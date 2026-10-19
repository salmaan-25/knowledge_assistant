"""
routes/chat.py — Document-grounded question answering via the LangGraph agent.

Endpoints:
  POST /chat — Ask a question about a selected document.

The endpoint routes the request through the full LangGraph workflow:
  identify_intent → retrieve_for_qa → generate_answer

Source references and a has_sufficient_context flag are returned alongside
the answer so the frontend can clearly indicate when the evidence is weak.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.learning_agent import get_learning_graph
from app.database.connection import get_db
from app.database.models import ChatMessage, DocumentStatus, MessageRole, User
from app.routes.auth import get_current_user
from app.routes.documents import _get_owned_document
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatMessageOut,
    ChatRequest,
    ChatResponse,
    ClearChatResponse,
    SourceReference,
)
from app.services.quota_service import check_quota, record_usage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a question about a document — returns a grounded answer with citations",
)
async def ask_question(
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """
    Document-grounded question answering with persistent history.

    Flow:
      1. Verify the document exists and belongs to the user.
      2. Check the document is in READY status.
      3. Verify user has sufficient daily token quota.
      4. Invoke the LangGraph workflow.
      5. Persist both the user question and the assistant response to the database.
      6. Record token usage and return the grounded answer with source citations.
    """
    doc = await _get_owned_document(body.document_id, current_user, db)

    if doc.status != DocumentStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Document '{doc.original_filename}' is not ready. "
                f"Current status: {doc.status.value}. "
                "Please wait for indexing to complete."
            ),
        )

    # Pre-flight quota check — raises HTTP 429 if the daily token limit is exceeded.
    await check_quota(current_user, db)

    graph = get_learning_graph()

    initial_state = {
        "document_id": body.document_id,
        "user_id": current_user.id,
        "question": body.question,
        "retry_count": 0,
        "has_sufficient_context": False,
        "difficulty": "medium",
        "question_count": 10,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("LangGraph invocation failed for document %s", body.document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your question: {exc}",
        ) from exc

    # Record usage AFTER successful invocation (commit=False batches with message persistence).
    await record_usage(current_user, db, tokens=0, commit=False)

    answer = final_state.get("answer") or "No answer could be generated."
    sources_raw: list[dict] = final_state.get("sources") or []
    intent = final_state.get("intent") or "question_answering"
    has_sufficient = final_state.get("has_sufficient_context", False)

    sources = [
        SourceReference(
            chunk_id=s["chunk_id"],
            page_number=s.get("page_number"),
            text_excerpt=s.get("text_excerpt", ""),
            relevance_score=s.get("relevance_score"),
        )
        for s in sources_raw
    ]

    # --- Persist to database ---
    user_msg = ChatMessage(
        user_id=current_user.id,
        document_id=body.document_id,
        role=MessageRole.USER.value,
        content=body.question,
    )
    db.add(user_msg)

    sources_data = [s.model_dump() for s in sources]
    assistant_msg = ChatMessage(
        user_id=current_user.id,
        document_id=body.document_id,
        role=MessageRole.ASSISTANT.value,
        content=answer,
        sources=sources_data,
        has_sufficient_context=has_sufficient,
    )
    db.add(assistant_msg)
    await db.commit()

    return ChatResponse(
        document_id=body.document_id,
        question=body.question,
        answer=answer,
        sources=sources,
        has_sufficient_context=has_sufficient,
        intent=intent,
    )


@router.get(
    "/{document_id}/history",
    response_model=ChatHistoryResponse,
    summary="Get chronological chat history for a document",
)
async def get_chat_history(
    document_id: str,
    limit: int = Query(default=100, ge=1, le=500, description="Max messages to return"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatHistoryResponse:
    """
    Fetch all chat messages exchanged with the specified document for the current user.
    Messages are returned in ascending chronological order (oldest first).
    """
    await _get_owned_document(document_id, current_user, db)

    result = await db.execute(
        select(ChatMessage)
        .where(
            ChatMessage.user_id == current_user.id,
            ChatMessage.document_id == document_id,
        )
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
    )
    messages = result.scalars().all()

    items = [
        ChatMessageOut(
            id=m.id,
            document_id=m.document_id,
            role=m.role,
            content=m.content,
            sources=[
                SourceReference(
                    chunk_id=s.get("chunk_id", ""),
                    page_number=s.get("page_number"),
                    text_excerpt=s.get("text_excerpt", ""),
                    relevance_score=s.get("relevance_score"),
                )
                for s in (m.sources or [])
            ],
            has_sufficient_context=m.has_sufficient_context,
            created_at=m.created_at,
        )
        for m in messages
    ]

    return ChatHistoryResponse(
        document_id=document_id,
        messages=items,
        total=len(items),
    )


@router.delete(
    "/{document_id}/history",
    response_model=ClearChatResponse,
    summary="Clear chat history for a document",
)
async def clear_chat_history(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ClearChatResponse:
    """
    Clear all previous conversation messages for the specified document and user.
    """
    await _get_owned_document(document_id, current_user, db)

    result = await db.execute(
        delete(ChatMessage).where(
            ChatMessage.user_id == current_user.id,
            ChatMessage.document_id == document_id,
        )
    )
    await db.commit()

    return ClearChatResponse(
        document_id=document_id,
        cleared_count=result.rowcount or 0,
    )

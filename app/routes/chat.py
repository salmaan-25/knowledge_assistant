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

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.learning_agent import get_learning_graph
from app.database.connection import get_db
from app.database.models import DocumentStatus, User
from app.routes.auth import get_current_user
from app.routes.documents import _get_owned_document
from app.schemas.chat import ChatRequest, ChatResponse, SourceReference

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
    Document-grounded question answering.

    Flow:
      1. Verify the document exists and belongs to the user.
      2. Check the document is in READY status.
      3. Invoke the LangGraph workflow.
      4. Return the answer with source references.

    The LangGraph agent:
      - Classifies intent (question_answering, summarization, etc.)
      - Retrieves relevant chunks from ChromaDB.
      - Generates a grounded answer using Gemini.
      - Includes source page references.
      - Sets has_sufficient_context=False when evidence is too weak.
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

    return ChatResponse(
        document_id=body.document_id,
        question=body.question,
        answer=answer,
        sources=sources,
        has_sufficient_context=has_sufficient,
        intent=intent,
    )

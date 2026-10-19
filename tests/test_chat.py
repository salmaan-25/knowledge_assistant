"""
tests/test_chat.py — Integration tests for Chat and Chat History endpoints.

Covers:
  - POST /chat auth requirement (401)
  - POST /chat non-existent document (404)
  - POST /chat document not ready (409)
  - POST /chat successful Q&A:
      - Validates both 'question' and 'message' aliases
      - Invokes mocked LangGraph
      - Records user message and assistant answer in ChatMessage table
      - Returns grounded answer with citations
  - GET /chat/{document_id}/history:
      - Requires authentication (401)
      - Returns 404 for non-existent or other user's document
      - Returns chronological messages (user and assistant) with sources
  - DELETE /chat/{document_id}/history:
      - Requires authentication (401)
      - Clears conversation for target document
      - Subsequent GET returns empty list
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Document, DocumentStatus, User


async def _create_test_document(
    db: AsyncSession,
    owner_id: str,
    status: DocumentStatus = DocumentStatus.READY,
) -> Document:
    """Helper to persist a test document directly to the test DB."""
    doc = Document(
        owner_id=owner_id,
        filename="test.pdf",
        original_filename="test.pdf",
        file_path="/tmp/test.pdf",
        file_size_bytes=1024,
        mime_type="application/pdf",
        status=status,
        title="Test Document",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


@pytest.mark.asyncio
async def test_chat_requires_auth(async_client: AsyncClient):
    """POST /chat requires valid JWT authentication."""
    resp = await async_client.post(
        "/chat",
        json={"document_id": "non-existent-id", "question": "What is this?"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_chat_history_requires_auth(async_client: AsyncClient):
    """GET /chat/{document_id}/history requires authentication."""
    resp = await async_client.get("/chat/some-doc-id/history")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_chat_clear_requires_auth(async_client: AsyncClient):
    """DELETE /chat/{document_id}/history requires authentication."""
    resp = await async_client.delete("/chat/some-doc-id/history")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_chat_document_not_found(async_client: AsyncClient, auth_headers: dict):
    """POST /chat on non-existent document returns 404."""
    resp = await async_client.post(
        "/chat",
        headers=auth_headers,
        json={"document_id": "00000000-0000-0000-0000-000000000000", "question": "Hello?"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_chat_document_not_ready_returns_409(
    async_client: AsyncClient, auth_headers: dict, test_db: AsyncSession
):
    """POST /chat returns 409 if document is still in PENDING or PROCESSING status."""
    me_resp = await async_client.get("/auth/me", headers=auth_headers)
    user_id = me_resp.json()["id"]

    doc = await _create_test_document(test_db, user_id, status=DocumentStatus.PENDING)

    resp = await async_client.post(
        "/chat",
        headers=auth_headers,
        json={"document_id": doc.id, "question": "What is this?"},
    )
    assert resp.status_code == 409
    assert "not ready" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_chat_and_history_lifecycle(
    async_client: AsyncClient, auth_headers: dict, test_db: AsyncSession
):
    """
    Test full chat lifecycle:
      1. Verify empty history
      2. Ask question with 'message' alias
      3. Verify response structure and database persistence
      4. Ask second question with 'question' field
      5. Verify GET /chat/{document_id}/history returns all 4 messages (2 user, 2 assistant)
      6. DELETE /chat/{document_id}/history clears messages
      7. Verify GET /chat/{document_id}/history is empty again
    """
    me_resp = await async_client.get("/auth/me", headers=auth_headers)
    user_id = me_resp.json()["id"]

    doc = await _create_test_document(test_db, user_id, status=DocumentStatus.READY)

    # 1. Initial history is empty
    history_resp = await async_client.get(f"/chat/{doc.id}/history", headers=auth_headers)
    assert history_resp.status_code == 200
    assert history_resp.json()["messages"] == []
    assert history_resp.json()["total"] == 0

    # Mock LangGraph invoke
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "answer": "Demarketing is marketing aimed at discouraging demand.",
        "sources": [
            {
                "chunk_id": "chunk-1",
                "page_number": 2,
                "text_excerpt": "Demarketing strategies reduce customer demand.",
                "relevance_score": 0.95,
            }
        ],
        "intent": "question_answering",
        "has_sufficient_context": True,
    }

    # 2. Ask first question using 'message' field alias
    with patch("app.routes.chat.get_learning_graph", return_value=mock_graph):
        resp1 = await async_client.post(
            "/chat",
            headers=auth_headers,
            json={"document_id": doc.id, "message": "what is demarketing?"},
        )

    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["question"] == "what is demarketing?"
    assert data1["answer"] == "Demarketing is marketing aimed at discouraging demand."
    assert data1["message"] == data1["answer"]  # computed field check
    assert len(data1["sources"]) == 1
    assert data1["has_sufficient_context"] is True

    # 3. Ask second question using 'question' field
    mock_graph.invoke.return_value = {
        "answer": "Examples include tobacco sin taxes and electricity peak pricing.",
        "sources": [],
        "intent": "question_answering",
        "has_sufficient_context": True,
    }

    with patch("app.routes.chat.get_learning_graph", return_value=mock_graph):
        resp2 = await async_client.post(
            "/chat",
            headers=auth_headers,
            json={"document_id": doc.id, "question": "Give me some examples."},
        )

    assert resp2.status_code == 200

    # 4. GET /chat/{document_id}/history returns 4 messages in chronological order
    history_resp2 = await async_client.get(f"/chat/{doc.id}/history", headers=auth_headers)
    assert history_resp2.status_code == 200
    history_data = history_resp2.json()
    assert history_data["total"] == 4
    msgs = history_data["messages"]

    # First turn: user question + assistant answer
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "what is demarketing?"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Demarketing is marketing aimed at discouraging demand."
    assert len(msgs[1]["sources"]) == 1
    assert msgs[1]["sources"][0]["page_number"] == 2

    # Second turn
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"] == "Give me some examples."
    assert msgs[3]["role"] == "assistant"
    assert "Examples include" in msgs[3]["content"]

    # 5. Clear history
    del_resp = await async_client.delete(f"/chat/{doc.id}/history", headers=auth_headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["cleared_count"] == 4

    # 6. Verify history is empty
    history_resp3 = await async_client.get(f"/chat/{doc.id}/history", headers=auth_headers)
    assert history_resp3.status_code == 200
    assert history_resp3.json()["messages"] == []
    assert history_resp3.json()["total"] == 0


@pytest.mark.asyncio
async def test_chat_history_isolation_between_users(
    async_client: AsyncClient, auth_headers: dict, test_db: AsyncSession
):
    """User B cannot access or clear chat history for User A's document."""
    # User A registers & creates doc
    me_resp = await async_client.get("/auth/me", headers=auth_headers)
    user_a_id = me_resp.json()["id"]
    doc_a = await _create_test_document(test_db, user_a_id, status=DocumentStatus.READY)

    # Register User B
    await async_client.post(
        "/auth/register",
        json={"email": "user_b@example.com", "password": "password123"},
    )
    login_b = await async_client.post(
        "/auth/login",
        data={"username": "user_b@example.com", "password": "password123"},
    )
    token_b = login_b.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # User B tries to get User A's history -> 403 (document not owned by B)
    resp = await async_client.get(f"/chat/{doc_a.id}/history", headers=headers_b)
    assert resp.status_code == 403

    # User B tries to delete User A's history -> 403
    del_resp = await async_client.delete(f"/chat/{doc_a.id}/history", headers=headers_b)
    assert del_resp.status_code == 403


@pytest.mark.asyncio
async def test_quota_tracking_and_blocking_on_zero_limit(
    async_client: AsyncClient, auth_headers: dict, test_db: AsyncSession
):
    """
    Verify:
      1. Initial /auth/me returns tokens_used_today=0, daily_token_limit=50000, queries_today=0
      2. Chat query increments tokens_used_today and queries_today
      3. Setting custom_daily_token_limit=0 blocks further queries with 429
    """
    me_resp = await async_client.get("/auth/me", headers=auth_headers)
    assert me_resp.status_code == 200
    user_data = me_resp.json()
    assert user_data["tokens_used_today"] == 0
    assert user_data["daily_token_limit"] == 50000
    assert user_data["queries_today"] == 0

    user_id = user_data["id"]
    doc = await _create_test_document(test_db, user_id, status=DocumentStatus.READY)

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "answer": "Test answer grounded in document.",
        "sources": [],
        "intent": "question_answering",
        "has_sufficient_context": True,
    }

    # First query succeeds
    with patch("app.routes.chat.get_learning_graph", return_value=mock_graph):
        resp = await async_client.post(
            "/chat",
            headers=auth_headers,
            json={"document_id": doc.id, "question": "First question?"},
        )
    assert resp.status_code == 200

    # /auth/me now reflects the usage
    me_after = await async_client.get("/auth/me", headers=auth_headers)
    assert me_after.json()["tokens_used_today"] == 1000
    assert me_after.json()["queries_today"] == 1

    # Admin / System blocks user by setting limit to 0
    from sqlalchemy import select

    res = await test_db.execute(select(User).where(User.id == user_id))
    user_db = res.scalar_one()
    user_db.custom_daily_token_limit = 0
    await test_db.commit()

    # Second query is rejected with HTTP 429
    with patch("app.routes.chat.get_learning_graph", return_value=mock_graph):
        resp_blocked = await async_client.post(
            "/chat",
            headers=auth_headers,
            json={"document_id": doc.id, "question": "Second question?"},
        )
    assert resp_blocked.status_code == 429
    err = resp_blocked.json()["detail"]
    assert "daily token limit" in err.lower()
    assert "midnight utc" in err.lower()

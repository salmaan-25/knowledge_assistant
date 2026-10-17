"""
tests/test_documents.py — Integration tests for document upload and management endpoints.

Note: Tests that call external services (Gemini embeddings, ChromaDB) are skipped
unless the GEMINI_API_KEY environment variable is set.  All other tests use mocks.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


# ------------------------------------------------------------------ #
# Upload endpoint tests (mocked external calls)
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_upload_requires_auth(async_client: AsyncClient, sample_pdf_bytes: bytes):
    """PDF upload requires authentication."""
    resp = await async_client.post(
        "/documents/upload",
        files={"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf(async_client: AsyncClient, auth_headers: dict):
    """Uploading a .txt file should be rejected with 422."""
    resp = await async_client.post(
        "/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(b"some text"), "text/plain")},
        headers=auth_headers,
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_upload_rejects_file_with_bad_magic_bytes(
    async_client: AsyncClient, auth_headers: dict
):
    """A file named .pdf but with non-PDF content must be rejected."""
    resp = await async_client.post(
        "/documents/upload",
        files={"file": ("fake.pdf", io.BytesIO(b"not a pdf content"), "application/pdf")},
        headers=auth_headers,
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
@patch("app.routes.documents._index_document_background", new_callable=AsyncMock)
async def test_upload_valid_pdf_returns_202(
    mock_bg, async_client: AsyncClient, auth_headers: dict, sample_pdf_bytes: bytes
):
    """A valid PDF upload returns 202 Accepted with a document_id."""
    resp = await async_client.post(
        "/documents/upload",
        files={"file": ("notes.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
        headers=auth_headers,
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "document_id" in data
    assert data["status"] == "pending"


# ------------------------------------------------------------------ #
# List documents
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_list_documents_empty(async_client: AsyncClient, auth_headers: dict):
    """A new user has no documents."""
    resp = await async_client.get("/documents", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["documents"] == []


@pytest.mark.asyncio
async def test_list_documents_requires_auth(async_client: AsyncClient):
    resp = await async_client.get("/documents")
    assert resp.status_code == 401


# ------------------------------------------------------------------ #
# Get / delete document (not found)
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_get_nonexistent_document_returns_404(async_client: AsyncClient, auth_headers: dict):
    resp = await async_client.get(
        "/documents/00000000-0000-0000-0000-000000000000", headers=auth_headers
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_document_returns_404(
    async_client: AsyncClient, auth_headers: dict
):
    resp = await async_client.delete(
        "/documents/00000000-0000-0000-0000-000000000000", headers=auth_headers
    )
    assert resp.status_code == 404

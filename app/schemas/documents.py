"""
schemas/documents.py — Pydantic request/response models for the documents API.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# Upload response
# ------------------------------------------------------------------ #
class PageExtract(BaseModel):
    """Extracted text from a single PDF page."""

    page_number: int
    text: str
    char_count: int


class DocumentUploadResponse(BaseModel):
    """Response returned after a PDF is uploaded and indexed."""

    document_id: str
    filename: str
    page_count: int
    total_chunks: int
    status: str
    message: str


# ------------------------------------------------------------------ #
# Document read models
# ------------------------------------------------------------------ #
class DocumentOut(BaseModel):
    """Public view of a document record (never exposes file paths)."""

    id: str
    filename: str
    original_filename: str
    file_size_bytes: int
    page_count: int | None = None
    total_chunks: int | None = None
    status: str
    title: str | None = None
    description: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Paginated list of documents belonging to the requesting user."""

    documents: list[DocumentOut]
    total: int


# ------------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------------ #
class SummaryRequest(BaseModel):
    """Request body for the document summary endpoint."""

    section_hint: str | None = Field(
        default=None,
        description="Optional hint to focus the summary on a specific topic or chapter.",
        max_length=500,
    )
    max_length: int | None = Field(
        default=None,
        ge=50,
        le=2000,
        description="Approximate target length of the summary in words.",
    )


class SummaryResponse(BaseModel):
    """Generated summary for a document or section."""

    document_id: str
    summary: str
    source_pages: list[int]
    section_hint: str | None = None

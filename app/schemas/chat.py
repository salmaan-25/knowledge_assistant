"""
schemas/chat.py — Pydantic models for the chat / question-answering endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """A passage retrieved from the document that supports the answer."""

    chunk_id: str
    page_number: int | None = None
    text_excerpt: str = Field(description="Short excerpt from the relevant chunk.")
    relevance_score: float | None = None


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    document_id: str = Field(description="UUID of the document to query.")
    question: str = Field(
        min_length=3,
        max_length=2000,
        description="The question to answer from the document.",
    )
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Optional previous turns for multi-turn conversation context.",
    )


class ChatResponse(BaseModel):
    """Response returned by the document Q&A endpoint."""

    document_id: str
    question: str
    answer: str
    sources: list[SourceReference]
    has_sufficient_context: bool = Field(
        description=(
            "False when the retrieved evidence is too weak to ground a reliable "
            "answer; the model will say so in the answer text as well."
        )
    )
    intent: str = Field(
        description="Intent identified by the LangGraph router (e.g. 'question_answering')."
    )

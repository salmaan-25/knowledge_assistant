"""
schemas/chat.py — Pydantic models for the chat / question-answering endpoint.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field, model_validator


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
        default="",
        max_length=2000,
        description="The question to answer from the document.",
    )
    message: str | None = Field(
        default=None,
        description="Alias for question used by some client implementations.",
    )
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Optional previous turns for multi-turn conversation context.",
    )

    @model_validator(mode="before")
    @classmethod
    def populate_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # If question is empty or missing, but message is provided, populate question
            if not data.get("question") and data.get("message"):
                data["question"] = data["message"]
            elif data.get("question") and not data.get("message"):
                data["message"] = data["question"]
        return data

    @model_validator(mode="after")
    def validate_content(self) -> ChatRequest:
        text = (self.question or "").strip()
        if len(text) < 3:
            raise ValueError("question (or message) must be at least 3 characters long.")
        self.question = text
        return self


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

    @computed_field
    @property
    def message(self) -> str:
        """Convenience alias for answer."""
        return self.answer

    @computed_field
    @property
    def response(self) -> str:
        """Convenience alias for answer."""
        return self.answer


class ChatMessageOut(BaseModel):
    """A persisted chat message (user question or assistant answer)."""

    id: str
    document_id: str
    role: str = Field(description="'user' or 'assistant'")
    content: str
    sources: list[SourceReference] = []
    has_sufficient_context: bool = True
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatHistoryResponse(BaseModel):
    """Chronological message history for a document."""

    document_id: str
    messages: list[ChatMessageOut]
    total: int


class ClearChatResponse(BaseModel):
    """Confirmation when chat history is cleared."""

    document_id: str
    cleared_count: int
    detail: str = "Chat history cleared successfully."

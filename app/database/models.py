"""
database/models.py — SQLAlchemy ORM models for the Knowledge Assistant.

Tables:
    users              — Registered users with role (user | admin) and token limits
    daily_token_usage  — Per-user daily token & query consumption tracking
    documents          — Uploaded PDF metadata and processing status
    document_chunks    — Relational chunk metadata (complement to ChromaDB)
    quiz_attempts      — A single quiz session per user per document
    quiz_answers       — Individual answer records per attempt
    topic_progress     — Aggregated weak-topic tracking per user
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


# ------------------------------------------------------------------ #
# Enumerations (stored as VARCHAR for portability)
# ------------------------------------------------------------------ #
class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Difficulty(str, enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionType(str, enum.Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"


# ------------------------------------------------------------------ #
# Users
# ------------------------------------------------------------------ #
class User(Base):
    """
    Represents an application user.

    Password hashing is handled in the auth service; this model only stores
    the hash.  email is unique and is the login identifier.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Role — never accepted from registration input, only set server-side.
    # Admins have unlimited token access and access to /admin/* endpoints.
    role: Mapped[str] = mapped_column(String(20), default=UserRole.USER, nullable=False, index=True)

    # Optional per-user token override (NULL → use global DEFAULT_DAILY_TOKEN_LIMIT).
    # Admins always bypass this limit regardless of the value here.
    custom_daily_token_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    # Relationships
    documents: Mapped[list[Document]] = relationship(
        "Document", back_populates="owner", cascade="all, delete-orphan"
    )
    quiz_attempts: Mapped[list[QuizAttempt]] = relationship(
        "QuizAttempt", back_populates="user", cascade="all, delete-orphan"
    )
    topic_progress: Mapped[list[TopicProgress]] = relationship(
        "TopicProgress", back_populates="user", cascade="all, delete-orphan"
    )
    daily_usage: Mapped[list[DailyTokenUsage]] = relationship(
        "DailyTokenUsage", back_populates="user", cascade="all, delete-orphan"
    )
    chat_messages: Mapped[list[ChatMessage]] = relationship(
        "ChatMessage", back_populates="user", cascade="all, delete-orphan"
    )

    @property
    def is_admin(self) -> bool:
        """Convenience check — avoids raw string comparisons in business logic."""
        return self.role == UserRole.ADMIN

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User id={self.id} email={self.email} role={self.role}>"


# ------------------------------------------------------------------ #
# Daily Token Usage
# ------------------------------------------------------------------ #
class DailyTokenUsage(Base):
    """
    Tracks daily token & query consumption per user.

    One row per (user_id, usage_date).  The application performs an
    upsert-style increment so only one row exists per user per calendar day.
    Admins are never written to this table — they bypass quota checks entirely.
    """

    __tablename__ = "daily_token_usage"
    __table_args__ = (UniqueConstraint("user_id", "usage_date", name="uq_user_usage_date"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid, index=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    usage_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    query_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="daily_usage")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<DailyTokenUsage user={self.user_id} date={self.usage_date} "
            f"tokens={self.tokens_used} queries={self.query_count}>"
        )


# ------------------------------------------------------------------ #
# Documents
# ------------------------------------------------------------------ #
class Document(Base):
    """
    Metadata for an uploaded PDF.

    The actual binary file is stored on disk (uploads/).  ChromaDB stores
    the indexed chunk embeddings.  This table records metadata and ownership
    so we can enforce per-user access control.
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid, index=True)
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(
            DocumentStatus, name="document_status", values_callable=lambda x: [e.value for e in x]
        ),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    # Relationships
    owner: Mapped[User] = relationship("User", back_populates="documents")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )
    quiz_attempts: Mapped[list[QuizAttempt]] = relationship(
        "QuizAttempt", back_populates="document"
    )
    chat_messages: Mapped[list[ChatMessage]] = relationship(
        "ChatMessage", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Document id={self.id} filename={self.filename} status={self.status}>"


# ------------------------------------------------------------------ #
# Document chunks (relational metadata mirror of ChromaDB)
# ------------------------------------------------------------------ #
class DocumentChunk(Base):
    """
    Stores chunk-level metadata in PostgreSQL to complement ChromaDB.

    ChromaDB holds the vector embeddings; this table keeps the source text
    and page reference so we can join chunk metadata with document ownership
    without querying the vector store for every lookup.
    """

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid, index=True)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chroma_chunk_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    # Relationships
    document: Mapped[Document] = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DocumentChunk id={self.id} doc={self.document_id} page={self.page_number}>"


# ------------------------------------------------------------------ #
# Quiz attempts
# ------------------------------------------------------------------ #
class QuizAttempt(Base):
    """
    A single quiz session.  The answer key is stored server-side (never sent
    to the client before submission) so scoring is deterministic and tamper-proof.
    """

    __tablename__ = "quiz_attempts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid, index=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    topic_filter: Mapped[str | None] = mapped_column(String(500), nullable=True)
    difficulty: Mapped[Difficulty] = mapped_column(
        Enum(Difficulty, name="difficulty", values_callable=lambda x: [e.value for e in x]),
        default=Difficulty.MEDIUM,
        nullable=False,
    )
    question_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # JSON blob: list of QuizQuestion dicts (includes correct_answer — server only)
    questions_json: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    percentage: Mapped[float | None] = mapped_column(Float, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="quiz_attempts")
    document: Mapped[Document] = relationship("Document", back_populates="quiz_attempts")
    answers: Mapped[list[QuizAnswer]] = relationship(
        "QuizAnswer", back_populates="attempt", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<QuizAttempt id={self.id} user={self.user_id} score={self.score}>"


# ------------------------------------------------------------------ #
# Quiz answers
# ------------------------------------------------------------------ #
class QuizAnswer(Base):
    """Individual answer submitted by the user for a single quiz question."""

    __tablename__ = "quiz_answers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid, index=True)
    attempt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("quiz_attempts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question_index: Mapped[int] = mapped_column(Integer, nullable=False)
    submitted_answer: Mapped[str] = mapped_column(String(500), nullable=False)
    correct_answer: Mapped[str] = mapped_column(String(500), nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_pages: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    # Relationships
    attempt: Mapped[QuizAttempt] = relationship("QuizAttempt", back_populates="answers")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<QuizAnswer attempt={self.attempt_id} q={self.question_index} correct={self.is_correct}>"


# ------------------------------------------------------------------ #
# Topic progress
# ------------------------------------------------------------------ #
class TopicProgress(Base):
    """
    Aggregated per-topic performance tracking per user.

    Updated after each quiz submission to identify weak areas and drive
    personalised revision recommendations in Phase 7.
    """

    __tablename__ = "topic_progress"
    __table_args__ = (
        UniqueConstraint("user_id", "document_id", "topic", name="uq_user_doc_topic"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid, index=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    correct: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_attempted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="topic_progress")

    @property
    def accuracy(self) -> float:
        """Return accuracy as a 0–1 float, or 0 if no attempts."""
        return self.correct / self.attempts if self.attempts > 0 else 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return f"<TopicProgress user={self.user_id} topic={self.topic} acc={self.accuracy:.0%}>"


# ------------------------------------------------------------------ #
# Chat History
# ------------------------------------------------------------------ #
class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(Base):
    """
    Persisted question-and-answer turn for a document.

    Records both user queries and assistant responses (with citations and context flags)
    so users can resume conversations across sessions.
    """

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid, index=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user' | 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    has_sufficient_context: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False, index=True
    )

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="chat_messages")
    document: Mapped[Document] = relationship("Document", back_populates="chat_messages")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ChatMessage id={self.id} role={self.role} doc={self.document_id}>"

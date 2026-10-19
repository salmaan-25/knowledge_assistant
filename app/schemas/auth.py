"""
schemas/auth.py — Pydantic models for user authentication and registration.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Request body for user registration."""

    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


class UserLogin(BaseModel):
    """Request body for user login."""

    email: EmailStr
    password: str


class UserOut(BaseModel):
    """Public user representation (never exposes the password hash)."""

    id: str
    email: str
    full_name: str | None = None
    is_active: bool
    role: str  # "user" | "admin"
    custom_daily_token_limit: int | None = None

    # Quota info for user dashboard / header display
    tokens_used_today: int = 0
    daily_token_limit: int = 50000  # -1 = unlimited (admin)
    queries_today: int = 0

    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    """JWT access token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds.")


class TokenData(BaseModel):
    """Decoded JWT payload."""

    user_id: str | None = None


class ProgressOut(BaseModel):
    """Learning progress record for a single topic."""

    topic: str
    document_id: str
    attempts: int
    correct: int
    accuracy: float
    last_attempted_at: datetime | None = None

    model_config = {"from_attributes": True}

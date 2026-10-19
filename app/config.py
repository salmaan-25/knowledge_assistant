"""
config.py — Application configuration loaded from environment variables.

All settings are read from the .env file (via python-dotenv) or from
real environment variables.  No secrets are hard-coded here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the repository root (.env lives there).
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from the repo root before Pydantic reads the values.
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    """
    Central settings object.  Every field maps to an environment variable
    of the same name (case-insensitive by default in Pydantic Settings v2).
    """

    # ------------------------------------------------------------------ #
    # Gemini / LLM
    # ------------------------------------------------------------------ #
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"

    # ------------------------------------------------------------------ #
    # PostgreSQL
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/knowledge_assistant"

    # ------------------------------------------------------------------ #
    # ChromaDB  (persisted locally; swap to hosted service in production)
    # ------------------------------------------------------------------ #
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "chroma_store")
    CHROMA_COLLECTION_NAME: str = "knowledge_chunks"

    # ------------------------------------------------------------------ #
    # File uploads
    # ------------------------------------------------------------------ #
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    MAX_UPLOAD_SIZE_MB: int = 50  # Maximum allowed PDF size in MB

    # ------------------------------------------------------------------ #
    # Chunking parameters
    # ------------------------------------------------------------------ #
    CHUNK_SIZE: int = 800  # Target characters per chunk
    CHUNK_OVERLAP: int = 150  # Overlap between consecutive chunks

    # ------------------------------------------------------------------ #
    # RAG / retrieval
    # ------------------------------------------------------------------ #
    RETRIEVAL_TOP_K: int = (
        4  # Number of chunks returned per query (optimized for low cost & high precision)
    )

    # ------------------------------------------------------------------ #
    # LangGraph / agent
    # ------------------------------------------------------------------ #
    MAX_LLM_RETRIES: int = 3  # Bounded retries for invalid LLM output

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    APP_ENV: str = "development"  # "development" | "production"
    SECRET_KEY: str = "change-me-in-production-use-a-long-random-string"
    ALLOWED_ORIGINS: list[str] | str = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
    ]

    # ------------------------------------------------------------------ #
    # Admin & Quota
    # ------------------------------------------------------------------ #
    # Email address that is automatically promoted to ADMIN on first startup.
    # Set this to your personal email in .env before the first run.
    SUPERADMIN_EMAIL: str = ""

    # Maximum tokens a basic user may consume per calendar day (UTC).
    # Set to 0 to disable quota enforcement for all users.
    DEFAULT_DAILY_TOKEN_LIMIT: int = 50000

    @field_validator("ALLOWED_ORIGINS")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return [str(i) for i in v]
        return []

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()

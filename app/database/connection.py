"""
database/connection.py — Async SQLAlchemy engine, session factory, and Base.

We use asyncpg for non-blocking PostgreSQL access.  The async_sessionmaker
is used via the get_db() dependency injected into FastAPI route handlers.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

# ------------------------------------------------------------------ #
# Engine
# ------------------------------------------------------------------ #
engine_kwargs: dict[str, Any] = {
    "echo": not settings.is_production,  # SQL logging in dev only
    "pool_pre_ping": True,  # Validate connections before using
}
if "sqlite" not in settings.DATABASE_URL:
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20

engine = create_async_engine(
    settings.DATABASE_URL,
    **engine_kwargs,
)

# ------------------------------------------------------------------ #
# Session factory
# ------------------------------------------------------------------ #
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ------------------------------------------------------------------ #
# Declarative base — all ORM models inherit from this
# ------------------------------------------------------------------ #
class Base(DeclarativeBase):
    """Shared declarative base for all SQLAlchemy models."""


# ------------------------------------------------------------------ #
# FastAPI dependency
# ------------------------------------------------------------------ #
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield a database session and guarantee it is closed after the request.

    Usage in a route:
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_all_tables() -> None:
    """Create all tables (dev helper).  Prefer Alembic migrations in production."""
    async with engine.begin() as conn:
        from app.database import models  # noqa: F401 — ensures models are registered

        await conn.run_sync(Base.metadata.create_all)

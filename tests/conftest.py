"""
tests/conftest.py — Shared pytest fixtures for the test suite.

Provides:
  - async_client: AsyncClient for end-to-end HTTP testing against the FastAPI app.
  - test_db: In-memory SQLite engine for database tests (avoids real PostgreSQL in CI).
  - sample_pdf_bytes: A minimal valid PDF for upload tests.
  - auth_headers: Pre-generated JWT headers for authenticated route tests.
"""

from __future__ import annotations

import os

# Set dummy env vars before importing app to satisfy pydantic Settings validation
os.environ.setdefault("GEMINI_API_KEY", "test_dummy_key_for_testing_1234567890")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_must_be_at_least_32_chars_long_12345")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.database import models  # noqa: F401 — register all ORM models
from app.database.connection import Base, get_db
from app.main import app

# ------------------------------------------------------------------ #
# In-memory SQLite for tests (avoids needing real PostgreSQL in CI)
# ------------------------------------------------------------------ #
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def test_engine():
    """Create a fresh in-memory SQLite engine for each test function."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_db(test_engine):
    """Yield an AsyncSession bound to the in-memory test database."""
    session_factory = async_sessionmaker(bind=test_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def async_client(test_engine):
    """
    Yield an httpx.AsyncClient wired to the FastAPI app with the test DB override.

    The get_db dependency is overridden to use the in-memory SQLite engine so
    tests don't touch the real PostgreSQL database.
    """
    session_factory = async_sessionmaker(bind=test_engine, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


# ------------------------------------------------------------------ #
# Minimal valid PDF bytes fixture
# ------------------------------------------------------------------ #
@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """
    Return the bytes of a minimal but structurally valid PDF with one text page.
    This avoids needing a real test PDF file on disk.
    """
    # Minimal PDF 1.4 with one page containing "Hello, World!"
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj
4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
5 0 obj<</Length 44>>
stream
BT /F1 12 Tf 100 700 Td (Hello, World!) Tj ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000352 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
445
%%EOF"""
    return pdf_content


@pytest_asyncio.fixture
async def auth_headers(async_client: AsyncClient) -> dict[str, str]:
    """Register a test user and return JWT auth headers."""
    await async_client.post(
        "/auth/register",
        json={"email": "test@example.com", "password": "testpassword123"},
    )
    resp = await async_client.post(
        "/auth/login",
        data={"username": "test@example.com", "password": "testpassword123"},
    )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

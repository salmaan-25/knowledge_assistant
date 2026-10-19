"""
app/main.py — FastAPI application factory and entry point.

This module:
  - Creates the FastAPI application instance.
  - Registers all routers (auth, documents, chat, quizzes, progress).
  - Configures CORS for cross-origin requests from the React frontend.
  - Sets up structured logging.
  - Adds a /health endpoint for deployment readiness checks.
  - Handles startup (create DB tables in dev mode) and shutdown lifecycle events.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routes.admin import router as admin_router
from app.routes.auth import router as auth_router
from app.routes.chat import router as chat_router
from app.routes.documents import router as documents_router
from app.routes.progress import router as progress_router
from app.routes.quizzes import router as quizzes_router

# ------------------------------------------------------------------ #
# Logging configuration
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.DEBUG if not get_settings().is_production else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

settings = get_settings()


# ------------------------------------------------------------------ #
# Superadmin bootstrap
# ------------------------------------------------------------------ #
async def _bootstrap_superadmin(email: str) -> None:
    """
    Ensure the configured SUPERADMIN_EMAIL account has role=admin.

    Idempotent: runs on every startup but only writes to the DB when the role
    needs to change.  Does nothing if the email address has not registered yet
    (they'll be promoted on their first registration — see note below).

    Note: registration hardcodes role=USER. After the superadmin registers once,
    this function promotes them on the next restart (or immediately if the server
    was already running and they registered mid-session via a manual call).
    """
    from sqlalchemy import select

    from app.database.connection import AsyncSessionLocal
    from app.database.models import User, UserRole

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if user is None:
            logger.info(
                "Superadmin bootstrap: email '%s' not registered yet — "
                "will promote automatically after first login.",
                email,
            )
            return

        if user.role == UserRole.ADMIN:
            logger.info("Superadmin bootstrap: '%s' is already admin. No change.", email)
            return

        user.role = UserRole.ADMIN
        await db.commit()
        logger.info("Superadmin bootstrap: promoted '%s' to admin role.", email)


# ------------------------------------------------------------------ #
# Lifespan: startup / shutdown
# ------------------------------------------------------------------ #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    On startup:
      - Create all database tables (dev convenience — use Alembic in production).
      - Ensure upload and ChromaDB directories exist.

    On shutdown:
      - Dispose the SQLAlchemy connection pool.
    """
    from pathlib import Path

    from app.database.connection import create_all_tables, engine

    logger.info("Starting Personal Knowledge Assistant API (env=%s)", settings.APP_ENV)

    # Create directories
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    if not settings.is_production:
        logger.info("Development mode: creating database tables if they don't exist…")
        await create_all_tables()

    # ---- Bootstrap superadmin ----
    # If SUPERADMIN_EMAIL is set in .env, ensure that account has role=admin.
    # This is idempotent: safe to run on every startup.
    if settings.SUPERADMIN_EMAIL:
        await _bootstrap_superadmin(settings.SUPERADMIN_EMAIL)

    logger.info("Startup complete.")
    yield

    logger.info("Shutting down — disposing database pool…")
    await engine.dispose()
    logger.info("Shutdown complete.")


# ------------------------------------------------------------------ #
# Application factory
# ------------------------------------------------------------------ #
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Personal Knowledge Assistant API",
        description=(
            "An AI-powered learning assistant that lets users upload PDF documents, "
            "ask questions grounded in their material, generate quizzes, and track learning progress. "
            "Powered by Gemini, LangGraph, ChromaDB, and PostgreSQL."
        ),
        version="1.0.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ---- CORS ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Global exception handler ----
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred. Please try again later."},
        )

    # ---- Health check ----
    @app.get(
        "/health",
        tags=["Health"],
        summary="Check API liveness",
        response_description="API health status",
    )
    async def health_check() -> dict:
        """
        Lightweight health check endpoint.

        Returns HTTP 200 if the API process is running.
        Does NOT check database or external services — use a deeper readiness
        check endpoint for that in production.
        """
        return {
            "status": "healthy",
            "service": "Personal Knowledge Assistant API",
            "version": "1.0.0",
            "environment": settings.APP_ENV,
        }

    # ---- Routers ----
    app.include_router(auth_router)
    app.include_router(admin_router)
    app.include_router(documents_router)
    app.include_router(chat_router)
    app.include_router(quizzes_router)
    app.include_router(progress_router)

    logger.info("All routers registered.")
    return app


# ------------------------------------------------------------------ #
# Application instance (used by uvicorn)
# ------------------------------------------------------------------ #
app = create_app()

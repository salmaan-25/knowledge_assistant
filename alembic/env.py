"""
alembic/env.py — Alembic migration environment.

Reads the DATABASE_URL from the application settings so we never store
credentials in alembic.ini or version-controlled files.

Supports async migrations via asyncpg using run_sync().
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

import app.database.models  # noqa: F401 — registers all ORM models with Base
from alembic import context

# ------------------------------------------------------------------ #
# Load application config and models
# ------------------------------------------------------------------ #
from app.config import get_settings
from app.database.connection import Base

settings = get_settings()

# ------------------------------------------------------------------ #
# Alembic config object
# ------------------------------------------------------------------ #
config = context.config

# Override the URL from alembic.ini with the real URL from settings.
# Escaping '%' as '%%' prevents configparser interpolation syntax errors when passwords
# contain URL-encoded characters (like '%40' for '@').
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL.replace("%", "%%"))

# Configure logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ------------------------------------------------------------------ #
# Offline mode (generate SQL without connecting)
# ------------------------------------------------------------------ #
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — generates SQL without a live connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ------------------------------------------------------------------ #
# Online mode (connect and migrate)
# ------------------------------------------------------------------ #
def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations asynchronously using asyncpg."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migration mode."""
    asyncio.run(run_async_migrations())


# ------------------------------------------------------------------ #
# Select mode and run
# ------------------------------------------------------------------ #
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

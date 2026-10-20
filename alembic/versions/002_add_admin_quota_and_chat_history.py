"""
alembic/versions/002_add_admin_quota_and_chat_history.py — Add admin roles, token quota, and chat history.

Adds:
  - users.role (VARCHAR(20), server_default='user', nullable=False)
  - users.custom_daily_token_limit (Integer, nullable=True)
  - daily_token_usage table (id, user_id, usage_date, tokens_used, query_count, uq_user_usage_date)
  - chat_messages table (id, user_id, document_id, role, content, sources, has_sufficient_context, created_at)

Revision ID: 002
Revises: 001
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ---- 1. users: role and token limit ----
    op.add_column("users", sa.Column("role", sa.String(20), nullable=False, server_default="user"))
    op.create_index("ix_users_role", "users", ["role"])
    op.add_column("users", sa.Column("custom_daily_token_limit", sa.Integer(), nullable=True))

    # ---- 2. daily_token_usage ----
    op.create_table(
        "daily_token_usage",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("usage_date", sa.Date(), nullable=False),
        sa.Column("tokens_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("query_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_daily_token_usage_id", "daily_token_usage", ["id"])
    op.create_index("ix_daily_token_usage_user_id", "daily_token_usage", ["user_id"])
    op.create_index("ix_daily_token_usage_usage_date", "daily_token_usage", ["usage_date"])
    op.create_unique_constraint(
        "uq_user_usage_date", "daily_token_usage", ["user_id", "usage_date"]
    )

    # ---- 3. chat_messages ----
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column("has_sufficient_context", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chat_messages_id", "chat_messages", ["id"])
    op.create_index("ix_chat_messages_user_id", "chat_messages", ["user_id"])
    op.create_index("ix_chat_messages_document_id", "chat_messages", ["document_id"])
    op.create_index("ix_chat_messages_created_at", "chat_messages", ["created_at"])


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("daily_token_usage")
    op.drop_column("users", "custom_daily_token_limit")
    op.drop_index("ix_users_role", table_name="users")
    op.drop_column("users", "role")

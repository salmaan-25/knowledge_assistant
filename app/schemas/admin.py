"""
schemas/admin.py — Pydantic models for the Admin Panel API.

All schemas here are used exclusively by /api/v1/admin/* endpoints and are
only reachable by users whose role == "admin".
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# User detail view (admin-only — includes token usage & limits)
# ------------------------------------------------------------------ #
class AdminUserOut(BaseModel):
    """Full user record as seen from the Admin Panel."""

    id: str
    email: str
    full_name: str | None = None
    is_active: bool
    role: str
    custom_daily_token_limit: int | None = None

    # Today's live usage (joined from daily_token_usage)
    tokens_used_today: int = 0
    queries_today: int = 0

    # Aggregated lifetime stats
    total_documents: int = 0

    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AdminUserListResponse(BaseModel):
    """Paginated list of users for the admin panel."""

    users: list[AdminUserOut]
    total: int
    page: int
    page_size: int


# ------------------------------------------------------------------ #
# Patch request bodies
# ------------------------------------------------------------------ #
class PatchRoleRequest(BaseModel):
    """Request body for PATCH /admin/users/{user_id}/role."""

    role: str = Field(..., pattern="^(user|admin)$", description="Target role: 'user' or 'admin'.")


class PatchStatusRequest(BaseModel):
    """Request body for PATCH /admin/users/{user_id}/status."""

    is_active: bool = Field(..., description="Set false to ban/deactivate the user.")


class PatchLimitRequest(BaseModel):
    """Request body for PATCH /admin/users/{user_id}/limit."""

    custom_daily_token_limit: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Override the global daily token limit for this user. "
            "Pass null to fall back to the system default (DEFAULT_DAILY_TOKEN_LIMIT). "
            "Pass 0 to completely block the user from making queries."
        ),
    )


# ------------------------------------------------------------------ #
# Platform stats response
# ------------------------------------------------------------------ #
class DailyUsageStat(BaseModel):
    """Token & query counts for a specific date (used in the stats endpoint)."""

    usage_date: date
    total_tokens: int
    total_queries: int
    active_users: int


class PlatformStats(BaseModel):
    """High-level platform metrics for the Admin Dashboard."""

    total_users: int
    admin_users: int
    basic_users: int
    active_users: int
    inactive_users: int

    tokens_consumed_today: int
    queries_today: int
    active_users_today: int

    total_documents: int

    # Last 7 days breakdown (newest first)
    daily_usage: list[DailyUsageStat] = []

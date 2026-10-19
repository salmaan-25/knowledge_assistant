"""
routes/admin.py — Secure Admin Panel API for the Knowledge Assistant.

Endpoints (all require role == "admin"):
  GET    /admin/users                        — Paginated user list with live usage stats.
  GET    /admin/users/{user_id}              — Full profile + today's usage for one user.
  PATCH  /admin/users/{user_id}/role        — Promote/demote a user (user ↔ admin).
  PATCH  /admin/users/{user_id}/status      — Activate or ban a user.
  PATCH  /admin/users/{user_id}/limit       — Override daily token limit for a user.
  GET    /admin/stats                        — Platform-wide usage dashboard.

Security:
  - Every endpoint is gated by require_admin (re-fetches user from DB on every request).
  - An admin cannot remove their own admin role to prevent accidental lockout.
  - All mutations are logged at INFO level for an audit trail.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.database.models import DailyTokenUsage, Document, User, UserRole
from app.routes.auth import require_admin
from app.schemas.admin import (
    AdminUserListResponse,
    AdminUserOut,
    DailyUsageStat,
    PatchLimitRequest,
    PatchRoleRequest,
    PatchStatusRequest,
    PlatformStats,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin Panel"])


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _today_utc() -> date:
    return datetime.now(UTC).date()


async def _build_admin_user_out(user: User, db: AsyncSession) -> AdminUserOut:
    """
    Enrich a User ORM object with today's token usage and document count.
    Returns an AdminUserOut ready to be serialised.
    """
    today = _today_utc()

    # Today's token usage
    usage_result = await db.execute(
        select(DailyTokenUsage).where(
            DailyTokenUsage.user_id == user.id,
            DailyTokenUsage.usage_date == today,
        )
    )
    usage = usage_result.scalar_one_or_none()
    tokens_today = usage.tokens_used if usage else 0
    queries_today = usage.query_count if usage else 0

    # Total documents owned
    doc_count_result = await db.execute(
        select(func.count(Document.id)).where(Document.owner_id == user.id)
    )
    total_docs = doc_count_result.scalar() or 0

    return AdminUserOut(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        role=user.role,
        custom_daily_token_limit=user.custom_daily_token_limit,
        tokens_used_today=tokens_today,
        queries_today=queries_today,
        total_documents=total_docs,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


async def _fetch_user_or_404(user_id: str, db: AsyncSession) -> User:
    """Fetch a User by ID or raise HTTP 404."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found.",
        )
    return user


# ================================================================== #
# GET /admin/users
# ================================================================== #
@router.get(
    "/users",
    response_model=AdminUserListResponse,
    summary="List all users with live token usage (Admin only)",
)
async def list_users(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Users per page."),
    email_filter: str | None = Query(
        default=None, description="Partial email search (case-insensitive)."
    ),
    role_filter: str | None = Query(
        default=None,
        description="Filter by role: 'user' or 'admin'.",
        pattern="^(user|admin)$",
    ),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> AdminUserListResponse:
    """
    Return a paginated list of all users.

    Includes live token usage for today and total document count for each user.
    Supports optional filtering by email substring or role.
    """
    query = select(User).order_by(User.created_at.desc())

    if email_filter:
        query = query.where(User.email.ilike(f"%{email_filter}%"))
    if role_filter:
        query = query.where(User.role == role_filter)

    # Total count for pagination metadata
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination
    query = query.offset((page - 1) * page_size).limit(page_size)
    users_result = await db.execute(query)
    users = users_result.scalars().all()

    enriched = [await _build_admin_user_out(u, db) for u in users]

    return AdminUserListResponse(
        users=enriched,
        total=total,
        page=page,
        page_size=page_size,
    )


# ================================================================== #
# GET /admin/users/{user_id}
# ================================================================== #
@router.get(
    "/users/{user_id}",
    response_model=AdminUserOut,
    summary="Get full profile + usage for a specific user (Admin only)",
)
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> AdminUserOut:
    """Return the full admin view of a single user including today's usage stats."""
    user = await _fetch_user_or_404(user_id, db)
    return await _build_admin_user_out(user, db)


# ================================================================== #
# PATCH /admin/users/{user_id}/role
# ================================================================== #
@router.patch(
    "/users/{user_id}/role",
    response_model=AdminUserOut,
    summary="Promote or demote a user's role (Admin only)",
)
async def update_user_role(
    user_id: str,
    body: PatchRoleRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> AdminUserOut:
    """
    Change a user's role to 'user' or 'admin'.

    Security safeguard: an admin cannot remove their own admin role.
    This prevents accidental self-lockout.
    """
    user = await _fetch_user_or_404(user_id, db)

    # Prevent self-demotion
    if user.id == admin.id and body.role == UserRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "You cannot remove your own admin role. "
                "Ask another admin to demote you, or promote another user first."
            ),
        )

    old_role = user.role
    user.role = body.role
    await db.commit()
    await db.refresh(user)

    logger.info(
        "Admin %s changed role of user %s: %s → %s",
        admin.email,
        user.email,
        old_role,
        body.role,
    )

    return await _build_admin_user_out(user, db)


# ================================================================== #
# PATCH /admin/users/{user_id}/status
# ================================================================== #
@router.patch(
    "/users/{user_id}/status",
    response_model=AdminUserOut,
    summary="Activate or ban a user account (Admin only)",
)
async def update_user_status(
    user_id: str,
    body: PatchStatusRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> AdminUserOut:
    """
    Toggle a user's active status.

    Setting is_active=false immediately invalidates all future API calls from
    this user (the get_current_user dependency checks is_active on every request).
    """
    user = await _fetch_user_or_404(user_id, db)

    # Prevent self-ban
    if user.id == admin.id and not body.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot deactivate your own account.",
        )

    old_status = user.is_active
    user.is_active = body.is_active
    await db.commit()
    await db.refresh(user)

    action = "activated" if body.is_active else "deactivated/banned"
    logger.info(
        "Admin %s %s user %s (was_active=%s)",
        admin.email,
        action,
        user.email,
        old_status,
    )

    return await _build_admin_user_out(user, db)


# ================================================================== #
# PATCH /admin/users/{user_id}/limit
# ================================================================== #
@router.patch(
    "/users/{user_id}/limit",
    response_model=AdminUserOut,
    summary="Set a custom daily token limit for a user (Admin only)",
)
async def update_user_limit(
    user_id: str,
    body: PatchLimitRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> AdminUserOut:
    """
    Override the daily token limit for a specific user.

    - Pass a positive integer to set a custom cap (overrides DEFAULT_DAILY_TOKEN_LIMIT).
    - Pass 0 to completely block the user from making queries.
    - Pass null to revert to the global system default.

    Note: Admin users ignore this field entirely — they are always unlimited.
    """
    user = await _fetch_user_or_404(user_id, db)

    old_limit = user.custom_daily_token_limit
    user.custom_daily_token_limit = body.custom_daily_token_limit
    await db.commit()
    await db.refresh(user)

    logger.info(
        "Admin %s updated token limit for user %s: %s → %s",
        admin.email,
        user.email,
        old_limit,
        body.custom_daily_token_limit,
    )

    return await _build_admin_user_out(user, db)


# ================================================================== #
# GET /admin/stats
# ================================================================== #
@router.get(
    "/stats",
    response_model=PlatformStats,
    summary="Platform-wide usage metrics for the Admin Dashboard (Admin only)",
)
async def get_platform_stats(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> PlatformStats:
    """
    Return aggregated platform metrics for the Admin Dashboard overview.

    Includes:
    - User counts broken down by role and active status.
    - Today's aggregate token and query usage.
    - Total document count across all users.
    - Last 7 days of daily usage breakdown.
    """
    today = _today_utc()

    # --- User counts ---
    total_users_r = await db.execute(select(func.count(User.id)))
    total_users = total_users_r.scalar() or 0

    admin_users_r = await db.execute(select(func.count(User.id)).where(User.role == UserRole.ADMIN))
    admin_users = admin_users_r.scalar() or 0

    active_users_r = await db.execute(
        select(func.count(User.id)).where(User.is_active == True)  # noqa: E712
    )
    active_users = active_users_r.scalar() or 0

    # --- Today's platform totals ---
    today_usage_r = await db.execute(
        select(
            func.coalesce(func.sum(DailyTokenUsage.tokens_used), 0),
            func.coalesce(func.sum(DailyTokenUsage.query_count), 0),
            func.count(DailyTokenUsage.user_id),
        ).where(DailyTokenUsage.usage_date == today)
    )
    tokens_today, queries_today, active_today = today_usage_r.one()

    # --- Total documents ---
    total_docs_r = await db.execute(select(func.count(Document.id)))
    total_docs = total_docs_r.scalar() or 0

    # --- Last 7 days breakdown ---
    from datetime import timedelta

    seven_days_ago = today - timedelta(days=6)
    daily_r = await db.execute(
        select(
            DailyTokenUsage.usage_date,
            func.sum(DailyTokenUsage.tokens_used).label("total_tokens"),
            func.sum(DailyTokenUsage.query_count).label("total_queries"),
            func.count(DailyTokenUsage.user_id).label("active_users"),
        )
        .where(DailyTokenUsage.usage_date >= seven_days_ago)
        .group_by(DailyTokenUsage.usage_date)
        .order_by(DailyTokenUsage.usage_date.desc())
    )
    daily_rows = daily_r.all()

    daily_stats = [
        DailyUsageStat(
            usage_date=row.usage_date,
            total_tokens=row.total_tokens or 0,
            total_queries=row.total_queries or 0,
            active_users=row.active_users or 0,
        )
        for row in daily_rows
    ]

    return PlatformStats(
        total_users=total_users,
        admin_users=admin_users,
        basic_users=total_users - admin_users,
        active_users=active_users,
        inactive_users=total_users - active_users,
        tokens_consumed_today=tokens_today,
        queries_today=queries_today,
        active_users_today=active_today,
        total_documents=total_docs,
        daily_usage=daily_stats,
    )

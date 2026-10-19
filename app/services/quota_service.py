"""
services/quota_service.py — Token quota enforcement for basic users.

How it works:
  1. BEFORE a query runs: check_quota() verifies the user has tokens remaining today.
     - Admins always pass.
     - If DEFAULT_DAILY_TOKEN_LIMIT == 0, quota is disabled globally.
     - Returns HTTP 429 (Too Many Requests) if the user is over their limit.

  2. AFTER a query runs: record_usage() increments tokens_used and query_count
     for today's DailyTokenUsage row.  Uses an upsert pattern so we never have
     duplicate (user_id, date) rows.

Token counting:
  - Actual token counts should come from the LLM response metadata when available.
  - Pass estimated_tokens=0 to record_usage when the exact count is unavailable;
    the function will use a configurable fallback estimate instead.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.models import DailyTokenUsage, User

logger = logging.getLogger(__name__)
settings = get_settings()

# Fallback token estimate per query when the LLM doesn't return usage metadata.
_FALLBACK_TOKENS_PER_QUERY = 1000


def _today_utc() -> date:
    return datetime.now(UTC).date()


def _effective_limit(user: User) -> int:
    """
    Return the token limit that applies to this user.

    Priority (highest to lowest):
      1. Admin  → unlimited (returns -1 as sentinel)
      2. user.custom_daily_token_limit is set  → use that value (0 = blocked)
      3. Global DEFAULT_DAILY_TOKEN_LIMIT from settings (<= 0 means disabled)
    """
    if user.is_admin:
        return -1  # sentinel: unlimited

    if user.custom_daily_token_limit is not None:
        return user.custom_daily_token_limit

    if settings.DEFAULT_DAILY_TOKEN_LIMIT <= 0:
        return -1  # global limit disabled

    return settings.DEFAULT_DAILY_TOKEN_LIMIT


async def _get_or_create_usage(user_id: str, db: AsyncSession) -> DailyTokenUsage:
    """
    Fetch today's DailyTokenUsage row for the user, creating it if absent.
    The caller is responsible for committing the session.
    """
    today = _today_utc()
    result = await db.execute(
        select(DailyTokenUsage).where(
            DailyTokenUsage.user_id == user_id,
            DailyTokenUsage.usage_date == today,
        )
    )
    usage = result.scalar_one_or_none()

    if usage is None:
        usage = DailyTokenUsage(
            user_id=user_id,
            usage_date=today,
            tokens_used=0,
            query_count=0,
        )
        db.add(usage)
        await db.flush()  # get the generated ID without committing the outer transaction

    return usage


async def check_quota(user: User, db: AsyncSession) -> None:
    """
    Pre-flight quota check. Call this BEFORE invoking the LLM.

    Raises HTTP 429 if the user has exhausted their daily token allowance.
    Admins (-1) and setups with limit < 0 pass immediately.
    A limit of 0 blocks the user immediately.
    """
    limit = _effective_limit(user)

    # -1 means admin (unlimited) or global limit is disabled
    if limit < 0:
        return

    usage = await _get_or_create_usage(user.id, db)

    if usage.tokens_used >= limit:
        logger.warning(
            "Quota exceeded: user=%s tokens_used=%d limit=%d",
            user.id,
            usage.tokens_used,
            limit,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"You have reached your daily token limit "
                f"({usage.tokens_used:,} / {limit:,} tokens). "
                f"Your allowance resets at midnight UTC."
            ),
        )


async def record_usage(
    user: User,
    db: AsyncSession,
    *,
    tokens: int = 0,
    commit: bool = True,
) -> None:
    """
    Post-query usage recorder. Call this AFTER the LLM response is received.

    If `tokens` is 0, a fallback estimate (_FALLBACK_TOKENS_PER_QUERY) is used
    so the quota is still approximately tracked even without exact counts.

    Admins are silently skipped — their usage is never written to the table.
    """
    if user.is_admin:
        return

    actual_tokens = tokens if tokens > 0 else _FALLBACK_TOKENS_PER_QUERY

    usage = await _get_or_create_usage(user.id, db)
    usage.tokens_used += actual_tokens
    usage.query_count += 1
    if commit:
        await db.commit()

    logger.debug(
        "Usage recorded: user=%s tokens_added=%d total_today=%d queries=%d",
        user.id,
        actual_tokens,
        usage.tokens_used,
        usage.query_count,
    )


async def get_user_quota_info(user: User, db: AsyncSession) -> dict[str, int]:
    """Return today's usage and effective limit for a user."""
    limit = _effective_limit(user)
    if user.is_admin:
        return {
            "tokens_used_today": 0,
            "daily_token_limit": -1,
            "queries_today": 0,
        }
    usage = await _get_or_create_usage(user.id, db)
    return {
        "tokens_used_today": usage.tokens_used,
        "daily_token_limit": limit,
        "queries_today": usage.query_count,
    }

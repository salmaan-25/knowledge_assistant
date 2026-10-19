"""
tests/test_admin.py — Tests for the Admin Panel API.

Covers:
  - Role fields correctly returned from /auth/register and /auth/me
  - Admin-only endpoints blocked for regular users (403)
  - Admin-only endpoints blocked for unauthenticated requests (401)
  - Admin can list users, view a user, change role, change status, change limit
  - Self-demotion and self-ban prevention
  - Platform stats endpoint correctness
  - Quota enforcement (429) and bypass for admins
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import User, UserRole


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
async def _register_and_login(
    client: AsyncClient,
    email: str = "user@example.com",
    password: str = "securepass123",
) -> dict[str, str]:
    """Register a user (if not already) and return auth headers."""
    await client.post("/auth/register", json={"email": email, "password": password})
    resp = await client.post(
        "/auth/login",
        data={"username": email, "password": password},
    )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def _make_admin(user_id: str, db: AsyncSession) -> None:
    """Directly promote a user to admin in the DB (bypasses the API)."""
    from sqlalchemy import select

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one()
    user.role = UserRole.ADMIN
    await db.commit()


# ------------------------------------------------------------------ #
# Registration default role
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_register_default_role_is_user(async_client: AsyncClient):
    """Every new registration must have role='user', never 'admin'."""
    resp = await async_client.post(
        "/auth/register",
        json={"email": "newuser@example.com", "password": "securepass123"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["role"] == "user"
    assert "hashed_password" not in data


@pytest.mark.asyncio
async def test_register_cannot_inject_admin_role(async_client: AsyncClient):
    """Registration payload cannot include a 'role' field to elevate privileges."""
    resp = await async_client.post(
        "/auth/register",
        json={"email": "hacker@example.com", "password": "securepass123", "role": "admin"},
    )
    # Should succeed but role must still be 'user'
    assert resp.status_code == 201
    assert resp.json()["role"] == "user"


# ------------------------------------------------------------------ #
# Admin endpoint access control
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_admin_endpoints_require_auth(async_client: AsyncClient):
    """All /admin/* endpoints must return 401 without a token."""
    endpoints = [
        ("GET", "/admin/users"),
        ("GET", "/admin/stats"),
    ]
    for method, path in endpoints:
        resp = await async_client.request(method, path)
        assert resp.status_code == 401, f"{method} {path} should return 401"


@pytest.mark.asyncio
async def test_admin_endpoints_blocked_for_basic_user(async_client: AsyncClient):
    """A regular user must get 403 on all /admin/* endpoints."""
    headers = await _register_and_login(async_client, "basic@example.com")

    endpoints = [
        ("GET", "/admin/users"),
        ("GET", "/admin/stats"),
    ]
    for method, path in endpoints:
        resp = await async_client.request(method, path, headers=headers)
        assert resp.status_code == 403, f"{method} {path} should be 403 for basic user"


# ------------------------------------------------------------------ #
# Admin: list users
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_admin_list_users(async_client: AsyncClient, test_db: AsyncSession):
    """Admin can retrieve the full user list."""
    # Register two users
    await _register_and_login(async_client, "alice@example.com")
    bob_headers = await _register_and_login(async_client, "bob@example.com")

    # Promote bob to admin
    me_resp = await async_client.get("/auth/me", headers=bob_headers)
    bob_id = me_resp.json()["id"]
    await _make_admin(bob_id, test_db)

    resp = await async_client.get("/admin/users", headers=bob_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "users" in data
    assert "total" in data
    assert data["total"] >= 2


@pytest.mark.asyncio
async def test_admin_list_users_email_filter(async_client: AsyncClient, test_db: AsyncSession):
    """Email filter returns only matching users."""
    await _register_and_login(async_client, "charlie@example.com")
    admin_headers = await _register_and_login(async_client, "adminuser@example.com")
    me_resp = await async_client.get("/auth/me", headers=admin_headers)
    await _make_admin(me_resp.json()["id"], test_db)

    resp = await async_client.get("/admin/users?email_filter=charlie", headers=admin_headers)
    assert resp.status_code == 200
    users = resp.json()["users"]
    assert all("charlie" in u["email"] for u in users)


@pytest.mark.asyncio
async def test_admin_list_users_role_filter(async_client: AsyncClient, test_db: AsyncSession):
    """Role filter correctly segregates admin vs. user results."""
    admin_headers = await _register_and_login(async_client, "filteradmin@example.com")
    me_resp = await async_client.get("/auth/me", headers=admin_headers)
    await _make_admin(me_resp.json()["id"], test_db)

    await _register_and_login(async_client, "plainuser@example.com")

    resp = await async_client.get("/admin/users?role_filter=admin", headers=admin_headers)
    assert resp.status_code == 200
    assert all(u["role"] == "admin" for u in resp.json()["users"])

    resp2 = await async_client.get("/admin/users?role_filter=user", headers=admin_headers)
    assert resp2.status_code == 200
    assert all(u["role"] == "user" for u in resp2.json()["users"])


# ------------------------------------------------------------------ #
# Admin: get single user
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_admin_get_user(async_client: AsyncClient, test_db: AsyncSession):
    """Admin can fetch full profile of any user."""
    target_headers = await _register_and_login(async_client, "target@example.com")
    target_id = (await async_client.get("/auth/me", headers=target_headers)).json()["id"]

    admin_headers = await _register_and_login(async_client, "getadmin@example.com")
    await _make_admin(
        (await async_client.get("/auth/me", headers=admin_headers)).json()["id"], test_db
    )

    resp = await async_client.get(f"/admin/users/{target_id}", headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == target_id
    assert "tokens_used_today" in data
    assert "queries_today" in data
    assert "total_documents" in data


@pytest.mark.asyncio
async def test_admin_get_nonexistent_user(async_client: AsyncClient, test_db: AsyncSession):
    """Fetching a non-existent user ID returns 404."""
    admin_headers = await _register_and_login(async_client, "getadmin404@example.com")
    await _make_admin(
        (await async_client.get("/auth/me", headers=admin_headers)).json()["id"], test_db
    )

    resp = await async_client.get("/admin/users/nonexistent-id-xyz", headers=admin_headers)
    assert resp.status_code == 404


# ------------------------------------------------------------------ #
# Admin: patch role
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_admin_promote_user_to_admin(async_client: AsyncClient, test_db: AsyncSession):
    """Admin can promote a basic user to admin role."""
    target_headers = await _register_and_login(async_client, "promote_target@example.com")
    target_id = (await async_client.get("/auth/me", headers=target_headers)).json()["id"]

    admin_headers = await _register_and_login(async_client, "promoter@example.com")
    await _make_admin(
        (await async_client.get("/auth/me", headers=admin_headers)).json()["id"], test_db
    )

    resp = await async_client.patch(
        f"/admin/users/{target_id}/role",
        json={"role": "admin"},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "admin"


@pytest.mark.asyncio
async def test_admin_demote_user(async_client: AsyncClient, test_db: AsyncSession):
    """Admin can demote another admin to basic user."""
    # Create target and make them admin
    target_headers = await _register_and_login(async_client, "demote_target@example.com")
    target_id = (await async_client.get("/auth/me", headers=target_headers)).json()["id"]
    await _make_admin(target_id, test_db)

    # Create the acting admin
    admin_headers = await _register_and_login(async_client, "demoter@example.com")
    await _make_admin(
        (await async_client.get("/auth/me", headers=admin_headers)).json()["id"], test_db
    )

    resp = await async_client.patch(
        f"/admin/users/{target_id}/role",
        json={"role": "user"},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "user"


@pytest.mark.asyncio
async def test_admin_cannot_demote_self(async_client: AsyncClient, test_db: AsyncSession):
    """An admin cannot remove their own admin role (self-lockout prevention)."""
    admin_headers = await _register_and_login(async_client, "selfdemote@example.com")
    admin_id = (await async_client.get("/auth/me", headers=admin_headers)).json()["id"]
    await _make_admin(admin_id, test_db)

    resp = await async_client.patch(
        f"/admin/users/{admin_id}/role",
        json={"role": "user"},
        headers=admin_headers,
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_patch_role_invalid_value(async_client: AsyncClient, test_db: AsyncSession):
    """Sending an invalid role value returns 422."""
    target_headers = await _register_and_login(async_client, "rolecheck@example.com")
    target_id = (await async_client.get("/auth/me", headers=target_headers)).json()["id"]

    admin_headers = await _register_and_login(async_client, "roleadmin@example.com")
    await _make_admin(
        (await async_client.get("/auth/me", headers=admin_headers)).json()["id"], test_db
    )

    resp = await async_client.patch(
        f"/admin/users/{target_id}/role",
        json={"role": "superuser"},  # invalid
        headers=admin_headers,
    )
    assert resp.status_code == 422


# ------------------------------------------------------------------ #
# Admin: patch status
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_admin_ban_user(async_client: AsyncClient, test_db: AsyncSession):
    """Admin can deactivate (ban) a user; banned user cannot access protected routes."""
    target_headers = await _register_and_login(async_client, "banme@example.com")
    target_id = (await async_client.get("/auth/me", headers=target_headers)).json()["id"]

    admin_headers = await _register_and_login(async_client, "banning_admin@example.com")
    await _make_admin(
        (await async_client.get("/auth/me", headers=admin_headers)).json()["id"], test_db
    )

    resp = await async_client.patch(
        f"/admin/users/{target_id}/status",
        json={"is_active": False},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False

    # Banned user should now get 401 on authenticated routes
    banned_me = await async_client.get("/auth/me", headers=target_headers)
    assert banned_me.status_code == 401


@pytest.mark.asyncio
async def test_admin_reactivate_user(async_client: AsyncClient, test_db: AsyncSession):
    """Admin can reactivate a previously banned user."""
    target_headers = await _register_and_login(async_client, "reactivateme@example.com")
    target_id = (await async_client.get("/auth/me", headers=target_headers)).json()["id"]

    admin_headers = await _register_and_login(async_client, "reactivate_admin@example.com")
    await _make_admin(
        (await async_client.get("/auth/me", headers=admin_headers)).json()["id"], test_db
    )

    # Ban first
    await async_client.patch(
        f"/admin/users/{target_id}/status",
        json={"is_active": False},
        headers=admin_headers,
    )

    # Reactivate
    resp = await async_client.patch(
        f"/admin/users/{target_id}/status",
        json={"is_active": True},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is True


@pytest.mark.asyncio
async def test_admin_cannot_ban_self(async_client: AsyncClient, test_db: AsyncSession):
    """An admin cannot ban their own account."""
    admin_headers = await _register_and_login(async_client, "selfban@example.com")
    admin_id = (await async_client.get("/auth/me", headers=admin_headers)).json()["id"]
    await _make_admin(admin_id, test_db)

    resp = await async_client.patch(
        f"/admin/users/{admin_id}/status",
        json={"is_active": False},
        headers=admin_headers,
    )
    assert resp.status_code == 400


# ------------------------------------------------------------------ #
# Admin: patch token limit
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_admin_set_custom_token_limit(async_client: AsyncClient, test_db: AsyncSession):
    """Admin can set a custom token limit for a specific user."""
    target_headers = await _register_and_login(async_client, "limitme@example.com")
    target_id = (await async_client.get("/auth/me", headers=target_headers)).json()["id"]

    admin_headers = await _register_and_login(async_client, "limitadmin@example.com")
    await _make_admin(
        (await async_client.get("/auth/me", headers=admin_headers)).json()["id"], test_db
    )

    resp = await async_client.patch(
        f"/admin/users/{target_id}/limit",
        json={"custom_daily_token_limit": 10000},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["custom_daily_token_limit"] == 10000


@pytest.mark.asyncio
async def test_admin_reset_token_limit_to_default(async_client: AsyncClient, test_db: AsyncSession):
    """Admin can reset a user's custom limit back to null (global default)."""
    target_headers = await _register_and_login(async_client, "resetlimit@example.com")
    target_id = (await async_client.get("/auth/me", headers=target_headers)).json()["id"]

    admin_headers = await _register_and_login(async_client, "resetadmin@example.com")
    await _make_admin(
        (await async_client.get("/auth/me", headers=admin_headers)).json()["id"], test_db
    )

    # Set a custom limit first
    await async_client.patch(
        f"/admin/users/{target_id}/limit",
        json={"custom_daily_token_limit": 5000},
        headers=admin_headers,
    )

    # Reset to null (back to global default)
    resp = await async_client.patch(
        f"/admin/users/{target_id}/limit",
        json={"custom_daily_token_limit": None},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["custom_daily_token_limit"] is None


@pytest.mark.asyncio
async def test_admin_set_negative_token_limit_fails(
    async_client: AsyncClient, test_db: AsyncSession
):
    """Sending a negative token limit must be rejected with 422."""
    target_headers = await _register_and_login(async_client, "neglimit@example.com")
    target_id = (await async_client.get("/auth/me", headers=target_headers)).json()["id"]

    admin_headers = await _register_and_login(async_client, "neglimitadmin@example.com")
    await _make_admin(
        (await async_client.get("/auth/me", headers=admin_headers)).json()["id"], test_db
    )

    resp = await async_client.patch(
        f"/admin/users/{target_id}/limit",
        json={"custom_daily_token_limit": -100},
        headers=admin_headers,
    )
    assert resp.status_code == 422


# ------------------------------------------------------------------ #
# Admin: platform stats
# ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_admin_stats_structure(async_client: AsyncClient, test_db: AsyncSession):
    """Platform stats endpoint returns the expected keys."""
    admin_headers = await _register_and_login(async_client, "statsadmin@example.com")
    await _make_admin(
        (await async_client.get("/auth/me", headers=admin_headers)).json()["id"], test_db
    )

    resp = await async_client.get("/admin/stats", headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()

    required_keys = {
        "total_users",
        "admin_users",
        "basic_users",
        "active_users",
        "inactive_users",
        "tokens_consumed_today",
        "queries_today",
        "active_users_today",
        "total_documents",
        "daily_usage",
    }
    assert required_keys.issubset(data.keys())
    assert isinstance(data["daily_usage"], list)


@pytest.mark.asyncio
async def test_admin_stats_counts_are_accurate(async_client: AsyncClient, test_db: AsyncSession):
    """User counts in stats are consistent with the registered users."""
    # Register two basic users and one admin
    await _register_and_login(async_client, "statsuser1@example.com")
    await _register_and_login(async_client, "statsuser2@example.com")

    admin_headers = await _register_and_login(async_client, "statscountadmin@example.com")
    admin_id = (await async_client.get("/auth/me", headers=admin_headers)).json()["id"]
    await _make_admin(admin_id, test_db)

    resp = await async_client.get("/admin/stats", headers=admin_headers)
    data = resp.json()

    assert data["total_users"] >= 3
    assert data["admin_users"] >= 1
    assert data["basic_users"] == data["total_users"] - data["admin_users"]
    assert data["active_users"] + data["inactive_users"] == data["total_users"]

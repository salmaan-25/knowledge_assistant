"""
tests/test_auth.py — Tests for user registration and login endpoints.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_success(async_client: AsyncClient):
    """A new user can register and receive their profile back."""
    resp = await async_client.post(
        "/auth/register",
        json={"email": "new@example.com", "password": "securepass123"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["email"] == "new@example.com"
    assert "id" in data
    assert "hashed_password" not in data


@pytest.mark.asyncio
async def test_register_duplicate_email(async_client: AsyncClient):
    """Registering the same email twice returns 409."""
    payload = {"email": "dupe@example.com", "password": "securepass123"}
    await async_client.post("/auth/register", json=payload)
    resp = await async_client.post("/auth/register", json=payload)
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_register_short_password(async_client: AsyncClient):
    """Registration with a password shorter than 8 characters is rejected."""
    resp = await async_client.post(
        "/auth/register",
        json={"email": "short@example.com", "password": "abc"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_login_success(async_client: AsyncClient):
    """A registered user can log in and receive a JWT access token."""
    await async_client.post(
        "/auth/register",
        json={"email": "login@example.com", "password": "securepass123"},
    )
    resp = await async_client.post(
        "/auth/login",
        data={"username": "login@example.com", "password": "securepass123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(async_client: AsyncClient):
    """Wrong password returns 401."""
    await async_client.post(
        "/auth/register",
        json={"email": "wrongpw@example.com", "password": "correctpass123"},
    )
    resp = await async_client.post(
        "/auth/login",
        data={"username": "wrongpw@example.com", "password": "wrongpass"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_endpoint_requires_auth(async_client: AsyncClient):
    """GET /auth/me returns 401 without a token."""
    resp = await async_client.get("/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_endpoint_authenticated(async_client: AsyncClient, auth_headers: dict):
    """GET /auth/me returns the current user's profile when authenticated."""
    resp = await async_client.get("/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "email" in data
    assert "id" in data
    # New field — every registered user defaults to role='user'
    assert data["role"] == "user"
    assert "hashed_password" not in data

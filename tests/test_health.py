"""
tests/test_health.py — Tests for the /health endpoint.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200(async_client: AsyncClient):
    """The health endpoint must return HTTP 200."""
    response = await async_client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_shape(async_client: AsyncClient):
    """The health response must contain status, service, and version keys."""
    response = await async_client.get("/health")
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data
    assert "environment" in data

"""
services/auth_service.py — Password hashing, JWT token creation and verification.

Uses:
  - passlib[bcrypt] for password hashing (never store plain-text passwords).
  - python-jose for JWT encoding/decoding.

All functions are synchronous (no I/O).  Database interactions are in the routes.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from jose import jwt
from passlib.context import CryptContext

from app.config import get_settings
from app.schemas.auth import TokenData

logger = logging.getLogger(__name__)
settings = get_settings()

# ------------------------------------------------------------------ #
# Password hashing
# ------------------------------------------------------------------ #
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
ALGORITHM = "HS256"


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of the given plain-text password."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if the plain-text password matches the stored hash."""
    return _pwd_context.verify(plain, hashed)


# ------------------------------------------------------------------ #
# JWT
# ------------------------------------------------------------------ #
def create_access_token(user_id: str, expires_delta: timedelta | None = None) -> str:
    """
    Create a signed JWT access token.

    The token payload contains:
      - sub: user ID
      - exp: expiry timestamp
      - iat: issued-at timestamp
    """
    now = datetime.now(UTC)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    payload = {
        "sub": user_id,
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> TokenData:
    """
    Decode and validate a JWT access token.

    Raises JWTError (from python-jose) on invalid or expired tokens.
    """
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    user_id: str | None = payload.get("sub")
    return TokenData(user_id=user_id)


def get_token_expires_in() -> int:
    """Return the token lifetime in seconds."""
    return ACCESS_TOKEN_EXPIRE_MINUTES * 60

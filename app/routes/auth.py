"""
routes/auth.py — User registration, login, and JWT-protected /me endpoint.

Endpoints:
  POST /auth/register   — Create a new user account.
  POST /auth/login      — Authenticate and receive a JWT.
  GET  /auth/me         — Return the current user's profile.

The get_current_user dependency is defined here and re-used by all other
protected routes across the application.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.database.models import User
from app.schemas.auth import Token, UserCreate, UserOut
from app.services.auth_service import (
    create_access_token,
    get_token_expires_in,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ------------------------------------------------------------------ #
# Dependency: get_current_user
# ------------------------------------------------------------------ #
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency that validates the Bearer JWT and returns the User ORM object.

    Used as a Depends() in any route that requires authentication.
    Raises HTTP 401 on invalid/expired token or if the user no longer exists.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        from app.services.auth_service import decode_access_token

        token_data = decode_access_token(token)
    except JWTError:
        raise credentials_exc from None

    if not token_data.user_id:
        raise credentials_exc

    result = await db.execute(select(User).where(User.id == token_data.user_id))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise credentials_exc

    return user


# ------------------------------------------------------------------ #
# POST /auth/register
# ------------------------------------------------------------------ #
@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(
    body: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> UserOut:
    """
    Create a new user account.

    - Checks for duplicate email.
    - Hashes password with bcrypt before storing.
    - Returns the public user profile (no password hash).
    """
    # Check duplicate
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
    )
    db.add(user)
    await db.flush()  # get generated ID before commit
    await db.refresh(user)

    logger.info("New user registered: %s", user.email)
    return UserOut.model_validate(user)


# ------------------------------------------------------------------ #
# POST /auth/login
# ------------------------------------------------------------------ #
@router.post(
    "/login",
    response_model=Token,
    summary="Authenticate and receive a JWT access token",
)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> Token:
    """
    OAuth2 password flow login.

    Returns a JWT access token on success.
    Raises HTTP 401 on invalid credentials.
    """
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account is inactive.",
        )

    token = create_access_token(user.id)
    logger.info("User logged in: %s", user.email)

    return Token(
        access_token=token,
        token_type="bearer",
        expires_in=get_token_expires_in(),
    )


# ------------------------------------------------------------------ #
# GET /auth/me
# ------------------------------------------------------------------ #
@router.get(
    "/me",
    response_model=UserOut,
    summary="Get the current authenticated user's profile",
)
async def get_me(current_user: User = Depends(get_current_user)) -> UserOut:
    """Return the authenticated user's profile."""
    return UserOut.model_validate(current_user)

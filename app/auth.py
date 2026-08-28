"""
Authentication & authorization.

- Passwords are hashed with bcrypt (never stored in plain text).
- On login, the server creates an AuthSession row (a revocable
  server-side session) and issues a JWT whose `jti` claim points at
  that row's id. The JWT is set as an HttpOnly, SameSite=Lax cookie —
  it is never exposed to, or readable by, frontend JavaScript.
- Every protected endpoint depends on `get_current_user`, which:
    1. Reads the cookie (never a client-supplied user_id/header).
    2. Verifies the JWT signature and expiry.
    3. Confirms the session still exists server-side (i.e. hasn't been
       logged out / revoked) and hasn't itself expired.
    4. Loads and returns the corresponding User row.
  If any step fails, the request is rejected with 401 before it ever
  reaches business logic — so a forged or replayed user id can never
  reach the RAG, document, or chat-history layers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.db_models import AuthSession, User

logger = logging.getLogger("rag_chatbot.auth")

COOKIE_NAME = "access_token"
JWT_ALGORITHM = "HS256"


# --------------------------------------------------------------------------
# Password hashing
# --------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed hash — never crash the login endpoint over it.
        return False


# --------------------------------------------------------------------------
# Sessions + JWTs
# --------------------------------------------------------------------------

def create_session_and_token(db: Session, user: User) -> tuple[str, datetime]:
    """
    Create a new server-side AuthSession for `user` and return
    (signed_jwt, expires_at). The JWT's `jti` is the AuthSession id,
    so it can be revoked independently of its natural expiry.
    """
    settings = get_settings()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)

    session_row = AuthSession(user_id=user.id, expires_at=expires_at)
    db.add(session_row)
    db.commit()
    db.refresh(session_row)

    payload = {
        "sub": user.id,
        "jti": session_row.id,
        "exp": expires_at,
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=JWT_ALGORITHM)
    return token, expires_at


def revoke_session(db: Session, session_id: str) -> None:
    """Delete a server-side session row, immediately invalidating its JWT."""
    row = db.get(AuthSession, session_id)
    if row is not None:
        db.delete(row)
        db.commit()


def _decode_token(token: str) -> dict:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Your session has expired. Please log in again.") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials.") from exc


def _resolve_user_from_cookie(token: str | None, db: Session) -> User | None:
    if not token:
        return None
    try:
        payload = _decode_token(token)
    except HTTPException:
        return None

    session_id = payload.get("jti")
    user_id = payload.get("sub")
    if not session_id or not user_id:
        return None

    session_row = db.get(AuthSession, session_id)
    if session_row is None or session_row.user_id != user_id:
        # Session was revoked (logout) or the token was tampered with.
        return None

    if session_row.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        db.delete(session_row)
        db.commit()
        return None

    user = db.get(User, user_id)
    return user


def get_current_user(
    access_token: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
) -> User:
    """Required-auth dependency: raises 401 if there is no valid, non-revoked session."""
    user = _resolve_user_from_cookie(access_token, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
        )
    return user


def get_current_user_optional(
    access_token: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
) -> User | None:
    """Optional-auth dependency used only for page routes that redirect rather than 401."""
    return _resolve_user_from_cookie(access_token, db)

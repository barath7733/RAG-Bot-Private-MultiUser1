"""
Database engine and session management.

Uses SQLAlchemy against a local SQLite file by default (configurable via
DATABASE_URL, so it can be pointed at Postgres/MySQL in production without
code changes). Stores:

- Users (accounts, hashed passwords)
- Sessions (server-side revocable login sessions backing the JWT cookie)
- Documents (per-user metadata for uploaded/indexed PDFs)
- Chat sessions + messages (per-user chat history)

This is the single source of truth for "who owns what" — every
user-scoped query in the app goes through the owner_user_id columns
defined in app/db_models.py.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    settings = get_settings()
    url = settings.database_url

    # Make sure the parent directory exists for file-based SQLite URLs.
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        db_path = url.replace("sqlite:///", "", 1)
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, future=True)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create all tables if they don't already exist. Safe to call on every startup."""
    from app import db_models  # noqa: F401  (ensures models are registered on Base)

    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """FastAPI dependency yielding a request-scoped database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

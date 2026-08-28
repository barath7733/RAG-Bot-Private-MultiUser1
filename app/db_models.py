"""
SQLAlchemy ORM models.

Every piece of user-owned data (documents, chat sessions, chat
messages) carries an `owner_user_id` foreign key back to `users.id`.
No query in this codebase should ever read/update/delete a row from
these tables without filtering on the authenticated caller's user id
— see app/rag.py and app/main.py, where every function that touches
one of these tables takes the current user's id as a required
parameter.

User IDs are opaque UUID4 hex strings — never the user's email — so
the primary internal identifier can't be guessed or reused if the
email changes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    sessions: Mapped[list["AuthSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    documents: Mapped[list["Document"]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    chat_sessions: Mapped[list["ChatSession"]] = relationship(back_populates="owner", cascade="all, delete-orphan")


class AuthSession(Base):
    """
    A server-side record of an active login session. The JWT issued at
    login embeds this row's id as its `jti` claim, so logging out (or an
    admin revoking a session) deletes the row here and the token stops
    working immediately — even though the JWT itself hasn't expired yet.
    """

    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user: Mapped["User"] = relationship(back_populates="sessions")


class Document(Base):
    """Per-user metadata for an uploaded/indexed PDF (the vectors themselves live in Pinecone)."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # matches the Pinecone document_id
    owner_user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id"), nullable=False, index=True)
    document_name: Mapped[str] = mapped_column(String(512), nullable=False)
    num_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    owner: Mapped["User"] = relationship(back_populates="documents")


class ChatSession(Base):
    """A single conversation thread belonging to exactly one user."""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    owner_user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    owner: Mapped["User"] = relationship(back_populates="chat_sessions")
    messages: Mapped[list["ChatMessageRow"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessageRow.id"
    )


class ChatMessageRow(Base):
    """A single message (user or assistant turn) within a chat session."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(32), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    mode_used: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")

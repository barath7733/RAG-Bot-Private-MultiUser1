"""
RAG orchestration layer.

Ties together PDF extraction, chunking, embedding, Pinecone storage,
retrieval, and Groq generation. Document metadata (name, chunk count,
upload time, size) and chat history now live in the SQL database,
keyed by `owner_user_id` — the vectors themselves live in Pinecone,
partitioned by per-user namespace (see app/pinecone_db.py).

Every public function below takes the authenticated user's id as a
required parameter and uses it both to scope Pinecone access and to
filter/authorize every database row it touches. There is no code path
here that can return or mutate another user's data.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app import groq_client, pinecone_db, web_search
from app.chunking import chunk_pages
from app.config import get_settings
from app.db_models import ChatMessageRow, ChatSession, Document
from app.embeddings import embed_query, embed_texts
from app.models import (
    ChatMode,
    ChatMessage,
    ChatHistoryMessage,
    ChatSessionDetail,
    ChatSessionSummary,
    DocumentInfo,
    SourceChunk,
    WebSource,
)
from app.pdf_processor import extract_text_from_pdf, PDFProcessingError
from app.web_search import WebSearchError

logger = logging.getLogger("rag_chatbot.rag")


class RAGError(Exception):
    """Raised for any user-facing RAG pipeline failure."""


# --------------------------------------------------------------------------
# Document registry (per-user, backed by the `documents` table)
# --------------------------------------------------------------------------

def _to_document_info(row: Document) -> DocumentInfo:
    return DocumentInfo(
        document_id=row.id,
        document_name=row.document_name,
        num_chunks=row.num_chunks,
        uploaded_at=row.uploaded_at.isoformat(),
        size_bytes=row.size_bytes,
    )


def list_documents(db: Session, user_id: str) -> list[DocumentInfo]:
    rows = db.query(Document).filter(Document.owner_user_id == user_id).order_by(Document.uploaded_at.desc()).all()
    return [_to_document_info(row) for row in rows]


def find_duplicate_document(db: Session, user_id: str, document_name: str, size_bytes: int) -> DocumentInfo | None:
    """Detect an existing document owned by this user with the same name and size."""
    row = (
        db.query(Document)
        .filter(
            Document.owner_user_id == user_id,
            Document.document_name == document_name,
            Document.size_bytes == size_bytes,
        )
        .first()
    )
    return _to_document_info(row) if row else None


def get_owned_document(db: Session, user_id: str, document_id: str) -> Document | None:
    """
    Fetch a document row only if it belongs to `user_id`. Returns None
    both when the document doesn't exist and when it belongs to
    someone else — callers must not be able to distinguish the two, or
    the endpoint would leak whether another user's document id exists.
    """
    return (
        db.query(Document)
        .filter(Document.id == document_id, Document.owner_user_id == user_id)
        .first()
    )


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------

def ingest_pdf(
    db: Session,
    user_id: str,
    file_bytes: bytes,
    original_filename: str,
    replace_existing: bool = False,
) -> DocumentInfo:
    """
    Full ingestion pipeline: extract -> clean & chunk -> embed -> upsert
    to the user's Pinecone namespace -> register in the database under
    that same user's ownership.
    """
    settings = get_settings()

    duplicate = find_duplicate_document(db, user_id, original_filename, len(file_bytes))
    if duplicate and not replace_existing:
        raise RAGError(
            f"A document named '{original_filename}' with the same content is already "
            "indexed. Delete it first or re-upload with replace enabled to re-index."
        )
    if duplicate and replace_existing:
        delete_document(db, user_id, duplicate.document_id)

    try:
        pages = extract_text_from_pdf(file_bytes)
    except PDFProcessingError as exc:
        raise RAGError(str(exc)) from exc

    chunks = chunk_pages(pages, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    if not chunks:
        raise RAGError("No usable text chunks could be produced from this document.")

    document_id = uuid.uuid4().hex[:16]
    chunk_ids = [f"chunk-{c.chunk_index}" for c in chunks]
    chunk_texts = [c.text for c in chunks]
    chunk_pages_list = [c.page for c in chunks]

    try:
        vectors = embed_texts(chunk_texts)
    except Exception as exc:  # noqa: BLE001
        raise RAGError(f"Failed to generate embeddings: {exc}") from exc

    try:
        pinecone_db.upsert_chunks(
            user_id=user_id,
            document_id=document_id,
            document_name=original_filename,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            chunk_vectors=vectors,
            chunk_pages=chunk_pages_list,
        )
    except Exception as exc:  # noqa: BLE001
        raise RAGError(f"Failed to store document in the vector database: {exc}") from exc

    row = Document(
        id=document_id,
        owner_user_id=user_id,
        document_name=original_filename,
        num_chunks=len(chunks),
        size_bytes=len(file_bytes),
        uploaded_at=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()

    logger.info(
        "Ingested document '%s' (%s) with %d chunks for user '%s'.",
        original_filename, document_id, len(chunks), user_id,
    )
    return _to_document_info(row)


def delete_document(db: Session, user_id: str, document_id: str) -> bool:
    """Delete a document, but only if it is owned by `user_id`."""
    row = get_owned_document(db, user_id, document_id)
    if row is None:
        return False

    db.delete(row)
    db.commit()

    try:
        pinecone_db.delete_document(user_id=user_id, document_id=document_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to delete vectors for document '%s': %s", document_id, exc)
        raise RAGError(f"Failed to remove document vectors: {exc}") from exc

    return True


# --------------------------------------------------------------------------
# Retrieval + answer generation
# --------------------------------------------------------------------------

def _history_to_dicts(history: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in history]


def retrieve_context(user_id: str, question: str, document_id: str | None = None) -> list[pinecone_db.RetrievedChunk]:
    settings = get_settings()
    query_vector = embed_query(question)
    matches = pinecone_db.query_similar_chunks(
        user_id=user_id,
        query_vector=query_vector,
        top_k=settings.top_k,
        document_id=document_id,
    )
    return [m for m in matches if m.score >= settings.similarity_threshold]


def _to_source_chunks(matches: list[pinecone_db.RetrievedChunk]) -> list[SourceChunk]:
    sources = []
    for match in matches:
        snippet = match.text[:280] + ("..." if len(match.text) > 280 else "")
        sources.append(
            SourceChunk(
                document_id=match.document_id,
                document_name=match.document_name,
                chunk_id=match.chunk_id,
                page=match.page,
                score=round(match.score, 4),
                snippet=snippet,
            )
        )
    return sources


def _to_web_sources(results: list[web_search.WebResult]) -> list[WebSource]:
    return [WebSource(title=r.title, url=r.url, snippet=r.snippet) for r in results]


def answer_question(
    db: Session,
    user_id: str,
    question: str,
    mode: ChatMode,
    history: list[ChatMessage],
    document_id: str | None = None,
) -> tuple[str, ChatMode, list[SourceChunk], list[WebSource], bool | None]:
    """
    Route the question to General AI, RAG, or Web Search mode and
    produce an answer. RAG/document lookups are always scoped to
    `user_id`'s own Pinecone namespace.

    Returns (answer, mode_actually_used, sources, web_sources, found_in_documents).
    """
    history_dicts = _history_to_dicts(history)

    if mode == ChatMode.GENERAL:
        answer = groq_client.generate_general_answer(question, history_dicts)
        return answer, ChatMode.GENERAL, [], [], None

    if mode == ChatMode.WEB:
        try:
            results = web_search.search_web(question, max_results=get_settings().web_search_max_results)
        except WebSearchError as exc:
            return str(exc), ChatMode.WEB, [], [], None

        if not results:
            answer = "I couldn't find any current web results for that question. Try rephrasing it."
            return answer, ChatMode.WEB, [], [], None

        context = "\n\n---\n\n".join(f"[{r.title}]({r.url})\n{r.snippet}" for r in results)
        answer = groq_client.generate_web_answer(question, context, history_dicts)
        return answer, ChatMode.WEB, [], _to_web_sources(results), None

    if mode == ChatMode.AUTO:
        has_documents = len(list_documents(db, user_id)) > 0
        wants_documents = has_documents and groq_client.classify_intent_needs_documents(question)
        if not wants_documents:
            answer = groq_client.generate_general_answer(question, history_dicts)
            return answer, ChatMode.GENERAL, [], [], None
        mode = ChatMode.RAG

    # RAG mode (explicit or resolved from AUTO)
    matches = retrieve_context(user_id, question, document_id=document_id)
    sources = _to_source_chunks(matches)

    if not matches:
        answer = (
            "I couldn't find relevant information about that in your uploaded "
            "documents. Please try rephrasing your question, upload a document "
            "that covers this topic, or switch to General AI mode."
        )
        return answer, ChatMode.RAG, [], [], False

    context = "\n\n---\n\n".join(
        f"[Source: {m.document_name}, page {m.page or 'n/a'}]\n{m.text}" for m in matches
    )
    answer = groq_client.generate_rag_answer(question, context, history_dicts)
    return answer, ChatMode.RAG, sources, [], True


# --------------------------------------------------------------------------
# Chat history (per-user chat sessions + messages)
# --------------------------------------------------------------------------

def get_owned_session(db: Session, user_id: str, session_id: str) -> ChatSession | None:
    """Fetch a chat session only if it belongs to `user_id` (see get_owned_document for why)."""
    return (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.owner_user_id == user_id)
        .first()
    )


def get_or_create_session(db: Session, user_id: str, session_id: str | None, first_message: str) -> ChatSession:
    if session_id:
        session = get_owned_session(db, user_id, session_id)
        if session is not None:
            return session
        # Unknown/foreign session id: silently start a fresh session for this
        # user rather than erroring, so a tampered id can't be used as an oracle.

    title = first_message.strip()[:60] or "New chat"
    session = ChatSession(owner_user_id=user_id, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def record_message(db: Session, session: ChatSession, role: str, content: str, mode_used: str | None = None) -> None:
    db.add(ChatMessageRow(session_id=session.id, role=role, content=content, mode_used=mode_used))
    session.updated_at = datetime.now(timezone.utc)
    db.add(session)
    db.commit()


def list_chat_sessions(db: Session, user_id: str) -> list[ChatSessionSummary]:
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.owner_user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    return [
        ChatSessionSummary(
            session_id=s.id,
            title=s.title,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
            message_count=len(s.messages),
        )
        for s in sessions
    ]


def get_chat_session_detail(db: Session, user_id: str, session_id: str) -> ChatSessionDetail | None:
    session = get_owned_session(db, user_id, session_id)
    if session is None:
        return None
    return ChatSessionDetail(
        session_id=session.id,
        title=session.title,
        messages=[
            ChatHistoryMessage(
                role=m.role,
                content=m.content,
                mode_used=m.mode_used,
                created_at=m.created_at.isoformat(),
            )
            for m in session.messages
        ],
    )


def delete_chat_session(db: Session, user_id: str, session_id: str) -> bool:
    session = get_owned_session(db, user_id, session_id)
    if session is None:
        return False
    db.delete(session)
    db.commit()
    return True

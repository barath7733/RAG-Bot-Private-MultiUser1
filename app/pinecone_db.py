"""
Pinecone vector database integration.

Responsible for:
- Ensuring the configured index exists with the correct dimension
  (derived at runtime from the embedding model — never hard-coded).
- Upserting document chunk embeddings with rich metadata.
- Running similarity search for a query embedding.
- Deleting all vectors belonging to a given document.
- Listing distinct documents currently indexed.

Internal Pinecone implementation details (namespaces, raw vector IDs,
index internals) are never leaked to the frontend — callers of this
module only ever see plain document/chunk metadata.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec

from app.config import get_settings
from app.embeddings import get_embedding_dimension

logger = logging.getLogger("rag_chatbot.pinecone_db")

_client_lock = threading.Lock()
_pc: Pinecone | None = None
_index = None
_ensured_dimension: int | None = None


@dataclass
class RetrievedChunk:
    document_id: str
    document_name: str
    chunk_id: str
    page: int | None
    score: float
    text: str


def _get_client() -> Pinecone:
    global _pc
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured.")
    with _client_lock:
        if _pc is None:
            _pc = Pinecone(api_key=settings.pinecone_api_key)
        return _pc


def get_index():
    """
    Return a ready-to-use Pinecone index handle, creating the index
    with the correct (model-derived) dimension if it does not exist
    yet.
    """
    global _index, _ensured_dimension
    settings = get_settings()
    pc = _get_client()
    dimension = get_embedding_dimension()

    with _client_lock:
        if _index is not None and _ensured_dimension == dimension:
            return _index

        existing = {idx["name"] for idx in pc.list_indexes()}
        if settings.pinecone_index_name not in existing:
            logger.info(
                "Creating Pinecone index '%s' with dimension %d ...",
                settings.pinecone_index_name,
                dimension,
            )
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        else:
            description = pc.describe_index(settings.pinecone_index_name)
            existing_dim = description.dimension
            if existing_dim != dimension:
                raise RuntimeError(
                    f"Pinecone index '{settings.pinecone_index_name}' has dimension "
                    f"{existing_dim}, but the configured embedding model "
                    f"'{settings.embedding_model}' produces {dimension}-dimensional "
                    "vectors. Use a different PINECONE_INDEX_NAME or recreate the index."
                )

        _index = pc.Index(settings.pinecone_index_name)
        _ensured_dimension = dimension
        return _index


def is_index_ready() -> bool:
    try:
        get_index()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pinecone index not ready: %s", exc)
        return False


def _user_namespace(user_id: str) -> str:
    """
    Map a user id to its Pinecone namespace. Namespaces give hard,
    engine-level isolation between users: a query scoped to one
    namespace can never return vectors stored under another, even if a
    metadata filter were accidentally omitted somewhere. Every call
    into this module below requires a user_id for exactly this reason.
    """
    return f"user_{user_id}"


def upsert_chunks(
    user_id: str,
    document_id: str,
    document_name: str,
    chunk_ids: list[str],
    chunk_texts: list[str],
    chunk_vectors: list[list[float]],
    chunk_pages: list[int | None],
) -> int:
    """Upsert a batch of chunk embeddings, scoped to the owning user's namespace."""
    if not (len(chunk_ids) == len(chunk_texts) == len(chunk_vectors) == len(chunk_pages)):
        raise ValueError("Mismatched chunk arrays passed to upsert_chunks.")

    index = get_index()
    namespace = _user_namespace(user_id)
    vectors = []
    for chunk_id, text, vector, page in zip(chunk_ids, chunk_texts, chunk_vectors, chunk_pages):
        vectors.append(
            {
                "id": f"{document_id}::{chunk_id}",
                "values": vector,
                "metadata": {
                    "document_id": document_id,
                    "document_name": document_name,
                    "owner_user_id": user_id,
                    "chunk_id": chunk_id,
                    "page": page if page is not None else -1,
                    "text": text,
                },
            }
        )

    batch_size = 100
    total = 0
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start:start + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        total += len(batch)

    logger.info("Upserted %d chunk(s) for document '%s' (namespace '%s').", total, document_id, namespace)
    return total


def query_similar_chunks(
    user_id: str,
    query_vector: list[float],
    top_k: int = 5,
    document_id: str | None = None,
) -> list[RetrievedChunk]:
    """
    Run a similarity search restricted to the requesting user's
    namespace only — it is structurally impossible for this call to
    return another user's chunks, regardless of `document_id`.
    """
    index = get_index()
    namespace = _user_namespace(user_id)
    filter_dict = {"document_id": {"$eq": document_id}} if document_id else None

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict,
        namespace=namespace,
    )

    matches: list[RetrievedChunk] = []
    for match in result.get("matches", []) if isinstance(result, dict) else result.matches:
        metadata = match["metadata"] if isinstance(match, dict) else match.metadata
        score = match["score"] if isinstance(match, dict) else match.score
        page_value = metadata.get("page", -1)
        matches.append(
            RetrievedChunk(
                document_id=metadata.get("document_id", ""),
                document_name=metadata.get("document_name", "unknown"),
                chunk_id=metadata.get("chunk_id", ""),
                page=None if page_value == -1 else int(page_value),
                score=float(score),
                text=metadata.get("text", ""),
            )
        )
    return matches


def delete_document(user_id: str, document_id: str) -> None:
    """Delete all vectors belonging to a document, scoped to the owner's namespace only."""
    index = get_index()
    namespace = _user_namespace(user_id)
    index.delete(filter={"document_id": {"$eq": document_id}}, namespace=namespace)
    logger.info("Deleted vectors for document '%s' (namespace '%s').", document_id, namespace)

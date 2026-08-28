"""
Lightweight Gemini API based embeddings.

Uses Gemini Embedding API instead of local sentence-transformers,
so the application does not load PyTorch/transformer models into
Render's limited RAM environment.
"""

from __future__ import annotations

import logging
import os

from google import genai
from google.genai import types

logger = logging.getLogger("rag_chatbot.embeddings")

EMBEDDING_MODEL = "gemini-embedding-001"

# Keep 384 dimensions so the existing Pinecone index can remain compatible
# with the previous all-MiniLM-L6-v2 embedding dimension.
EMBEDDING_DIMENSION = 384

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client

    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not configured."
            )

        _client = genai.Client(api_key=api_key)

    return _client


def get_embedding_dimension() -> int:
    """Return the embedding vector dimension."""
    return EMBEDDING_DIMENSION


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple document chunks."""

    if not texts:
        return []

    client = _get_client()

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION,
            task_type="RETRIEVAL_DOCUMENT",
        ),
    )

    if not result.embeddings:
        raise RuntimeError("Gemini returned no embeddings.")

    return [embedding.values for embedding in result.embeddings]


def embed_query(text: str) -> list[float]:
    """Generate an embedding for a user query."""

    if not text.strip():
        return []

    client = _get_client()

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION,
            task_type="RETRIEVAL_QUERY",
        ),
    )

    if not result.embeddings:
        raise RuntimeError("Gemini returned no embedding.")

    return result.embeddings[0].values
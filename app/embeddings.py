"""
Lightweight Gemini API based embeddings.

Uses Gemini Embedding API instead of local sentence-transformers,
so the application does not load PyTorch/transformer models.
"""

from __future__ import annotations

import logging
import os
import time

from google import genai
from google.genai import types

logger = logging.getLogger("rag_chatbot.embeddings")

EMBEDDING_MODEL = "gemini-embedding-001"

# Keep 384 dimensions so the existing Pinecone index remains compatible.
EMBEDDING_DIMENSION = 384

# Gemini allows at most 100 requests in one batch.
# Use 80 to stay safely below the limit.
EMBEDDING_BATCH_SIZE = 80

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


def _embed_batch(
    client: genai.Client,
    texts: list[str],
) -> list[list[float]]:
    """Generate embeddings for one safe-sized batch."""

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


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple document chunks.

    Large lists are automatically split into smaller batches so that
    Gemini never receives more than EMBEDDING_BATCH_SIZE requests
    in a single API call.
    """

    if not texts:
        return []

    client = _get_client()

    all_embeddings: list[list[float]] = []

    total = len(texts)

    logger.info(
        "Generating embeddings for %d chunks using batch size %d.",
        total,
        EMBEDDING_BATCH_SIZE,
    )

    for start in range(0, total, EMBEDDING_BATCH_SIZE):
        end = min(start + EMBEDDING_BATCH_SIZE, total)

        batch = texts[start:end]

        logger.info(
            "Embedding batch: chunks %d-%d of %d",
            start + 1,
            end,
            total,
        )

        embeddings = _embed_batch(client, batch)

        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Gemini returned {len(embeddings)} embeddings "
                f"for {len(batch)} texts."
            )

        all_embeddings.extend(embeddings)

        # Small pause between batches to reduce the chance of rate limits.
        if end < total:
            time.sleep(0.2)

    logger.info(
        "Successfully generated %d embeddings.",
        len(all_embeddings),
    )

    return all_embeddings


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
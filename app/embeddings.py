"""
Local sentence-transformers based embeddings.

Uses all-MiniLM-L6-v2 locally instead of the Gemini Embedding API.

Embedding dimension:
    384

This is compatible with a Pinecone index configured with dimension=384.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag_chatbot.embeddings")


# ---------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 produces 384-dimensional vectors.
EMBEDDING_DIMENSION = 384


# ---------------------------------------------------------
# Load model only once
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Load the sentence-transformers model once and reuse it.

    The model is downloaded automatically the first time the
    application starts if it is not already available.
    """

    logger.info(
        "Loading embedding model: %s",
        EMBEDDING_MODEL,
    )

    model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info(
        "Embedding model loaded successfully: %s",
        EMBEDDING_MODEL,
    )

    return model


# ---------------------------------------------------------
# Dimension
# ---------------------------------------------------------

def get_embedding_dimension() -> int:
    """
    Return the embedding vector dimension.
    """

    return EMBEDDING_DIMENSION


# ---------------------------------------------------------
# Document embeddings
# ---------------------------------------------------------

def embed_texts(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embeddings for multiple document chunks.

    Uses local sentence-transformers instead of an external
    Gemini embedding API.

    Parameters
    ----------
    texts:
        List of document chunks.

    Returns
    -------
    list[list[float]]
        One 384-dimensional embedding vector per text.
    """

    if not texts:
        return []

    model = _get_model()

    logger.info(
        "Generating embeddings for %d document chunks using %s",
        len(texts),
        EMBEDDING_MODEL,
    )

    # Normalize embeddings so cosine similarity works consistently.
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    result = embeddings.tolist()

    # Safety check
    if len(result) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: "
            f"generated {len(result)} embeddings "
            f"for {len(texts)} texts."
        )

    for index, embedding in enumerate(result):
        if len(embedding) != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Invalid embedding dimension at index {index}: "
                f"expected {EMBEDDING_DIMENSION}, "
                f"got {len(embedding)}."
            )

    logger.info(
        "Successfully generated %d embeddings.",
        len(result),
    )

    return result


# ---------------------------------------------------------
# Query embedding
# ---------------------------------------------------------

def embed_query(
    text: str,
) -> list[float]:
    """
    Generate an embedding for a user search query.

    Uses the same local model as document embeddings so that
    document vectors and query vectors are compatible.
    """

    if not text or not text.strip():
        return []

    model = _get_model()

    logger.info(
        "Generating query embedding using %s",
        EMBEDDING_MODEL,
    )

    embedding = model.encode(
        text.strip(),
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    result = embedding.tolist()

    if len(result) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Invalid query embedding dimension: "
            f"expected {EMBEDDING_DIMENSION}, "
            f"got {len(result)}."
        )

    return result
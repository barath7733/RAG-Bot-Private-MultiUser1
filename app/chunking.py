"""
Text cleaning and chunking.

Splits extracted page text into overlapping, meaning-preserving
chunks suitable for embedding and vector search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.pdf_processor import ExtractedPage


@dataclass
class TextChunk:
    chunk_index: int
    text: str
    page: int | None


def clean_text(text: str) -> str:
    """Normalize whitespace and strip stray control characters."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (no heavy NLP dependency required)."""
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_pages(
    pages: list[ExtractedPage],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[TextChunk]:
    """
    Chunk cleaned page text into overlapping windows of approximately
    `chunk_size` characters, respecting sentence boundaries where
    possible so chunks remain semantically coherent.

    Each chunk retains a reference to the source page number (the
    page where the chunk *starts*) for source attribution.
    """
    chunks: list[TextChunk] = []
    chunk_index = 0

    for page in pages:
        cleaned = clean_text(page.text)
        sentences = _split_into_sentences(cleaned)
        if not sentences:
            continue

        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence

            if len(candidate) <= chunk_size:
                current = candidate
                continue

            # Flush the current chunk, then start a new one with overlap.
            if current:
                chunks.append(TextChunk(chunk_index=chunk_index, text=current, page=page.page_number))
                chunk_index += 1
                overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
                current = f"{overlap_text} {sentence}".strip()
            else:
                # A single sentence longer than chunk_size: hard-split it.
                for start in range(0, len(sentence), chunk_size - chunk_overlap):
                    piece = sentence[start:start + chunk_size]
                    if piece.strip():
                        chunks.append(TextChunk(chunk_index=chunk_index, text=piece.strip(), page=page.page_number))
                        chunk_index += 1
                current = ""

        if current:
            chunks.append(TextChunk(chunk_index=chunk_index, text=current, page=page.page_number))
            chunk_index += 1

    return chunks

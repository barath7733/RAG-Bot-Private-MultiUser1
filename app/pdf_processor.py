"""
PDF text extraction.

Uses PyPDF to safely extract text page-by-page from an uploaded PDF,
handling corrupted, encrypted, empty, or scanned (image-only) PDFs
without raising unhandled exceptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger("rag_chatbot.pdf_processor")


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be safely read or contains no usable text."""


@dataclass
class ExtractedPage:
    page_number: int  # 1-indexed, matches how a human would reference the page
    text: str


def extract_text_from_pdf(file_bytes: bytes) -> list[ExtractedPage]:
    """
    Extract text from a PDF's bytes, page by page.

    Returns a list of ExtractedPage (only pages with non-empty text).
    Raises PDFProcessingError for corrupted, encrypted (without a
    crackable empty password), or fully empty/image-only documents.
    """
    if not file_bytes:
        raise PDFProcessingError("The uploaded file is empty.")

    try:
        reader = PdfReader(BytesIOCompat(file_bytes))
    except PdfReadError as exc:
        raise PDFProcessingError(f"The PDF appears to be corrupted or unreadable: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - we want to convert any parser failure
        raise PDFProcessingError(f"Could not open the PDF file: {exc}") from exc

    if reader.is_encrypted:
        try:
            # Try an empty password first — common for "protected but not really" PDFs.
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise PDFProcessingError(
                "This PDF is password-protected and cannot be processed."
            ) from exc

    pages: list[ExtractedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - a single bad page shouldn't kill the whole doc
            logger.warning("Failed to extract text from page %s: %s", index, exc)
            raw_text = ""

        cleaned = raw_text.strip()
        if cleaned:
            pages.append(ExtractedPage(page_number=index, text=cleaned))

    if not pages:
        raise PDFProcessingError(
            "No extractable text was found in this PDF. It may be a scanned "
            "or image-only document that requires OCR, which is not supported."
        )

    logger.info("Extracted text from %d/%d page(s).", len(pages), len(reader.pages))
    return pages


def BytesIOCompat(file_bytes: bytes):
    """Small helper kept separate so extraction failures are easy to unit test."""
    import io

    return io.BytesIO(file_bytes)

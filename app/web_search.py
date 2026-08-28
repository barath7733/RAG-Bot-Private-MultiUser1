"""
Web search integration (Tavily API).

Used for the "Web Search" chat mode, so the assistant can answer
questions about current events, prices, or anything past the LLM's
training cutoff — grounded in freshly retrieved web results, with
source URLs returned to the frontend, the same way RAG mode grounds
answers in uploaded documents.

Requires TAVILY_API_KEY (free tier available at https://tavily.com).
If it is not configured, callers get a clear RuntimeError instead of
a silent failure.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from app.config import get_settings
load_dotenv()

logger = logging.getLogger("rag_chatbot.web_search")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class WebSearchError(Exception):
    """Raised for any user-facing web search failure."""


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str


def is_web_search_configured() -> bool:
    return bool(os.getenv("TAVILY_API_KEY"))


def search_web(query: str, max_results: int = 5) -> list[WebResult]:
    """Run a web search and return the top results (title, url, snippet)."""
    settings = get_settings()
    if not settings.tavily_api_key:
        raise WebSearchError("TAVILY_API_KEY is not configured — Web Search mode is unavailable.")

    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
    }

    try:
        response = httpx.post(TAVILY_SEARCH_URL, json=payload, timeout=20.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("Tavily search failed with status %s: %s", exc.response.status_code, exc)
        raise WebSearchError(f"Web search request failed ({exc.response.status_code}).") from exc
    except httpx.HTTPError as exc:
        logger.error("Tavily search request error: %s", exc)
        raise WebSearchError(f"Web search request failed: {exc}") from exc

    data = response.json()
    results: list[WebResult] = []
    for item in data.get("results", []):
        results.append(
            WebResult(
                title=item.get("title", "Untitled"),
                url=item.get("url", ""),
                snippet=(item.get("content", "") or "")[:500],
            )
        )

    if not results:
        logger.info("Web search for '%s' returned no results.", query)

    return results

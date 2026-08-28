"""
AI image generation.

Uses the free Pollinations.ai image API (https://pollinations.ai),
which requires no API key — a plain HTTPS request with a URL-encoded
prompt returns a generated image. This keeps the "get it working
today" bar low; if you later want higher-quality or commercially
licensed output, swap this module for a provider like Stability AI
or Together AI (both take an API key the same way Groq/Pinecone do
here — see the README for notes).

ACCURACY: raw user prompts are frequently short/vague ("a cat on a
skateboard"), and diffusion models are very sensitive to how specific
a prompt is. Before generating, the raw prompt is expanded into a
detailed, unambiguous prompt via Groq (see
app.groq_client.enhance_image_prompt) — preserving the user's exact
subject/intent while adding the visual specificity (composition,
lighting, style) needed to get an accurate result. This step degrades
gracefully to the raw prompt if the LLM call fails.
"""

from __future__ import annotations

import logging
import urllib.parse
import uuid

import httpx

from app.config import get_settings
from app.groq_client import enhance_image_prompt

logger = logging.getLogger("rag_chatbot.image_gen")

POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt"


class ImageGenerationError(Exception):
    """Raised for any user-facing image generation failure."""


def build_image_url(prompt: str, width: int = 1024, height: int = 1024) -> str:
    """
    Build a direct, shareable image URL for the given (already final)
    prompt. The frontend can render this URL straight into an <img>
    tag — the image is generated on Pollinations' side on first fetch.
    """
    if not prompt or not prompt.strip():
        raise ImageGenerationError("Please provide a description of the image you want to generate.")

    settings = get_settings()
    encoded_prompt = urllib.parse.quote(prompt.strip())
    # A random seed keeps repeated identical prompts from being cached to the same image.
    seed = uuid.uuid4().int % 1_000_000
    return (
        f"{POLLINATIONS_BASE_URL}/{encoded_prompt}"
        f"?width={width}&height={height}&seed={seed}&nologo=true&model={settings.image_gen_model}"
    )


def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    enhance: bool = True,
) -> tuple[str, str]:
    """
    Generate an image for the prompt and return (image_url, final_prompt),
    after verifying the generation actually succeeds (Pollinations
    returns a real error status for disallowed/invalid prompts rather
    than a valid image).

    `final_prompt` is what was actually sent to the image model — when
    `enhance=True` (the default) this is the Groq-expanded version of
    the user's prompt, returned to the caller so the frontend can show
    the user exactly what was generated from, for transparency.
    """
    if not prompt or not prompt.strip():
        raise ImageGenerationError("Please provide a description of the image you want to generate.")

    final_prompt = enhance_image_prompt(prompt) if enhance else prompt.strip()

    url = build_image_url(final_prompt, width=width, height=height)

    try:
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Image generation failed with status %s (prompt: %r)",
            exc.response.status_code, final_prompt,
        )
        raise ImageGenerationError(
            "Image generation failed. Try rephrasing your prompt."
        ) from exc
    except httpx.HTTPError as exc:
        logger.error("Image generation request error: %s", exc)
        raise ImageGenerationError(f"Image generation request failed: {exc}") from exc

    return url, final_prompt
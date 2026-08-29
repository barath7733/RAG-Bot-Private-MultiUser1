"""
AI image generation using the current Pollinations API.
"""

from __future__ import annotations

import logging
import urllib.parse
import uuid

import httpx

from app.config import get_settings
from app.groq_client import enhance_image_prompt

logger = logging.getLogger("rag_chatbot.image_gen")

POLLINATIONS_BASE_URL = "https://gen.pollinations.ai/image"


class ImageGenerationError(Exception):
    """Raised for any user-facing image generation failure."""


def build_image_url(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
) -> str:
    """
    Build a Pollinations image URL using the current API.
    """

    if not prompt or not prompt.strip():
        raise ImageGenerationError(
            "Please provide a description of the image you want to generate."
        )

    settings = get_settings()

    encoded_prompt = urllib.parse.quote(prompt.strip())
    seed = uuid.uuid4().int % 1_000_000

    url = (
        f"{POLLINATIONS_BASE_URL}/{encoded_prompt}"
        f"?model={settings.image_gen_model}"
        f"&width={width}"
        f"&height={height}"
        f"&seed={seed}"
        f"&nologo=true"
    )

    return url


def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    enhance: bool = True,
) -> tuple[str, str]:
    """
    Generate an image using Pollinations and return:

        (image_url, final_prompt)
    """

    if not prompt or not prompt.strip():
        raise ImageGenerationError(
            "Please provide a description of the image you want to generate."
        )

    settings = get_settings()

    # Enhance the user's prompt using Groq.
    if enhance:
        try:
            final_prompt = enhance_image_prompt(prompt)
        except Exception as exc:
            logger.warning(
                "Image prompt enhancement failed, using original prompt: %s",
                exc,
            )
            final_prompt = prompt.strip()
    else:
        final_prompt = prompt.strip()

    url = build_image_url(
        final_prompt,
        width=width,
        height=height,
    )

    if not settings.pollinations_api_key:
        logger.error("POLLINATIONS_API_KEY is not configured.")

        raise ImageGenerationError(
            "Image generation is not configured. "
            "Please add POLLINATIONS_API_KEY in the server environment."
        )

    headers = {
        "Authorization": f"Bearer {settings.pollinations_api_key}",
    }

    try:
        response = httpx.get(
            url,
            headers=headers,
            timeout=120.0,
            follow_redirects=True,
        )

        response.raise_for_status()

    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code

        logger.error(
            "Pollinations image generation failed. "
            "Status=%s Prompt=%r",
            status_code,
            final_prompt,
        )

        if status_code == 401:
            raise ImageGenerationError(
                "Image generation authentication failed. "
                "Please check the Pollinations API key."
            ) from exc

        if status_code == 402:
            raise ImageGenerationError(
                "Image generation requires available Pollen/credits."
            ) from exc

        if status_code == 429:
            raise ImageGenerationError(
                "Image generation rate limit reached. Please try again later."
            ) from exc

        raise ImageGenerationError(
            "Image generation failed. Please try again."
        ) from exc

    except httpx.HTTPError as exc:
        logger.error(
            "Pollinations image generation request error: %s",
            exc,
        )

        raise ImageGenerationError(
            "Unable to connect to the image generation service."
        ) from exc

    return url, final_prompt
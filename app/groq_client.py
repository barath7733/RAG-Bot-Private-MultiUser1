"""
Groq LLM client.

Groq is the ONLY LLM provider used in this project. No OpenAI SDK,
models, or API keys are used anywhere.
"""

from __future__ import annotations

import logging

from groq import Groq

from app.config import get_settings

logger = logging.getLogger("rag_chatbot.groq_client")

_client: Groq | None = None


# --------------------------------------------------------------------------
# General AI
# --------------------------------------------------------------------------

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable general-purpose AI assistant. "
    "Answer the user's question directly and accurately using your own "
    "knowledge. Be clear, well-organized, and easy to understand.\n\n"

    "ANSWER FORMATTING RULES:\n"
    "1. Choose the clearest format based on the question.\n"
    "2. Use clear Markdown headings (## and ###) when the answer has "
    "multiple sections.\n"
    "3. Use bullet points for lists.\n"
    "4. Use numbered lists for steps, procedures, or ordered information.\n"
    "5. Use short paragraphs for explanations.\n"
    "6. Use **bold text** for important terms when useful.\n"
    "7. Use a Markdown table only when it genuinely improves clarity, "
    "especially for comparisons, differences, classifications, or "
    "structured data.\n"
    "8. Do not force information into a table when normal paragraphs or "
    "bullet points are clearer.\n"
    "9. Keep the answer logically ordered and easy to read.\n"
    "10. Match the amount of detail to the question.\n"
    "11. If you are not certain about something, say so honestly instead "
    "of guessing."
)


# --------------------------------------------------------------------------
# Web Search
# --------------------------------------------------------------------------

WEB_SYSTEM_PROMPT = (
    "You are a helpful assistant answering using freshly retrieved web "
    "search results provided below.\n\n"

    "ANSWER FORMATTING RULES:\n"
    "1. Choose the clearest format based on the question.\n"
    "2. Use clear Markdown headings (## and ###) when useful.\n"
    "3. Use bullet points for lists.\n"
    "4. Use numbered lists for steps or ordered information.\n"
    "5. Use short paragraphs for explanations.\n"
    "6. Use **bold text** for important terms when useful.\n"
    "7. Use a Markdown table only when it genuinely improves clarity, "
    "especially for comparisons or structured data.\n"
    "8. Do not force normal explanations into a table.\n"
    "9. Keep the answer logically ordered and easy to read.\n\n"

    "WEB SEARCH RULES:\n"
    "1. Base your answer on the retrieved web results — they contain "
    "more current information than your own training data.\n"
    "2. Do not invent facts, numbers, or dates not present in the results.\n"
    "3. If the results don't answer the question, say so honestly.\n"
    "4. Be concise and cite which source(s) support key claims by name "
    "when it's natural to do so.\n"
    "5. Note that results may be time-sensitive; do not overstate certainty."
)


# --------------------------------------------------------------------------
# RAG / Document
# --------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = (
    "You are a document-grounded assistant. Answer the user's question "
    "PRIMARILY using the retrieved context provided below, which comes "
    "from documents the user has uploaded.\n\n"

    "ANSWER FORMATTING RULES:\n"
    "1. Choose the clearest format based on the user's question.\n"
    "2. Use a Markdown table ONLY when it genuinely improves clarity, "
    "such as for comparison, difference, classification, specifications, "
    "or structured data.\n"
    "3. For explanations, concepts, descriptions, definitions, and general "
    "questions, prefer headings, subheadings, short paragraphs, bullet "
    "points, and numbered lists instead of a table.\n"
    "4. For procedures, processes, methods, or step-by-step questions, "
    "use numbered lists in the correct order.\n"
    "5. For multiple related points, use bullet points.\n"
    "6. Use clear Markdown headings (##) and subheadings (###) when the "
    "answer contains multiple sections.\n"
    "7. Use short and readable paragraphs for explanations.\n"
    "8. Use **bold text** to highlight important terms when useful.\n"
    "9. Keep the answer logically ordered from introduction to explanation "
    "and then key points or conclusion when appropriate.\n"
    "10. Do not unnecessarily repeat the question.\n"
    "11. Do not add unnecessary introductions or filler.\n"
    "12. Match the level of detail to the question.\n"
    "13. Make the response look clean and readable, similar to a modern "
    "AI assistant response.\n\n"

    "DOCUMENT GROUNDING RULES:\n"
    "1. Base your answer on the retrieved context whenever it is relevant.\n"
    "2. Do not invent, assume, or fabricate any facts, numbers, or details "
    "that are not present in the retrieved context.\n"
    "3. If the retrieved context does not contain enough information to "
    "answer the question, clearly state that the information could not "
    "be found in the uploaded documents rather than guessing.\n"
    "4. You may use general knowledge only to explain or clarify terms "
    "found in the context — never to supply document-specific facts "
    "that are not in the context."
)


# --------------------------------------------------------------------------
# Groq Client
# --------------------------------------------------------------------------

def _get_client() -> Groq:
    global _client

    settings = get_settings()

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)

    return _client


def _run_completion(messages: list[dict[str, str]]) -> str:
    settings = get_settings()
    client = _get_client()

    try:
        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("Groq API call failed: %s", exc)
        raise RuntimeError(
            f"The AI provider request failed: {exc}"
        ) from exc

    choice = completion.choices[0] if completion.choices else None

    if (
        not choice
        or not choice.message
        or not choice.message.content
    ):
        raise RuntimeError(
            "The AI provider returned an empty response."
        )

    return choice.message.content.strip()


# --------------------------------------------------------------------------
# General AI Answer
# --------------------------------------------------------------------------

def generate_general_answer(
    question: str,
    history: list[dict[str, str]],
) -> str:
    """General AI mode: answer directly with no document context."""

    messages = [
        {
            "role": "system",
            "content": GENERAL_SYSTEM_PROMPT,
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    return _run_completion(messages)


# --------------------------------------------------------------------------
# RAG Answer
# --------------------------------------------------------------------------

def generate_rag_answer(
    question: str,
    context: str,
    history: list[dict[str, str]],
) -> str:
    """RAG mode: answer grounded in retrieved document context."""

    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT,
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": (
                "Retrieved context from uploaded documents:\n"
                "---\n"
                f"{context}\n"
                "---\n\n"
                f"Question: {question}"
            ),
        }
    )

    return _run_completion(messages)


# --------------------------------------------------------------------------
# Web Search Answer
# --------------------------------------------------------------------------

def generate_web_answer(
    question: str,
    context: str,
    history: list[dict[str, str]],
) -> str:
    """Web Search mode: answer grounded in freshly retrieved web results."""

    messages = [
        {
            "role": "system",
            "content": WEB_SYSTEM_PROMPT,
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": (
                "Web search results:\n"
                "---\n"
                f"{context}\n"
                "---\n\n"
                f"Question: {question}"
            ),
        }
    )

    return _run_completion(messages)


# --------------------------------------------------------------------------
# Intent Classification
# --------------------------------------------------------------------------

def classify_intent_needs_documents(
    question: str,
) -> bool:
    """
    Lightweight LLM-based classification used only in AUTO mode:
    does this question likely require looking at uploaded documents?

    Falls back to a conservative default (True — try RAG first) if the
    classification call itself fails, since RAG mode gracefully reports
    "not found in documents" when nothing relevant is retrieved.
    """

    settings = get_settings()
    client = _get_client()

    try:
        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify whether answering the user's question "
                        "would require looking up information in uploaded "
                        "documents (as opposed to being answerable from "
                        "general knowledge or being small talk). Reply with "
                        "exactly one word: 'DOCUMENT' or 'GENERAL'."
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            temperature=0,
            max_tokens=5,
        )

        label = (
            completion.choices[0].message.content or ""
        ).strip().upper()

        return "DOCUMENT" in label

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Intent classification failed, defaulting to document search: %s",
            exc,
        )

        return True


# --------------------------------------------------------------------------
# Image Prompt Enhancement
# --------------------------------------------------------------------------

IMAGE_PROMPT_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for text-to-image diffusion models. "
    "Rewrite the user's image request into a single, highly detailed, "
    "unambiguous image-generation prompt that will make the model produce "
    "EXACTLY what the user asked for.\n\n"

    "STRICT RULES:\n"
    "1. Preserve every explicit detail the user gave — subject(s), exact "
    "count of objects/people, colors, actions, setting, text, style, mood — "
    "word for word in meaning. Never drop, change, or contradict anything "
    "the user specified.\n"
    "2. Never invent NEW subjects, objects, people, or text that the user "
    "did not ask for. You may only ADD descriptive detail (lighting, "
    "camera angle, composition, texture, atmosphere, art style) that makes "
    "the described scene clearer and more visually specific — never detail "
    "that changes what is depicted.\n"
    "3. If the user's request is already very short (e.g. 'a red apple'), "
    "expand it with concrete visual specifics (framing, lighting, "
    "background, level of detail) while keeping the exact same subject.\n"
    "4. If the user specified an art style, medium, or aspect (e.g. "
    "'photo', 'cartoon', 'oil painting', 'logo', 'anime'), keep that style "
    "as the dominant instruction and reinforce it.\n"
    "5. Do not add camera/photo-realism language if the user asked for a "
    "clearly non-photographic style (e.g. cartoon, sketch, icon).\n"
    "6. Output ONLY the final rewritten prompt as plain text — no quotes, "
    "no labels, no explanations, no markdown."
)


def enhance_image_prompt(raw_prompt: str) -> str:
    """
    Expand a user's (often short/vague) image request into a detailed,
    unambiguous prompt for the image model, using Groq.

    Falls back to the raw prompt unchanged if the LLM call fails,
    so a Groq outage never blocks image generation entirely.
    """

    raw_prompt = raw_prompt.strip()

    try:
        completion = _get_client().chat.completions.create(
            model=get_settings().groq_model,
            messages=[
                {
                    "role": "system",
                    "content": IMAGE_PROMPT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": raw_prompt,
                },
            ],
            temperature=0.4,
            max_tokens=300,
        )

        enhanced = (
            completion.choices[0].message.content or ""
        ).strip().strip('"')

        return enhanced if enhanced else raw_prompt

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Image prompt enhancement failed, using raw prompt: %s",
            exc,
        )

        return raw_prompt


# --------------------------------------------------------------------------
# Vision / Image Analysis
# --------------------------------------------------------------------------

VISION_SYSTEM_PROMPT = (
    "You are a precise visual assistant. Look carefully at the provided "
    "image and answer the user's question about it accurately. If the "
    "user gave no specific question, provide a clear, well-organized "
    "description covering: the main subject(s), notable objects, setting, "
    "colors, any visible text, and anything else a person would find "
    "useful to know. Only describe what is actually visible — never guess "
    "or invent details you cannot see. If the image is unclear or you "
    "cannot confidently identify something, say so honestly."
)


def analyze_image(
    image_data_url: str,
    question: str | None,
) -> str:
    """
    Send an uploaded image to a vision-capable Groq model and return
    its answer/description.
    """

    settings = get_settings()

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    client = _get_client()

    user_text = (
        question.strip()
        if question and question.strip()
        else "Describe this image in detail."
    )

    try:
        completion = client.chat.completions.create(
            model=settings.vision_model,
            messages=[
                {
                    "role": "system",
                    "content": VISION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                            },
                        },
                    ],
                },
            ],
            temperature=0.3,
            max_tokens=1024,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Groq vision API call failed: %s",
            exc,
        )

        raise RuntimeError(
            f"Image analysis failed: {exc}. The configured vision model "
            f"({settings.vision_model}) may have changed — check "
            "console.groq.com/docs/vision for the current model name."
        ) from exc

    choice = (
        completion.choices[0]
        if completion.choices
        else None
    )

    if (
        not choice
        or not choice.message
        or not choice.message.content
    ):
        raise RuntimeError(
            "The vision model returned an empty response."
        )

    return choice.message.content.strip()
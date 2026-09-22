"""Groq-backed suggestion assistant.

The assistant is grounded: every request carries the poster image *and* the
numbers this project computed (detections, attention percentages, PES
components). It is told to treat those numbers as authoritative rather than
re-estimating them by eye, so the chat cannot contradict the score on screen.
"""

from __future__ import annotations

import base64
from typing import Any, Iterable

import cv2
import httpx
import numpy as np

from config.settings import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_IMAGE_MAX_EDGE,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
)


class ChatConfigError(RuntimeError):
    """Raised when the assistant is not configured (missing key)."""


class ChatUpstreamError(RuntimeError):
    """Raised when Groq rejects or fails the request."""


SYSTEM_PROMPT = """You are a poster design consultant for food and beverage advertising.

You are given a poster image plus a Poster Effectiveness Score (PES) analysis \
computed by a YOLOv8 detector and a saliency-based attention model.

Ground rules:
- The supplied numbers are authoritative. Never recompute, re-estimate, or \
contradict the detections, attention percentages, or component scores.
- You may comment on what you see in the image (colour, contrast, typography, \
composition, food styling) to *explain* those numbers.
- Be specific and actionable. "Move the CTA into the lower-right third and \
raise its contrast against the background" beats "improve the CTA".
- Prioritise the weakest component score unless the user asks about something else.
- If the analysis found no detections, say the detector found nothing and that \
advice is therefore based on the image alone.
- Keep replies under 200 words unless asked for more."""


def is_configured() -> bool:
    return bool(GROQ_API_KEY)


def encode_image_for_chat(image: np.ndarray) -> str:
    """Downscale and JPEG-encode a poster as a data URI within Groq's size limit."""
    height, width = image.shape[:2]
    longest_edge = max(height, width)

    if longest_edge > GROQ_IMAGE_MAX_EDGE:
        scale = GROQ_IMAGE_MAX_EDGE / longest_edge
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ChatUpstreamError("Could not encode the poster for the assistant.")

    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_analysis_digest(analysis: dict[str, Any]) -> str:
    """Flatten the analysis payload into compact text for the model."""
    pes = analysis.get("pes") or {}
    components = pes.get("components") or {}
    percentages = pes.get("attention_percentages") or {}
    detections = analysis.get("detections") or []

    lines: list[str] = []
    lines.append(f"PES: {pes.get('score', 0):.1f}/100 ({pes.get('category', 'n/a')})")

    if components:
        lines.append("Component scores (0-100, weighted):")
        weights = pes.get("weights") or {}
        for name, value in components.items():
            weight = weights.get(name, 0)
            lines.append(f"  - {name}: {value:.1f} (weight {weight:g}%)")

    if percentages:
        lines.append("Estimated attention share:")
        for label, pct in sorted(percentages.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  - {label}: {pct:.1f}%")

    if detections:
        lines.append(f"Detected elements ({len(detections)}):")
        for det in detections:
            lines.append(
                f"  - {det.get('class_name')} at {det.get('relative_position')}, "
                f"confidence {float(det.get('confidence', 0)) * 100:.0f}%, "
                f"covers {float(det.get('normalized_area', 0)) * 100:.1f}% of the poster"
            )
    else:
        lines.append("Detected elements: none.")

    insights = pes.get("insights") or []
    if insights:
        lines.append("Rule-based findings:")
        lines.extend(f"  - {item}" for item in insights)

    return "\n".join(lines)


def build_messages(
    analysis: dict[str, Any],
    history: Iterable[dict[str, str]],
    user_message: str,
    image_data_uri: str | None,
) -> list[dict[str, Any]]:
    """Assemble the Groq chat payload.

    The poster image rides on the first user turn only. Groq caps a request at
    three images and charges 2048 tokens each, so resending it every turn would
    burn the context window for no benefit — the model keeps it in history.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    digest = build_analysis_digest(analysis)
    opening: list[dict[str, Any]] = [
        {"type": "text", "text": f"Here is the analysis of my poster:\n\n{digest}"},
    ]
    if image_data_uri:
        opening.append({"type": "image_url", "image_url": {"url": image_data_uri}})

    messages.append({"role": "user", "content": opening})
    messages.append(
        {
            "role": "assistant",
            "content": "I have the poster and its analysis. What would you like to improve?",
        }
    )

    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages


def list_models() -> list[str]:
    """Return model ids the configured key can reach, for troubleshooting."""
    if not is_configured():
        raise ChatConfigError("GROQ_API_KEY is not set.")

    try:
        response = httpx.get(
            f"{GROQ_BASE_URL}/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=GROQ_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ChatUpstreamError(f"Could not reach Groq: {exc}") from exc

    payload = response.json()
    return sorted(item["id"] for item in payload.get("data", []) if "id" in item)


def ask(
    analysis: dict[str, Any],
    history: Iterable[dict[str, str]],
    user_message: str,
    image_data_uri: str | None = None,
) -> str:
    """Send one grounded chat turn to Groq and return the reply text."""
    if not is_configured():
        raise ChatConfigError(
            "GROQ_API_KEY is not set. Add it to a .env file in the project root "
            "to enable the suggestion assistant."
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": build_messages(analysis, history, user_message, image_data_uri),
        "max_tokens": GROQ_MAX_TOKENS,
        "temperature": 0.6,
    }

    try:
        response = httpx.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=GROQ_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        raise ChatUpstreamError(f"Could not reach Groq: {exc}") from exc

    if response.status_code == 401:
        raise ChatConfigError("Groq rejected the API key (401). Check GROQ_API_KEY.")

    if response.status_code == 404:
        raise ChatUpstreamError(
            f"Groq has no model '{GROQ_MODEL}'. Call GET /api/chat/models to list "
            "the ids your key can use, then set GROQ_MODEL in .env."
        )

    if response.status_code >= 400:
        raise ChatUpstreamError(f"Groq error {response.status_code}: {response.text[:400]}")

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ChatUpstreamError("Groq returned no reply.")

    return (choices[0].get("message") or {}).get("content", "").strip()

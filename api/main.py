from __future__ import annotations

import base64
import sys
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import chat
from analysis.aoi import AOIRecord, draw_aoi_boxes
from analysis.detection import detect_poster_elements
from analysis.recommendations import build_recommendations
from analysis.saliency import (
    attention_percentages_from_saliency,
    saliency_heatmap_overlay,
)
from analysis.scoring import calculate_pes, score_poster
from config.settings import GROQ_MODEL
from models.detector import ModelNotFoundError

app = FastAPI(title="Poster PES API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Analyses are kept in-process so the chat turn can reference the poster and its
# numbers without the browser re-uploading either. Bounded because this is a
# long-running server, not a request-scoped cache; a restart clears it, which is
# acceptable for a single-node deployment.
_MAX_CACHED_ANALYSES = 32
_analyses: OrderedDict[str, dict[str, Any]] = OrderedDict()


class ChatRequest(BaseModel):
    analysis_id: str
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(default_factory=list)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "chat_configured": chat.is_configured(),
        "chat_model": GROQ_MODEL,
    }


@app.get("/api/chat/models")
def chat_models() -> dict[str, Any]:
    """List model ids the configured Groq key can reach (troubleshooting aid)."""
    try:
        return {"configured_model": GROQ_MODEL, "available": chat.list_models()}
    except chat.ChatConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except chat.ChatUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/analyze-poster")
async def analyze_poster(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")

    image_bytes = await file.read()
    image = decode_image(image_bytes)

    try:
        records, source = detect_poster_elements(image, conf=0.25)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Poster detection failed. Check MODEL_WEIGHTS and model compatibility. {exc}",
        ) from exc

    # Attention share drives PES, so it is estimated from saliency rather than
    # box area — see analysis/saliency.py for why.
    percentages = attention_percentages_from_saliency(image, records)
    pes = calculate_pes(percentages, records)
    legacy_scores = score_poster(records, percentages)
    recommendations = build_recommendations(percentages, records, legacy_scores)

    analysis_id = uuid.uuid4().hex
    payload = {
        "analysis_id": analysis_id,
        "filename": file.filename,
        "source": source,
        "detections": [record.to_dict() for record in records],
        "pes": {
            "mode": "saliency_attention",
            "score": pes["score"],
            "category": pes["category"],
            "components": pes["components"],
            "weights": pes["weights"],
            "insights": pes["insights"],
            "attention_percentages": percentages,
        },
        "recommendations": recommendations,
        "annotated_image": encode_annotated_image(image, records),
        "saliency_image": encode_image_data_uri(saliency_heatmap_overlay(image)),
    }

    _remember_analysis(analysis_id, payload, image)
    return payload


@app.post("/api/chat")
def chat_turn(request: ChatRequest) -> dict[str, str]:
    cached = _analyses.get(request.analysis_id)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="That analysis is no longer in memory. Re-analyze the poster to continue.",
        )

    try:
        reply = chat.ask(
            analysis=cached["payload"],
            history=request.history,
            user_message=request.message,
            image_data_uri=cached["image_data_uri"],
        )
    except chat.ChatConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except chat.ChatUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"reply": reply}


def _remember_analysis(analysis_id: str, payload: dict[str, Any], image: np.ndarray) -> None:
    _analyses[analysis_id] = {
        "payload": payload,
        "image_data_uri": chat.encode_image_for_chat(image),
    }
    while len(_analyses) > _MAX_CACHED_ANALYSES:
        _analyses.popitem(last=False)


def decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")
    return image


def encode_image_data_uri(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode image.")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def encode_annotated_image(image: np.ndarray, records: list[AOIRecord]) -> str:
    preview_rgb, _ = draw_aoi_boxes(image, records)
    return encode_image_data_uri(cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))

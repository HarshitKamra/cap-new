"""Central configuration loaded from environment variables with sensible defaults."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Populate os.environ from a project-root .env, without overriding real env vars.

    Uses python-dotenv when installed and falls back to a minimal parser so the
    app runs on a bare `pip install -r requirements.txt` either way.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

DATASET_DIR = Path(os.getenv("CAPSTONE_DATASET_DIR", PROJECT_ROOT / "Capstone.yolov8"))
DATASET_YAML = DATASET_DIR / "data.yaml"
DATASET_IMAGE_DIR = DATASET_DIR / "train" / "images"
DATASET_LABEL_DIR = DATASET_DIR / "train" / "labels"

DEFAULT_MODEL_WEIGHTS = Path(
    os.getenv(
        "MODEL_WEIGHTS",
        PROJECT_ROOT / "models" / "weights" / "best.pt",
    )
)

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".avif")

# Class order matches Capstone.yolov8/data.yaml
CLASS_NAMES: dict[int, str] = {
    0: "CTA",
    1: "Headline",
    2: "Price",
    3: "Product",
    4: "logo",
}

CLASS_NAME_TO_ID = {name: class_id for class_id, name in CLASS_NAMES.items()}

CORE_AOI_ORDER = ["Product", "Headline", "CTA", "Price", "logo"]
BACKGROUND_LABEL = "Background"

RAW_DATA_DEFAULT_FIXATION_MS = 300

# --- Groq chat (suggestion assistant) -------------------------------------
# The key is read server-side only and is never sent to the browser.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Groq rotates its hosted model line-up, so this is configurable rather than
# pinned in code. If the default 404s, GET /api/chat/models lists the ids your
# key can actually reach and you set GROQ_MODEL to one of them.
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

# Groq vision limits: 20MB per request, 3 images max, 2048 tokens per image.
# Posters are downscaled below this edge length before being sent.
GROQ_IMAGE_MAX_EDGE = int(os.getenv("GROQ_IMAGE_MAX_EDGE", "1024"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "900"))
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "60"))

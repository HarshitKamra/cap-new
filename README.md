# Capstone — Advertisement Poster Analysis

Upload a food & beverage poster, detect its design elements with YOLOv8, and get a
Poster Effectiveness Score (PES) plus a grounded chat assistant for improvements.

Prerequisites
- Python 3.10+
- Node 18+ (for the React frontend)
- Recommended: create a virtualenv and install `pip install -r requirements.txt`.

Key components
- `api/main.py` — FastAPI backend (`/api/analyze-poster`, `/api/chat`).
- `frontend/` — React + Vite UI (upload, boxes, PES breakdown, chat).
- `analysis/` — modular analysis helpers (poster, gaze, attention, detection, saliency, scoring, chat).
- `models/detector.py` — YOLO inference helpers.
- `training/` — `split_dataset.py`, `train.py`, and `inference.py` using Ultralytics YOLOv8.
- `aoi_visualizer.py` — legacy CLI for poster AOI preview and gaze analysis.

## How the PES score is computed

1. **Detect AOIs** — YOLOv8 returns boxes classed `Product`, `Headline`, `CTA`,
   `Price`, `logo`.
2. **Estimate attention** — `analysis/saliency.py` builds a spectral-residual
   saliency map with a centre-bias prior, then assigns saliency mass to AOIs
   using a smallest-box-wins ownership rule so overlapping boxes cannot
   double-count. This replaces box-area as the attention proxy, so a small
   high-contrast CTA can out-score a large flat logo.
3. **Score five components** (0–100 each) in `analysis/scoring.py`:
   Product Attention, CTA Visibility, and Headline Engagement are scored against
   ideal attention bands; Attention Balance rewards coverage and penalises
   one-element dominance; Visual Hierarchy checks that Product leads with
   Headline and CTA close behind.
4. **Weighted sum** using the weights in `config/scoring.yaml` (sums to 100).
5. **Category** — ≥80 Excellent, ≥65 Good, ≥50 Average, else Needs Improvement.

All bands, weights, and category cut-offs live in `config/scoring.yaml`, so they
can be tuned without touching code.

**Caveat:** the bands and weights are design heuristics, not values validated
against real advertising performance. PES measures internal design quality, not
predicted click-through. Grounding it properly means calibrating against real
gaze data — `webgazer_choose_poster.html` collects that.

## Configuration

Copy `.env.example` to `.env` (gitignored) and fill in:

```
GROQ_API_KEY=gsk_...        # enables the chat assistant
GROQ_MODEL=...              # optional; see GET /api/chat/models
MODEL_WEIGHTS=...           # optional; defaults to models/weights/best.pt
```

The Groq key is read server-side only and is never sent to the browser. Groq
rotates its hosted models, so if the default model returns 404, start the API and
call `GET /api/chat/models` to list the ids your key can reach.

Notes
- The repository does not include trained YOLO weights. Set `MODEL_WEIGHTS` or pass `--weights` to `training/inference.py`.
- The WebGazer capture UI is in `webgazer_choose_poster.html` and exports CSVs compatible with the gaze parser.

Quick commands

Run the smoke test suite:
```
python -m pytest -q
```

Run inference (requires a `.pt` checkpoint):
```
python training/inference.py --weights path/to/best.pt --source path/to/poster.jpg --out inference_out
```

Start training (requires `ultralytics`):
```
python training/train.py --data Capstone.yolov8/data.yaml --epochs 50
```

Run the Streamlit app (local):
```
pip install -r requirements.txt
streamlit run app/app.py
```

Build Docker image:
```
docker build -t capstone-poster:latest .
```

Run tests locally:
```
python -m pytest -q
```

CI and deployment notes
-----------------------
- To enable the CI workflow to push Docker images, add the following repository secrets in GitHub:
	- `DOCKERHUB_USERNAME` — your Docker Hub user name
	- `DOCKERHUB_TOKEN` — a Docker Hub access token (not your password)

- To run production Compose, set `MODEL_WEIGHTS` and `DOCKER_IMAGE` env vars, for example:

```
export MODEL_WEIGHTS=/path/to/best.pt
export DOCKER_IMAGE=myuser/capstone-poster:latest
docker compose -f docker-compose.prod.yml up -d --build
```

Security note: Do not commit trained weights to the repository; reference them via environment variables or an external artifact store.

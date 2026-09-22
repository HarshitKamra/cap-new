"""Run YOLO inference on one image or a directory and save predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# Run directly as `python training/inference.py`, so the project root is not on
# sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.aoi import draw_aoi_boxes  # noqa: E402
from analysis.detection import detections_to_aoi_records  # noqa: E402
from analysis.poster import read_image  # noqa: E402
from config.settings import DEFAULT_MODEL_WEIGHTS  # noqa: E402
from models.detector import ModelNotFoundError, PosterDetector  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run poster YOLO inference")
    parser.add_argument("source", type=Path, help="Image file or directory")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_MODEL_WEIGHTS,
        help="Path to trained .pt weights",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference"),
        help="Directory for JSON predictions and optional visualizations",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save annotated poster images",
    )
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
        return sorted(
            path for path in source.iterdir() if path.suffix.lower() in extensions
        )
    raise FileNotFoundError(f"Source not found: {source}")


def main() -> None:
    args = parse_args()
    detector = PosterDetector(weights_path=args.weights, conf_threshold=args.conf)

    if not detector.is_available:
        raise ModelNotFoundError(
            f"Weights not found: {args.weights}\nTrain first: python training/train.py --copy-weights"
        )

    images = collect_images(args.source)
    if not images:
        raise FileNotFoundError(f"No images found at: {args.source}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for image_path in images:
        image = read_image(image_path)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        detections = detector.predict(image_path, conf=args.conf)
        records = detections_to_aoi_records(detections, image.shape)

        result = {
            "image": str(image_path),
            "detection_count": len(records),
            "detections": [record.to_dict() for record in records],
        }
        all_results.append(result)

        stem = image_path.stem
        json_path = args.output_dir / f"{stem}_predictions.json"
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)

        print(f"{image_path.name}: {len(records)} detections -> {json_path.name}")

        if args.save_viz and records:
            preview, _ = draw_aoi_boxes(image, records)
            viz_path = args.output_dir / f"{stem}_viz.jpg"
            cv2.imwrite(str(viz_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            print(f"  visualization -> {viz_path.name}")

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)

    print(f"\nProcessed {len(all_results)} image(s). Summary: {summary_path}")


if __name__ == "__main__":
    main()

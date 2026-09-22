"""Train YOLOv8 poster element detector on Capstone.yolov8 dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Run directly as `python training/train.py`, so the project root is not on
# sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (  # noqa: E402
    DATASET_DIR,
    DATASET_YAML,
    DEFAULT_MODEL_WEIGHTS,
    PROJECT_ROOT,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Capstone poster dataset")
    parser.add_argument("--data", type=Path, default=DATASET_YAML, help="Path to data.yaml")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model (yolov8n.pt, yolov8s.pt, etc.)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="", help="CUDA device id or 'cpu'")
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "runs" / "detect",
        help="Ultralytics project directory",
    )
    parser.add_argument("--name", default="capstone_poster", help="Run name")
    parser.add_argument(
        "--copy-weights",
        action="store_true",
        help="Copy best.pt to models/weights/best.pt after training",
    )
    return parser.parse_args()


def write_resolved_dataset_yaml(data_yaml: Path, output_dir: Path) -> Path:
    """Write a copy of data.yaml with an absolute `path`, and return its location.

    The committed data.yaml uses `path: .` so the dataset stays portable, but
    Ultralytics resolves a relative `path` against the working directory rather
    than against the yaml's own folder — which sends it looking for
    `<cwd>/valid/images`. Rewriting `path` to the real dataset root before
    training makes it work from any cwd without editing the tracked file.
    """
    import yaml

    with open(data_yaml, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    config["path"] = str(data_yaml.resolve().parent)

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "data.resolved.yaml"
    with open(resolved_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return resolved_path


def main() -> None:
    args = parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    valid_images = DATASET_DIR / "valid" / "images"
    if not valid_images.is_dir() or not any(valid_images.iterdir()):
        raise FileNotFoundError(
            "Validation split not found. Run first:\n"
            "  python training/split_dataset.py"
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics: pip install ultralytics") from exc

    resolved_data = write_resolved_dataset_yaml(args.data, args.project)

    model = YOLO(args.model)
    results = model.train(
        data=str(resolved_data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else args.project / args.name
    best_weights = save_dir / "weights" / "best.pt"

    print(f"\nTraining complete. Run directory: {save_dir}")
    if best_weights.is_file():
        print(f"Best weights: {best_weights}")
        if args.copy_weights:
            DEFAULT_MODEL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_weights, DEFAULT_MODEL_WEIGHTS)
            print(f"Copied to: {DEFAULT_MODEL_WEIGHTS}")
    else:
        print("Warning: best.pt not found in run directory.")


if __name__ == "__main__":
    main()

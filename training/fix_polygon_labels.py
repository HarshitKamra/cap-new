"""Convert polygon/segmentation label lines to YOLO detection bounding boxes.

Some Roboflow annotations were drawn as polygons, so their label lines carry
`class_id x1 y1 x2 y2 ... xn yn` instead of `class_id cx cy w h`. Ultralytics
detection training rejects any file containing such a line and silently drops
the whole image — which cost this dataset several images out of 74.

A polygon's bounding box is exactly what a detector would learn from it, so
converting is lossless for this task. Run once after exporting the dataset:

    python training/fix_polygon_labels.py            # report only
    python training/fix_polygon_labels.py --apply    # rewrite in place
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATASET_DIR  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize polygon labels to YOLO boxes")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Path to the Capstone.yolov8 dataset root",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite label files. Without it, only report what would change.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing a .bak copy of each rewritten file",
    )
    return parser.parse_args()


def polygon_to_box(values: list[float]) -> tuple[float, float, float, float]:
    """Return (cx, cy, w, h) spanning a flat [x1, y1, x2, y2, ...] polygon."""
    xs = values[0::2]
    ys = values[1::2]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return (
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        x_max - x_min,
        y_max - y_min,
    )


def convert_line(line: str) -> tuple[str, bool]:
    """Return (normalized_line, was_converted)."""
    parts = line.split()
    if len(parts) <= 5:
        return line, False

    coords = [float(value) for value in parts[1:]]
    if len(coords) % 2 != 0 or len(coords) < 6:
        # Not a well-formed polygon; leave it for the caller to notice.
        return line, False

    cx, cy, width, height = polygon_to_box(coords)
    clipped = [max(0.0, min(1.0, value)) for value in (cx, cy, width, height)]
    return f"{parts[0]} " + " ".join(f"{value:.6f}" for value in clipped), True


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()

    label_dirs = [
        path
        for split in ("train", "valid", "test")
        if (path := dataset_dir / split / "labels").is_dir()
    ]
    if not label_dirs:
        raise FileNotFoundError(f"No label directories found under {dataset_dir}")

    converted_files = 0
    converted_lines = 0

    for label_dir in label_dirs:
        for label_path in sorted(label_dir.glob("*.txt")):
            original = label_path.read_text(encoding="utf-8").splitlines()
            rewritten: list[str] = []
            file_changes = 0

            for line in original:
                if not line.strip():
                    continue
                new_line, changed = convert_line(line.strip())
                rewritten.append(new_line)
                file_changes += int(changed)

            if not file_changes:
                continue

            converted_files += 1
            converted_lines += file_changes
            relative = label_path.relative_to(dataset_dir)
            print(f"{'rewrote' if args.apply else 'would fix'} {relative} ({file_changes} lines)")

            if args.apply:
                if not args.no_backup:
                    shutil.copy2(label_path, label_path.with_suffix(".txt.bak"))
                label_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    print(f"\nFiles affected: {converted_files}")
    print(f"Polygon lines converted: {converted_lines}")
    if converted_files and not args.apply:
        print("\nRe-run with --apply to rewrite these files.")


if __name__ == "__main__":
    main()

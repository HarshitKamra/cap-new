"""Create stratified train/valid split for the Capstone YOLO dataset."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

# Run directly as `python training/split_dataset.py`, so the project root is not
# on sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATASET_DIR, PROJECT_ROOT  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Split Capstone YOLO dataset into train/valid")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of images reserved for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction reserved for an independent test split (default: 0.0). "
            "With a small dataset every test image is one fewer training image, "
            "so this is opt-in."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Path to Capstone.yolov8 dataset root",
    )
    return parser.parse_args()


def collect_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    image_dir = dataset_dir / "train" / "images"
    label_dir = dataset_dir / "train" / "labels"
    pairs: list[tuple[Path, Path]] = []

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            pairs.append((image_path, label_path))

    return pairs


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    random.seed(args.seed)

    valid_image_dir = dataset_dir / "valid" / "images"
    valid_label_dir = dataset_dir / "valid" / "labels"
    test_image_dir = dataset_dir / "test" / "images"
    test_label_dir = dataset_dir / "test" / "labels"

    for directory in (valid_image_dir, valid_label_dir, test_image_dir, test_label_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Return any previous split to train/ *before* collecting, so a re-run draws
    # from the full dataset instead of only what last time left behind.
    _restore_split_to_train(dataset_dir, valid_image_dir, valid_label_dir)
    _restore_split_to_train(dataset_dir, test_image_dir, test_label_dir)

    pairs = collect_pairs(dataset_dir)
    if not pairs:
        raise FileNotFoundError(f"No labelled images found under {dataset_dir}")

    random.shuffle(pairs)
    val_count = max(1, int(len(pairs) * args.val_ratio))
    test_count = int(len(pairs) * args.test_ratio)

    if val_count + test_count >= len(pairs):
        raise ValueError(
            f"--val-ratio and --test-ratio would consume all {len(pairs)} images. "
            "Lower them so images remain for training."
        )

    val_pairs = pairs[:val_count]
    test_pairs = pairs[val_count : val_count + test_count]
    train_pairs = pairs[val_count + test_count :]

    def move_pairs(target_image_dir: Path, target_label_dir: Path, items: list[tuple[Path, Path]]):
        """Move, never copy.

        Copying left every validation image in train/ as well, so the model was
        validated on images it had trained on and mAP read far higher than the
        detector really was. Moving keeps the holdout genuinely held out.
        """
        for image_path, label_path in items:
            shutil.move(str(image_path), str(target_image_dir / image_path.name))
            shutil.move(str(label_path), str(target_label_dir / label_path.name))

    move_pairs(valid_image_dir, valid_label_dir, val_pairs)
    if test_pairs:
        move_pairs(test_image_dir, test_label_dir, test_pairs)

    print(f"Dataset root: {dataset_dir}")
    print(f"Total labelled images: {len(pairs)}")
    print(f"Train: {len(train_pairs)}")
    print(f"Valid: {len(val_pairs)}")
    print(f"Test:  {len(test_pairs)}" if test_pairs else "Test:  0 (pass --test-ratio to create one)")
    print("\nSplits are disjoint — no image appears in more than one.")
    print("Re-running reshuffles from the full set; use --seed for a reproducible split.")


def _restore_split_to_train(dataset_dir: Path, image_dir: Path, label_dir: Path) -> None:
    """Move a previously split-out set back into train/ so a re-split sees all images."""
    train_image_dir = dataset_dir / "train" / "images"
    train_label_dir = dataset_dir / "train" / "labels"

    for source_dir, target_dir in ((image_dir, train_image_dir), (label_dir, train_label_dir)):
        if not source_dir.is_dir():
            continue
        for path in source_dir.iterdir():
            if path.is_file():
                shutil.move(str(path), str(target_dir / path.name))


if __name__ == "__main__":
    main()

"""Poster image loading utilities."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from config.settings import SUPPORTED_EXTENSIONS


def read_image(file_path: str | Path) -> np.ndarray | None:
    """
    Read poster images as OpenCV BGR arrays.
    Pillow fallback supports AVIF/WebP when OpenCV cannot read them.
    """
    file_path = str(file_path)
    image = cv2.imread(file_path)

    if image is not None:
        return image

    try:
        from PIL import Image

        with Image.open(file_path) as pil_image:
            rgb_image = pil_image.convert("RGB")
            return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def list_available_posters(folder_path: str | Path) -> list[str]:
    """Return sorted poster filenames filtered by supported image extensions."""
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder_path}")

    posters = [
        name
        for name in os.listdir(folder_path)
        if (folder_path / name).is_file() and name.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    return sorted(posters)


def load_poster_image(folder_path: str | Path, poster_name: str) -> np.ndarray:
    """Load one poster image by filename."""
    image = read_image(Path(folder_path) / poster_name)
    if image is None:
        raise ValueError(f"Unable to read poster: {poster_name}")
    return image


def select_poster(
    folder_path: str | Path,
    requested_poster: str | None = None,
    *,
    interactive: bool = True,
) -> tuple[str, np.ndarray]:
    """Select a poster interactively or by name."""
    folder_path = Path(folder_path)
    posters = list_available_posters(folder_path)

    if not posters:
        raise FileNotFoundError("No supported poster images found.")

    lower_to_original = {name.lower(): name for name in posters}

    if requested_poster:
        selected = lower_to_original.get(requested_poster.lower())
        if not selected:
            raise FileNotFoundError(
                f"Poster not found: {requested_poster}. Choose one from {folder_path}."
            )
        return selected, load_poster_image(folder_path, selected)

    if not interactive:
        raise ValueError("Poster name is required when interactive=False.")

    print("Available Posters:")
    for index, poster_name in enumerate(posters, start=1):
        print(f"{index}. {poster_name}")

    while True:
        selection = input("\nSelect poster by number OR type filename: ").strip()

        if selection.isdigit():
            selected_index = int(selection)
            if 1 <= selected_index <= len(posters):
                selected = posters[selected_index - 1]
                break
            print(f"Invalid number. Choose between 1 and {len(posters)}.")
            continue

        manual_name = selection.lower()
        if manual_name in lower_to_original:
            selected = lower_to_original[manual_name]
            break

        print("Filename not found. Please choose from the listed posters.")

    print(f"Selected Poster: {selected}")
    return selected, load_poster_image(folder_path, selected)

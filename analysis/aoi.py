"""Standardized Areas-of-Interest (AOI) representation and YOLO label parsing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from config.settings import CLASS_NAMES, CORE_AOI_ORDER


@dataclass(frozen=True)
class AOIRecord:
    """Internal representation for one detected or annotated poster element."""

    class_name: str
    class_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    area: int
    normalized_area: float
    center_x: float
    center_y: float
    relative_position: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_legacy_box(self) -> tuple[str, int, int, int, int]:
        """Return (label, x1, y1, x2, y2) for backward compatibility."""
        return (self.class_name, self.x1, self.y1, self.x2, self.y2)


def _relative_position(center_x: float, center_y: float, width: int, height: int) -> str:
    """Map center point to a nine-zone grid label."""
    x_ratio = center_x / max(width, 1)
    y_ratio = center_y / max(height, 1)

    if y_ratio < 1 / 3:
        vertical = "top"
    elif y_ratio < 2 / 3:
        vertical = "center"
    else:
        vertical = "bottom"

    if x_ratio < 1 / 3:
        horizontal = "left"
    elif x_ratio < 2 / 3:
        horizontal = "center"
    else:
        horizontal = "right"

    if vertical == "center" and horizontal == "center":
        return "center"
    if horizontal == "center":
        return vertical
    if vertical == "center":
        return horizontal
    return f"{vertical}-{horizontal}"


def build_aoi_record(
    class_id: int,
    x_center_norm: float,
    y_center_norm: float,
    width_norm: float,
    height_norm: float,
    image_width: int,
    image_height: int,
    confidence: float = 1.0,
    class_names: dict[int, str] | None = None,
) -> AOIRecord:
    """Convert YOLO-normalized box to a pixel AOI record."""
    names = class_names or CLASS_NAMES
    poster_area = max(image_width * image_height, 1)

    x_center = x_center_norm * image_width
    y_center = y_center_norm * image_height
    box_width = width_norm * image_width
    box_height = height_norm * image_height

    x1 = int(round(x_center - box_width / 2))
    y1 = int(round(y_center - box_height / 2))
    x2 = int(round(x_center + box_width / 2))
    y2 = int(round(y_center + box_height / 2))

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    area = width * height
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return AOIRecord(
        class_name=names.get(class_id, f"class_{class_id}"),
        class_id=class_id,
        confidence=float(confidence),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        width=width,
        height=height,
        area=area,
        normalized_area=area / poster_area,
        center_x=center_x,
        center_y=center_y,
        relative_position=_relative_position(center_x, center_y, image_width, image_height),
    )


def parse_yolo_label_lines(
    lines: list[str],
    image_shape: tuple[int, ...],
    class_names: dict[int, str] | None = None,
    confidence: float = 1.0,
) -> list[AOIRecord]:
    """Parse YOLO annotation lines into AOI records."""
    height, width = image_shape[:2]
    records: list[AOIRecord] = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]

        if len(values) == 4:
            x_center, y_center, box_w, box_h = values
        elif len(values) >= 6 and len(values) % 2 == 0:
            # Polygon/segmentation annotation: collapse it to its bounding box
            # rather than reading the first four numbers as one, which would
            # place the box on the first two vertices.
            xs, ys = values[0::2], values[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
            box_w, box_h = x_max - x_min, y_max - y_min
        else:
            continue

        records.append(
            build_aoi_record(
                class_id=class_id,
                x_center_norm=x_center,
                y_center_norm=y_center,
                width_norm=box_w,
                height_norm=box_h,
                image_width=width,
                image_height=height,
                confidence=confidence,
                class_names=class_names,
            )
        )

    return records


def parse_yolo_labels(
    label_path: str,
    image_shape: tuple[int, ...],
    class_names: dict[int, str] | None = None,
    confidence: float = 1.0,
) -> list[AOIRecord]:
    """Load a YOLO label file and return AOI records."""
    with open(label_path, encoding="utf-8") as file:
        lines = file.readlines()
    return parse_yolo_label_lines(lines, image_shape, class_names, confidence)


def get_present_aoi_labels(records: list[AOIRecord]) -> list[str]:
    """Return unique AOI class names in a stable marketing order."""
    labels = {record.class_name for record in records}
    ordered = [label for label in CORE_AOI_ORDER if label in labels]
    extras = sorted(labels - set(CORE_AOI_ORDER))
    return ordered + extras


def aoi_records_to_legacy_boxes(records: list[AOIRecord]) -> list[tuple[str, int, int, int, int]]:
    return [record.to_legacy_box() for record in records]


def draw_aoi_boxes(
    image: np.ndarray,
    records: list[AOIRecord],
) -> tuple[np.ndarray, list[tuple[str, int, int, int, int]]]:
    """Draw AOI rectangles on a poster preview and return RGB image + legacy boxes."""
    preview = image.copy()

    for record in records:
        cv2.rectangle(preview, (record.x1, record.y1), (record.x2, record.y2), (0, 255, 0), 2)
        cv2.putText(
            preview,
            record.class_name,
            (record.x1, max(record.y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    return preview_rgb, aoi_records_to_legacy_boxes(records)

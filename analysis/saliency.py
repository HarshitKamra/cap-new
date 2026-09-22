"""Saliency-based attention estimation.

PES is defined over *attention percentages* — the share of viewer gaze each AOI
captures. With no eye tracker in the loop we estimate that share from a visual
saliency map instead of from box area, so that a small high-contrast CTA can
out-score a large flat logo.

Method: spectral-residual saliency (Hou & Zhang, CVPR 2007), implemented on
numpy FFT because `cv2.saliency` lives in opencv-contrib, which this project
does not depend on. The raw map is multiplied by a centre-bias prior, since
viewers reliably fixate the middle of a poster first.

Saliency mass is then assigned to AOIs with a single-owner rule (see
`_ownership_map`) so overlapping boxes cannot double-count attention.
"""

from __future__ import annotations

import cv2
import numpy as np

from analysis.aoi import AOIRecord
from config.settings import BACKGROUND_LABEL

# Spectral residual is computed at low resolution by design — the technique
# relies on the coarse amplitude spectrum, and 64px is the size used in the
# original paper.
_WORK_SIZE = 64

# Strength of the centre-bias prior, as a fraction of the poster's half-width.
# Larger sigma = flatter prior = less centre preference.
_CENTRE_SIGMA = 0.45

# Floor applied to the centre prior so edge content is damped, never zeroed.
_CENTRE_FLOOR = 0.25


def compute_saliency_map(image: np.ndarray) -> np.ndarray:
    """Return a per-pixel saliency map, same H×W as `image`, non-negative."""
    if image is None or image.size == 0:
        raise ValueError("Cannot compute saliency for an empty image.")

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (_WORK_SIZE, _WORK_SIZE), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float64)

    spectrum = np.fft.fft2(small)
    log_amplitude = np.log(np.abs(spectrum) + 1e-8)
    phase = np.angle(spectrum)

    # The "expected" log spectrum is the locally averaged one; whatever the
    # image does *beyond* that average is the part that draws the eye.
    averaged = cv2.blur(log_amplitude, (3, 3))
    residual = log_amplitude - averaged

    reconstructed = np.fft.ifft2(np.exp(residual + 1j * phase))
    saliency = np.abs(reconstructed) ** 2

    saliency = cv2.GaussianBlur(saliency, (9, 9), 2.5)
    saliency = cv2.resize(saliency, (width, height), interpolation=cv2.INTER_CUBIC)

    return np.clip(saliency, 0.0, None)


def centre_bias(shape: tuple[int, int]) -> np.ndarray:
    """Gaussian prior favouring the poster centre, floored so edges still count."""
    height, width = shape[:2]

    ys = (np.linspace(0.0, 1.0, height) - 0.5).reshape(-1, 1)
    xs = (np.linspace(0.0, 1.0, width) - 0.5).reshape(1, -1)
    squared_distance = (ys / _CENTRE_SIGMA) ** 2 + (xs / _CENTRE_SIGMA) ** 2

    prior = np.exp(-0.5 * squared_distance)
    return _CENTRE_FLOOR + (1.0 - _CENTRE_FLOOR) * prior


def _ownership_map(shape: tuple[int, int], records: list[AOIRecord]) -> np.ndarray:
    """Assign every pixel to at most one AOI.

    Boxes overlap constantly on real posters — a Headline drawn on top of a
    Product, a Price badge inside a CTA. Summing saliency per box would count
    those pixels twice and push the percentages past 100. Painting boxes
    largest-first means the smallest box containing a pixel wins it, which
    matches how a viewer attributes a glance to the tighter element.

    Returns an int array of indices into `records`, or -1 for background.
    """
    height, width = shape[:2]
    owner = np.full((height, width), -1, dtype=np.int32)

    order = sorted(
        range(len(records)),
        key=lambda i: records[i].width * records[i].height,
        reverse=True,
    )

    for index in order:
        record = records[index]
        x1 = max(0, min(width, record.x1))
        y1 = max(0, min(height, record.y1))
        x2 = max(0, min(width, record.x2))
        y2 = max(0, min(height, record.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        owner[y1:y2, x1:x2] = index

    return owner


def attention_percentages_from_saliency(
    image: np.ndarray,
    records: list[AOIRecord],
) -> dict[str, float]:
    """Estimate per-label attention percentages, including background.

    Percentages sum to 100 across all AOI labels plus `BACKGROUND_LABEL`, which
    is what `analysis.scoring.calculate_pes` expects.
    """
    saliency = compute_saliency_map(image)
    saliency = saliency * centre_bias(saliency.shape)

    total_mass = float(saliency.sum())
    labels = sorted({record.class_name for record in records})

    if total_mass <= 0 or not records:
        percentages = {label: 0.0 for label in labels}
        percentages[BACKGROUND_LABEL] = 100.0 if not records else 0.0
        return percentages

    owner = _ownership_map(saliency.shape, records)

    mass_by_label: dict[str, float] = {label: 0.0 for label in labels}
    for index, record in enumerate(records):
        mass_by_label[record.class_name] += float(saliency[owner == index].sum())

    background_mass = float(saliency[owner == -1].sum())

    percentages = {
        label: round((mass / total_mass) * 100.0, 2)
        for label, mass in mass_by_label.items()
    }
    percentages[BACKGROUND_LABEL] = round((background_mass / total_mass) * 100.0, 2)

    return percentages


def saliency_heatmap_overlay(image: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a colourised saliency map over the poster, for the UI preview."""
    saliency = compute_saliency_map(image)
    saliency = saliency * centre_bias(saliency.shape)

    peak = float(saliency.max())
    normalized = saliency / peak if peak > 0 else saliency
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    return cv2.addWeighted(colored, alpha, image, 1.0 - alpha, 0.0)

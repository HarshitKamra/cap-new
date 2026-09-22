"""Tests for saliency-based attention estimation and the PES it feeds."""

import cv2
import numpy as np
import pytest

from analysis.aoi import build_aoi_record
from analysis.saliency import (
    attention_percentages_from_saliency,
    centre_bias,
    compute_saliency_map,
)
from analysis.scoring import calculate_pes, get_pes_category

POSTER_W, POSTER_H = 400, 600


def make_record(class_id, x1, y1, x2, y2):
    return build_aoi_record(
        class_id=class_id,
        x_center_norm=((x1 + x2) / 2) / POSTER_W,
        y_center_norm=((y1 + y2) / 2) / POSTER_H,
        width_norm=(x2 - x1) / POSTER_W,
        height_norm=(y2 - y1) / POSTER_H,
        image_width=POSTER_W,
        image_height=POSTER_H,
        confidence=0.9,
    )


@pytest.fixture
def poster():
    """A flat pale poster with one large dull block and one small vivid block."""
    image = np.full((POSTER_H, POSTER_W, 3), 210, np.uint8)
    cv2.rectangle(image, (40, 40), (360, 300), (205, 205, 205), -1)  # large, low contrast
    cv2.rectangle(image, (150, 480), (260, 530), (0, 0, 255), -1)  # small, high contrast
    return image


def test_saliency_map_matches_image_size_and_is_non_negative(poster):
    saliency = compute_saliency_map(poster)
    assert saliency.shape == (POSTER_H, POSTER_W)
    assert saliency.min() >= 0.0
    assert saliency.max() > 0.0


def test_empty_image_is_rejected():
    with pytest.raises(ValueError):
        compute_saliency_map(np.zeros((0, 0, 3), np.uint8))


def test_centre_bias_peaks_at_centre_and_never_zeroes_edges():
    prior = centre_bias((POSTER_H, POSTER_W))
    assert prior.shape == (POSTER_H, POSTER_W)
    assert prior[POSTER_H // 2, POSTER_W // 2] == pytest.approx(prior.max())
    assert prior.min() > 0.0


def test_percentages_sum_to_one_hundred(poster):
    records = [make_record(3, 40, 40, 360, 300), make_record(0, 150, 480, 260, 530)]
    percentages = attention_percentages_from_saliency(poster, records)
    assert sum(percentages.values()) == pytest.approx(100.0, abs=0.1)
    assert "Background" in percentages


def test_overlapping_boxes_do_not_double_count(poster):
    """A Headline drawn inside a Product box must not inflate the total past 100."""
    records = [
        make_record(3, 40, 40, 360, 300),  # Product, large
        make_record(1, 80, 100, 300, 200),  # Headline, fully inside Product
    ]
    percentages = attention_percentages_from_saliency(poster, records)
    assert sum(percentages.values()) == pytest.approx(100.0, abs=0.1)
    # The inner box owns its pixels, so both labels get a non-negative share.
    assert percentages["Headline"] > 0
    assert percentages["Product"] >= 0


def test_attention_is_not_merely_area(poster):
    """The large dull block covers far more area than its attention share."""
    records = [make_record(3, 40, 40, 360, 300), make_record(0, 150, 480, 260, 530)]
    percentages = attention_percentages_from_saliency(poster, records)

    product_area_share = ((360 - 40) * (300 - 40)) / (POSTER_W * POSTER_H) * 100
    assert product_area_share > 30  # it really is a big box
    assert percentages["Product"] < product_area_share  # yet wins less attention


def test_no_detections_puts_everything_in_background(poster):
    percentages = attention_percentages_from_saliency(poster, [])
    assert percentages["Background"] == pytest.approx(100.0)


def test_calculate_pes_accepts_aoi_records(poster):
    records = [make_record(3, 40, 40, 360, 300), make_record(0, 150, 480, 260, 530)]
    percentages = attention_percentages_from_saliency(poster, records)
    pes = calculate_pes(percentages, records)

    assert 0 <= pes["score"] <= 100
    assert set(pes["components"]) == {
        "Product Attention",
        "CTA Visibility",
        "Headline Engagement",
        "Attention Balance",
        "Visual Hierarchy",
    }
    assert pes["category"] == get_pes_category(pes["score"])
    assert isinstance(pes["insights"], list)


def test_calculate_pes_accepts_legacy_tuples():
    """aoi_visualizer passes (label, x1, y1, x2, y2) tuples, not AOIRecords."""
    boxes = [("Product", 0, 0, 100, 100), ("CTA", 20, 20, 40, 40)]
    pes = calculate_pes({"Product": 40, "CTA": 15, "Background": 10}, boxes)
    assert 0 <= pes["score"] <= 100


def test_pes_weights_sum_to_one_hundred():
    from analysis.scoring import PES_WEIGHTS

    assert sum(PES_WEIGHTS.values()) == pytest.approx(100.0)

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from analysis.aoi import AOIRecord, get_present_aoi_labels
from config.settings import BACKGROUND_LABEL, CORE_AOI_ORDER, PROJECT_ROOT


LABEL_WEIGHTS = {
    "CTA": 0.4,
    "Product": 0.35,
    "Headline": 0.15,
    "Price": 0.05,
    "logo": -0.1,
}

# Fallbacks used when config/scoring.yaml is absent or unreadable.
_DEFAULT_PES_WEIGHTS = {
    "Product Attention": 25,
    "CTA Visibility": 20,
    "Headline Engagement": 20,
    "Attention Balance": 15,
    "Visual Hierarchy": 20,
}

_DEFAULT_IDEAL_ATTENTION_RANGES = {
    "Product": (30, 45),
    "Headline": (15, 30),
    "CTA": (10, 20),
    "Price": (5, 15),
    "logo": (3, 10),
}

_DEFAULT_PES_CATEGORIES = {"excellent": 80, "good": 65, "average": 50}

SCORING_CONFIG_PATH = PROJECT_ROOT / "config" / "scoring.yaml"


def _load_scoring_config() -> dict[str, Any]:
    """Read config/scoring.yaml, falling back to built-in defaults."""
    try:
        import yaml

        with open(SCORING_CONFIG_PATH, encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
    except Exception:
        loaded = {}

    weights = loaded.get("pes_weights") or _DEFAULT_PES_WEIGHTS
    ranges = loaded.get("ideal_attention_ranges") or _DEFAULT_IDEAL_ATTENTION_RANGES
    categories = loaded.get("pes_categories") or _DEFAULT_PES_CATEGORIES

    return {
        "pes_weights": {str(name): float(weight) for name, weight in weights.items()},
        "ideal_attention_ranges": {
            str(label): (float(bounds[0]), float(bounds[1]))
            for label, bounds in ranges.items()
        },
        "pes_categories": {str(name): float(cut) for name, cut in categories.items()},
    }


_CONFIG = _load_scoring_config()

PES_WEIGHTS: dict[str, float] = _CONFIG["pes_weights"]
IDEAL_ATTENTION_RANGES: dict[str, tuple[float, float]] = _CONFIG["ideal_attention_ranges"]
PES_CATEGORIES: dict[str, float] = _CONFIG["pes_categories"]


def _labels_of(items: Iterable[Any]) -> List[str]:
    """Return AOI labels from either AOIRecord objects or legacy (label, x1, y1, x2, y2) tuples."""
    items = list(items)
    if items and not isinstance(items[0], AOIRecord):
        labels = {item[0] for item in items}
        ordered = [label for label in CORE_AOI_ORDER if label in labels]
        return ordered + sorted(labels - set(CORE_AOI_ORDER))
    return get_present_aoi_labels(items)


def clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
    """Keep a score inside the 0-100 range."""
    return max(minimum, min(maximum, value))


def score_attention(attention_percentages: Dict[str, float]) -> float:
    """Compute a 0-100 attention score using label weights and attention percentages."""
    score = 0.0
    for label, pct in attention_percentages.items():
        weight = LABEL_WEIGHTS.get(label, 0.0)
        score += weight * pct

    # map expected range roughly to 0-100, clamp
    val = max(0.0, min(100.0, score))
    return float(val)


def presence_score(records: List[AOIRecord]) -> float:
    labels = set(get_present_aoi_labels(records))
    # simple heuristic: require Product and CTA for good ad
    score = 0.0
    if "Product" in labels:
        score += 50.0
    if "CTA" in labels:
        score += 50.0
    return float(score)


def score_poster(records: List[AOIRecord], attention_percentages: Dict[str, float]) -> Dict[str, float]:
    att = score_attention(attention_percentages)
    pres = presence_score(records)
    overall = 0.7 * att + 0.3 * pres
    overall = max(0.0, min(100.0, overall))
    return {"attention_score": att, "presence_score": pres, "overall_score": overall}


def score_for_ideal_range(value: float, low: float, high: float) -> float:
    """
    Score an attention percentage against a healthy marketing range.
    Values inside the range get 100; weak or excessive attention is penalized.
    """
    if low <= value <= high:
        return 100

    if value < low:
        return clamp((value / low) * 100) if low else 100

    remaining_space = 100 - high
    if remaining_space <= 0:
        return 0

    return clamp(100 - ((value - high) / remaining_space) * 100)


def calculate_balance_score(percentages: Dict[str, float], boxes: Iterable[Any]) -> float:
    """
    Reward posters where important AOIs receive some attention while background
    attention and one-element domination stay low.
    """
    labels = _labels_of(boxes)

    if not labels:
        return 0

    meaningful_labels = [
        label
        for label in labels
        if percentages.get(label, 0) >= 5
    ]
    coverage_score = (len(meaningful_labels) / len(labels)) * 100

    dominant_attention = max(percentages.get(label, 0) for label in labels)
    dominance_score = (
        100
        if dominant_attention <= 55
        else clamp(100 - ((dominant_attention - 55) * 2))
    )

    background_attention = percentages.get(BACKGROUND_LABEL, 0)
    background_score = clamp(100 - (background_attention * 2))

    return (
        coverage_score * 0.45
        + dominance_score * 0.35
        + background_score * 0.20
    )


def calculate_hierarchy_score(percentages: Dict[str, float], boxes: Iterable[Any]) -> float:
    """
    Reward a marketing-friendly visual hierarchy:
    Product should lead, while Headline and CTA should appear near the top.
    """
    labels = _labels_of(boxes)
    ranked_labels = [
        label
        for label, _ in sorted(
            ((label, percentages.get(label, 0)) for label in labels),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    if not ranked_labels or max(percentages.get(label, 0) for label in labels) == 0:
        return 0

    rank_map = {
        label: rank
        for rank, label in enumerate(ranked_labels, start=1)
    }

    score = 0

    product_rank = rank_map.get("Product")
    if product_rank == 1:
        score += 35
    elif product_rank == 2:
        score += 25
    elif product_rank == 3:
        score += 15

    headline_rank = rank_map.get("Headline")
    if headline_rank and headline_rank <= 3:
        score += 25
    elif percentages.get("Headline", 0) > 0:
        score += 10

    cta_rank = rank_map.get("CTA")
    if cta_rank and cta_rank <= 3:
        score += 25
    elif percentages.get("CTA", 0) > 0:
        score += 10

    background_attention = percentages.get(BACKGROUND_LABEL, 0)
    if background_attention <= 10:
        score += 15
    elif background_attention <= 20:
        score += 8

    return clamp(score)


def get_pes_category(score: float) -> str:
    """Translate a numeric PES into a simple interpretation."""
    if score >= PES_CATEGORIES.get("excellent", 80):
        return "Excellent"
    if score >= PES_CATEGORIES.get("good", 65):
        return "Good"
    if score >= PES_CATEGORIES.get("average", 50):
        return "Average"
    return "Needs Improvement"


def build_design_insights(
    percentages: Dict[str, float],
    boxes: Iterable[Any],
    component_scores: Dict[str, float],
) -> List[str]:
    """Generate human-readable poster improvement notes from PES inputs."""
    labels = set(_labels_of(boxes))
    insights = []

    if "Product" not in labels:
        insights.append("Product AOI is missing, so the main food item cannot be evaluated.")
    elif percentages.get("Product", 0) < 25:
        insights.append("Product attention is low; make the food item larger, clearer, or more central.")
    elif percentages.get("Product", 0) > 55:
        insights.append("Product dominates strongly; supporting text or CTA may need more visual weight.")
    else:
        insights.append("Product visibility is healthy.")

    if "CTA" not in labels:
        insights.append("CTA is missing; add a clear action such as order now, visit, or buy.")
    elif percentages.get("CTA", 0) < 8:
        insights.append("CTA is being ignored; improve contrast, size, or placement.")
    else:
        insights.append("CTA is receiving useful attention.")

    if "Headline" not in labels:
        insights.append("Headline is missing; add a short message to guide viewer understanding.")
    elif percentages.get("Headline", 0) < 10:
        insights.append("Headline engagement is weak; improve readability and position.")

    if percentages.get(BACKGROUND_LABEL, 0) > 25:
        insights.append("Too much attention is going outside AOIs; reduce clutter or mark important regions.")

    if component_scores["Attention Balance"] < 50:
        insights.append("Attention is concentrated on too few elements; improve balance across key AOIs.")

    if component_scores["Visual Hierarchy"] < 60:
        insights.append("Visual hierarchy is weak; Product, Headline, and CTA should guide the viewer in order.")

    return insights


def calculate_pes(
    percentages: Dict[str, float],
    aoi_records: Iterable[Any],
) -> dict[str, Any]:
    """Calculate Poster Effectiveness Score from AOI attention analytics.

    Accepts either AOIRecord objects or legacy (label, x1, y1, x2, y2) tuples.
    """
    component_scores = {
        "Product Attention": score_for_ideal_range(
            percentages.get("Product", 0), *IDEAL_ATTENTION_RANGES["Product"]
        ),
        "CTA Visibility": score_for_ideal_range(
            percentages.get("CTA", 0), *IDEAL_ATTENTION_RANGES["CTA"]
        ),
        "Headline Engagement": score_for_ideal_range(
            percentages.get("Headline", 0), *IDEAL_ATTENTION_RANGES["Headline"]
        ),
        "Attention Balance": calculate_balance_score(percentages, aoi_records),
        "Visual Hierarchy": calculate_hierarchy_score(percentages, aoi_records),
    }

    score = sum(component_scores[name] * weight / 100 for name, weight in PES_WEIGHTS.items())

    return {
        "score": clamp(score),
        "category": get_pes_category(score),
        "components": component_scores,
        "weights": PES_WEIGHTS,
        "insights": build_design_insights(percentages, aoi_records, component_scores),
    }

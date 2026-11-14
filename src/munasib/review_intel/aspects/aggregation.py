"""Per-product aspect statistics and the documented classification rules.

All numbers come from stored review predictions and stored per-review aspect
lists. A mention is one **review** that contains the aspect, so the three label
counts always sum to ``mention_count``.

Pain point (all four must hold):

1. ``mention_count >= min_mentions`` (enforced before classification);
2. ``negativity_lift >= pain_point_min_lift``;
3. ``negative_count >= pain_point_min_negative_mentions``;
4. ``negative_share >= pain_point_min_negative_share``.

Strength (mutually exclusive with pain point):

1. ``mention_count >= min_mentions``;
2. ``positive_share >= strength_min_positive_share``;
3. ``negativity_lift is None`` (no product-level negative baseline) or
   ``negativity_lift <= strength_max_negative_lift``.

Everything else is ``neutral``. The lift formula uses the product's own
predicted negative share as the baseline and a configurable pseudo-count:

``smoothed_negative_share = (negative + smoothing * baseline) / (mentions + smoothing)``

``negativity_lift = smoothed_negative_share / baseline``

When the product has no predicted-negative reviews the lift is undefined and
stored as ``None``; it is never replaced with a fake number.

Examples follow the classification: pain points quote the most helpful negative
reviews, strengths the most helpful positive reviews, and neutral aspects quote
the dominant label (ties resolve negative, neutral, positive). Helpfulness
descending then ``review_id`` ascending makes selection deterministic. Text is
an exact prefix of at most ``example_max_chars`` characters, with no added
ellipsis.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import groupby

from munasib.core.config import AspectsConfig
from munasib.review_intel.aspects.types import (
    ASPECT_NEUTRAL,
    ASPECT_PAIN_POINT,
    ASPECT_STRENGTH,
    AspectExample,
    AspectLabelCounts,
    ProductAspectStats,
)


@dataclass(frozen=True)
class AspectPredictionRow:
    """One predicted review and its stored aspects (input to aggregation)."""

    asin: str
    review_id: str
    predicted_label: int
    aspects: Sequence[str]
    text: str
    helpful_votes: int


def negativity_lift(
    negative_count: int,
    mentions: int,
    product_negative_share: float,
    smoothing: float,
) -> float | None:
    """Compute the smoothed negativity lift against the product baseline.

    Args:
        negative_count: Mentions from reviews predicted negative.
        mentions: All mentions of the aspect (predicted reviews only).
        product_negative_share: Negative predicted reviews / all predicted
            reviews for the product.
        smoothing: Pseudo-count at the product baseline (must be positive).

    Returns:
        Lift (1.0 = exactly the product baseline), or None when the product has
        no negative reviews or there are no mentions.

    Raises:
        ValueError: If a count, share or smoothing value is out of range.
    """
    if negative_count < 0 or mentions < 0:
        raise ValueError("negative_count and mentions must be non-negative")
    if negative_count > mentions:
        raise ValueError("negative_count cannot exceed mentions")
    if not 0.0 <= product_negative_share <= 1.0:
        raise ValueError("product_negative_share must be in [0, 1]")
    if smoothing <= 0.0:
        raise ValueError(f"smoothing must be > 0, got {smoothing}")
    if mentions == 0 or product_negative_share == 0.0:
        return None
    smoothed = (
        negative_count + smoothing * product_negative_share
    ) / (mentions + smoothing)
    return smoothed / product_negative_share


def classify_aspect(
    counts: AspectLabelCounts,
    lift: float | None,
    config: AspectsConfig,
) -> str:
    """Classify an aspect as pain point, strength or neutral.

    Args:
        counts: Review-level label counts.
        lift: Precomputed negativity lift, or None.
        config: Aspect configuration.

    Returns:
        One of ``pain_point``, ``strength`` or ``neutral``.
    """
    if (
        lift is not None
        and lift >= config.pain_point_min_lift
        and counts.negative >= config.pain_point_min_negative_mentions
        and counts.negative_share >= config.pain_point_min_negative_share
    ):
        return ASPECT_PAIN_POINT
    if counts.positive_share >= config.strength_min_positive_share and (
        lift is None or lift <= config.strength_max_negative_lift
    ):
        return ASPECT_STRENGTH
    return ASPECT_NEUTRAL


def example_label(classification: str, counts: AspectLabelCounts) -> int:
    """Return which predicted label supplies the examples.

    Args:
        classification: Stored classification.
        counts: Review-level label counts.

    Returns:
        Negative for pain points, positive for strengths, and the dominant label
        for neutral aspects.
    """
    if classification == ASPECT_PAIN_POINT:
        return 0
    if classification == ASPECT_STRENGTH:
        return 2
    return counts.dominant_label()


def select_examples(
    examples_by_label: dict[int, list[AspectExample]],
    classification: str,
    counts: AspectLabelCounts,
    config: AspectsConfig,
) -> list[AspectExample]:
    """Select the most helpful examples for a classified aspect.

    Args:
        examples_by_label: Candidate examples keyed by predicted label.
        classification: Stored classification.
        counts: Review-level label counts.
        config: Aspect configuration (``max_examples``).

    Returns:
        Up to ``max_examples`` examples, most helpful first with ``review_id``
        ascending as the tie-breaker.
    """
    label = example_label(classification, counts)
    candidates = examples_by_label.get(label, [])
    return sorted(candidates, key=lambda example: example.sort_key())[: config.max_examples]


def aggregate_product(
    asin: str,
    rows: Iterable[AspectPredictionRow],
    config: AspectsConfig,
) -> list[ProductAspectStats]:
    """Aggregate one product's predicted reviews into aspect statistics.

    Args:
        asin: Product ASIN.
        rows: Every predicted review of the product (including reviews with no
            aspects, so the product negative share uses the full denominator).
        config: Aspect and aggregation thresholds.

    Returns:
        One statistic per aspect with at least ``min_mentions`` mentions,
        ordered by mention count descending then aspect.

    Raises:
        ValueError: If a row belongs to another product or carries an unknown
            predicted label.
    """
    materialised = list(rows)
    labeled_reviews = len(materialised)
    product_counts = AspectLabelCounts()
    aspect_counts: defaultdict[str, AspectLabelCounts] = defaultdict(AspectLabelCounts)
    examples: defaultdict[str, defaultdict[int, list[AspectExample]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in materialised:
        if row.asin != asin:
            raise ValueError(f"row for {row.asin!r} passed to aggregate_product({asin!r})")
        if row.predicted_label not in (0, 1, 2):
            raise ValueError(
                f"review {row.review_id!r} has unknown predicted label "
                f"{row.predicted_label!r}"
            )
        product_counts.add(row.predicted_label)
        unique_aspects = {aspect for aspect in row.aspects if aspect}
        if not unique_aspects:
            continue
        example = AspectExample(
            review_id=row.review_id,
            label=row.predicted_label,
            helpful_votes=row.helpful_votes,
            text=row.text[: config.example_max_chars],
        )
        for aspect in unique_aspects:
            aspect_counts[aspect].add(row.predicted_label)
            if example.text:
                examples[aspect][row.predicted_label].append(example)

    if labeled_reviews == 0:
        return []
    product_negative_share = product_counts.negative / labeled_reviews

    results: list[ProductAspectStats] = []
    for aspect, counts in aspect_counts.items():
        if counts.mentions < config.min_mentions:
            continue
        lift = negativity_lift(
            counts.negative,
            counts.mentions,
            product_negative_share,
            config.lift_smoothing,
        )
        classification = classify_aspect(counts, lift, config)
        selected = select_examples(examples[aspect], classification, counts, config)
        results.append(
            ProductAspectStats(
                asin=asin,
                aspect=aspect,
                counts=counts,
                product_labeled_reviews=labeled_reviews,
                product_negative_share=product_negative_share,
                negativity_lift=lift,
                classification=classification,
                examples=selected,
            )
        )
    results.sort(key=lambda stat: (-stat.counts.mentions, stat.aspect))
    return results


def iter_product_aspects(
    rows: Iterable[AspectPredictionRow],
    config: AspectsConfig,
) -> Iterator[ProductAspectStats]:
    """Group a review stream by product and yield every aspect statistic.

    Args:
        rows: Predicted review rows sorted by ``(asin, review_id)``.
        config: Aspect and aggregation thresholds.

    Yields:
        Aspect statistics for every product in ascending ASIN order.
    """
    for asin, group in groupby(rows, key=lambda row: row.asin):
        yield from aggregate_product(asin, group, config)


__all__ = [
    "AspectPredictionRow",
    "aggregate_product",
    "classify_aspect",
    "example_label",
    "iter_product_aspects",
    "negativity_lift",
    "select_examples",
]

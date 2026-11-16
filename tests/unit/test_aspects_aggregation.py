"""Tests for the documented aspect aggregation rules.

These tests pin the exact pain-point rule requested for P13: all of lift,
negative mention count, and negative share must pass. They also pin the
classification-directed example selection and the deterministic truncation.
"""

from __future__ import annotations

import pytest

from munasib.core.config import AspectsConfig
from munasib.review_intel.aspects.aggregation import (
    AspectPredictionRow,
    aggregate_product,
    classification_counts,
    classify_aspect,
    negativity_lift,
)
from munasib.review_intel.aspects.types import (
    ASPECT_NEUTRAL,
    ASPECT_PAIN_POINT,
    ASPECT_STRENGTH,
    AspectExample,
    AspectLabelCounts,
)


def row(
    review_id: str,
    label: int,
    aspects: list[str] | None = None,
    *,
    text: str | None = None,
    helpful: int = 0,
    asin: str = "A1",
) -> AspectPredictionRow:
    """Build one aggregation input row.

    Args:
        review_id: Stable review id.
        label: Predicted sentiment label.
        aspects: Mentioned aspects (None = no aspect).
        text: Review text (defaults to a short sentence).
        helpful: Helpful-vote count.
        asin: Product ASIN.

    Returns:
        Prediction row.
    """
    return AspectPredictionRow(
        asin=asin,
        review_id=review_id,
        predicted_label=label,
        aspects=aspects or [],
        text=text if text is not None else f"review {review_id}",
        helpful_votes=helpful,
    )


def product_with_baseline(
    *,
    negative: int,
    positive: int,
    aspect_negative: int,
    aspect_positive: int,
    aspect: str = "filter",
) -> list[AspectPredictionRow]:
    """Build a product whose one aspect has the requested label counts.

    The remaining reviews are positive and carry no aspect, so the product
    negative baseline is ``negative / (negative + positive)``.

    Args:
        negative: Negative reviews without the aspect.
        positive: Positive reviews without the aspect.
        aspect_negative: Negative reviews that mention the aspect.
        aspect_positive: Positive reviews that mention the aspect.
        aspect: Aspect phrase.

    Returns:
        Aggregation rows.
    """
    rows: list[AspectPredictionRow] = []
    index = 0
    for _ in range(negative):
        rows.append(row(f"n{index:03d}", 0))
        index += 1
    for _ in range(positive):
        rows.append(row(f"p{index:03d}", 2))
        index += 1
    for _ in range(aspect_negative):
        rows.append(row(f"an{index:03d}", 0, [aspect]))
        index += 1
    for _ in range(aspect_positive):
        rows.append(row(f"ap{index:03d}", 2, [aspect]))
    return rows


def test_lift_formula_and_zero_baseline() -> None:
    """Lift uses the smoothing pseudo-count and is None with a zero baseline."""
    # baseline 0.2, 4/10 negative: (4 + 5*0.2)/(10+5) / 0.2 = 5/15/0.2 = 1.6667
    assert negativity_lift(4, 10, 0.2, 5.0) == pytest.approx(5.0 / 15.0 / 0.2)
    assert negativity_lift(0, 10, 0.0, 5.0) is None


def test_small_negative_aspect_is_not_a_pain_point() -> None:
    """1% negative product, 5 mentions, 1 negative: share and count both fail."""
    config = AspectsConfig()
    rows = product_with_baseline(
        negative=0,
        positive=95,
        aspect_negative=1,
        aspect_positive=4,
    )
    stats = aggregate_product("A1", rows, config)
    assert len(stats) == 1
    stat = stats[0]
    assert stat.product_negative_share == pytest.approx(0.01)
    assert stat.counts.mentions == 5
    assert stat.counts.negative_share == pytest.approx(0.2)
    assert stat.classification == ASPECT_NEUTRAL


def test_six_mentions_four_negative_is_a_pain_point() -> None:
    """The same low baseline passes once every pain-point condition holds."""
    config = AspectsConfig()
    rows = product_with_baseline(
        negative=0,
        positive=94,
        aspect_negative=4,
        aspect_positive=2,
    )
    stats = aggregate_product("A1", rows, config)
    assert stats[0].counts.mentions == 6
    assert stats[0].classification == ASPECT_PAIN_POINT
    assert stats[0].counts.negative >= config.pain_point_min_negative_mentions


def test_strength_rule_is_unchanged() -> None:
    """High positive share and low lift classify as strength."""
    config = AspectsConfig()
    rows = product_with_baseline(
        negative=20,
        positive=80,
        aspect_negative=0,
        aspect_positive=10,
    )
    stats = aggregate_product("A1", rows, config)
    assert stats[0].counts.positive_share == 1.0
    assert stats[0].negativity_lift is not None
    assert stats[0].negativity_lift < config.strength_max_negative_lift
    assert stats[0].classification == ASPECT_STRENGTH


def test_pain_point_examples_are_negative_even_when_positives_are_more_numerous() -> None:
    """Examples follow the classification, not the majority label."""
    config = AspectsConfig(max_examples=2)
    rows = product_with_baseline(
        negative=10,
        positive=90,
        aspect_negative=4,
        aspect_positive=8,
    )
    # The aspect has more positive than negative mentions, yet four negatives
    # against a 10% baseline pass the pain-point rule.
    rows = [
        row(item.review_id, item.predicted_label, item.aspects, helpful=5, asin=item.asin)
        for item in rows
    ]
    stats = aggregate_product("A1", rows, config)
    stat = stats[0]
    assert stat.classification == ASPECT_PAIN_POINT
    assert stat.counts.positive > stat.counts.negative
    assert [example.label for example in stat.examples] == [0, 0]
    assert all(example.helpful_votes == 5 for example in stat.examples)


def test_strength_examples_are_positive_and_pain_examples_are_selected() -> None:
    """Strength quotes positive reviews; pain point quotes the most helpful negative."""
    config = AspectsConfig(max_examples=1)
    rows = [
        row("n1", 0, ["motor"], text="bad motor", helpful=1),
        row("n2", 0, ["motor"], text="worse motor", helpful=9),
        row("n3", 0, ["motor"], text="broken motor", helpful=2),
        row("n4", 0, ["motor"], text="dead motor", helpful=0),
        row("p1", 2, ["motor"], text="great motor", helpful=8),
        row("n5", 0, []),
        *[row(f"p{i}", 2, []) for i in range(2, 10)],
    ]
    stats = aggregate_product("A1", rows, config)
    stat = stats[0]
    assert stat.classification == ASPECT_PAIN_POINT
    assert stat.examples[0].text == "worse motor"

    positive_rows = [
        row("p1", 2, ["motor"], text="great motor", helpful=8),
        row("p2", 2, ["motor"], text="fine motor", helpful=2),
        row("p3", 2, ["motor"], text="good motor", helpful=1),
        row("p4", 2, ["motor"], text="nice motor", helpful=0),
        row("p5", 2, ["motor"], text="ok motor", helpful=0),
        row("n1", 0, []),
        row("n2", 0, []),
    ]
    positive_stats = aggregate_product("A1", positive_rows, config)
    assert positive_stats[0].classification == ASPECT_STRENGTH
    assert positive_stats[0].examples[0].text == "great motor"


def test_dominant_label_ties_resolve_negative_neutral_positive() -> None:
    """The documented tie order is negative, then neutral, then positive."""
    counts = AspectLabelCounts(negative=2, neutral=2, positive=2)
    assert counts.dominant_label() == 0
    counts = AspectLabelCounts(negative=0, neutral=2, positive=2)
    assert counts.dominant_label() == 1
    counts = AspectLabelCounts(negative=0, neutral=0, positive=2)
    assert counts.dominant_label() == 2


def test_example_text_is_an_exact_prefix_without_ellipsis() -> None:
    """Truncation never appends an ellipsis or reflows the text."""
    config = AspectsConfig(max_examples=1, example_max_chars=10, min_mentions=1)
    text = "a long review text"
    rows = [row("r1", 0, ["motor"], text=text), *[row(f"n{i}", 0) for i in range(5)]]
    stat = aggregate_product("A1", rows, config)[0]
    assert stat.examples[0].text == text[:10]
    assert "…" not in stat.examples[0].text


def test_min_mentions_drops_low_support_aspects() -> None:
    """An aspect below the mention floor is not stored, even when neutral."""
    config = AspectsConfig(min_mentions=5)
    rows = [row("r1", 0, ["rare"]), row("r2", 0, ["rare"])]
    assert aggregate_product("A1", rows, config) == []


def test_product_negative_share_uses_reviews_without_aspects() -> None:
    """The baseline denominator includes labelled reviews with no aspect."""
    config = AspectsConfig(min_mentions=1)
    rows = [row("r1", 0, ["filter"]), row("r2", 2, ["filter"]), row("r3", 2, [])]
    stat = aggregate_product("A1", rows, config)[0]
    assert stat.product_labeled_reviews == 3
    assert stat.product_negative_share == pytest.approx(1 / 3)


def test_unknown_label_is_rejected() -> None:
    """A corrupt label cannot silently enter the counts."""
    with pytest.raises(ValueError, match="unknown predicted label"):
        aggregate_product("A1", [row("r1", 7, ["filter"])], AspectsConfig())


def test_classify_aspect_requires_all_pain_point_conditions() -> None:
    """Directly pin the conjunction: lift alone or share alone is not enough."""
    config = AspectsConfig()
    counts = AspectLabelCounts(negative=1, neutral=4, positive=0)
    assert classify_aspect(counts, 10.0, config) == ASPECT_NEUTRAL
    counts = AspectLabelCounts(negative=3, neutral=3, positive=0)
    assert classify_aspect(counts, 1.0, config) == ASPECT_NEUTRAL
    assert classify_aspect(counts, 2.0, config) == ASPECT_PAIN_POINT


def test_example_sort_key_uses_helpfulness_then_review_id() -> None:
    """Equal helpfulness falls back to the stable review id."""
    low = AspectExample("b", 0, 1, "x")
    high = AspectExample("a", 0, 1, "y")
    assert sorted([low, high], key=lambda example: example.sort_key()) == [high, low]


def test_classification_counts_summarise_rows_for_the_before_after_report() -> None:
    """The counter reports rows, products and pain-point products exactly."""
    config = AspectsConfig()
    pain_rows = product_with_baseline(
        negative=1, positive=9, aspect_negative=4, aspect_positive=1
    )
    strength_rows = [row(f"m{index}", 2, ["motor"], asin="A2") for index in range(6)]
    neutral_rows = [
        row(f"n{index}", 0, ["box"], asin="A3") for index in range(2)
    ] + [row(f"p{index}", 2, ["box"], asin="A3") for index in range(3)]
    stats = (
        list(aggregate_product("A1", pain_rows, config))
        + list(aggregate_product("A2", strength_rows, config))
        + list(aggregate_product("A3", neutral_rows, config))
    )
    assert [stat.classification for stat in stats] == [
        ASPECT_PAIN_POINT,
        ASPECT_STRENGTH,
        ASPECT_NEUTRAL,
    ]

    before = classification_counts(stats)
    assert before.rows == 3
    assert before.products == 3
    assert before.pain_points == 1
    assert before.strengths == 1
    assert before.neutral == 1
    assert before.products_with_pain_point == 1

    # Excluding the pain row shrinks the table the way the report shows it.
    after = classification_counts(
        [stat for stat in stats if not (stat.asin == "A1")]
    )
    assert after.rows == 2
    assert after.products == 2
    assert after.pain_points == 0
    assert after.products_with_pain_point == 0
    assert after.strengths == 1
    assert after.neutral == 1

"""Tests for the pure per-product evidence indicator arithmetic.

Every expected value here is hand-computed from the synthetic inputs (star
histograms, entropy, weighted means, OLS slopes, sliding-window counts) - the
tests do not re-derive expectations by calling the implementation. Hermetic:
no database, no network, no clock.
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import pytest

from munasib.core.config import EvidenceIndicatorsConfig, EvidenceStrengthConfig
from munasib.review_intel.evidence.indicators import (
    DIRECTION_BENEFIT,
    DIRECTION_COST,
    MONTH_SECONDS,
    Indicator,
    StrengthResult,
    burst_indicator,
    composite_strength,
    concentration_indicators,
    helpfulness_indicators,
    normalized_entropy,
    rating_agreement_indicator,
    rating_text_agreement_indicator,
    recency_indicators,
    review_volume_sufficiency,
    sentiment_shares_indicator,
    star_counts,
    verified_purchase_indicator,
    volume_indicator,
)

REFERENCE = datetime(2018, 10, 4)


def _indicators_config(**overrides: object) -> EvidenceIndicatorsConfig:
    """Build the indicator config with defaults and explicit overrides.

    Args:
        **overrides: Field values overriding the defaults.

    Returns:
        Validated configuration.
    """
    return EvidenceIndicatorsConfig(**overrides)


def _times(values: list[str]) -> pd.Series:
    """Build a naive-UTC datetime series.

    Args:
        values: ISO date strings.

    Returns:
        Datetime series.
    """
    return pd.Series(pd.to_datetime(values))


def _indicator(name: str, normalized: float | None, reason: str | None = None) -> Indicator:
    """Build a minimal indicator stand-in for composite tests.

    Args:
        name: Indicator key.
        normalized: Normalized value (None = missing).
        reason: Missing reason.

    Returns:
        The indicator.
    """
    return Indicator(
        name=name,
        raw_value=None if normalized is None else normalized,
        normalized=normalized,
        direction=DIRECTION_BENEFIT,
        description="stand-in",
        missing_reason=reason,
    )


# ------------------------------------------------------------------------------
# Volume
# ------------------------------------------------------------------------------


def test_review_volume_sufficiency_curve() -> None:
    """Zero reviews score 0, the saturation point scores 1, the curve is monotone."""
    assert review_volume_sufficiency(0, 200) == 0.0
    assert review_volume_sufficiency(200, 200) == pytest.approx(1.0)
    assert review_volume_sufficiency(5_000, 200) == 1.0
    small = review_volume_sufficiency(10, 200)
    large = review_volume_sufficiency(100, 200)
    assert 0.0 < small < large < 1.0


def test_volume_indicator_hand_computed() -> None:
    """The indicator exposes the review count and the log-scaled sufficiency."""
    indicator = volume_indicator(200, _indicators_config())
    assert indicator.raw_value == 200.0
    assert indicator.normalized == pytest.approx(1.0)
    assert indicator.direction == DIRECTION_BENEFIT
    assert indicator.missing_reason is None


def test_volume_indicator_without_reviews_is_missing() -> None:
    """A product with no reviews has no computable volume evidence (never a 0)."""
    indicator = volume_indicator(0, _indicators_config())
    assert indicator.raw_value is None
    assert indicator.normalized is None
    assert indicator.missing_reason == "no_reviews"


# ------------------------------------------------------------------------------
# Rating distribution
# ------------------------------------------------------------------------------


def test_star_counts_and_normalized_entropy() -> None:
    """The histogram and its normalised entropy are hand-checkable."""
    counts = star_counts(pd.Series([1, 1, 5, 5, float("nan")]))
    assert counts == [2, 0, 0, 0, 2]
    assert normalized_entropy(counts) == pytest.approx((1.0, 1.0 / math.log2(5)))
    assert normalized_entropy([3, 0, 0, 0, 0]) == (0.0, 0.0)
    assert normalized_entropy([0, 0, 0, 0, 0]) == (0.0, 0.0)


def test_rating_agreement_unanimous_is_perfect() -> None:
    """Four five-star reviews carry zero entropy, so agreement is 1.0."""
    indicator = rating_agreement_indicator(pd.Series([5, 5, 5, 5]), _indicators_config())
    assert indicator.normalized == 1.0
    assert indicator.extras["entropy_bits"] == 0.0
    assert indicator.extras["mean_rating"] == pytest.approx(5.0)
    assert indicator.extras["star_counts"]["5"] == 4
    assert indicator.extras["polarized"] is False


def test_rating_agreement_polarized_split() -> None:
    """A 50/50 one-star / five-star split is polarized and carries one bit."""
    indicator = rating_agreement_indicator(pd.Series([1, 1, 5, 5]), _indicators_config())
    assert indicator.extras["polarized"] is True
    assert indicator.extras["entropy_bits"] == pytest.approx(1.0)
    assert indicator.normalized == pytest.approx(1.0 - 1.0 / math.log2(5))
    assert indicator.raw_value == indicator.normalized


def test_rating_agreement_one_sided_low_is_not_polarized() -> None:
    """Only reachable both ways: a pure one-star pile is not "polarized"."""
    indicator = rating_agreement_indicator(pd.Series([1, 1, 1, 1]), _indicators_config())
    assert indicator.extras["polarized"] is False


def test_rating_agreement_without_ratings() -> None:
    """No rated reviews means no agreement, reported as missing."""
    ratings = pd.Series([float("nan"), float("nan")], dtype="float64")
    indicator = rating_agreement_indicator(ratings, _indicators_config())
    assert indicator.normalized is None
    assert indicator.missing_reason == "no_ratings"


# ------------------------------------------------------------------------------
# Sentiment shares
# ------------------------------------------------------------------------------


def test_sentiment_shares_hand_computed() -> None:
    """Half negative / half positive with symmetric net sentiment scores 0.5."""
    predicted = pd.Series([0.0, 2.0])
    net = pd.Series([-0.85, 0.85])
    ratings = pd.Series([1.0, 5.0])
    indicator = sentiment_shares_indicator(predicted, net, ratings)
    assert indicator.raw_value == pytest.approx(0.0)
    assert indicator.normalized == pytest.approx(0.5)
    extras = indicator.extras
    assert extras["predictions"] == 2
    assert extras["predicted_negative_share"] == pytest.approx(0.5)
    assert extras["predicted_positive_share"] == pytest.approx(0.5)
    assert extras["star_negative_share"] == pytest.approx(0.5)


def test_sentiment_shares_ignore_unpredicted_reviews() -> None:
    """Reviews without a prediction stay in the extras' context but not the shares."""
    predicted = pd.Series([0.0, float("nan")])
    net = pd.Series([-0.5, float("nan")])
    ratings = pd.Series([1.0, 3.0])
    indicator = sentiment_shares_indicator(predicted, net, ratings)
    assert indicator.extras["predictions"] == 1
    assert indicator.extras["predicted_negative_share"] == pytest.approx(1.0)
    assert indicator.normalized == pytest.approx(0.25)  # (−0.5 + 1) / 2


def test_sentiment_shares_without_predictions() -> None:
    """No predictions at all means the shares are missing, not zero."""
    indicator = sentiment_shares_indicator(
        pd.Series([float("nan")]), pd.Series([float("nan")]), pd.Series([4.0])
    )
    assert indicator.normalized is None
    assert indicator.missing_reason == "no_predictions"


# ------------------------------------------------------------------------------
# Helpfulness
# ------------------------------------------------------------------------------


def test_helpfulness_hand_computed() -> None:
    """Weights (1 + votes) turn [1, -1] with votes [0, 3] into a -0.6 mean."""
    votes = pd.Series([0.0, 3.0])
    net = pd.Series([1.0, -1.0])
    has_prediction = pd.Series([True, True])
    coverage, weighted = helpfulness_indicators(votes, net, has_prediction)
    assert coverage.raw_value == pytest.approx(0.5)
    assert coverage.normalized == pytest.approx(0.5)
    assert coverage.extras["reviews_with_votes"] == 1
    assert weighted.raw_value == pytest.approx(-0.6)
    assert weighted.normalized == pytest.approx(0.2)
    assert weighted.extras["unweighted_net_sentiment"] == pytest.approx(0.0)
    assert weighted.extras["difference"] == pytest.approx(-0.6)


def test_helpfulness_without_reviews_or_predictions() -> None:
    """Both helpfulness indicators name their missing reason explicitly."""
    empty = pd.Series(dtype="float64")
    coverage, weighted = helpfulness_indicators(
        empty, empty, pd.Series(dtype="bool")
    )
    assert coverage.missing_reason == "no_reviews"
    assert weighted.missing_reason == "no_predictions"


# ------------------------------------------------------------------------------
# Verified purchases
# ------------------------------------------------------------------------------


def test_verified_purchase_ratio_excludes_unknowns() -> None:
    """NULL is unknown: it leaves both the numerator and the denominator."""
    indicator = verified_purchase_indicator(
        pd.Series([True, True, False, None], dtype="object")
    )
    assert indicator.raw_value == pytest.approx(2 / 3)
    assert indicator.extras["verified_known"] == 3
    assert indicator.extras["verified_count"] == 2
    assert indicator.extras["verified_unknown"] == 1


def test_verified_purchase_ratio_all_unknown() -> None:
    """With no known flag at all the ratio is missing."""
    indicator = verified_purchase_indicator(pd.Series([None, None], dtype="object"))
    assert indicator.normalized is None
    assert indicator.missing_reason == "verified_unknown"


# ------------------------------------------------------------------------------
# Recency, shift and trend
# ------------------------------------------------------------------------------


def test_recency_and_staleness_anchor_on_the_reference_date() -> None:
    """An old-only product has zero coverage and a large staleness cost."""
    times = _times(["2005-01-01", "2016-10-03"])
    predicted = pd.Series([0.0, 2.0])
    net = pd.Series([-0.9, 0.9])
    has_prediction = predicted.notna() & net.notna()
    coverage, staleness, _shift, _trend = recency_indicators(
        times, predicted, net, has_prediction, REFERENCE, _indicators_config()
    )
    assert coverage.raw_value == 0.0  # 2016-10-03 is one day before the window
    assert coverage.extras["recent_reviews"] == 0
    assert coverage.extras["window_start"] == "2016-10-04T00:00:00Z"
    # 2016-10-03 -> 2018-10-04 is 731 days = 731 * 86400 seconds, i.e.
    # 731 / 30.436875 = 24.0169 mean-Gregorian months.
    expected_months = 731 * 86400.0 / MONTH_SECONDS
    assert staleness.raw_value == pytest.approx(expected_months)
    assert staleness.normalized == pytest.approx(1.0 / (1.0 + expected_months / 12.0))
    assert staleness.direction == DIRECTION_COST


def test_recency_coverage_counts_only_the_window() -> None:
    """The window is inclusive at both ends; earlier reviews fall outside."""
    times = _times(["2016-10-03", "2016-10-04", "2018-10-04"])
    predicted = pd.Series([2.0, 2.0, 2.0])
    net = pd.Series([0.9, 0.9, 0.9])
    has_prediction = predicted.notna() & net.notna()
    coverage, _staleness, _shift, _trend = recency_indicators(
        times, predicted, net, has_prediction, REFERENCE, _indicators_config()
    )
    assert coverage.raw_value == pytest.approx(2 / 3)
    assert coverage.extras["recent_reviews"] == 2


def test_recent_negative_shift_hand_computed() -> None:
    """Recent 60% negative vs overall 50% gives a +0.1 delta and 0.9 score."""
    times = _times(["2018-01-01"] * 10 + ["2005-01-01"] * 10)
    predicted = pd.Series([0.0] * 6 + [2.0] * 4 + [0.0] * 4 + [2.0] * 6)
    net = pd.Series([-0.8] * 6 + [0.8] * 4 + [-0.8] * 4 + [0.8] * 6)
    has_prediction = predicted.notna() & net.notna()
    _coverage, _staleness, shift, _trend = recency_indicators(
        times, predicted, net, has_prediction, REFERENCE, _indicators_config()
    )
    assert shift.raw_value == pytest.approx(0.1)
    assert shift.normalized == pytest.approx(0.9)
    assert shift.extras["recent_negative_share"] == pytest.approx(0.6)
    assert shift.extras["overall_negative_share"] == pytest.approx(0.5)
    assert shift.direction == DIRECTION_COST


def test_recent_negative_shift_needs_the_minimum_recent_predictions() -> None:
    """Nine predicted reviews in the window is below the ten required."""
    times = _times(["2018-01-01"] * 9 + ["2005-01-01"])
    predicted = pd.Series([0.0] * 9 + [2.0])
    net = pd.Series([-0.8] * 10)
    has_prediction = predicted.notna() & net.notna()
    _coverage, _staleness, shift, _trend = recency_indicators(
        times, predicted, net, has_prediction, REFERENCE, _indicators_config()
    )
    assert shift.normalized is None
    assert shift.missing_reason == "recent_window_below_min_reviews"


def test_sentiment_trend_hand_computed_slope() -> None:
    """A +0.1 mean-net change per quarter scores 0.5 + 0.5*(0.1/0.25) = 0.7."""
    quarter_dates = [
        "2016-10-15",
        "2017-01-15",
        "2017-04-15",
        "2017-07-15",
        "2017-10-15",
        "2018-01-15",
        "2018-04-15",
        "2018-07-15",
    ]
    times = _times([date for date in quarter_dates for _ in range(5)])
    predicted = pd.Series([2.0] * 40)
    net = pd.Series([0.1 * index for index in range(8) for _ in range(5)])
    has_prediction = predicted.notna() & net.notna()
    _coverage, _staleness, _shift, trend = recency_indicators(
        times, predicted, net, has_prediction, REFERENCE, _indicators_config()
    )
    assert trend.raw_value == pytest.approx(0.1, rel=1e-9)
    assert trend.normalized == pytest.approx(0.7)
    assert len(trend.extras["quarters_used"]) == 8
    assert trend.extras["quarters_used"][0]["quarter"] == "2016-Q4"
    assert trend.extras["quarters_used"][-1]["quarter"] == "2018-Q3"


def test_sentiment_trend_excludes_the_partial_reference_quarter() -> None:
    """Reviews inside the reference quarter cannot move the slope."""
    quarter_dates = [
        "2016-10-15",
        "2017-01-15",
        "2017-04-15",
        "2017-07-15",
        "2017-10-15",
        "2018-01-15",
        "2018-04-15",
        "2018-07-15",
    ]
    times = _times([date for date in quarter_dates for _ in range(5)] + ["2018-10-04"] * 5)
    predicted = pd.Series([2.0] * 45)
    net = pd.Series(
        [0.1 * index for index in range(8) for _ in range(5)] + [-1.0] * 5
    )
    has_prediction = predicted.notna() & net.notna()
    _coverage, _staleness, _shift, trend = recency_indicators(
        times, predicted, net, has_prediction, REFERENCE, _indicators_config()
    )
    assert trend.raw_value == pytest.approx(0.1, rel=1e-9)
    assert len(trend.extras["quarters_used"]) == 8


def test_sentiment_trend_needs_enough_qualifying_quarters() -> None:
    """Two quarters with enough predictions are below the minimum of four."""
    times = _times(["2018-01-15"] * 5 + ["2018-04-15"] * 5)
    predicted = pd.Series([2.0] * 10)
    net = pd.Series([0.1] * 5 + [0.5] * 5)
    has_prediction = predicted.notna() & net.notna()
    _coverage, _staleness, _shift, trend = recency_indicators(
        times, predicted, net, has_prediction, REFERENCE, _indicators_config()
    )
    assert trend.normalized is None
    assert trend.missing_reason == "trend_insufficient_quarters"


# ------------------------------------------------------------------------------
# Temporal burst (sliding window)
# ------------------------------------------------------------------------------


def test_temporal_burst_joins_march_to_august() -> None:
    """A March-August run is one burst even though half-year buckets split it."""
    times = _times(
        [
            "2018-03-01",
            "2018-04-01",
            "2018-05-01",
            "2018-06-01",
            "2018-06-30",
            "2018-07-01",
            "2018-07-31",
            "2018-08-01",
            "2018-08-15",
            "2018-08-31",
        ]
    )
    indicator = burst_indicator(times, 10, _indicators_config())
    # Five fall in Jan-Jun and five in Jul-Dec, so no calendar half holds more
    # than half the negatives; the sliding window holds all ten.
    assert indicator.raw_value == pytest.approx(1.0)
    assert indicator.normalized == pytest.approx(0.0)
    assert indicator.extras["window_count"] == 10
    assert indicator.extras["window_start"] == "2018-03-01T00:00:00Z"
    assert indicator.extras["window_end"] == "2018-09-01T00:00:00Z"
    assert indicator.extras["flag"] is True
    assert indicator.direction == DIRECTION_COST


def test_temporal_burst_spread_out_negatives_score_low() -> None:
    """One negative per year is not a burst."""
    times = _times([f"{year}-06-01" for year in range(2005, 2019)])
    indicator = burst_indicator(times, len(times), _indicators_config())
    assert indicator.raw_value == pytest.approx(1 / 14)
    assert indicator.extras["flag"] is False
    assert indicator.normalized == pytest.approx(1.0 - 1 / 14)


def test_temporal_burst_tie_resolves_to_the_earliest_window() -> None:
    """Two equal clusters report the earlier window start."""
    times = _times(
        ["2005-01-01", "2005-01-10", "2005-02-01", "2005-02-10", "2005-03-01"]
        + ["2010-01-01", "2010-01-10", "2010-02-01", "2010-02-10", "2010-03-01"]
    )
    indicator = burst_indicator(times, 10, _indicators_config())
    assert indicator.raw_value == pytest.approx(0.5)
    assert indicator.extras["window_start"] == "2005-01-01T00:00:00Z"
    assert indicator.extras["flag"] is True  # 0.5 >= 0.50


def test_temporal_burst_below_the_negative_minimum_is_missing() -> None:
    """A single negative review cannot be a burst (minimum is ten)."""
    indicator = burst_indicator(_times(["2018-01-01"]), 1, _indicators_config())
    assert indicator.normalized is None
    assert indicator.missing_reason == "below_min_negative_reviews"
    assert indicator.extras["flag"] is False


# ------------------------------------------------------------------------------
# Negative-feedback concentration
# ------------------------------------------------------------------------------


def test_concentration_hand_computed() -> None:
    """HHI 0.8^2 + 0.2^2 = 0.68, and the top pain points cover every mention."""
    counts = pd.Series(
        [8.0, 2.0, 0.0], index=pd.Index(["battery", "sound", "price"], name="aspect")
    )
    classes = pd.Series(
        ["pain_point", "pain_point", "strength"], index=counts.index
    )
    concentration, top = concentration_indicators(
        counts, classes, 10, _indicators_config()
    )
    assert concentration.raw_value == pytest.approx(0.68)
    assert concentration.normalized == pytest.approx(0.32)
    assert concentration.extras["negative_mentions"] == 10
    assert concentration.direction == DIRECTION_COST
    assert top.raw_value == pytest.approx(1.0)
    assert top.normalized == pytest.approx(0.0)
    assert top.extras["top_pain_points"] == [
        {"aspect": "battery", "negative_mentions": 8},
        {"aspect": "sound", "negative_mentions": 2},
    ]


def test_concentration_top_pain_point_tie_breaks_alphabetically() -> None:
    """Equal negative counts order the pain points by aspect name ascending."""
    counts = pd.Series([3.0, 3.0, 1.0], index=pd.Index(["beta", "alpha", "gamma"]))
    classes = pd.Series(["pain_point"] * 3, index=counts.index)
    _concentration, top = concentration_indicators(
        counts, classes, 10, _indicators_config(top_pain_points=1)
    )
    assert top.extras["top_pain_points"] == [
        {"aspect": "alpha", "negative_mentions": 3}
    ]


def test_concentration_without_pain_points_scores_zero_share() -> None:
    """No pain-point aspect means a 0% top share (best), not a missing value."""
    counts = pd.Series([3.0, 1.0], index=pd.Index(["speed", "price"]))
    classes = pd.Series(["strength", "strength"], index=counts.index)
    _concentration, top = concentration_indicators(
        counts, classes, 10, _indicators_config()
    )
    assert top.raw_value == 0.0
    assert top.normalized == 1.0
    assert top.extras["top_pain_points"] == []


def test_concentration_below_the_negative_minimum_is_missing() -> None:
    """Fewer than ten predicted-negative reviews makes both inputs undefined."""
    counts = pd.Series([5.0], index=pd.Index(["battery"]))
    classes = pd.Series(["pain_point"], index=counts.index)
    concentration, top = concentration_indicators(
        counts, classes, 5, _indicators_config()
    )
    assert concentration.missing_reason == "below_min_negative_reviews"
    assert top.missing_reason == "below_min_negative_reviews"


def test_concentration_without_negative_mentions_is_missing() -> None:
    """Ten negatives but no aspect mention at all is a named missing case."""
    counts = pd.Series([0.0], index=pd.Index(["battery"]))
    classes = pd.Series(["pain_point"], index=counts.index)
    concentration, top = concentration_indicators(
        counts, classes, 10, _indicators_config()
    )
    assert concentration.missing_reason == "no_negative_mentions"
    assert top.missing_reason == "no_negative_mentions"


# ------------------------------------------------------------------------------
# Rating / text agreement
# ------------------------------------------------------------------------------


def test_rating_text_agreement_hand_computed() -> None:
    """One conflict in ten reviews is half of the 0.20 saturation point."""
    ratings = pd.Series([1.0] + [5.0] * 9)
    predicted = pd.Series([0.0] + [2.0] * 9)
    prob_positive = pd.Series([0.8] + [0.8] * 9)
    prob_negative = pd.Series([0.2] + [0.1] * 9)
    indicator = rating_text_agreement_indicator(
        ratings, predicted, prob_positive, prob_negative, _indicators_config()
    )
    assert indicator.raw_value == pytest.approx(0.1)
    assert indicator.normalized == pytest.approx(0.5)  # 1 − 0.1/0.2
    assert indicator.extras["conflicts"] == 1
    assert indicator.extras["evaluated"] == 10
    assert indicator.direction == DIRECTION_COST


def test_rating_text_agreement_conflict_on_the_other_side() -> None:
    """Five stars with a high negative probability is a conflict too."""
    ratings = pd.Series([5.0])
    predicted = pd.Series([2.0])
    prob_positive = pd.Series([0.05])
    prob_negative = pd.Series([0.9])
    indicator = rating_text_agreement_indicator(
        ratings, predicted, prob_positive, prob_negative, _indicators_config()
    )
    assert indicator.raw_value == pytest.approx(1.0)
    assert indicator.normalized == 0.0  # saturated


def test_rating_text_agreement_neutral_stars_never_conflict() -> None:
    """Three-star reviews are outside both conflict rules."""
    indicator = rating_text_agreement_indicator(
        pd.Series([3.0]),
        pd.Series([1.0]),
        pd.Series([0.99]),
        pd.Series([0.99]),
        _indicators_config(),
    )
    assert indicator.raw_value == 0.0
    assert indicator.normalized == 1.0


def test_rating_text_agreement_without_predictions_is_missing() -> None:
    """No paired rating+prediction means no conflict rate."""
    indicator = rating_text_agreement_indicator(
        pd.Series([4.0]),
        pd.Series([float("nan")]),
        pd.Series([float("nan")]),
        pd.Series([float("nan")]),
        _indicators_config(),
    )
    assert indicator.normalized is None
    assert indicator.missing_reason == "no_predictions"


# ------------------------------------------------------------------------------
# Indicator payload and composite
# ------------------------------------------------------------------------------


def test_indicator_payload_shape_and_rounding() -> None:
    """The stored payload has stable keys and rounds floats to six decimals."""
    indicator = Indicator(
        name="x",
        raw_value=1 / 3,
        normalized=2 / 3,
        direction=DIRECTION_BENEFIT,
        description="text",
        extras={"n": 3},
    )
    payload = indicator.to_payload()
    assert set(payload) == {
        "raw_value",
        "normalized",
        "direction",
        "description",
        "missing_reason",
        "extras",
    }
    assert payload["raw_value"] == pytest.approx(0.333333)
    assert payload["normalized"] == pytest.approx(0.666667)
    assert payload["missing_reason"] is None
    assert payload["extras"] == {"n": 3}


def test_composite_strength_renormalizes_around_missing_components() -> None:
    """A missing verified ratio drops out and the rest share its weight."""
    indicators = {
        "review_volume": _indicator("review_volume", 1.0),
        "recency_coverage": _indicator("recency_coverage", 0.5),
        "verified_purchase_ratio": _indicator(
            "verified_purchase_ratio", None, reason="verified_unknown"
        ),
        "rating_agreement": _indicator("rating_agreement", 0.5),
        "rating_text_agreement": _indicator("rating_text_agreement", 1.0),
        "helpfulness_coverage": _indicator("helpfulness_coverage", 0.0),
    }
    result = composite_strength(indicators, EvidenceStrengthConfig())
    expected = (0.2 * 1.0 + 0.2 * 0.5 + 0.15 * 0.5 + 0.15 * 1.0 + 0.15 * 0.0) / 0.85
    assert isinstance(result, StrengthResult)
    assert result.value == pytest.approx(expected)
    assert result.components_used == (
        "review_volume",
        "recency_coverage",
        "rating_agreement",
        "rating_text_consistency",
        "helpfulness_coverage",
    )
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert result.missing == {"verified_purchase_ratio": "verified_unknown"}


def test_composite_strength_with_all_components_available() -> None:
    """With nothing missing the weights are the configured ones."""
    indicators = {
        "review_volume": _indicator("review_volume", 0.8),
        "recency_coverage": _indicator("recency_coverage", 0.4),
        "verified_purchase_ratio": _indicator("verified_purchase_ratio", 1.0),
        "rating_agreement": _indicator("rating_agreement", 0.6),
        "rating_text_agreement": _indicator("rating_text_agreement", 0.2),
        "helpfulness_coverage": _indicator("helpfulness_coverage", 0.5),
    }
    result = composite_strength(indicators, EvidenceStrengthConfig())
    expected = (
        0.2 * 0.8 + 0.2 * 0.4 + 0.15 * 1.0 + 0.15 * 0.6 + 0.15 * 0.2 + 0.15 * 0.5
    )
    assert result.value == pytest.approx(expected)
    for name, value in EvidenceStrengthConfig().weights.model_dump().items():
        assert result.weights[name] == pytest.approx(value)
    assert result.missing == {}


def test_composite_strength_without_any_component_is_none() -> None:
    """When every component is missing the strength is None, never zero."""
    names = (
        "review_volume",
        "recency_coverage",
        "verified_purchase_ratio",
        "rating_agreement",
        "rating_text_agreement",
        "helpfulness_coverage",
    )
    indicators = {
        name: _indicator(name, None, reason="no_reviews") for name in names
    }
    result = composite_strength(indicators, EvidenceStrengthConfig())
    assert result.value is None
    assert result.components_used == ()
    assert result.weights == {}
    assert set(result.missing) == {
        "review_volume",
        "recency_coverage",
        "verified_purchase_ratio",
        "rating_agreement",
        "rating_text_consistency",
        "helpfulness_coverage",
    }

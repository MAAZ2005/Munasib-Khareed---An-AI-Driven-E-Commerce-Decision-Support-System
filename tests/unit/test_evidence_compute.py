"""Tests for the evidence assembly (``compute_evidence``) and its summary.

Every expectation is hand-built from synthetic frames or hand-written
``EvidenceRow`` objects; nothing here needs the real cache. The summary tests
verify the reported distributions/flags/correlations against values computed by
hand from the manual rows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from munasib.core.config import EvidenceConfig
from munasib.review_intel.evidence.compute import (
    JSON_FILENAME,
    MARKDOWN_FILENAME,
    SUMMARY_FLAGS,
    EvidenceRow,
    compute_evidence,
    render_summary_markdown,
    summarize,
    write_summary_reports,
)
from munasib.review_intel.evidence.indicators import STRENGTH_COMPONENT_KEYS

REFERENCE = datetime(2018, 10, 4)
COMPUTED_AT = datetime(2026, 10, 24, 6, 15, 6)

#: Indicator keys the hand-written summary rows carry (the composite's inputs).
_COMPONENT_INDICATORS = (
    "review_volume",
    "recency_coverage",
    "verified_purchase_ratio",
    "rating_agreement",
    "rating_text_agreement",
    "helpfulness_coverage",
)


def _review(
    asin: str,
    rating: float,
    when: str,
    *,
    votes: int = 0,
    verified: bool | None = True,
    predicted: float | None = 2.0,
    prob_negative: float | None = 0.05,
    prob_neutral: float | None = 0.05,
    prob_positive: float | None = 0.90,
) -> dict[str, object]:
    """Build one review row for the compute input frame.

    Args:
        asin: Product ASIN.
        rating: Star rating.
        when: ISO date of the review.
        votes: Helpful-vote count.
        verified: Verified flag (None = unknown).
        predicted: Predicted label code (None = no prediction).
        prob_negative: Predicted negative probability.
        prob_neutral: Predicted neutral probability.
        prob_positive: Predicted positive probability.

    Returns:
        Row mapping matching ``store.load_reviews`` column names.
    """
    return {
        "asin": asin,
        "rating": rating,
        "helpful_votes": votes,
        "verified": verified,
        "review_time": pd.Timestamp(when),
        "predicted_label": predicted,
        "prob_negative": prob_negative,
        "prob_neutral": prob_neutral,
        "prob_positive": prob_positive,
    }


def _reviews(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Convert review rows into a frame with a datetime column.

    Args:
        rows: Review row mappings.

    Returns:
        The review frame.
    """
    frame = pd.DataFrame(rows)
    frame["review_time"] = pd.to_datetime(frame["review_time"])
    return frame


def _products(*pairs: tuple[str, str]) -> pd.DataFrame:
    """Build the products frame from ``(asin, split)`` pairs.

    Args:
        *pairs: ASIN/split tuples.

    Returns:
        The products frame.
    """
    return pd.DataFrame({"asin": [asin for asin, _ in pairs], "split": [s for _, s in pairs]})


def _empty_aspects() -> pd.DataFrame:
    """Build an empty aspect frame with the expected columns.

    Returns:
        Empty frame.
    """
    return pd.DataFrame(
        {
            "asin": pd.Series(dtype="object"),
            "aspect": pd.Series(dtype="object"),
            "classification": pd.Series(dtype="object"),
            "negative_count": pd.Series(dtype="int64"),
        }
    )


def _mixed_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build four products covering the interesting evidence shapes.

    Returns:
        ``(products, reviews, aspects)`` frames: GOOD (60 recent, voted,
        predicted positive), OLD (12 one-star reviews from 2005), POLAR (20
        recent 1/5-star reviews with unknown verified flags and some missing
        predictions) and ZERO (no reviews at all).
    """
    rows: list[dict[str, object]] = []
    # GOOD: 60 recent reviews, votes on every third, all predicted positive.
    for index in range(60):
        rows.append(
            _review(
                "GOOD000001",
                5 if index % 5 else 4,
                f"2018-{1 + index % 9:02d}-{1 + index % 27:02d}",
                votes=index % 3,
            )
        )
    # OLD: 12 reviews in 2005, all one star and predicted negative.
    for index in range(12):
        rows.append(
            _review(
                "OLD0000001",
                1,
                f"2005-{1 + index % 12:02d}-01",
                predicted=0.0,
                prob_negative=0.9,
                prob_neutral=0.05,
                prob_positive=0.05,
            )
        )
    # POLAR: 20 recent reviews split 1/5 stars, one quarter missing predictions,
    # with every verified flag unknown.
    for index in range(20):
        rows.append(
            _review(
                "POLAR00001",
                1 if index % 2 else 5,
                f"2018-{1 + index % 9:02d}-15",
                verified=None,
                predicted=None if index % 4 == 0 else (0.0 if index % 2 else 2.0),
                prob_negative=None if index % 4 == 0 else (0.9 if index % 2 else 0.05),
                prob_neutral=None if index % 4 == 0 else 0.05,
                prob_positive=None if index % 4 == 0 else (0.05 if index % 2 else 0.9),
            )
        )
    aspects = pd.DataFrame(
        {
            "asin": ["GOOD000001", "GOOD000001", "OLD0000001"],
            "aspect": ["battery", "price", "battery"],
            "classification": ["pain_point", "strength", "pain_point"],
            "negative_count": [1, 0, 12],
        }
    )
    products = _products(
        ("GOOD000001", "train"),
        ("OLD0000001", "val"),
        ("POLAR00001", "test"),
        ("ZERO000001", "train"),
    )
    return products, _reviews(rows), aspects


def _compute() -> list[EvidenceRow]:
    """Run compute_evidence over the mixed synthetic inputs.

    Returns:
        The computed rows.
    """
    products, reviews, aspects = _mixed_inputs()
    return compute_evidence(
        products,
        reviews,
        aspects,
        reference_date=REFERENCE,
        computed_at=COMPUTED_AT,
        config=EvidenceConfig(),
    )


# ------------------------------------------------------------------------------
# compute_evidence
# ------------------------------------------------------------------------------


def test_compute_evidence_is_deterministic() -> None:
    """Two runs over identical frames produce identical payloads."""
    first = [row.to_payload() for row in _compute()]
    second = [row.to_payload() for row in _compute()]
    assert first == second
    # The stored JSON column (``indicators``) must be serialisable exactly as stored.
    json.dumps([row.indicators for row in _compute()])


def test_compute_evidence_orders_products_by_asin() -> None:
    """Rows come back in ascending ASIN order, independent of frame order."""
    rows = _compute()
    assert [row.asin for row in rows] == sorted(row.asin for row in rows)


def test_payload_structure_directions_and_flags() -> None:
    """Every indicator entry carries the stored contract and consistent flags."""
    for row in _compute():
        payload = row.indicators
        assert set(payload) == {
            "split",
            "indicators",
            "flags",
            "polarized",
            "no_recent_reviews",
            "temporal_burst",
            "low_evidence",
            "strength",
        }
        assert payload["split"] == row.split
        indicators = payload["indicators"]
        assert len(indicators) == 14
        for name, entry in indicators.items():
            assert set(entry) == {
                "raw_value",
                "normalized",
                "direction",
                "description",
                "missing_reason",
                "extras",
            }
            assert entry["direction"] in ("benefit", "cost"), name
            assert isinstance(entry["description"], str) and entry["description"]
            if entry["missing_reason"] is not None:
                assert entry["normalized"] is None
        expected_flags = {
            name: entry["missing_reason"]
            for name, entry in indicators.items()
            if entry["missing_reason"] is not None
        }
        assert payload["flags"] == expected_flags
        for name in SUMMARY_FLAGS:
            assert isinstance(payload[name], bool)


def test_all_normalized_values_stay_in_range() -> None:
    """Every stored normalized value is inside [0, 1]."""
    for row in _compute():
        for name, entry in row.indicators["indicators"].items():
            value = entry["normalized"]
            if value is not None:
                assert 0.0 <= value <= 1.0, f"{row.asin}:{name}={value}"


def test_three_review_product_is_low_evidence() -> None:
    """Three recent five-star reviews compute a row but stay low evidence."""
    rows = compute_evidence(
        _products(("TINY000001", "train")),
        _reviews([_review("TINY000001", 5, "2018-06-01") for _ in range(3)]),
        _empty_aspects(),
        reference_date=REFERENCE,
        computed_at=COMPUTED_AT,
        config=EvidenceConfig(),
    )
    row = rows[0]
    assert row.n_reviews == 3
    assert row.evidence_strength is not None
    payload = row.indicators
    assert payload["low_evidence"] is True
    volume = payload["indicators"]["review_volume"]
    # min(1, ln(4)/ln(201)) = 0.2614...
    assert volume["normalized"] == pytest.approx(0.2614, abs=5e-5)
    assert payload["indicators"]["review_volume"]["raw_value"] == 3.0


def test_product_without_reviews_has_no_strength() -> None:
    """A product with no stored reviews has every indicator missing and NULL strength."""
    rows = compute_evidence(
        _products(("ZERO000001", "train")),
        _reviews([_review("OTHER00001", 5, "2018-06-01")]),
        _empty_aspects(),
        reference_date=REFERENCE,
        computed_at=COMPUTED_AT,
        config=EvidenceConfig(),
    )
    row = rows[0]
    assert row.n_reviews == 0
    assert row.evidence_strength is None
    payload = row.indicators
    assert payload["low_evidence"] is True
    assert payload["no_recent_reviews"] is True
    assert payload["strength"]["value"] is None
    assert payload["strength"]["components_used"] == []
    for name, entry in payload["indicators"].items():
        assert entry["normalized"] is None, name
        assert entry["missing_reason"] is not None, name
    assert payload["flags"]["review_volume"] == "no_reviews"
    assert payload["flags"]["rating_agreement"] == "no_ratings"
    assert payload["flags"]["temporal_burst"] == "below_min_negative_reviews"


def test_missing_predictions_are_counted_not_imputed() -> None:
    """Prediction-based indicators only use predicted reviews and say so."""
    row = {r.asin: r for r in _compute()}["POLAR00001"]
    payload = row.indicators
    shares = payload["indicators"]["sentiment_shares"]
    assert row.n_reviews == 20
    assert shares["extras"]["predictions"] == 15  # 5 of 20 lack a prediction
    assert "sentiment_shares" not in payload["flags"]
    # The star-derived negative share still covers every rated review.
    assert shares["extras"]["star_negative_share"] == pytest.approx(0.5)
    assert payload["polarized"] is True


def test_verified_unknown_drops_the_component_and_renormalizes() -> None:
    """All-NULL verified flags drop the component; the weights renormalize."""
    row = {r.asin: r for r in _compute()}["POLAR00001"]
    strength = row.indicators["strength"]
    assert "verified_purchase_ratio" not in strength["components_used"]
    assert strength["missing_components"]["verified_purchase_ratio"] == "verified_unknown"
    assert set(strength["weights"]) == set(strength["components_used"])
    assert sum(strength["weights"].values()) == pytest.approx(1.0, abs=1e-4)
    assert row.evidence_strength is not None and 0.0 < row.evidence_strength < 1.0


def test_old_product_has_no_recent_reviews() -> None:
    """A 2005-only history has zero coverage and is flagged accordingly."""
    row = {r.asin: r for r in _compute()}["OLD0000001"]
    payload = row.indicators
    assert payload["no_recent_reviews"] is True
    assert payload["indicators"]["recency_coverage"]["normalized"] == 0.0
    assert payload["indicators"]["staleness"]["raw_value"] > 120  # months
    assert payload["flags"]["recent_negative_shift"] == "recent_window_below_min_reviews"


def test_stored_strength_equals_recomputed_component_mean() -> None:
    """The stored composite matches the weighted mean of its own components."""
    for row in _compute():
        strength = row.indicators["strength"]
        if strength["value"] is None:
            continue
        indicators = row.indicators["indicators"]
        expected = 0.0
        for component in strength["components_used"]:
            indicator_name = (
                "rating_text_agreement"
                if component == "rating_text_consistency"
                else component
            )
            expected += strength["weights"][component] * indicators[indicator_name][
                "normalized"
            ]
        assert row.evidence_strength == pytest.approx(expected, abs=1e-5)


def test_rating_text_consistency_uses_the_agreement_indicator() -> None:
    """The stored components map 1:1 onto the documented indicators."""
    payload = _compute()[0].indicators
    assert set(payload["strength"]["components_used"]) <= set(STRENGTH_COMPONENT_KEYS)


def test_compute_evidence_rejects_timezone_aware_timestamps() -> None:
    """The storage contract is naive UTC; aware timestamps fail loudly."""
    products, reviews, aspects = _mixed_inputs()
    with pytest.raises(ValueError, match="naive UTC"):
        compute_evidence(
            products,
            reviews,
            aspects,
            reference_date=datetime(2018, 10, 4, tzinfo=UTC),
            computed_at=COMPUTED_AT,
            config=EvidenceConfig(),
        )


def test_compute_evidence_validates_the_products_frame() -> None:
    """A products frame without asin/split is a caller bug."""
    _products_frame, reviews, aspects = _mixed_inputs()
    with pytest.raises(ValueError, match="split"):
        compute_evidence(
            pd.DataFrame({"asin": ["A"]}),
            reviews,
            aspects,
            reference_date=REFERENCE,
            computed_at=COMPUTED_AT,
            config=EvidenceConfig(),
        )


# ------------------------------------------------------------------------------
# summarize (hand-written rows)
# ------------------------------------------------------------------------------


def _manual_row(
    asin: str,
    split: str,
    strength: float | None,
    values: dict[str, float | None],
    *,
    n_reviews: int = 100,
    reasons: dict[str, str] | None = None,
    flags: dict[str, bool] | None = None,
) -> EvidenceRow:
    """Build a hand-written evidence row for summary tests.

    Args:
        asin: Product ASIN.
        split: Product split.
        strength: Composite value.
        values: Normalized value per component indicator.
        n_reviews: Review count.
        reasons: Missing reason per indicator.
        flags: Values of the summary booleans.

    Returns:
        The row.
    """
    reasons = reasons or {}
    flags = flags or {}
    indicators = {
        name: {
            "raw_value": values.get(name),
            "normalized": values.get(name),
            "direction": "benefit",
            "description": "manual",
            "missing_reason": reasons.get(name),
            "extras": {},
        }
        for name in _COMPONENT_INDICATORS
    }
    payload = {
        "split": split,
        "indicators": indicators,
        "flags": dict(reasons),
        "polarized": flags.get("polarized", False),
        "low_evidence": flags.get("low_evidence", False),
        "no_recent_reviews": flags.get("no_recent_reviews", False),
        "temporal_burst": flags.get("temporal_burst", False),
        "strength": {},
    }
    return EvidenceRow(
        asin=asin,
        split=split,
        n_reviews=n_reviews,
        evidence_strength=strength,
        indicators=payload,
        reference_date=REFERENCE,
        computed_at=COMPUTED_AT,
    )


def _three_manual_rows() -> list[EvidenceRow]:
    """Build three rows with rank-perfect and reversed signals.

    Returns:
        The rows (strength and recency_coverage both increase; verified
        decreases; n_reviews is constant).
    """
    common = {
        "review_volume": 0.5,
        "rating_agreement": 0.5,
        "rating_text_agreement": 0.5,
        "helpfulness_coverage": 0.5,
    }
    return [
        _manual_row(
            "A000000001",
            "train",
            0.1,
            {**common, "recency_coverage": 0.0, "verified_purchase_ratio": 1.0},
            flags={"polarized": True},
        ),
        _manual_row(
            "A000000002",
            "train",
            0.5,
            {**common, "recency_coverage": 0.5, "verified_purchase_ratio": 0.5},
        ),
        _manual_row(
            "A000000003",
            "test",
            0.9,
            {**common, "recency_coverage": 1.0, "verified_purchase_ratio": 0.0,
             "rating_agreement": None},
            reasons={"rating_agreement": "no_ratings"},
            flags={"low_evidence": True, "no_recent_reviews": True},
        ),
    ]


def test_summarize_distributions_and_flags() -> None:
    """Distributions carry hand-checked min/median/max and flags are counted."""
    summary = summarize(_three_manual_rows())
    assert summary.products == 3
    recency = summary.distributions["recency_coverage"]
    assert recency == {"count": 3, "min": 0.0, "median": 0.5, "max": 1.0}
    volume = summary.distributions["review_volume"]
    assert volume == {"count": 3, "min": 0.5, "median": 0.5, "max": 0.5}
    assert summary.flags == {
        "polarized": 1,
        "low_evidence": 1,
        "no_recent_reviews": 1,
        "temporal_burst": 0,
    }


def test_summarize_per_split_and_missing_reasons() -> None:
    """Per-split medians and the missing reasons are aggregated per indicator."""
    summary = summarize(_three_manual_rows())
    assert summary.per_split == {
        "train": {"products": 2, "strength_median": 0.3},
        "test": {"products": 1, "strength_median": 0.9},
    }
    assert summary.missing == {
        "rating_agreement": {"count": 1, "reasons": {"no_ratings": 1}}
    }


def test_summarize_correlations_are_hand_checked() -> None:
    """Spearman values follow the ranks built into the manual rows."""
    summary = summarize(_three_manual_rows())
    correlations = summary.correlations
    assert set(correlations) == {*STRENGTH_COMPONENT_KEYS, "n_reviews"}
    assert correlations["recency_coverage"]["spearman"] == pytest.approx(1.0)
    assert correlations["recency_coverage"]["n"] == 3
    assert correlations["verified_purchase_ratio"]["spearman"] == pytest.approx(-1.0)
    # Constant signals and the missing third row give no defined correlation.
    assert correlations["review_volume"]["spearman"] is None
    assert correlations["rating_agreement"]["spearman"] is None
    assert correlations["rating_agreement"]["n"] == 2
    assert correlations["n_reviews"]["spearman"] is None


def test_summarize_rejects_an_empty_run() -> None:
    """Summarizing nothing is a caller bug, not an empty report."""
    with pytest.raises(ValueError, match="empty"):
        summarize([])


def test_summary_lines_and_payload_are_json_safe() -> None:
    """The printed lines and the JSON payload expose the same figures."""
    summary = summarize(_three_manual_rows())
    lines = summary.lines()
    assert any("products written" in line for line in lines)
    assert any("2018-10-04" in line for line in lines)
    assert any("spearman recency_coverage" in line for line in lines)
    payload = summary.to_payload()
    json.dumps(payload)
    assert set(payload) == {
        "products",
        "reference_date",
        "computed_at",
        "distributions",
        "flags",
        "per_split",
        "missing",
        "correlations",
    }


def test_write_summary_reports_is_deterministic(tmp_path: Path) -> None:
    """Re-running the report writer produces byte-identical files."""
    summary = summarize(_three_manual_rows())
    first = write_summary_reports(summary, tmp_path / "r1")
    second = write_summary_reports(summary, tmp_path / "r2")
    assert [path.name for path in first] == [JSON_FILENAME, MARKDOWN_FILENAME]
    assert first[0].read_bytes() == second[0].read_bytes()
    assert first[1].read_bytes() == second[1].read_bytes()
    payload = json.loads(first[0].read_text(encoding="utf-8"))
    assert payload["reference_date"] == "2018-10-04T00:00:00+00:00"
    markdown = first[1].read_text(encoding="utf-8")
    assert "# Evidence summary" in markdown
    assert "2018-10-04" in markdown


def test_render_summary_markdown_lists_every_section() -> None:
    """The Markdown report carries distributions, flags, splits and correlations."""
    markdown = render_summary_markdown(summarize(_three_manual_rows()))
    for heading in (
        "## Distributions",
        "## Flags",
        "## Per split",
        "## Missing values",
        "## Correlation with evidence_strength (Spearman)",
    ):
        assert heading in markdown
    assert "`no_ratings`" in markdown

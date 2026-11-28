"""Tests for the risk assembly, the agreement-scale rule and the run summary.

Synthetic frames only: no cache, no network, no clock. Every expected value is
derived by hand from the formulas in ``docs/risk_indicators.md``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from munasib.core.config import RiskConfig
from munasib.core.exceptions import DataError
from munasib.risk.compute import (
    CRITERION_ORDER,
    JSON_FILENAME,
    MARKDOWN_FILENAME,
    agreement_component,
    choose_agreement_scale,
    compute_risk_indicators,
    render_summary_markdown,
    summarize,
    write_summary_reports,
)
from munasib.risk.factors import FACTOR_CODES
from munasib.risk.indicators import (
    product_sentiment_totals,
    read_criteria_directions,
)

CONFIG = RiskConfig()
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "criteria_weights.yaml"
DIRECTIONS = read_criteria_directions(CONFIG_PATH)
REFERENCE = datetime(2018, 10, 4)
COMPUTED_AT = datetime(2026, 10, 24, 6, 15, 6)

PRODUCTS = [
    ("T1", "train", "Train one", 4.5),
    ("T2", "train", "Train two", 4.0),
    ("T3", "train", "Train three", 3.5),
    ("V1", "val", "Validation one", 4.2),
    ("S1", "test", "Test one", 4.1),
]


def _entry(
    raw: float | None = None,
    normalized: float | None = None,
    extras: dict | None = None,
) -> dict:
    """Build one stored indicator entry.

    Args:
        raw: Raw value.
        normalized: Normalized value.
        extras: Extra payload.

    Returns:
        Indicator mapping.
    """
    return {
        "raw_value": raw,
        "normalized": normalized,
        "direction": "benefit",
        "description": "",
        "missing_reason": None,
        "extras": extras or {},
    }


def _payload(strength: float = 0.7, *, polarized: bool = False) -> dict:
    """Build a complete stored evidence payload.

    Args:
        strength: Composite evidence strength.
        polarized: Stored polarization flag.

    Returns:
        Evidence payload mapping.
    """
    return {
        "split": "train",
        "indicators": {
            "rating_agreement": _entry(0.5, 0.5, {"polarized": polarized}),
            "rating_text_agreement": _entry(
                None, 0.8, {"conflicts": 2, "evaluated": 20}
            ),
            "helpfulness_weighted_sentiment": _entry(
                None, None, {"difference": 0.02}
            ),
            "recent_negative_shift": _entry(
                0.05,
                0.95,
                {
                    "recent_negative_share": 0.3,
                    "overall_negative_share": 0.25,
                    "recent_predictions": 12,
                },
            ),
            "top_pain_point_share": _entry(
                0.4,
                0.6,
                {
                    "top_pain_points": [{"aspect": "battery", "negative_mentions": 2}],
                    "negative_mentions": 5,
                },
            ),
            "aspect_concentration": _entry(0.4, 0.6),
            "temporal_burst": _entry(0.4, 0.6, {"flag": False}),
            "staleness": _entry(10.0, 0.8, {"last_review_time": "2017-12-01"}),
            "verified_purchase_ratio": _entry(
                0.9, 0.9, {"verified_known": 100, "verified_count": 90}
            ),
        },
        "polarized": polarized,
        "temporal_burst": False,
        "low_evidence": False,
        "strength": {"value": strength},
    }


def _products() -> pd.DataFrame:
    """Build the product frame.

    Returns:
        Frame with the columns the stage requires.
    """
    return pd.DataFrame(
        [
            {
                "asin": asin,
                "split": split,
                "title": title,
                "mean_rating": rating,
                "n_reviews": 100,
            }
            for asin, split, title, rating in PRODUCTS
        ]
    )


def _predictions() -> pd.DataFrame:
    """Build four predictions per product (3 positive, 1 negative).

    Returns:
        Frame with the stored prediction columns.
    """
    rows = []
    for asin, *_ in PRODUCTS:
        for index in range(4):
            label = 0.0 if index == 0 else 2.0
            rows.append(
                {
                    "asin": asin,
                    "predicted_label": label,
                    "prob_negative": 0.8 if label == 0 else 0.1,
                    "prob_positive": 0.1 if label == 0 else 0.8,
                }
            )
    return pd.DataFrame(rows)


def _sentiment() -> pd.DataFrame:
    """Per-product sentiment totals.

    Returns:
        Frame with ``asin``, ``n_predictions``, ``mean_net``, ``negative_share``.
    """
    return product_sentiment_totals(_predictions())


def _evidence(*, exclude: str | None = None, strength: float = 0.7) -> pd.DataFrame:
    """Build the evidence frame.

    Args:
        exclude: ASIN to leave without an evidence row.
        strength: Composite evidence strength.

    Returns:
        Frame with ``asin``, ``evidence_strength`` and ``indicators``.
    """
    rows = []
    for asin, *_ in PRODUCTS:
        if asin == exclude:
            continue
        rows.append(
            {
                "asin": asin,
                "evidence_strength": strength,
                "indicators": _payload(strength),
            }
        )
    return pd.DataFrame(rows)


def _features(*, percentile: float | None = 0.5) -> pd.DataFrame:
    """Build the product-features frame.

    Args:
        percentile: Price percentile, or None for a missing price.

    Returns:
        Frame with ``asin`` and ``indicators``.
    """
    indicators = {
        "price_percentile_category": percentile,
        "price_missing": percentile is None,
        "price_peer_group": "Home & Kitchen > Appliances",
        "price_peer_count": 40,
        "price_peer_fallback": False,
    }
    return pd.DataFrame(
        [{"asin": asin, "indicators": dict(indicators)} for asin, *_ in PRODUCTS]
    )


def _compute(
    *,
    products: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
    evidence: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    targets: list[str] | None = None,
):
    """Run the stage with the synthetic frames.

    Args:
        products: Optional product frame override.
        sentiment: Optional sentiment override.
        evidence: Optional evidence override.
        features: Optional features override.
        targets: Optional ASIN subset.

    Returns:
        The ``RiskResult``.
    """
    return compute_risk_indicators(
        _products() if products is None else products,
        _sentiment() if sentiment is None else sentiment,
        _evidence() if evidence is None else evidence,
        _features() if features is None else features,
        reference_date=REFERENCE,
        computed_at=COMPUTED_AT,
        config=CONFIG,
        directions=DIRECTIONS,
        target_asins=targets,
    )


def test_compute_hand_computed_criteria() -> None:
    """Every criterion of T1 matches the hand-derived formula."""
    result = _compute()
    row = next(row for row in result.rows if row.asin == "T1")
    quality = row.criteria["product_quality"]
    # rating: (100*4.5 + 20*4.0)/120 = 4.4166667 -> (4.4166667-1)/4
    # net: 3 positive (+0.7) and 1 negative (-0.7) -> 0.35; prior 0.35
    # -> shrunk 0.35 -> (0.35+1)/2 = 0.675; recent 0.95
    expected_quality = (
        0.5 * ((4.5 * 100 + 4.0 * 20) / 120 - 1) / 4
        + 0.3 * 0.675
        + 0.2 * 0.95
    )
    assert quality.value == pytest.approx(expected_quality)
    assert quality.value == pytest.approx(0.819583, abs=1e-6)
    assert quality.confidence == pytest.approx(1.0)

    reliability = row.criteria["rating_reliability"]
    # agreement 0.5, non-polarized 1.0, consistency 0.8, helpfulness gap 0.02
    # at the chosen scale 0.10 -> 1 - 0.02/0.10 = 0.8.
    expected_reliability = 0.35 * 0.5 + 0.20 * 1.0 + 0.25 * 0.8 + 0.20 * 0.8
    assert reliability.value == pytest.approx(expected_reliability)

    concentration = row.criteria["negative_feedback_concentration"]
    # shrunk = 0.25 (own == prior), prevalence = 0.25/0.4 = 0.625, focus = 0.4
    assert concentration.value == pytest.approx(0.625 * (0.5 + 0.5 * 0.4))

    assert row.criteria["price_position"].value == pytest.approx(0.5)
    assert row.criteria["evidence_strength"].value == pytest.approx(0.7)
    assert row.split == "train"
    assert row.title == "Train one"


def test_compute_is_deterministic() -> None:
    """Two runs over the same frames produce identical payloads."""
    first = _compute()
    second = _compute()
    first_payloads = [
        json.dumps(row.to_payload(), sort_keys=True, default=str) for row in first.rows
    ]
    second_payloads = [
        json.dumps(row.to_payload(), sort_keys=True, default=str) for row in second.rows
    ]
    assert first_payloads == second_payloads
    assert first.agreement == second.agreement


def test_compute_priors_use_the_full_train_split_under_a_limit() -> None:
    """A limited run writes one row but keeps the full-train priors."""
    limited = _compute(targets=["V1"])
    full = _compute()
    assert [row.asin for row in limited.rows] == ["V1"]
    assert limited.priors == full.priors
    assert limited.priors["n_train_products"] == 3


def test_compute_rejects_an_unknown_target() -> None:
    """A typo in ``--asins`` cannot silently shrink the run."""
    with pytest.raises(DataError, match="not stored products"):
        _compute(targets=["NOPE"])


def test_compute_missing_price_is_none_not_zero() -> None:
    """A product without a price gets a null criterion, never 0 or 0.5."""
    result = _compute(features=_features(percentile=None))
    for row in result.rows:
        price = row.criteria["price_position"]
        assert price.value is None
        assert price.missing_reason == "price_missing"
        assert price.confidence == 0.0
        assert row.to_payload()["indicators"]["vector"][4] is None


def test_compute_missing_evidence_is_null_and_flagged() -> None:
    """No evidence row -> null strength, a high weak-evidence factor, no fabrication."""
    result = _compute(evidence=_evidence(exclude="S1"))
    row = next(row for row in result.rows if row.asin == "S1")
    assert row.criteria["evidence_strength"].value is None
    assert row.criteria["evidence_strength"].missing_reason == "no_evidence_row"
    assert row.criteria["rating_reliability"].value is None
    assert row.criteria["rating_reliability"].missing_reason == "all_components_missing"
    factors = {factor.code: factor for factor in row.risk_factors}
    assert factors["weak_evidence"].severity == "high"
    # The rating and sentiment components still exist, so quality is not null.
    assert row.criteria["product_quality"].value is not None
    assert row.criteria["product_quality"].confidence == pytest.approx(0.80)


def test_compute_uses_no_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patching ``datetime.now`` and ``time.time`` to raise cannot break a run."""
    class _NoClock:
        """Stand-in for the datetime class whose ``now`` is forbidden."""

        @classmethod
        def now(cls, tz=None):  # noqa: ANN001, ANN206 - test double
            raise AssertionError("the risk stage must not read the wall clock")

        @classmethod
        def utcnow(cls):
            raise AssertionError("the risk stage must not read the wall clock")

    def _no_time() -> float:
        raise AssertionError("the risk stage must not read the wall clock")

    monkeypatch.setattr("munasib.risk.compute.datetime", _NoClock)
    monkeypatch.setattr(time, "time", _no_time)
    result = _compute()
    assert len(result.rows) == len(PRODUCTS)


def test_agreement_component_saturates() -> None:
    """The component is 1 at gap 0 and exactly 0 at the scale."""
    assert agreement_component(0.0, 0.2) == pytest.approx(1.0)
    assert agreement_component(0.1, 0.2) == pytest.approx(0.5)
    assert agreement_component(0.2, 0.2) == 0.0
    assert agreement_component(-0.5, 0.2) == 0.0
    with pytest.raises(ValueError, match="positive"):
        agreement_component(0.1, 0.0)


def test_choose_agreement_scale_moves_past_a_saturated_candidate() -> None:
    """31.7%-style saturation at 0.10 moves to 0.20, exactly as documented."""
    gaps = [0.01] * 7 + [0.15] * 3
    choice = choose_agreement_scale(
        gaps, configured=0.10, candidates=[0.10, 0.20, 0.30], max_zero_share=0.25
    )
    assert choice.scale == pytest.approx(0.20)
    assert choice.zero_share == pytest.approx(0.0)
    assert choice.zero_share_by_candidate[0.10] == pytest.approx(0.30)
    assert choice.meets_target is True
    assert choice.median_component == pytest.approx(0.95)


def test_choose_agreement_scale_keeps_the_configured_scale_when_safe() -> None:
    """Below the cap the configured scale is used unchanged."""
    choice = choose_agreement_scale(
        [0.01] * 9 + [0.2],
        configured=0.10,
        candidates=[0.10, 0.20, 0.30],
        max_zero_share=0.25,
    )
    assert choice.scale == pytest.approx(0.10)
    assert choice.zero_share == pytest.approx(0.10)


def test_choose_agreement_scale_falls_back_to_the_largest() -> None:
    """When no candidate meets the cap, the largest is used and reported."""
    choice = choose_agreement_scale(
        [0.5, 0.6],
        configured=0.10,
        candidates=[0.10, 0.20, 0.30],
        max_zero_share=0.25,
    )
    assert choice.scale == pytest.approx(0.30)
    assert choice.meets_target is False
    assert choice.zero_share == pytest.approx(1.0)


def test_choose_agreement_scale_handles_no_gaps() -> None:
    """No usable gap keeps the configured scale with a zero share."""
    choice = choose_agreement_scale(
        [], configured=0.10, candidates=[0.10, 0.20, 0.30], max_zero_share=0.25
    )
    assert choice.scale == pytest.approx(0.10)
    assert choice.gap_count == 0
    assert choice.zero_share == 0.0
    assert choice.median_component is None


def test_summarize_reports_every_required_section() -> None:
    """The summary carries stats, missing reasons, raw/normalized inputs and matrices."""
    result = _compute()
    summary = summarize(
        result.rows, _evidence(), result.agreement, result.priors
    )
    assert summary.products == len(PRODUCTS)
    assert summary.priors["mean_rating"] == pytest.approx(4.0)
    for name in CRITERION_ORDER:
        assert summary.criteria_stats[name].count == len(PRODUCTS)
    assert summary.missing["price_position"] == {}
    assert set(summary.cost_raw) >= {
        "negative_feedback_concentration",
        "price_position",
        "top_pain_point_share",
        "aspect_concentration",
        "temporal_burst",
    }
    assert "normalized (1 - raw)" in summary.cost_raw["top_pain_point_share"]
    assert "raw (HHI)" in summary.cost_raw["aspect_concentration"]
    assert "normalized (1 - value)" in summary.cost_raw[
        "negative_feedback_concentration"
    ]
    assert summary.cost_raw["negative_feedback_concentration"][
        "normalized (1 - value)"
    ].median == pytest.approx(0.5625)
    assert summary.cost_raw["top_pain_point_share"]["raw"].median == pytest.approx(0.4)
    assert summary.correlations["product_quality"]["product_quality"] == 1.0
    assert summary.train_correlations["product_quality"]["product_quality"] == 1.0
    assert set(summary.split_medians) == {"train", "val", "test"}
    assert sum(summary.factor_counts.values()) == sum(
        len(row.risk_factors) for row in result.rows
    )


def test_summarize_best_and_worst_are_ordered_by_quality() -> None:
    """Best/worst lists are sorted by product_quality and carry display fields."""
    result = _compute()
    summary = summarize(result.rows, _evidence(), result.agreement, result.priors)
    assert summary.best[0]["asin"] == "T1"
    assert summary.worst[0]["asin"] == "T3"
    assert summary.best[0]["title"] == "Train one"
    assert summary.best[0]["n_reviews"] == 100
    assert summary.best[0]["net_sentiment"] == pytest.approx(0.35)


def test_summary_lines_keep_raw_and_normalized_apart() -> None:
    """The printed report labels raw and normalized statistics explicitly."""
    result = _compute()
    summary = summarize(result.rows, _evidence(), result.agreement, result.priors)
    text = "\n".join(summary.lines())
    assert "value (cost)" in text
    assert "normalized (1 - raw)" in text
    assert "normalized (1 - value)" in text
    assert "raw (HHI)" in text
    assert "spearman correlations (all products" in text
    assert "spearman correlations (train split only" in text
    assert "best 5 by product_quality" in text
    assert "worst 5 by product_quality" in text
    assert "risk factors (severity counts" in text


def test_summary_flags_dead_factors_on_a_tiny_run() -> None:
    """With five products every untouched rule is below the 0.5% floor."""
    result = _compute()
    summary = summarize(result.rows, _evidence(), result.agreement, result.priors)
    assert any("<0.5%" in note for note in summary.notes)


def test_summary_payload_is_json_safe_and_complete() -> None:
    """The JSON payload carries every section and survives a JSON round-trip."""
    result = _compute()
    summary = summarize(result.rows, _evidence(), result.agreement, result.priors)
    payload = summary.to_payload()

    assert payload["products"] == len(PRODUCTS)
    assert payload["reference_date"] == "2018-10-04T00:00:00+00:00"
    assert payload["computed_at"] == "2026-10-24T06:15:06+00:00"
    assert payload["priors"]["n_train_products"] == 3
    assert payload["agreement"]["scale"] == summary.agreement.scale
    assert payload["agreement"]["zero_share_by_candidate"]
    assert set(payload["criteria"]) == set(CRITERION_ORDER)
    assert set(payload["cost_indicators"]) == set(summary.cost_raw)
    assert set(payload["missing"]) == set(CRITERION_ORDER)
    assert set(payload["correlations"]) == set(CRITERION_ORDER)
    assert set(payload["train_correlations"]) == set(CRITERION_ORDER)
    assert set(payload["split_medians"]) == {"train", "val", "test"}
    assert set(payload["factors"]) == set(FACTOR_CODES)
    assert len(payload["best"]) == len(PRODUCTS)
    assert len(payload["worst"]) == len(PRODUCTS)
    assert json.loads(json.dumps(payload)) == payload


def test_summary_markdown_renders_every_section() -> None:
    """The Markdown report mirrors the JSON sections and never shows elapsed time."""
    result = _compute()
    summary = summarize(result.rows, _evidence(), result.agreement, result.priors)
    text = render_summary_markdown(summary)

    for heading in (
        "# Risk indicators summary",
        "## Category priors (train split)",
        "## Agreement scale",
        "## Criterion values",
        "## Cost-direction inputs (raw and normalized)",
        "## Missing values (None, never imputed)",
        "## Spearman correlations (all products, pairwise complete)",
        "## Spearman correlations (train split only, pairwise complete)",
        "## Median criterion value by split",
        "## Risk factors",
        "## Best 5 by product_quality",
        "## Worst 5 by product_quality",
    ):
        assert heading in text
    assert "elapsed" not in text
    assert "2018-10-04" in text


def test_write_summary_reports_writes_both_files(tmp_path: Path) -> None:
    """The writer creates both reports and reproduces them byte for byte."""
    result = _compute()
    summary = summarize(result.rows, _evidence(), result.agreement, result.priors)

    paths = write_summary_reports(summary, tmp_path)

    assert [path.name for path in paths] == [JSON_FILENAME, MARKDOWN_FILENAME]
    first = [path.read_bytes() for path in paths]
    write_summary_reports(summary, tmp_path)
    assert [path.read_bytes() for path in paths] == first

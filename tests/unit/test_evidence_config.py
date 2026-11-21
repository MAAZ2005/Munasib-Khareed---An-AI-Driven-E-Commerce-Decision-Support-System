"""Tests for the ``review_intel.evidence`` indicator and strength configuration.

Covers the documented defaults, the shipped config file declaring every key,
and the validation that keeps the composite honest: weights must sum to 1.0,
windows and counts must be positive, shares must be proper fractions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from munasib.core.config import (
    STRENGTH_WEIGHT_TOLERANCE,
    EvidenceConfig,
    EvidenceIndicatorsConfig,
    EvidenceStrengthConfig,
    Settings,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def test_documented_defaults() -> None:
    """The defaults match those documented in docs/evidence.md."""
    config = EvidenceIndicatorsConfig()
    assert config.n_saturation == 200
    assert config.polarization_min_share == 0.25
    assert config.recent_months == 24
    assert config.min_recent_reviews == 10
    assert config.trend_quarters == 8
    assert config.min_trend_quarters == 4
    assert config.min_quarter_reviews == 5
    assert config.burst_window_months == 6
    assert config.burst_flag_share == 0.50
    assert config.min_negative_reviews == 10
    assert config.conflict_probability_threshold == 0.5
    assert config.conflict_saturation == 0.20
    assert config.top_pain_points == 3
    assert config.staleness_scale_months == 12.0
    assert config.trend_saturation == 0.25


def test_strength_defaults_and_weight_sum() -> None:
    """The six composite weights ship as documented and sum to 1.0."""
    strength = EvidenceStrengthConfig()
    weights = strength.weights.model_dump()
    assert weights == {
        "review_volume": 0.25,
        "recency_coverage": 0.20,
        "verified_purchase_ratio": 0.15,
        "rating_agreement": 0.20,
        "rating_text_consistency": 0.15,
        "helpfulness_coverage": 0.05,
    }
    assert abs(sum(weights.values()) - 1.0) <= STRENGTH_WEIGHT_TOLERANCE
    assert strength.low_evidence_threshold == 0.40
    assert strength.min_reviews_for_confidence == 50


def test_shipped_config_declares_every_key() -> None:
    """The repo config file names each key, so nothing falls back silently."""
    text = REPO_CONFIG.read_text(encoding="utf-8")
    for key in (
        # The untouched live-servicing weights.
        "verified_purchase:",
        "helpfulness_ratio:",
        "review_depth:",
        # Indicator block.
        "n_saturation:",
        "polarization_min_share:",
        "recent_months:",
        "min_recent_reviews:",
        "trend_quarters:",
        "min_trend_quarters:",
        "min_quarter_reviews:",
        "burst_window_months:",
        "burst_flag_share:",
        "min_negative_reviews:",
        "conflict_probability_threshold:",
        "conflict_saturation:",
        "top_pain_points:",
        "staleness_scale_months:",
        "trend_saturation:",
        # Strength block.
        "low_evidence_threshold:",
        "min_reviews_for_confidence:",
        "rating_text_consistency:",
    ):
        assert f"{key}" in text, f"config.yaml is missing {key}"


def test_settings_exposes_the_blocks() -> None:
    """``Settings`` surfaces both new blocks under the review-intel config."""
    evidence = Settings().review_intel.evidence
    assert evidence.indicators.recent_months == 24
    assert evidence.strength.weights.recency_coverage == 0.20
    # The live three-indicator weights are untouched by the new blocks.
    assert evidence.weights.verified_purchase == 0.50


def test_weights_not_summing_to_one_are_rejected() -> None:
    """A weight set that does not sum to 1.0 is a configuration error."""
    with pytest.raises(ValidationError, match="sum to 1.0"):
        EvidenceStrengthConfig(weights={"review_volume": 0.9})


def test_negative_weight_is_rejected() -> None:
    """A negative weight would make the composite nonsensical."""
    with pytest.raises(ValidationError, match="must be positive"):
        EvidenceStrengthConfig(weights={"review_volume": -0.1})


def test_low_evidence_threshold_range() -> None:
    """The strength threshold lives in [0, 1]."""
    with pytest.raises(ValidationError, match="low_evidence_threshold"):
        EvidenceStrengthConfig(low_evidence_threshold=1.5)
    with pytest.raises(ValidationError, match="low_evidence_threshold"):
        EvidenceStrengthConfig(low_evidence_threshold=-0.1)


def test_min_reviews_for_confidence_must_be_positive() -> None:
    """A zero review minimum would make the guard meaningless."""
    with pytest.raises(ValidationError, match="min_reviews_for_confidence"):
        EvidenceStrengthConfig(min_reviews_for_confidence=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_saturation", 1),
        ("recent_months", 0),
        ("min_recent_reviews", 0),
        ("trend_quarters", 1),
        ("min_trend_quarters", 0),
        ("min_quarter_reviews", 0),
        ("burst_window_months", 0),
        ("min_negative_reviews", 0),
        ("top_pain_points", 0),
    ],
)
def test_non_positive_counts_are_rejected(field: str, value: int) -> None:
    """Every window and count must be at least its documented minimum."""
    with pytest.raises(ValidationError, match=field):
        EvidenceIndicatorsConfig(**{field: value})


def test_trend_minimum_cannot_exceed_the_window() -> None:
    """Requiring more qualifying quarters than the window holds is impossible."""
    with pytest.raises(ValidationError, match="min_trend_quarters"):
        EvidenceIndicatorsConfig(trend_quarters=8, min_trend_quarters=9)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("polarization_min_share", 0.0),
        ("polarization_min_share", 1.0),
        ("burst_flag_share", 0.0),
        ("burst_flag_share", 1.0),
        ("conflict_probability_threshold", 0.0),
        ("conflict_probability_threshold", 1.0),
    ],
)
def test_shares_must_be_proper_fractions(field: str, value: float) -> None:
    """A 0 or 1 share would make its rule either always or never fire."""
    with pytest.raises(ValidationError, match=field):
        EvidenceIndicatorsConfig(**{field: value})


@pytest.mark.parametrize(
    "field", ["conflict_saturation", "staleness_scale_months", "trend_saturation"]
)
def test_scales_must_be_positive(field: str) -> None:
    """A zero scale would divide by zero in the normalisation."""
    with pytest.raises(ValidationError, match=field):
        EvidenceIndicatorsConfig(**{field: 0.0})


def test_evidence_config_keeps_the_live_weights_intact() -> None:
    """The P14 blocks are additive: the live three-indicator weights are untouched."""
    config = EvidenceConfig()
    live = config.weights
    assert (live.verified_purchase, live.helpfulness_ratio, live.review_depth) == (
        0.50,
        0.30,
        0.20,
    )

"""Configuration tests for the risk criteria and risk-factor blocks.

The defaults must stay in sync with ``configs/config.yaml`` and every validator
must reject a value that would silently break the formulas.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from munasib.core.config import (
    ConcentratedComplaintsFactorConfig,
    ConcentrationCriteriaConfig,
    LowVerifiedFactorConfig,
    PolarizedFactorConfig,
    PriceAbovePeersFactorConfig,
    PricePositionCriteriaConfig,
    ProductQualityCriteriaConfig,
    RatingReliabilityCriteriaConfig,
    RecentDeclineFactorConfig,
    RiskConfig,
    RiskCriteriaConfig,
    StaleReviewsFactorConfig,
    WeakEvidenceFactorConfig,
    load_settings,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def test_risk_config_defaults_match_the_documented_contract() -> None:
    """The declared defaults are the ones the formulas assume."""
    config = RiskConfig()
    assert config.criteria.shrinkage_m == pytest.approx(20.0)

    quality = config.criteria.product_quality.weights
    assert quality.model_dump() == {
        "shrunk_rating": pytest.approx(0.50),
        "shrunk_sentiment": pytest.approx(0.30),
        "recent_sentiment": pytest.approx(0.20),
    }
    assert sum(quality.model_dump().values()) == pytest.approx(1.0)

    reliability = config.criteria.rating_reliability
    assert reliability.agreement_scale == pytest.approx(0.10)
    assert reliability.agreement_scale_candidates == pytest.approx([0.10, 0.20, 0.30])
    assert reliability.agreement_scale_max_zero_share == pytest.approx(0.25)
    assert sum(reliability.weights.model_dump().values()) == pytest.approx(1.0)

    concentration = config.criteria.negative_feedback_concentration
    assert concentration.p_saturation == pytest.approx(0.40)
    assert concentration.focus_floor == pytest.approx(0.50)

    assert config.criteria.price_position.fallback_confidence == pytest.approx(0.50)


def test_risk_factor_defaults_match_the_documented_bands() -> None:
    """Every band threshold has the value quoted in docs/risk_indicators.md."""
    factors = RiskConfig().factors
    assert factors.recent_sentiment_decline.model_dump() == {
        "min_recent_predictions": 10,
        "min_delta": pytest.approx(0.10),
        "medium_delta": pytest.approx(0.20),
        "high_delta": pytest.approx(0.35),
    }
    assert factors.polarized_ratings.model_dump() == {
        "medium_rating_agreement": pytest.approx(0.25),
        "high_rating_agreement": pytest.approx(0.10),
    }
    assert factors.low_verified_share.max_verified_share == pytest.approx(0.70)
    assert factors.weak_evidence.high_strength == pytest.approx(0.20)
    assert factors.stale_reviews.high_months == pytest.approx(48.0)
    assert factors.price_above_peers_with_low_quality.max_product_quality == pytest.approx(0.40)
    assert factors.concentrated_complaints.high_top_share == pytest.approx(0.80)
    assert factors.negative_burst.high_share == pytest.approx(0.80)
    assert factors.rating_text_conflict.high_conflict_rate == pytest.approx(0.20)


def test_config_yaml_declares_the_same_values_as_the_defaults() -> None:
    """The shipped YAML is the documented configuration, not a divergent copy."""
    settings = load_settings(CONFIG_PATH)
    defaults = RiskConfig()
    assert settings.risk.criteria.model_dump() == defaults.criteria.model_dump()
    assert settings.risk.factors.model_dump() == defaults.factors.model_dump()


def test_product_quality_weights_must_sum_to_one() -> None:
    """A weight vector that does not sum to 1 is rejected."""
    with pytest.raises(ValidationError, match="sum"):
        ProductQualityCriteriaConfig(
            weights={
                "shrunk_rating": 0.5,
                "shrunk_sentiment": 0.3,
                "recent_sentiment": 0.3,
            }
        )


def test_product_quality_weights_must_be_positive() -> None:
    """A zero or negative component weight is rejected."""
    with pytest.raises(ValidationError, match="positive"):
        ProductQualityCriteriaConfig(
            weights={
                "shrunk_rating": 0.8,
                "shrunk_sentiment": 0.2,
                "recent_sentiment": 0.0,
            }
        )


def test_rating_reliability_scale_must_be_positive() -> None:
    """A non-positive agreement scale is rejected."""
    with pytest.raises(ValidationError, match="positive"):
        RatingReliabilityCriteriaConfig(agreement_scale=0.0)


def test_rating_reliability_scale_must_be_a_candidate() -> None:
    """A scale outside the declared candidates is rejected."""
    with pytest.raises(ValidationError, match="one of the candidates"):
        RatingReliabilityCriteriaConfig(
            agreement_scale=0.15,
            agreement_scale_candidates=[0.10, 0.20],
        )


def test_rating_reliability_candidates_must_be_ascending() -> None:
    """Duplicate or descending candidates are rejected."""
    with pytest.raises(ValidationError, match="ascending"):
        RatingReliabilityCriteriaConfig(
            agreement_scale=0.10,
            agreement_scale_candidates=[0.20, 0.10],
        )
    with pytest.raises(ValidationError, match="ascending"):
        RatingReliabilityCriteriaConfig(
            agreement_scale=0.10,
            agreement_scale_candidates=[0.10, 0.10, 0.20],
        )


def test_rating_reliability_candidates_must_not_be_empty() -> None:
    """An empty candidate list is rejected."""
    with pytest.raises(ValidationError, match="must not be empty"):
        RatingReliabilityCriteriaConfig(
            agreement_scale=0.10,
            agreement_scale_candidates=[],
        )


def test_rating_reliability_zero_share_must_be_a_share() -> None:
    """The saturation cap is a share in [0, 1)."""
    with pytest.raises(ValidationError, match=r"\[0, 1\)"):
        RatingReliabilityCriteriaConfig(agreement_scale_max_zero_share=1.0)


def test_shrinkage_must_be_positive() -> None:
    """A zero pseudo-count would divide by zero in the shrinkage."""
    with pytest.raises(ValidationError, match="positive"):
        RiskCriteriaConfig(shrinkage_m=0.0)


def test_concentration_constants_must_be_in_range() -> None:
    """``p_saturation`` and ``focus_floor`` are in (0, 1]."""
    with pytest.raises(ValidationError, match=r"\(0, 1\]"):
        ConcentrationCriteriaConfig(p_saturation=0.0)
    with pytest.raises(ValidationError, match=r"\(0, 1\]"):
        ConcentrationCriteriaConfig(focus_floor=1.5)


def test_price_fallback_confidence_must_be_in_range() -> None:
    """The confidence multiplier is in [0, 1]."""
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        PricePositionCriteriaConfig(fallback_confidence=-0.1)


def test_recent_decline_bands_must_ascend() -> None:
    """The decline bands are strictly ascending."""
    with pytest.raises(ValidationError, match="ascending"):
        RecentDeclineFactorConfig(min_delta=0.30, medium_delta=0.20, high_delta=0.35)


def test_recent_decline_needs_at_least_one_recent_prediction() -> None:
    """A zero recent-prediction floor would make the rule meaningless."""
    with pytest.raises(ValidationError, match=">= 1"):
        RecentDeclineFactorConfig(min_recent_predictions=0)


def test_descending_bands_reject_inversion() -> None:
    """Polarized, low-verified and weak-evidence bands descend."""
    with pytest.raises(ValidationError, match="descending"):
        PolarizedFactorConfig(medium_rating_agreement=0.10, high_rating_agreement=0.25)
    with pytest.raises(ValidationError, match="descending"):
        LowVerifiedFactorConfig(max_verified_share=0.40, medium_verified_share=0.55)
    with pytest.raises(ValidationError, match="descending"):
        WeakEvidenceFactorConfig(max_strength=0.20, medium_strength=0.30)


def test_stale_reviews_rejects_negative_months() -> None:
    """A negative staleness floor is rejected."""
    with pytest.raises(ValidationError, match=r"in \[0"):
        StaleReviewsFactorConfig(min_months=-1.0)


def test_price_above_peers_rejects_an_out_of_range_quality_ceiling() -> None:
    """The quality gate is a 0-1 value."""
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        PriceAbovePeersFactorConfig(max_product_quality=1.5)


def test_concentrated_complaints_bands_must_ascend() -> None:
    """The complaint-share bands are strictly ascending."""
    with pytest.raises(ValidationError, match="ascending"):
        ConcentratedComplaintsFactorConfig(min_top_share=0.80, medium_top_share=0.65)

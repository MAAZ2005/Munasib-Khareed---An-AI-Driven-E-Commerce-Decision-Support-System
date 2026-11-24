"""Boundary and severity tests for the nine rule-based risk factors.

Each test drives ``generate_risk_factors`` with a synthetic stored payload; every
threshold comes from the default ``RiskFactorsConfig`` (the same defaults as
``configs/config.yaml``).
"""

from __future__ import annotations

from typing import Any

import pytest

from munasib.core.config import RiskFactorsConfig
from munasib.risk.factors import FACTOR_CODES, RiskFactor, generate_risk_factors
from munasib.risk.indicators import Criterion

CONFIG = RiskFactorsConfig()


def _entry(
    raw: float | None = None,
    normalized: float | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        "direction": "cost",
        "description": "",
        "missing_reason": None,
        "extras": extras or {},
    }


def _payload(
    indicators: dict[str, Any] | None = None,
    *,
    polarized: bool = False,
    burst: bool = False,
) -> dict[str, Any]:
    """Build a minimal stored evidence payload.

    Args:
        indicators: Indicator entries keyed by name.
        polarized: Stored polarization flag.
        burst: Stored temporal-burst flag.

    Returns:
        Evidence payload mapping.
    """
    return {
        "indicators": indicators or {},
        "polarized": polarized,
        "temporal_burst": burst,
        "low_evidence": False,
    }


def _factors(
    payload: dict[str, Any] | None,
    *,
    strength: float | None = 0.8,
    low_evidence: bool | None = False,
    features: dict[str, Any] | None = None,
    quality: float | None = None,
) -> dict[str, RiskFactor]:
    """Run the rules and index the triggered factors by code.

    Args:
        payload: Stored evidence payload.
        strength: Stored composite evidence strength.
        low_evidence: Stored low-evidence flag.
        features: Stored product-features payload.
        quality: Product-quality value (None omits the criterion).

    Returns:
        Triggered factors keyed by code.
    """
    criteria = {}
    if quality is not None:
        criteria["product_quality"] = Criterion(
            name="product_quality",
            value=quality,
            direction="benefit",
            description="",
            confidence=1.0,
        )
    triggered = generate_risk_factors(
        criteria=criteria,
        evidence_strength=strength,
        low_evidence=low_evidence,
        evidence_payload=payload,
        features_payload=features,
        config=CONFIG,
    )
    return {factor.code: factor for factor in triggered}


def _decline(delta: float, predictions: int = 12) -> dict[str, Any]:
    """Payload with only the recent-shift indicator.

    Args:
        delta: Recent minus overall negative share.
        predictions: Recent predicted-review count.

    Returns:
        Evidence payload.
    """
    return _payload(
        {
            "recent_negative_shift": _entry(
                raw=delta,
                extras={
                    "recent_negative_share": 0.5,
                    "overall_negative_share": 0.5 - delta,
                    "recent_predictions": predictions,
                },
            )
        }
    )


def test_recent_sentiment_decline_boundary_and_bands() -> None:
    """0.10 fires low, 0.20 medium, 0.35 high; just below does not fire."""
    assert "recent_sentiment_decline" not in _factors(_decline(0.0999))
    assert _factors(_decline(0.10))["recent_sentiment_decline"].severity == "low"
    assert _factors(_decline(0.20))["recent_sentiment_decline"].severity == "medium"
    assert _factors(_decline(0.35))["recent_sentiment_decline"].severity == "high"


def test_recent_sentiment_decline_needs_enough_recent_predictions() -> None:
    """Below the configured recent-prediction count the rule stays silent."""
    assert "recent_sentiment_decline" not in _factors(_decline(0.5, predictions=9))


def test_polarized_ratings_needs_the_flag_and_bands_agreement() -> None:
    """The flag triggers; lower agreement escalates."""
    agreement = _entry(
        normalized=0.30, extras={"star_shares": {"1": 0.2, "2": 0.1, "5": 0.5}}
    )
    quiet = _payload({"rating_agreement": agreement}, polarized=False)
    assert "polarized_ratings" not in _factors(quiet)

    loud = _payload({"rating_agreement": agreement}, polarized=True)
    assert _factors(loud)["polarized_ratings"].severity == "low"

    medium = _payload(
        {"rating_agreement": _entry(normalized=0.25)}, polarized=True
    )
    assert _factors(medium)["polarized_ratings"].severity == "medium"

    high = _payload({"rating_agreement": _entry(normalized=0.10)}, polarized=True)
    factor = _factors(high)["polarized_ratings"]
    assert factor.severity == "high"
    assert factor.supporting["rating_agreement"] == pytest.approx(0.10)


def test_low_verified_share_bands() -> None:
    """Above 0.70 is silent; 0.70/0.55/0.40 are low/medium/high."""
    assert "low_verified_share" not in _factors(
        _payload({"verified_purchase_ratio": _entry(raw=0.71)})
    )
    assert (
        _factors(_payload({"verified_purchase_ratio": _entry(raw=0.70)}))[
            "low_verified_share"
        ].severity
        == "low"
    )
    assert (
        _factors(_payload({"verified_purchase_ratio": _entry(raw=0.55)}))[
            "low_verified_share"
        ].severity
        == "medium"
    )
    assert (
        _factors(_payload({"verified_purchase_ratio": _entry(raw=0.40)}))[
            "low_verified_share"
        ].severity
        == "high"
    )


def test_weak_evidence_handles_null_and_bands() -> None:
    """A NULL composite is high severity; bands descend with the strength."""
    assert _factors(None, strength=None)["weak_evidence"].severity == "high"
    assert _factors(None, strength=0.39)["weak_evidence"].severity == "low"
    assert _factors(None, strength=0.30)["weak_evidence"].severity == "medium"
    assert _factors(None, strength=0.20)["weak_evidence"].severity == "high"
    assert "weak_evidence" not in _factors(None, strength=0.40)


def test_weak_evidence_fires_on_the_low_evidence_flag() -> None:
    """A thin-but-strong product still triggers, at low severity."""
    factor = _factors(None, strength=0.55, low_evidence=True)["weak_evidence"]
    assert factor.severity == "low"
    assert "low-evidence flag" in factor.description


def test_stale_reviews_bands() -> None:
    """Exactly 24 months is not stale; beyond it escalates at 36 and 48."""
    def stale(months: float) -> dict[str, RiskFactor]:
        return _factors(
            _payload(
                {
                    "staleness": _entry(
                        raw=months,
                        extras={"last_review_time": "2016-01-01T00:00:00"},
                    )
                }
            )
        )

    assert "stale_reviews" not in stale(24.0)
    assert stale(24.1)["stale_reviews"].severity == "low"
    assert stale(36.0)["stale_reviews"].severity == "medium"
    assert stale(48.0)["stale_reviews"].severity == "high"


def test_price_above_peers_requires_both_gates() -> None:
    """Expensive AND low quality; either gate failing means no factor."""
    features = {
        "price_percentile_category": 0.80,
        "price_peer_group": "group",
        "price_peer_count": 30,
    }
    assert "price_above_peers_with_low_quality" not in _factors(
        None, features=features, quality=0.40
    )
    assert "price_above_peers_with_low_quality" not in _factors(
        None,
        features={**features, "price_percentile_category": 0.75},
        quality=0.39,
    )
    factor = _factors(None, features=features, quality=0.39)[
        "price_above_peers_with_low_quality"
    ]
    assert factor.severity == "low"

    medium = _factors(
        None,
        features={**features, "price_percentile_category": 0.85},
        quality=0.20,
    )["price_above_peers_with_low_quality"]
    assert medium.severity == "medium"

    high = _factors(
        None,
        features={**features, "price_percentile_category": 0.95},
        quality=0.20,
    )["price_above_peers_with_low_quality"]
    assert high.severity == "high"


def test_concentrated_complaints_bands_and_supporting_aspect() -> None:
    """The top pain-point share drives the bands and names the aspect."""
    def concentrated(share: float) -> dict[str, RiskFactor]:
        return _factors(
            _payload(
                {
                    "top_pain_point_share": _entry(
                        raw=share,
                        extras={
                            "top_pain_points": [
                                {"aspect": "battery", "negative_mentions": 5}
                            ],
                            "negative_mentions": 10,
                        },
                    )
                }
            )
        )

    assert "concentrated_complaints" not in concentrated(0.4999)
    assert concentrated(0.50)["concentrated_complaints"].severity == "low"
    assert concentrated(0.65)["concentrated_complaints"].severity == "medium"
    factor = concentrated(0.80)["concentrated_complaints"]
    assert factor.severity == "high"
    assert factor.supporting["aspect"] == "battery"
    assert factor.supporting["negative_mentions"] == pytest.approx(5)
    assert "battery" in factor.description


def test_negative_burst_needs_the_flag_and_bands_the_share() -> None:
    """The stored flag triggers; 0.50/0.65/0.80 are low/medium/high."""
    def burst(share: float, flag: bool = True) -> dict[str, RiskFactor]:
        return _factors(
            _payload(
                {
                    "temporal_burst": _entry(
                        raw=share,
                        extras={
                            "flag": flag,
                            "window_start": "2018-01-01T00:00:00",
                            "window_end": "2018-04-01T00:00:00",
                            "window_months": 3,
                            "window_count": 5,
                            "negative_reviews": 10,
                        },
                    )
                },
                burst=flag,
            )
        )

    assert "negative_burst" not in burst(0.60, flag=False)
    assert burst(0.50)["negative_burst"].severity == "low"
    assert burst(0.65)["negative_burst"].severity == "medium"
    assert burst(0.80)["negative_burst"].severity == "high"


def test_rating_text_conflict_bands() -> None:
    """The conflict rate 0.10/0.15/0.20 is low/medium/high."""
    def conflict(conflicts: int, evaluated: int = 100) -> dict[str, RiskFactor]:
        return _factors(
            _payload(
                {
                    "rating_text_agreement": _entry(
                        extras={"conflicts": conflicts, "evaluated": evaluated}
                    )
                }
            )
        )

    assert "rating_text_conflict" not in conflict(9)
    assert conflict(10)["rating_text_conflict"].severity == "low"
    assert conflict(15)["rating_text_conflict"].severity == "medium"
    assert conflict(20)["rating_text_conflict"].severity == "high"
    assert "rating_text_conflict" not in conflict(1, evaluated=0)


def test_generate_risk_factors_keeps_the_stable_order() -> None:
    """Multiple triggers come back in the documented FACTOR_CODES order."""
    payload = _payload(
        {
            "recent_negative_shift": _entry(
                raw=0.5,
                extras={
                    "recent_negative_share": 0.9,
                    "overall_negative_share": 0.4,
                    "recent_predictions": 20,
                },
            ),
            "rating_agreement": _entry(normalized=0.05),
            "verified_purchase_ratio": _entry(raw=0.2),
            "top_pain_point_share": _entry(
                raw=0.9,
                extras={
                    "top_pain_points": [{"aspect": "x", "negative_mentions": 9}],
                    "negative_mentions": 10,
                },
            ),
        },
        polarized=True,
    )
    triggered = generate_risk_factors(
        criteria={},
        evidence_strength=0.1,
        low_evidence=True,
        evidence_payload=payload,
        features_payload=None,
        config=CONFIG,
    )
    codes = [factor.code for factor in triggered]
    assert codes == [
        code for code in FACTOR_CODES if code in codes
    ]
    assert len(codes) == len(set(codes))
    assert set(codes) >= {
        "recent_sentiment_decline",
        "polarized_ratings",
        "low_verified_share",
        "weak_evidence",
        "concentrated_complaints",
    }

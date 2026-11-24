"""Rule-based risk factors derived from stored evidence and criteria values.

Each rule is a pure function of one product's already-computed inputs:

* the stored ``product_evidence.indicators`` payload (or ``None`` when the
  product was never scored),
* the stored ``product_features.indicators`` payload (or ``None``),
* the five computed criteria (only ``product_quality`` is read here),
* the stored composite evidence strength and its ``low_evidence`` flag.

A factor is emitted only when its trigger threshold is met. Severity is
**banded**: the rule triggers at its first config threshold (``low``) and
escalates at the next two (``medium``, ``high``). Lower-is-worse inputs
(verified share, agreement, strength) have descending bands; higher-is-worse
inputs (decline delta, months, share, percentile) have ascending bands.

No value is imputed: a rule whose inputs are missing simply does not fire, and
the ``weak_evidence`` rule fires ``high`` when no composite exists at all (the
absence of evidence is itself the finding). Every threshold comes from
``risk.factors`` in ``configs/config.yaml``; the formulas are documented in
``docs/risk_indicators.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from munasib.core.config import RiskFactorsConfig
from munasib.risk.indicators import Criterion

#: Factor codes in the stable emission order (also the summary order).
FACTOR_CODES: tuple[str, ...] = (
    "recent_sentiment_decline",
    "polarized_ratings",
    "low_verified_share",
    "weak_evidence",
    "stale_reviews",
    "price_above_peers_with_low_quality",
    "concentrated_complaints",
    "negative_burst",
    "rating_text_conflict",
)

#: Severity labels, weakest first.
SEVERITIES: tuple[str, ...] = ("low", "medium", "high")


@dataclass(frozen=True)
class RiskFactor:
    """One triggered risk factor.

    Attributes:
        code: Factor code (one of :data:`FACTOR_CODES`).
        severity: ``low``/``medium``/``high``.
        description: One sentence citing the computed numbers.
        supporting: JSON-safe numbers/flags the trigger was based on.
    """

    code: str
    severity: str
    description: str
    supporting: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Render the stored JSON for this factor.

        Returns:
            A JSON-safe mapping.
        """
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")
        return {
            "code": self.code,
            "severity": self.severity,
            "description": self.description,
            "supporting": dict(self.supporting),
        }


def _ascending_severity(value: float, low: float, medium: float, high: float) -> str:
    """Severity of a higher-is-worse value.

    Args:
        value: Observed value (already at or above ``low``).
        low: Trigger threshold (severity ``low``).
        medium: Medium threshold.
        high: High threshold.

    Returns:
        ``high`` when ``value >= high``, else ``medium`` when ``>= medium``,
        else ``low``.
    """
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def _descending_severity(value: float, low: float, medium: float, high: float) -> str:
    """Severity of a lower-is-worse value.

    Args:
        value: Observed value (already at or below ``low``).
        low: Trigger threshold (severity ``low``).
        medium: Medium threshold.
        high: High threshold.

    Returns:
        ``high`` when ``value <= high``, else ``medium`` when ``<= medium``,
        else ``low``.
    """
    if value <= high:
        return "high"
    if value <= medium:
        return "medium"
    return "low"


def _indicator(payload: Mapping[str, Any] | None, name: str) -> Mapping[str, Any] | None:
    """Fetch one stored indicator entry.

    Args:
        payload: Stored evidence ``indicators`` payload, or None.
        name: Indicator name.

    Returns:
        The indicator mapping, or None when the payload or the entry is absent.
    """
    if not isinstance(payload, Mapping):
        return None
    indicators = payload.get("indicators")
    if not isinstance(indicators, Mapping):
        return None
    entry = indicators.get(name)
    return entry if isinstance(entry, Mapping) else None


def _number(value: Any) -> float | None:
    """Convert a stored number to ``float``; anything else becomes ``None``.

    Args:
        value: Stored JSON value.

    Returns:
        The float value, or None when it is missing or not numeric.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _recent_sentiment_decline(
    payload: Mapping[str, Any] | None, config: RiskFactorsConfig
) -> RiskFactor | None:
    """Fire when the recent negative share rose well above the overall share.

    Args:
        payload: Stored evidence payload.
        config: Factor thresholds.

    Returns:
        The factor, or None when the shift is missing or below the threshold.
    """
    settings = config.recent_sentiment_decline
    entry = _indicator(payload, "recent_negative_shift")
    if entry is None:
        return None
    delta = _number(entry.get("raw_value"))
    extras = entry.get("extras")
    if not isinstance(extras, Mapping):
        return None
    predictions = extras.get("recent_predictions")
    recent = _number(extras.get("recent_negative_share"))
    overall = _number(extras.get("overall_negative_share"))
    if (
        delta is None
        or not isinstance(predictions, int)
        or predictions < settings.min_recent_predictions
        or recent is None
        or overall is None
        or delta < settings.min_delta
    ):
        return None
    severity = _ascending_severity(
        delta, settings.min_delta, settings.medium_delta, settings.high_delta
    )
    description = (
        f"The recent negative share is {recent:.1%} over {predictions:,} recent "
        f"predicted reviews versus {overall:.1%} overall ({delta:+.4f})."
    )
    return RiskFactor(
        code="recent_sentiment_decline",
        severity=severity,
        description=description,
        supporting={
            "recent_negative_share": recent,
            "overall_negative_share": overall,
            "delta": delta,
            "recent_predictions": predictions,
        },
    )


def _polarized_ratings(
    payload: Mapping[str, Any] | None, config: RiskFactorsConfig
) -> RiskFactor | None:
    """Fire when the stored rating distribution is flagged as polarized.

    Args:
        payload: Stored evidence payload.
        config: Factor thresholds.

    Returns:
        The factor, or None when the product is not flagged or has no ratings.
    """
    if not isinstance(payload, Mapping) or not payload.get("polarized"):
        return None
    entry = _indicator(payload, "rating_agreement")
    if entry is None:
        return None
    agreement = _number(entry.get("normalized"))
    if agreement is None:
        return None
    settings = config.polarized_ratings
    severity = (
        "high"
        if agreement <= settings.high_rating_agreement
        else "medium"
        if agreement <= settings.medium_rating_agreement
        else "low"
    )
    extras = entry.get("extras")
    shares = extras.get("star_shares") if isinstance(extras, Mapping) else None
    low_stars = high_stars = None
    if isinstance(shares, Mapping):
        one = _number(shares.get("1")) or 0.0
        two = _number(shares.get("2")) or 0.0
        five = _number(shares.get("5")) or 0.0
        low_stars, high_stars = one + two, five
    detail = ""
    if low_stars is not None and high_stars is not None:
        detail = f" ({low_stars:.1%} are 1-2 stars, {high_stars:.1%} are 5 stars)"
    return RiskFactor(
        code="polarized_ratings",
        severity=severity,
        description=(
            f"The rating distribution is polarized (normalized agreement "
            f"{agreement:.4f}){detail}."
        ),
        supporting={
            "rating_agreement": agreement,
            "low_star_share": low_stars,
            "high_star_share": high_stars,
        },
    )


def _low_verified_share(
    payload: Mapping[str, Any] | None, config: RiskFactorsConfig
) -> RiskFactor | None:
    """Fire when too few reviews with a known flag are verified purchases.

    Args:
        payload: Stored evidence payload.
        config: Factor thresholds.

    Returns:
        The factor, or None when the ratio is missing or above the threshold.
    """
    entry = _indicator(payload, "verified_purchase_ratio")
    if entry is None:
        return None
    share = _number(entry.get("raw_value"))
    if share is None:
        return None
    settings = config.low_verified_share
    if share > settings.max_verified_share:
        return None
    severity = _descending_severity(
        share,
        settings.max_verified_share,
        settings.medium_verified_share,
        settings.high_verified_share,
    )
    extras = entry.get("extras") if isinstance(entry.get("extras"), Mapping) else {}
    known = _number(extras.get("verified_known")) if extras else None
    count = _number(extras.get("verified_count")) if extras else None
    detail = (
        f" ({count:.0f} of {known:.0f} reviews with a known flag)"
        if known is not None and count is not None
        else ""
    )
    return RiskFactor(
        code="low_verified_share",
        severity=severity,
        description=(
            f"Only {share:.1%} of the reviews with a known verified flag are "
            f"verified purchases{detail}."
        ),
        supporting={"verified_share": share},
    )


def _weak_evidence(
    evidence_strength: float | None,
    low_evidence: bool | None,
    config: RiskFactorsConfig,
) -> RiskFactor | None:
    """Fire when the evidence composite is thin, weak or missing entirely.

    Args:
        evidence_strength: Stored composite (None when absent or NULL).
        low_evidence: Stored low-evidence flag (None when no evidence row).
        config: Factor thresholds.

    Returns:
        The factor, or None when the evidence is strong enough.
    """
    settings = config.weak_evidence
    if evidence_strength is None:
        return RiskFactor(
            code="weak_evidence",
            severity="high",
            description=(
                "No composite evidence strength exists for this product (no stored "
                "evidence row or no component could be computed)."
            ),
            supporting={"evidence_strength": None, "low_evidence": low_evidence},
        )
    if evidence_strength < settings.max_strength:
        severity = _descending_severity(
            evidence_strength,
            settings.max_strength,
            settings.medium_strength,
            settings.high_strength,
        )
        description = (
            f"The composite evidence strength is {evidence_strength:.4f}, below "
            f"the weak-evidence threshold {settings.max_strength:g}."
        )
    elif low_evidence:
        severity = "low"
        description = (
            f"The composite evidence strength is {evidence_strength:.4f} but the "
            "stored low-evidence flag is set (thin review count), so the evidence "
            "cannot be trusted."
        )
    else:
        return None
    return RiskFactor(
        code="weak_evidence",
        severity=severity,
        description=description,
        supporting={"evidence_strength": evidence_strength, "low_evidence": low_evidence},
    )


def _stale_reviews(
    payload: Mapping[str, Any] | None, config: RiskFactorsConfig
) -> RiskFactor | None:
    """Fire when the newest review is older than the configured months.

    Args:
        payload: Stored evidence payload.
        config: Factor thresholds.

    Returns:
        The factor, or None when staleness is missing or below the threshold.
    """
    entry = _indicator(payload, "staleness")
    if entry is None:
        return None
    months = _number(entry.get("raw_value"))
    if months is None:
        return None
    settings = config.stale_reviews
    if months <= settings.min_months:
        return None
    severity = _ascending_severity(
        months, settings.min_months, settings.medium_months, settings.high_months
    )
    extras = entry.get("extras") if isinstance(entry.get("extras"), Mapping) else {}
    last_review = extras.get("last_review_time") if extras else None
    detail = f"; the newest review is from {last_review}" if last_review else ""
    return RiskFactor(
        code="stale_reviews",
        severity=severity,
        description=(
            f"The newest review is {months:.1f} months old at the reference date "
            f"(threshold {settings.min_months:g} months){detail}."
        ),
        supporting={"staleness_months": months, "last_review_time": last_review},
    )


def _price_above_peers_with_low_quality(
    criteria: Mapping[str, Criterion],
    features: Mapping[str, Any] | None,
    config: RiskFactorsConfig,
) -> RiskFactor | None:
    """Fire when a costly product also has below-threshold product quality.

    Args:
        criteria: Computed criteria (only ``product_quality`` is read).
        features: Stored ``product_features.indicators`` payload, or None.
        config: Factor thresholds.

    Returns:
        The factor, or None when either input is missing or a gate fails.
    """
    if not isinstance(features, Mapping):
        return None
    percentile = _number(features.get("price_percentile_category"))
    quality = criteria["product_quality"].value if "product_quality" in criteria else None
    settings = config.price_above_peers_with_low_quality
    if percentile is None or quality is None:
        return None
    if percentile <= settings.min_price_percentile or quality >= settings.max_product_quality:
        return None
    severity = _ascending_severity(
        percentile,
        settings.min_price_percentile,
        settings.medium_price_percentile,
        settings.high_price_percentile,
    )
    group = features.get("price_peer_group")
    peer_count = features.get("price_peer_count")
    return RiskFactor(
        code="price_above_peers_with_low_quality",
        severity=severity,
        description=(
            f"The price sits at percentile {percentile:.4f} of its peer group "
            f"{group!r} ({peer_count} priced products) while product quality is "
            f"{quality:.4f}, below {settings.max_product_quality:g}."
        ),
        supporting={
            "price_percentile_category": percentile,
            "product_quality": quality,
            "price_peer_group": group,
            "price_peer_count": peer_count,
        },
    )


def _concentrated_complaints(
    payload: Mapping[str, Any] | None, config: RiskFactorsConfig
) -> RiskFactor | None:
    """Fire when a few pain-point aspects cover most negative mentions.

    Args:
        payload: Stored evidence payload.
        config: Factor thresholds.

    Returns:
        The factor, or None when the share is missing or below the threshold.
    """
    entry = _indicator(payload, "top_pain_point_share")
    if entry is None:
        return None
    share = _number(entry.get("raw_value"))
    if share is None:
        return None
    settings = config.concentrated_complaints
    if share < settings.min_top_share:
        return None
    severity = _ascending_severity(
        share, settings.min_top_share, settings.medium_top_share, settings.high_top_share
    )
    extras = entry.get("extras") if isinstance(entry.get("extras"), Mapping) else {}
    top = extras.get("top_pain_points") if extras else None
    first = top[0] if isinstance(top, list) and top else None
    mentions = _number(extras.get("negative_mentions")) if extras else None
    detail = ""
    aspect = negative_mentions = None
    if isinstance(first, Mapping):
        aspect = first.get("aspect")
        negative_mentions = _number(first.get("negative_mentions"))
        if negative_mentions is not None and mentions is not None:
            detail = (
                f"; the largest is {aspect!r} with {negative_mentions:.0f} of "
                f"{mentions:.0f} negative mentions"
            )
        else:
            detail = f"; the largest is {aspect!r}"
    mentions_text = (
        f"the {mentions:.0f} negative-review aspect mentions"
        if mentions is not None
        else "the negative-review aspect mentions"
    )
    return RiskFactor(
        code="concentrated_complaints",
        severity=severity,
        description=(
            f"The top pain-point aspects cover {share:.1%} of {mentions_text}{detail}."
        ),
        supporting={
            "top_pain_point_share": share,
            "aspect": aspect,
            "negative_mentions": negative_mentions,
            "total_negative_mentions": mentions,
        },
    )


def _negative_burst(
    payload: Mapping[str, Any] | None, config: RiskFactorsConfig
) -> RiskFactor | None:
    """Fire when a short window holds a large share of the negative reviews.

    Args:
        payload: Stored evidence payload.
        config: Factor thresholds.

    Returns:
        The factor, or None when the burst flag is absent.
    """
    if not isinstance(payload, Mapping) or not payload.get("temporal_burst"):
        return None
    entry = _indicator(payload, "temporal_burst")
    if entry is None:
        return None
    share = _number(entry.get("raw_value"))
    if share is None:
        return None
    settings = config.negative_burst
    if share < settings.min_share:
        return None
    severity = _ascending_severity(
        share, settings.min_share, settings.medium_share, settings.high_share
    )
    extras = entry.get("extras") if isinstance(entry.get("extras"), Mapping) else {}
    window_start = extras.get("window_start") if extras else None
    window_end = extras.get("window_end") if extras else None
    window_months = extras.get("window_months") if extras else None
    count = extras.get("window_count") if extras else None
    total = extras.get("negative_reviews") if extras else None
    window = ""
    if window_start and window_end:
        window = (
            f" between {window_start} and {window_end}"
            f" ({window_months} months, {count} of {total} negative reviews)"
        )
    return RiskFactor(
        code="negative_burst",
        severity=severity,
        description=f"{share:.1%} of the negative reviews fall in one window{window}.",
        supporting={
            "burst_share": share,
            "window_start": window_start,
            "window_end": window_end,
            "window_months": window_months,
            "window_count": count,
            "negative_reviews": total,
        },
    )


def _rating_text_conflict(
    payload: Mapping[str, Any] | None, config: RiskFactorsConfig
) -> RiskFactor | None:
    """Fire when star ratings and review text contradict each other often.

    Args:
        payload: Stored evidence payload.
        config: Factor thresholds.

    Returns:
        The factor, or None when the conflict rate is missing or below threshold.
    """
    entry = _indicator(payload, "rating_text_agreement")
    if entry is None:
        return None
    extras = entry.get("extras")
    if not isinstance(extras, Mapping):
        return None
    conflicts = _number(extras.get("conflicts"))
    evaluated = _number(extras.get("evaluated"))
    if conflicts is None or evaluated is None or evaluated <= 0:
        return None
    rate = conflicts / evaluated
    settings = config.rating_text_conflict
    if rate < settings.min_conflict_rate:
        return None
    severity = _ascending_severity(
        rate,
        settings.min_conflict_rate,
        settings.medium_conflict_rate,
        settings.high_conflict_rate,
    )
    return RiskFactor(
        code="rating_text_conflict",
        severity=severity,
        description=(
            f"{conflicts:.0f} of {evaluated:.0f} rated, predicted reviews ({rate:.1%}) "
            "have a star rating that contradicts the review text."
        ),
        supporting={
            "conflicts": conflicts,
            "evaluated": evaluated,
            "conflict_rate": rate,
        },
    )


def generate_risk_factors(
    *,
    criteria: Mapping[str, Criterion],
    evidence_strength: float | None,
    low_evidence: bool | None,
    evidence_payload: Mapping[str, Any] | None,
    features_payload: Mapping[str, Any] | None,
    config: RiskFactorsConfig,
) -> tuple[RiskFactor, ...]:
    """Run every rule over one product, in the stable :data:`FACTOR_CODES` order.

    Args:
        criteria: The five computed criteria.
        evidence_strength: Stored composite evidence strength.
        low_evidence: Stored low-evidence flag.
        evidence_payload: Stored ``product_evidence.indicators`` payload.
        features_payload: Stored ``product_features.indicators`` payload.
        config: Factor thresholds.

    Returns:
        The triggered factors, ordered by :data:`FACTOR_CODES`.
    """
    candidates = (
        _recent_sentiment_decline(evidence_payload, config),
        _polarized_ratings(evidence_payload, config),
        _low_verified_share(evidence_payload, config),
        _weak_evidence(evidence_strength, low_evidence, config),
        _stale_reviews(evidence_payload, config),
        _price_above_peers_with_low_quality(criteria, features_payload, config),
        _concentrated_complaints(evidence_payload, config),
        _negative_burst(evidence_payload, config),
        _rating_text_conflict(evidence_payload, config),
    )
    triggered = tuple(factor for factor in candidates if factor is not None)
    order = {code: index for index, code in enumerate(FACTOR_CODES)}
    return tuple(sorted(triggered, key=lambda factor: order[factor.code]))


__all__ = ["FACTOR_CODES", "SEVERITIES", "RiskFactor", "generate_risk_factors"]

"""Pure functions computing the five per-product decision criteria.

The risk stage turns stored, already-computed values into the five criteria of
``configs/criteria_weights.yaml`` - it never runs a model, never touches the
database and never reads the clock. Everything here is a pure function of its
arguments, so the same inputs always produce the same row.

Every criterion is either a weighted mean of 0-1 components or a single mapped
value. A component that is unavailable (``None``) is **dropped**, the remaining
weights are renormalized over the components actually used, and the row records
``components_used``, the renormalized ``weights``, ``missing_components`` with
their reasons, and a ``confidence`` equal to the share of component weight that
was actually available. Nothing is imputed.

The formulas, constants and thresholds are documented in
``docs/risk_indicators.md``; every constant comes from
:mod:`munasib.core.config` (``risk.criteria``), never from this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from munasib.core.config import RiskCriteriaConfig
from munasib.core.exceptions import DataError

#: Decimal places kept in every stored float.
DIGITS = 6

#: Direction of a criterion: a higher value is better (benefit) or worse (cost).
DIRECTION_BENEFIT = "benefit"
DIRECTION_COST = "cost"

#: The five criteria in the fixed TOPSIS vector order (``fuzzy_risk`` is the
#: fuzzy stage's output and is not computed here).
CRITERION_ORDER: tuple[str, ...] = (
    "product_quality",
    "evidence_strength",
    "rating_reliability",
    "negative_feedback_concentration",
    "price_position",
)

#: Expected directions as declared in ``configs/criteria_weights.yaml``. The
#: loader compares the file against this map and fails loudly on a mismatch.
EXPECTED_DIRECTIONS: Mapping[str, str] = {
    "product_quality": DIRECTION_BENEFIT,
    "evidence_strength": DIRECTION_BENEFIT,
    "rating_reliability": DIRECTION_BENEFIT,
    "negative_feedback_concentration": DIRECTION_COST,
    "price_position": DIRECTION_COST,
}


def _clamp_unit(value: float) -> float:
    """Clamp a value into ``[0, 1]``.

    Args:
        value: Raw value.

    Returns:
        ``min(1, max(0, value))``.
    """
    return min(1.0, max(0.0, float(value)))


def _rounded(value: float | None) -> float | None:
    """Round a float to :data:`DIGITS` decimals, preserving ``None``.

    Args:
        value: Value to round.

    Returns:
        The rounded value, or ``None``.
    """
    return None if value is None else round(float(value), DIGITS)


def shrink(value: float, n: float, prior: float, m: float) -> float:
    """Bayesian shrinkage of a product value toward the category prior.

    ``shrunk = (n * value + m * prior) / (n + m)``: ``m`` pseudo-reviews worth
    of the prior are mixed in, so a product with three reviews barely moves the
    prior while one with three hundred is almost its own mean.

    Args:
        value: The product's own value.
        n: Number of observations behind it (reviews or predictions).
        prior: Category prior (computed from TRAIN-split products only).
        m: Shrinkage pseudo-count (``risk.criteria.shrinkage_m``).

    Returns:
        The shrunk value.

    Raises:
        ValueError: If ``n`` is negative or ``m`` is not positive.
    """
    if n < 0:
        raise ValueError(f"observation count must be non-negative, got {n}")
    if m <= 0:
        raise ValueError(f"shrinkage pseudo-count must be positive, got {m}")
    return (n * float(value) + m * float(prior)) / (n + m)


@dataclass(frozen=True)
class Criterion:
    """One decision criterion, with everything needed to audit it.

    Attributes:
        name: Criterion key (one of :data:`CRITERION_ORDER`).
        value: 0-1 value, or ``None`` when no component was available.
        direction: ``benefit`` or ``cost`` (from ``criteria_weights.yaml``).
        description: One sentence citing the computed numbers.
        confidence: Share of component weight actually available (0-1).
        components_used: Components that contributed, in config order.
        weights: Renormalized component weights (sum 1.0 when non-empty).
        missing_components: Component name -> reason it was dropped.
        missing_reason: Reason the whole criterion is ``None`` (only then).
        extras: Supporting numbers (flags, intermediate values).
    """

    name: str
    value: float | None
    direction: str
    description: str
    confidence: float
    components_used: tuple[str, ...] = ()
    weights: Mapping[str, float] = field(default_factory=dict)
    missing_components: Mapping[str, str] = field(default_factory=dict)
    missing_reason: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Render the stored JSON for this criterion.

        Returns:
            A JSON-safe mapping with rounded floats.
        """
        return {
            "value": _rounded(self.value),
            "direction": self.direction,
            "description": self.description,
            "confidence": _rounded(self.confidence),
            "components_used": list(self.components_used),
            "weights": {name: _rounded(weight) for name, weight in self.weights.items()},
            "missing_components": dict(self.missing_components),
            "missing_reason": self.missing_reason,
            "extras": dict(self.extras),
        }


def _weighted_components(
    components: Mapping[str, tuple[float | None, str | None]],
    weights: Mapping[str, float],
) -> tuple[float | None, tuple[str, ...], dict[str, float], dict[str, str], float]:
    """Weighted mean over the available components with weight renormalization.

    Args:
        components: Component name -> ``(value or None, missing reason)``.
        weights: Component name -> configured weight (sums to 1.0).

    Returns:
        ``(value, components_used, renormalized_weights, missing, confidence)``.
        ``value`` is ``None`` only when every component is missing.
    """
    total_weight = float(sum(weights[name] for name in components))
    available: dict[str, float] = {}
    missing: dict[str, str] = {}
    for name, (value, reason) in components.items():
        if value is None:
            missing[name] = reason or "unavailable"
        else:
            available[name] = float(value)
    if not available:
        return None, (), {}, missing, 0.0
    used_weight = sum(weights[name] for name in available)
    renormalized = {name: weights[name] / used_weight for name in available}
    value = sum(renormalized[name] * available[name] for name in available)
    confidence = used_weight / total_weight if total_weight > 0 else 0.0
    return _clamp_unit(value), tuple(available), renormalized, missing, confidence


def product_quality_criterion(
    *,
    mean_rating: float | None,
    n_reviews: int,
    mean_net: float | None,
    n_predictions: int,
    recent_negative_shift: float | None,
    recent_negative_shift_reason: str | None,
    prior_rating: float,
    prior_net: float,
    config: RiskCriteriaConfig,
) -> Criterion:
    """Product quality from the shrunk rating, shrunk sentiment and recent shift.

    Components:

    * ``shrunk_rating`` = ``((n_reviews * mean_rating + m * prior_rating)
      / (n_reviews + m) - 1) / 4`` (a 1-5 rating mapped onto 0-1).
    * ``shrunk_sentiment`` = ``(shrunk_net + 1) / 2`` with the same shrinkage
      over the product's predicted reviews.
    * ``recent_sentiment`` = the stored ``recent_negative_shift`` normalized
      value (1.0 = no worsening recent shift).

    Args:
        mean_rating: Product's mean star rating (None when unrated).
        n_reviews: Rated reviews behind ``mean_rating``.
        mean_net: Mean per-review net sentiment over predicted reviews.
        n_predictions: Predicted reviews behind ``mean_net``.
        recent_negative_shift: Stored normalized shift (None when unavailable).
        recent_negative_shift_reason: Why the shift is unavailable.
        prior_rating: Train-split rating prior.
        prior_net: Train-split net-sentiment prior.
        config: Risk criteria configuration.

    Returns:
        The ``product_quality`` criterion.
    """
    m = config.shrinkage_m
    weights = config.product_quality.weights.model_dump()
    parts: dict[str, tuple[float | None, str | None]] = {}
    notes: list[str] = []

    rating_component: float | None = None
    shrunk_rating: float | None = None
    if mean_rating is not None and n_reviews > 0:
        shrunk_rating = shrink(mean_rating, n_reviews, prior_rating, m)
        rating_component = _clamp_unit((shrunk_rating - 1.0) / 4.0)
        parts["shrunk_rating"] = (rating_component, None)
        notes.append(
            f"mean rating {mean_rating:.2f} over {n_reviews:,} reviews shrinks to "
            f"{shrunk_rating:.4f} against the train prior {prior_rating:.4f} "
            f"(m={m:g}) -> {rating_component:.4f}"
        )
    else:
        parts["shrunk_rating"] = (None, "no_ratings")

    sentiment_component: float | None = None
    shrunk_net: float | None = None
    if mean_net is not None and n_predictions > 0:
        shrunk_net = shrink(mean_net, n_predictions, prior_net, m)
        sentiment_component = _clamp_unit((shrunk_net + 1.0) / 2.0)
        parts["shrunk_sentiment"] = (sentiment_component, None)
        notes.append(
            f"net sentiment {mean_net:+.4f} over {n_predictions:,} predictions shrinks "
            f"to {shrunk_net:+.4f} (prior {prior_net:+.4f}) -> "
            f"{sentiment_component:.4f}"
        )
    else:
        parts["shrunk_sentiment"] = (None, "no_predictions")

    if recent_negative_shift is not None:
        parts["recent_sentiment"] = (_clamp_unit(recent_negative_shift), None)
        notes.append(f"recent negative shift {_clamp_unit(recent_negative_shift):.4f}")
    else:
        parts["recent_sentiment"] = (
            None,
            recent_negative_shift_reason or "no_recent_sentiment",
        )

    value, used, weights_used, missing, confidence = _weighted_components(parts, weights)
    if value is None:
        description = (
            "No product-quality component is available (no rated reviews, no "
            "predictions and no recent-sentiment shift)."
        )
        return Criterion(
            name="product_quality",
            value=None,
            direction=DIRECTION_BENEFIT,
            description=description,
            confidence=0.0,
            components_used=(),
            weights={},
            missing_components=missing,
            missing_reason="all_components_missing",
            extras={
                "prior_rating": _rounded(prior_rating),
                "prior_net": _rounded(prior_net),
                "shrinkage_m": m,
            },
        )
    description = (
        "; ".join(notes)
        + f". Weighted mean {value:.4f} over {len(used)} of {len(parts)} components "
        f"(confidence {confidence:.2f})."
    )
    return Criterion(
        name="product_quality",
        value=value,
        direction=DIRECTION_BENEFIT,
        description=description,
        confidence=confidence,
        components_used=used,
        weights=weights_used,
        missing_components=missing,
        extras={
            "prior_rating": _rounded(prior_rating),
            "prior_net": _rounded(prior_net),
            "shrinkage_m": m,
            "shrunk_rating": _rounded(shrunk_rating),
            "shrunk_net_sentiment": _rounded(shrunk_net),
            "components": {
                "shrunk_rating": _rounded(rating_component),
                "shrunk_sentiment": _rounded(sentiment_component),
                "recent_sentiment": _rounded(recent_negative_shift),
            },
        },
    )


def evidence_strength_criterion(
    strength: float | None,
    *,
    low_evidence: bool | None,
    missing_reason: str | None,
) -> Criterion:
    """The stored evidence composite, unchanged.

    Args:
        strength: ``product_evidence.evidence_strength`` (None when absent/NULL).
        low_evidence: The stored low-evidence flag (None when no evidence row).
        missing_reason: Why the strength is absent.

    Returns:
        The ``evidence_strength`` criterion.
    """
    if strength is None:
        return Criterion(
            name="evidence_strength",
            value=None,
            direction=DIRECTION_BENEFIT,
            description=(
                "No stored evidence composite is available for this product "
                f"({missing_reason or 'no_evidence_row'})."
            ),
            confidence=0.0,
            missing_reason=missing_reason or "no_evidence_row",
            extras={"low_evidence": low_evidence},
        )
    value = _clamp_unit(strength)
    return Criterion(
        name="evidence_strength",
        value=value,
        direction=DIRECTION_BENEFIT,
        description=(
            f"Stored composite evidence strength {value:.4f} "
            f"(low_evidence={bool(low_evidence)})."
        ),
        confidence=1.0,
        components_used=("evidence_strength",),
        weights={"evidence_strength": 1.0},
        extras={"low_evidence": bool(low_evidence)},
    )


def rating_reliability_criterion(
    *,
    rating_agreement: float | None,
    polarized: bool | None,
    rating_text_consistency: float | None,
    helpfulness_gap: float | None,
    config: RiskCriteriaConfig,
    agreement_scale: float | None = None,
) -> Criterion:
    """Rating reliability from agreement, polarization, text and helpfulness.

    Components:

    * ``rating_agreement`` - the stored normalized agreement indicator.
    * ``non_polarization`` - 1.0 when the ratings are not polarized, else 0.0.
      Only available when ``rating_agreement`` is (an unrated product has no
      polarization signal; a stored ``False`` would otherwise fabricate 1.0).
    * ``rating_text_consistency`` - the stored normalized text agreement.
    * ``helpfulness_agreement`` = ``1 - min(1, gap / agreement_scale)`` with
      ``gap = |helpfulness-weighted net - unweighted net|``.

    Args:
        rating_agreement: Stored normalized rating agreement (None = no ratings).
        polarized: Stored polarization flag (None = no evidence row).
        rating_text_consistency: Stored normalized rating/text consistency.
        helpfulness_gap: Absolute weighted/unweighted net-sentiment gap.
        config: Risk criteria configuration.
        agreement_scale: Scale actually used (None = the configured one). The
            run may pick a smaller candidate after the saturation check.

    Returns:
        The ``rating_reliability`` criterion.
    """
    weights = config.rating_reliability.weights.model_dump()
    scale = (
        config.rating_reliability.agreement_scale
        if agreement_scale is None
        else float(agreement_scale)
    )
    parts: dict[str, tuple[float | None, str | None]] = {}
    notes: list[str] = []

    if rating_agreement is not None:
        parts["rating_agreement"] = (_clamp_unit(rating_agreement), None)
        notes.append(f"rating agreement {_clamp_unit(rating_agreement):.4f}")
    else:
        parts["rating_agreement"] = (None, "no_ratings")

    if rating_agreement is not None and polarized is not None:
        non_polarization = 0.0 if polarized else 1.0
        parts["non_polarization"] = (non_polarization, None)
        notes.append(f"non-polarization {non_polarization:.1f} (polarized={polarized})")
    else:
        parts["non_polarization"] = (None, "no_ratings")

    if rating_text_consistency is not None:
        parts["rating_text_consistency"] = (_clamp_unit(rating_text_consistency), None)
        notes.append(
            f"rating/text consistency {_clamp_unit(rating_text_consistency):.4f}"
        )
    else:
        parts["rating_text_consistency"] = (None, "no_rating_text_pairs")

    helpfulness_agreement: float | None = None
    if helpfulness_gap is not None:
        helpfulness_agreement = 1.0 - min(1.0, float(helpfulness_gap) / scale)
        parts["helpfulness_agreement"] = (_clamp_unit(helpfulness_agreement), None)
        notes.append(
            f"helpfulness agreement {helpfulness_agreement:.4f} "
            f"(gap {helpfulness_gap:.4f}, scale {scale:g})"
        )
    else:
        parts["helpfulness_agreement"] = (None, "no_predictions")

    value, used, weights_used, missing, confidence = _weighted_components(parts, weights)
    extras: dict[str, Any] = {
        "agreement_scale": scale,
        "helpfulness_gap": _rounded(helpfulness_gap),
        "helpfulness_agreement": _rounded(helpfulness_agreement),
        "polarized": polarized,
    }
    if value is None:
        return Criterion(
            name="rating_reliability",
            value=None,
            direction=DIRECTION_BENEFIT,
            description=(
                "No rating-reliability component is available (no ratings, no "
                "rating/text pairs and no predictions)."
            ),
            confidence=0.0,
            components_used=(),
            weights={},
            missing_components=missing,
            missing_reason="all_components_missing",
            extras=extras,
        )
    description = (
        "; ".join(notes)
        + f". Weighted mean {value:.4f} over {len(used)} of {len(parts)} components "
        f"(confidence {confidence:.2f})."
    )
    return Criterion(
        name="rating_reliability",
        value=value,
        direction=DIRECTION_BENEFIT,
        description=description,
        confidence=confidence,
        components_used=used,
        weights=weights_used,
        missing_components=missing,
        extras=extras,
    )


def negative_feedback_concentration_criterion(
    *,
    n_predictions: int,
    negative_share: float | None,
    prior_negative_share: float,
    top_pain_point_share: float | None,
    aspect_concentration: float | None,
    temporal_burst: float | None,
    config: RiskCriteriaConfig,
) -> Criterion:
    """Negative feedback concentration: prevalence times focus.

    ``prevalence = min(1, shrunk_negative_share / p_saturation)`` and
    ``focus`` = mean of the available raw values among ``top_pain_point_share``,
    ``aspect_concentration`` (the Herfindahl index) and ``temporal_burst`` (the
    largest sliding-window share). ``value = prevalence * (focus_floor +
    (1 - focus_floor) * focus)``, so diffuse negative feedback still counts the
    floor amount. When every focus input is missing (fewer than 10
    predicted-negative reviews) ``focus = 0`` and ``focus_unknown`` is recorded;
    the value is still reported because prevalence alone is real.

    Args:
        n_predictions: Reviews carrying a stored prediction.
        negative_share: Share of those reviews predicted negative.
        prior_negative_share: Train-split negative-share prior.
        top_pain_point_share: Stored raw top-pain-point share (None = missing).
        aspect_concentration: Stored raw Herfindahl index (None = missing).
        temporal_burst: Stored raw burst share (None = missing).
        config: Risk criteria configuration.

    Returns:
        The ``negative_feedback_concentration`` criterion (cost).
    """
    parameters = config.negative_feedback_concentration
    m = config.shrinkage_m
    if negative_share is None or n_predictions <= 0:
        return Criterion(
            name="negative_feedback_concentration",
            value=None,
            direction=DIRECTION_COST,
            description=(
                "No review of the product carries a stored sentiment prediction, so "
                "the negative-feedback prevalence is undefined."
            ),
            confidence=0.0,
            missing_reason="no_predictions",
            extras={"n_predictions": n_predictions},
        )

    shrunk = shrink(negative_share, n_predictions, prior_negative_share, m)
    prevalence = min(1.0, shrunk / parameters.p_saturation)
    focus_inputs = {
        "top_pain_point_share": top_pain_point_share,
        "aspect_concentration": aspect_concentration,
        "temporal_burst": temporal_burst,
    }
    available_focus = {
        name: _clamp_unit(value)
        for name, value in focus_inputs.items()
        if value is not None
    }
    focus_unknown = not available_focus
    focus = (
        sum(available_focus.values()) / len(available_focus) if available_focus else 0.0
    )
    value = prevalence * (
        parameters.focus_floor + (1.0 - parameters.focus_floor) * focus
    )
    parts = (
        f"shrunken negative share {shrunk:.4f} from {n_predictions:,} predictions "
        f"(prior {prior_negative_share:.4f}, m={m:g}) gives prevalence "
        f"{prevalence:.4f} against p_saturation {parameters.p_saturation:g}; "
    )
    if focus_unknown:
        parts += (
            "no concentration input is available (fewer than 10 predicted-negative "
            "reviews), so focus_unknown is recorded and focus is 0; "
        )
    else:
        parts += (
            "focus "
            f"{focus:.4f} over {len(available_focus)} of 3 inputs "
            f"({', '.join(sorted(available_focus))}); "
        )
    parts += (
        f"value {_clamp_unit(value):.4f} with focus_floor "
        f"{parameters.focus_floor:g}."
    )
    return Criterion(
        name="negative_feedback_concentration",
        value=_clamp_unit(value),
        direction=DIRECTION_COST,
        description=parts,
        confidence=1.0,
        components_used=("negative_share",),
        weights={"negative_share": 1.0},
        extras={
            "n_predictions": n_predictions,
            "shrunk_negative_share": _rounded(shrunk),
            "prevalence": _rounded(prevalence),
            "focus": _rounded(focus),
            "focus_components": sorted(available_focus),
            "focus_unknown": focus_unknown,
            "p_saturation": parameters.p_saturation,
            "focus_floor": parameters.focus_floor,
        },
    )


def price_position_criterion(
    percentile: float | None,
    *,
    price_missing: bool,
    peer_group: str | None,
    peer_count: int | None,
    peer_fallback: bool | None,
    config: RiskCriteriaConfig,
) -> Criterion:
    """Price position from the stored category price percentile (cost).

    Args:
        percentile: ``price_percentile_category`` (0 = cheapest, 1 = priciest).
        price_missing: Stored missing-price flag.
        peer_group: Stored price peer group.
        peer_count: Stored number of priced products in the group.
        peer_fallback: Stored fallback flag (weaker comparison).
        config: Risk criteria configuration.

    Returns:
        The ``price_position`` criterion; ``None`` (never imputed) without a price.
    """
    extras: dict[str, Any] = {
        "price_missing": price_missing,
        "price_peer_group": peer_group,
        "price_peer_count": peer_count,
        "price_peer_fallback": peer_fallback,
    }
    if percentile is None:
        return Criterion(
            name="price_position",
            value=None,
            direction=DIRECTION_COST,
            description=(
                "The product has no usable price percentile, so the price position "
                "is unknown and is not imputed."
            ),
            confidence=0.0,
            missing_reason="price_missing" if price_missing else "price_percentile_missing",
            extras=extras,
        )
    fallback_confidence = config.price_position.fallback_confidence
    confidence = 1.0 if not peer_fallback else fallback_confidence
    where = (
        f"peer group {peer_group!r} ({peer_count:,} priced products)"
        if peer_group is not None and peer_count is not None
        else "its price peer group"
    )
    fallback_note = (
        f"; the group is a fallback (confidence {confidence:.2f})"
        if peer_fallback
        else ""
    )
    return Criterion(
        name="price_position",
        value=_clamp_unit(percentile),
        direction=DIRECTION_COST,
        description=(
            f"Price percentile {_clamp_unit(percentile):.4f} within {where} "
            f"(0 = cheapest, 1 = priciest){fallback_note}."
        ),
        confidence=confidence,
        components_used=("price_percentile_category",),
        weights={"price_percentile_category": 1.0},
        extras=extras,
    )


def criteria_vector(criteria: Mapping[str, Criterion]) -> tuple[float | None, ...]:
    """The five criterion values in the fixed TOPSIS vector order.

    Args:
        criteria: Computed criteria keyed by name.

    Returns:
        ``(product_quality, evidence_strength, rating_reliability,
        negative_feedback_concentration, price_position)``; a missing criterion
        stays ``None`` (never 0).
    """
    return tuple(
        criteria[name].value if name in criteria else None for name in CRITERION_ORDER
    )


def read_criteria_directions(path: Path) -> dict[str, str]:
    """Read and check the five criterion directions from the weights config.

    The directions are never assumed in code: they are read from
    ``configs/criteria_weights.yaml`` and compared against
    :data:`EXPECTED_DIRECTIONS`, so a changed orientation fails loudly instead of
    silently flipping a criterion.

    Args:
        path: Path to ``criteria_weights.yaml``.

    Returns:
        Criterion name -> ``benefit``/``cost`` for the five criteria.

    Raises:
        DataError: If the file is missing or unparsable, a criterion is absent,
            a direction is not ``benefit``/``cost``, or a direction disagrees
            with :data:`EXPECTED_DIRECTIONS`.
    """
    if not path.exists():
        raise DataError(f"criteria weights file not found at {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DataError(f"could not read criteria weights file at {path}: {exc}") from exc
    configured = raw.get("criteria") if isinstance(raw, dict) else None
    if not isinstance(configured, dict):
        raise DataError(f"criteria weights file at {path} declares no 'criteria' mapping")

    directions: dict[str, str] = {}
    problems: list[str] = []
    for name in CRITERION_ORDER:
        body = configured.get(name)
        if not isinstance(body, dict):
            problems.append(f"criterion {name!r} is missing")
            continue
        direction = body.get("type")
        if direction not in (DIRECTION_BENEFIT, DIRECTION_COST):
            problems.append(
                f"criterion {name!r} has direction {direction!r}; expected "
                "'benefit' or 'cost'"
            )
            continue
        directions[name] = str(direction)
    if problems:
        raise DataError("; ".join(problems))

    mismatched = {
        name: (directions[name], EXPECTED_DIRECTIONS[name])
        for name in CRITERION_ORDER
        if directions[name] != EXPECTED_DIRECTIONS[name]
    }
    if mismatched:
        detail = ", ".join(
            f"{name}: file={declared!r} expected={expected!r}"
            for name, (declared, expected) in mismatched.items()
        )
        raise DataError(
            f"criteria weights file at {path} declares directions that contradict "
            f"the risk criteria: {detail}"
        )
    return directions


def product_sentiment_totals(reviews: pd.DataFrame) -> pd.DataFrame:
    """Per-product prediction counts, mean net sentiment and negative share.

    Uses only reviews that carry a stored prediction with both probabilities;
    a review without a prediction is excluded from the counts, never imputed.

    Args:
        reviews: Frame with ``asin``, ``predicted_label``, ``prob_negative`` and
            ``prob_positive`` columns.

    Returns:
        Frame with one row per product that has at least one prediction:
        ``asin``, ``n_predictions``, ``mean_net`` and ``negative_share``.
    """
    required = {"asin", "predicted_label", "prob_negative", "prob_positive"}
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"reviews frame lacks column(s) {sorted(missing)}")
    usable = (
        reviews["predicted_label"].notna()
        & reviews["prob_negative"].notna()
        & reviews["prob_positive"].notna()
    )
    frame = pd.DataFrame(
        {
            "asin": reviews.loc[usable, "asin"].astype(str),
            "net": reviews.loc[usable, "prob_positive"].astype(float)
            - reviews.loc[usable, "prob_negative"].astype(float),
            "negative": reviews.loc[usable, "predicted_label"].astype(float) == 0.0,
        }
    )
    if frame.empty:
        return pd.DataFrame(
            {
                "asin": pd.Series(dtype="object"),
                "n_predictions": pd.Series(dtype="int64"),
                "mean_net": pd.Series(dtype="float64"),
                "negative_share": pd.Series(dtype="float64"),
            }
        )
    grouped = frame.groupby("asin", sort=True)
    return grouped.agg(
        n_predictions=("net", "size"),
        mean_net=("net", "mean"),
        negative_share=("negative", "mean"),
    ).reset_index()


def category_priors(products: pd.DataFrame, sentiment: pd.DataFrame) -> dict[str, Any]:
    """Train-split-only category priors used by the shrinkage.

    Each prior is a **product-macro mean**: every TRAIN product contributes one
    value, so a product with thousands of reviews cannot dominate the prior.
    Validation and test products never influence it.

    Args:
        products: Frame with ``asin``, ``split`` and ``mean_rating`` columns.
        sentiment: Output of :func:`product_sentiment_totals`.

    Returns:
        JSON-safe mapping with the three priors and the train counts.

    Raises:
        ValueError: If an input frame lacks a required column.
        DataError: If the train split has no usable rating or prediction values
            (a prior cannot be fabricated).
    """
    required_products = {"asin", "split", "mean_rating"}
    required_sentiment = {"asin", "n_predictions", "mean_net", "negative_share"}
    missing_products = required_products - set(products.columns)
    missing_sentiment = required_sentiment - set(sentiment.columns)
    if missing_products:
        raise ValueError(f"products frame lacks column(s) {sorted(missing_products)}")
    if missing_sentiment:
        raise ValueError(f"sentiment frame lacks column(s) {sorted(missing_sentiment)}")

    train = products[products["split"].astype(str) == "train"]
    train_asins = set(train["asin"].astype(str))
    ratings = train["mean_rating"].dropna().astype(float)
    predicted = sentiment[
        sentiment["asin"].astype(str).isin(train_asins)
        & (sentiment["n_predictions"] > 0)
    ]
    nets = predicted["mean_net"].dropna().astype(float)
    negative_shares = predicted["negative_share"].dropna().astype(float)
    if ratings.empty:
        raise DataError(
            "the train split has no product with a mean rating; the rating prior "
            "cannot be computed"
        )
    if nets.empty or negative_shares.empty:
        raise DataError(
            "the train split has no product with stored predictions; the sentiment "
            "priors cannot be computed"
        )
    return {
        "split": "train",
        "n_train_products": int(len(train)),
        "n_train_with_rating": int(ratings.size),
        "n_train_with_predictions": int(nets.size),
        "mean_rating": _rounded(float(ratings.mean())),
        "net_sentiment": _rounded(float(nets.mean())),
        "negative_share": _rounded(float(negative_shares.mean())),
    }


__all__ = [
    "CRITERION_ORDER",
    "DIRECTION_BENEFIT",
    "DIRECTION_COST",
    "EXPECTED_DIRECTIONS",
    "Criterion",
    "category_priors",
    "criteria_vector",
    "evidence_strength_criterion",
    "negative_feedback_concentration_criterion",
    "price_position_criterion",
    "product_quality_criterion",
    "product_sentiment_totals",
    "rating_reliability_criterion",
    "read_criteria_directions",
    "shrink",
]

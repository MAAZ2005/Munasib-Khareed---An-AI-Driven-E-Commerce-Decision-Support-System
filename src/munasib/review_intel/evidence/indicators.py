"""Pure per-product evidence indicators (every formula is in docs/evidence.md).

This module holds the arithmetic only: functions take pandas series/scalars and
return :class:`Indicator` values. They read no database, no file and no clock -
the reference date is always passed in - so the whole stage is deterministic and
testable on synthetic frames.

Every indicator carries:

* ``raw_value`` - the measured quantity in its natural unit (or None);
* ``normalized`` - 0-1 with 1.0 = best for the consumer (or None);
* ``direction`` - ``benefit`` (higher raw is better) or ``cost`` (higher raw is
  worse), matching the API's ``IndicatorDirection`` values;
* ``description`` - one sentence citing the numbers it was computed from;
* ``extras`` - the supporting counts/shares needed to audit it;
* ``missing_reason`` - the flag name when the value could not be computed.

A None indicator is never imputed: it is dropped from the composite and the
remaining weights are renormalized (see :func:`composite_strength`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from munasib.core.config import EvidenceIndicatorsConfig, EvidenceStrengthConfig
from munasib.core.constants import (
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABEL_POSITIVE,
)
from munasib.review_intel.aspects.types import ASPECT_PAIN_POINT

#: Stored direction values (identical to the API's ``IndicatorDirection``).
DIRECTION_BENEFIT = "benefit"
DIRECTION_COST = "cost"

#: Composite components in storage/report order (docs/evidence.md).
STRENGTH_COMPONENT_KEYS: tuple[str, ...] = (
    "review_volume",
    "recency_coverage",
    "verified_purchase_ratio",
    "rating_agreement",
    "rating_text_consistency",
    "helpfulness_coverage",
)

#: Composite component -> the indicator whose normalized value it consumes.
COMPONENT_INDICATOR: dict[str, str] = {
    "review_volume": "review_volume",
    "recency_coverage": "recency_coverage",
    "verified_purchase_ratio": "verified_purchase_ratio",
    "rating_agreement": "rating_agreement",
    "rating_text_consistency": "rating_text_agreement",
    "helpfulness_coverage": "helpfulness_coverage",
}

#: Mean Gregorian month length in seconds (365.2425 / 12 days), the unit of the
#: staleness indicator (fixed duration, so no calendar arithmetic is involved).
MONTH_SECONDS = 30.436875 * 86400.0

#: Decimal places kept in every stored float (deterministic, compact JSON).
DIGITS = 6


def _round(value: float) -> float:
    """Round one stored float to :data:`DIGITS` decimal places.

    Args:
        value: Raw float.

    Returns:
        The rounded value.
    """
    return round(float(value), DIGITS)


def _optional(value: float | None) -> float | None:
    """Round a float, preserving None (missing stays missing).

    Args:
        value: Raw float or None.

    Returns:
        The rounded float, or None.
    """
    return None if value is None else _round(value)


def _clamp_unit(value: float) -> float:
    """Clamp a float into ``[0.0, 1.0]``.

    Args:
        value: Candidate value.

    Returns:
        The clamped value.
    """
    return max(0.0, min(1.0, float(value)))


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce a column to float, turning non-numeric entries into NaN.

    Args:
        series: Source column.

    Returns:
        Float series (missing stays NaN).
    """
    return pd.to_numeric(series, errors="coerce").astype("float64")


@dataclass(frozen=True)
class Indicator:
    """One evidence indicator: the value, its provenance and its audit extras."""

    name: str
    raw_value: float | None
    normalized: float | None
    direction: str
    description: str
    extras: dict[str, object] = field(default_factory=dict)
    missing_reason: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Render the stored JSON entry (stable keys, rounded floats).

        Returns:
            Mapping with ``raw_value``, ``normalized``, ``direction``,
            ``description``, ``missing_reason`` and ``extras``.
        """
        return {
            "raw_value": _optional(self.raw_value),
            "normalized": _optional(self.normalized),
            "direction": self.direction,
            "description": self.description,
            "missing_reason": self.missing_reason,
            "extras": dict(self.extras),
        }


@dataclass(frozen=True)
class StrengthResult:
    """Composite evidence strength plus the components it actually used."""

    value: float | None
    components_used: tuple[str, ...]
    weights: dict[str, float]
    missing: dict[str, str]


# ------------------------------------------------------------------------------
# Volume
# ------------------------------------------------------------------------------


def review_volume_sufficiency(n_reviews: int, n_saturation: int) -> float:
    """Log-scaled review-count sufficiency in ``[0, 1]``.

    ``min(1, log(1 + n) / log(1 + n_saturation))`` - the first reviews matter
    much more than the one-hundredth, and the score saturates at
    ``n_saturation``.

    Args:
        n_reviews: Number of reviews.
        n_saturation: Review count at which the score reaches 1.0.

    Returns:
        Sufficiency in ``[0, 1]`` (0 when there are no reviews).
    """
    if n_reviews <= 0:
        return 0.0
    denominator = math.log1p(max(int(n_saturation), 2))
    return _clamp_unit(math.log1p(n_reviews) / denominator)


def volume_indicator(n_reviews: int, config: EvidenceIndicatorsConfig) -> Indicator:
    """Build the review-volume indicator.

    A product with no stored reviews has no computable volume evidence either
    (sufficiency is only meaningful relative to an actual review set), so the
    value is None with reason ``no_reviews`` rather than a fabricated 0.0.

    Args:
        n_reviews: Number of stored reviews of the product.
        config: Indicator configuration.

    Returns:
        ``review_volume`` (benefit; raw = review count).
    """
    if n_reviews <= 0:
        return Indicator(
            name="review_volume",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_BENEFIT,
            description=(
                "The product has no stored reviews, so there is no volume evidence."
            ),
            extras={"n_reviews": 0, "n_saturation": config.n_saturation},
            missing_reason="no_reviews",
        )
    normalized = review_volume_sufficiency(n_reviews, config.n_saturation)
    description = (
        f"{n_reviews:,} reviews for this product; the log-scaled sufficiency score "
        f"reaches 1.0 at {config.n_saturation:,} reviews, giving {normalized:.4f} here."
    )
    return Indicator(
        name="review_volume",
        raw_value=float(n_reviews),
        normalized=normalized,
        direction=DIRECTION_BENEFIT,
        description=description,
        extras={"n_reviews": n_reviews, "n_saturation": config.n_saturation},
    )


# ------------------------------------------------------------------------------
# Rating distribution
# ------------------------------------------------------------------------------


def star_counts(ratings: pd.Series) -> list[int]:
    """Count reviews per star (1-5), ignoring missing ratings.

    Args:
        ratings: Star ratings (may contain NaN).

    Returns:
        Five counts, index 0 = one star.
    """
    rated = _numeric(ratings).dropna()
    return [int((rated == star).sum()) for star in range(1, 6)]


def normalized_entropy(counts: Sequence[int]) -> tuple[float, float]:
    """Shannon entropy of a count vector and its normalisation by ``log2(5)``.

    Args:
        counts: Non-negative counts.

    Returns:
        ``(entropy_bits, entropy_normalized)``; both 0.0 for an empty vector.
    """
    total = sum(counts)
    if total <= 0:
        return 0.0, 0.0
    entropy = -sum((count / total) * math.log2(count / total) for count in counts if count > 0)
    return entropy, _clamp_unit(entropy / math.log2(5))


def rating_agreement_indicator(
    ratings: pd.Series, config: EvidenceIndicatorsConfig
) -> Indicator:
    """Build the rating-agreement indicator from the star histogram.

    ``rating_agreement = 1 - H / log2(5)`` (1.0 = every review gives the same
    star). The full histogram, mean rating and polarization flag travel in the
    extras; the polarization rule is ``share(1-2 stars) >= min_share AND
    share(5 stars) >= min_share``.

    Args:
        ratings: Star ratings (may contain NaN).
        config: Indicator configuration.

    Returns:
        ``rating_agreement`` (benefit; raw = agreement).
    """
    counts = star_counts(ratings)
    total = sum(counts)
    if total <= 0:
        return Indicator(
            name="rating_agreement",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_BENEFIT,
            description="The product has no rated reviews, so rating agreement is undefined.",
            extras={"star_counts": {}, "star_shares": {}, "polarized": False},
            missing_reason="no_ratings",
        )
    shares = [count / total for count in counts]
    entropy, entropy_normalized = normalized_entropy(counts)
    agreement = 1.0 - entropy_normalized
    low_star_share = shares[0] + shares[1]
    high_star_share = shares[4]
    polarized = (
        low_star_share >= config.polarization_min_share
        and high_star_share >= config.polarization_min_share
    )
    mean_rating = float(_numeric(ratings).dropna().mean())
    split = "/".join(str(count) for count in counts)
    description = (
        f"Rating agreement is {agreement:.4f}: the {split} (1-5 star) split has "
        f"{entropy:.4f} bits of entropy ({entropy_normalized:.4f} of log2(5)) and a mean "
        f"rating of {mean_rating:.2f}. The 1-2 star share is {low_star_share:.1%} and the "
        f"5 star share {high_star_share:.1%}, so the distribution is "
        f"{'polarized' if polarized else 'not polarized'} (both must reach "
        f"{config.polarization_min_share:.0%})."
    )
    return Indicator(
        name="rating_agreement",
        raw_value=agreement,
        normalized=agreement,
        direction=DIRECTION_BENEFIT,
        description=description,
        extras={
            "star_counts": {str(star): counts[star - 1] for star in range(1, 6)},
            "star_shares": {str(star): _round(shares[star - 1]) for star in range(1, 6)},
            "mean_rating": _round(mean_rating),
            "entropy_bits": _round(entropy),
            "entropy_normalized": _round(entropy_normalized),
            "rated_reviews": total,
            "polarized": polarized,
        },
    )


# ------------------------------------------------------------------------------
# Sentiment shares and helpfulness
# ------------------------------------------------------------------------------


def sentiment_shares_indicator(
    predicted: pd.Series, net: pd.Series, ratings: pd.Series
) -> Indicator:
    """Build the predicted-sentiment-share indicator.

    Shares use reviews with a stored prediction; ``net`` is
    ``prob_positive - prob_negative``. The star-derived negative share
    (rating <= 2) is reported separately because the weak labels exist for every
    rated review, not only for the predicted ones.

    Args:
        predicted: Predicted label codes (NaN where no prediction exists).
        net: Per-review net sentiment ``prob_positive - prob_negative``.
        ratings: Star ratings.

    Returns:
        ``sentiment_shares`` (benefit; raw = mean net sentiment in ``[-1, 1]``).
    """
    valid = predicted.notna() & net.notna()
    total = int(valid.sum())
    if total == 0:
        return Indicator(
            name="sentiment_shares",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_BENEFIT,
            description="No review of the product carries a stored sentiment prediction.",
            extras={"predictions": 0},
            missing_reason="no_predictions",
        )
    labels = predicted[valid].astype(int)
    negative_share = float((labels == SENTIMENT_LABEL_NEGATIVE).mean())
    neutral_share = float((labels == SENTIMENT_LABEL_NEUTRAL).mean())
    positive_share = float((labels == SENTIMENT_LABEL_POSITIVE).mean())
    mean_net = float(net[valid].mean())
    normalized = _clamp_unit((mean_net + 1.0) / 2.0)
    rated = _numeric(ratings).dropna()
    star_negative_share: float | None = (
        None if rated.empty else float((rated <= 2).mean())
    )
    star_note = (
        "undefined (no rated reviews)"
        if star_negative_share is None
        else f"{star_negative_share:.1%} (weak labels)"
    )
    description = (
        f"{total:,} reviews carry a prediction: {negative_share:.1%} negative, "
        f"{neutral_share:.1%} neutral and {positive_share:.1%} positive; the mean net "
        f"sentiment is {mean_net:.4f} (normalised {normalized:.4f}). The star-derived "
        f"negative share is {star_note}."
    )
    return Indicator(
        name="sentiment_shares",
        raw_value=mean_net,
        normalized=normalized,
        direction=DIRECTION_BENEFIT,
        description=description,
        extras={
            "predictions": total,
            "predicted_negative_share": _round(negative_share),
            "predicted_neutral_share": _round(neutral_share),
            "predicted_positive_share": _round(positive_share),
            "mean_net_sentiment": _round(mean_net),
            "star_negative_share": _optional(star_negative_share),
        },
    )


def helpfulness_indicators(
    helpful_votes: pd.Series, net: pd.Series, has_prediction: pd.Series
) -> tuple[Indicator, Indicator]:
    """Build the helpfulness coverage and helpfulness-weighted sentiment.

    Coverage is the share of all reviews with at least one helpful vote. The
    weighted net sentiment uses ``1 + helpful_votes`` as the weight (so every
    review counts at least once) over the reviews with a prediction, and the
    difference from the unweighted mean is stored alongside it.

    Args:
        helpful_votes: Helpful-vote counts (missing means 0 in the data; NaN is
            treated as 0 here too).
        net: Per-review net sentiment.
        has_prediction: Boolean mask of reviews with a stored prediction.

    Returns:
        ``(helpfulness_coverage, helpfulness_weighted_sentiment)``.
    """
    votes = _numeric(helpful_votes).fillna(0.0)
    total = len(votes)
    if total == 0:
        coverage = Indicator(
            name="helpfulness_coverage",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_BENEFIT,
            description="The product has no reviews, so helpfulness coverage is undefined.",
            extras={"reviews_with_votes": 0},
            missing_reason="no_reviews",
        )
    else:
        with_votes = int((votes >= 1).sum())
        share = with_votes / total
        coverage = Indicator(
            name="helpfulness_coverage",
            raw_value=share,
            normalized=share,
            direction=DIRECTION_BENEFIT,
            description=(
                f"{with_votes:,} of {total:,} reviews carry at least one helpful vote "
                f"({share:.1%}); the mean helpful-vote count per review is "
                f"{float(votes.mean()):.4f}."
            ),
            extras={
                "reviews_with_votes": with_votes,
                "mean_helpful_votes": _round(float(votes.mean())),
            },
        )
    mask = has_prediction & net.notna()
    if not bool(mask.any()):
        weighted = Indicator(
            name="helpfulness_weighted_sentiment",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_BENEFIT,
            description="No review carries a stored prediction, so weighted sentiment is undefined.",
            extras={"predictions": 0},
            missing_reason="no_predictions",
        )
        return coverage, weighted
    weights = 1.0 + votes[mask]
    values = net[mask]
    weight_total = float(weights.sum())
    weighted_net = float((weights * values).sum() / weight_total)
    unweighted_net = float(values.mean())
    difference = weighted_net - unweighted_net
    normalized = _clamp_unit((weighted_net + 1.0) / 2.0)
    description = (
        f"Net sentiment weighted by (1 + helpful votes) is {weighted_net:.4f} over "
        f"{int(mask.sum()):,} predicted reviews ({weight_total:.0f} total weight), "
        f"{'above' if difference > 0 else 'below' if difference < 0 else 'equal to'} "
        f"the unweighted {unweighted_net:.4f} by {difference:+.4f}."
    )
    weighted = Indicator(
        name="helpfulness_weighted_sentiment",
        raw_value=weighted_net,
        normalized=normalized,
        direction=DIRECTION_BENEFIT,
        description=description,
        extras={
            "predictions": int(mask.sum()),
            "weight_total": _round(weight_total),
            "unweighted_net_sentiment": _round(unweighted_net),
            "difference": _round(difference),
        },
    )
    return coverage, weighted


# ------------------------------------------------------------------------------
# Verified purchases
# ------------------------------------------------------------------------------


def verified_purchase_indicator(verified: pd.Series) -> Indicator:
    """Build the verified-purchase ratio.

    The denominator is the reviews with a **known** verified flag; NULL means
    unknown, not False, so it is excluded from both sides and the known count is
    reported.

    Args:
        verified: Nullable booleans (True/False/None).

    Returns:
        ``verified_purchase_ratio`` (benefit; raw = ratio over known flags).
    """
    known = verified.dropna()
    unknown = int(len(verified) - len(known))
    if known.empty:
        return Indicator(
            name="verified_purchase_ratio",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_BENEFIT,
            description=(
                "Every review of the product has an unknown verified flag, so the "
                "verified-purchase ratio is undefined."
            ),
            extras={"verified_known": 0, "verified_count": 0, "verified_unknown": unknown},
            missing_reason="verified_unknown",
        )
    verified_count = int(known.astype(bool).sum())
    ratio = verified_count / len(known)
    description = (
        f"{verified_count:,} of {len(known):,} reviews with a known verified flag are "
        f"verified purchases ({ratio:.1%}); {unknown:,} reviews carry an unknown flag "
        "and are excluded from both the numerator and the denominator."
    )
    return Indicator(
        name="verified_purchase_ratio",
        raw_value=ratio,
        normalized=ratio,
        direction=DIRECTION_BENEFIT,
        description=description,
        extras={
            "verified_known": int(len(known)),
            "verified_count": verified_count,
            "verified_unknown": unknown,
        },
    )


# ------------------------------------------------------------------------------
# Recency, recent shift and trend
# ------------------------------------------------------------------------------


def _iso(timestamp: pd.Timestamp) -> str:
    """Render a naive UTC pandas timestamp as an ISO 8601 string.

    Args:
        timestamp: Naive UTC timestamp.

    Returns:
        ISO 8601 string with a ``Z`` suffix.
    """
    return f"{pd.Timestamp(timestamp).isoformat()}Z"


def recency_indicators(
    times: pd.Series,
    predicted: pd.Series,
    net: pd.Series,
    has_prediction: pd.Series,
    reference_date: datetime,
    config: EvidenceIndicatorsConfig,
) -> tuple[Indicator, Indicator, Indicator, Indicator]:
    """Build recency coverage, staleness, the recent negative shift and the trend.

    All windows are anchored on ``reference_date`` (the dataset's latest review
    timestamp), never on today's date:

    * ``recency_coverage`` - share of reviews inside ``recent_months``.
    * ``staleness`` - months since the latest review, cost-coded with
      ``1 / (1 + months / staleness_scale_months)``.
    * ``recent_negative_shift`` - recent minus overall predicted-negative share,
      only with at least ``min_recent_reviews`` predicted reviews in the window.
    * ``sentiment_trend`` - OLS slope of quarterly mean net sentiment over the
      ``trend_quarters`` complete quarters before the reference quarter,
      requiring ``min_trend_quarters`` quarters with at least
      ``min_quarter_reviews`` predicted reviews each.

    Args:
        times: Review timestamps (naive UTC).
        predicted: Predicted label codes (NaN = no prediction).
        net: Per-review net sentiment.
        has_prediction: Boolean mask of reviews with a stored prediction.
        reference_date: Dataset reference date (naive UTC).
        config: Indicator configuration.

    Returns:
        ``(recency_coverage, staleness, recent_negative_shift, sentiment_trend)``.
    """
    reference = pd.Timestamp(reference_date)
    window_start = reference - pd.DateOffset(months=config.recent_months)
    recent = (times >= window_start) & (times <= reference)
    total = len(times)
    if total == 0:
        coverage = Indicator(
            name="recency_coverage",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_BENEFIT,
            description="The product has no reviews, so recency coverage is undefined.",
            extras={"recent_reviews": 0, "window_start": _iso(window_start)},
            missing_reason="no_reviews",
        )
        staleness = Indicator(
            name="staleness",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_COST,
            description="The product has no reviews, so staleness is undefined.",
            extras={"last_review_time": None},
            missing_reason="no_reviews",
        )
    else:
        recent_count = int(recent.sum())
        coverage_share = recent_count / total
        coverage = Indicator(
            name="recency_coverage",
            raw_value=coverage_share,
            normalized=coverage_share,
            direction=DIRECTION_BENEFIT,
            description=(
                f"{recent_count:,} of {total:,} reviews ({coverage_share:.1%}) fall in the "
                f"{config.recent_months}-month window starting {_iso(window_start)}, "
                f"anchored on the dataset reference date {_iso(reference)}."
            ),
            extras={
                "recent_reviews": recent_count,
                "window_start": _iso(window_start),
                "reference_date": _iso(reference),
            },
        )
        last_review = pd.Timestamp(times.max())
        months = max(0.0, (reference - last_review).total_seconds() / MONTH_SECONDS)
        normalized = 1.0 / (1.0 + months / config.staleness_scale_months)
        staleness = Indicator(
            name="staleness",
            raw_value=months,
            normalized=normalized,
            direction=DIRECTION_COST,
            description=(
                f"The latest review is {months:.2f} months before the reference date "
                f"{_iso(reference)}; the cost curve gives {normalized:.4f} with a "
                f"{config.staleness_scale_months:g}-month scale."
            ),
            extras={
                "last_review_time": _iso(last_review),
                "staleness_scale_months": config.staleness_scale_months,
            },
        )

    recent_predicted = recent & has_prediction
    recent_predictions = int(recent_predicted.sum())
    if recent_predictions < config.min_recent_reviews:
        shift = Indicator(
            name="recent_negative_shift",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_COST,
            description=(
                f"Only {recent_predictions:,} predicted reviews fall in the recent "
                f"{config.recent_months}-month window (minimum "
                f"{config.min_recent_reviews:,}), so the recent shift is undefined."
            ),
            extras={"recent_predictions": recent_predictions},
            missing_reason="recent_window_below_min_reviews",
        )
    else:
        overall_negative = float((predicted[has_prediction] == SENTIMENT_LABEL_NEGATIVE).mean())
        recent_negative = float(
            (predicted[recent_predicted] == SENTIMENT_LABEL_NEGATIVE).mean()
        )
        delta = recent_negative - overall_negative
        normalized = 1.0 - _clamp_unit(delta)
        shift = Indicator(
            name="recent_negative_shift",
            raw_value=delta,
            normalized=normalized,
            direction=DIRECTION_COST,
            description=(
                f"The recent negative share is {recent_negative:.1%} over "
                f"{recent_predictions:,} predicted reviews versus {overall_negative:.1%} "
                f"overall ({delta:+.4f}); only a worsening shift is penalised, giving "
                f"{normalized:.4f}."
            ),
            extras={
                "recent_negative_share": _round(recent_negative),
                "overall_negative_share": _round(overall_negative),
                "recent_predictions": recent_predictions,
            },
        )

    if total == 0:
        trend = Indicator(
            name="sentiment_trend",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_BENEFIT,
            description="The product has no reviews, so no sentiment trend can be estimated.",
            extras={"quarters_used": []},
            missing_reason="no_reviews",
        )
        return coverage, staleness, shift, trend

    if not bool(has_prediction.any()):
        trend = Indicator(
            name="sentiment_trend",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_BENEFIT,
            description="No review carries a stored prediction, so no sentiment trend can be estimated.",
            extras={"quarters_used": []},
            missing_reason="no_predictions",
        )
        return coverage, staleness, shift, trend

    reference_quarter = reference.year * 4 + (reference.month - 1) // 3
    first_quarter = reference_quarter - config.trend_quarters
    quarter_index = times.dt.year * 4 + (times.dt.month - 1) // 3
    in_window = has_prediction & (quarter_index >= first_quarter) & (
        quarter_index < reference_quarter
    )
    if not bool(in_window.any()):
        trend = Indicator(
            name="sentiment_trend",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_BENEFIT,
            description=(
                f"No predicted review falls in the {config.trend_quarters} complete "
                f"quarters before the reference quarter "
                f"{reference_quarter // 4}-Q{reference_quarter % 4 + 1}, so no trend is reported."
            ),
            extras={"quarters_used": []},
            missing_reason="trend_insufficient_quarters",
        )
        return coverage, staleness, shift, trend

    frame = pd.DataFrame(
        {
            "quarter": quarter_index[in_window].astype(int),
            "net": net[in_window].astype(float),
        }
    )
    grouped = frame.groupby("quarter")["net"].agg(["mean", "count"])
    qualifying = grouped[grouped["count"] >= config.min_quarter_reviews]
    if len(qualifying) < max(2, config.min_trend_quarters):
        trend = Indicator(
            name="sentiment_trend",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_BENEFIT,
            description=(
                f"Only {len(qualifying)} of the {config.trend_quarters} quarters before the "
                f"reference quarter hold at least {config.min_quarter_reviews:,} predicted "
                f"reviews (minimum {config.min_trend_quarters}), so no trend is reported."
            ),
            extras={
                "quarters_used": [],
                "qualifying_quarters": int(len(qualifying)),
            },
            missing_reason="trend_insufficient_quarters",
        )
        return coverage, staleness, shift, trend

    x = qualifying.index.to_numpy(dtype=float) - float(first_quarter)
    y = qualifying["mean"].to_numpy(dtype=float)
    x_centred = x - x.mean()
    denominator = float((x_centred**2).sum())
    slope = float((x_centred * (y - y.mean())).sum() / denominator) if denominator else 0.0
    normalized = 0.5 + 0.5 * max(-1.0, min(1.0, slope / config.trend_saturation))
    quarter_keys = qualifying.index.to_numpy(dtype=int)
    quarter_counts = qualifying["count"].to_numpy(dtype=int)
    quarter_means = qualifying["mean"].to_numpy(dtype=float)
    quarters = [
        {
            "quarter": f"{int(quarter) // 4}-Q{int(quarter) % 4 + 1}",
            "reviews": int(count),
            "mean_net_sentiment": _round(float(mean)),
        }
        for quarter, count, mean in zip(
            quarter_keys, quarter_counts, quarter_means, strict=True
        )
    ]
    trend = Indicator(
        name="sentiment_trend",
        raw_value=slope,
        normalized=_clamp_unit(normalized),
        direction=DIRECTION_BENEFIT,
        description=(
            f"The quarterly mean net sentiment changes by {slope:+.4f} per quarter over "
            f"{len(qualifying)} qualifying quarters ({quarters[0]['quarter']} to "
            f"{quarters[-1]['quarter']}); a slope of "
            f"{config.trend_saturation:g} saturates the score at {_clamp_unit(normalized):.4f}."
        ),
        extras={
            "quarters_used": quarters,
            "slope_per_quarter": _round(slope),
        },
    )
    return coverage, staleness, shift, trend


# ------------------------------------------------------------------------------
# Negative-feedback concentration inputs
# ------------------------------------------------------------------------------


def concentration_indicators(
    aspect_negative_counts: pd.Series,
    aspect_classifications: pd.Series,
    n_negative_reviews: int,
    config: EvidenceIndicatorsConfig,
) -> tuple[Indicator, Indicator]:
    """Build the aspect-HHI and top-pain-point-share inputs.

    Both are computed over **negative-review aspect mentions**: a product-aspect
    row's ``negative_count`` is the number of predicted-negative reviews that
    mention the aspect, so a review mentioning two aspects counts in both. The
    Herfindahl index is ``sum((mentions_i / total_mentions) ** 2)``; the top
    share sums the ``top_pain_points`` pain-point aspects with the most negative
    mentions. They are reported raw and are not composite components; the risk
    stage maps them to its criterion in a later stage.

    Args:
        aspect_negative_counts: Negative mention count per aspect (index = aspect).
        aspect_classifications: Stored classification per aspect (same index).
        n_negative_reviews: Predicted-negative reviews of the product.
        config: Indicator configuration.

    Returns:
        ``(aspect_concentration, top_pain_point_share)``, both cost-coded with
        ``normalized = 1 - raw``.
    """
    if n_negative_reviews < config.min_negative_reviews:
        reason = "below_min_negative_reviews"
        description = (
            f"The product has {n_negative_reviews:,} predicted-negative reviews "
            f"(minimum {config.min_negative_reviews:,}), so this input is undefined."
        )
        return (
            Indicator(
                name="aspect_concentration",
                raw_value=None,
                normalized=None,
                direction=DIRECTION_COST,
                description=description,
                extras={"negative_reviews": n_negative_reviews},
                missing_reason=reason,
            ),
            Indicator(
                name="top_pain_point_share",
                raw_value=None,
                normalized=None,
                direction=DIRECTION_COST,
                description=description,
                extras={"negative_reviews": n_negative_reviews},
                missing_reason=reason,
            ),
        )

    counts = aspect_negative_counts[_numeric(aspect_negative_counts) > 0].astype(float)
    total_mentions = float(counts.sum())
    if total_mentions <= 0:
        description = (
            f"The product has {n_negative_reviews:,} predicted-negative reviews but none of "
            "its stored aspects carries a negative mention."
        )
        return (
            Indicator(
                name="aspect_concentration",
                raw_value=None,
                normalized=None,
                direction=DIRECTION_COST,
                description=description,
                extras={"negative_reviews": n_negative_reviews, "negative_mentions": 0},
                missing_reason="no_negative_mentions",
            ),
            Indicator(
                name="top_pain_point_share",
                raw_value=None,
                normalized=None,
                direction=DIRECTION_COST,
                description=description,
                extras={"negative_reviews": n_negative_reviews, "negative_mentions": 0},
                missing_reason="no_negative_mentions",
            ),
        )

    shares = counts / total_mentions
    hhi = float((shares**2).sum())
    concentration = Indicator(
        name="aspect_concentration",
        raw_value=hhi,
        normalized=1.0 - _clamp_unit(hhi),
        direction=DIRECTION_COST,
        description=(
            f"The {int(total_mentions):,} negative-review aspect mentions are spread over "
            f"{len(counts):,} aspects with a Herfindahl index of {hhi:.4f} "
            f"(1.0 = every mention on one aspect); the inverted value is "
            f"{1.0 - _clamp_unit(hhi):.4f}."
        ),
        extras={
            "negative_reviews": n_negative_reviews,
            "negative_mentions": int(total_mentions),
            "aspects_with_negative_mentions": int(len(counts)),
        },
    )

    classifications = aspect_classifications.reindex(counts.index)
    pain = counts[classifications == ASPECT_PAIN_POINT].rename_axis("aspect")
    top = (
        pain.rename("negative_count")
        .reset_index()
        .sort_values(["negative_count", "aspect"], ascending=[False, True])
        .head(config.top_pain_points)
    )
    top_total = float(top["negative_count"].sum()) if not top.empty else 0.0
    top_share = top_total / total_mentions
    top_payload = [
        {"aspect": str(row["aspect"]), "negative_mentions": int(row["negative_count"])}
        for _, row in top.iterrows()
    ]
    top_indicator = Indicator(
        name="top_pain_point_share",
        raw_value=top_share,
        normalized=1.0 - _clamp_unit(top_share),
        direction=DIRECTION_COST,
        description=(
            f"The top {config.top_pain_points} pain-point aspects cover {top_total:.0f} of "
            f"the {int(total_mentions):,} negative-review aspect mentions "
            f"({top_share:.1%}); no pain point exists when the list is empty."
        ),
        extras={
            "negative_reviews": n_negative_reviews,
            "negative_mentions": int(total_mentions),
            "top_pain_points": top_payload,
        },
    )
    return concentration, top_indicator


def burst_indicator(
    negative_times: pd.Series,
    n_negative_reviews: int,
    config: EvidenceIndicatorsConfig,
) -> Indicator:
    """Build the sliding-window temporal-burst indicator.

    For every predicted-negative review time ``t`` the window is
    ``[t, t + burst_window_months)``; the indicator is the largest number of the
    product's predicted-negative reviews inside one such window divided by the
    product's total predicted-negative reviews. Ties resolve to the earliest
    window start. A fixed calendar bucketing would split a March-August run
    across two halves, which is exactly what a sliding window avoids.

    Args:
        negative_times: Review times of the predicted-negative reviews.
        n_negative_reviews: Total predicted-negative reviews of the product.
        config: Indicator configuration.

    Returns:
        ``temporal_burst`` (cost; raw = largest window share).
    """
    if n_negative_reviews < config.min_negative_reviews:
        return Indicator(
            name="temporal_burst",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_COST,
            description=(
                f"The product has {n_negative_reviews:,} predicted-negative reviews "
                f"(minimum {config.min_negative_reviews:,}), so the burst share is undefined."
            ),
            extras={"negative_reviews": n_negative_reviews, "flag": False},
            missing_reason="below_min_negative_reviews",
        )
    times = negative_times.dropna().sort_values().reset_index(drop=True)
    total = len(times)
    if total == 0:
        return Indicator(
            name="temporal_burst",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_COST,
            description="No predicted-negative review timestamp is available.",
            extras={"negative_reviews": 0, "flag": False},
            missing_reason="no_negative_timestamps",
        )
    values = times.to_numpy(dtype="datetime64[ns]")
    window_ends = (times + pd.DateOffset(months=config.burst_window_months)).to_numpy(
        dtype="datetime64[ns]"
    )
    end_positions = np.searchsorted(values, window_ends, side="left")
    counts = end_positions - np.arange(total)
    best_index = int(np.argmax(counts))
    best_count = int(counts[best_index])
    share = best_count / total
    flag = share >= config.burst_flag_share
    start = pd.Timestamp(values[best_index])
    end = pd.Timestamp(window_ends[best_index])
    normalized = 1.0 - _clamp_unit(share)
    return Indicator(
        name="temporal_burst",
        raw_value=share,
        normalized=normalized,
        direction=DIRECTION_COST,
        description=(
            f"{best_count:,} of {total:,} predicted-negative reviews ({share:.1%}) fall inside "
            f"the sliding {config.burst_window_months}-month window starting {_iso(start)}; "
            f"the product is {'flagged' if flag else 'not flagged'} as a burst "
            f"(threshold {config.burst_flag_share:.0%})."
        ),
        extras={
            "negative_reviews": total,
            "window_start": _iso(start),
            "window_end": _iso(end),
            "window_count": best_count,
            "window_months": config.burst_window_months,
            "flag": flag,
        },
    )


# ------------------------------------------------------------------------------
# Rating / text agreement
# ------------------------------------------------------------------------------


def rating_text_agreement_indicator(
    ratings: pd.Series,
    predicted: pd.Series,
    prob_positive: pd.Series,
    prob_negative: pd.Series,
    config: EvidenceIndicatorsConfig,
) -> Indicator:
    """Build the rating/text conflict indicator.

    A review conflicts when its stars and its text prediction strongly disagree:
    1-2 stars with ``prob_positive > threshold``, or 4-5 stars with
    ``prob_negative > threshold``. The raw value is the conflict rate; the
    normalized value is ``1 - min(1, rate / conflict_saturation)`` so 0.0 means
    the conflict rate saturated.

    Args:
        ratings: Star ratings.
        predicted: Predicted label codes.
        prob_positive: Predicted positive probabilities.
        prob_negative: Predicted negative probabilities.
        config: Indicator configuration.

    Returns:
        ``rating_text_agreement`` (cost; raw = conflict rate).
    """
    valid = (
        predicted.notna()
        & ratings.notna()
        & prob_positive.notna()
        & prob_negative.notna()
    )
    evaluated = int(valid.sum())
    if evaluated == 0:
        return Indicator(
            name="rating_text_agreement",
            raw_value=None,
            normalized=None,
            direction=DIRECTION_COST,
            description=(
                "No review carries both a star rating and a stored prediction, so the "
                "conflict rate is undefined."
            ),
            extras={"conflicts": 0, "evaluated": 0},
            missing_reason="no_predictions",
        )
    threshold = config.conflict_probability_threshold
    stars = _numeric(ratings)[valid]
    positive_probability = _numeric(prob_positive)[valid]
    negative_probability = _numeric(prob_negative)[valid]
    conflict = ((stars <= 2) & (positive_probability > threshold)) | (
        (stars >= 4) & (negative_probability > threshold)
    )
    conflicts = int(conflict.sum())
    rate = conflicts / evaluated
    consistency = 1.0 - min(1.0, rate / config.conflict_saturation)
    return Indicator(
        name="rating_text_agreement",
        raw_value=rate,
        normalized=consistency,
        direction=DIRECTION_COST,
        description=(
            f"{conflicts:,} of {evaluated:,} reviews ({rate:.1%}) strongly disagree with "
            f"their stars (1-2 stars with positive probability above {threshold:.2f}, or "
            f"4-5 stars with negative probability above it); consistency is "
            f"{consistency:.4f} and saturates at a {config.conflict_saturation:.0%} "
            "conflict rate."
        ),
        extras={
            "conflicts": conflicts,
            "evaluated": evaluated,
            "conflict_saturation": config.conflict_saturation,
            "probability_threshold": threshold,
        },
    )


# ------------------------------------------------------------------------------
# Composite
# ------------------------------------------------------------------------------


def composite_strength(
    indicators: Mapping[str, Indicator], config: EvidenceStrengthConfig
) -> StrengthResult:
    """Weighted mean of the available normalized components.

    A component with a None normalized value is dropped and the remaining
    weights are renormalized over the components actually used, so a missing
    signal can never be silently counted as zero.

    Args:
        indicators: Every indicator of one product, keyed by name.
        config: Composite configuration (weights + low-evidence guard).

    Returns:
        The composite value (None when no component is available), the used
        component names, their post-renormalization weights and the missing
        components with their reasons.
    """
    weights = config.weights.model_dump()
    missing: dict[str, str] = {}
    available: list[str] = []
    values: dict[str, float] = {}
    for component in STRENGTH_COMPONENT_KEYS:
        indicator = indicators[COMPONENT_INDICATOR[component]]
        if indicator.normalized is None:
            missing[component] = indicator.missing_reason or "unavailable"
        else:
            available.append(component)
            values[component] = float(indicator.normalized)
    if not available:
        return StrengthResult(value=None, components_used=(), weights={}, missing=missing)
    total_weight = sum(weights[component] for component in available)
    renormalized = {
        component: weights[component] / total_weight for component in available
    }
    value = sum(
        renormalized[component] * values[component] for component in available
    )
    return StrengthResult(
        value=value,
        components_used=tuple(available),
        weights=renormalized,
        missing=missing,
    )


__all__ = [
    "COMPONENT_INDICATOR",
    "DIGITS",
    "DIRECTION_BENEFIT",
    "DIRECTION_COST",
    "Indicator",
    "MONTH_SECONDS",
    "STRENGTH_COMPONENT_KEYS",
    "StrengthResult",
    "burst_indicator",
    "composite_strength",
    "concentration_indicators",
    "helpfulness_indicators",
    "normalized_entropy",
    "rating_agreement_indicator",
    "rating_text_agreement_indicator",
    "recency_indicators",
    "review_volume_sufficiency",
    "sentiment_shares_indicator",
    "star_counts",
    "verified_purchase_indicator",
    "volume_indicator",
]

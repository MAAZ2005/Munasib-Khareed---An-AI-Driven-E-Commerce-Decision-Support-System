"""Analysis service contract and the deterministic stub implementation.

The API layer depends on :class:`AnalysisService` only, so the real
fuzzy + TOPSIS + SHAP engine can replace :class:`StubAnalysisService` later
without touching a single router (locked design decision: the UI never imports
ML modules, it only calls the API).

The stub splits every field honestly in two:

**Real (computed from the serving cache rows).** Review counts, the star
histogram, weak-label sentiment shares, monthly temporal buckets, the verified
and helpfulness evidence indicators, review depth, the composite evidence
strength, the product's own review date range, the dataset ``reference_date``
from ``data/processed/meta.json``, and the alternatives read from the real
``related_products`` edges.

**Placeholder (seeded from the ASIN).** The fuzzy assessment, the TOPSIS
ranking, the SHAP explanation, the risk score, the verdict label, the concerns
and the aspects. Each is ``hashlib.sha256``-derived, never Python's randomised
``hash()``, so two calls for the same ASIN are byte-identical and two different
ASINs get different values. Every stub response carries
``pipeline_version="stub-0"`` and a ``warnings`` entry beginning with
``"STUB:"``; nothing placeholder is ever presented as a result.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from munasib.api.schemas.alternatives import (
    AlternativeProduct,
    AlternativeRelation,
    AlternativesResult,
    ComparisonDeltas,
)
from munasib.api.schemas.analysis import AnalysisResponse, DataBasis
from munasib.api.schemas.decision import (
    CRITERION_NAMES,
    CriterionName,
    DecisionAssessment,
    FiredRule,
    FuzzyAssessment,
    MembershipProfile,
    TopsisCriterion,
    TopsisResult,
)
from munasib.api.schemas.evidence import (
    EvidenceIndicator,
    EvidenceIndicators,
    IndicatorDirection,
)
from munasib.api.schemas.explanation import (
    ShapContribution,
    ShapExplanation,
    ShapReason,
    SurrogateFidelity,
)
from munasib.api.schemas.products import ProductDetail, ProductSummary
from munasib.api.schemas.review_analysis import (
    RatingDistribution,
    ReviewAnalysis,
    SentimentBreakdown,
    TemporalBucket,
    TemporalTrend,
)
from munasib.api.schemas.verdict import Concern, ConcernSeverity, ConsumerVerdict, VerdictLabel
from munasib.core.config import Settings
from munasib.core.constants import (
    SENTIMENT_LABELS,
    VERDICT_LABELS,
)
from munasib.core.exceptions import DataError, ProductNotFoundError
from munasib.db.models import Product, Review
from munasib.db.repositories import RelatedRepository

#: Version string stamped on every stub response (a placeholder, never a result).
STUB_PIPELINE_VERSION = "stub-0"

#: Every stub warning starts with this prefix so a client can filter them out.
STUB_WARNING_PREFIX = "STUB:"

#: How many alternatives the stub actually compares (``usable`` still reports
#: the true distinct count, and ``items`` may never exceed it).
MAX_STUB_ALTERNATIVES = 20

#: Namespace mixed into every hash so stub values cannot collide with the real
#: engine's future outputs.
_STUB_NAMESPACE = "munasib-stub-0"

#: Decimals used for placeholder degrees/shares and risk scores, so repeated
#: calls serialise identically.
_DEGREE_DIGITS = 4
_RISK_DIGITS = 2


# ------------------------------------------------------------------------------
# Deterministic seeding
# ------------------------------------------------------------------------------


def _seed_int(asin: str, *parts: str) -> int:
    """Derive a stable 64-bit integer from an ASIN and a salt.

    Args:
        asin: Product ASIN.
        *parts: Extra salt strings (one per placeholder family).

    Returns:
        A deterministic integer in ``[0, 2**64)``.
    """
    payload = "|".join((_STUB_NAMESPACE, asin, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _unit(seed: int) -> float:
    """Map a seed onto the unit interval.

    Args:
        seed: Integer from :func:`_seed_int`.

    Returns:
        A float in ``[0.0, 1.0)``.
    """
    return seed / float(1 << 64)


def _ranged(seed: int, low: float, high: float) -> float:
    """Map a seed onto ``[low, high]``.

    Args:
        seed: Integer from :func:`_seed_int`.
        low: Inclusive lower bound.
        high: Inclusive upper bound.

    Returns:
        A deterministic float in ``[low, high]``.
    """
    return low + _unit(seed) * (high - low)


def _criterion_seed(asin: str, criterion: str) -> int:
    """Return the seed for one criterion's placeholder value.

    Args:
        asin: Product ASIN.
        criterion: Criterion key from ``configs/criteria_weights.yaml``.

    Returns:
        A deterministic integer.
    """
    return _seed_int(asin, "criterion", criterion)


# ------------------------------------------------------------------------------
# Config-backed specs (names, weights, directions, rule ids come from config)
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class CriterionSpec:
    """One TOPSIS criterion exactly as declared in the weights config."""

    name: str
    description: str
    weight: float
    direction: str


@dataclass(frozen=True)
class CriteriaSpec:
    """The whole TOPSIS problem declared in ``configs/criteria_weights.yaml``."""

    criteria: tuple[CriterionSpec, ...]
    sum_tolerance: float

    @property
    def risk_criteria(self) -> tuple[CriterionSpec, ...]:
        """The five 0-1 criteria that feed the risk stage (no ``fuzzy_risk``)."""
        return tuple(c for c in self.criteria if c.name != CriterionName.FUZZY_RISK.value)


@dataclass(frozen=True)
class FuzzySpec:
    """Fuzzy inputs, their terms and the rule base from the fuzzy rules config."""

    inputs: tuple[str, ...]
    input_terms: dict[str, tuple[str, ...]]
    output_terms: tuple[str, ...]
    rules: tuple[dict[str, Any], ...]
    defuzzification_method: str


@dataclass(frozen=True)
class VerdictThresholds:
    """Verdict bands and the weak-evidence guardrail from the thresholds config."""

    consider_max_risk: float
    consider_min_evidence: float
    consider_min_verified: float
    risk_min_risk: float


@dataclass(frozen=True)
class DatasetMeta:
    """Dataset-wide provenance read from ``data/processed/meta.json``."""

    category: str
    reference_date: datetime
    built_at: datetime


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    """Read one YAML config file.

    Args:
        path: File to read.
        label: Human-readable name used in error messages.

    Returns:
        The parsed mapping (empty when the file holds nothing).

    Raises:
        DataError: If the file is missing or not a YAML mapping.
    """
    if not path.exists():
        raise DataError(f"{label} not found at {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise DataError(f"{label} at {path} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise DataError(f"{label} at {path} must contain a mapping at the top level")
    return loaded


def load_criteria_spec(settings: Settings) -> CriteriaSpec:
    """Load the six TOPSIS criteria from ``configs/criteria_weights.yaml``.

    Args:
        settings: Project settings (``decision.criteria_weights_file``).

    Returns:
        The parsed criteria with weights, directions and the sum tolerance.

    Raises:
        DataError: If the file is missing, a criterion lacks a name, weight or
            direction, a direction is not ``benefit``/``cost``, a weight is
            outside (0, 1], or the weights do not sum to 1.0 within the
            configured tolerance.
    """
    raw = _load_yaml(Path(settings.decision.criteria_weights_file), "criteria weights file")
    configured = raw.get("criteria")
    if not isinstance(configured, dict) or not configured:
        raise DataError("criteria weights file declares no 'criteria' mapping")

    criteria: list[CriterionSpec] = []
    for name, body in configured.items():
        if not isinstance(body, dict):
            raise DataError(f"criterion {name!r} must be a mapping")
        weight = body.get("weight")
        direction = body.get("type")
        description = body.get("description")
        if not isinstance(weight, int | float) or isinstance(weight, bool):
            raise DataError(f"criterion {name!r} has no numeric 'weight'")
        if direction not in ("benefit", "cost"):
            raise DataError(
                f"criterion {name!r} has direction {direction!r}; expected 'benefit' or 'cost'"
            )
        if not 0.0 < float(weight) <= 1.0:
            raise DataError(f"criterion {name!r} weight {weight} is outside (0, 1]")
        criteria.append(
            CriterionSpec(
                name=str(name),
                description=str(description or name),
                weight=float(weight),
                direction=str(direction),
            )
        )

    tolerance = float(raw.get("validation", {}).get("sum_tolerance", 0.001))
    total = sum(criterion.weight for criterion in criteria)
    if abs(total - 1.0) > tolerance:
        raise DataError(
            f"criterion weights must sum to 1.0 within {tolerance}; got {total:.4f}"
        )

    declared = tuple(criterion.name for criterion in criteria)
    if set(declared) != set(CRITERION_NAMES):
        raise DataError(
            f"criteria weights file declares {sorted(declared)} but the schema "
            f"expects {sorted(CRITERION_NAMES)}"
        )
    return CriteriaSpec(criteria=tuple(criteria), sum_tolerance=tolerance)


def load_fuzzy_spec(settings: Settings) -> FuzzySpec:
    """Load the fuzzy inputs, terms and rules from ``configs/fuzzy_rules.yaml``.

    Args:
        settings: Project settings (``risk.fuzzy_rules_file``).

    Returns:
        The parsed fuzzy specification.

    Raises:
        DataError: If the file is missing or malformed.
    """
    raw = _load_yaml(Path(settings.risk.fuzzy_rules_file), "fuzzy rules file")
    variables = raw.get("variables")
    inputs = variables.get("inputs") if isinstance(variables, dict) else None
    output = variables.get("output") if isinstance(variables, dict) else None
    if not isinstance(inputs, dict) or not inputs:
        raise DataError("fuzzy rules file declares no 'variables.inputs' mapping")
    if not isinstance(output, dict) or not output:
        raise DataError("fuzzy rules file declares no 'variables.output' mapping")

    input_terms: dict[str, tuple[str, ...]] = {}
    for name, body in inputs.items():
        terms = body.get("terms") if isinstance(body, dict) else None
        if not isinstance(terms, dict) or not terms:
            raise DataError(f"fuzzy input {name!r} declares no 'terms'")
        input_terms[str(name)] = tuple(str(term) for term in terms)

    output_terms: tuple[str, ...] = ()
    for body in output.values():
        if isinstance(body, dict) and isinstance(body.get("terms"), dict):
            output_terms = tuple(str(term) for term in body["terms"])
    if not output_terms:
        raise DataError("fuzzy rules file declares no output terms")

    rules_raw = raw.get("rules")
    if not isinstance(rules_raw, list) or not rules_raw:
        raise DataError("fuzzy rules file declares no 'rules' list")
    rules = tuple(rule for rule in rules_raw if isinstance(rule, dict))

    return FuzzySpec(
        inputs=tuple(input_terms),
        input_terms=input_terms,
        output_terms=output_terms,
        rules=rules,
        defuzzification_method=settings.risk.default_defuzzification_method,
    )


def load_verdict_thresholds(settings: Settings) -> VerdictThresholds:
    """Load the verdict bands from ``configs/verdict_thresholds.yaml``.

    Args:
        settings: Project settings (``verdict.thresholds_file``).

    Returns:
        The parsed thresholds.

    Raises:
        DataError: If the file is missing or a band is missing its key.
    """
    raw = _load_yaml(Path(settings.verdict.thresholds_file), "verdict thresholds file")
    verdicts = raw.get("verdicts")
    if not isinstance(verdicts, dict):
        raise DataError("verdict thresholds file declares no 'verdicts' mapping")
    consider = verdicts.get("consider")
    risk = verdicts.get("consumer_risk")
    if not isinstance(consider, dict) or not isinstance(risk, dict):
        raise DataError("verdict thresholds file needs 'consider' and 'consumer_risk' bands")
    for band, keys in ((consider, ("max_risk_score", "min_evidence_strength", "min_verified_ratio")), (risk, ("min_risk_score",))):
        for key in keys:
            if band.get(key) is None:
                raise DataError(f"verdict band is missing {key!r}")
    return VerdictThresholds(
        consider_max_risk=float(consider["max_risk_score"]),
        consider_min_evidence=float(consider["min_evidence_strength"]),
        consider_min_verified=float(consider["min_verified_ratio"]),
        risk_min_risk=float(risk["min_risk_score"]),
    )


def load_dataset_meta(settings: Settings) -> DatasetMeta:
    """Read the dataset-wide provenance from ``data/processed/meta.json``.

    Args:
        settings: Project settings (``paths.processed_data_dir``).

    Returns:
        The category, the ``reference_date`` every recency feature is anchored
        on, and the build timestamp used as ``generated_at``.

    Raises:
        DataError: If the file is missing or lacks the review date range.
    """
    path = Path(settings.paths.processed_data_dir) / "meta.json"
    if not path.exists():
        raise DataError(
            f"dataset metadata not found at {path}; run scripts/build_dataset.py"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"dataset metadata at {path} is not valid JSON: {exc}") from exc

    date_range = raw.get("review_date_range")
    if not isinstance(date_range, dict) or not date_range.get("max"):
        raise DataError(f"dataset metadata at {path} declares no review_date_range.max")
    category = raw.get("category")
    if not isinstance(category, str) or not category:
        raise DataError(f"dataset metadata at {path} declares no category")
    return DatasetMeta(
        category=category,
        reference_date=_as_utc(str(date_range["max"])),
        built_at=_as_utc(str(raw.get("built_at") or date_range["max"])),
    )


def _as_utc(value: str) -> datetime:
    """Parse an ISO timestamp into an aware UTC datetime.

    Args:
        value: ISO 8601 string with or without an offset.

    Returns:
        The same instant in UTC.
    """
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


# ------------------------------------------------------------------------------
# Small numeric helpers
# ------------------------------------------------------------------------------


def _share(count: int, total: int) -> float:
    """Divide safely, returning 0.0 for an empty population.

    Args:
        count: Numerator.
        total: Denominator.

    Returns:
        ``count / total`` or 0.0 when ``total`` is 0.
    """
    if total <= 0:
        return 0.0
    return count / total


def _round(value: float, digits: int) -> float:
    """Round for stable serialisation.

    Args:
        value: Number to round.
        digits: Decimal places.

    Returns:
        The rounded value.
    """
    return round(float(value), digits)


def _clamp_unit(value: float) -> float:
    """Clamp onto the unit interval (guards against float drift).

    Args:
        value: Candidate value.

    Returns:
        The value bounded to ``[0.0, 1.0]``.
    """
    return min(1.0, max(0.0, float(value)))


def _ols_slope(values: Sequence[float]) -> float | None:
    """Ordinary-least-squares slope of ``values`` against their index.

    Args:
        values: Series ordered by ascending period.

    Returns:
        The slope, or None when fewer than two points exist.
    """
    if len(values) < 2:
        return None
    count = len(values)
    mean_x = (count - 1) / 2.0
    mean_y = sum(values) / count
    ss_xy = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    ss_xx = sum((index - mean_x) ** 2 for index in range(count))
    if ss_xx == 0:
        return None
    return ss_xy / ss_xx


def _star_key(rating: float | None) -> int | None:
    """Map a stored star rating onto the 1-5 integer histogram key.

    Args:
        rating: Rating as stored (may be None).

    Returns:
        The star in 1-5, or None when unrated.
    """
    if rating is None:
        return None
    star = int(round(float(rating)))
    return star if 1 <= star <= 5 else None


# ------------------------------------------------------------------------------
# Real blocks computed from review rows
# ------------------------------------------------------------------------------


def _rating_distribution(reviews: Sequence[Review]) -> RatingDistribution:
    """Build the star histogram for one product.

    Args:
        reviews: Every stored review of the product.

    Returns:
        Counts and shares for all five stars; ``total`` is the number of rated
        reviews and shares are 0.0 when nothing is rated.
    """
    counts = dict.fromkeys(range(1, 6), 0)
    rated = 0
    total_rating = 0.0
    for review in reviews:
        star = _star_key(review.rating)
        if star is None:
            continue
        counts[star] += 1
        rated += 1
        total_rating += star
    shares = {star: _share(counts[star], rated) for star in range(1, 6)}
    mean = _round(total_rating / rated, 6) if rated else None
    return RatingDistribution(
        counts=counts,
        shares={star: _round(value, _DEGREE_DIGITS) for star, value in shares.items()},
        mean_rating=mean,
        total=rated,
    )


def _sentiment_breakdown(reviews: Sequence[Review]) -> SentimentBreakdown:
    """Build the weak-label sentiment breakdown for one product.

    Args:
        reviews: Every stored review of the product.

    Returns:
        Counts and shares per weak label; ``unlabelled_count`` counts reviews
        with no rating, and the shares use every review as denominator.
    """
    total = len(reviews)
    counts = dict.fromkeys(SENTIMENT_LABELS, 0)
    unlabelled = 0
    for review in reviews:
        code = review.sentiment_label_weak
        if code is None or code not in counts:
            unlabelled += 1
            continue
        counts[code] += 1
    return SentimentBreakdown(
        total=total,
        negative_count=counts[SENTIMENT_LABELS[0]],
        neutral_count=counts[SENTIMENT_LABELS[1]],
        positive_count=counts[SENTIMENT_LABELS[2]],
        unlabelled_count=unlabelled,
        negative_share=_round(_share(counts[SENTIMENT_LABELS[0]], total), _DEGREE_DIGITS),
        neutral_share=_round(_share(counts[SENTIMENT_LABELS[1]], total), _DEGREE_DIGITS),
        positive_share=_round(_share(counts[SENTIMENT_LABELS[2]], total), _DEGREE_DIGITS),
    )


def _month_key(review: Review) -> tuple[int, int]:
    """Return the ``(year, month)`` bucket of a review.

    Args:
        review: Review row with a UTC ``review_time``.

    Returns:
        The calendar month key.
    """
    moment = review.review_time
    return (moment.year, moment.month)


def _temporal_trend(reviews: Sequence[Review]) -> TemporalTrend:
    """Bucket reviews by calendar month and compute the positive-share slope.

    Args:
        reviews: Every stored review of the product.

    Returns:
        One bucket per non-empty month (empty months are never fabricated),
        ascending, plus the OLS slope of the positive share per period.
    """
    grouped: dict[tuple[int, int], list[int | None]] = {}
    for review in reviews:
        grouped.setdefault(_month_key(review), []).append(review.sentiment_label_weak)

    buckets: list[TemporalBucket] = []
    positive_shares: list[float] = []
    for (year, month) in sorted(grouped):
        labels = grouped[(year, month)]
        total = len(labels)
        negative = sum(1 for code in labels if code == SENTIMENT_LABELS[0])
        neutral = sum(1 for code in labels if code == SENTIMENT_LABELS[1])
        positive = sum(1 for code in labels if code == SENTIMENT_LABELS[2])
        positive_share = _share(positive, total)
        positive_shares.append(positive_share)
        buckets.append(
            TemporalBucket(
                period_start=datetime(year, month, 1, tzinfo=UTC),
                n_reviews=total,
                negative_share=_round(_share(negative, total), _DEGREE_DIGITS),
                neutral_share=_round(_share(neutral, total), _DEGREE_DIGITS),
                positive_share=_round(positive_share, _DEGREE_DIGITS),
            )
        )

    slope = _ols_slope(positive_shares)
    return TemporalTrend(
        period="month",
        buckets=buckets,
        positive_share_slope=None if slope is None else _round(slope, 6),
        n_reviews_used=sum(bucket.n_reviews for bucket in buckets),
    )


def _entropy(counts: Sequence[int]) -> float:
    """Shannon entropy of a histogram in bits.

    Args:
        counts: Non-negative bin sizes.

    Returns:
        The entropy in bits (0.0 for an empty or single-bin histogram).
    """
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        share = count / total
        entropy -= share * math.log2(share)
    return entropy


def _evidence_indicators(
    reviews: Sequence[Review],
    rating_distribution: RatingDistribution,
    sentiment: SentimentBreakdown,
    settings: Settings,
) -> EvidenceIndicators:
    """Compute the evidence indicators honestly from the review rows.

    The three fixed indicators are the configured weights of
    ``review_intel.evidence``; ``details`` carries the extra signals that are
    also directly computable from the rows (star entropy, helpfulness-weighted
    sentiment, negative severity).

    Args:
        reviews: Every stored review of the product.
        rating_distribution: Pre-computed star histogram.
        sentiment: Pre-computed weak-label breakdown.
        settings: Project settings (``review_intel.evidence``).

    Returns:
        The populated evidence block.
    """
    total = len(reviews)
    verified = sum(1 for review in reviews if review.verified is True)
    helpful = sum(1 for review in reviews if (review.helpful_votes or 0) >= 1)
    votes = sum(int(review.helpful_votes or 0) for review in reviews)
    mean_length = sum(int(review.text_length or 0) for review in reviews) / total if total else 0.0

    verified_share = _share(verified, total)
    helpfulness_share = _share(helpful, total)
    mean_votes = _share(votes, total)
    scale = float(settings.review_intel.evidence.length_scale_chars) or 1.0
    depth = _clamp_unit(mean_length / scale)

    weights = settings.review_intel.evidence.weights
    strength = (
        weights.verified_purchase * verified_share
        + weights.helpfulness_ratio * helpfulness_share
        + weights.review_depth * depth
    )

    star_counts = [rating_distribution.counts[star] for star in range(1, 6)]
    star_total = sum(star_counts)
    entropy = _entropy(star_counts)
    entropy_norm = _clamp_unit(entropy / math.log2(5)) if star_total else 0.0

    weighted_labels = sum(
        int(review.helpful_votes or 0) * int(code)
        for review, code in (
            (review, review.sentiment_label_weak) for review in reviews
        )
        if code is not None
    )
    weighted_votes = sum(
        int(review.helpful_votes or 0)
        for review in reviews
        if review.sentiment_label_weak is not None
    )
    weighted_mean = _share(weighted_labels, weighted_votes) if weighted_votes else 0.0
    # Unweighted comparison on the SAME 0-1 scale as the weighted value, so the
    # "above/below" sentence below is a real comparison and not a units mismatch.
    labelled = sum(1 for review in reviews if review.sentiment_label_weak is not None)
    unweighted_mean = _share(
        sum(int(code) for code in (r.sentiment_label_weak for r in reviews) if code is not None),
        labelled,
    )
    unweighted_rescaled = _clamp_unit(unweighted_mean / 2.0)
    weighted_value = _clamp_unit(weighted_mean / 2.0)

    one_star_negative = sum(
        1
        for review in reviews
        if review.sentiment_label_weak == SENTIMENT_LABELS[0]
        and _star_key(review.rating) == 1
    )

    return EvidenceIndicators(
        verified_purchase=EvidenceIndicator(
            name="verified_purchase",
            value=_round(verified_share, _DEGREE_DIGITS),
            raw_value=_round(verified_share, _DEGREE_DIGITS),
            direction=IndicatorDirection.BENEFIT,
            description=(
                f"Share of the product's reviews that are verified purchases "
                f"({verified} of {total} here)."
            ),
        ),
        helpfulness=EvidenceIndicator(
            name="helpfulness_ratio",
            value=_round(helpfulness_share, _DEGREE_DIGITS),
            raw_value=_round(mean_votes, _DEGREE_DIGITS),
            direction=IndicatorDirection.BENEFIT,
            description=(
                f"Share of reviews that at least one other shopper found helpful "
                f"({helpful} of {total} here); the raw value is the mean "
                f"helpful-vote count per review."
            ),
        ),
        review_depth=EvidenceIndicator(
            name="review_depth",
            value=_round(depth, _DEGREE_DIGITS),
            raw_value=_round(mean_length, 2),
            direction=IndicatorDirection.BENEFIT,
            description=(
                f"How substantive the review text is, from the mean character "
                f"length of the reviews ({mean_length:.2f} characters here)."
            ),
        ),
        evidence_strength=_round(strength, _DEGREE_DIGITS),
        details=[
            EvidenceIndicator(
                name="rating_distribution_entropy",
                value=_round(entropy_norm, _DEGREE_DIGITS),
                raw_value=_round(entropy, _DEGREE_DIGITS),
                direction=IndicatorDirection.COST,
                description=(
                    "Shannon entropy of the star distribution normalised by "
                    f"log(5) (0 = one shared star, 1 = perfectly spread): "
                    f"{entropy_norm:.4f} here over the "
                    f"{'/'.join(str(rating_distribution.counts[star]) for star in range(1, 6))} "
                    f"star split, so opinions span all five stars and the "
                    f"{(rating_distribution.mean_rating or 0.0):.2f} mean hides a divided audience."
                ),
            ),
            EvidenceIndicator(
                name="helpfulness_weighted_sentiment",
                value=_round(weighted_value, _DEGREE_DIGITS),
                raw_value=_round(weighted_mean, _DEGREE_DIGITS),
                direction=IndicatorDirection.BENEFIT,
                description=(
                    "Weak labels (0 negative, 1 neutral, 2 positive) averaged "
                    "with each review's helpful votes as weight and rescaled to "
                    f"0-1 (raw 0-2 mean label {weighted_mean:.4f}): "
                    f"{weighted_value:.4f} here from {weighted_votes} helpful "
                    f"votes across {labelled} labelled reviews, "
                    f"{'above' if weighted_value > unweighted_rescaled else 'below'} "
                    f"the {unweighted_rescaled:.4f} unweighted value - the reviews "
                    "shoppers voted useful read "
                    f"{'better' if weighted_value > unweighted_rescaled else 'worse'} "
                    "than average."
                ),
            ),
            EvidenceIndicator(
                name="negative_severity_share",
                value=_round(_share(one_star_negative, sentiment.negative_count), _DEGREE_DIGITS),
                raw_value=float(one_star_negative),
                direction=IndicatorDirection.COST,
                description=(
                    "Share of the weak-negative reviews that are one-star reviews "
                    f"({one_star_negative} of {sentiment.negative_count} here): "
                    "negative feedback packed into the harshest bucket rather "
                    "than spread over 2-3 stars."
                ),
            ),
        ],
    )


# ------------------------------------------------------------------------------
# Placeholder blocks (seeded from the ASIN, never presented as results)
# ------------------------------------------------------------------------------


def _placeholder_fuzzy(asin: str, fuzzy: FuzzySpec) -> FuzzyAssessment:
    """Build the placeholder fuzzy assessment for one product.

    Input names, linguistic terms, rule ids/antecedents/consequents and the
    defuzzification method all come from the real config; only the degrees,
    firing strengths and the defuzzified value are seeded placeholders.

    Args:
        asin: Product ASIN.
        fuzzy: The fuzzy specification loaded from config.

    Returns:
        A schema-valid fuzzy assessment with placeholder numbers.
    """
    memberships = {
        name: MembershipProfile(
            term_degrees={
                term: _round(_unit(_seed_int(asin, "fuzzy", name, term)), _DEGREE_DIGITS)
                for term in fuzzy.input_terms[name]
            }
        )
        for name in fuzzy.inputs
    }

    # A deterministic slice of the real rule base: only the strengths are
    # placeholders, so the rules an auditor sees are genuinely configured ones.
    rule_count = len(fuzzy.rules)
    start = _seed_int(asin, "fired", "start") % rule_count if rule_count else 0
    chosen = [fuzzy.rules[(start + offset) % rule_count] for offset in range(min(3, rule_count))]

    fired: list[FiredRule] = []
    for rule in chosen:
        rule_id = int(rule.get("id", 0) or 0)
        antecedents_raw = rule.get("antecedents")
        antecedents = (
            {str(k): str(v) for k, v in antecedents_raw.items()}
            if isinstance(antecedents_raw, dict)
            else {}
        )
        consequent_raw = rule.get("consequent")
        consequent = ""
        if isinstance(consequent_raw, dict):
            consequent = str(next(iter(consequent_raw.values()), ""))
        fired.append(
            FiredRule(
                rule_id=rule_id,
                antecedents=antecedents,
                consequent=consequent,
                strength=_round(_unit(_seed_int(asin, "rule", str(rule_id))), _DEGREE_DIGITS),
            )
        )
    fired.sort(key=lambda item: item.strength, reverse=True)

    output_degrees = {
        term: _round(_unit(_seed_int(asin, "output", term)), _DEGREE_DIGITS)
        for term in fuzzy.output_terms
    }
    # Renormalise so the degrees read as an aggregate a centroid could produce.
    total = sum(output_degrees.values())
    if total > 0:
        output_degrees = {
            term: _round(value / total, _DEGREE_DIGITS) for term, value in output_degrees.items()
        }

    return FuzzyAssessment(
        input_memberships=memberships,
        fired_rules=fired,
        output_degrees=output_degrees,
        defuzzified=_round(
            _ranged(_seed_int(asin, "defuzzified"), 5.0, 95.0), _RISK_DIGITS
        ),
        defuzzification_method=fuzzy.defuzzification_method,
    )


def _placeholder_risk(asin: str) -> float:
    """Return the placeholder consumer risk for one product.

    Seeded only from the ASIN, so the subject and every alternative get a stable
    score and the comparison deltas stay consistent between calls.

    Args:
        asin: Product ASIN.

    Returns:
        A deterministic risk in [5.0, 95.0].
    """
    return _round(_ranged(_seed_int(asin, "risk"), 5.0, 95.0), _RISK_DIGITS)


def _placeholder_topsis(
    asin: str,
    criteria: CriteriaSpec,
    fuzzy: FuzzySpec,
    evidence_strength: float,
) -> TopsisResult:
    """Build the placeholder TOPSIS result for one product.

    Names, weights, directions and descriptions come from the real config, and
    ``evidence_strength``/``fuzzy_risk`` carry real (respectively internally
    consistent) values; every other value, the normalised value and the
    contribution are seeded placeholders.

    Args:
        asin: Product ASIN.
        criteria: The criteria loaded from the weights config.
        fuzzy: The fuzzy specification loaded from the rules config.
        evidence_strength: The product's real evidence strength.

    Returns:
        A schema-valid TOPSIS result with placeholder values.
    """
    fuzzy_value = _placeholder_fuzzy(asin, fuzzy).defuzzified

    entries: list[TopsisCriterion] = []
    for criterion in criteria.criteria:
        name = criterion.name
        if name == CriterionName.EVIDENCE_STRENGTH.value:
            value = float(evidence_strength)
        elif name == CriterionName.FUZZY_RISK.value:
            value = float(fuzzy_value)
        elif name in (
            CriterionName.PRODUCT_QUALITY.value,
            CriterionName.RATING_RELIABILITY.value,
            CriterionName.NEGATIVE_FEEDBACK_CONCENTRATION.value,
            CriterionName.PRICE_POSITION.value,
        ):
            value = _round(_unit(_criterion_seed(asin, name)), _DEGREE_DIGITS)
        else:  # pragma: no cover - guarded by load_criteria_spec
            value = 0.0
        entries.append(
            TopsisCriterion(
                name=CriterionName(name),
                description=criterion.description,
                weight=criterion.weight,
                direction=criterion.direction,
                value=value,
                normalized_value=_round(_unit(_criterion_seed(asin, name)), _DEGREE_DIGITS),
                contribution=_round(
                    criterion.weight * _unit(_criterion_seed(asin, name)), _DEGREE_DIGITS
                ),
            )
        )

    peer_count = 20
    ranked_count = peer_count + 1
    rank = 1 + (_seed_int(asin, "rank") % peer_count)
    percentile = 100.0 * (ranked_count - rank) / ranked_count
    return TopsisResult(
        criteria=entries,
        closeness=_round(_unit(_seed_int(asin, "closeness")), _DEGREE_DIGITS),
        rank=rank,
        ranked_count=ranked_count,
        peer_count=peer_count,
        percentile=_round(percentile, 2),
        risk_score=_placeholder_risk(asin),
    )


def _placeholder_explanation(asin: str, sentiment: SentimentBreakdown) -> ShapExplanation:
    """Build the placeholder SHAP explanation for one product.

    The three features are real pipeline features with the product's real values;
    the contributions, base value and fidelity are seeded placeholders, and
    ``meets_minimum`` is deliberately False because no surrogate was fitted.

    Args:
        asin: Product ASIN.
        sentiment: The product's real weak-label breakdown.

    Returns:
        A schema-valid explanation whose numbers are labelled placeholders.
    """
    features = (
        ("negative_sentiment_share", sentiment.negative_share),
        ("neutral_sentiment_share", sentiment.neutral_share),
        ("positive_sentiment_share", sentiment.positive_share),
    )
    contributions = [
        ShapContribution(
            feature=name,
            value=_round(value, _DEGREE_DIGITS),
            contribution=_round(
                _ranged(_seed_int(asin, "shap", name), -20.0, 20.0), _RISK_DIGITS
            ),
        )
        for name, value in features
    ]
    contributions.sort(key=lambda item: abs(item.contribution), reverse=True)

    return ShapExplanation(
        base_value=_round(_ranged(_seed_int(asin, "shap", "base"), 20.0, 60.0), _RISK_DIGITS),
        contributions=contributions,
        surrogate_fidelity=SurrogateFidelity(
            r2=0.0,
            mae=0.0,
            n_training_points=0,
            meets_minimum=False,
        ),
        reasons=[
            ShapReason(
                text=(
                    f"{sentiment.negative_count} of {sentiment.total} reviews "
                    f"({sentiment.negative_share:.1%}) carry the weak negative label "
                    "(placeholder ranking, not a fitted surrogate)."
                ),
                feature="negative_sentiment_share",
                raw_value=_round(sentiment.negative_share, _DEGREE_DIGITS),
            )
        ],
    )


# ------------------------------------------------------------------------------
# Verdict synthesis (bands from configs/verdict_thresholds.yaml)
# ------------------------------------------------------------------------------


def _verdict_label(
    risk_score: float,
    evidence_strength: float,
    verified_share: float,
    thresholds: VerdictThresholds,
) -> VerdictLabel:
    """Pick the verdict label from the risk score and the evidence guardrail.

    Args:
        risk_score: Consumer risk on 0-100.
        evidence_strength: Evidence strength on 0-1.
        verified_share: Share of verified purchases on 0-1.
        thresholds: The bands loaded from the thresholds config.

    Returns:
        One of the three locked labels. Very weak evidence can never yield a
        clean ``CONSIDER``.
    """
    if risk_score >= thresholds.risk_min_risk:
        return VerdictLabel.CONSUMER_RISK
    clean_consider = (
        risk_score <= thresholds.consider_max_risk
        and evidence_strength >= thresholds.consider_min_evidence
        and verified_share >= thresholds.consider_min_verified
    )
    return VerdictLabel.CONSIDER if clean_consider else VerdictLabel.CONSIDER_WITH_CAUTION


def _severity(index: int) -> ConcernSeverity:
    """Map a verdict-band distance onto a concern severity.

    Args:
        index: 0 for the lowest band, 1 for the middle, 2 for the highest.

    Returns:
        The matching severity.
    """
    return (ConcernSeverity.LOW, ConcernSeverity.MEDIUM, ConcernSeverity.HIGH)[index]


def _placeholder_verdict(
    asin: str,
    risk_score: float,
    evidence_strength: float,
    reviews: Sequence[Review],
    rating_distribution: RatingDistribution,
    sentiment: SentimentBreakdown,
    thresholds: VerdictThresholds,
) -> ConsumerVerdict:
    """Build the verdict from the placeholder risk score and the real evidence.

    Args:
        asin: Product ASIN.
        risk_score: Placeholder consumer risk on 0-100.
        evidence_strength: Real evidence strength on 0-1.
        reviews: Every stored review of the product.
        rating_distribution: Pre-computed star histogram.
        sentiment: Pre-computed weak-label breakdown.
        thresholds: The bands loaded from the thresholds config.

    Returns:
        A consumer verdict whose rationale cites only real numbers.
    """
    total = len(reviews)
    verified = sum(1 for review in reviews if review.verified is True)
    verified_share = _share(verified, total)
    label = _verdict_label(risk_score, evidence_strength, verified_share, thresholds)

    mean_rating = rating_distribution.mean_rating
    one_star = rating_distribution.counts[1]
    concerns: list[Concern] = []
    if sentiment.negative_share >= 0.25:
        concerns.append(
            Concern(
                code="high_negative_sentiment",
                text=(
                    f"{sentiment.negative_count} of {sentiment.total} reviews "
                    f"({sentiment.negative_share:.1%}) carry the weak negative label, "
                    "above the 25% caution band."
                ),
                severity=_severity(2),
                supporting_numbers={
                    "negative_share": sentiment.negative_share,
                    "negative_count": float(sentiment.negative_count),
                    "total_reviews": float(sentiment.total),
                },
            )
        )
    if mean_rating is not None and mean_rating < 3.0:
        concerns.append(
            Concern(
                code="low_mean_rating",
                text=(
                    f"The mean rating is {mean_rating:.2f} out of 5, with "
                    f"{one_star} of {sentiment.total} reviews rated one star."
                ),
                severity=_severity(1),
                supporting_numbers={
                    "mean_rating": mean_rating,
                    "one_star_count": float(one_star),
                    "one_star_share": rating_distribution.shares[1],
                },
            )
        )
    if evidence_strength < thresholds.consider_min_evidence:
        concerns.append(
            Concern(
                code="weak_evidence",
                text=(
                    f"Evidence strength is {evidence_strength:.4f}, below the "
                    f"{thresholds.consider_min_evidence:.2f} needed for a clean "
                    "CONSIDER."
                ),
                severity=_severity(1),
                supporting_numbers={
                    "evidence_strength": evidence_strength,
                    "min_evidence_strength": thresholds.consider_min_evidence,
                },
            )
        )

    rationale = (
        f"{label.value}: placeholder risk {risk_score:.1f}/100 against an evidence base "
        f"of {total} reviews ({verified} of {total} verified purchases, "
        f"{sentiment.negative_count} weakly negative). This verdict is a stub: the "
        "risk score it summarises is seeded from the ASIN, not computed."
    )
    return ConsumerVerdict(
        label=label,
        risk_score=risk_score,
        evidence_strength=evidence_strength,
        major_concerns=concerns,
        rationale=rationale,
    )


def _stub_warnings(reviews: Sequence[Review], sentiment: SentimentBreakdown) -> list[str]:
    """Build the warnings every stub analysis carries.

    Args:
        reviews: Every stored review of the product.
        sentiment: Pre-computed weak-label breakdown.

    Returns:
        The warning list, always led by a ``"STUB:"`` entry.
    """
    warnings = [
        f"{STUB_WARNING_PREFIX} The risk assessment (fuzzy, TOPSIS, SHAP, risk score, "
        "verdict and aspects) is a deterministic placeholder seeded from the ASIN, "
        "not a computed result; the real engine replaces it in a later prompt.",
    ]
    warnings.append(
        "Sentiment labels are weak supervision from star ratings (1-2 negative, 3 "
        "neutral, 4-5 positive), not model sentiment."
    )
    warnings.append(
        "Aspects are empty because the aspect-extraction stage has not run yet."
    )
    if sentiment.total < 20:
        warnings.append(
            f"Evidence base of {sentiment.total} reviews is very small; treat every "
            "number here as thin evidence."
        )
    if any(review.verified is None for review in reviews):
        unknown = sum(1 for review in reviews if review.verified is None)
        warnings.append(
            f"{unknown} review(s) have no verified-purchase flag (NULL, not False); "
            "the verified share counts only explicit True rows."
        )
    return warnings


# ------------------------------------------------------------------------------
# Service contract
# ------------------------------------------------------------------------------


class AnalysisService(ABC):
    """What the API layer needs from an analysis backend.

    Both methods take the session-bound repositories implicitly through the
    constructor, so swapping the stub for the real engine is a one-line change
    in :mod:`munasib.api.deps` and no router changes.
    """

    @abstractmethod
    def analyze(self, asin: str) -> AnalysisResponse:
        """Return the full analysis for one product.

        Args:
            asin: Product ASIN.

        Returns:
            The composed analysis response.

        Raises:
            ProductNotFoundError: If the ASIN is not in the serving cache.
        """

    @abstractmethod
    def alternatives(self, asin: str) -> AlternativesResult:
        """Return the usable alternatives for one product.

        Args:
            asin: Product ASIN.

        Returns:
            The alternatives, or an empty result with an explicit reason.

        Raises:
            ProductNotFoundError: If the ASIN is not in the serving cache.
        """


class StubAnalysisService(AnalysisService):
    """Deterministic placeholder analysis built on the real serving-cache rows."""

    def __init__(self, session: Session, settings: Settings) -> None:
        """Wire the service to a session and the project settings.

        Args:
            session: Open SQLAlchemy session over the serving cache.
            settings: Project settings (config paths and thresholds).
        """
        self._session = session
        self._settings = settings
        self._criteria = load_criteria_spec(settings)
        self._fuzzy = load_fuzzy_spec(settings)
        self._thresholds = load_verdict_thresholds(settings)
        self._meta = load_dataset_meta(settings)
        self._related = RelatedRepository(session)

    # -- public API ----------------------------------------------------------

    def analyze(self, asin: str) -> AnalysisResponse:
        """Return the full stub analysis for one product.

        Args:
            asin: Product ASIN.

        Returns:
            The composed analysis response stamped ``pipeline_version="stub-0"``.

        Raises:
            ProductNotFoundError: If the ASIN is not in the serving cache.
        """
        product = self._get_product(asin)
        reviews = self._all_reviews(asin)

        rating = _rating_distribution(reviews)
        sentiment = _sentiment_breakdown(reviews)
        trend = _temporal_trend(reviews)
        review_analysis = ReviewAnalysis(
            rating_distribution=rating,
            sentiment=sentiment,
            temporal_trend=trend,
            aspects=[],
        )
        evidence = _evidence_indicators(reviews, rating, sentiment, self._settings)

        fuzzy = _placeholder_fuzzy(asin, self._fuzzy)
        topsis = _placeholder_topsis(
            asin, self._criteria, self._fuzzy, evidence.evidence_strength
        )
        # The headline risk is the TOPSIS risk; the fuzzy stage's defuzzified
        # value is exactly what TOPSIS receives as its extra cost criterion.
        risk_score = topsis.risk_score

        verdict = _placeholder_verdict(
            asin,
            risk_score,
            evidence.evidence_strength,
            reviews,
            rating,
            sentiment,
            self._thresholds,
        )

        return AnalysisResponse(
            asin=asin,
            product=self._product_detail(product, reviews),
            review_analysis=review_analysis,
            evidence=evidence,
            risk_indicators=self._risk_indicators(asin, evidence.evidence_strength),
            decision=DecisionAssessment(fuzzy=fuzzy, topsis=topsis),
            explanation=_placeholder_explanation(asin, sentiment),
            alternatives=self.alternatives(asin),
            verdict=verdict,
            risk_score=risk_score,
            evidence_strength=evidence.evidence_strength,
            pipeline_version=STUB_PIPELINE_VERSION,
            generated_at=self._meta.built_at,
            data_basis=DataBasis(
                category=self._meta.category,
                n_reviews=len(reviews),
                date_range_start=_first_review_time(reviews, self._meta.reference_date),
                date_range_end=_last_review_time(reviews, self._meta.reference_date),
                reference_date=self._meta.reference_date,
            ),
            warnings=_stub_warnings(reviews, sentiment),
        )

    def alternatives(self, asin: str) -> AlternativesResult:
        """Return the usable alternatives for one product.

        Both counts are per distinct **related product**, never per edge:

        * ``total_related`` is the number of distinct related products of this
          product across ``also_bought`` and ``also_viewed`` (a target linked by
          both edge types counts once) excluding the product itself.
        * ``usable`` is how many of those distinct related products are
          themselves in the processed subset, so it can never exceed
          ``total_related``.

        Args:
            asin: Product ASIN.

        Returns:
            The alternatives result; empty with an explicit reason when nothing
            is comparable.

        Raises:
            ProductNotFoundError: If the ASIN is not in the serving cache.
        """
        self._get_product(asin)
        edges = list(self._related.get_related(asin, only_in_subset=False))

        # Collapse the edges onto distinct related products: one entry per target
        # holding every edge type that links it, with a self-reference skipped
        # (a product is never its own alternative).
        edge_types: dict[str, set[str]] = {}
        in_subset_targets: set[str] = set()
        for edge in edges:
            if edge.related_asin == asin:
                continue
            edge_types.setdefault(edge.related_asin, set()).add(edge.relation_type)
            if edge.in_subset:
                in_subset_targets.add(edge.related_asin)

        total_related = len(edge_types)
        usable_asins = sorted(edge_types.keys() & in_subset_targets)

        if not usable_asins:
            if total_related == 0:
                reason = (
                    "no related products found (0 distinct targets across "
                    "also_bought / also_viewed)"
                )
            else:
                reason = (
                    f"{total_related} distinct related product(s) found, but none of them "
                    "are in the processed subset, so no alternative can be compared"
                )
            return AlternativesResult(
                items=[],
                empty_reason=reason,
                total_related=total_related,
                usable=0,
            )

        subject_risk = _placeholder_risk(asin)
        subject_strength = self._evidence_strength_for(asin)
        subject_label = self._verdict_for(asin, subject_risk, subject_strength)

        items: list[AlternativeProduct] = []
        for related in usable_asins[:MAX_STUB_ALTERNATIVES]:
            product = self._session.get(Product, related)
            if product is None:  # pragma: no cover - in_subset guarantees a row
                continue
            risk = _placeholder_risk(related)
            strength = self._evidence_strength_for(related)
            label = self._verdict_for(related, risk, strength)
            items.append(
                AlternativeProduct(
                    summary=ProductSummary.model_validate(product),
                    risk_score=risk,
                    verdict=label,
                    deltas=ComparisonDeltas(
                        risk_score_delta=_round(risk - subject_risk, _RISK_DIGITS),
                        evidence_strength_delta=_round(
                            strength - subject_strength, _DEGREE_DIGITS
                        ),
                        mean_rating_delta=self._mean_rating_delta(product, asin),
                    ),
                    relation=_relation(subject_label, label),
                    relation_types=sorted(edge_types.get(related, set())),
                )
            )

        return AlternativesResult(
            items=items,
            empty_reason=None,
            total_related=total_related,
            usable=len(usable_asins),
        )

    # -- helpers -------------------------------------------------------------

    def _get_product(self, asin: str) -> Product:
        """Fetch one product, raising the domain error when absent.

        Args:
            asin: Product ASIN.

        Returns:
            The product row.

        Raises:
            ProductNotFoundError: If the ASIN is not in the serving cache.
        """
        product = self._session.get(Product, asin)
        if product is None:
            raise ProductNotFoundError(asin, f"product {asin!r} not found in the serving cache")
        return product

    def _all_reviews(self, asin: str) -> list[Review]:
        """Fetch every review of one product.

        Args:
            asin: Product ASIN.

        Returns:
            The review rows in a stable order.
        """
        return list(
            self._session.scalars(select(Review).where(Review.asin == asin).order_by(Review.id))
        )

    def _product_detail(self, product: Product, reviews: Sequence[Review]) -> ProductDetail:
        """Build the product profile, filling a missing date range from reviews.

        Args:
            product: The product row.
            reviews: The product's review rows.

        Returns:
            The full product detail.
        """
        detail = ProductDetail.model_validate(product)
        if detail.first_review_time is None:
            detail = detail.model_copy(
                update={"first_review_time": _first_review_time(reviews, self._meta.reference_date)}
            )
        if detail.last_review_time is None:
            detail = detail.model_copy(
                update={"last_review_time": _last_review_time(reviews, self._meta.reference_date)}
            )
        return detail

    def _mean_rating_delta(self, alternative: Product, asin: str) -> float | None:
        """Difference between an alternative's and the subject's mean rating.

        Args:
            alternative: The alternative's product row.
            asin: Subject product ASIN.

        Returns:
            ``alternative - subject``, or None when either has no rating.
        """
        subject = self._session.get(Product, asin)
        if alternative.mean_rating is None or subject is None or subject.mean_rating is None:
            return None
        return _round(alternative.mean_rating - subject.mean_rating, 6)

    def _verdict_for(self, asin: str, risk: float, strength: float) -> VerdictLabel:
        """Derive the verdict label for any product from the configured bands.

        Args:
            asin: Product ASIN.
            risk: Risk score on 0-100.
            strength: Evidence strength on 0-1.

        Returns:
            One of the three locked labels.
        """
        return _verdict_label(
            risk, strength, _verified_share(self._all_reviews(asin)), self._thresholds
        )

    def _evidence_strength_for(self, asin: str) -> float:
        """Compute the real evidence strength for any product.

        Args:
            asin: Product ASIN.

        Returns:
            The evidence strength on 0-1.
        """
        reviews = self._all_reviews(asin)
        rating = _rating_distribution(reviews)
        sentiment = _sentiment_breakdown(reviews)
        return _evidence_indicators(reviews, rating, sentiment, self._settings).evidence_strength

    def _risk_indicators(self, asin: str, evidence_strength: float) -> list[EvidenceIndicator]:
        """Build the risk stage's five criteria as top-level indicators.

        ``evidence_strength`` carries the real computed value; the other four are
        seeded placeholders.

        Args:
            asin: Product ASIN.
            evidence_strength: The product's real evidence strength.

        Returns:
            One indicator per 0-1 criterion, in config order.
        """
        indicators: list[EvidenceIndicator] = []
        for criterion in self._criteria.risk_criteria:
            if criterion.name == CriterionName.EVIDENCE_STRENGTH.value:
                value = float(evidence_strength)
                description = (
                    f"{criterion.description}; mirrors evidence.evidence_strength "
                    f"({evidence_strength})."
                )
            else:
                value = _round(_unit(_criterion_seed(asin, criterion.name)), _DEGREE_DIGITS)
                description = (
                    f"{criterion.description}. Placeholder value seeded from the ASIN; "
                    "the risk stage that computes this criterion has not run yet."
                )
            indicators.append(
                EvidenceIndicator(
                    name=criterion.name,
                    value=value,
                    raw_value=value,
                    direction=IndicatorDirection(criterion.direction),
                    description=description,
                )
            )
        return indicators


def _first_review_time(reviews: Sequence[Review], fallback: datetime) -> datetime:
    """Return the oldest review timestamp.

    Args:
        reviews: The product's review rows.
        fallback: Value used when the product has no reviews (the dataset
            ``reference_date``, so the result never depends on the wall clock).

    Returns:
        A UTC datetime.
    """
    if not reviews:
        return fallback
    return min(review.review_time for review in reviews)


def _last_review_time(reviews: Sequence[Review], fallback: datetime) -> datetime:
    """Return the newest review timestamp.

    Args:
        reviews: The product's review rows.
        fallback: Value used when the product has no reviews (the dataset
            ``reference_date``, so the result never depends on the wall clock).

    Returns:
        A UTC datetime.
    """
    if not reviews:
        return fallback
    return max(review.review_time for review in reviews)


def _verified_share(reviews: Sequence[Review]) -> float:
    """Share of reviews explicitly flagged as verified purchases.

    Args:
        reviews: The product's review rows.

    Returns:
        A share on 0-1; NULL flags are not counted as True.
    """
    return _share(sum(1 for review in reviews if review.verified is True), len(reviews))


def _relation(subject: VerdictLabel, alternative: VerdictLabel) -> AlternativeRelation:
    """Classify an alternative by verdict severity.

    Args:
        subject: The subject's verdict label.
        alternative: The alternative's verdict label.

    Returns:
        ``better`` when the alternative's label is less severe, ``worse`` when
        more severe, ``comparable`` when identical.
    """
    subject_rank = VERDICT_LABELS.index(str(subject))
    alternative_rank = VERDICT_LABELS.index(str(alternative))
    if alternative_rank < subject_rank:
        return AlternativeRelation.BETTER
    if alternative_rank > subject_rank:
        return AlternativeRelation.WORSE
    return AlternativeRelation.COMPARABLE


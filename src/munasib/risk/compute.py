"""Assemble per-product decision criteria and risk factors, plus the run summary.

The stage reads only stored tables (``products``, ``reviews`` +
``review_features``, ``product_features`` and ``product_evidence``): no model
runs, no network and no clock. Recency is anchored on the dataset reference date
and ``computed_at`` is the dataset build timestamp, so two runs over the same
cache produce byte-identical rows.

Criteria (formulas in ``docs/risk_indicators.md``):

* ``product_quality`` - weighted mean of the shrunk mean rating, the shrunk mean
  net sentiment and the stored recent-sentiment shift.
* ``evidence_strength`` - the stored composite, unchanged.
* ``rating_reliability`` - agreement, non-polarization, rating/text consistency
  and helpfulness agreement.
* ``negative_feedback_concentration`` - shrunk negative prevalence times the
  focus of the negative feedback (cost).
* ``price_position`` - the stored category price percentile (cost).

Category priors are **product-macro means over TRAIN-split products only**, and
they are computed over the full frame even when only a subset of products is
written (``--limit``/``--asins``), so a limited run never changes a prior.

The helpfulness agreement component saturates at exactly 0 when the absolute
weighted/unweighted gap reaches ``agreement_scale``. When more than
``agreement_scale_max_zero_share`` of the written products would sit at 0, the
smallest candidate scale that avoids that saturation is used instead; the choice
and the per-candidate zero shares are reported, never fitted to ground truth.

The printed summary is also written, from computed values only, to
``reports/risk_indicators_summary.json`` and ``.md``; only the dataset's own
timestamps (reference date and build time) appear there, never the wall clock.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from munasib.core.config import RiskConfig
from munasib.core.exceptions import DataError
from munasib.risk.factors import FACTOR_CODES, SEVERITIES, RiskFactor, generate_risk_factors
from munasib.risk.indicators import (
    CRITERION_ORDER,
    Criterion,
    category_priors,
    criteria_vector,
    evidence_strength_criterion,
    negative_feedback_concentration_criterion,
    price_position_criterion,
    product_quality_criterion,
    rating_reliability_criterion,
)

DIGITS = 6

#: Report file names written under ``paths.reports_dir``.
JSON_FILENAME = "risk_indicators_summary.json"
MARKDOWN_FILENAME = "risk_indicators_summary.md"

#: Short labels used in the correlation matrix rendering.
CRITERION_LABELS: Mapping[str, str] = {
    "product_quality": "pq",
    "evidence_strength": "es",
    "rating_reliability": "rr",
    "negative_feedback_concentration": "nfc",
    "price_position": "pp",
}


def _iso(value: datetime | None) -> str | None:
    """Render a naive UTC datetime as an ISO 8601 string.

    Args:
        value: Naive UTC datetime, or None when the run carried none.

    Returns:
        ISO 8601 string with ``+00:00``, or None.
    """
    if value is None:
        return None
    return value.replace(tzinfo=UTC).isoformat()


def _json_ready(value: Any) -> Any:
    """Convert a summary value into a JSON-safe, deterministically rounded copy.

    Floats are rounded to :data:`DIGITS` so two runs over the same cache
    serialise byte-identically, datetimes become ISO 8601 strings and mappings
    and sequences are converted recursively.

    Args:
        value: Any summary value.

    Returns:
        The JSON-safe equivalent.
    """
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, float):
        return _rounded(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _stat_payload(stat: Stat) -> dict[str, Any]:
    """Render one statistic as a JSON-safe min/median/max mapping.

    Args:
        stat: The statistic (rounding happens in :func:`_json_ready`).

    Returns:
        Mapping with ``count``, ``min``, ``median`` and ``max``.
    """
    return {
        "count": stat.count,
        "min": stat.minimum,
        "median": stat.median,
        "max": stat.maximum,
    }


def _as_int(value: Any) -> int:
    """Convert a stored count to ``int``, treating missing/NaN as 0.

    Args:
        value: Stored JSON/scalar value.

    Returns:
        The integer value, or 0 when it is missing or not numeric.
    """
    number = _as_float(value)
    return 0 if number is None else int(number)


def _indicator_entry(
    indicators: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    """Fetch one stored indicator entry from a product's payload.

    Args:
        indicators: The ``indicators`` mapping of the stored evidence payload.
        name: Indicator name.

    Returns:
        The indicator mapping, or an empty mapping when it is absent.
    """
    entry = indicators.get(name)
    return entry if isinstance(entry, Mapping) else {}


def _validate_timestamp(value: datetime, label: str) -> datetime:
    """Reject timezone-aware timestamps (the cache stores naive UTC).

    Args:
        value: Timestamp to check.
        label: Argument name used in the error message.

    Returns:
        The timestamp unchanged.

    Raises:
        ValueError: If the timestamp carries a timezone.
    """
    if value.tzinfo is not None:
        raise ValueError(
            f"{label} must be naive UTC (the cache stores naive timestamps), got {value!r}"
        )
    return value


def _rounded(value: float | None) -> float | None:
    """Round a float to :data:`DIGITS` decimals, preserving ``None``.

    Args:
        value: Value to round.

    Returns:
        The rounded value, or ``None``.
    """
    return None if value is None else round(float(value), DIGITS)


def _as_float(value: Any) -> float | None:
    """Convert a stored value to float, treating NaN and non-numbers as missing.

    Args:
        value: Stored JSON/scalar value.

    Returns:
        The float value, or ``None``.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return None if pd.isna(number) else number
    return None


@dataclass(frozen=True)
class AgreementScaleChoice:
    """Outcome of the helpfulness-agreement saturation rule.

    Attributes:
        scale: The scale actually used.
        zero_share: Share of products whose agreement component is exactly 0.
        median_component: Median agreement component at the chosen scale.
        median_gap: Median absolute weighted/unweighted gap.
        zero_share_by_candidate: Zero share at every candidate scale.
        gap_count: Products with a usable gap.
        meets_target: Whether some candidate kept the zero share under the cap.
    """

    scale: float
    zero_share: float
    median_component: float | None
    median_gap: float | None
    zero_share_by_candidate: Mapping[float, float]
    gap_count: int
    meets_target: bool

    def to_payload(self) -> dict[str, Any]:
        """Render a JSON-safe report of the choice.

        Returns:
            Mapping with the chosen scale and the per-candidate zero shares.
        """
        return {
            "scale": self.scale,
            "zero_share": _rounded(self.zero_share),
            "median_component": _rounded(self.median_component),
            "median_gap": _rounded(self.median_gap),
            "zero_share_by_candidate": {
                str(scale): _rounded(share)
                for scale, share in self.zero_share_by_candidate.items()
            },
            "gap_count": self.gap_count,
            "meets_target": self.meets_target,
        }


def agreement_component(gap: float, scale: float) -> float:
    """Helpfulness agreement component for one absolute gap.

    ``component = 1 - min(1, gap / scale)``, so the component is exactly 0
    once the gap reaches the scale.

    Args:
        gap: ``abs(helpfulness_weighted_net - unweighted_net)``.
        scale: Agreement scale (positive).

    Returns:
        The component in ``[0, 1]``.
    """
    if scale <= 0:
        raise ValueError(f"agreement scale must be positive, got {scale}")
    return 1.0 - min(1.0, abs(float(gap)) / scale)


def choose_agreement_scale(
    gaps: Sequence[float],
    *,
    configured: float,
    candidates: Sequence[float],
    max_zero_share: float,
) -> AgreementScaleChoice:
    """Pick the smallest candidate scale that keeps saturation under the cap.

    Args:
        gaps: Absolute weighted/unweighted net gaps of the written products.
        configured: Configured scale (must be one of ``candidates``).
        candidates: Candidate scales, ascending.
        max_zero_share: Share at exactly 0 above which the next scale is used.

    Returns:
        The chosen scale with its diagnostics.

    Raises:
        ValueError: If the scale or candidates are invalid, ``configured`` is
            not a candidate, or the cap is outside ``[0, 1)``.
    """
    if configured <= 0:
        raise ValueError(f"configured agreement scale must be positive, got {configured}")
    if not candidates:
        raise ValueError("agreement scale candidates must not be empty")
    if list(candidates) != sorted(candidates) or len(set(candidates)) != len(candidates):
        raise ValueError(f"agreement scale candidates must be strictly ascending, got {list(candidates)}")
    if configured not in candidates:
        raise ValueError(f"configured agreement scale {configured} is not in {list(candidates)}")
    if not 0.0 <= max_zero_share < 1.0:
        raise ValueError(
            f"max zero share must be in [0, 1), got {max_zero_share}"
        )

    values = [abs(float(gap)) for gap in gaps]
    count = len(values)
    zero_share_by_candidate: dict[float, float] = {}
    for scale in candidates:
        zeros = sum(1 for gap in values if agreement_component(gap, scale) == 0.0)
        zero_share_by_candidate[scale] = zeros / count if count else 0.0

    chosen: float | None = None
    for scale in candidates:
        if zero_share_by_candidate[scale] < max_zero_share:
            chosen = scale
            break
    meets_target = chosen is not None
    if chosen is None:
        chosen = candidates[-1]

    components = [agreement_component(gap, chosen) for gap in values]
    return AgreementScaleChoice(
        scale=chosen,
        zero_share=zero_share_by_candidate[chosen],
        median_component=statistics.median(components) if components else None,
        median_gap=statistics.median(values) if values else None,
        zero_share_by_candidate=dict(zero_share_by_candidate),
        gap_count=count,
        meets_target=meets_target,
    )


@dataclass(frozen=True)
class RiskRow:
    """One product's computed criteria and risk factors.

    Attributes:
        asin: Product ASIN.
        split: Dataset split (stored for report slicing; not a criterion).
        title: Product title (display only, never stored in the row).
        mean_rating: Product mean star rating (display only).
        n_reviews: Stored review count (display only).
        mean_net: Mean predicted net sentiment over predicted reviews.
        criteria: The five criteria keyed by name.
        risk_factors: Triggered factors in stable order.
        directions: Criterion -> ``benefit``/``cost``.
        priors: Train-split category priors.
        reference_date: Dataset reference date (naive UTC).
        computed_at: Dataset build timestamp (naive UTC).
    """

    asin: str
    split: str
    title: str
    mean_rating: float | None
    n_reviews: int
    mean_net: float | None
    criteria: Mapping[str, Criterion]
    risk_factors: tuple[RiskFactor, ...]
    directions: Mapping[str, str]
    priors: Mapping[str, Any]
    reference_date: datetime
    computed_at: datetime

    def to_payload(self) -> dict[str, object]:
        """Render the database row payload.

        Returns:
            ``asin``, ``indicators``, ``risk_factors``, ``priors``,
            ``reference_date`` and ``computed_at``.
        """
        return {
            "asin": self.asin,
            "indicators": {
                "split": self.split,
                "criteria": {
                    name: criterion.to_payload()
                    for name, criterion in self.criteria.items()
                },
                "vector": [_rounded(value) for value in criteria_vector(self.criteria)],
                "directions": dict(self.directions),
            },
            "risk_factors": [factor.to_payload() for factor in self.risk_factors],
            "priors": dict(self.priors),
            "reference_date": self.reference_date,
            "computed_at": self.computed_at,
        }


@dataclass(frozen=True)
class RiskResult:
    """Everything one run produced.

    Attributes:
        rows: Computed rows, ascending by ASIN.
        agreement: The agreement-scale decision.
        priors: The category priors every row was shrunk toward.
    """

    rows: tuple[RiskRow, ...]
    agreement: AgreementScaleChoice
    priors: Mapping[str, Any]


def _evidence_gaps(evidence: pd.DataFrame) -> list[float]:
    """Absolute helpfulness gaps over every product with a stored difference.

    The agreement scale is a dataset-level setting: it is chosen over the whole
    evidence table (not just the written subset) so a ``--limit`` run cannot
    store a different scale - and therefore different ``rating_reliability``
    values - than a full run.

    Args:
        evidence: Evidence frame with ``asin`` and ``indicators``.

    Returns:
        One gap per product carrying a stored weighted/unweighted difference.
    """
    gaps: list[float] = []
    for row in evidence.itertuples(index=False):
        payload = row.indicators
        if not isinstance(payload, Mapping):
            continue
        indicators = payload.get("indicators")
        if not isinstance(indicators, Mapping):
            continue
        entry = indicators.get("helpfulness_weighted_sentiment")
        if not isinstance(entry, Mapping):
            continue
        extras = entry.get("extras")
        if not isinstance(extras, Mapping):
            continue
        difference = _as_float(extras.get("difference"))
        if difference is not None:
            gaps.append(abs(difference))
    return gaps


def compute_risk_indicators(
    products: pd.DataFrame,
    sentiment: pd.DataFrame,
    evidence: pd.DataFrame,
    features: pd.DataFrame,
    *,
    reference_date: datetime,
    computed_at: datetime,
    config: RiskConfig,
    directions: Mapping[str, str],
    target_asins: Sequence[str] | None = None,
) -> RiskResult:
    """Compute every written product's criteria and risk factors.

    Args:
        products: Frame with ``asin``, ``split``, ``title``, ``mean_rating`` and
            ``n_reviews``. Must hold **every** product: the priors are computed
            from the full train split.
        sentiment: Output of
            :func:`munasib.risk.indicators.product_sentiment_totals` over every
            product.
        evidence: Frame with ``asin``, ``evidence_strength`` and ``indicators``.
        features: Frame with ``asin`` and ``indicators`` (product features).
        reference_date: Dataset reference date (naive UTC).
        computed_at: Dataset build timestamp (naive UTC).
        config: Risk configuration.
        directions: Criterion directions read from ``criteria_weights.yaml``.
        target_asins: ASINs to write, or None for every product.

    Returns:
        The rows (ascending by ASIN), the agreement-scale decision and priors.

    Raises:
        ValueError: If a frame lacks a required column or a timestamp is aware.
        DataError: If the train split cannot supply the priors or a target ASIN
            is not a stored product.
    """
    _validate_timestamp(reference_date, "reference_date")
    _validate_timestamp(computed_at, "computed_at")

    required_products = {"asin", "split", "title", "mean_rating", "n_reviews"}
    required_evidence = {"asin", "evidence_strength", "indicators"}
    required_features = {"asin", "indicators"}
    for label, frame, required in (
        ("products", products, required_products),
        ("evidence", evidence, required_evidence),
        ("features", features, required_features),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} frame lacks column(s) {sorted(missing)}")

    directions = dict(directions)
    for name in CRITERION_ORDER:
        if name not in directions:
            raise ValueError(f"directions map lacks criterion {name!r}")

    priors = category_priors(products, sentiment)
    agreement = choose_agreement_scale(
        _evidence_gaps(evidence),
        configured=config.criteria.rating_reliability.agreement_scale,
        candidates=config.criteria.rating_reliability.agreement_scale_candidates,
        max_zero_share=config.criteria.rating_reliability.agreement_scale_max_zero_share,
    )

    wanted = (
        set(products["asin"].astype(str))
        if target_asins is None
        else {str(asin) for asin in target_asins}
    )
    stored = set(products["asin"].astype(str))
    unknown = sorted(wanted - stored)
    if unknown:
        raise DataError(
            f"{len(unknown)} requested ASIN(s) are not stored products: {unknown[:5]}"
        )

    sentiment_by_asin = {
        str(row.asin): row for row in sentiment.itertuples(index=False)
    }
    evidence_by_asin = {str(row.asin): row for row in evidence.itertuples(index=False)}
    features_by_asin = {str(row.asin): row for row in features.itertuples(index=False)}

    prior_rating = float(priors["mean_rating"])
    prior_net = float(priors["net_sentiment"])
    prior_negative = float(priors["negative_share"])
    scale = agreement.scale

    rows: list[RiskRow] = []
    ordered = products.assign(
        row_asin=products["asin"].astype(str)
    ).sort_values("row_asin")
    for product in ordered.itertuples(index=False):
        asin = str(product.asin)
        if asin not in wanted:
            continue

        sent = sentiment_by_asin.get(asin)
        n_predictions = _as_int(sent.n_predictions) if sent is not None else 0
        mean_net = _as_float(sent.mean_net) if sent is not None else None
        negative_share = (
            _as_float(sent.negative_share) if sent is not None else None
        )

        evidence_row = evidence_by_asin.get(asin)
        strength = (
            _as_float(evidence_row.evidence_strength)
            if evidence_row is not None
            else None
        )
        payload = evidence_row.indicators if evidence_row is not None else None
        payload = payload if isinstance(payload, Mapping) else None
        low_evidence = payload.get("low_evidence") if payload else None
        raw_indicators = payload.get("indicators") if payload else None
        indicators: Mapping[str, Any] = (
            raw_indicators if isinstance(raw_indicators, Mapping) else {}
        )

        shift_entry = _indicator_entry(indicators, "recent_negative_shift")
        recent_shift = _as_float(shift_entry.get("normalized"))
        if recent_shift is None:
            recent_shift_reason = (
                str(shift_entry.get("missing_reason"))
                if shift_entry.get("missing_reason")
                else ("no_evidence_row" if payload is None else "recent_shift_missing")
            )
        else:
            recent_shift_reason = None

        quality = product_quality_criterion(
            mean_rating=_as_float(product.mean_rating),
            n_reviews=_as_int(product.n_reviews),
            mean_net=mean_net,
            n_predictions=n_predictions,
            recent_negative_shift=recent_shift,
            recent_negative_shift_reason=recent_shift_reason,
            prior_rating=prior_rating,
            prior_net=prior_net,
            config=config.criteria,
        )

        strength_criterion = evidence_strength_criterion(
            strength,
            low_evidence=bool(low_evidence) if low_evidence is not None else None,
            missing_reason=(
                None if payload is not None else "no_evidence_row"
            ),
        )

        agreement_entry = _indicator_entry(indicators, "rating_agreement")
        helpfulness_entry = _indicator_entry(indicators, "helpfulness_weighted_sentiment")
        helpfulness_extras = helpfulness_entry.get("extras")
        helpfulness_gap = (
            _as_float(helpfulness_extras.get("difference"))
            if isinstance(helpfulness_extras, Mapping)
            else None
        )
        if helpfulness_gap is not None:
            helpfulness_gap = abs(helpfulness_gap)
        reliability = rating_reliability_criterion(
            rating_agreement=_as_float(agreement_entry.get("normalized")),
            polarized=(
                bool(payload.get("polarized"))
                if payload is not None and "polarized" in payload
                else None
            ),
            rating_text_consistency=_as_float(
                _indicator_entry(indicators, "rating_text_agreement").get("normalized")
            ),
            helpfulness_gap=helpfulness_gap,
            config=config.criteria,
            agreement_scale=scale,
        )

        concentration = negative_feedback_concentration_criterion(
            n_predictions=n_predictions,
            negative_share=negative_share,
            prior_negative_share=prior_negative,
            top_pain_point_share=_as_float(
                _indicator_entry(indicators, "top_pain_point_share").get("raw_value")
            ),
            aspect_concentration=_as_float(
                _indicator_entry(indicators, "aspect_concentration").get("raw_value")
            ),
            temporal_burst=_as_float(_indicator_entry(indicators, "temporal_burst").get("raw_value")),
            config=config.criteria,
        )

        features_row = features_by_asin.get(asin)
        features_payload = (
            features_row.indicators if features_row is not None else None
        )
        features_payload = (
            features_payload if isinstance(features_payload, Mapping) else None
        )
        price = price_position_criterion(
            _as_float(
                features_payload.get("price_percentile_category")
                if features_payload
                else None
            ),
            price_missing=bool(features_payload.get("price_missing"))
            if features_payload and "price_missing" in features_payload
            else True,
            peer_group=features_payload.get("price_peer_group")
            if features_payload
            else None,
            peer_count=(
                int(features_payload["price_peer_count"])
                if features_payload
                and isinstance(features_payload.get("price_peer_count"), int)
                else None
            ),
            peer_fallback=(
                bool(features_payload.get("price_peer_fallback"))
                if features_payload and "price_peer_fallback" in features_payload
                else None
            ),
            config=config.criteria,
        )

        criteria = {
            "product_quality": _with_direction(quality, directions),
            "evidence_strength": _with_direction(strength_criterion, directions),
            "rating_reliability": _with_direction(reliability, directions),
            "negative_feedback_concentration": _with_direction(
                concentration, directions
            ),
            "price_position": _with_direction(price, directions),
        }

        factors = generate_risk_factors(
            criteria=criteria,
            evidence_strength=strength,
            low_evidence=bool(low_evidence) if low_evidence is not None else None,
            evidence_payload=payload,
            features_payload=features_payload,
            config=config.factors,
        )

        rows.append(
            RiskRow(
                asin=asin,
                split=str(product.split),
                title=str(product.title),
                mean_rating=_as_float(product.mean_rating),
                n_reviews=_as_int(product.n_reviews),
                mean_net=mean_net,
                criteria=criteria,
                risk_factors=factors,
                directions=directions,
                priors=priors,
                reference_date=reference_date,
                computed_at=computed_at,
            )
        )
    return RiskResult(rows=tuple(rows), agreement=agreement, priors=priors)


def _with_direction(criterion: Criterion, directions: Mapping[str, str]) -> Criterion:
    """Return the criterion stamped with the declared direction.

    Args:
        criterion: Computed criterion.
        directions: Directions read from ``criteria_weights.yaml``.

    Returns:
        The criterion with its declared direction (a mismatch would mean the
        computation and the config disagree, so it is rejected).
    """
    declared = directions[criterion.name]
    if declared != criterion.direction:
        raise DataError(
            f"criterion {criterion.name!r} was computed as {criterion.direction!r} "
            f"but criteria_weights.yaml declares {declared!r}"
        )
    return criterion


@dataclass(frozen=True)
class Stat:
    """Min/median/max of one labelled series.

    Attributes:
        count: Non-missing values.
        minimum: Smallest value, or None when ``count`` is 0.
        median: Median, or None when ``count`` is 0.
        maximum: Largest value, or None when ``count`` is 0.
    """

    count: int
    minimum: float | None
    median: float | None
    maximum: float | None

    def label(self) -> str:
        """Render the statistic as ``min=... median=... max=... (n=...)``.

        Returns:
            One compact label.
        """
        if self.count == 0:
            return "n=0 (no values)"
        return (
            f"min={self.minimum:.4f} median={self.median:.4f} "
            f"max={self.maximum:.4f} (n={self.count:,})"
        )


def _stat(values: Sequence[float | None]) -> Stat:
    """Build a :class:`Stat` from possibly-missing values.

    Args:
        values: Values (None entries are dropped).

    Returns:
        The statistic.
    """
    present = [
        float(value)
        for value in values
        if value is not None and not pd.isna(value)
    ]
    if not present:
        return Stat(count=0, minimum=None, median=None, maximum=None)
    return Stat(
        count=len(present),
        minimum=min(present),
        median=float(statistics.median(present)),
        maximum=max(present),
    )


def _spearman(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Pairwise-complete Spearman matrix over the five criteria.

    Each pair is ranked and correlated on the rows where **both** values are
    present, so a product missing one criterion cannot distort a pair it does
    not belong to.

    Args:
        frame: One column per criterion, NaN for missing values.

    Returns:
        Nested mapping criterion -> criterion -> rho (None when a pair has
        fewer than two complete observations or a constant series).
    """
    result: dict[str, dict[str, float | None]] = {}
    for row in CRITERION_ORDER:
        result[row] = {}
        for column in CRITERION_ORDER:
            if row == column:
                result[row][column] = 1.0 if frame[row].notna().sum() >= 2 else None
                continue
            pair = frame[[row, column]].dropna()
            if len(pair) < 2 or pair[row].nunique() < 2 or pair[column].nunique() < 2:
                result[row][column] = None
                continue
            value = pair[row].corr(pair[column], method="spearman")
            result[row][column] = None if pd.isna(value) else round(float(value), 4)
    return result


@dataclass(frozen=True)
class RiskSummary:
    """Everything the CLI prints (and tests assert) about one run.

    Attributes:
        products: Rows written.
        priors: Train-split category priors the values were shrunk toward.
        criteria_stats: Value min/median/max per criterion.
        missing: Criterion -> missing reason -> product count.
        cost_raw: Cost inputs with their raw and normalized statistics.
        correlations: 5x5 Spearman over all written products.
        train_correlations: 5x5 Spearman over the written train-split products.
        split_medians: Split -> criterion -> median.
        factor_counts: Factor -> number of products triggering it.
        factor_severities: Factor -> severity -> count.
        agreement: The agreement-scale choice.
        best: Five highest-``product_quality`` products.
        worst: Five lowest-``product_quality`` products.
        reference_date: Dataset reference date (naive UTC), or None for an
            empty run.
        computed_at: Dataset build timestamp (naive UTC), or None for an
            empty run.
        notes: Flagged correlations and factor shares.
    """

    products: int
    priors: Mapping[str, Any]
    criteria_stats: Mapping[str, Stat]
    missing: Mapping[str, Mapping[str, int]]
    cost_raw: Mapping[str, Mapping[str, Stat]]
    correlations: Mapping[str, Mapping[str, float | None]]
    train_correlations: Mapping[str, Mapping[str, float | None]]
    split_medians: Mapping[str, Mapping[str, float | None]]
    factor_counts: Mapping[str, int]
    factor_severities: Mapping[str, Mapping[str, int]]
    agreement: AgreementScaleChoice
    best: tuple[Mapping[str, Any], ...]
    worst: tuple[Mapping[str, Any], ...]
    reference_date: datetime | None = None
    computed_at: datetime | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def lines(self) -> list[str]:
        """Render the human-readable summary.

        Returns:
            One line per report row.
        """
        lines: list[str] = []
        lines.append(f"products written                   : {self.products:,}")
        if self.priors:
            lines.append(
                f"priors (train split)               : "
                f"mean_rating={self.priors.get('mean_rating')} "
                f"net={self.priors.get('net_sentiment'):+.4f} "
                f"negative={self.priors.get('negative_share'):.4f} "
                f"(train={self.priors.get('n_train_products'):,} products, "
                f"rated={self.priors.get('n_train_with_rating'):,}, "
                f"predicted={self.priors.get('n_train_with_predictions'):,})"
            )
        choice = self.agreement
        candidates = ", ".join(
            f"{scale:g}={share:.1%}"
            for scale, share in choice.zero_share_by_candidate.items()
        )
        median_component = (
            f"{choice.median_component:.4f}"
            if choice.median_component is not None
            else "n/a"
        )
        lines.append(
            f"agreement scale                    : {choice.scale:g} "
            f"(zero share {choice.zero_share:.1%} over {choice.gap_count:,} products, "
            f"median component {median_component}; candidates {candidates})"
        )
        lines.append("")
        lines.append("criterion values (higher = riskier for cost directions):")
        for name in CRITERION_ORDER:
            lines.append(
                f"  {name:<31} : {self.criteria_stats[name].label()}"
            )
        lines.append("")
        lines.append("missing criteria (None, never imputed):")
        for name in CRITERION_ORDER:
            reasons = self.missing.get(name, {})
            total = sum(reasons.values())
            if not reasons:
                lines.append(f"  {name:<31} : 0")
            else:
                detail = ", ".join(
                    f"{reason}={count:,}" for reason, count in sorted(reasons.items())
                )
                lines.append(f"  {name:<31} : {total:,} ({detail})")
        lines.append("")
        lines.append("cost-direction indicators: value and raw inputs")
        lines.append(
            "  (raw and stored normalized values are labelled separately; the "
            "stored"
        )
        lines.append("   normalization of the three evidence inputs is 1 - raw)")
        for name in (
            "negative_feedback_concentration",
            "price_position",
            "top_pain_point_share",
            "aspect_concentration",
            "temporal_burst",
        ):
            parts = self.cost_raw.get(name, {})
            rendered = " | ".join(
                f"{label} {stat.label()}" for label, stat in parts.items()
            )
            lines.append(f"  {name:<31} : {rendered}")
        lines.append("")
        lines.append("spearman correlations (all products, pairwise complete):")
        lines.extend(self._matrix_lines(self.correlations))
        lines.append("")
        lines.append("spearman correlations (train split only, pairwise complete):")
        lines.extend(self._matrix_lines(self.train_correlations))
        lines.append("")
        lines.append("median criterion value by split:")
        for split in sorted(self.split_medians):
            rendered = ", ".join(
                f"{CRITERION_LABELS[name]}="
                f"{'n/a' if value is None else f'{value:.4f}'}"
                for name, value in self.split_medians[split].items()
            )
            lines.append(f"  {split:<5} : {rendered}")
        lines.append("")
        lines.append("risk factors (severity counts and share of products):")
        for code in FACTOR_CODES:
            count = self.factor_counts.get(code, 0)
            share = count / self.products if self.products else 0.0
            severities = self.factor_severities.get(code, {})
            detail = " ".join(
                f"{severity}={severities.get(severity, 0):,}" for severity in SEVERITIES
            )
            lines.append(
                f"  {code:<38} : {count:,} ({share:.2%}) {detail}"
            )
        if self.notes:
            lines.append("")
            lines.append("notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        lines.append("")
        lines.append("best 5 by product_quality:")
        lines.extend(self._product_lines(self.best))
        lines.append("worst 5 by product_quality:")
        lines.extend(self._product_lines(self.worst))
        return lines

    def to_payload(self) -> dict[str, Any]:
        """Render the JSON report payload (computed values only).

        Returns:
            Mapping with the run metadata, the train priors and their counts,
            the agreement-scale decision with every candidate's saturation
            share, min/median/max of every criterion plus the raw and
            normalized cost inputs, missing counts with reasons, both Spearman
            matrices, the per-split medians, the factor counts by severity, the
            notes and the five best/worst products. The only timestamps are the
            dataset's own (reference date and build time); no wall-clock value
            is included.
        """
        factors = {
            code: {
                "count": self.factor_counts.get(code, 0),
                "share": (
                    round(self.factor_counts.get(code, 0) / self.products, DIGITS)
                    if self.products
                    else 0.0
                ),
                "severities": {
                    severity: self.factor_severities.get(code, {}).get(severity, 0)
                    for severity in SEVERITIES
                },
            }
            for code in FACTOR_CODES
        }
        missing = {
            name: {
                "count": sum(reasons.values()),
                "reasons": dict(sorted(reasons.items())),
            }
            for name, reasons in self.missing.items()
        }
        payload = {
            "products": self.products,
            "reference_date": self.reference_date,
            "computed_at": self.computed_at,
            "priors": dict(self.priors),
            "agreement": self.agreement.to_payload(),
            "criteria": {
                name: _stat_payload(self.criteria_stats[name])
                for name in CRITERION_ORDER
            },
            "cost_indicators": {
                name: {label: _stat_payload(stat) for label, stat in parts.items()}
                for name, parts in self.cost_raw.items()
            },
            "missing": missing,
            "correlations": self.correlations,
            "train_correlations": self.train_correlations,
            "split_medians": self.split_medians,
            "factors": factors,
            "best": list(self.best),
            "worst": list(self.worst),
            "notes": list(self.notes),
        }
        return cast(dict[str, Any], _json_ready(payload))

    def _matrix_lines(
        self, matrix: Mapping[str, Mapping[str, float | None]]
    ) -> list[str]:
        """Render one labelled Spearman matrix.

        Args:
            matrix: Criterion -> criterion -> rho.

        Returns:
            Header, key and one line per criterion.
        """
        header = "        " + " ".join(
            f"{CRITERION_LABELS[name]:>6}" for name in CRITERION_ORDER
        )
        lines = [
            "        key: "
            + " ".join(f"{CRITERION_LABELS[name]}={name}" for name in CRITERION_ORDER),
            header,
        ]
        for row in CRITERION_ORDER:
            cells = []
            for column in CRITERION_ORDER:
                value = matrix.get(row, {}).get(column)
                cells.append(f"{'n/a':>6}" if value is None else f"{value:>6.3f}")
            lines.append(f"  {CRITERION_LABELS[row]:>4}  " + " ".join(cells))
        return lines

    @staticmethod
    def _product_lines(entries: Sequence[Mapping[str, Any]]) -> list[str]:
        """Render best/worst product rows.

        Args:
            entries: Mappings with ``asin``, ``title``, ``mean_rating``,
                ``n_reviews``, ``net_sentiment`` and ``value``.

        Returns:
            One line per product.
        """
        if not entries:
            return ["  (no product has a product_quality value)"]
        lines = []
        for entry in entries:
            rating = (
                f"{entry['mean_rating']:.2f}/5"
                if entry["mean_rating"] is not None
                else "unrated"
            )
            net = (
                f"{entry['net_sentiment']:+.4f}"
                if entry["net_sentiment"] is not None
                else "n/a"
            )
            title = str(entry["title"])
            if len(title) > 52:
                title = title[:49] + "..."
            lines.append(
                f"  {entry['asin']}  quality={entry['value']:.4f}  {rating}  "
                f"n={entry['n_reviews']:,}  net={net}  {title}"
            )
        return lines


def summarize(
    rows: Sequence[RiskRow],
    evidence: pd.DataFrame,
    agreement: AgreementScaleChoice,
    priors: Mapping[str, Any] | None = None,
) -> RiskSummary:
    """Summarize a run: distributions, missing counts, matrices and factors.

    Args:
        rows: Computed rows.
        evidence: The same evidence frame that fed the computation (for the
            cost-indicator raw inputs).
        agreement: The agreement-scale choice.
        priors: Category priors (defaults to the first row's copy).

    Returns:
        The summary.
    """
    products = len(rows)
    frame = pd.DataFrame(
        {
            name: [row.criteria[name].value for row in rows]
            for name in CRITERION_ORDER
        },
        columns=list(CRITERION_ORDER),
        dtype="float64",
    )
    criteria_stats = {
        name: _stat(frame[name].tolist()) for name in CRITERION_ORDER
    }

    missing: dict[str, dict[str, int]] = {name: {} for name in CRITERION_ORDER}
    for row in rows:
        for name in CRITERION_ORDER:
            criterion = row.criteria[name]
            if criterion.value is None:
                reason = criterion.missing_reason or "unavailable"
                bucket = missing[name]
                bucket[reason] = bucket.get(reason, 0) + 1

    # Cost-direction indicators: value plus raw/normalized inputs, kept apart.
    concentration_values = [
        row.criteria["negative_feedback_concentration"].value for row in rows
    ]
    prevalence_values = [
        _as_float(
            row.criteria["negative_feedback_concentration"].extras.get("prevalence")
        )
        for row in rows
    ]
    focus_values = [
        _as_float(row.criteria["negative_feedback_concentration"].extras.get("focus"))
        for row in rows
    ]
    price_values = [row.criteria["price_position"].value for row in rows]
    price_normalized = [
        None if value is None else 1.0 - float(value) for value in price_values
    ]
    concentration_normalized = [
        None if value is None else 1.0 - float(value) for value in concentration_values
    ]
    raw_inputs: dict[str, dict[str, Stat]] = {
        "negative_feedback_concentration": {
            "value (cost)": _stat(concentration_values),
            "normalized (1 - value)": _stat(concentration_normalized),
            "prevalence": _stat(prevalence_values),
            "focus": _stat(focus_values),
        },
        "price_position": {
            "value (cost percentile)": _stat(price_values),
            "normalized (1 - value)": _stat(price_normalized),
        },
    }
    raw_names = ("top_pain_point_share", "aspect_concentration", "temporal_burst")
    stored_inputs: dict[str, dict[str, tuple[float | None, float | None]]] = {}
    for evidence_row in evidence.itertuples(index=False):
        payload = evidence_row.indicators
        if not isinstance(payload, Mapping):
            continue
        indicators = payload.get("indicators")
        if not isinstance(indicators, Mapping):
            continue
        values: dict[str, tuple[float | None, float | None]] = {}
        for name in raw_names:
            entry = indicators.get(name)
            if isinstance(entry, Mapping):
                values[name] = (
                    _as_float(entry.get("raw_value")),
                    _as_float(entry.get("normalized")),
                )
        stored_inputs[str(evidence_row.asin)] = values
    for name in raw_names:
        raws = [stored_inputs.get(row.asin, {}).get(name, (None, None))[0] for row in rows]
        normalizeds = [
            stored_inputs.get(row.asin, {}).get(name, (None, None))[1] for row in rows
        ]
        label = "raw (HHI)" if name == "aspect_concentration" else "raw"
        raw_inputs[name] = {
            label: _stat(raws),
            "normalized (1 - raw)": _stat(normalizeds),
        }

    train_frame = pd.DataFrame(
        {
            name: [
                row.criteria[name].value for row in rows if row.split == "train"
            ]
            for name in CRITERION_ORDER
        },
        columns=list(CRITERION_ORDER),
        dtype="float64",
    )
    correlations = _spearman(frame)
    train_correlations = _spearman(train_frame)

    split_medians: dict[str, dict[str, float | None]] = {}
    for split in sorted({row.split for row in rows}):
        split_rows = [row for row in rows if row.split == split]
        split_medians[split] = {
            name: (
                _stat([row.criteria[name].value for row in split_rows]).median
            )
            for name in CRITERION_ORDER
        }

    factor_counts: dict[str, int] = dict.fromkeys(FACTOR_CODES, 0)
    factor_severities: dict[str, dict[str, int]] = {
        code: dict.fromkeys(SEVERITIES, 0) for code in FACTOR_CODES
    }
    for row in rows:
        for factor in row.risk_factors:
            factor_counts[factor.code] += 1
            factor_severities[factor.code][factor.severity] += 1

    def entries(order: Sequence[RiskRow]) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            {
                "asin": row.asin,
                "title": row.title,
                "mean_rating": row.mean_rating,
                "n_reviews": row.n_reviews,
                "net_sentiment": row.mean_net,
                "value": row.criteria["product_quality"].value,
            }
            for row in order
        )

    ranked = [row for row in rows if row.criteria["product_quality"].value is not None]
    best = entries(
        sorted(
            ranked,
            key=lambda row: (
                -float(row.criteria["product_quality"].value),  # type: ignore[arg-type]
                row.asin,
            ),
        )[:5]
    )
    worst = entries(
        sorted(
            ranked,
            key=lambda row: (
                float(row.criteria["product_quality"].value),  # type: ignore[arg-type]
                row.asin,
            ),
        )[:5]
    )

    notes: list[str] = []
    for row_name in CRITERION_ORDER:
        for column in CRITERION_ORDER:
            if column <= row_name:
                continue
            value = correlations.get(row_name, {}).get(column)
            if value is not None and abs(value) > 0.8:
                notes.append(
                    f"criteria {row_name!r} and {column!r} correlate at rho="
                    f"{value:+.4f} (>|0.8|), so they are not independent signals"
                )
    if products:
        for code in FACTOR_CODES:
            share = factor_counts[code] / products
            if share > 0.60:
                notes.append(
                    f"factor {code!r} triggers for {share:.1%} of products (>60%), "
                    "so it is close to a constant warning"
                )
            elif share < 0.005:
                notes.append(
                    f"factor {code!r} triggers for {share:.2%} of products (<0.5%), "
                    "so it is close to dead"
                )
    if not agreement.meets_target:
        notes.append(
            "no candidate agreement scale kept the saturation share under the cap; "
            f"the largest ({agreement.scale:g}) was used"
        )

    return RiskSummary(
        products=products,
        priors=dict(priors) if priors is not None else (dict(rows[0].priors) if rows else {}),
        criteria_stats=criteria_stats,
        missing=missing,
        cost_raw=raw_inputs,
        correlations=correlations,
        train_correlations=train_correlations,
        split_medians=split_medians,
        factor_counts=factor_counts,
        factor_severities=factor_severities,
        agreement=agreement,
        best=best,
        worst=worst,
        reference_date=rows[0].reference_date if rows else None,
        computed_at=rows[0].computed_at if rows else None,
        notes=tuple(notes),
    )


def _md_cell(value: Any) -> str:
    """Render one summary value as a Markdown table cell.

    Args:
        value: Any JSON-safe summary value.

    Returns:
        The rendered cell (never empty).
    """
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _matrix_markdown(matrix: Mapping[str, Mapping[str, float | None]]) -> list[str]:
    """Render one labelled Spearman matrix as a Markdown table.

    Args:
        matrix: Criterion -> criterion -> rho.

    Returns:
        A key line plus one row per criterion.
    """
    key = "Key: " + ", ".join(
        f"{CRITERION_LABELS[name]}={name}" for name in CRITERION_ORDER
    )
    header = "| | " + " | ".join(
        f"`{CRITERION_LABELS[name]}`" for name in CRITERION_ORDER
    ) + " |"
    lines = [
        key,
        header,
        "|---|" + "---:|" * len(CRITERION_ORDER),
    ]
    for row in CRITERION_ORDER:
        cells = []
        for column in CRITERION_ORDER:
            value = matrix.get(row, {}).get(column)
            cells.append("n/a" if value is None else f"{value:.3f}")
        lines.append(f"| `{CRITERION_LABELS[row]}` | " + " | ".join(cells) + " |")
    return lines


def render_summary_markdown(summary: RiskSummary) -> str:
    """Render the Markdown report from the same figures as the JSON report.

    Args:
        summary: Run summary.

    Returns:
        GitHub-flavoured Markdown.
    """
    lines = [
        "# Risk indicators summary",
        "",
        "Computed from the stored cache by `python scripts/compute_risk_indicators.py`; "
        "every number below is produced by the code, none is illustrative.",
        "",
        f"- **Products written:** {summary.products:,}",
        f"- **Reference date (recency anchor):** {_iso(summary.reference_date) or 'n/a'} "
        "- the latest review timestamp in `data/processed/meta.json`, never today's "
        "date.",
        f"- **Computed at:** {_iso(summary.computed_at) or 'n/a'} - the dataset build "
        "timestamp, never the wall clock.",
        "",
        "## Category priors (train split)",
        "",
        "| Prior | Value |",
        "|---|---:|",
    ]
    for name, value in summary.priors.items():
        lines.append(f"| `{name}` | {_md_cell(value)} |")
    choice = summary.agreement
    lines += [
        "",
        "## Agreement scale",
        "",
        f"- **Chosen scale:** {choice.scale:g}",
        f"- **Share of products saturated at 0:** {choice.zero_share:.2%} over "
        f"{choice.gap_count:,} products",
        f"- **Median component:** {_md_cell(_rounded(choice.median_component))}",
        f"- **Median gap:** {_md_cell(_rounded(choice.median_gap))}",
        f"- **Kept the saturation share under the cap:** "
        f"{str(choice.meets_target).lower()}",
        "",
        "| Candidate scale | Share at 0 |",
        "|---:|---:|",
    ]
    for scale, share in choice.zero_share_by_candidate.items():
        lines.append(f"| {scale:g} | {share:.2%} |")
    lines += [
        "",
        "## Criterion values",
        "",
        "| Criterion | n | min | median | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CRITERION_ORDER:
        stat = summary.criteria_stats[name]
        lines.append(
            f"| `{name}` | {stat.count:,} | {_md_cell(_rounded(stat.minimum))} | "
            f"{_md_cell(_rounded(stat.median))} | {_md_cell(_rounded(stat.maximum))} |"
        )
    lines += [
        "",
        "## Cost-direction inputs (raw and normalized)",
        "",
        "| Indicator | Input | n | min | median | max |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, parts in summary.cost_raw.items():
        for label, stat in parts.items():
            lines.append(
                f"| `{name}` | {label} | {stat.count:,} | "
                f"{_md_cell(_rounded(stat.minimum))} | "
                f"{_md_cell(_rounded(stat.median))} | "
                f"{_md_cell(_rounded(stat.maximum))} |"
            )
    lines += [
        "",
        "## Missing values (None, never imputed)",
        "",
        "| Criterion | Products | Reasons |",
        "|---|---:|---|",
    ]
    for name in CRITERION_ORDER:
        reasons = summary.missing.get(name, {})
        total = sum(reasons.values())
        detail = ", ".join(
            f"`{reason}`={count:,}" for reason, count in sorted(reasons.items())
        )
        lines.append(f"| `{name}` | {total:,} | {detail or '-'} |")
    lines += [
        "",
        "## Spearman correlations (all products, pairwise complete)",
        "",
    ]
    lines += _matrix_markdown(summary.correlations)
    lines += [
        "",
        "## Spearman correlations (train split only, pairwise complete)",
        "",
    ]
    lines += _matrix_markdown(summary.train_correlations)
    lines += [
        "",
        "## Median criterion value by split",
        "",
        "| Split | " + " | ".join(f"`{CRITERION_LABELS[n]}`" for n in CRITERION_ORDER) + " |",
        "|---|" + "---:|" * len(CRITERION_ORDER),
    ]
    for split in sorted(summary.split_medians):
        cells = [
            _md_cell(_rounded(summary.split_medians[split].get(name)))
            for name in CRITERION_ORDER
        ]
        lines.append(f"| {split} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Risk factors",
        "",
        "| Factor | Products | Share | " + " | ".join(SEVERITIES) + " |",
        "|---|---:|---:|" + "---:|" * len(SEVERITIES),
    ]
    for code in FACTOR_CODES:
        count = summary.factor_counts.get(code, 0)
        share = count / summary.products if summary.products else 0.0
        severities = summary.factor_severities.get(code, {})
        cells = [f"{severities.get(severity, 0):,}" for severity in SEVERITIES]
        lines.append(
            f"| `{code}` | {count:,} | {share:.2%} | " + " | ".join(cells) + " |"
        )
    if summary.notes:
        lines += ["", "## Notes", ""]
        lines.extend(f"- {note}" for note in summary.notes)
    for heading, entries in (
        ("Best 5 by product_quality", summary.best),
        ("Worst 5 by product_quality", summary.worst),
    ):
        lines += [
            "",
            f"## {heading}",
            "",
            "| ASIN | product_quality | mean_rating | n_reviews | net_sentiment | Title |",
            "|---|---:|---:|---:|---:|---|",
        ]
        if not entries:
            lines.append("| - | n/a | n/a | n/a | n/a | no product has a value |")
        for entry in entries:
            title = str(entry["title"]).replace("|", "\\|")
            lines.append(
                f"| {entry['asin']} | {_md_cell(_rounded(entry['value']))} | "
                f"{_md_cell(_rounded(entry['mean_rating']))} | "
                f"{_md_cell(entry['n_reviews'])} | "
                f"{_md_cell(_rounded(entry['net_sentiment']))} | {title} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_summary_reports(summary: RiskSummary, reports_dir: Path) -> list[Path]:
    """Write ``risk_indicators_summary.json`` and ``risk_indicators_summary.md``.

    Args:
        summary: Run summary.
        reports_dir: Target directory (created when missing).

    Returns:
        The two written paths ``[json_path, markdown_path]``.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / JSON_FILENAME
    markdown_path = reports_dir / MARKDOWN_FILENAME
    json_path.write_text(
        json.dumps(summary.to_payload(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_summary_markdown(summary), encoding="utf-8")
    return [json_path, markdown_path]


__all__ = [
    "CRITERION_LABELS",
    "JSON_FILENAME",
    "MARKDOWN_FILENAME",
    "AgreementScaleChoice",
    "RiskResult",
    "RiskRow",
    "RiskSummary",
    "Stat",
    "agreement_component",
    "choose_agreement_scale",
    "compute_risk_indicators",
    "render_summary_markdown",
    "summarize",
    "write_summary_reports",
]

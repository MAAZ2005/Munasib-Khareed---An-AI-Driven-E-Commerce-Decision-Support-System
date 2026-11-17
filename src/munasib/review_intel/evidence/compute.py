"""Assemble per-product evidence indicators, the composite and the run summary.

The stage reads only stored data (``reviews`` + ``review_features`` and
``product_aspects``): no model runs, no network, no clock. Every recency window
is anchored on the dataset reference date passed in by the caller, and
``computed_at`` is the dataset build timestamp, so two runs over the same cache
produce identical rows and identical reports.

The public entry points are :func:`compute_evidence` (one list of
:class:`EvidenceRow` for the selected products) and :func:`summarize` /
:func:`write_summary_reports` (the printed/stored summary). Every formula lives in
:mod:`munasib.review_intel.evidence.indicators`; see ``docs/evidence.md``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from munasib.core.config import EvidenceConfig
from munasib.core.constants import SENTIMENT_LABEL_NEGATIVE, SPLIT_ORDER
from munasib.review_intel.evidence.indicators import (
    COMPONENT_INDICATOR,
    DIGITS,
    STRENGTH_COMPONENT_KEYS,
    Indicator,
    StrengthResult,
    burst_indicator,
    composite_strength,
    concentration_indicators,
    helpfulness_indicators,
    rating_agreement_indicator,
    rating_text_agreement_indicator,
    recency_indicators,
    sentiment_shares_indicator,
    verified_purchase_indicator,
    volume_indicator,
)

#: Report filenames written under ``settings.paths.reports_dir``.
JSON_FILENAME = "evidence_summary.json"
MARKDOWN_FILENAME = "evidence_summary.md"

#: Flag names reported by the summary, in fixed order.
SUMMARY_FLAGS: tuple[str, ...] = (
    "polarized",
    "low_evidence",
    "no_recent_reviews",
    "temporal_burst",
)

_NUMERIC_COLUMNS = (
    "rating",
    "helpful_votes",
    "predicted_label",
    "prob_negative",
    "prob_neutral",
    "prob_positive",
)


@dataclass(frozen=True)
class EvidenceRow:
    """One product's evidence result, ready for ``product_evidence``."""

    asin: str
    split: str
    n_reviews: int
    evidence_strength: float | None
    indicators: dict[str, Any]
    reference_date: datetime
    computed_at: datetime

    def to_payload(self) -> dict[str, object]:
        """Render the database row payload.

        Returns:
            Mapping with every ``product_evidence`` column (timestamps are naive
            UTC, matching the serving-cache contract).
        """
        return {
            "asin": self.asin,
            "n_reviews": self.n_reviews,
            "evidence_strength": self.evidence_strength,
            "indicators": self.indicators,
            "reference_date": self.reference_date,
            "computed_at": self.computed_at,
        }


def _iso(value: datetime) -> str:
    """Render a naive UTC datetime as an ISO 8601 string.

    Args:
        value: Naive UTC datetime.

    Returns:
        ISO 8601 string with ``+00:00``.
    """
    return value.replace(tzinfo=UTC).isoformat()


def _prepare_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Normalise the review frame once (numeric columns, naive UTC times, net).

    Args:
        reviews: ``reviews`` joined with ``review_features`` (see
            :func:`munasib.review_intel.evidence.store.load_reviews`).

    Returns:
        A copy with float numeric columns, ``review_time`` as naive UTC and a
        ``net`` column (``prob_positive - prob_negative``).
    """
    if reviews.empty:
        return reviews.copy()
    frame = reviews.copy()
    for column in _NUMERIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = float("nan")
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    if "verified" not in frame.columns:
        frame["verified"] = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    frame["review_time"] = pd.to_datetime(frame["review_time"], utc=True).dt.tz_localize(None)
    frame["net"] = frame["prob_positive"] - frame["prob_negative"]
    return frame


def _aspect_series(aspect_frame: pd.DataFrame | None) -> tuple[pd.Series, pd.Series]:
    """Split one product's aspect rows into negative counts and classifications.

    Args:
        aspect_frame: Rows of ``product_aspects`` for one product, or None.

    Returns:
        ``(negative_counts, classifications)`` indexed by aspect name; both are
        empty when the product has no stored aspect rows.
    """
    empty_index = pd.Index([], name="aspect", dtype="object")
    if aspect_frame is None or aspect_frame.empty:
        return (
            pd.Series(dtype="float64", index=empty_index),
            pd.Series(dtype="object", index=empty_index),
        )
    counts = pd.to_numeric(aspect_frame["negative_count"], errors="coerce").fillna(0.0)
    counts.index = pd.Index(aspect_frame["aspect"].astype(str), name="aspect")
    classifications = pd.Series(
        aspect_frame["classification"].astype(str).to_numpy(), index=counts.index
    )
    return counts.astype("float64"), classifications


def _product_indicators(
    frame: pd.DataFrame | None,
    aspect_frame: pd.DataFrame | None,
    reference_date: datetime,
    config: EvidenceConfig,
) -> tuple[dict[str, Indicator], StrengthResult]:
    """Compute every indicator of one product plus the composite.

    Args:
        frame: The product's prepared review rows, or None when it has none.
        aspect_frame: The product's stored aspect rows, or None.
        reference_date: Dataset reference date (naive UTC).
        config: Evidence configuration.

    Returns:
        ``(indicators by name, composite strength)``.
    """
    empty_float = pd.Series(dtype="float64")
    empty_bool = pd.Series(dtype="bool")
    empty_time = pd.Series(dtype="datetime64[ns]")
    if frame is None or frame.empty:
        n_reviews = 0
        ratings = empty_float
        votes = empty_float
        verified = pd.Series(dtype="object")
        times = empty_time
        predicted = empty_float
        net = empty_float
        prob_positive = empty_float
        prob_negative = empty_float
        has_prediction = empty_bool
    else:
        n_reviews = len(frame)
        ratings = frame["rating"]
        votes = frame["helpful_votes"]
        verified = frame["verified"]
        times = frame["review_time"]
        predicted = frame["predicted_label"]
        net = frame["net"]
        prob_positive = frame["prob_positive"]
        prob_negative = frame["prob_negative"]
        has_prediction = predicted.notna() & net.notna()

    indicator_config = config.indicators
    indicators: dict[str, Indicator] = {}
    indicators["review_volume"] = volume_indicator(n_reviews, indicator_config)
    indicators["rating_agreement"] = rating_agreement_indicator(ratings, indicator_config)
    indicators["sentiment_shares"] = sentiment_shares_indicator(predicted, net, ratings)
    coverage, weighted = helpfulness_indicators(votes, net, has_prediction)
    indicators["helpfulness_coverage"] = coverage
    indicators["helpfulness_weighted_sentiment"] = weighted
    indicators["verified_purchase_ratio"] = verified_purchase_indicator(verified)
    recency, staleness, shift, trend = recency_indicators(
        times, predicted, net, has_prediction, reference_date, indicator_config
    )
    indicators["recency_coverage"] = recency
    indicators["staleness"] = staleness
    indicators["recent_negative_shift"] = shift
    indicators["sentiment_trend"] = trend

    negative_counts, classifications = _aspect_series(aspect_frame)
    negative_mask = has_prediction & (predicted == SENTIMENT_LABEL_NEGATIVE)
    n_negative = int(negative_mask.sum())
    concentration, top_share = concentration_indicators(
        negative_counts, classifications, n_negative, indicator_config
    )
    indicators["aspect_concentration"] = concentration
    indicators["top_pain_point_share"] = top_share
    indicators["temporal_burst"] = burst_indicator(
        times[negative_mask], n_negative, indicator_config
    )
    indicators["rating_text_agreement"] = rating_text_agreement_indicator(
        ratings, predicted, prob_positive, prob_negative, indicator_config
    )
    return indicators, composite_strength(indicators, config.strength)


def _indicator_payload(
    split: str,
    indicators: Mapping[str, Indicator],
    strength: StrengthResult,
    n_reviews: int,
    config: EvidenceConfig,
) -> dict[str, Any]:
    """Render the stored ``indicators`` JSON of one product.

    Args:
        split: Product split (stored for self-describing reports).
        indicators: Every indicator of the product, keyed by name.
        strength: Composite result.
        n_reviews: Review count the indicators were computed over.
        config: Evidence configuration (low-evidence guard).

    Returns:
        The JSON payload: ``indicators``, ``flags``, the four summary booleans,
        ``low_evidence`` and the ``strength`` breakdown.
    """
    flags = {
        name: indicator.missing_reason
        for name, indicator in indicators.items()
        if indicator.missing_reason is not None
    }
    recency = indicators["recency_coverage"]
    recent_reviews = recency.extras.get("recent_reviews")
    recent_count = int(recent_reviews) if isinstance(recent_reviews, int) else 0
    no_recent = recency.missing_reason is not None or recent_count == 0
    low_evidence = (
        strength.value is None
        or strength.value < config.strength.low_evidence_threshold
        or n_reviews < config.strength.min_reviews_for_confidence
    )
    return {
        "split": split,
        "indicators": {name: indicator.to_payload() for name, indicator in indicators.items()},
        "flags": flags,
        "polarized": bool(indicators["rating_agreement"].extras.get("polarized", False)),
        "no_recent_reviews": no_recent,
        "temporal_burst": bool(indicators["temporal_burst"].extras.get("flag", False)),
        "low_evidence": low_evidence,
        "strength": {
            "value": None if strength.value is None else round(strength.value, DIGITS),
            "components_used": list(strength.components_used),
            "weights": {
                name: round(weight, DIGITS) for name, weight in strength.weights.items()
            },
            "missing_components": dict(strength.missing),
        },
    }


def compute_evidence(
    products: pd.DataFrame,
    reviews: pd.DataFrame,
    aspects: pd.DataFrame,
    *,
    reference_date: datetime,
    computed_at: datetime,
    config: EvidenceConfig,
) -> list[EvidenceRow]:
    """Compute one evidence row per product, in ascending ASIN order.

    Args:
        products: Products to score (``asin``, ``split``).
        reviews: Reviews joined with stored predictions (``asin``, ``rating``,
            ``helpful_votes``, ``verified``, ``review_time``,
            ``predicted_label``, ``prob_negative``/``prob_neutral``/
            ``prob_positive``).
        aspects: Stored aspect rows (``asin``, ``aspect``, ``classification``,
            ``negative_count``).
        reference_date: Dataset reference date (naive UTC, from
            ``review_date_range.max``); every recency window is anchored here.
        computed_at: Dataset build timestamp (naive UTC, from ``built_at``).
        config: Evidence configuration.

    Returns:
        One row per product, deterministic for identical inputs.

    Raises:
        ValueError: If a timestamp is timezone-aware (storage contract), the
            product frame is empty, or the product frame lacks ``asin``/``split``.
    """
    if reference_date.tzinfo is not None or computed_at.tzinfo is not None:
        raise ValueError(
            "reference_date and computed_at must be naive UTC (see munasib.db.models); "
            f"got reference tzinfo={reference_date.tzinfo!r} and "
            f"computed tzinfo={computed_at.tzinfo!r}"
        )
    missing_columns = [column for column in ("asin", "split") if column not in products.columns]
    if missing_columns:
        raise ValueError(f"products frame lacks column(s) {missing_columns}")
    if products.empty:
        raise ValueError("cannot compute evidence without products")

    prepared = _prepare_reviews(reviews)
    reviews_by_asin = (
        {str(asin): frame for asin, frame in prepared.groupby("asin", sort=False)}
        if not prepared.empty
        else {}
    )
    aspects_by_asin = (
        {str(asin): frame for asin, frame in aspects.groupby("asin", sort=False)}
        if not aspects.empty
        else {}
    )

    rows: list[EvidenceRow] = []
    for record in products.sort_values("asin").to_dict("records"):
        asin = str(record["asin"])
        split = str(record["split"])
        frame = reviews_by_asin.get(asin)
        indicators, strength = _product_indicators(
            frame, aspects_by_asin.get(asin), reference_date, config
        )
        n_reviews = 0 if frame is None else len(frame)
        rows.append(
            EvidenceRow(
                asin=asin,
                split=split,
                n_reviews=n_reviews,
                evidence_strength=(
                    None if strength.value is None else round(strength.value, DIGITS)
                ),
                indicators=_indicator_payload(split, indicators, strength, n_reviews, config),
                reference_date=reference_date,
                computed_at=computed_at,
            )
        )
    return rows


# ------------------------------------------------------------------------------
# Summary (printed and written to reports/)
# ------------------------------------------------------------------------------


def _normalized(row: EvidenceRow, indicator_name: str) -> float | None:
    """Read one stored normalized value from a row.

    Args:
        row: Evidence row.
        indicator_name: Indicator key.

    Returns:
        The normalized value, or None when the indicator is missing.
    """
    entry = row.indicators["indicators"][indicator_name]
    value = entry.get("normalized") if isinstance(entry, dict) else None
    return None if value is None else float(value)


def _missing_reason(row: EvidenceRow, indicator_name: str) -> str | None:
    """Read one stored missing reason from a row.

    Args:
        row: Evidence row.
        indicator_name: Indicator key.

    Returns:
        The reason, or None when the indicator is available.
    """
    entry = row.indicators["indicators"][indicator_name]
    reason = entry.get("missing_reason") if isinstance(entry, dict) else None
    return None if reason is None else str(reason)


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    """Describe one numeric distribution.

    Args:
        values: Non-missing values.

    Returns:
        ``{"count", "min", "median", "max"}`` with floats rounded to 6 decimals.
    """
    return {
        "count": len(values),
        "min": round(min(values), DIGITS),
        "median": round(float(median(values)), DIGITS),
        "max": round(max(values), DIGITS),
    }


def _spearman(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Spearman rank correlation without a scipy dependency.

    Args:
        pairs: ``(composite, signal)`` pairs with both values present.

    Returns:
        The rank correlation, or None when fewer than two pairs exist or either
        side is constant (an undefined correlation is not reported as 0).
    """
    if len(pairs) < 2:
        return None
    frame = pd.DataFrame(pairs, columns=["strength", "signal"])
    if frame["strength"].nunique() < 2 or frame["signal"].nunique() < 2:
        return None
    rho = float(frame["strength"].rank().corr(frame["signal"].rank()))
    return None if math.isnan(rho) else round(rho, DIGITS)


@dataclass(frozen=True)
class EvidenceSummary:
    """Computed figures of one run (printed and written to ``reports/``)."""

    products: int
    reference_date: datetime
    computed_at: datetime
    distributions: dict[str, dict[str, Any]]
    flags: dict[str, int]
    per_split: dict[str, dict[str, Any]]
    missing: dict[str, dict[str, Any]]
    correlations: dict[str, dict[str, Any]]

    def lines(self) -> list[str]:
        """Render the printed summary (computed values only).

        Returns:
            One line per reported figure, in a fixed order.
        """
        lines = [
            f"{'products written':<34}: {self.products:,}",
            f"{'reference date':<34}: {_iso(self.reference_date)}",
            f"{'computed at':<34}: {_iso(self.computed_at)}",
        ]
        for name, stats in self.distributions.items():
            lines.append(
                f"{name:<34}: min={float(stats['min']):.6f} "
                f"median={float(stats['median']):.6f} max={float(stats['max']):.6f} "
                f"(n={int(stats['count']):,})"
            )
        if self.products:
            for name, count in self.flags.items():
                lines.append(f"{name:<34}: {count:,} ({count / self.products:.2%})")
        for split in SPLIT_ORDER:
            split_stats = self.per_split.get(split)
            if split_stats is None:
                continue
            value = split_stats["strength_median"]
            strength_text = "n/a" if value is None else f"{float(value):.6f}"
            lines.append(
                f"{'split ' + split:<34}: n={int(split_stats['products']):,} "
                f"strength median={strength_text}"
            )
        for name, entry in self.missing.items():
            reasons = entry["reasons"]
            reason_text = ", ".join(
                f"{reason}={count:,}" for reason, count in sorted(reasons.items())
            )
            lines.append(f"{'missing ' + name:<34}: {int(entry['count']):,} ({reason_text})")
        for name, entry in self.correlations.items():
            value = entry["spearman"]
            rho_text = "n/a" if value is None else f"{float(value):+.4f}"
            lines.append(f"{'spearman ' + name:<34}: {rho_text} (n={int(entry['n']):,})")
        return lines

    def to_payload(self) -> dict[str, Any]:
        """Render the JSON report payload.

        Returns:
            Mapping with the run metadata, distributions, flag shares, per-split
            figures, missing counts and correlations. No wall-clock value is
            included.
        """
        flags = {
            name: {
                "count": count,
                "share": round(count / self.products, DIGITS) if self.products else 0.0,
            }
            for name, count in self.flags.items()
        }
        return {
            "products": self.products,
            "reference_date": _iso(self.reference_date),
            "computed_at": _iso(self.computed_at),
            "distributions": self.distributions,
            "flags": flags,
            "per_split": self.per_split,
            "missing": self.missing,
            "correlations": self.correlations,
        }


def summarize(rows: Sequence[EvidenceRow]) -> EvidenceSummary:
    """Aggregate the printed/stored figures for one run.

    Args:
        rows: Rows written this run.

    Returns:
        Distributions of the composite and every normalized indicator, the four
        summary flag counts, per-split counts and strength medians, the number
        of products behind each missing-data flag (with reasons), and the
        Spearman correlation of the composite with every component and with the
        review count.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    materialised = list(rows)
    if not materialised:
        raise ValueError("cannot summarize an empty evidence run")
    indicator_names = list(materialised[0].indicators["indicators"])

    distributions: dict[str, dict[str, float | int]] = {}
    strengths = [
        row.evidence_strength
        for row in materialised
        if row.evidence_strength is not None
    ]
    if strengths:
        distributions["evidence_strength"] = _distribution(strengths)
    for name in indicator_names:
        values = [
            value
            for row in materialised
            if (value := _normalized(row, name)) is not None
        ]
        if values:
            distributions[name] = _distribution(values)

    flags = {
        name: sum(1 for row in materialised if bool(row.indicators[name]))
        for name in SUMMARY_FLAGS
    }

    per_split: dict[str, dict[str, float | int | None]] = {}
    for split in SPLIT_ORDER:
        subset = [row for row in materialised if row.split == split]
        if not subset:
            continue
        split_strengths = [
            row.evidence_strength for row in subset if row.evidence_strength is not None
        ]
        per_split[split] = {
            "products": len(subset),
            "strength_median": (
                round(float(median(split_strengths)), DIGITS) if split_strengths else None
            ),
        }

    missing: dict[str, dict[str, Any]] = {}
    for name in indicator_names:
        reasons: Counter[str] = Counter()
        for row in materialised:
            reason = _missing_reason(row, name)
            if reason is not None:
                reasons[reason] += 1
        if reasons:
            missing[name] = {
                "count": sum(reasons.values()),
                "reasons": dict(sorted(reasons.items())),
            }

    correlations: dict[str, dict[str, float | int | None]] = {}
    targets: list[tuple[str, str | None]] = [
        (component, COMPONENT_INDICATOR[component]) for component in STRENGTH_COMPONENT_KEYS
    ]
    targets.append(("n_reviews", None))
    for label, indicator_name in targets:
        pairs: list[tuple[float, float]] = []
        for row in materialised:
            if row.evidence_strength is None:
                continue
            signal = (
                float(row.n_reviews)
                if indicator_name is None
                else _normalized(row, indicator_name)
            )
            if signal is None:
                continue
            pairs.append((row.evidence_strength, signal))
        correlations[label] = {"spearman": _spearman(pairs), "n": len(pairs)}

    return EvidenceSummary(
        products=len(materialised),
        reference_date=materialised[0].reference_date,
        computed_at=materialised[0].computed_at,
        distributions=distributions,
        flags=flags,
        per_split=per_split,
        missing=missing,
        correlations=correlations,
    )


def render_summary_markdown(summary: EvidenceSummary) -> str:
    """Render the Markdown report from the same figures as the JSON report.

    Args:
        summary: Run summary.

    Returns:
        GitHub-flavoured Markdown.
    """
    lines = [
        "# Evidence summary",
        "",
        "Computed from the stored sentiment predictions and aspect rows by "
        "`python scripts/compute_evidence.py`; every number below is produced by the "
        "code, none is illustrative.",
        "",
        f"- **Products:** {summary.products:,}",
        f"- **Reference date (recency anchor):** {_iso(summary.reference_date)} - the "
        "latest review timestamp in `data/processed/meta.json`, never today's date.",
        f"- **Computed at:** {_iso(summary.computed_at)} - the dataset build timestamp.",
        "- **Weak supervision:** sentiment labels come from the stored predictor, whose "
        "training targets are star-derived weak labels; the model was trained on the "
        "train-split products, so train predictions are in-sample (see `docs/evidence.md`).",
        "",
        "## Distributions",
        "",
        "| Indicator | n | min | median | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in summary.distributions.items():
        lines.append(
            f"| `{name}` | {int(stats['count']):,} | {float(stats['min']):.6f} | "
            f"{float(stats['median']):.6f} | {float(stats['max']):.6f} |"
        )
    lines += ["", "## Flags", "", "| Flag | Products | Share |", "|---|---:|---:|"]
    for name, count in summary.flags.items():
        share = count / summary.products if summary.products else 0.0
        lines.append(f"| `{name}` | {count:,} | {share:.2%} |")
    lines += [
        "",
        "## Per split",
        "",
        "| Split | Products | evidence_strength median |",
        "|---|---:|---:|",
    ]
    for split, stats in summary.per_split.items():
        value = stats["strength_median"]
        text = "n/a" if value is None else f"{float(value):.6f}"
        lines.append(f"| {split} | {int(stats['products']):,} | {text} |")
    lines += [
        "",
        "## Missing values",
        "",
        "| Indicator | Products | Reasons |",
        "|---|---:|---|",
    ]
    for name, entry in summary.missing.items():
        reasons = entry["reasons"]
        text = ", ".join(f"`{reason}`={count:,}" for reason, count in sorted(reasons.items()))
        lines.append(f"| `{name}` | {int(entry['count']):,} | {text} |")
    lines += [
        "",
        "## Correlation with evidence_strength (Spearman)",
        "",
        "| Signal | Spearman | n |",
        "|---|---:|---:|",
    ]
    for name, entry in summary.correlations.items():
        value = entry["spearman"]
        text = "n/a" if value is None else f"{float(value):+.4f}"
        lines.append(f"| `{name}` | {text} | {int(entry['n']):,} |")
    lines.append("")
    return "\n".join(lines)


def write_summary_reports(summary: EvidenceSummary, reports_dir: Path) -> list[Path]:
    """Write ``evidence_summary.json`` and ``evidence_summary.md``.

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
    "JSON_FILENAME",
    "MARKDOWN_FILENAME",
    "SUMMARY_FLAGS",
    "EvidenceRow",
    "EvidenceSummary",
    "compute_evidence",
    "render_summary_markdown",
    "summarize",
    "write_summary_reports",
]

"""Database access and persistence for the per-product evidence stage.

The loaders read the serving cache only:

* ``reviews`` joined with ``review_features`` - the stored weak-label-trained
  predictions, never a fresh model run;
* ``product_aspects`` - the stored aspect counts and classifications;
* ``products`` - the ASIN/split list the rows are computed for.

``read_reference_date`` mirrors the API/serving rules: the recency anchor is
``review_date_range.max`` from ``data/processed/meta.json`` (the latest review
timestamp in the dataset), never the wall clock. ``write_evidence`` upserts
through :class:`~munasib.db.repositories.ProductEvidenceRepository`, so re-runs
are idempotent and partial runs refresh only their own products.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from json import JSONDecodeError
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from munasib.core.config import Settings
from munasib.core.exceptions import DataError
from munasib.db.models import ProductEvidence
from munasib.db.repositories import ProductEvidenceRepository
from munasib.product_intel.features import META_FILENAME
from munasib.review_intel.evidence.compute import EvidenceRow

#: Reviews joined with their stored predictions (one row per review).
_REVIEW_QUERY = """
SELECT r.asin AS asin,
       r.rating AS rating,
       r.helpful_votes AS helpful_votes,
       r.verified AS verified,
       r.review_time AS review_time,
       f.predicted_label AS predicted_label,
       f.prob_negative AS prob_negative,
       f.prob_neutral AS prob_neutral,
       f.prob_positive AS prob_positive
FROM reviews AS r
LEFT JOIN review_features AS f ON f.review_id = r.review_id
ORDER BY r.id
"""

_PRODUCT_QUERY = """
SELECT asin AS asin, split AS split
FROM products
ORDER BY asin
"""

_ASPECT_QUERY = """
SELECT asin AS asin, aspect AS aspect, classification AS classification,
       negative_count AS negative_count
FROM product_aspects
ORDER BY asin, aspect
"""


def read_reference_date(settings: Settings) -> datetime:
    """Read the dataset reference date from ``data/processed/meta.json``.

    The reference date is the latest review timestamp of the dataset
    (``review_date_range.max``). Every recency window of the evidence stage is
    anchored on it, never on the wall clock; the user-facing warning printed by
    ``scripts/build_dataset.py`` documents the same rule.

    Args:
        settings: Project settings (``paths.processed_data_dir``).

    Returns:
        Naive UTC datetime (the serving-cache storage contract).

    Raises:
        DataError: If the metadata file is missing, unreadable, or declares no
            valid ``review_date_range.max``.
    """
    path = Path(settings.paths.processed_data_dir) / META_FILENAME
    if not path.exists():
        raise DataError(
            f"dataset metadata not found at {path}; run scripts/build_dataset.py first"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (JSONDecodeError, OSError) as exc:
        raise DataError(f"could not read dataset metadata at {path}: {exc}") from exc
    date_range = raw.get("review_date_range") if isinstance(raw, dict) else None
    maximum = date_range.get("max") if isinstance(date_range, dict) else None
    if not isinstance(maximum, str) or not maximum.strip():
        raise DataError(
            f"dataset metadata at {path} declares no review_date_range.max; "
            "the recency anchor cannot be resolved"
        )
    try:
        parsed = datetime.fromisoformat(maximum.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataError(
            f"dataset metadata at {path} has an unparsable review_date_range.max "
            f"{maximum!r}: {exc}"
        ) from exc
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(UTC).replace(tzinfo=None)


def load_products(engine: Engine) -> pd.DataFrame:
    """Load the product list (``asin``, ``split``) in ascending ASIN order.

    Args:
        engine: SQLite serving-cache engine.

    Returns:
        Frame with one row per product.
    """
    return pd.read_sql(_PRODUCT_QUERY, engine)


def load_reviews(engine: Engine) -> pd.DataFrame:
    """Load every review with its stored sentiment prediction.

    The join is a LEFT JOIN because a review may have no ``review_features``
    row or a row with NULL probabilities (the predictor skipped it); both are
    handled as "no prediction", never imputed.

    Args:
        engine: SQLite serving-cache engine.

    Returns:
        Frame with ``asin``, ``rating``, ``helpful_votes``, ``verified``,
        ``review_time`` and the four stored prediction columns.
    """
    return pd.read_sql(_REVIEW_QUERY, engine)


def load_aspects(engine: Engine) -> pd.DataFrame:
    """Load every stored aspect row (the per-product counts and classes).

    Args:
        engine: SQLite serving-cache engine.

    Returns:
        Frame with ``asin``, ``aspect``, ``classification``, ``negative_count``.
    """
    return pd.read_sql(_ASPECT_QUERY, engine)


def write_evidence(session: Session, rows: Sequence[EvidenceRow]) -> int:
    """Upsert one ``product_evidence`` row per product.

    Args:
        session: Open session (the caller owns the transaction).
        rows: Rows from :func:`~munasib.review_intel.evidence.compute.compute_evidence`.

    Returns:
        Number of rows written.

    Raises:
        ValueError: If a row carries a timezone-aware timestamp (the storage
            contract is naive UTC).
    """
    for row in rows:
        if row.reference_date.tzinfo is not None or row.computed_at.tzinfo is not None:
            raise ValueError(
                "evidence timestamps must be naive UTC (see munasib.db.models); "
                f"got reference tzinfo={row.reference_date.tzinfo!r} and "
                f"computed tzinfo={row.computed_at.tzinfo!r}"
            )
    payloads = [row.to_payload() for row in rows]
    return ProductEvidenceRepository(session).upsert_evidence(payloads)


def read_evidence(session: Session, asin: str) -> ProductEvidence | None:
    """Fetch one stored evidence row.

    Args:
        session: Open session.
        asin: Product ASIN.

    Returns:
        The row, or None when the product was never scored.
    """
    return ProductEvidenceRepository(session).get_evidence(asin)


def count_evidence(session: Session) -> int:
    """Count stored evidence rows.

    Args:
        session: Open session.

    Returns:
        Row count.
    """
    return ProductEvidenceRepository(session).count_rows()


__all__ = [
    "count_evidence",
    "load_aspects",
    "load_products",
    "load_reviews",
    "read_evidence",
    "read_reference_date",
    "write_evidence",
]

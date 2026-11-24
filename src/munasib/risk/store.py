"""Database access and persistence for the per-product risk-indicator stage.

The loaders read the serving cache only:

* ``products`` - ASIN, split, title, mean rating and review count;
* ``reviews`` joined with ``review_features`` - the stored sentiment
  predictions, never a fresh model run;
* ``product_features`` - the stored metadata indicators (price percentile and
  peer-group provenance);
* ``product_evidence`` - the stored evidence composite and indicator payload.

The recency anchor and the build timestamp are reused from the stages that
already define them (``review_date_range.max`` and ``built_at`` in
``data/processed/meta.json``), so the risk row is reproducible from the cache
alone and never reads the wall clock.

``write_risk_indicators`` upserts through
:class:`~munasib.db.repositories.ProductRiskIndicatorRepository`, so re-runs are
idempotent and partial runs refresh only their own products.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from json import JSONDecodeError
from typing import Any

import pandas as pd
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from munasib.db.models import ProductRiskIndicator
from munasib.db.repositories import ProductRiskIndicatorRepository
from munasib.product_intel.features import read_build_timestamp
from munasib.review_intel.evidence.store import read_reference_date
from munasib.risk.compute import RiskRow

#: Products (display columns included so best/worst lists need no extra join).
_PRODUCT_QUERY = """
SELECT asin AS asin,
       split AS split,
       title AS title,
       mean_rating AS mean_rating,
       n_reviews AS n_reviews
FROM products
ORDER BY asin
"""

#: Reviews joined with their stored prediction (one row per review).
_PREDICTION_QUERY = """
SELECT r.asin AS asin,
       f.predicted_label AS predicted_label,
       f.prob_negative AS prob_negative,
       f.prob_positive AS prob_positive
FROM reviews AS r
LEFT JOIN review_features AS f ON f.review_id = r.review_id
ORDER BY r.id
"""

#: Stored evidence rows (composite + full indicator payload).
_EVIDENCE_QUERY = """
SELECT asin AS asin,
       evidence_strength AS evidence_strength,
       indicators AS indicators
FROM product_evidence
ORDER BY asin
"""

#: Stored product-feature indicator payloads (price fields live here).
_FEATURE_QUERY = """
SELECT asin AS asin,
       indicators AS indicators
FROM product_features
ORDER BY asin
"""


def _decode_json_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Decode a SQLite JSON column that pandas returned as text.

    SQLAlchemy's JSON type serialises to TEXT in SQLite; ``pd.read_sql`` reads
    the raw text instead of applying the ORM's JSON decoding. Every payload is
    decoded here so downstream code always sees a mapping (or ``None``), never
    a string.

    Args:
        frame: Frame read from SQL.
        column: JSON column name.

    Returns:
        The same frame with the column decoded in place.
    """

    def decode(value: Any) -> Any:
        if isinstance(value, (str, bytes)):
            try:
                return json.loads(value)
            except (JSONDecodeError, UnicodeDecodeError):
                return None
        return value if isinstance(value, dict) or value is None else value

    frame[column] = frame[column].map(decode)
    return frame


def load_products(engine: Engine) -> pd.DataFrame:
    """Load every catalog product with the columns the stage needs.

    Args:
        engine: SQLite serving-cache engine.

    Returns:
        Frame with ``asin``, ``split``, ``title``, ``mean_rating`` and
        ``n_reviews`` (one row per product).
    """
    return pd.read_sql(_PRODUCT_QUERY, engine)


def load_predictions(engine: Engine) -> pd.DataFrame:
    """Load every review with its stored sentiment prediction.

    The join is a LEFT JOIN because a review may have no ``review_features``
    row or a row with NULL probabilities; both are treated as "no prediction",
    never imputed.

    Args:
        engine: SQLite serving-cache engine.

    Returns:
        Frame with ``asin``, ``predicted_label``, ``prob_negative`` and
        ``prob_positive``.
    """
    return pd.read_sql(_PREDICTION_QUERY, engine)


def load_evidence(engine: Engine) -> pd.DataFrame:
    """Load every stored evidence row.

    Args:
        engine: SQLite serving-cache engine.

    Returns:
        Frame with ``asin``, ``evidence_strength`` and the stored ``indicators``
        payload.
    """
    return _decode_json_column(pd.read_sql(_EVIDENCE_QUERY, engine), "indicators")


def load_features(engine: Engine) -> pd.DataFrame:
    """Load every stored product-feature row.

    The ``indicators`` payload is decoded by
    :class:`~munasib.db.models.ProductFeature`'s JSON column, so the price
    percentile and its provenance arrive as a mapping.

    Args:
        engine: SQLite serving-cache engine.

    Returns:
        Frame with ``asin`` and the stored ``indicators`` payload.
    """
    return _decode_json_column(pd.read_sql(_FEATURE_QUERY, engine), "indicators")


def write_risk_indicators(session: Session, rows: Sequence[RiskRow]) -> int:
    """Upsert one ``product_risk_indicators`` row per product.

    Args:
        session: Open session (the caller owns the transaction).
        rows: Rows from
            :func:`~munasib.risk.compute.compute_risk_indicators`.

    Returns:
        Number of rows written.

    Raises:
        ValueError: If a row carries a timezone-aware timestamp (the storage
            contract is naive UTC).
    """
    for row in rows:
        if row.reference_date.tzinfo is not None or row.computed_at.tzinfo is not None:
            raise ValueError(
                "risk-indicator timestamps must be naive UTC (see munasib.db.models); "
                f"got reference tzinfo={row.reference_date.tzinfo!r} and "
                f"computed tzinfo={row.computed_at.tzinfo!r}"
            )
    payloads = [row.to_payload() for row in rows]
    return ProductRiskIndicatorRepository(session).upsert_risk_indicators(payloads)


def read_risk_indicators(session: Session, asin: str) -> ProductRiskIndicator | None:
    """Fetch one stored risk-indicator row.

    Args:
        session: Open session.
        asin: Product ASIN.

    Returns:
        The row, or None when the product was never scored.
    """
    return ProductRiskIndicatorRepository(session).get_risk_indicators(asin)


def count_risk_indicators(session: Session) -> int:
    """Count stored risk-indicator rows.

    Args:
        session: Open session.

    Returns:
        Row count.
    """
    return ProductRiskIndicatorRepository(session).count_rows()


__all__ = [
    "count_risk_indicators",
    "load_evidence",
    "load_features",
    "load_predictions",
    "load_products",
    "read_build_timestamp",
    "read_reference_date",
    "read_risk_indicators",
    "write_risk_indicators",
]

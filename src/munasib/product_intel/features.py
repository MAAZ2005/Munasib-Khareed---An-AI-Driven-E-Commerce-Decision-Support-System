"""Persist product profiles into the serving cache's ``product_features`` table.

The table (see :class:`munasib.db.models.ProductFeature`) holds one JSON blob
of indicators per ASIN plus a ``computed_at`` timestamp. Two properties make
re-runs trustworthy:

* **Deterministic timestamp.** ``computed_at`` is the dataset build timestamp
  from ``data/processed/meta.json`` (``built_at``), never the wall clock, so
  running the CLI twice produces byte-identical rows. The timestamp is stored
  naive-UTC, matching the database contract in :mod:`munasib.db.models`.
* **Idempotent writes.** Profiles are upserted on the ``asin`` primary key, so
  re-running overwrites the row instead of duplicating it.

A stored blob round-trips through :meth:`~munasib.product_intel.profile.ProductProfile.from_indicators`,
which refuses a payload that is missing a key rather than defaulting it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from munasib.core.config import Settings
from munasib.core.exceptions import DataError
from munasib.db.models import ProductFeature
from munasib.product_intel.profile import ProductProfile

#: File holding the dataset build timestamp (written by scripts/build_dataset.py).
META_FILENAME = "meta.json"


def read_build_timestamp(settings: Settings) -> datetime:
    """Read the dataset build timestamp that stamps every profile write.

    Args:
        settings: Project settings (``paths.processed_data_dir``).

    Returns:
        The ``built_at`` timestamp as a **naive UTC** datetime (the storage
        contract of every datetime column in the serving cache).

    Raises:
        DataError: If ``meta.json`` is missing, unparsable, or declares no
            usable ``built_at`` - the profile writer never falls back to the
            wall clock, because that would make identical inputs produce
            different rows.
    """
    path = Path(settings.paths.processed_data_dir) / META_FILENAME
    if not path.exists():
        raise DataError(f"dataset metadata not found at {path}; run scripts/build_dataset.py")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"dataset metadata at {path} is not valid JSON: {exc}") from exc

    built_at = raw.get("built_at")
    if not isinstance(built_at, str) or not built_at.strip():
        raise DataError(f"dataset metadata at {path} declares no built_at timestamp")
    try:
        parsed = datetime.fromisoformat(built_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataError(f"built_at {built_at!r} in {path} is not an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        # meta.json timestamps are UTC by construction; be explicit anyway.
        return parsed
    return parsed.astimezone(UTC).replace(tzinfo=None)


def write_product_features(
    session: Session,
    profiles: list[ProductProfile],
    computed_at: datetime,
) -> int:
    """Upsert one ``product_features`` row per profile.

    Args:
        session: Open session (the caller owns the transaction, e.g.
            ``session_scope``).
        profiles: Profiles to store.
        computed_at: Dataset build timestamp (naive UTC, from
            :func:`read_build_timestamp`); stored verbatim on every row.

    Returns:
        Number of rows written (inserted or refreshed).

    Raises:
        ValueError: If ``computed_at`` is timezone-aware, so a caller cannot
            accidentally store a mixed-offset timestamp.
        Exception: Any integrity failure (e.g. an ASIN missing from
            ``products``) propagates and the caller's transaction rolls back.
    """
    if computed_at.tzinfo is not None:
        raise ValueError(
            "computed_at must be naive UTC (see munasib.db.models); "
            f"got tzinfo={computed_at.tzinfo!r}"
        )
    for profile in profiles:
        session.merge(
            ProductFeature(
                asin=profile.asin,
                indicators=profile.to_indicators(),
                computed_at=computed_at,
            )
        )
    session.flush()
    return len(profiles)


def read_indicators(session: Session, asin: str) -> dict | None:
    """Read the raw stored indicator blob for one product.

    Args:
        session: Open session.
        asin: Product ASIN.

    Returns:
        The stored ``indicators`` mapping, or None when the product was never
        profiled (missing data stays missing - nothing is fabricated).
    """
    row = session.get(ProductFeature, asin)
    return None if row is None else row.indicators


def read_profile(session: Session, asin: str) -> ProductProfile | None:
    """Read one stored profile back as a typed :class:`ProductProfile`.

    Args:
        session: Open session.
        asin: Product ASIN.

    Returns:
        The reconstructed profile, or None when no row exists.

    Raises:
        ValueError: If the stored blob is missing an indicator key (a corrupt
            row must fail loudly rather than load as a partial profile).
    """
    payload = read_indicators(session, asin)
    if payload is None:
        return None
    return ProductProfile.from_indicators(asin, payload)


def read_profiles(session: Session, asins: list[str] | None = None) -> list[ProductProfile]:
    """Read stored profiles for several products, ordered by ASIN.

    Args:
        session: Open session.
        asins: ASINs to load, or None for every stored profile.

    Returns:
        One reconstructed profile per existing row, ascending by ASIN.
    """
    statement = select(ProductFeature).order_by(ProductFeature.asin)
    if asins is not None:
        statement = statement.where(ProductFeature.asin.in_(asins))
    return [
        ProductProfile.from_indicators(row.asin, row.indicators)
        for row in session.scalars(statement)
        if row.indicators is not None
    ]


def list_profiled_asins(session: Session) -> list[str]:
    """List every ASIN that already has stored product features.

    Args:
        session: Open session.

    Returns:
        Stored ASINs, ascending (stable, so diffing two runs is meaningful).
    """
    return list(session.scalars(select(ProductFeature.asin).order_by(ProductFeature.asin)))


__all__ = [
    "list_profiled_asins",
    "read_build_timestamp",
    "read_indicators",
    "read_profile",
    "read_profiles",
    "write_product_features",
]

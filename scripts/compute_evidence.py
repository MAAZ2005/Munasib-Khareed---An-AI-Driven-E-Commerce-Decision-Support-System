"""Compute and store per-product evidence indicators for the candidate set.

Reads the SQLite serving cache only - stored sentiment predictions from
``review_features`` joined onto ``reviews``, plus the stored aspect rows in
``product_aspects`` - and upserts one ``product_evidence`` row per ASIN. No
model runs, no network. Every recency window is anchored on the dataset
reference date from ``data/processed/meta.json`` (``review_date_range.max``),
never on the wall clock; ``computed_at`` is the dataset build timestamp, so two
runs over the same cache write identical rows. See ``docs/evidence.md`` for
every formula, window and missing-data rule.

The printed summary contains only computed values: reference date and computed
at, min/median/max of the composite and of every non-missing normalized
indicator, the four flag shares, per-split counts with the strength median, the
number of products behind each missing-data flag (with reasons) and the
Spearman correlation of the composite with each component and with the review
count. The same figures are written to ``reports/evidence_summary.json`` and
``reports/evidence_summary.md``; elapsed time is printed but never written to
the reports because it is not a data value.

Usage::

    python scripts/compute_evidence.py
    python scripts/compute_evidence.py --limit 100
    python scripts/compute_evidence.py --asins B000050JN4 B00004T8A4
    python scripts/compute_evidence.py --config configs/config.yaml

Exit codes:
  0 - evidence rows written (a filter may legitimately match nothing new).
  1 - the run could not start (bad config, unreadable cache, missing
      ``meta.json``, or an ``--asins`` entry that is not in the cache).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from munasib.core.config import Settings, load_settings
from munasib.core.exceptions import ConfigError, DataError, ProductNotFoundError
from munasib.db.models import Product
from munasib.db.repositories import ProductRepository
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)
from munasib.product_intel.features import read_build_timestamp
from munasib.review_intel.evidence.compute import (
    EvidenceSummary,
    compute_evidence,
    summarize,
    write_summary_reports,
)
from munasib.review_intel.evidence.store import (
    load_aspects,
    load_products,
    load_reviews,
    read_reference_date,
    write_evidence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("munasib.compute_evidence")


@dataclass(frozen=True)
class EvidenceRun:
    """Outcome of one evidence run (computed values only)."""

    #: Evidence rows written this run.
    written: int
    #: The printed/reported summary.
    summary: EvidenceSummary
    #: Files written under ``settings.paths.reports_dir``.
    report_paths: list[Path]
    #: Wall-clock duration of the run (printed, never reported).
    elapsed_seconds: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute evidence indicators and store them in product_evidence."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score at most N products (ASIN order); default: every product.",
    )
    parser.add_argument(
        "--asins",
        nargs="+",
        default=None,
        metavar="ASIN",
        help="Score exactly these ASINs (overrides --limit).",
    )
    return parser.parse_args(argv)


def select_targets(
    session: Session, limit: int | None, asins: Sequence[str] | None
) -> list[Product]:
    """Choose which products to score, in stable ASIN order.

    Args:
        session: Open session over the serving cache.
        limit: Maximum number of products (None = no cap).
        asins: Explicit ASINs to score, or None to take them in ASIN order.

    Returns:
        The selected product rows, ascending by ASIN.

    Raises:
        ProductNotFoundError: If a requested ASIN is not in the cache, so a
            typo cannot silently shrink the run.
        ValueError: If ``limit`` is negative.
    """
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")
    repository = ProductRepository(session)
    if asins:
        wanted = sorted(set(asins))
        rows = repository.get_products(wanted)
        found = {row.asin for row in rows}
        unknown = [asin for asin in wanted if asin not in found]
        if unknown:
            raise ProductNotFoundError(
                ", ".join(unknown),
                f"ASIN(s) not in the serving cache: {unknown}",
            )
        return rows
    rows = repository.list_all_products()
    return rows if limit is None else rows[:limit]


def run(settings: Settings, limit: int | None, asins: Sequence[str] | None) -> EvidenceRun:
    """Score the selected products, store the rows and report the summary.

    Args:
        settings: Loaded project settings.
        limit: Maximum number of products (None = every product).
        asins: Explicit ASINs, or None to take them in ASIN order.

    Returns:
        The run outcome (rows written, summary, report paths, elapsed time).

    Raises:
        DataError: If the cache is empty or ``meta.json`` cannot supply the
            reference date.
        ProductNotFoundError: If an ``--asins`` entry is unknown.
        ValueError: If ``limit`` is negative or a frame is malformed.
    """
    started = time.perf_counter()
    engine = create_engine_from_settings(settings)
    init_db(engine)
    factory = create_session_factory(engine)
    report_paths: list[Path] = []
    try:
        computed_at = read_build_timestamp(settings)
        reference_date = read_reference_date(settings)
        with session_scope(factory) as session:
            if not ProductRepository(session).list_all_products():
                raise DataError(
                    "the serving cache holds no products; run scripts/build_dataset.py "
                    "and scripts/ingest_db.py first"
                )
            targets = select_targets(session, limit, asins)
        target_asins = {row.asin for row in targets}
        logger.info("targets=%d", len(targets))
        logger.info("reference_date=%s computed_at=%s", reference_date, computed_at)

        products = load_products(engine)
        reviews = load_reviews(engine)
        aspects = load_aspects(engine)
        products = products[products["asin"].isin(target_asins)]
        reviews = reviews[reviews["asin"].isin(target_asins)]
        aspects = aspects[aspects["asin"].isin(target_asins)]
        logger.info(
            "data_loaded reviews=%d aspect_rows=%d", len(reviews), len(aspects)
        )

        rows = compute_evidence(
            products,
            reviews,
            aspects,
            reference_date=reference_date,
            computed_at=computed_at,
            config=settings.review_intel.evidence,
        )
        with session_scope(factory) as session:
            written = write_evidence(session, rows)
        logger.info("evidence_written rows=%d", written)

        summary = summarize(rows)
        report_paths = write_summary_reports(summary, Path(settings.paths.reports_dir))
        elapsed = time.perf_counter() - started
        logger.info("run_complete elapsed=%.2fs", elapsed)
        return EvidenceRun(
            written=written,
            summary=summary,
            report_paths=report_paths,
            elapsed_seconds=elapsed,
        )
    finally:
        engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    """Compute and store evidence indicators, then print the summary.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 = stored, 1 = could not start).
    """
    args = parse_args(argv)
    try:
        settings: Settings = load_settings(args.config)
    except ConfigError as exc:
        logger.error("config_error path=%s detail=%s", args.config, exc)
        return 1

    try:
        outcome = run(settings, args.limit, args.asins)
    except (ConfigError, DataError, ProductNotFoundError, ValueError) as exc:
        logger.error("run_failed detail=%s", exc)
        return 1

    print()
    for line in outcome.summary.lines():
        print(line)
    print(f"{'elapsed':<34}: {outcome.elapsed_seconds:.2f}s")
    print()
    for path in outcome.report_paths:
        print(f"{'report written':<34}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

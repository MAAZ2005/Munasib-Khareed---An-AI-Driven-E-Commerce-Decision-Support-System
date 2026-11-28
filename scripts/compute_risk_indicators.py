"""Compute and store per-product decision criteria and rule-based risk factors.

Reads the SQLite serving cache only - ``products``, the stored sentiment
predictions (``reviews`` joined with ``review_features``), the stored
``product_features`` payload and the stored ``product_evidence`` payload - and
upserts one ``product_risk_indicators`` row per ASIN. No model runs and no
network: the stage only reshapes and combines values that earlier stages
already computed and stored.

Recency is anchored on the dataset reference date
(``review_date_range.max`` from ``data/processed/meta.json``) and ``computed_at``
is the dataset build timestamp, so two runs over the same cache write identical
rows. Category priors are product-macro means over the **full train split**, even
when only a subset of ASINs is written, so a limited run never changes a prior.
See ``docs/risk_indicators.md`` for every formula, threshold and missing-data
rule.

The printed summary contains only computed values: the agreement-scale decision
(with the share of products saturated at 0), min/median/max of every criterion,
missing counts with their reasons, raw **and** normalized statistics for the
cost-direction indicators, the 5x5 Spearman matrix over all written products and
again over the train split, per-split medians, the severity counts of every risk
factor with its trigger share, correlation/factor warnings, and the five best and
worst products by ``product_quality``. The same figures are written to
``reports/risk_indicators_summary.json`` and ``reports/risk_indicators_summary.md``;
only the dataset's own timestamps (reference date and build time) appear in
them, never the wall clock and never the elapsed time.

Usage::

    python scripts/compute_risk_indicators.py
    python scripts/compute_risk_indicators.py --limit 100
    python scripts/compute_risk_indicators.py --asins B000050JN4 B00004T8A4
    python scripts/compute_risk_indicators.py --config configs/config.yaml

Exit codes:
  0 - risk-indicator rows written (a filter may legitimately match nothing new).
  1 - the run could not start (bad config, unreadable cache or metadata, a
      criteria-weights mismatch, or an ``--asins`` entry not in the cache).
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
from munasib.risk.compute import (
    RiskSummary,
    compute_risk_indicators,
    summarize,
    write_summary_reports,
)
from munasib.risk.indicators import (
    product_sentiment_totals,
    read_criteria_directions,
)
from munasib.risk.store import (
    load_evidence,
    load_features,
    load_predictions,
    load_products,
    read_build_timestamp,
    read_reference_date,
    write_risk_indicators,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("munasib.compute_risk_indicators")


@dataclass(frozen=True)
class RiskRun:
    """Outcome of one risk-indicator run (computed values only)."""

    #: Rows written this run.
    written: int
    #: The printed summary.
    summary: RiskSummary
    #: Files written under ``settings.paths.reports_dir``.
    report_paths: list[Path]
    #: Wall-clock duration of the run (printed, never stored).
    elapsed_seconds: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute decision criteria and risk factors and store them in "
            "product_risk_indicators."
        )
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
        ProductNotFoundError: If a requested ASIN is not in the cache, so a typo
            cannot silently shrink the run.
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


def run(settings: Settings, limit: int | None, asins: Sequence[str] | None) -> RiskRun:
    """Score the selected products, store the rows and build the summary.

    Args:
        settings: Loaded project settings.
        limit: Maximum number of products (None = every product).
        asins: Explicit ASINs, or None to take them in ASIN order.

    Returns:
        The run outcome (rows written, summary, report paths, elapsed time).

    Raises:
        DataError: If the cache is empty, metadata cannot supply the reference
            date, the train split cannot supply the priors, or the criteria
            weights file contradicts the computed directions.
        ProductNotFoundError: If an ``--asins`` entry is unknown.
        ValueError: If ``limit`` is negative or a frame is malformed.
    """
    started = time.perf_counter()
    engine = create_engine_from_settings(settings)
    init_db(engine)
    factory = create_session_factory(engine)
    try:
        computed_at = read_build_timestamp(settings)
        reference_date = read_reference_date(settings)
        directions = read_criteria_directions(
            Path(settings.decision.criteria_weights_file)
        )
        with session_scope(factory) as session:
            if not ProductRepository(session).list_all_products():
                raise DataError(
                    "the serving cache holds no products; run scripts/build_dataset.py "
                    "and scripts/ingest_db.py first"
                )
            targets = select_targets(session, limit, asins)
        target_asins = [row.asin for row in targets]
        logger.info("targets=%d", len(targets))
        logger.info("reference_date=%s computed_at=%s", reference_date, computed_at)

        products = load_products(engine)
        predictions = load_predictions(engine)
        evidence = load_evidence(engine)
        features = load_features(engine)
        sentiment = product_sentiment_totals(predictions)
        logger.info(
            "data_loaded products=%d predictions=%d evidence=%d features=%d",
            len(products),
            len(predictions),
            len(evidence),
            len(features),
        )

        result = compute_risk_indicators(
            products,
            sentiment,
            evidence,
            features,
            reference_date=reference_date,
            computed_at=computed_at,
            config=settings.risk,
            directions=directions,
            target_asins=target_asins,
        )
        with session_scope(factory) as session:
            written = write_risk_indicators(session, result.rows)
        logger.info("risk_indicators_written rows=%d", written)

        summary = summarize(result.rows, evidence, result.agreement, result.priors)
        report_paths = write_summary_reports(summary, Path(settings.paths.reports_dir))
        elapsed = time.perf_counter() - started
        logger.info("run_complete elapsed=%.2fs", elapsed)
        return RiskRun(
            written=written,
            summary=summary,
            report_paths=report_paths,
            elapsed_seconds=elapsed,
        )
    finally:
        engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    """Compute and store risk indicators, then print the summary.

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

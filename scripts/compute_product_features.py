"""Compute and store product-intelligence profiles for the candidate set.

Reads the SQLite serving cache (never the network), builds the peer
distributions once, profiles the requested products and upserts one
``product_features`` row per ASIN. See ``docs/product_intelligence.md`` for
every formula, tie rule and threshold.

Determinism: ``computed_at`` is the dataset build timestamp from
``data/processed/meta.json``, so two runs over the same cache write identical
rows; re-running is therefore idempotent (upsert on the ``asin`` key). The
summary printed at the end reports **counts and percentile statistics only** -
no invented numbers.

Usage::

    python scripts/compute_product_features.py
    python scripts/compute_product_features.py --limit 50
    python scripts/compute_product_features.py --asins B000050JN4 B00004T8A4
    python scripts/compute_product_features.py --config configs/config.yaml

Exit codes:
  0 - profiles written (a filter may legitimately match nothing new).
  1 - the run could not start (bad config, unreadable cache, missing
      ``meta.json``, or an ``--asins`` entry that is not in the cache).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median

from sqlalchemy.orm import Session

from munasib.core.config import Settings, load_settings
from munasib.core.exceptions import ConfigError, DataError, ProductNotFoundError
from munasib.db.models import Product
from munasib.db.repositories import ProductRepository, RelatedRepository
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)
from munasib.product_intel.features import read_build_timestamp, write_product_features
from munasib.product_intel.peers import PeerStats
from munasib.product_intel.profile import ProductProfile, build_profiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("munasib.compute_product_features")

#: How many products are profiled between two progress log lines.
PROGRESS_EVERY = 250


@dataclass(frozen=True)
class RunSummary:
    """Counts and percentile statistics for one run (printed, never guessed)."""

    #: Products whose profile was written this run.
    written: int
    #: Written profiles with no price / no rank / no brand.
    without_price: int
    without_rank: int
    without_brand: int
    #: Written profiles that carry a brand price percentile.
    with_brand_percentile: int
    #: ``name -> (min, median, max)`` over the non-None percentiles written.
    percentile_stats: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    def lines(self) -> list[str]:
        """Render the human-readable summary.

        Returns:
            One line per reported figure (always the same shape, so a diff of
            two runs reads cleanly).
        """
        return [
            f"profiles written            : {self.written:,}",
            f"without price               : {self.without_price:,}",
            f"without rank                : {self.without_rank:,}",
            f"without brand               : {self.without_brand:,}",
            f"with brand percentile       : {self.with_brand_percentile:,}",
            *(
                f"{name:<28}: min={stats[0]:.4f} median={stats[1]:.4f} max={stats[2]:.4f}"
                for name, stats in sorted(self.percentile_stats.items())
            ),
        ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute product-intelligence profiles and store them in product_features."
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
        help="Profile at most N products (ASIN order); default: every product.",
    )
    parser.add_argument(
        "--asins",
        nargs="+",
        default=None,
        metavar="ASIN",
        help="Profile exactly these ASINs (overrides --limit).",
    )
    return parser.parse_args(argv)


def select_targets(
    session: Session, limit: int | None, asins: Sequence[str] | None
) -> list[Product]:
    """Choose which products to profile, in stable ASIN order.

    Args:
        session: Open session over the serving cache.
        limit: Maximum number of products (None = no cap).
        asins: Explicit ASINs to profile, or None to take them in ASIN order.

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


def summarize(profiles: Sequence[ProductProfile]) -> RunSummary:
    """Aggregate the printed figures for one batch of profiles.

    Args:
        profiles: Profiles written this run.

    Returns:
        Counts of missing values and min/median/max over each non-None
        percentile. A percentile that is None for every profile is omitted
        rather than reported as zero.
    """
    names = (
        "price_percentile_category",
        "price_percentile_brand",
        "rank_percentile",
    )
    stats: dict[str, tuple[float, float, float]] = {}
    for name in names:
        values = sorted(
            value for profile in profiles if (value := getattr(profile, name)) is not None
        )
        if values:
            stats[name] = (values[0], float(median(values)), values[-1])
    return RunSummary(
        written=len(profiles),
        without_price=sum(1 for profile in profiles if profile.price_missing),
        without_rank=sum(1 for profile in profiles if profile.rank_missing),
        without_brand=sum(1 for profile in profiles if profile.brand_missing),
        with_brand_percentile=sum(
            1 for profile in profiles if profile.price_percentile_brand is not None
        ),
        percentile_stats=stats,
    )


def run(settings: Settings, limit: int | None, asins: Sequence[str] | None) -> RunSummary:
    """Profile the selected products and store the results.

    Args:
        settings: Loaded project settings.
        limit: Maximum number of products (None = every product).
        asins: Explicit ASINs, or None to take them in ASIN order.

    Returns:
        The run summary (counts and percentile statistics).

    Raises:
        DataError: If ``meta.json`` cannot supply the build timestamp.
        ProductNotFoundError: If an ``--asins`` entry is unknown.
    """
    engine = create_engine_from_settings(settings)
    init_db(engine)
    factory = create_session_factory(engine)
    computed_at: datetime = read_build_timestamp(settings)
    try:
        with session_scope(factory) as session:
            # Peer distributions always cover the WHOLE candidate set, even when
            # --limit/--asins profile a slice: a percentile against a truncated
            # set would silently change meaning.
            all_products = ProductRepository(session).list_all_products()
            if not all_products:
                raise DataError(
                    "the serving cache holds no products; run scripts/build_dataset.py "
                    "and scripts/ingest_db.py first"
                )
            peers = PeerStats.from_products(all_products, settings.product_intel)
            logger.info(
                "peer_stats products=%d priced=%d", peers.n_products, peers.n_priced
            )

            targets = select_targets(session, limit, asins)
            target_asins = [row.asin for row in targets]
            logger.info("targets=%d", len(targets))

            related = RelatedRepository(session).count_related(target_asins)
            profiles = build_profiles(targets, peers, settings.product_intel, related)
            for index in range(0, len(profiles), PROGRESS_EVERY):
                logger.info(
                    "profiled %d/%d", min(index + PROGRESS_EVERY, len(profiles)), len(profiles)
                )
            written = write_product_features(session, profiles, computed_at)
            logger.info("stored profiles=%d computed_at=%s", written, computed_at.isoformat())
            return summarize(profiles)
    finally:
        engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    """Compute and store product features, then print the summary.

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
        summary = run(settings, args.limit, args.asins)
    except (ConfigError, DataError, ProductNotFoundError, ValueError) as exc:
        logger.error("run_failed detail=%s", exc)
        return 1

    print()
    for line in summary.lines():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

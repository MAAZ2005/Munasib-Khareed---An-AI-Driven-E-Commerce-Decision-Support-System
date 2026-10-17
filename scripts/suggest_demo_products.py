"""Suggest candidate products for a live demo of the dashboard.

Reads the SQLite serving cache (never the network) and prints three groups of
products that make the analysis interesting, so a demo can be driven from real
ASINs instead of hand-picked ones:

* **well-reviewed** - high mean rating with plenty of reviews,
* **polarized** - a material share of both 5-star and 1-2 star reviews,
* **low-rated** - a low mean rating.

Every suggestion must also carry a stored ``image_url`` (so the thumbnail cache
can be filled for it) and at least ``demo_products.min_usable_alternatives``
usable in-subset alternatives (so the comparison/TOPSIS story has something to
compare against). All thresholds come from ``configs/config.yaml``
(``demo_products``) - nothing is hardcoded.

Usage::

    python scripts/suggest_demo_products.py
    python scripts/suggest_demo_products.py --config configs/config.yaml

Exit codes:
  0 - suggestions printed (a group may legitimately be empty).
  1 - the run could not start (bad config or an unreadable serving cache).
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import Engine, case, func, select

from munasib.core.config import DemoProductsConfig, Settings, load_settings
from munasib.core.exceptions import ConfigError
from munasib.db.models import Product, RelatedProduct, Review
from munasib.db.session import create_engine_from_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("munasib.suggest_demo_products")

#: The rating that counts as a "five star" review and the top of the "low" band.
FIVE_STAR = 5.0
LOW_STAR_MAX = 2.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Suggest demo ASINs (well-reviewed / polarized / low-rated)."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration (default: configs/config.yaml).",
    )
    return parser.parse_args(argv)


def load_candidates(engine: Engine) -> list[dict[str, object]]:
    """Load every product that has a stored image URL, with its rating shares.

    The shares are computed from the stored reviews: ``five_star_share`` and
    ``low_star_share`` count the product's 5-star and 1-2 star reviews over its
    ``n_reviews``. ``usable_alternatives`` counts DISTINCT in-subset related
    products (the product itself is never an edge in ``related_products``).

    Args:
        engine: Serving-cache engine.

    Returns:
        One dict per candidate product: ``asin``, ``title``, ``n_reviews``,
        ``mean_rating``, ``five_star_share``, ``low_star_share``,
        ``usable_alternatives``.
    """
    rating_counts = (
        select(
            Review.asin.label("asin"),
            func.sum(case((Review.rating == FIVE_STAR, 1), else_=0)).label("five_star"),
            func.sum(case((Review.rating <= LOW_STAR_MAX, 1), else_=0)).label("low_star"),
        )
        .group_by(Review.asin)
        .subquery()
    )
    alternative_counts = (
        select(
            RelatedProduct.asin.label("asin"),
            func.count(func.distinct(RelatedProduct.related_asin)).label("usable"),
        )
        .where(RelatedProduct.in_subset.is_(True))
        .group_by(RelatedProduct.asin)
        .subquery()
    )
    statement = (
        select(
            Product.asin,
            Product.title,
            Product.n_reviews,
            Product.mean_rating,
            func.coalesce(rating_counts.c.five_star, 0),
            func.coalesce(rating_counts.c.low_star, 0),
            func.coalesce(alternative_counts.c.usable, 0),
        )
        .join(rating_counts, rating_counts.c.asin == Product.asin, isouter=True)
        .join(alternative_counts, alternative_counts.c.asin == Product.asin, isouter=True)
        .where(Product.image_url.is_not(None))
        .order_by(Product.asin)
    )
    candidates: list[dict[str, object]] = []
    with engine.connect() as connection:
        for asin, title, n_reviews, mean_rating, five_star, low_star, usable in connection.execute(
            statement
        ):
            total = int(n_reviews) or 1
            candidates.append(
                {
                    "asin": str(asin),
                    "title": str(title) if title is not None else "",
                    "n_reviews": int(n_reviews),
                    "mean_rating": float(mean_rating) if mean_rating is not None else None,
                    "five_star_share": int(five_star) / total,
                    "low_star_share": int(low_star) / total,
                    "usable_alternatives": int(usable),
                }
            )
    return candidates


def select_groups(
    candidates: list[dict[str, object]], config: DemoProductsConfig
) -> dict[str, list[dict[str, object]]]:
    """Split the candidates into the three demo groups using the config thresholds.

    Args:
        candidates: Rows from :func:`load_candidates`.
        config: ``demo_products`` thresholds.

    Returns:
        ``{group_name: [candidate, ...]}`` where each group is ordered by review
        count (descending), then ASIN, and truncated to ``config.per_group``.
    """

    def eligible(candidate: dict[str, object]) -> bool:
        return int(candidate["usable_alternatives"]) >= config.min_usable_alternatives  # type: ignore[arg-type]

    def mean_rating(candidate: dict[str, object]) -> float | None:
        value = candidate["mean_rating"]
        return float(value) if value is not None else None  # type: ignore[arg-type]

    well_reviewed = [
        candidate
        for candidate in candidates
        if eligible(candidate)
        and mean_rating(candidate) is not None
        and mean_rating(candidate) >= config.well_reviewed.mean_rating_min  # type: ignore[operator]
        and int(candidate["n_reviews"]) >= config.well_reviewed.min_reviews  # type: ignore[arg-type]
    ]
    polarized = [
        candidate
        for candidate in candidates
        if eligible(candidate)
        and float(candidate["five_star_share"]) >= config.polarized.five_star_share_min  # type: ignore[arg-type]
        and float(candidate["low_star_share"]) >= config.polarized.low_star_share_min  # type: ignore[arg-type]
    ]
    low_rated = [
        candidate
        for candidate in candidates
        if eligible(candidate)
        and mean_rating(candidate) is not None
        and mean_rating(candidate) <= config.low_rated.mean_rating_max  # type: ignore[operator]
    ]

    groups = {
        "well-reviewed": well_reviewed,
        "polarized": polarized,
        "low-rated": low_rated,
    }
    for members in groups.values():
        members.sort(key=lambda item: (-int(item["n_reviews"]), str(item["asin"])))  # type: ignore[arg-type]
    return {name: members[: config.per_group] for name, members in groups.items()}


def _fmt_rating(value: object) -> str:
    """Format a mean rating for the console table.

    Args:
        value: Mean rating or None.

    Returns:
        ``"n/a"`` for None, else two decimals.
    """
    return "n/a" if value is None else f"{float(value):.2f}"  # type: ignore[arg-type]


def print_groups(groups: dict[str, list[dict[str, object]]], config: DemoProductsConfig) -> None:
    """Print the selected groups as console tables.

    Args:
        groups: Output of :func:`select_groups`.
        config: ``demo_products`` thresholds (used for the header line only).
    """
    for name, members in groups.items():
        print()
        print(f"## {name} ({len(members)}/{config.per_group})")
        if not members:
            print("   none match the configured thresholds")
            continue
        print(
            f"   {'ASIN':<11}{'reviews':>8}{'mean':>7}{'5-star':>8}{'1-2 star':>10}"
            f"{'alts':>6}  title"
        )
        for member in members:
            title = str(member["title"])
            print(
                f"   {str(member['asin']):<11}{int(member['n_reviews']):>8,}"
                f"{_fmt_rating(member['mean_rating']):>7}"
                f"{float(member['five_star_share']):>8.2f}"
                f"{float(member['low_star_share']):>10.2f}"
                f"{int(member['usable_alternatives']):>6}  {title[:60]}"
            )


def main(argv: list[str] | None = None) -> int:
    """Print demo ASIN suggestions from the serving cache.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 = printed, 1 = could not start).
    """
    args = parse_args(argv)
    try:
        settings: Settings = load_settings(args.config)
    except ConfigError as exc:
        logger.error("config_error path=%s detail=%s", args.config, exc)
        return 1

    config = settings.demo_products
    engine = create_engine_from_settings(settings)
    try:
        candidates = load_candidates(engine)
    except Exception as exc:  # pragma: no cover - defensive: mirror ingest_db ergonomics
        logger.error("cache_unreadable detail=%s", exc)
        return 1
    if not candidates:
        logger.error(
            "no_candidates detail=no product with an image_url; run "
            "python scripts/build_dataset.py and python scripts/ingest_db.py"
        )
        return 1

    groups = select_groups(candidates, config)
    print(f"Candidates with an image URL and >= {config.min_usable_alternatives} usable alternatives: {len(candidates):,}")
    print_groups(groups, config)
    for name, members in groups.items():
        for member in members:
            logger.info("suggested group=%s asin=%s", name, member["asin"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

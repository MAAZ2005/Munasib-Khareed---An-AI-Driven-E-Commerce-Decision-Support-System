"""Build the processed parquet dataset from the downloaded raw category files.

Runs the four-pass preprocessing pipeline in :mod:`munasib.data.preprocessing`
(candidate selection -> cleaning -> de-duplication -> feature derivation ->
product-level split) and writes ``products.parquet``, ``reviews.parquet``,
``related_products.parquet`` and ``meta.json`` into ``paths.processed_data_dir``
(``data/processed/``), plus ``reports/data_quality_report.{json,md}`` into
``paths.reports_dir``.

All thresholds, paths, label mappings and split fractions/seeds come from the
config file (``dataset.audit.candidate_min_reviews``,
``review_intel.sentiment.rating_mapping``, ``dataset.split``); the category
defaults to ``dataset.category`` and can be overridden with ``--category``.

Re-running is idempotent: with the same config, inputs and split seed the build
produces the same tables and splits. If the existing outputs were produced by
the same settings and input files, the run is skipped (``--force`` rebuilds
anyway).

Exit codes:
  0 - dataset built (or outputs already current).
  1 - build failed (missing raw files, no candidates); reason logged.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from munasib.core.config import Settings, load_settings
from munasib.core.exceptions import ConfigError, DataError
from munasib.data.loaders import DEFAULT_CHUNK_SIZE
from munasib.data.preprocessing import build_processed_dataset, current_settings_block

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("munasib.build_dataset")

_META_FILENAME = "meta.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build data/processed/*.parquet from the downloaded raw category files."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "UCSD Amazon Review Data (2018) category to build "
            "(default: dataset.category from the config file)."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Records per streamed chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the existing outputs are current for these settings/inputs.",
    )
    return parser.parse_args(argv)


def _missing_raw_files(settings: Settings) -> list[Path]:
    """Raw files the build needs but that do not exist yet.

    Args:
        settings: Loaded (possibly category-overridden) settings.

    Returns:
        Paths of missing files; empty when everything is present or the category
        has no registered sources (the build raises its own error then).
    """
    source = settings.dataset.sources.get(settings.dataset.category)
    if source is None:
        return []
    raw_dir = settings.paths.raw_data_dir
    expected = [
        raw_dir / Path(source.reviews_url).name,
        raw_dir / Path(source.metadata_url).name,
    ]
    return [path for path in expected if not path.exists()]


def _outputs_are_current(
    settings: Settings,
    chunk_size: int,
    *,
    processed_dir: Path,
    reports_dir: Path,
) -> dict[str, Any] | None:
    """Check whether existing outputs match this run's settings and inputs.

    Freshness = same category, same settings block (thresholds, mapping, split
    seed/fractions, chunk size), same input file sizes, and every recorded
    output file present. Anything less forces a rebuild.

    Args:
        settings: Loaded settings the run would use.
        chunk_size: Chunk size the run would use.
        processed_dir: Directory holding ``meta.json`` and the parquet tables.
        reports_dir: Directory holding the data-quality report files.

    Returns:
        The parsed ``meta.json`` dict when outputs are current, else ``None``.
    """
    meta_path = processed_dir / _META_FILENAME
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("category") != settings.dataset.category:
        return None
    if meta.get("settings") != current_settings_block(settings, chunk_size):
        return None
    for entry in meta.get("inputs", {}).values():
        path = Path(entry.get("path", ""))
        if not path.exists() or path.stat().st_size != int(entry.get("bytes", -1)):
            return None
    for name in meta.get("files", []):
        if not (processed_dir / name).exists():
            return None
    for name in meta.get("report_files", []):
        if not (reports_dir / name).exists():
            return None
    if not meta.get("report_files"):
        return None  # outputs from before the quality report existed
    return meta


def _print_summary(meta: dict[str, Any]) -> None:
    """Print the recorded dataset facts (used when the build is skipped)."""
    row_counts = meta.get("row_counts", {})
    splits = meta.get("splits", {}).get("sizes", {})
    print()
    print(f"Category                : {meta.get('category')}")
    print(f"Products (final)        : {row_counts.get('products_final', 0):,}")
    print(f"Reviews (final)         : {row_counts.get('reviews_final', 0):,}")
    print(f"Related edges           : {row_counts.get('related_edges', 0):,}")
    for split in ("train", "val", "test"):
        counts = splits.get(split, {})
        print(
            f"Split {split:<7}: {counts.get('products', 0):,} products / "
            f"{counts.get('reviews', 0):,} reviews"
        )
    print("Outputs                 : " + ", ".join(meta.get("files", [])))
    print("Reports                 : " + ", ".join(meta.get("report_files", [])))


def main(argv: list[str] | None = None) -> int:
    """Run the dataset build and print a summary.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 = built or current, 1 = failed).
    """
    args = parse_args(argv)
    if args.chunk_size < 1:
        logger.error("invalid_chunk_size value=%d minimum=1", args.chunk_size)
        return 1
    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        logger.error("config_error path=%s detail=%s", args.config, exc)
        return 1
    if args.category:
        settings = settings.model_copy(deep=True)
        settings.dataset.category = args.category

    category = settings.dataset.category
    processed_dir = Path(settings.paths.processed_data_dir)
    reports_dir = Path(settings.paths.reports_dir)

    missing = _missing_raw_files(settings)
    if missing:
        logger.error(
            "raw_files_missing category=%s files=%s",
            category,
            ",".join(str(path) for path in missing),
        )
        logger.error(
            "action=python scripts/download_data.py --category %s", category
        )
        return 1

    if not args.force:
        current = _outputs_are_current(
            settings, args.chunk_size, processed_dir=processed_dir, reports_dir=reports_dir
        )
        if current is not None:
            logger.info(
                "outputs_current category=%s detail=same settings, inputs and files; "
                "pass --force to rebuild",
                category,
            )
            _print_summary(current)
            return 0

    split = settings.dataset.split
    logger.info(
        "build_start category=%s chunk_size=%d split_seed=%d fractions=%.2f/%.2f/%.2f "
        "config=%s",
        category,
        args.chunk_size,
        split.seed,
        split.train_fraction,
        split.val_fraction,
        split.test_fraction,
        args.config,
    )
    try:
        report = build_processed_dataset(
            settings, config_path=str(args.config), chunk_size=args.chunk_size
        )
    except DataError as exc:
        logger.error("build_failed category=%s detail=%s", category, exc)
        return 1

    print()
    print(f"Category                : {report.category}")
    print(f"Candidate threshold     : >= {report.candidate_min_reviews} reviews/product")
    print(f"Products (candidates)   : {report.products_selected:,}")
    print(f"Products (final)        : {report.products_final:,}")
    print(f"Reviews (final)         : {report.reviews_final:,}")
    print(f"  duplicates dropped    : {report.duplicate_reviews_dropped:,}")
    print("  without review text   : flagged has_text=False (rows kept)")
    print(f"Related edges           : {report.related_edges:,}")
    print(f"  in subset             : {report.related_in_subset:,}")
    print(f"Review date range       : {report.review_date_min} .. {report.review_date_max}")
    for split_name in ("train", "val", "test"):
        counts = report.split_sizes.get(split_name, {})
        print(
            f"Split {split_name:<7}: {counts.get('products', 0):,} products / "
            f"{counts.get('reviews', 0):,} reviews"
        )
    print("Outputs                 : " + ", ".join(report.output_files))
    print("Reports                 : " + ", ".join(report.report_files))
    logger.info(
        "build_done category=%s products=%d reviews=%d related=%d split_seed=%d",
        category,
        report.products_final,
        report.reviews_final,
        report.related_edges,
        split.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

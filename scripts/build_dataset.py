"""Build the processed parquet dataset from the downloaded raw category files.

Runs the four-pass preprocessing pipeline in :mod:`munasib.data.preprocessing`
(candidate selection -> cleaning -> de-duplication -> feature derivation) and
writes ``products.parquet``, ``reviews.parquet``, ``related_products.parquet``
and ``meta.json`` into ``paths.processed_data_dir`` (``data/processed/``).

All thresholds, paths and label mappings come from the config file
(``dataset.audit.candidate_min_reviews``, ``review_intel.sentiment.rating_mapping``);
switching category stays a one-line change in ``dataset.category``.

Exit codes:
  0 - dataset built; outputs listed in the final summary.
  1 - build failed (missing raw files, no candidates); reason logged.
"""

from __future__ import annotations

import argparse
import logging
import sys

from munasib.core.config import load_settings
from munasib.core.exceptions import DataError
from munasib.data.loaders import DEFAULT_CHUNK_SIZE
from munasib.data.preprocessing import build_processed_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("munasib.build_dataset")


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
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Records per streamed chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the dataset build and print a summary.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 = built, 1 = failed).
    """
    args = parse_args(argv)
    if args.chunk_size < 1:
        logger.error("--chunk-size must be >= 1")
        return 1
    settings = load_settings(args.config)

    try:
        report = build_processed_dataset(
            settings, config_path=str(args.config), chunk_size=args.chunk_size
        )
    except DataError as exc:
        logger.error("%s", exc)
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
    print("Outputs                 : " + ", ".join(report.output_files))
    return 0


if __name__ == "__main__":
    sys.exit(main())

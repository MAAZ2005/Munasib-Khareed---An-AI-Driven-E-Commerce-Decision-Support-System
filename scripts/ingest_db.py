"""Load data/processed/*.parquet into the SQLite serving cache.

The database is a cache of the processed subset (locked design decision #6),
so this script mirrors the three parquet tables into SQLite:

* ``products`` / ``reviews`` / ``related_products`` - bulk-inserted in batches,
  one transaction per table (``munasib.db.session.session_scope``).
* ``review_features`` / ``product_features`` / ``analysis_results`` - created
  empty; later pipeline stages fill them.

Null handling is explicit: pandas nulls (``None``/``NaN``/``pd.NA``/``NaT``)
become SQL NULL, so e.g. an unknown ``verified`` stays NULL and is never
turned into ``False``; timestamps are stored as UTC (converted with
``tz_convert("UTC")`` before the offset is dropped - review dates span
2001-11-08 .. 2018-10-04, today's date never appears).

Usage::

    python scripts/ingest_db.py --force   # drop + rebuild + verify
    python scripts/ingest_db.py           # idempotent: skips when current
    python scripts/ingest_db.py --verify  # verification only, no loading

Re-running without ``--force`` is idempotent: when the stored row counts and
the products' split distribution already match the parquet files, loading is
skipped; otherwise the cache tables are cleared (children before parents) and
reloaded. Either way the final row counts are identical to the parquet counts
and no duplicates can exist (primary keys/unique constraints enforce it).

The one cache value with no parquet counterpart is ``products.image_cached``
(whether the thumbnail exists locally). It is never copied from parquet: after
every run the flag is re-derived from the ``<asin>.jpg`` files that actually
exist in ``images.directory``, so a rebuild keeps the images that were already
downloaded and drops flags for files that are gone. ``--verify`` fails when the
two disagree.

Exit codes:
  0 - loaded or skipped, and every verification check passed.
  1 - failed (missing parquet, load error, or a verification failure).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import delete, func, insert, select, text, update
from sqlalchemy.engine import Engine

from munasib.core.config import load_settings
from munasib.core.constants import ASIN_PATTERN, IMAGE_FILENAME_SUFFIX
from munasib.core.exceptions import ConfigError, DataError
from munasib.db.models import (
    AnalysisResult,
    Base,
    Product,
    ProductFeature,
    RelatedProduct,
    Review,
    ReviewFeature,
)
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("munasib.ingest_db")

#: Records per bulk-insert batch (one transaction per table, many batches).
DEFAULT_INGEST_BATCH_SIZE = 10_000

PRODUCT_FILENAME = "products.parquet"
REVIEW_FILENAME = "reviews.parquet"
RELATED_FILENAME = "related_products.parquet"

# Parquet -> table column renames (documented in munasib.db.models).
PRODUCT_RENAME = {"description_clean": "description", "features_clean": "features"}
REVIEW_RENAME = {"review_datetime": "review_time", "review_year": "year", "review_month": "month"}

# Exact parquet columns this script consumes (schema-drift guard: a missing or
# unexpected column fails loudly instead of silently ingesting wrong data).
EXPECTED_PRODUCT_COLUMNS = (
    "asin", "split", "title", "title_clean", "brand", "brand_clean", "brand_missing",
    "price", "price_max", "price_status", "price_missing", "rank", "rank_value",
    "rank_category", "rank_missing", "description_clean", "features_clean",
    "n_reviews", "mean_rating", "first_review_time", "last_review_time",
    "image_url", "image_missing",
)
EXPECTED_REVIEW_COLUMNS = (
    "review_id", "asin", "split", "reviewer_id", "rating", "review_text_raw",
    "review_text_clean", "summary_raw", "summary_clean", "has_text", "text_length",
    "helpful_votes", "verified", "review_datetime", "review_year", "review_month",
    "sentiment_label_weak",
)
EXPECTED_RELATED_COLUMNS = ("asin", "related_asin", "relation_type", "in_subset")

#: Table columns that have no parquet counterpart, with their ingest default.
#: ``image_cached`` is a serving-cache flag (see munasib.db.models.Product): a
#: freshly ingested row has no local thumbnail yet, so it starts False. The real
#: value is then derived from the thumbnail files on disk by
#: :func:`sync_image_flags`, so a rebuild never loses what was already downloaded.
EXTRAS = {
    "products": {"image_cached": False},
}

#: Columns ``--verify`` requires on the products table (schema-drift check).
VERIFIED_PRODUCT_COLUMNS = ("image_url", "image_missing", "image_cached")

#: Tables loaded from parquet, children before parents for clearing.
CACHE_TABLES = ("reviews", "related_products", "products")
#: Tables filled by later pipeline stages; reported but never loaded here.
EMPTY_TABLES = ("review_features", "product_features", "analysis_results")
CLEAR_ORDER = (*EMPTY_TABLES, "related_products", "reviews", "products")

_TABLE_MODELS = {
    "products": Product,
    "reviews": Review,
    "related_products": RelatedProduct,
    "review_features": ReviewFeature,
    "product_features": ProductFeature,
    "analysis_results": AnalysisResult,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Load data/processed/*.parquet into the SQLite serving cache."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and rebuild every table even when the cache already matches the parquet files.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run only the verification report (no loading) and exit.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_INGEST_BATCH_SIZE,
        help=f"Rows per bulk-insert batch (default: {DEFAULT_INGEST_BATCH_SIZE}).",
    )
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------
# Value conversion (pandas -> DBAPI-safe Python values)
# ------------------------------------------------------------------------------


def _to_scalar(value: object) -> object:
    """Convert one parquet cell into a DBAPI-safe value with proper NULLs.

    Rules (never fabricate data):

    * ``None``/``pd.NA``/``pd.NaT``/``NaN`` -> ``None`` (SQL NULL).
    * numpy scalars -> native Python scalars (the sqlite3 driver rejects
      ``np.bool_`` etc.).
    * tz-aware timestamps -> UTC wall time with the offset dropped (SQLite
      stores no offset; every datetime in this database is UTC by contract).

    Args:
        value: One cell from ``DataFrame.to_dict("records")``.

    Returns:
        ``None`` (NULL) or a native Python scalar.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        if value.tzinfo is not None:
            return value.tz_convert("UTC").tz_localize(None).to_pydatetime()
        return value.to_pydatetime()
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


def _to_features_json(value: object) -> list | None:
    """JSON-encode the product feature bullet list (or NULL it).

    Args:
        value: ``features_clean`` cell: list of strings, or a missing marker.

    Returns:
        The list unchanged (the JSON column serialises it) or ``None``.

    Raises:
        DataError: If the cell holds something that is neither a list nor a
            missing value - stored data is never guessed at.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        # pandas 3 materialises list-typed parquet cells as numpy arrays.
        value = value.tolist()
    if isinstance(value, list | tuple):
        return list(value)
    raise DataError(f"features_clean cell is not a list: {value!r}")


def _convert_record(record: dict, rename: dict[str, str], extras: dict | None = None) -> dict:
    """Convert one parquet record into an insert payload for a table.

    Args:
        record: One row as a ``{parquet_column: value}`` dict.
        rename: Parquet-to-table column renames.
        extras: Table columns with no parquet counterpart (e.g. the cache-only
            ``image_cached``), added with their default value.

    Returns:
        ``{table_column: converted_value}`` with proper NULLs.
    """
    row: dict = {}
    for column, value in record.items():
        target = rename.get(column, column)
        if target == "features":
            row[target] = _to_features_json(value)
        else:
            row[target] = _to_scalar(value)
    if extras:
        for column, value in extras.items():
            row.setdefault(column, value)
    return row


def _check_columns(frame: pd.DataFrame, expected: Sequence[str], table: str) -> None:
    """Guard against parquet schema drift before ingesting.

    Args:
        frame: Loaded parquet table.
        expected: Exact column names the script knows how to load.
        table: Table name for the error message.

    Raises:
        DataError: On any missing or unexpected column.
    """
    actual = list(frame.columns)
    missing = [column for column in expected if column not in actual]
    extra = [column for column in actual if column not in expected]
    if missing or extra:
        raise DataError(
            f"{table} schema mismatch; missing={missing} unexpected={extra} - "
            "re-run python scripts/build_dataset.py or update scripts/ingest_db.py"
        )


# ------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------


def _batches(frame: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    """Yield consecutive row slices of ``frame``.

    Args:
        frame: Source rows.
        batch_size: Maximum rows per slice.

    Yields:
        Slices with at most ``batch_size`` rows.
    """
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size]


def _load_table(
    engine: Engine,
    model: type,
    frame: pd.DataFrame,
    rename: dict[str, str],
    batch_size: int,
    extras: dict | None = None,
) -> int:
    """Bulk-insert one parquet frame in a single transaction per table.

    Args:
        engine: Target engine.
        model: ORM class of the destination table.
        frame: Source rows (columns already verified).
        rename: Parquet-to-table column renames.
        batch_size: Rows per bulk-insert batch.
        extras: Table columns with no parquet counterpart, with ingest defaults.

    Returns:
        Number of rows inserted.
    """
    if frame.empty:
        return 0
    factory = create_session_factory(engine)
    inserted = 0
    with session_scope(factory) as session:
        for batch in _batches(frame, batch_size):
            rows = [
                _convert_record(record, rename, extras)
                for record in batch.to_dict("records")
            ]
            session.execute(insert(model), rows)
            inserted += len(rows)
    return inserted


def clear_cache_tables(engine: Engine) -> None:
    """Delete every row, children before parents (foreign keys stay valid).

    Args:
        engine: Target engine.
    """
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        for name in CLEAR_ORDER:
            session.execute(delete(Base.metadata.tables[name]))
    logger.info("cache_cleared tables=%s", ",".join(CLEAR_ORDER))


def cached_image_asins(images_dir: Path) -> set[str]:
    """ASINs with a thumbnail file in the local image cache.

    Only ``<10 alphanumeric chars>.jpg`` files count, so a stray file (or the
    ``.gitkeep`` placeholder) can never flip a product's flag.

    Args:
        images_dir: Thumbnail cache directory (``images.directory``).

    Returns:
        The ASINs whose thumbnail is present; empty when the directory is absent.
    """
    if not images_dir.is_dir():
        return set()
    return {
        path.stem
        for path in images_dir.glob(f"*{IMAGE_FILENAME_SUFFIX}")
        if ASIN_PATTERN.match(path.stem)
    }


def sync_image_flags(engine: Engine, images_dir: Path) -> int:
    """Point ``products.image_cached`` at the thumbnails that exist on disk.

    ``image_cached`` is the only serving-cache value that is not in the parquet
    files, so it is derived here: a product is flagged exactly when
    ``<images_dir>/<asin>.jpg`` exists. Running this after every load (and on a
    plain ``ingest_db.py`` run) means a rebuild or a re-ingest can never lose the
    flags the downloader set, and never keep a flag for a file that was removed.

    Args:
        engine: Target engine (the ``products`` table must already exist).
        images_dir: Thumbnail cache directory (``images.directory``).

    Returns:
        Number of products flagged ``image_cached=true`` afterwards.
    """
    asins = cached_image_asins(images_dir)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        session.execute(update(Product).values(image_cached=False))
        if asins:
            session.execute(
                update(Product).where(Product.asin.in_(asins)).values(image_cached=True)
            )
        flagged = session.scalar(
            select(func.count()).select_from(Product).where(Product.image_cached.is_(True))
        )
    logger.info(
        "image_flags_synced cache=%s files=%d flagged=%d",
        images_dir,
        len(asins),
        int(flagged or 0),
    )
    return int(flagged or 0)


def ingest_parquet(
    engine: Engine,
    processed_dir: Path,
    batch_size: int,
) -> dict[str, int]:
    """Load the three processed parquet tables into the serving cache.

    Args:
        engine: Target engine (tables must already exist).
        processed_dir: Directory holding ``*.parquet``.
        batch_size: Rows per bulk-insert batch.

    Returns:
        Rows inserted per table.

    Raises:
        DataError: If a parquet file is missing or its schema drifted.
    """
    counts: dict[str, int] = {}
    for filename, table, rename, expected in (
        (PRODUCT_FILENAME, "products", PRODUCT_RENAME, EXPECTED_PRODUCT_COLUMNS),
        (REVIEW_FILENAME, "reviews", REVIEW_RENAME, EXPECTED_REVIEW_COLUMNS),
        (RELATED_FILENAME, "related_products", {}, EXPECTED_RELATED_COLUMNS),
    ):
        path = processed_dir / filename
        if not path.exists():
            raise DataError(
                f"missing processed file: {path} - run python scripts/build_dataset.py first"
            )
        logger.info("loading table=%s file=%s", table, path)
        frame = pd.read_parquet(path)
        _check_columns(frame, expected, table)
        # Products first: reviews/related reference them (foreign keys are ON).
        counts[table] = _load_table(
            engine, _TABLE_MODELS[table], frame, rename, batch_size, EXTRAS.get(table)
        )
        logger.info("loaded table=%s rows=%d", table, counts[table])
        del frame
    return counts


# ------------------------------------------------------------------------------
# Freshness and verification
# ------------------------------------------------------------------------------


def parquet_row_counts(processed_dir: Path) -> dict[str, int]:
    """Read parquet row counts from file metadata (no full load).

    Args:
        processed_dir: Directory holding ``*.parquet``.

    Returns:
        Row counts for the three cache tables.

    Raises:
        DataError: If a parquet file is missing.
    """
    counts: dict[str, int] = {}
    for filename, table in (
        (PRODUCT_FILENAME, "products"),
        (REVIEW_FILENAME, "reviews"),
        (RELATED_FILENAME, "related_products"),
    ):
        path = processed_dir / filename
        if not path.exists():
            raise DataError(
                f"missing processed file: {path} - run python scripts/build_dataset.py first"
            )
        counts[table] = pq.ParquetFile(path).metadata.num_rows
    return counts


def db_row_counts(engine: Engine) -> dict[str, int]:
    """Count rows in every table of the serving cache.

    Args:
        engine: Target engine.

    Returns:
        ``{table_name: row_count}`` for all six tables.
    """
    counts: dict[str, int] = {}
    with engine.connect() as connection:
        for name in _TABLE_MODELS:
            # Table names come from Base.metadata only, never from user input.
            total = connection.scalar(text(f"SELECT COUNT(*) FROM {name}"))
            counts[name] = int(total or 0)
    return counts


def products_split_distribution(processed_dir: Path) -> dict[str, int]:
    """Products-per-split from ``products.parquet``.

    Args:
        processed_dir: Directory holding ``products.parquet``.

    Returns:
        ``{split: product_count}`` (e.g. ``{"train": 1444, ...}``).
    """
    frame = pd.read_parquet(processed_dir / PRODUCT_FILENAME, columns=["split"])
    distribution = frame["split"].value_counts(sort=False)
    return {str(key): int(value) for key, value in distribution.items()}


def db_split_distribution(engine: Engine) -> dict[str, int]:
    """Products-per-split from the ``products`` table.

    Args:
        engine: Target engine.

    Returns:
        ``{split: product_count}``.
    """
    with engine.connect() as connection:
        rows = connection.execute(
            select(Product.split, func.count()).group_by(Product.split).order_by(Product.split)
        )
        return {str(split): int(total) for split, total in rows}


def cache_is_fresh(engine: Engine, processed_dir: Path) -> bool:
    """Decide whether the cache already mirrors the parquet files.

    Fresh = same row counts in all three cache tables **and** the same
    products-per-split distribution (the distribution catches a cache built
    with a different split seed, where plain counts would coincidentally match).

    Args:
        engine: Target engine.
        processed_dir: Directory holding the parquet files.

    Returns:
        True when loading can be skipped safely.
    """
    expected = parquet_row_counts(processed_dir)
    stored = db_row_counts(engine)
    if any(stored[table] != expected[table] for table in expected):
        return False
    return db_split_distribution(engine) == products_split_distribution(processed_dir)


def verify(engine: Engine, processed_dir: Path, images_dir: Path) -> bool:
    """Verify the serving cache against the parquet files and SQLite itself.

    Prints, for every check, only values the code computes:

    1. Row counts per cache table versus the parquet row counts.
    2. Row counts of the pipeline-owned tables (informational).
    3. The ``products`` table carries the image columns (parquet mirror plus the
       cache-only ``image_cached`` flag) - catches a stale database built before
       the columns were added, which ``create_all`` never migrates.
    4. ``image_cached`` agrees with the thumbnail files on disk: no flagged
       product may lack its file (the API would answer 404 for a product that
       claims ``has_image=true``) and no cached file may be unflagged.
    5. ``PRAGMA foreign_key_check`` - must return no rows.
    6. ``PRAGMA integrity_check`` - must return exactly ``ok``.
    7. Database file size on disk.

    Args:
        engine: Target engine.
        processed_dir: Directory holding the parquet files.
        images_dir: Thumbnail cache directory (``images.directory``).

    Returns:
        True when every check passes (a raised ``DataError`` on missing
        parquet files propagates to the caller as a failure).
    """
    # Idempotent: guarantees the tables exist so a fresh database reports
    # zero counts (and therefore a FAIL) instead of raising.
    init_db(engine)
    expected = parquet_row_counts(processed_dir)
    stored = db_row_counts(engine)
    ok = True

    print()
    print(f"{'Table':<22}{'parquet':>12}{'database':>12}   status")
    print("-" * 62)
    for table in ("products", "reviews", "related_products"):
        match = stored[table] == expected[table]
        ok = ok and match
        print(
            f"{table:<22}{expected[table]:>12,}{stored[table]:>12,}   "
            f"{'ok' if match else 'MISMATCH'}"
        )
    for table in EMPTY_TABLES:
        print(f"{table:<22}{'-':>12}{stored[table]:>12,}   filled by later pipeline stages")

    with engine.connect() as connection:
        product_columns = {
            str(row[1])
            for row in connection.exec_driver_sql("PRAGMA table_info(products)").fetchall()
        }
        fk_rows = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
        integrity = [
            row[0] for row in connection.exec_driver_sql("PRAGMA integrity_check").fetchall()
        ]
        # Only usable once the flag column is known to exist (previous check).
        flag_rows = (
            connection.execute(select(Product.asin, Product.image_cached)).all()
            if VERIFIED_PRODUCT_COLUMNS[-1] in product_columns
            else []
        )
    missing_columns = [c for c in VERIFIED_PRODUCT_COLUMNS if c not in product_columns]
    columns_ok = not missing_columns
    fk_ok = not fk_rows
    integrity_ok = integrity == ["ok"]

    flagged = {str(asin) for asin, cached in flag_rows if cached}
    product_asins = {str(asin) for asin, _ in flag_rows}
    files = cached_image_asins(images_dir)
    dangling = sorted(flagged - files)  # claims an image the API cannot serve
    unmarked = sorted((files & product_asins) - flagged)  # downloaded but unflagged
    stale_files = sorted(files - product_asins)  # not a product in this subset
    images_ok = columns_ok and not dangling and not unmarked
    ok = ok and columns_ok and images_ok and fk_ok and integrity_ok

    print()
    print(
        "products columns         : "
        + ("ok" if columns_ok else f"MISSING {', '.join(missing_columns)}")
    )
    print(f"images cached on disk    : {len(files & product_asins):,} (in this subset)")
    if columns_ok:
        print(
            f"image_cached flag        : {len(flagged):,} flagged, "
            f"{len(dangling):,} without a file, {len(unmarked):,} file(s) unflagged"
        )
    else:
        print("image_cached flag        : not checked (column missing)")
    if stale_files:
        print(f"  files with no product  : {len(stale_files):,} (ignored)")
    print(f"PRAGMA foreign_key_check : {'empty (ok)' if fk_ok else f'{len(fk_rows)} violation(s)'}")
    print(f"PRAGMA integrity_check    : {', '.join(map(str, integrity))}")

    db_file = engine.url.database
    if db_file and db_file not in (":memory:",):
        size = Path(db_file).stat().st_size if Path(db_file).exists() else 0
        print(f"Database file            : {db_file} ({size:,} bytes)")
    else:
        print("Database file            : in-memory (no file)")

    print(f"\nVERIFICATION              : {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Load (or skip) the serving cache and verify it.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 = loaded/skipped and verified, 1 = failure).
    """
    args = parse_args(argv)
    if args.batch_size < 1:
        logger.error("invalid_batch_size value=%d minimum=1", args.batch_size)
        return 1
    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        logger.error("config_error path=%s detail=%s", args.config, exc)
        return 1

    processed_dir = Path(settings.paths.processed_data_dir)
    images_dir = Path(settings.images.directory)
    # SQLite creates the file but not its parent directory.
    db_file = Path(settings.paths.db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine_from_settings(settings)
    try:
        if args.verify:
            logger.info("verify_only db=%s", engine.url)
            try:
                return 0 if verify(engine, processed_dir, images_dir) else 1
            except DataError as exc:
                logger.error("verify_failed detail=%s", exc)
                return 1

        # Fail on missing parquet files before touching the existing schema.
        parquet_row_counts(processed_dir)

        if args.force:
            Base.metadata.drop_all(engine)
            logger.info("schema_dropped detail=--force rebuild")
            init_db(engine)
            loaded = ingest_parquet(engine, processed_dir, args.batch_size)
            logger.info(
                "ingest_done source=force products=%d reviews=%d related=%d",
                loaded["products"],
                loaded["reviews"],
                loaded["related_products"],
            )
        else:
            init_db(engine)
            if cache_is_fresh(engine, processed_dir):
                counts = db_row_counts(engine)
                logger.info(
                    "cache_current detail=counts and split distribution match parquet; "
                    "pass --force to reload"
                )
                logger.info(
                    "cache_counts products=%d reviews=%d related=%d",
                    counts["products"],
                    counts["reviews"],
                    counts["related_products"],
                )
            else:
                clear_cache_tables(engine)
                loaded = ingest_parquet(engine, processed_dir, args.batch_size)
                logger.info(
                    "ingest_done source=reload products=%d reviews=%d related=%d",
                    loaded["products"],
                    loaded["reviews"],
                    loaded["related_products"],
                )

        # ``image_cached`` lives only in the cache, so it is re-derived from the
        # thumbnail files after every load (and on a no-op run) rather than copied
        # from parquet: a rebuild can then never lose the flags of the images that
        # were already downloaded.
        sync_image_flags(engine, images_dir)

        ok = verify(engine, processed_dir, images_dir)
        return 0 if ok else 1
    except DataError as exc:
        logger.error("ingest_failed detail=%s", exc)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())

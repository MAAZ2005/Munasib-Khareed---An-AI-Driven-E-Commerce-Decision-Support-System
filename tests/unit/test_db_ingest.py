"""Tests for the ``scripts/ingest_db.py`` CLI wrapper.

Builds a tiny processed-parquet fixture (3 products, 4 reviews, 3 related
edges) in a temp directory plus a temp config pointing ``db_path`` there, then
exercises the full CLI flows: load + verify, idempotent re-run, ``--force``,
``--verify``, stale-split detection, missing-parquet failure, and the pandas
value-conversion helpers (NULL vs False, numpy scalars, UTC timestamps).
Hermetic: never touches the real ``data/`` directories or network.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select, text

from munasib.core.constants import (
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABEL_POSITIVE,
)
from munasib.core.exceptions import DataError
from munasib.db.models import Product, Review
from munasib.db.session import create_engine_from_settings, init_db

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ingest_db.py"


def _load_module():
    """Import ``scripts/ingest_db.py`` as a module.

    Returns:
        The loaded script module.
    """
    spec = importlib.util.spec_from_file_location("munasib_ingest_db_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Tiny processed-parquet fixture (real column names, real dtypes)
# ------------------------------------------------------------------------------


def _products_frame() -> pd.DataFrame:
    """Build the fixture ``products.parquet`` content (3 rows).

    Returns:
        Frame with the exact parquet columns, including a missing brand,
        price, rank and feature list to prove NULL handling.
    """
    return pd.DataFrame(
        {
            "asin": ["B000000001", "B000000002", "B000000003"],
            "split": ["train", "val", "train"],
            "title": ["Acme Widget", "Beta Gadget", "Gamma Slicer"],
            "title_clean": ["acme widget", "beta gadget", "gamma slicer"],
            "brand": ["Acme", None, "GammaCo"],
            "brand_clean": ["acme", None, "gammaco"],
            "brand_missing": [False, True, False],
            "price": [19.99, np.nan, 5.25],
            "price_max": pd.Series([np.nan, np.nan, np.nan], dtype="float64"),
            "price_status": ["ok", "missing", "ok"],
            "price_missing": [False, True, False],
            "rank": ["#1 in Widgets", None, "#9 in Slicers"],
            "rank_value": pd.array([1, None, 9], dtype="Int64"),
            "rank_category": ["Widgets", None, "Slicers"],
            "rank_missing": [False, True, False],
            "description_clean": ["a widget", None, "a slicer"],
            "features_clean": [["steel", "compact"], None, ["plastic"]],
            "n_reviews": [2, 1, 1],
            "mean_rating": [4.5, 3.0, 5.0],
            "first_review_time": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2018-01-03"], utc=True
            ),
            "last_review_time": pd.to_datetime(
                ["2018-01-02", "2018-01-03", "2018-01-04"], utc=True
            ),
            # Row 2 has no usable image URL (the flag mirrors the null).
            "image_url": [
                "https://example.com/one.jpg",
                None,
                "https://example.com/three.jpg",
            ],
            "image_missing": [False, True, False],
        }
    )


def _reviews_frame() -> pd.DataFrame:
    """Build the fixture ``reviews.parquet`` content (4 rows).

    Returns:
        Frame with the exact parquet columns; row 3 has ``verified`` unknown
        (NULL) and row 4 is a kept no-text review (``has_text=False``).
    """
    return pd.DataFrame(
        {
            "review_id": [
                "0000000000000001",
                "0000000000000002",
                "0000000000000003",
                "0000000000000004",
            ],
            "asin": ["B000000001", "B000000001", "B000000002", "B000000003"],
            "split": ["train", "train", "val", "train"],
            "reviewer_id": ["u1", "u2", "u3", "u4"],
            "rating": [5.0, 1.0, 3.0, 5.0],
            "review_text_raw": ["Great widget", "Terrible", "It is okay", None],
            "review_text_clean": ["great widget", "terrible", "it is okay", None],
            "summary_raw": ["Great", "Bad", "Ok", None],
            "summary_clean": ["great", "bad", "ok", None],
            "has_text": [True, True, True, False],
            "text_length": [12, 8, 9, 0],
            "helpful_votes": [2, 0, 1, 0],
            "verified": pd.array([True, False, None, True], dtype="boolean"),
            "review_datetime": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], utc=True
            ),
            "review_year": pd.array([2018, 2018, 2018, 2018], dtype="Int16"),
            "review_month": pd.array([1, 1, 1, 1], dtype="Int8"),
            "sentiment_label_weak": pd.array(
                [
                    SENTIMENT_LABEL_POSITIVE,
                    SENTIMENT_LABEL_NEGATIVE,
                    SENTIMENT_LABEL_NEUTRAL,
                    SENTIMENT_LABEL_POSITIVE,
                ],
                dtype="Int8",
            ),
        }
    )


def _related_frame() -> pd.DataFrame:
    """Build the fixture ``related_products.parquet`` content (3 edges).

    Returns:
        Frame with the exact parquet columns, including one edge pointing
        outside the subset.
    """
    return pd.DataFrame(
        {
            "asin": ["B000000001", "B000000001", "B000000002"],
            "related_asin": ["B000000002", "OUTSIDE99", "B000000001"],
            "relation_type": ["also_bought", "also_viewed", "also_bought"],
            "in_subset": [True, False, True],
        }
    )


def _write_parquet_fixture(processed: Path) -> None:
    """Write the three fixture parquet tables.

    Args:
        processed: Directory to create and fill.
    """
    processed.mkdir(parents=True, exist_ok=True)
    _products_frame().to_parquet(processed / "products.parquet", index=False)
    _reviews_frame().to_parquet(processed / "reviews.parquet", index=False)
    _related_frame().to_parquet(processed / "related_products.parquet", index=False)


def _write_config(path: Path, processed: Path, db_path: Path) -> Path:
    """Write a minimal config pointing at the temporary processed dir and DB.

    Args:
        path: Destination YAML path.
        processed: Directory holding the fixture parquet tables.
        db_path: SQLite file to create/use.

    Returns:
        The config path.
    """
    path.write_text(
        f"""
paths:
  processed_data_dir: "{processed.as_posix()}"
  db_path: "{db_path.as_posix()}"
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def workspace(tmp_path: Path) -> Iterator[dict[str, Path]]:
    """Temp workspace with fixture parquet, config and DB path.

    Yields:
        Dict with ``processed``, ``config`` and ``db`` paths.
    """
    processed = tmp_path / "processed"
    _write_parquet_fixture(processed)
    db_path = tmp_path / "db" / "cache.sqlite"
    config = _write_config(tmp_path / "config.yaml", processed, db_path)
    yield {"processed": processed, "config": config, "db": db_path}


def _counts(db_path: Path) -> dict[str, int]:
    """Row counts of every table in the fixture database.

    Args:
        db_path: SQLite file.

    Returns:
        ``{table: row_count}``.
    """
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    init_db(engine)
    try:
        with engine.connect() as connection:
            return {
                name: int(connection.scalar(text(f"SELECT COUNT(*) FROM {name}")) or 0)
                for name in (
                    "products",
                    "reviews",
                    "related_products",
                    "review_features",
                    "product_features",
                    "analysis_results",
                )
            }
    finally:
        engine.dispose()


# ------------------------------------------------------------------------------
# CLI flows
# ------------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    """Defaults point at the master config and the standard batch size."""
    module = _load_module()
    args = module.parse_args([])
    assert args.config == "configs/config.yaml"
    assert args.force is False
    assert args.verify is False
    assert args.batch_size == module.DEFAULT_INGEST_BATCH_SIZE == 10_000


def test_parse_args_overrides() -> None:
    """Flags override config path, batch size and booleans."""
    module = _load_module()
    args = module.parse_args(
        ["--config", "other.yaml", "--batch-size", "500", "--force", "--verify"]
    )
    assert args.config == "other.yaml"
    assert args.batch_size == 500
    assert args.force is True
    assert args.verify is True


def test_main_rejects_invalid_batch_size(workspace: dict[str, Path]) -> None:
    """A non-positive batch size exits 1 before opening anything."""
    module = _load_module()
    assert module.main(["--config", str(workspace["config"]), "--batch-size", "0"]) == 1


def test_ingest_loads_every_table(workspace: dict[str, Path]) -> None:
    """A full run loads the three cache tables and passes verification."""
    module = _load_module()
    assert module.main(["--config", str(workspace["config"])]) == 0
    counts = _counts(workspace["db"])
    assert counts["products"] == 3
    assert counts["reviews"] == 4
    assert counts["related_products"] == 3
    assert counts["review_features"] == 0  # filled by later prompts
    assert counts["product_features"] == 0
    assert counts["analysis_results"] == 0


def test_ingest_preserves_null_versus_false(workspace: dict[str, Path]) -> None:
    """verified NULL stays NULL, False stays False; NULLs are not fabricated."""
    module = _load_module()
    assert module.main(["--config", str(workspace["config"])]) == 0
    engine = create_engine_from_settings(db_url=f"sqlite:///{workspace['db']}")
    try:
        with engine.connect() as connection:
            verified = {
                row[0]: row[1]
                for row in connection.execute(
                    text("SELECT review_id, verified FROM reviews ORDER BY review_id")
                )
            }
            missing_price = connection.execute(
                select(Product.price, Product.price_missing).where(Product.asin == "B000000002")
            ).one()
            no_text = connection.execute(
                select(Review.has_text, Review.review_text_clean).where(
                    Review.review_id == "0000000000000004"
                )
            ).one()
            features = connection.execute(
                select(Product.features).where(Product.asin == "B000000001")
            ).scalar_one()
            naive_time = connection.execute(
                select(Product.last_review_time).where(Product.asin == "B000000003")
            ).scalar_one()
    finally:
        engine.dispose()
    assert verified["0000000000000001"] == 1  # True
    assert verified["0000000000000002"] == 0  # False, not NULL
    assert verified["0000000000000003"] is None  # unknown, kept NULL
    assert missing_price.price is None and missing_price.price_missing is True
    assert no_text.has_text is False and no_text.review_text_clean is None
    assert features == ["steel", "compact"]
    assert naive_time.tzinfo is None  # stored as naive UTC


def test_ingest_sets_image_columns(workspace: dict[str, Path]) -> None:
    """``image_url``/``image_missing`` mirror the parquet; ``image_cached`` starts False."""
    module = _load_module()
    assert module.main(["--config", str(workspace["config"])]) == 0
    engine = create_engine_from_settings(db_url=f"sqlite:///{workspace['db']}")
    try:
        with engine.connect() as connection:
            rows = {
                row.asin: (row.image_url, row.image_missing, row.image_cached)
                for row in connection.execute(
                    select(
                        Product.asin, Product.image_url, Product.image_missing, Product.image_cached
                    )
                )
            }
    finally:
        engine.dispose()
    assert rows["B000000001"] == ("https://example.com/one.jpg", False, False)
    assert rows["B000000002"] == (None, True, False)
    assert rows["B000000003"] == ("https://example.com/three.jpg", False, False)
    assert all(entry[2] is False for entry in rows.values())


def test_verify_requires_the_image_columns(workspace: dict[str, Path]) -> None:
    """A cache missing the image columns fails --verify (create_all never migrates)."""
    module = _load_module()
    assert module.main(["--config", str(workspace["config"])]) == 0
    engine = create_engine_from_settings(db_url=f"sqlite:///{workspace['db']}")
    try:
        with engine.begin() as connection:
            try:
                connection.exec_driver_sql("ALTER TABLE products DROP COLUMN image_cached")
            except Exception as exc:  # pragma: no cover - old SQLite without DROP COLUMN
                pytest.skip(f"SQLite cannot DROP COLUMN here: {exc}")
    finally:
        engine.dispose()
    assert module.main(["--config", str(workspace["config"]), "--verify"]) == 1


def test_missing_image_column_aborts_the_load(workspace: dict[str, Path]) -> None:
    """A parquet without the image columns aborts instead of ingesting wrong data."""
    module = _load_module()
    without_images = _products_frame().drop(columns=["image_url", "image_missing"])
    without_images.to_parquet(workspace["processed"] / "products.parquet", index=False)
    assert module.main(["--config", str(workspace["config"])]) == 1


def test_convert_record_adds_extras() -> None:
    """The cache-only ``image_cached`` default is applied to every product row."""
    module = _load_module()
    row = module._convert_record({"asin": "B000000001"}, {}, {"image_cached": False})
    assert row == {"asin": "B000000001", "image_cached": False}
    # A parquet column with the same name would win (setdefault semantics).
    row = module._convert_record(
        {"asin": "B000000001", "image_cached": True}, {}, {"image_cached": False}
    )
    assert row["image_cached"] is True


def test_ingest_is_idempotent(workspace: dict[str, Path]) -> None:
    """A second run without --force skips loading and keeps identical counts."""
    module = _load_module()
    assert module.main(["--config", str(workspace["config"])]) == 0
    first = _counts(workspace["db"])
    assert module.main(["--config", str(workspace["config"])]) == 0
    second = _counts(workspace["db"])
    assert first == second


def test_force_rebuilds_and_stays_identical(workspace: dict[str, Path]) -> None:
    """--force drops and rebuilds; final counts match a normal load exactly."""
    module = _load_module()
    assert module.main(["--config", str(workspace["config"])]) == 0
    normal = _counts(workspace["db"])
    assert module.main(["--config", str(workspace["config"]), "--force"]) == 0
    forced = _counts(workspace["db"])
    assert forced == normal


def test_verify_only_passes_then_fails_after_tampering(
    workspace: dict[str, Path],
) -> None:
    """--verify alone loads nothing; after tampering it reports failure."""
    module = _load_module()
    # Verify against an empty (created-on-demand) DB fails on count mismatch.
    assert module.main(["--config", str(workspace["config"]), "--verify"]) == 1
    assert module.main(["--config", str(workspace["config"])]) == 0
    assert module.main(["--config", str(workspace["config"]), "--verify"]) == 0
    engine = create_engine_from_settings(db_url=f"sqlite:///{workspace['db']}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("DELETE FROM reviews WHERE review_id='0000000000000004'")
    finally:
        engine.dispose()
    assert module.main(["--config", str(workspace["config"]), "--verify"]) == 1


def test_missing_parquet_exits_nonzero(tmp_path: Path) -> None:
    """An empty processed directory fails with a clear error, exit code 1."""
    module = _load_module()
    processed = tmp_path / "empty_processed"
    processed.mkdir()
    config = _write_config(tmp_path / "config.yaml", processed, tmp_path / "db.sqlite")
    assert module.main(["--config", str(config)]) == 1
    assert module.main(["--config", str(config), "--verify"]) == 1


def test_schema_drift_exits_nonzero(workspace: dict[str, Path]) -> None:
    """An unexpected parquet column aborts the load instead of guessing."""
    module = _load_module()
    drifted = _products_frame()
    drifted["future_column"] = ["x", "y", "z"]
    drifted.to_parquet(workspace["processed"] / "products.parquet", index=False)
    assert module.main(["--config", str(workspace["config"])]) == 1


def test_stale_split_distribution_triggers_reload(workspace: dict[str, Path]) -> None:
    """Same counts but a different split distribution forces a reload.

    This is the "cache built with another split seed" hazard: plain row counts
    would match, so the split distribution is compared as well.
    """
    module = _load_module()
    assert module.main(["--config", str(workspace["config"])]) == 0
    engine = create_engine_from_settings(db_url=f"sqlite:///{workspace['db']}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("UPDATE products SET split='test' WHERE asin='B000000001'")
    finally:
        engine.dispose()
    # Re-run: distribution mismatch -> clear + reload -> parquet truth restored.
    assert module.main(["--config", str(workspace["config"])]) == 0
    engine = create_engine_from_settings(db_url=f"sqlite:///{workspace['db']}")
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text("SELECT split, COUNT(*) FROM products GROUP BY split ORDER BY split")
            ).fetchall()
            distribution = {str(row[0]): int(row[1]) for row in rows}
    finally:
        engine.dispose()
    assert distribution == {"train": 2, "val": 1}


# ------------------------------------------------------------------------------
# Value conversion helpers
# ------------------------------------------------------------------------------


def test_to_scalar_null_markers() -> None:
    """Every pandas null marker becomes None (SQL NULL)."""
    module = _load_module()
    assert module._to_scalar(None) is None
    assert module._to_scalar(pd.NA) is None
    assert module._to_scalar(pd.NaT) is None
    assert module._to_scalar(float("nan")) is None
    assert module._to_scalar(np.float64("nan")) is None
    assert module._to_scalar(np.nan) is None


def test_to_scalar_numpy_scalars_become_native() -> None:
    """numpy scalars are converted to driver-safe Python types."""
    module = _load_module()
    assert module._to_scalar(np.bool_(True)) is True
    assert module._to_scalar(np.int64(7)) == 7
    assert module._to_scalar(np.float64(2.5)) == 2.5


def test_to_scalar_drops_utc_offset_only() -> None:
    """Timestamps are converted to UTC wall time, not shifted arbitrarily."""
    module = _load_module()
    aware = pd.Timestamp("2018-10-04 12:00", tz="Asia/Karachi")  # UTC+05:00
    stored = module._to_scalar(aware)
    assert stored == pd.Timestamp("2018-10-04 07:00").to_pydatetime()
    assert stored.tzinfo is None
    naive = pd.Timestamp("2018-10-04 12:00")
    assert module._to_scalar(naive) == naive.to_pydatetime()


def test_to_features_json_list_and_nulls() -> None:
    """Feature lists pass through; missing markers become None."""
    module = _load_module()
    assert module._to_features_json(["steel", "compact"]) == ["steel", "compact"]
    assert module._to_features_json(None) is None
    assert module._to_features_json(pd.NA) is None
    assert module._to_features_json(float("nan")) is None
    with pytest.raises(DataError, match="not a list"):
        module._to_features_json("not a list")

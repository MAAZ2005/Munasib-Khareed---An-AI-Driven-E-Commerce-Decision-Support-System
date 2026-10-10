"""Tests for the serving-cache schema and session layer.

Covers table creation, foreign-key enforcement, unique constraints, the
per-connection PRAGMAs (``foreign_keys=ON``, busy timeout, WAL) and the
``session_scope`` commit/rollback contract - all against in-memory SQLite with
small ORM-built fixtures. Hermetic: no network, no GPU, no real parquet files.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import insert, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from munasib.core.constants import SPLIT_TRAIN
from munasib.db.models import (
    AnalysisResult,
    Product,
    RelatedProduct,
    Review,
    ReviewFeature,
)
from munasib.db.session import (
    BUSY_TIMEOUT_MS,
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)

ALL_TABLES = {
    "products",
    "reviews",
    "related_products",
    "review_features",
    "product_features",
    "analysis_results",
}

#: Minimal valid product payload for raw (non-ORM) inserts.
PRODUCT_ROW = {
    "asin": "B000000001",
    "split": SPLIT_TRAIN,
    "title": "Widget",
    "title_clean": "widget",
    "brand": "Acme",
    "brand_clean": "acme",
    "brand_missing": False,
    "price": 9.99,
    "price_max": None,
    "price_status": "ok",
    "price_missing": False,
    "rank": None,
    "rank_value": None,
    "rank_category": None,
    "rank_missing": True,
    "description": None,
    "features": None,
    "n_reviews": 1,
    "mean_rating": 4.0,
    "first_review_time": datetime(2018, 1, 1),
    "last_review_time": datetime(2018, 1, 1),
}

#: Minimal valid review payload (``verified`` intentionally NULL = unknown).
REVIEW_ROW = {
    "review_id": "0123456789abcdef",
    "asin": "B000000001",
    "split": SPLIT_TRAIN,
    "reviewer_id": "u1",
    "rating": 4.0,
    "review_text_raw": "fine",
    "review_text_clean": "fine",
    "summary_raw": "fine",
    "summary_clean": "fine",
    "has_text": True,
    "text_length": 4,
    "helpful_votes": 0,
    "verified": None,
    "review_time": datetime(2018, 1, 1),
    "year": 2018,
    "month": 1,
    "sentiment_label_weak": 2,
}


@pytest.fixture()
def engine() -> Iterator[Engine]:
    """In-memory engine with every table created.

    Yields:
        The engine.
    """
    engine = create_engine_from_settings(db_url="sqlite://")
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def session(engine: Engine) -> Iterator[Session]:
    """Open session over the seeded engine (rolled back and closed after)."""
    factory = create_session_factory(engine)
    session = factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# ------------------------------------------------------------------------------
# Table creation
# ------------------------------------------------------------------------------


def test_init_db_creates_all_six_tables(engine) -> None:
    """create_all builds exactly the six declared tables."""
    with engine.connect() as connection:
        names = set(
            connection.scalars(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            )
        )
    assert names == ALL_TABLES


def test_pipeline_owned_tables_start_empty(engine) -> None:
    """The three cache tables load later; feature/analysis tables start empty."""
    with engine.connect() as connection:
        for name in ("review_features", "product_features", "analysis_results"):
            total = connection.scalar(text(f"SELECT COUNT(*) FROM {name}"))
            assert total == 0


def test_init_db_is_idempotent(engine) -> None:
    """Calling create_all twice does not fail or drop data."""
    with session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), [PRODUCT_ROW])
    init_db(engine)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT COUNT(*) FROM products")) == 1


# ------------------------------------------------------------------------------
# Foreign keys and unique constraints
# ------------------------------------------------------------------------------


def test_foreign_key_rejects_unknown_asin(engine) -> None:
    """A review row referencing a missing product violates the FK."""
    bad_review = {**REVIEW_ROW, "review_id": "ffffffffffffffff", "asin": "NOPE000000"}
    with pytest.raises(IntegrityError), session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Review), [bad_review])


def test_foreign_key_allows_existing_asin(engine) -> None:
    """A review row referencing a stored product inserts cleanly."""
    with session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), [PRODUCT_ROW])
        session.execute(insert(Review), [REVIEW_ROW])
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT COUNT(*) FROM reviews")) == 1


def test_related_asin_may_point_outside_subset(engine) -> None:
    """related_asin is NOT a foreign key: unknown targets are allowed."""
    with session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), [PRODUCT_ROW])
        session.execute(
            insert(RelatedProduct),
            [
                {
                    "asin": "B000000001",
                    "related_asin": "OUTSIDE999",
                    "relation_type": "also_viewed",
                    "in_subset": False,
                }
            ],
        )
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT COUNT(*) FROM related_products")) == 1


def test_products_primary_key_rejects_duplicate_asin(engine) -> None:
    """Inserting the same ASIN twice violates the products primary key."""
    with pytest.raises(IntegrityError), session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), [PRODUCT_ROW, PRODUCT_ROW])


def test_reviews_unique_review_id(engine) -> None:
    """review_id is unique - the stable key for review_features."""
    second = {**REVIEW_ROW, "reviewer_id": "u2"}
    with pytest.raises(IntegrityError), session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), [PRODUCT_ROW])
        session.execute(insert(Review), [REVIEW_ROW, second])


def test_related_unique_triple(engine) -> None:
    """(asin, related_asin, relation_type) is unique - same edge twice fails."""
    edge = {
        "asin": "B000000001",
        "related_asin": "B000000002",
        "relation_type": "also_bought",
        "in_subset": False,
    }
    with pytest.raises(IntegrityError), session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), [PRODUCT_ROW])
        session.execute(insert(RelatedProduct), [edge, dict(edge)])


def test_related_same_pair_different_type_allowed(engine) -> None:
    """The same pair with a different relation_type is a distinct edge."""
    edges = [
        {
            "asin": "B000000001",
            "related_asin": "B000000002",
            "relation_type": kind,
            "in_subset": False,
        }
        for kind in ("also_bought", "also_viewed")
    ]
    with session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), [PRODUCT_ROW])
        session.execute(insert(RelatedProduct), edges)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT COUNT(*) FROM related_products")) == 2


def test_related_relation_type_check_constraint(engine) -> None:
    """relation_type is constrained to the two UCSD relation names."""
    edge = {
        "asin": "B000000001",
        "related_asin": "B000000002",
        "relation_type": "also_viewed_together",
        "in_subset": False,
    }
    with pytest.raises(IntegrityError), session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), [PRODUCT_ROW])
        session.execute(insert(RelatedProduct), [edge])


def test_verified_null_versus_false_roundtrip(engine) -> None:
    """NULL (unknown) and False are stored as distinct values."""
    rows = [
        {**REVIEW_ROW, "review_id": "aaaa000000000001", "verified": None},
        {**REVIEW_ROW, "review_id": "aaaa000000000002", "verified": False},
        {**REVIEW_ROW, "review_id": "aaaa000000000003", "verified": True},
    ]
    with session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), [PRODUCT_ROW])
        session.execute(insert(Review), rows)
    with engine.connect() as connection:
        stored = {
            row[0]: row[1]
            for row in connection.execute(
                text("SELECT review_id, verified FROM reviews ORDER BY review_id")
            )
        }
    assert stored["aaaa000000000001"] is None
    assert stored["aaaa000000000002"] == 0  # False stored as 0, not NULL
    assert stored["aaaa000000000003"] == 1


# ------------------------------------------------------------------------------
# PRAGMAs and session_scope
# ------------------------------------------------------------------------------


def test_pragmas_applied_on_connect(engine) -> None:
    """Every connection gets foreign_keys=ON and the busy timeout."""
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert connection.exec_driver_sql("PRAGMA busy_timeout").scalar() == BUSY_TIMEOUT_MS


def test_wal_journal_mode_for_file_database(tmp_path: Path) -> None:
    """File databases open in WAL mode (in-memory databases report 'memory')."""
    db_path = tmp_path / "cache.sqlite"
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    try:
        init_db(engine)
        with engine.connect() as connection:
            mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar()
        assert str(mode).lower() == "wal"
    finally:
        engine.dispose()


def test_memory_database_reports_memory_journal_mode(engine) -> None:
    """In-memory engines skip WAL without error (journal_mode reports memory)."""
    with engine.connect() as connection:
        mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar()
    assert str(mode).lower() == "memory"


def test_session_scope_commits_on_success(engine) -> None:
    """Data written inside the scope is visible after the scope exits."""
    with session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), [PRODUCT_ROW])
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT COUNT(*) FROM products")) == 1


def test_session_scope_rolls_back_on_error(engine) -> None:
    """An exception inside the scope discards every pending change."""

    def boom(session: Session) -> None:
        session.execute(insert(Product), [PRODUCT_ROW])
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"), session_scope(
        create_session_factory(engine)
    ) as session:
        boom(session)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT COUNT(*) FROM products")) == 0


def test_analysis_results_composite_key() -> None:
    """analysis_results is keyed by (asin, pipeline_version)."""
    mapped = {column.name for column in AnalysisResult.__table__.primary_key.columns}
    assert mapped == {"asin", "pipeline_version"}


def test_review_features_keyed_by_review_id() -> None:
    """review_features is keyed by review_id, which references reviews."""
    table = ReviewFeature.__table__
    assert [column.name for column in table.primary_key.columns] == ["review_id"]
    assert table.c.review_id.foreign_keys, "review_id must be a foreign key"


def test_indexes_exist_for_api_queries(engine) -> None:
    """The indexes backing the API's queries were created."""
    with engine.connect() as connection:
        names = set(
            connection.scalars(
                text("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
            )
        )
    assert {
        "ix_products_brand",
        "ix_products_title",
        "ix_reviews_asin",
        "ix_reviews_asin_review_time",
        "ix_reviews_split",
        "ix_related_products_asin",
        "ix_related_products_asin_in_subset",
    } <= names


def test_products_datetime_roundtrip_is_utc_naive(engine) -> None:
    """Stored datetimes come back as naive UTC values (offsets are dropped)."""
    aware = datetime(2018, 10, 4, 12, 30, tzinfo=UTC)
    row = {**PRODUCT_ROW, "first_review_time": aware}
    with session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), [row])
    with session_scope(create_session_factory(engine)) as session:
        stored = session.scalars(select(Product)).one()
        assert stored.first_review_time == datetime(2018, 10, 4, 12, 30)
        assert stored.first_review_time.tzinfo is None

"""Tests for the ``product_evidence`` model, repository and store layer.

Covers the repository upsert contract (idempotent merge on ``asin``), the
point/filter/count queries, the store loaders over a seeded cache (including
the LEFT JOIN that leaves predictions NULL), the reference-date reader (naive
UTC contract, every failure mode named) and the timezone guard on writes.
Hermetic: an in-memory SQLite database, no network, no real ``data/`` files.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import insert, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from munasib.core.config import Settings
from munasib.core.constants import SENTIMENT_LABEL_POSITIVE
from munasib.core.exceptions import DataError
from munasib.db.models import Product, ProductAspect, ProductEvidence, Review, ReviewFeature
from munasib.db.repositories import ProductEvidenceRepository
from munasib.db.session import create_engine_from_settings, create_session_factory, init_db
from munasib.review_intel.evidence.compute import EvidenceRow
from munasib.review_intel.evidence.store import (
    count_evidence,
    load_aspects,
    load_products,
    load_reviews,
    read_evidence,
    read_reference_date,
    write_evidence,
)

BUILD_AT = datetime(2026, 10, 24, 6, 15, 6)
REFERENCE = datetime(2018, 10, 4)

PRODUCTS = ("B0000000EA", "B0000000EB")


def _product_row(asin: str) -> dict:
    """Build one products-table insert payload.

    Args:
        asin: Product ASIN.

    Returns:
        Insert payload with every NOT NULL column filled.
    """
    return {
        "asin": asin,
        "split": "train",
        "title": f"Product {asin}",
        "title_clean": f"product {asin}",
        "brand": "Acme",
        "brand_clean": "acme",
        "brand_missing": False,
        "price": 10.0,
        "price_max": None,
        "price_status": "single",
        "price_missing": False,
        "rank": None,
        "rank_value": None,
        "rank_category": None,
        "rank_missing": True,
        "description": "A perfectly ordinary description.",
        "features": ["bullet one"],
        "n_reviews": 0,
        "mean_rating": None,
        "first_review_time": None,
        "last_review_time": None,
        "image_url": None,
        "image_missing": True,
        "image_cached": False,
    }


def _review_row(asin: str, review_id: str, when: datetime, rating: float) -> dict:
    """Build one reviews-table insert payload.

    Args:
        asin: Owning product ASIN.
        review_id: Stable 16-hex review key.
        when: Naive UTC review time.
        rating: Star rating.

    Returns:
        Insert payload.
    """
    return {
        "review_id": review_id,
        "asin": asin,
        "split": "train",
        "reviewer_id": f"user-{review_id[-2:]}",
        "rating": rating,
        "review_text_raw": "text",
        "review_text_clean": "text",
        "summary_raw": "summary",
        "summary_clean": "summary",
        "has_text": True,
        "text_length": 4,
        "helpful_votes": 1,
        "verified": True,
        "review_time": when,
        "year": when.year,
        "month": when.month,
        "sentiment_label_weak": 0 if rating <= 2 else 2,
    }


def _prediction(review_id: str, label: int) -> dict:
    """Build one review_features insert payload.

    Args:
        review_id: The review the prediction belongs to.
        label: Predicted label code.

    Returns:
        Insert payload with the matching probability triplet.
    """
    probabilities = {
        0: (0.90, 0.05, 0.05),
        1: (0.05, 0.90, 0.05),
        2: (0.05, 0.05, 0.90),
    }[label]
    return {
        "review_id": review_id,
        "predicted_label": label,
        "prob_negative": probabilities[0],
        "prob_neutral": probabilities[1],
        "prob_positive": probabilities[2],
    }


def _aspect_row(asin: str, classification: str, negative_count: int) -> dict:
    """Build one product_aspects insert payload.

    Args:
        asin: Product ASIN.
        classification: Stored classification.
        negative_count: Negative mention count.

    Returns:
        Insert payload with consistent shares.
    """
    mention = max(negative_count, 1)
    return {
        "asin": asin,
        "aspect": "battery",
        "mention_count": mention,
        "negative_count": negative_count,
        "neutral_count": 0,
        "positive_count": mention - negative_count,
        "negative_share": negative_count / mention,
        "neutral_share": 0.0,
        "positive_share": (mention - negative_count) / mention,
        "net_sentiment": (mention - 2 * negative_count) / mention,
        "product_labeled_reviews": 1,
        "product_negative_share": 0.0 if classification != "pain_point" else 1.0,
        "negativity_lift": None,
        "classification": classification,
        "examples": None,
        "computed_at": BUILD_AT,
    }


@pytest.fixture()
def engine() -> Iterator[Engine]:
    """Seeded in-memory engine with products, reviews, predictions and aspects.

    Yields:
        Engine holding two products; the first has one predicted review plus
        one unpredicted review, the second has one predicted negative review.
    """
    engine = create_engine_from_settings(db_url="sqlite://")
    init_db(engine)
    with create_session_factory(engine)() as session:
        session.execute(insert(Product), [_product_row(asin) for asin in PRODUCTS])
        session.execute(
            insert(Review),
            [
                _review_row("B0000000EA", "000000000000ea01", datetime(2018, 6, 1), 5.0),
                _review_row("B0000000EA", "000000000000ea02", datetime(2018, 6, 2), 4.0),
                _review_row("B0000000EB", "000000000000eb01", datetime(2005, 3, 1), 1.0),
            ],
        )
        session.execute(
            insert(ReviewFeature),
            [
                _prediction("000000000000ea01", SENTIMENT_LABEL_POSITIVE),
                _prediction("000000000000eb01", 0),
            ],
        )
        session.execute(
            insert(ProductAspect),
            [
                _aspect_row("B0000000EA", "neutral", 0),
                _aspect_row("B0000000EB", "pain_point", 1),
            ],
        )
        session.commit()
    yield engine
    engine.dispose()


@pytest.fixture()
def session(engine: Engine) -> Iterator[Session]:
    """Open a session over the seeded engine.

    Args:
        engine: Seeded engine.

    Yields:
        Open session.
    """
    with create_session_factory(engine)() as open_session:
        yield open_session


def _evidence_row(asin: str, strength: float | None = 0.5, n_reviews: int = 2) -> EvidenceRow:
    """Build a minimal evidence row for persistence tests.

    Args:
        asin: Product ASIN.
        strength: Composite value (None means no component was computable).
        n_reviews: Review count stored alongside the payload.

    Returns:
        The row with a minimal (but complete) indicator payload.
    """
    return EvidenceRow(
        asin=asin,
        split="train",
        n_reviews=n_reviews,
        evidence_strength=strength,
        indicators={
            "split": "train",
            "indicators": {},
            "flags": {},
            "polarized": False,
            "no_recent_reviews": False,
            "temporal_burst": False,
            "low_evidence": True,
            "strength": {},
        },
        reference_date=REFERENCE,
        computed_at=BUILD_AT,
    )


# ------------------------------------------------------------------------------
# Repository
# ------------------------------------------------------------------------------


def test_upsert_evidence_stores_and_updates_one_row_per_asin(session: Session) -> None:
    """Upserting twice merges on the primary key instead of duplicating."""
    repository = ProductEvidenceRepository(session)
    written = repository.upsert_evidence([_evidence_row("B0000000EA").to_payload()])
    assert written == 1
    # A second run refreshes the same row with new values.
    updated = _evidence_row("B0000000EA", strength=0.7, n_reviews=9)
    assert repository.upsert_evidence([updated.to_payload()]) == 1
    stored = repository.get_evidence("B0000000EA")
    assert stored is not None
    assert repository.count_rows() == 1
    assert stored.n_reviews == 9
    assert stored.evidence_strength == pytest.approx(0.7)
    assert stored.reference_date == REFERENCE
    assert stored.computed_at == BUILD_AT
    assert stored.indicators["low_evidence"] is True


def test_upsert_evidence_stores_a_null_strength(session: Session) -> None:
    """A product with no computable component stores NULL strength."""
    repository = ProductEvidenceRepository(session)
    repository.upsert_evidence([_evidence_row("B0000000EB", strength=None).to_payload()])
    stored = repository.get_evidence("B0000000EB")
    assert stored is not None
    assert stored.evidence_strength is None


def test_list_evidence_orders_by_asin_and_filters(session: Session) -> None:
    """Listing is ascending by ASIN and honours an explicit ASIN filter."""
    repository = ProductEvidenceRepository(session)
    repository.upsert_evidence(
        [
            _evidence_row("B0000000EB").to_payload(),
            _evidence_row("B0000000EA").to_payload(),
        ]
    )
    assert [row.asin for row in repository.list_evidence()] == list(PRODUCTS)
    assert [row.asin for row in repository.list_evidence(["B0000000EB"])] == ["B0000000EB"]
    assert repository.list_evidence(["B00000000000"]) == []


def test_get_evidence_returns_none_for_an_unscored_product(session: Session) -> None:
    """A product never scored has no evidence row."""
    repository = ProductEvidenceRepository(session)
    repository.upsert_evidence([_evidence_row("B0000000EA").to_payload()])
    assert repository.get_evidence("B0000000EB") is None


def test_upsert_evidence_rejects_an_incomplete_row(session: Session) -> None:
    """A payload missing a required column is a caller bug."""
    repository = ProductEvidenceRepository(session)
    payload = _evidence_row("B0000000EA").to_payload()
    del payload["n_reviews"]
    with pytest.raises(ValueError, match="n_reviews"):
        repository.upsert_evidence([payload])


def test_null_strength_is_stored_as_null(session: Session) -> None:
    """Reading back a NULL strength yields None, not 0.0."""
    repository = ProductEvidenceRepository(session)
    repository.upsert_evidence([_evidence_row("B0000000EA", strength=None).to_payload()])
    stored = session.scalars(select(ProductEvidence.evidence_strength)).one()
    assert stored is None


# ------------------------------------------------------------------------------
# Store layer
# ------------------------------------------------------------------------------


def test_write_evidence_rejects_timezone_aware_timestamps(session: Session) -> None:
    """The storage contract is naive UTC; aware timestamps fail loudly."""
    row = EvidenceRow(
        asin="B0000000EA",
        split="train",
        n_reviews=1,
        evidence_strength=0.5,
        indicators={"indicators": {}},
        reference_date=datetime(2018, 10, 4, tzinfo=UTC),
        computed_at=BUILD_AT,
    )
    with pytest.raises(ValueError, match="naive UTC"):
        write_evidence(session, [row])


def test_write_evidence_round_trips_through_the_repository(session: Session) -> None:
    """Rows written through the store are readable through the repository."""
    rows = [_evidence_row("B0000000EA"), _evidence_row("B0000000EB", strength=None)]
    assert write_evidence(session, rows) == 2
    assert count_evidence(session) == 2
    stored = read_evidence(session, "B0000000EA")
    assert stored is not None and stored.evidence_strength == pytest.approx(0.5)


def test_load_products_returns_the_asin_split_list(engine: Engine) -> None:
    """The product loader returns both columns in ascending ASIN order."""
    frame = load_products(engine)
    assert list(frame.columns) == ["asin", "split"]
    assert frame["asin"].tolist() == list(PRODUCTS)


def test_load_reviews_left_joins_the_predictions(engine: Engine) -> None:
    """Reviews without a feature row keep a NULL prediction, never a zero."""
    frame = load_reviews(engine)
    assert len(frame) == 3
    missing = frame[frame["prob_positive"].isna()]
    assert missing["asin"].tolist() == ["B0000000EA"]
    assert missing["predicted_label"].isna().all()
    assert frame["verified"].notna().all()


def test_load_aspects_returns_the_counts(engine: Engine) -> None:
    """The aspect loader carries classification and negative counts."""
    frame = load_aspects(engine)
    assert list(frame.columns) == [
        "asin",
        "aspect",
        "classification",
        "negative_count",
    ]
    assert frame["negative_count"].tolist() == [0, 1]


# ------------------------------------------------------------------------------
# read_reference_date
# ------------------------------------------------------------------------------


def _settings(processed: Path) -> Settings:
    """Build settings pointing at a processed dir.

    Args:
        processed: Directory holding (or missing) ``meta.json``.

    Returns:
        The settings.
    """
    return Settings.model_validate(
        {"paths": {"processed_data_dir": str(processed), "db_path": "unused.sqlite"}}
    )


def test_read_reference_date_normalises_to_naive_utc(tmp_path: Path) -> None:
    """An aware dataset timestamp becomes naive UTC (the storage contract)."""
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "meta.json").write_text(
        json.dumps({"review_date_range": {"max": "2018-10-04T00:00:00+00:00"}}),
        encoding="utf-8",
    )
    assert read_reference_date(_settings(processed)) == REFERENCE


def test_read_reference_date_accepts_a_naive_source(tmp_path: Path) -> None:
    """A naive dataset timestamp passes through unchanged."""
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "meta.json").write_text(
        json.dumps({"review_date_range": {"max": "2018-10-04T05:06:07"}}),
        encoding="utf-8",
    )
    assert read_reference_date(_settings(processed)) == datetime(2018, 10, 4, 5, 6, 7)


def test_read_reference_date_requires_the_metadata_file(tmp_path: Path) -> None:
    """A missing meta.json fails loudly; there is no wall-clock fallback."""
    with pytest.raises(DataError, match="not found"):
        read_reference_date(_settings(tmp_path))


def test_read_reference_date_rejects_unreadable_json(tmp_path: Path) -> None:
    """Broken JSON is a data error, not a crash."""
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DataError, match="could not read"):
        read_reference_date(_settings(processed))


def test_read_reference_date_requires_the_max_field(tmp_path: Path) -> None:
    """Metadata without ``review_date_range.max`` cannot anchor recency."""
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "meta.json").write_text(
        json.dumps({"review_date_range": {"min": "2001-11-08T00:00:00+00:00"}}),
        encoding="utf-8",
    )
    with pytest.raises(DataError, match="review_date_range.max"):
        read_reference_date(_settings(processed))


def test_read_reference_date_rejects_an_unparsable_max(tmp_path: Path) -> None:
    """An unparsable timestamp is a data error, not a silent fallback."""
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "meta.json").write_text(
        json.dumps({"review_date_range": {"max": "not-a-date"}}),
        encoding="utf-8",
    )
    with pytest.raises(DataError, match="unparsable"):
        read_reference_date(_settings(processed))

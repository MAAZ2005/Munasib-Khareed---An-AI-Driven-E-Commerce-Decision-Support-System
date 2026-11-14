"""Tests for the resumable candidate cache and aspect storage helpers."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import insert
from sqlalchemy.engine import Engine

from munasib.core.config import AspectsConfig
from munasib.db.models import Product, Review, ReviewFeature
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)
from munasib.review_intel.aspects.cache import (
    AspectCacheIncomplete,
    AspectCacheMismatch,
    CandidateCache,
    extraction_fingerprint,
    iter_part_records,
    source_signature,
)
from munasib.review_intel.aspects.storage import (
    ReviewAspectWriter,
    count_text_reviews,
    iter_prediction_rows,
    iter_text_reviews,
    read_review_aspects,
    write_review_aspects,
)


def product_row(asin: str = "B000000001") -> dict:
    """Build a minimal product insert payload.

    Args:
        asin: Product ASIN.

    Returns:
        Insert payload with every NOT NULL column.
    """
    return {
        "asin": asin,
        "split": "train",
        "brand_missing": True,
        "price_missing": True,
        "rank_missing": True,
        "n_reviews": 3,
        "image_missing": True,
    }


def review_row(review_id: str, text: str | None, asin: str = "B000000001") -> dict:
    """Build a minimal review insert payload.

    Args:
        review_id: Stable review id.
        text: Clean review text, or None for a textless review.
        asin: Owning product.

    Returns:
        Insert payload.
    """
    return {
        "review_id": review_id,
        "asin": asin,
        "split": "train",
        "reviewer_id": "u1",
        "review_text_raw": text,
        "review_text_clean": text,
        "summary_raw": None,
        "summary_clean": None,
        "has_text": text is not None,
        "text_length": len(text or ""),
        "helpful_votes": 0,
        "verified": None,
        "review_time": datetime(2018, 1, 1),
        "year": 2018,
        "month": 1,
        "sentiment_label_weak": 2,
    }


@pytest.fixture()
def engine() -> Iterator[Engine]:
    """In-memory cache with one product and three reviews.

    Yields:
        Engine with predictions for the two text reviews.
    """
    database = create_engine_from_settings(db_url="sqlite://")
    init_db(database)
    factory = create_session_factory(database)
    with session_scope(factory) as session:
        session.execute(insert(Product), [product_row()])
        session.execute(
            insert(Review),
            [
                review_row("r1", "filter works"),
                review_row("r2", "motor broke"),
                review_row("r3", None),
            ],
        )
        session.execute(
            insert(ReviewFeature),
            [
                {
                    "review_id": "r1",
                    "predicted_label": 2,
                    "prob_negative": 0.1,
                    "prob_neutral": 0.2,
                    "prob_positive": 0.7,
                    "aspects_json": ["filter"],
                },
                {
                    "review_id": "r2",
                    "predicted_label": 0,
                    "prob_negative": 0.7,
                    "prob_neutral": 0.2,
                    "prob_positive": 0.1,
                    "aspects_json": ["motor"],
                },
                {
                    "review_id": "r3",
                    "aspects_json": None,
                },
            ],
        )
    yield database
    database.dispose()


def test_source_signature_and_keyset_pagination(engine: Engine) -> None:
    """The source signature counts text only; keyset pages never overlap."""
    factory = create_session_factory(engine)
    session = factory()
    try:
        signature = source_signature(session)
        assert signature["text_reviews"] == 2
        assert signature["text_chars"] == len("filter works") + len("motor broke")
        assert count_text_reviews(session) == 2
        first = iter_text_reviews(session, limit=1)
        second = iter_text_reviews(session, after_review_id=first[-1].review_id, limit=10)
        assert [row.review_id for row in first + second] == ["r1", "r2"]
    finally:
        session.close()


def test_prediction_stream_excludes_unpredicted_and_keeps_textless(
    engine: Engine,
) -> None:
    """Textless reviews may have no prediction; only labelled rows are streamed."""
    factory = create_session_factory(engine)
    session = factory()
    try:
        rows = list(iter_prediction_rows(session))
    finally:
        session.close()
    assert [row.review_id for row in rows] == ["r1", "r2"]
    assert rows[0].aspects == ["filter"]
    assert rows[1].predicted_label == 0


def test_candidate_cache_roundtrip_and_resume(tmp_path: Path) -> None:
    """Complete parts are reused; missing parts require --resume."""
    cache = CandidateCache(tmp_path / "parts", chunk_rows=2)
    fingerprint = extraction_fingerprint(
        config=AspectsConfig(),
        source={"text_reviews": 3},
    )
    status = cache.prepare(
        fingerprint=fingerprint,
        fingerprint_payload={"source": {"text_reviews": 3}},
        total_reviews=3,
        resume=False,
    )
    assert status.missing_parts == [0, 1]
    cache.write_part(
        0,
        [
            {"review_id": "r1", "asin": "A", "aspects": ["filter"]},
            {"review_id": "r2", "asin": "A", "aspects": []},
        ],
    )
    with pytest.raises(AspectCacheIncomplete, match="--resume"):
        cache.prepare(
            fingerprint=fingerprint,
            fingerprint_payload={"source": {"text_reviews": 3}},
            total_reviews=3,
            resume=False,
        )
    resumed = cache.prepare(
        fingerprint=fingerprint,
        fingerprint_payload={"source": {"text_reviews": 3}},
        total_reviews=3,
        resume=True,
    )
    assert resumed.completed_parts == [0]
    assert resumed.missing_parts == [1]
    cache.write_part(
        1,
        [{"review_id": "r3", "asin": "B", "aspects": ["motor"]}],
    )
    complete = cache.prepare(
        fingerprint=fingerprint,
        fingerprint_payload={"source": {"text_reviews": 3}},
        total_reviews=3,
        resume=False,
    )
    assert complete.complete
    assert cache.last_review_id(1) == "r3"
    records = [record for frame in cache.iter_parts() for record in iter_part_records(frame)]
    assert records == [
        ("r1", "A", ["filter"]),
        ("r2", "A", []),
        ("r3", "B", ["motor"]),
    ]


def test_candidate_cache_fingerprint_mismatch_is_refused(tmp_path: Path) -> None:
    """Parts from another source/settings cannot be mixed silently."""
    cache = CandidateCache(tmp_path / "parts", chunk_rows=2)
    cache.prepare(
        fingerprint="one",
        fingerprint_payload={},
        total_reviews=2,
        resume=False,
    )
    with pytest.raises(AspectCacheMismatch, match="does not match"):
        cache.prepare(
            fingerprint="two",
            fingerprint_payload={},
            total_reviews=2,
            resume=True,
        )


def test_candidate_cache_rejects_parts_without_a_manifest(tmp_path: Path) -> None:
    """A bare parts directory is ambiguous and must be removed."""
    parts = tmp_path / "parts"
    parts.mkdir()
    pd.DataFrame(
        [{"review_id": "r1", "asin": "A", "aspects": ["filter"]}],
        columns=["review_id", "asin", "aspects"],
    ).to_parquet(parts / "part-00000.parquet", index=False)
    with pytest.raises(AspectCacheMismatch, match="no manifest"):
        CandidateCache(parts, chunk_rows=2).prepare(
            fingerprint="x",
            fingerprint_payload={},
            total_reviews=1,
            resume=False,
        )


def test_review_aspect_writer_roundtrip(tmp_path: Path) -> None:
    """The streaming writer and the convenience writer produce the same contract."""
    path = tmp_path / "review_aspects.parquet"
    with ReviewAspectWriter(path, batch_size=1) as writer:
        writer.write("r1", ["filter"])
        writer.write("r2", [])
    assert read_review_aspects(path) == [("r1", ["filter"]), ("r2", [])]
    second = tmp_path / "second.parquet"
    assert write_review_aspects([("r3", ["motor"])], second) == 1
    assert read_review_aspects(second) == [("r3", ["motor"])]

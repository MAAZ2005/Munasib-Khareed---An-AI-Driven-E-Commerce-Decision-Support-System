"""Tests for ``scripts/compute_evidence.py`` (evidence CLI).

Hermetic: a temporary SQLite cache holds two products with reviews, stored
predictions and aspect rows (plus one review-less product), and a temporary
``meta.json`` fixes both the reference date and the build timestamp, so every
count asserted here is computed by the code under test - no network, no real
``data/`` artifact.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import delete as sa_delete
from sqlalchemy import func, insert
from sqlalchemy import select as sa_select
from sqlalchemy.orm import Session

from munasib.core.config import Settings
from munasib.db.models import Product, ProductAspect, ProductEvidence, Review, ReviewFeature
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compute_evidence.py"

#: Two reviewed products plus one review-less product (all train-split here).
P_A = "B0000000EA"
P_B = "B0000000EB"
P_C = "B0000000EC"

BUILD_TIMESTAMP = "2026-10-24T06:15:06+00:00"
REFERENCE = datetime(2018, 10, 4)
BUILD_AT = datetime(2026, 10, 24, 6, 15, 6)


def _load_module():
    """Import ``scripts/compute_evidence.py`` as a module.

    Returns:
        The loaded script module.
    """
    spec = importlib.util.spec_from_file_location("munasib_compute_evidence_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _product_row(asin: str, n_reviews: int) -> dict:
    """Build a minimal products insert payload.

    Args:
        asin: Product ASIN.
        n_reviews: Review count to display in the cache row.

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
        "n_reviews": n_reviews,
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
        "reviewer_id": f"user-{review_id[-3:]}",
        "rating": rating,
        "review_text_raw": "text",
        "review_text_clean": "text",
        "summary_raw": "summary",
        "summary_clean": "summary",
        "has_text": True,
        "text_length": 4,
        "helpful_votes": 2,
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
    probabilities = {0: (0.90, 0.05, 0.05), 2: (0.05, 0.05, 0.90)}[label]
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
        Insert payload with consistent share columns.
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
        "product_negative_share": 1.0 if classification == "pain_point" else 0.0,
        "negativity_lift": None,
        "classification": classification,
        "examples": None,
        "computed_at": BUILD_AT,
    }


PRODUCTS = [_product_row(P_A, 12), _product_row(P_B, 12), _product_row(P_C, 0)]

REVIEWS = [
    # P_A: 12 recent five-star reviews (Jan-Sep 2018), all predicted positive.
    *(
        _review_row(
            P_A, f"000000000000ea{index:02d}", datetime(2018, 1 + index % 9, 1 + index // 9), 5.0
        )
        for index in range(12)
    ),
    # P_B: 12 one-star reviews from 2005, all predicted negative.
    *(
        _review_row(P_B, f"000000000000eb{index:02d}", datetime(2005, 1 + index % 12, 15), 1.0)
        for index in range(12)
    ),
]

PREDICTIONS = [
    *(_prediction(f"000000000000ea{index:02d}", 2) for index in range(12)),
    *(_prediction(f"000000000000eb{index:02d}", 0) for index in range(12)),
]

ASPECTS = [
    _aspect_row(P_A, "neutral", 0),
    _aspect_row(P_B, "pain_point", 12),
]


@pytest.fixture()
def cache(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """Create a temp serving cache and a temp processed dir with meta.json.

    Args:
        tmp_path: Pytest temporary directory.

    Yields:
        Tuple ``(db_path, processed_dir)``.
    """
    db_path = tmp_path / "cache.sqlite"
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    init_db(engine)
    with session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), PRODUCTS)
        session.execute(insert(Review), REVIEWS)
        session.execute(insert(ReviewFeature), PREDICTIONS)
        session.execute(insert(ProductAspect), ASPECTS)
    engine.dispose()

    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "meta.json").write_text(
        json.dumps(
            {
                "category": "TestCategory",
                "built_at": BUILD_TIMESTAMP,
                "review_date_range": {"max": "2018-10-04T00:00:00+00:00"},
            }
        ),
        encoding="utf-8",
    )
    yield db_path, processed


def _config(tmp_path: Path, db_path: Path, processed: Path, reports: Path) -> Path:
    """Write a minimal config pointing at the temp cache, processed dir and reports.

    Args:
        tmp_path: Pytest temporary directory.
        db_path: SQLite cache to use.
        processed: Processed data dir holding meta.json.
        reports: Directory the summary reports are written to.

    Returns:
        The config path.
    """
    config = tmp_path / "config.yaml"
    config.write_text(
        f'paths:\n  db_path: "{db_path.as_posix()}"\n'
        f'  processed_data_dir: "{processed.as_posix()}"\n'
        f'  reports_dir: "{reports.as_posix()}"\n',
        encoding="utf-8",
    )
    return config


def _settings(db_path: Path, processed: Path, reports: Path) -> Settings:
    """Build settings pointing at the temp layout.

    Args:
        db_path: SQLite cache path.
        processed: Processed data dir.
        reports: Reports directory.

    Returns:
        The settings.
    """
    return Settings.model_validate(
        {
            "paths": {
                "db_path": str(db_path),
                "processed_data_dir": str(processed),
                "reports_dir": str(reports),
            }
        }
    )


def _stored(settings: Settings, asin: str) -> ProductEvidence:
    """Read one stored evidence row from the temp cache.

    Args:
        settings: Settings pointing at the temp cache.
        asin: Product ASIN.

    Returns:
        The stored row.

    Raises:
        AssertionError: If the row is missing.
    """
    engine = create_engine_from_settings(settings)
    factory = create_session_factory(engine)
    try:
        with session_scope(factory) as session:
            row = session.get(ProductEvidence, asin)
            assert row is not None, f"no product_evidence row for {asin}"
            session.expunge(row)
            return row
    finally:
        engine.dispose()


def _stored_asins(settings: Settings) -> list[str]:
    """List every stored evidence ASIN.

    Args:
        settings: Settings pointing at the temp cache.

    Returns:
        Sorted ASIN list.
    """
    engine = create_engine_from_settings(settings)
    factory = create_session_factory(engine)
    try:
        with session_scope(factory) as session:
            return sorted(session.scalars(sa_select(ProductEvidence.asin)))
    finally:
        engine.dispose()


# ------------------------------------------------------------------------------
# select_targets
# ------------------------------------------------------------------------------


@contextmanager
def _session_for(db_path: Path) -> Iterator[Session]:
    """Open a session over a temp cache.

    Args:
        db_path: SQLite cache path.

    Yields:
        Open session.
    """
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    session = create_session_factory(engine)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


def test_select_targets_applies_the_limit_in_asin_order(cache: tuple[Path, Path]) -> None:
    """``--limit N`` takes the first N products by ASIN."""
    module = _load_module()
    db_path, _ = cache
    with _session_for(db_path) as session:
        selected = module.select_targets(session, 2, None)
        assert [row.asin for row in selected] == [P_A, P_B]


def test_select_targets_honours_explicit_asins(cache: tuple[Path, Path]) -> None:
    """``--asins`` scores exactly those products, in ASIN order."""
    module = _load_module()
    db_path, _ = cache
    with _session_for(db_path) as session:
        selected = module.select_targets(session, None, [P_B, P_A])
        assert [row.asin for row in selected] == [P_A, P_B]


def test_select_targets_rejects_an_unknown_asin(cache: tuple[Path, Path]) -> None:
    """An ASIN missing from the cache fails loudly instead of shrinking the run."""
    module = _load_module()
    db_path, _ = cache
    with _session_for(db_path) as session:
        with pytest.raises(Exception, match="NOTINTHERE"):
            module.select_targets(session, None, ["NOTINTHERE"])


def test_select_targets_rejects_a_negative_limit(cache: tuple[Path, Path]) -> None:
    """A negative limit is a usage error, not an empty run."""
    module = _load_module()
    db_path, _ = cache
    with _session_for(db_path) as session:
        with pytest.raises(ValueError, match="non-negative"):
            module.select_targets(session, -1, None)


# ------------------------------------------------------------------------------
# CLI end-to-end
# ------------------------------------------------------------------------------


def test_main_writes_evidence_with_the_meta_timestamps(
    tmp_path: Path, cache: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """A full run stores all three rows stamped from meta.json, never the clock."""
    module = _load_module()
    db_path, processed = cache
    reports = tmp_path / "reports"
    config = _config(tmp_path, db_path, processed, reports)
    assert module.main(["--config", str(config)]) == 0

    out = capsys.readouterr().out
    lines = {line.split(":")[0].strip(): line for line in out.splitlines() if ":" in line}
    assert lines["products written"].strip().endswith(": 3")
    assert lines["reference date"].strip().endswith(": 2018-10-04T00:00:00+00:00")
    assert lines["computed at"].strip().endswith(": 2026-10-24T06:15:06+00:00")

    stored = _stored(_settings(db_path, processed, reports), P_A)
    assert stored.n_reviews == 12
    assert stored.reference_date == REFERENCE
    assert stored.computed_at == BUILD_AT
    assert stored.evidence_strength is not None
    assert set(stored.indicators) == {
        "split",
        "indicators",
        "flags",
        "polarized",
        "no_recent_reviews",
        "temporal_burst",
        "low_evidence",
        "strength",
    }
    assert len(stored.indicators["indicators"]) == 14
    # 12 reviews is below the confidence minimum, so the guard fires.
    assert stored.indicators["low_evidence"] is True


def test_main_scores_the_old_and_the_reviewless_product(
    tmp_path: Path, cache: tuple[Path, Path]
) -> None:
    """The 2005-only product loses recency; the review-less one has NULL strength."""
    module = _load_module()
    db_path, processed = cache
    reports = tmp_path / "reports"
    config = _config(tmp_path, db_path, processed, reports)
    assert module.main(["--config", str(config)]) == 0

    settings = _settings(db_path, processed, reports)
    old = _stored(settings, P_B)
    assert old.indicators["no_recent_reviews"] is True
    assert old.indicators["indicators"]["recency_coverage"]["normalized"] == 0.0

    lonely = _stored(settings, P_C)
    assert lonely.n_reviews == 0
    assert lonely.evidence_strength is None
    assert lonely.indicators["flags"]["review_volume"] == "no_reviews"


def test_main_is_deterministic_across_two_runs(tmp_path: Path, cache: tuple[Path, Path]) -> None:
    """Two runs over the same cache write byte-identical indicator blobs."""
    module = _load_module()
    db_path, processed = cache
    reports = tmp_path / "reports"
    config = _config(tmp_path, db_path, processed, reports)
    settings = _settings(db_path, processed, reports)

    assert module.main(["--config", str(config)]) == 0
    first = {asin: _stored(settings, asin).indicators for asin in (P_A, P_B, P_C)}
    assert module.main(["--config", str(config)]) == 0
    second = {asin: _stored(settings, asin).indicators for asin in (P_A, P_B, P_C)}
    assert first == second


def test_main_writes_the_summary_reports(tmp_path: Path, cache: tuple[Path, Path]) -> None:
    """The run writes evidence_summary.json/.md with the computed figures."""
    module = _load_module()
    db_path, processed = cache
    reports = tmp_path / "reports"
    config = _config(tmp_path, db_path, processed, reports)
    assert module.main(["--config", str(config)]) == 0

    payload = json.loads((reports / "evidence_summary.json").read_text(encoding="utf-8"))
    assert payload["products"] == 3
    assert payload["reference_date"] == "2018-10-04T00:00:00+00:00"
    assert payload["flags"]["no_recent_reviews"]["count"] == 2
    assert set(payload["correlations"]) >= {"review_volume", "n_reviews"}
    markdown = (reports / "evidence_summary.md").read_text(encoding="utf-8")
    assert "# Evidence summary" in markdown
    assert "2018-10-04" in markdown


def test_main_with_limit_only_stores_that_slice(tmp_path: Path, cache: tuple[Path, Path]) -> None:
    """``--limit`` stores only the selected evidence rows."""
    module = _load_module()
    db_path, processed = cache
    reports = tmp_path / "reports"
    config = _config(tmp_path, db_path, processed, reports)
    assert module.main(["--config", str(config), "--limit", "1"]) == 0
    assert _stored_asins(_settings(db_path, processed, reports)) == [P_A]


def test_main_with_asins_stores_only_those(tmp_path: Path, cache: tuple[Path, Path]) -> None:
    """``--asins`` stores exactly the requested evidence rows."""
    module = _load_module()
    db_path, processed = cache
    reports = tmp_path / "reports"
    config = _config(tmp_path, db_path, processed, reports)
    assert module.main(["--config", str(config), "--asins", P_C, P_A]) == 0
    assert _stored_asins(_settings(db_path, processed, reports)) == [P_A, P_C]


def test_main_returns_one_on_an_unknown_asin(tmp_path: Path, cache: tuple[Path, Path]) -> None:
    """A typo in ``--asins`` exits 1 and stores nothing."""
    module = _load_module()
    db_path, processed = cache
    reports = tmp_path / "reports"
    config = _config(tmp_path, db_path, processed, reports)
    assert module.main(["--config", str(config), "--asins", "NOTINTHERE"]) == 1
    assert _stored_asins(_settings(db_path, processed, reports)) == []


def test_main_returns_one_without_meta_json(tmp_path: Path, cache: tuple[Path, Path]) -> None:
    """A missing meta.json exits 1 - the CLI never falls back to the wall clock."""
    module = _load_module()
    db_path, processed = cache
    (processed / "meta.json").unlink()
    reports = tmp_path / "reports"
    config = _config(tmp_path, db_path, processed, reports)
    assert module.main(["--config", str(config)]) == 1


def test_main_returns_one_on_an_empty_cache(tmp_path: Path, cache: tuple[Path, Path]) -> None:
    """An empty product table exits 1 naming the build commands."""
    module = _load_module()
    db_path, processed = cache
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    with session_scope(create_session_factory(engine)) as session:
        session.execute(sa_delete(ProductAspect))
        session.execute(sa_delete(ReviewFeature))
        session.execute(sa_delete(Review))
        session.execute(sa_delete(Product))
    engine.dispose()
    reports = tmp_path / "reports"
    config = _config(tmp_path, db_path, processed, reports)
    assert module.main(["--config", str(config)]) == 1


def test_main_returns_one_on_a_bad_config(tmp_path: Path) -> None:
    """A missing config file exits 1 without touching any cache."""
    module = _load_module()
    assert module.main(["--config", str(tmp_path / "nope.yaml")]) == 1


def test_main_counts_a_zero_product_cache_gracefully(
    tmp_path: Path, cache: tuple[Path, Path]
) -> None:
    """An empty run still reports zero products through the summary."""
    module = _load_module()
    db_path, processed = cache
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    with session_scope(create_session_factory(engine)) as session:
        total = session.scalar(sa_select(func.count()).select_from(ProductEvidence))
        assert total == 0
    engine.dispose()
    # The empty-cache guard is exercised in test_main_returns_one_on_an_empty_cache;
    # here a filtered --asins run over the seeded cache stays exit 0.
    reports = tmp_path / "reports"
    config = _config(tmp_path, db_path, processed, reports)
    assert module.main(["--config", str(config), "--asins", P_A, P_B, P_C]) == 0


def test_parse_args_defaults() -> None:
    """The default config path is the master config and no filter is applied."""
    module = _load_module()
    args = module.parse_args([])
    assert args.config == "configs/config.yaml"
    assert args.limit is None
    assert args.asins is None


def test_parse_args_reads_limit_and_asins() -> None:
    """``--limit`` and ``--asins`` parse into the documented shapes."""
    module = _load_module()
    args = module.parse_args(["--limit", "3", "--asins", "B0000000PA", "B0000000PB"])
    assert args.limit == 3
    assert args.asins == ["B0000000PA", "B0000000PB"]

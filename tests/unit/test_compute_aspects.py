"""End-to-end tests for the staged ``scripts/compute_aspects.py`` CLI.

The spaCy pass is replaced by a deterministic fake extractor, so the test still
exercises the real candidate cache, vocabulary filter, database upserts,
aggregation and CLI wiring without loading a model.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from collections.abc import Iterator
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pytest
from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from munasib.db.models import Product, ProductAspect, Review, ReviewFeature
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)
from munasib.review_intel.aspects.extraction import ExtractionTiming

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compute_aspects.py"
BUILD_TIMESTAMP = "2026-10-24T06:15:06+00:00"


def _load_module():
    """Import the script module.

    Returns:
        Loaded module.
    """
    return _load_module_cached()


@lru_cache(maxsize=1)
def _load_module_cached():
    """Load and cache the script module (one patch target per test session).

    Returns:
        Loaded module.
    """
    spec = importlib.util.spec_from_file_location("munasib_compute_aspects_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _product_row(asin: str, n_reviews: int) -> dict:
    """Build a minimal product row.

    Args:
        asin: Product ASIN.
        n_reviews: Stored review count.

    Returns:
        Insert payload.
    """
    return {
        "asin": asin,
        "split": "train",
        "title": f"Product {asin}",
        "title_clean": f"product {asin}",
        "brand_missing": True,
        "price_missing": True,
        "rank_missing": True,
        "n_reviews": n_reviews,
        "image_missing": True,
    }


def _review_row(asin: str, review_id: str, text: str, label: int) -> dict:
    """Build a minimal review row.

    Args:
        asin: Owning product.
        review_id: Stable review id.
        text: Clean review text.
        label: Weak label (not used by the aspect stage).

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
        "has_text": True,
        "text_length": len(text),
        "helpful_votes": 0,
        "verified": None,
        "review_time": datetime(2018, 1, 1),
        "year": 2018,
        "month": 1,
        "sentiment_label_weak": label,
    }


def _seed(engine: Engine) -> None:
    """Seed two products with a pain point and a strength.

    Product A has 16 reviews: 5 predicted negative (4 mention "filter"), 11
    predicted positive (2 mention "filter"). Product B has 10 positive reviews,
    all mentioning "motor". One extra review mentions "husband", which the
    stoplist must remove.

    Args:
        engine: Target cache.
    """
    rows: list[dict] = []
    products = [_product_row("B0000000A1", 16), _product_row("B0000000B1", 11)]
    reviews: list[dict] = []
    features: list[dict] = []
    # Product A: no-aspect reviews.
    reviews.append(_review_row("B0000000A1", "a00", "works fine", 0))
    for index in range(1, 10):
        reviews.append(_review_row("B0000000A1", f"a{index:02d}", "works fine", 2))
    # Product A: the filter aspect, 4 negatives and 2 positives.
    for index in range(10, 14):
        reviews.append(_review_row("B0000000A1", f"a{index:02d}", "bad filter", 0))
    for index in range(14, 16):
        reviews.append(_review_row("B0000000A1", f"a{index:02d}", "good filter", 2))
    # Product B: a clean strength.
    for index in range(10):
        reviews.append(_review_row("B0000000B1", f"b{index:02d}", "great motor", 2))
    # A single generic candidate that the stoplist must remove.
    reviews.append(_review_row("B0000000B1", "b99", "my husband likes it", 2))
    for review in reviews:
        label = review["sentiment_label_weak"]
        features.append(
            {
                "review_id": review["review_id"],
                "predicted_label": label,
                "prob_negative": 0.8 if label == 0 else 0.1,
                "prob_neutral": 0.1,
                "prob_positive": 0.7 if label == 2 else 0.1,
                "aspects_json": None,
            }
        )
    rows.extend(products)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        session.execute(insert(Product), rows)
        session.execute(insert(Review), reviews)
        session.execute(insert(ReviewFeature), features)


@pytest.fixture()
def workspace(tmp_path: Path) -> Iterator[dict[str, Path]]:
    """Create a seeded cache, processed dir and config.

    Yields:
        Paths for ``db``, ``config``, ``processed`` and the aspect artifacts.
    """
    db_path = tmp_path / "cache.sqlite"
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    init_db(engine)
    _seed(engine)
    engine.dispose()

    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "meta.json").write_text(
        json.dumps({"built_at": BUILD_TIMESTAMP}), encoding="utf-8"
    )
    aspects = tmp_path / "aspects"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
project:
  random_seed: 42
paths:
  db_path: "{db_path.as_posix()}"
  processed_data_dir: "{processed.as_posix()}"
review_intel:
  aspects:
    min_review_count: 2
    min_mentions: 5
    chunk_rows: 1000
    speed_sample_size: 10
    candidate_parts_dir: "{(aspects / 'parts').as_posix()}"
    vocabulary_path: "{(aspects / 'vocabulary.parquet').as_posix()}"
    vocabulary_csv_path: "{(aspects / 'vocabulary.csv').as_posix()}"
    review_aspects_path: "{(aspects / 'review_aspects.parquet').as_posix()}"
    audit_sample_path: "{(aspects / 'audit.csv').as_posix()}"
""",
        encoding="utf-8",
    )
    yield {
        "db": db_path,
        "config": config,
        "processed": processed,
        "aspects": aspects,
    }


@pytest.fixture()
def fake_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace spaCy with a deterministic keyword extractor.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    module = _load_module()

    def fake_iter(nlp, texts, config, *, apply_stoplist: bool = False):
        for text in list(texts):
            if "filter" in text:
                yield ["filter"]
            elif "motor" in text:
                yield ["motor"]
            elif "husband" in text:
                yield ["husband"]
            else:
                yield []

    monkeypatch.setattr(module, "load_nlp", lambda config: object())
    monkeypatch.setattr(module, "iter_extracted_aspects", fake_iter)
    monkeypatch.setattr(
        module,
        "measure_extraction_speed",
        lambda nlp, texts, config, apply_stoplist=False: ExtractionTiming(
            documents=len(texts), seconds=0.01
        ),
    )


def _stored(workspace: dict[str, Path]) -> tuple[list[ProductAspect], list[ReviewFeature]]:
    """Read stored aspect rows and review features.

    Args:
        workspace: Fixture paths.

    Returns:
        ``(product_aspects, review_features)``.
    """
    engine = create_engine_from_settings(db_url=f"sqlite:///{workspace['db']}")
    try:
        factory = create_session_factory(engine)
        with session_scope(factory) as session:
            aspects = list(
                session.scalars(
                    select(ProductAspect).order_by(ProductAspect.asin, ProductAspect.aspect)
                )
            )
            features = list(
                session.scalars(select(ReviewFeature).order_by(ReviewFeature.review_id))
            )
            return aspects, features
    finally:
        engine.dispose()


def test_all_runs_the_three_stages_and_writes_expected_classifications(
    workspace: dict[str, Path],
    fake_extraction: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A full run writes the vocabulary, per-review aspects and product rows."""
    module = _load_module()
    assert module.main(["--config", str(workspace["config"]), "--all"]) == 0
    out = capsys.readouterr().out
    assert "vocabulary size" in out
    assert "reviews processed" in out
    assert "pain points" in out
    assert "Example products:" in out

    aspects, features = _stored(workspace)
    by_key = {(row.asin, row.aspect): row for row in aspects}
    assert by_key[("B0000000A1", "filter")].classification == "pain_point"
    assert by_key[("B0000000A1", "filter")].negative_count == 4
    assert by_key[("B0000000B1", "motor")].classification == "strength"
    assert by_key[("B0000000B1", "motor")].negative_count == 0
    assert ("B0000000B1", "husband") not in by_key
    assert len(features) == 27
    assert all(feature.aspects_json is not None for feature in features)
    assert workspace["aspects"].joinpath("vocabulary.csv").exists()
    assert workspace["aspects"].joinpath("review_aspects.parquet").exists()


def test_complete_cache_is_reused_without_calling_the_extractor(
    workspace: dict[str, Path],
    fake_extraction: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second vocabulary run reads parts and never touches spaCy."""
    module = _load_module()
    assert module.main(["--config", str(workspace["config"]), "--vocabulary"]) == 0

    def exploding(*args, **kwargs):
        raise AssertionError("the extractor must not run for a complete cache")

    monkeypatch.setattr(module, "iter_extracted_aspects", exploding)
    summary = module.run_vocabulary(module.load_settings(str(workspace["config"])), resume=False)
    assert summary.extracted_now == 0
    assert summary.reused_parts == 1
    assert summary.vocabulary_size == 2


def test_source_change_requires_removing_the_cache(
    workspace: dict[str, Path],
    fake_extraction: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A fingerprint mismatch exits 2 and never mixes parts."""
    module = _load_module()
    assert module.main(["--config", str(workspace["config"]), "--vocabulary"]) == 0
    engine = create_engine_from_settings(db_url=f"sqlite:///{workspace['db']}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "UPDATE reviews SET review_text_clean='changed text' WHERE review_id='a00'"
            )
    finally:
        engine.dispose()
    with caplog.at_level(logging.ERROR):
        assert module.main(
            ["--config", str(workspace["config"]), "--vocabulary", "--resume"]
        ) == 2
    assert "cache_conflict" in caplog.text


def test_limit_is_rejected_for_all(workspace: dict[str, Path]) -> None:
    """A partial review pass cannot produce a whole-category aggregation."""
    module = _load_module()
    assert module.main(["--config", str(workspace["config"]), "--all", "--limit", "2"]) == 1


def test_parse_args_requires_exactly_one_stage() -> None:
    """The CLI has no implicit stage and accepts one stage at a time."""
    module = _load_module()
    with pytest.raises(SystemExit):
        module.parse_args([])
    with pytest.raises(SystemExit):
        module.parse_args(["--vocabulary", "--reviews"])


def test_aggregate_requires_complete_review_aspects(
    workspace: dict[str, Path], fake_extraction: None
) -> None:
    """Aggregation refuses to run on a partial per-review pass."""
    module = _load_module()
    assert module.main(["--config", str(workspace["config"]), "--vocabulary"]) == 0
    assert module.main(["--config", str(workspace["config"]), "--reviews", "--limit", "2"]) == 0
    assert module.main(["--config", str(workspace["config"]), "--aggregate"]) == 1


def test_stored_product_aspects_have_the_build_timestamp(
    workspace: dict[str, Path], fake_extraction: None
) -> None:
    """``computed_at`` comes from meta.json, never the wall clock."""
    module = _load_module()
    assert module.main(["--config", str(workspace["config"]), "--all"]) == 0
    aspects, _ = _stored(workspace)
    assert aspects
    assert all(row.computed_at == datetime(2026, 10, 24, 6, 15, 6) for row in aspects)

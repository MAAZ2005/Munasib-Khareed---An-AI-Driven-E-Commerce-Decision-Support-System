"""Hermetic tests for ``scripts/compute_risk_indicators.py``.

A temporary SQLite cache holds three train products with stored predictions,
evidence payloads and price features; a temporary ``meta.json`` fixes the
reference date and build timestamp. Every number asserted here is produced by the
code under test - no network, no real ``data/`` artifact.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import insert

from munasib.db.models import (
    Product,
    ProductEvidence,
    ProductFeature,
    ProductRiskIndicator,
    Review,
    ReviewFeature,
)
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compute_risk_indicators.py"

P_A = "B0000000KA"
P_B = "B0000000KB"
P_C = "B0000000KC"

BUILD_TIMESTAMP = "2026-10-24T06:15:06+00:00"
REFERENCE = datetime(2018, 10, 4)
BUILD_AT = datetime(2026, 10, 24, 6, 15, 6)


def _load_module():
    """Import ``scripts/compute_risk_indicators.py`` as a module.

    Returns:
        The loaded script module.
    """
    spec = importlib.util.spec_from_file_location("munasib_compute_risk_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _product_row(asin: str) -> dict:
    """Build a minimal products insert payload.

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
        "n_reviews": 2,
        "mean_rating": 4.0,
        "first_review_time": None,
        "last_review_time": None,
        "image_url": None,
        "image_missing": True,
        "image_cached": False,
    }


def _review_row(asin: str, review_id: str, rating: float) -> dict:
    """Build one reviews-table insert payload.

    Args:
        asin: Owning product ASIN.
        review_id: Stable review key.
        rating: Star rating.

    Returns:
        Insert payload.
    """
    when = datetime(2018, 5, 1)
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
        "helpful_votes": 1,
        "verified": True,
        "review_time": when,
        "year": when.year,
        "month": when.month,
        "sentiment_label_weak": 2 if rating >= 4 else 0,
    }


def _prediction(review_id: str, label: int) -> dict:
    """Build one review_features insert payload.

    Args:
        review_id: Owning review key.
        label: Predicted label (0 negative, 2 positive).

    Returns:
        Insert payload.
    """
    return {
        "review_id": review_id,
        "predicted_label": label,
        "prob_negative": 0.8 if label == 0 else 0.1,
        "prob_neutral": 0.1,
        "prob_positive": 0.1 if label == 0 else 0.8,
        "aspects_json": None,
    }


def _entry(raw: float | None, normalized: float | None, extras: dict | None = None) -> dict:
    """Build one stored indicator entry.

    Args:
        raw: Raw value.
        normalized: Normalized value.
        extras: Extra payload.

    Returns:
        Indicator mapping.
    """
    return {
        "raw_value": raw,
        "normalized": normalized,
        "direction": "benefit",
        "description": "",
        "missing_reason": None,
        "extras": extras or {},
    }


def _evidence_payload() -> dict:
    """Build a complete stored evidence payload.

    Returns:
        Evidence payload mapping.
    """
    return {
        "split": "train",
        "indicators": {
            "rating_agreement": _entry(0.5, 0.5, {"polarized": False}),
            "rating_text_agreement": _entry(None, 0.8, {"conflicts": 2, "evaluated": 20}),
            "helpfulness_weighted_sentiment": _entry(None, None, {"difference": 0.02}),
            "recent_negative_shift": _entry(
                0.05,
                0.95,
                {
                    "recent_negative_share": 0.5,
                    "overall_negative_share": 0.45,
                    "recent_predictions": 12,
                },
            ),
            "top_pain_point_share": _entry(
                0.4,
                0.6,
                {
                    "top_pain_points": [{"aspect": "battery", "negative_mentions": 1}],
                    "negative_mentions": 2,
                },
            ),
            "aspect_concentration": _entry(0.4, 0.6),
            "temporal_burst": _entry(0.4, 0.6, {"flag": False}),
            "staleness": _entry(10.0, 0.8, {"last_review_time": "2017-12-01"}),
            "verified_purchase_ratio": _entry(
                0.9, 0.9, {"verified_known": 2, "verified_count": 2}
            ),
        },
        "polarized": False,
        "temporal_burst": False,
        "low_evidence": False,
        "strength": {"value": 0.7},
    }


def _cache_rows() -> dict:
    """Build every insert payload for the temp cache.

    Returns:
        Mapping of table name to row list.
    """
    products = [_product_row(asin) for asin in (P_A, P_B, P_C)]
    reviews = []
    features = []
    for asin in (P_A, P_B, P_C):
        for index, label in enumerate((2, 0)):
            review_id = f"000000000000{asin[-2:].lower()}{index:02d}"
            reviews.append(_review_row(asin, review_id, 5.0 if label == 2 else 1.0))
            features.append(_prediction(review_id, label))
    evidence = [
        {
            "asin": asin,
            "n_reviews": 2,
            "evidence_strength": 0.7,
            "indicators": _evidence_payload(),
            "reference_date": REFERENCE,
            "computed_at": BUILD_AT,
        }
        for asin in (P_A, P_B, P_C)
    ]
    product_features = [
        {
            "asin": asin,
            "indicators": {
                "price_percentile_category": 0.5,
                "price_missing": False,
                "price_peer_group": "Home & Kitchen > Appliances",
                "price_peer_count": 40,
                "price_peer_fallback": False,
            },
            "computed_at": BUILD_AT,
        }
        for asin in (P_A, P_B, P_C)
    ]
    return {
        "products": products,
        "reviews": reviews,
        "features": features,
        "evidence": evidence,
        "product_features": product_features,
    }


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
    rows = _cache_rows()
    with session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), rows["products"])
        session.execute(insert(Review), rows["reviews"])
        session.execute(insert(ReviewFeature), rows["features"])
        session.execute(insert(ProductEvidence), rows["evidence"])
        session.execute(insert(ProductFeature), rows["product_features"])
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


def _config(tmp_path: Path, db_path: Path, processed: Path) -> Path:
    """Write a minimal config pointing at the temp cache and processed dir.

    Args:
        tmp_path: Pytest temporary directory.
        db_path: SQLite cache to use.
        processed: Processed data dir holding meta.json.

    Returns:
        The config path.
    """
    config = tmp_path / "config.yaml"
    config.write_text(
        f'paths:\n  db_path: "{db_path.as_posix()}"\n'
        f'  processed_data_dir: "{processed.as_posix()}"\n'
        f'  reports_dir: "{(tmp_path / "reports").as_posix()}"\n',
        encoding="utf-8",
    )
    return config


def _stored_rows(db_path: Path) -> list[ProductRiskIndicator]:
    """Read every stored risk row from the temp cache.

    Args:
        db_path: SQLite cache path.

    Returns:
        The stored rows, ascending by ASIN.
    """
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    factory = create_session_factory(engine)
    try:
        with session_scope(factory) as session:
            rows = list(
                session.query(ProductRiskIndicator).order_by(ProductRiskIndicator.asin)
            )
            for row in rows:
                session.expunge(row)
            return rows
    finally:
        engine.dispose()


def test_cli_writes_every_product_and_prints_the_summary(
    cache: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default run stores one row per product and prints the full report."""
    db_path, processed = cache
    module = _load_module()
    config = _config(tmp_path, db_path, processed)

    assert module.main(["--config", str(config)]) == 0

    rows = _stored_rows(db_path)
    assert len(rows) == 3
    assert {row.asin for row in rows} == {P_A, P_B, P_C}
    payload = rows[0].indicators
    assert list(payload["criteria"]) == [
        "product_quality",
        "evidence_strength",
        "rating_reliability",
        "negative_feedback_concentration",
        "price_position",
    ]
    assert len(payload["vector"]) == 5
    assert rows[0].priors["n_train_products"] == 3

    output = capsys.readouterr().out
    assert "products written" in output
    assert "agreement scale" in output
    assert "value (cost)" in output
    assert "normalized (1 - raw)" in output
    assert "spearman correlations (all products" in output
    assert "spearman correlations (train split only" in output
    assert "best 5 by product_quality" in output
    assert "worst 5 by product_quality" in output


def test_cli_limit_writes_a_stable_prefix(
    cache: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--limit`` scores the first N products in ASIN order."""
    db_path, processed = cache
    module = _load_module()
    config = _config(tmp_path, db_path, processed)

    assert module.main(["--config", str(config), "--limit", "2"]) == 0

    rows = _stored_rows(db_path)
    assert [row.asin for row in rows] == [P_A, P_B]
    assert "products written" in capsys.readouterr().out


def test_cli_asins_writes_only_the_requested_products(
    cache: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--asins`` overrides ``--limit`` and writes exactly those products."""
    db_path, processed = cache
    module = _load_module()
    config = _config(tmp_path, db_path, processed)

    assert module.main(["--config", str(config), "--asins", P_C]) == 0

    rows = _stored_rows(db_path)
    assert [row.asin for row in rows] == [P_C]
    assert "products written" in capsys.readouterr().out


def test_cli_unknown_asin_fails_without_writing(
    cache: tuple[Path, Path], tmp_path: Path
) -> None:
    """A typo cannot silently shrink the run."""
    db_path, processed = cache
    module = _load_module()
    config = _config(tmp_path, db_path, processed)

    assert module.main(["--config", str(config), "--asins", "NOPE"]) == 1
    assert _stored_rows(db_path) == []


def test_cli_negative_limit_fails_without_writing(
    cache: tuple[Path, Path], tmp_path: Path
) -> None:
    """A negative limit is a usage error."""
    db_path, processed = cache
    module = _load_module()
    config = _config(tmp_path, db_path, processed)

    assert module.main(["--config", str(config), "--limit", "-1"]) == 1
    assert _stored_rows(db_path) == []


def test_cli_missing_config_fails(tmp_path: Path) -> None:
    """An unreadable config exits 1 instead of raising."""
    module = _load_module()
    assert module.main(["--config", str(tmp_path / "missing.yaml")]) == 1

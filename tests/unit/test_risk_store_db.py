"""Database tests for the risk-indicator loaders, writer and repository.

Hermetic: a temporary SQLite cache holds a tiny catalog, two reviews (one with a
prediction, one without), one evidence row and one feature row. The loaders must
decode SQLite's JSON TEXT into mappings, and the writer must upsert idempotently.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import Engine, insert

from munasib.core.config import Settings
from munasib.db.models import (
    Product,
    ProductEvidence,
    ProductFeature,
    Review,
    ReviewFeature,
)
from munasib.db.repositories import ProductRiskIndicatorRepository
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)
from munasib.risk.compute import RiskRow
from munasib.risk.factors import FACTOR_CODES, RiskFactor
from munasib.risk.indicators import EXPECTED_DIRECTIONS, Criterion
from munasib.risk.store import (
    _decode_json_column,
    count_risk_indicators,
    load_evidence,
    load_features,
    load_predictions,
    load_products,
    read_risk_indicators,
    write_risk_indicators,
)

P_A = "B0000000RA"
P_B = "B0000000RB"
REFERENCE = datetime(2018, 10, 4)
COMPUTED_AT = datetime(2026, 10, 24, 6, 15, 6)


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
        "mean_rating": 4.5,
        "first_review_time": None,
        "last_review_time": None,
        "image_url": None,
        "image_missing": True,
        "image_cached": False,
    }


def _review_row(asin: str, review_id: str) -> dict:
    """Build one reviews-table insert payload.

    Args:
        asin: Owning product ASIN.
        review_id: Stable review key.

    Returns:
        Insert payload.
    """
    when = datetime(2018, 5, 1)
    return {
        "review_id": review_id,
        "asin": asin,
        "split": "train",
        "reviewer_id": f"user-{review_id[-3:]}",
        "rating": 5.0,
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
        "sentiment_label_weak": 2,
    }


def _criterion(name: str, value: float | None, direction: str) -> Criterion:
    """Build one criterion for a stored row.

    Args:
        name: Criterion key.
        value: Criterion value.
        direction: ``benefit`` or ``cost``.

    Returns:
        The criterion.
    """
    return Criterion(
        name=name,
        value=value,
        direction=direction,
        description=f"{name} description",
        confidence=1.0 if value is not None else 0.0,
        missing_reason=None if value is not None else "unavailable",
    )


def _risk_row(asin: str, quality: float = 0.8) -> RiskRow:
    """Build one complete computed row.

    Args:
        asin: Product ASIN.
        quality: Product-quality value.

    Returns:
        The computed row.
    """
    criteria = {
        "product_quality": _criterion("product_quality", quality, "benefit"),
        "evidence_strength": _criterion("evidence_strength", 0.7, "benefit"),
        "rating_reliability": _criterion("rating_reliability", 0.6, "benefit"),
        "negative_feedback_concentration": _criterion(
            "negative_feedback_concentration", None, "cost"
        ),
        "price_position": _criterion("price_position", 0.5, "cost"),
    }
    return RiskRow(
        asin=asin,
        split="train",
        title=f"Product {asin}",
        mean_rating=4.5,
        n_reviews=2,
        mean_net=0.5,
        criteria=criteria,
        risk_factors=(
            RiskFactor(
                code="weak_evidence",
                severity="high",
                description="evidence strength 0.10 is below 0.20",
                supporting={"evidence_strength": 0.10},
            ),
        ),
        directions=dict(EXPECTED_DIRECTIONS),
        priors={
            "mean_rating": 4.3,
            "net_sentiment": 0.6,
            "negative_share": 0.1,
            "shrinkage_m": 20.0,
            "n_train_products": 3,
        },
        reference_date=REFERENCE,
        computed_at=COMPUTED_AT,
    )


@pytest.fixture()
def engine(tmp_path: Path) -> Iterator[Engine]:
    """Create a temp cache with products, reviews, evidence and features.

    Args:
        tmp_path: Pytest temporary directory.

    Yields:
        The open engine (disposed afterwards).
    """
    db_path = tmp_path / "cache.sqlite"
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    init_db(engine)
    with session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), [_product_row(P_A), _product_row(P_B)])
        session.execute(
            insert(Review),
            [_review_row(P_A, "0000000000000a01"), _review_row(P_A, "0000000000000a02")],
        )
        # Only the first review has a stored prediction; the second must stay
        # "no prediction" rather than being imputed.
        session.execute(
            insert(ReviewFeature),
            [
                {
                    "review_id": "0000000000000a01",
                    "predicted_label": 2,
                    "prob_negative": 0.1,
                    "prob_neutral": 0.1,
                    "prob_positive": 0.8,
                    "aspects_json": None,
                }
            ],
        )
        session.execute(
            insert(ProductEvidence),
            [
                {
                    "asin": P_A,
                    "n_reviews": 2,
                    "evidence_strength": 0.7,
                    "indicators": {
                        "indicators": {
                            "rating_agreement": {"raw_value": 0.5, "normalized": 0.5}
                        },
                        "polarized": False,
                    },
                    "reference_date": REFERENCE,
                    "computed_at": COMPUTED_AT,
                }
            ],
        )
        session.execute(
            insert(ProductFeature),
            [
                {
                    "asin": P_A,
                    "indicators": {
                        "price_percentile_category": 0.4,
                        "price_missing": False,
                        "price_peer_fallback": False,
                    },
                    "computed_at": COMPUTED_AT,
                }
            ],
        )
    try:
        yield engine
    finally:
        engine.dispose()


def test_load_products_returns_the_display_columns(engine: Engine) -> None:
    """Products load in ASIN order with the columns the stage needs."""
    frame = load_products(engine)
    assert list(frame.columns) == ["asin", "split", "title", "mean_rating", "n_reviews"]
    assert frame["asin"].tolist() == [P_A, P_B]


def test_load_predictions_keeps_unpredicted_reviews_as_null(engine: Engine) -> None:
    """A review without a feature row is a null prediction, not an imputed one."""
    frame = load_predictions(engine)
    assert len(frame) == 2
    assert frame["predicted_label"].isna().sum() == 1
    predicted = frame[frame["predicted_label"].notna()].iloc[0]
    assert predicted["prob_positive"] == pytest.approx(0.8)


def test_load_evidence_decodes_the_sqlite_json_text(engine: Engine) -> None:
    """The evidence payload arrives as a mapping, never a JSON string."""
    frame = load_evidence(engine)
    assert len(frame) == 1
    payload = frame.iloc[0]["indicators"]
    assert isinstance(payload, dict)
    assert payload["polarized"] is False
    assert payload["indicators"]["rating_agreement"]["normalized"] == pytest.approx(0.5)


def test_load_features_decodes_the_sqlite_json_text(engine: Engine) -> None:
    """The feature payload arrives as a mapping, never a JSON string."""
    frame = load_features(engine)
    assert len(frame) == 1
    payload = frame.iloc[0]["indicators"]
    assert isinstance(payload, dict)
    assert payload["price_percentile_category"] == pytest.approx(0.4)


def test_decode_json_column_handles_invalid_and_missing_values() -> None:
    """Broken JSON becomes None; an existing mapping is left alone."""
    frame = pd.DataFrame({"indicators": ["{not json", '{"a": 1}', None, {"b": 2}]})
    decoded = _decode_json_column(frame, "indicators")
    assert decoded["indicators"].tolist() == [None, {"a": 1}, None, {"b": 2}]


def test_write_read_roundtrip_and_count(engine: Engine) -> None:
    """A computed row round-trips through the table with its payload intact."""
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        assert write_risk_indicators(session, [_risk_row(P_A)]) == 1
    with session_scope(factory) as session:
        assert count_risk_indicators(session) == 1
        row = read_risk_indicators(session, P_A)
        assert row is not None
        assert row.indicators["vector"] == [0.8, 0.7, 0.6, None, 0.5]
        assert row.indicators["directions"] == dict(EXPECTED_DIRECTIONS)
        assert row.risk_factors[0]["code"] == "weak_evidence"
        assert row.risk_factors[0]["severity"] == "high"
        assert row.priors["n_train_products"] == 3
        assert row.reference_date == REFERENCE
        assert row.computed_at == COMPUTED_AT


def test_upsert_refreshes_instead_of_duplicating(engine: Engine) -> None:
    """A second write for the same ASIN replaces the values, keeping one row."""
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        write_risk_indicators(session, [_risk_row(P_A, quality=0.8)])
    with session_scope(factory) as session:
        write_risk_indicators(session, [_risk_row(P_A, quality=0.3)])
    with session_scope(factory) as session:
        assert count_risk_indicators(session) == 1
        row = read_risk_indicators(session, P_A)
        assert row is not None
        assert row.indicators["criteria"]["product_quality"]["value"] == pytest.approx(0.3)


def test_repository_lists_in_asin_order_and_filters(engine: Engine) -> None:
    """Listing is stable and honours an ASIN subset."""
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        write_risk_indicators(session, [_risk_row(P_B), _risk_row(P_A)])
    with session_scope(factory) as session:
        repository = ProductRiskIndicatorRepository(session)
        assert [row.asin for row in repository.list_risk_indicators()] == [P_A, P_B]
        assert [
            row.asin for row in repository.list_risk_indicators([P_B])
        ] == [P_B]
        assert repository.get_risk_indicators(P_A) is not None
        assert repository.get_risk_indicators("missing") is None


def test_writer_rejects_timezone_aware_timestamps(engine: Engine) -> None:
    """The storage contract is naive UTC; an aware timestamp is a usage error."""
    factory = create_session_factory(engine)
    aware = _risk_row(P_A)
    object.__setattr__(
        aware,
        "computed_at",
        datetime(2026, 10, 24, 6, 15, 6, tzinfo=UTC),
    )
    with session_scope(factory) as session:
        with pytest.raises(ValueError, match="naive UTC"):
            write_risk_indicators(session, [aware])


def test_real_cache_rows_are_well_formed() -> None:
    """When the real cache exists, every stored row has the five criteria."""
    settings = Settings.model_validate({})
    db_path = Path(settings.paths.db_path)
    if not db_path.exists():
        pytest.skip(f"real serving cache not present at {db_path}")
    engine = create_engine_from_settings(settings)
    factory = create_session_factory(engine)
    try:
        with session_scope(factory) as session:
            rows = ProductRiskIndicatorRepository(session).list_risk_indicators()
        if not rows:
            pytest.skip("the real cache has no product_risk_indicators rows yet")
        for row in rows:
            criteria = row.indicators["criteria"]
            assert list(criteria) == [
                "product_quality",
                "evidence_strength",
                "rating_reliability",
                "negative_feedback_concentration",
                "price_position",
            ]
            assert len(row.indicators["vector"]) == 5
            assert row.indicators["directions"] == dict(EXPECTED_DIRECTIONS)
            for factor in row.risk_factors:
                assert factor["code"] in FACTOR_CODES
                assert factor["severity"] in {"low", "medium", "high"}
            assert row.priors["n_train_products"] > 0
    finally:
        engine.dispose()

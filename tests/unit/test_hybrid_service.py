"""Unit tests for HybridAnalysisService.

Validates that:
1. Output validates against AnalysisResponse.
2. Real fields equal stored values (aspects, evidence composite, predicted shares, rating distribution).
3. Placeholders (fuzzy, TOPSIS, SHAP, risk score, verdict, concerns, alternatives' risk scores)
   are deterministic and identical to the stub logic.
4. Warnings carry the PARTIAL: prefix and pipeline_version is 'hybrid-0'.
5. Missing aspects, evidence, or risk indicator rows degrade gracefully with explicit reasons.
6. Repeated calls on the same product are strictly deterministic.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from munasib.api.schemas.analysis import AnalysisResponse
from munasib.api.schemas.evidence import IndicatorDirection
from munasib.api.schemas.review_analysis import AspectPolarity
from munasib.core.config import Settings, get_settings
from munasib.core.constants import (
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABEL_POSITIVE,
    SPLIT_TRAIN,
)
from munasib.core.exceptions import ProductNotFoundError
from munasib.db.models import (
    Product,
    ProductAspect,
    ProductEvidence,
    ProductRiskIndicator,
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
from munasib.pipeline.service import (
    HYBRID_PIPELINE_VERSION,
    HYBRID_WARNING_PREFIX,
    HybridAnalysisService,
    StubAnalysisService,
    _hybrid_aspects,
)

SUBJECT = "B000000001"
ALT_ONE = "B000000002"
NO_ASPECTS_PROD = "B000000003"
NO_EVIDENCE_PROD = "B000000004"
NULL_STRENGTH_PROD = "B000000005"

META = {
    "category": "Appliances",
    "built_at": "2026-10-11T10:45:51+00:00",
    "review_date_range": {
        "min": "2018-01-01T00:00:00+00:00",
        "max": "2018-10-04T00:00:00+00:00",
    },
    "splits": {
        "sizes": {
            "train": {"products": 4, "reviews": 8},
            "val": {"products": 0, "reviews": 0},
            "test": {"products": 0, "reviews": 0},
        }
    },
}


def _product_row(asin: str, *, n_reviews: int, mean_rating: float | None) -> dict:
    return {
        "asin": asin,
        "split": SPLIT_TRAIN,
        "title": f"Product {asin}",
        "title_clean": f"product {asin}",
        "brand": "TestBrand",
        "brand_clean": "testbrand",
        "brand_missing": False,
        "price": 19.99,
        "price_max": None,
        "price_status": "single",
        "price_missing": False,
        "rank": "#1 in Widgets",
        "rank_value": 1,
        "rank_category": "Widgets",
        "rank_missing": False,
        "description": "Description text",
        "features": ["Feature A", "Feature B"],
        "n_reviews": n_reviews,
        "mean_rating": mean_rating,
        "first_review_time": datetime(2018, 1, 1),
        "last_review_time": datetime(2018, 3, 1),
    }


def _review_row(
    asin: str,
    review_id: str,
    when: datetime,
    *,
    rating: float | None,
    weak_label: int | None,
    verified: bool | None,
    helpful: int,
    text_length: int,
) -> dict:
    return {
        "review_id": review_id,
        "asin": asin,
        "split": SPLIT_TRAIN,
        "reviewer_id": f"user-{review_id[-1]}",
        "rating": rating,
        "review_text_raw": "Review body text",
        "review_text_clean": "review body text",
        "summary_raw": "Summary",
        "summary_clean": "summary",
        "has_text": text_length > 0,
        "text_length": text_length,
        "helpful_votes": helpful,
        "verified": verified,
        "review_time": when,
        "year": when.year,
        "month": when.month,
        "sentiment_label_weak": weak_label,
    }


def _review_feature_row(
    review_id: str,
    predicted_label: int | None,
    prob_neg: float,
    prob_neu: float,
    prob_pos: float,
) -> dict:
    return {
        "review_id": review_id,
        "predicted_label": predicted_label,
        "prob_negative": prob_neg,
        "prob_neutral": prob_neu,
        "prob_positive": prob_pos,
        "aspects_json": [],
    }


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    db_file = tmp_path / "test.db"
    meta_file = tmp_path / "meta.json"
    meta_file.write_text(json.dumps(META), encoding="utf-8")

    settings = get_settings().model_copy(
        update={
            "paths": get_settings().paths.model_copy(
                update={
                    "db_path": str(db_file),
                    "processed_data_dir": str(tmp_path),
                }
            ),
            "analysis": get_settings().analysis.model_copy(update={"service": "hybrid"}),
        }
    )
    return settings


@pytest.fixture
def engine(test_settings: Settings) -> Engine:
    eng = create_engine_from_settings(test_settings)
    init_db(eng)
    return eng


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    factory = create_session_factory(engine)
    with session_scope(factory) as sess:
        # 1. Products
        sess.execute(
            insert(Product),
            [
                _product_row(SUBJECT, n_reviews=4, mean_rating=3.25),
                _product_row(ALT_ONE, n_reviews=2, mean_rating=4.5),
                _product_row(NO_ASPECTS_PROD, n_reviews=1, mean_rating=4.0),
                _product_row(NO_EVIDENCE_PROD, n_reviews=1, mean_rating=2.0),
                _product_row(NULL_STRENGTH_PROD, n_reviews=1, mean_rating=3.0),
            ],
        )

        # 2. Reviews
        sess.execute(
            insert(Review),
            [
                _review_row(
                    SUBJECT, "00000000000000a1", datetime(2018, 1, 1),
                    rating=5.0, weak_label=SENTIMENT_LABEL_POSITIVE, verified=True,
                    helpful=2, text_length=100,
                ),
                _review_row(
                    SUBJECT, "00000000000000a2", datetime(2018, 2, 1),
                    rating=1.0, weak_label=SENTIMENT_LABEL_NEGATIVE, verified=False,
                    helpful=0, text_length=50,
                ),
                _review_row(
                    SUBJECT, "00000000000000a3", datetime(2018, 3, 1),
                    rating=3.0, weak_label=SENTIMENT_LABEL_NEUTRAL, verified=None,
                    helpful=1, text_length=200,
                ),
                _review_row(
                    SUBJECT, "00000000000000a4", datetime(2018, 3, 1),
                    rating=4.0, weak_label=SENTIMENT_LABEL_POSITIVE, verified=True,
                    helpful=0, text_length=50,
                ),
                _review_row(
                    ALT_ONE, "00000000000000b1", datetime(2018, 1, 5),
                    rating=5.0, weak_label=SENTIMENT_LABEL_POSITIVE, verified=True,
                    helpful=1, text_length=200,
                ),
                _review_row(
                    ALT_ONE, "00000000000000b2", datetime(2018, 2, 5),
                    rating=4.0, weak_label=SENTIMENT_LABEL_POSITIVE, verified=True,
                    helpful=0, text_length=200,
                ),
                _review_row(
                    NO_ASPECTS_PROD, "00000000000000c1", datetime(2018, 1, 10),
                    rating=4.0, weak_label=SENTIMENT_LABEL_POSITIVE, verified=True,
                    helpful=0, text_length=80,
                ),
                _review_row(
                    NO_EVIDENCE_PROD, "00000000000000d1", datetime(2018, 1, 12),
                    rating=2.0, weak_label=SENTIMENT_LABEL_NEGATIVE, verified=False,
                    helpful=0, text_length=60,
                ),
                _review_row(
                    NULL_STRENGTH_PROD, "00000000000000e1", datetime(2018, 1, 14),
                    rating=3.0, weak_label=SENTIMENT_LABEL_NEUTRAL, verified=True,
                    helpful=0, text_length=70,
                ),
            ],
        )

        # 3. Review Features (predicted labels)
        sess.execute(
            insert(ReviewFeature),
            [
                _review_feature_row("00000000000000a1", 2, 0.05, 0.15, 0.80),
                _review_feature_row("00000000000000a2", 0, 0.85, 0.10, 0.05),
                _review_feature_row("00000000000000a3", 1, 0.10, 0.80, 0.10),
                _review_feature_row("00000000000000a4", 2, 0.05, 0.15, 0.80),
                _review_feature_row("00000000000000b1", 2, 0.05, 0.05, 0.90),
                _review_feature_row("00000000000000b2", 2, 0.05, 0.10, 0.85),
            ],
        )

        # 4. Related edges
        sess.execute(
            insert(RelatedProduct),
            [
                {
                    "asin": SUBJECT,
                    "related_asin": ALT_ONE,
                    "relation_type": "also_bought",
                    "in_subset": True,
                }
            ],
        )

        # 5. Product Aspects (SUBJECT has pain_point + strength + neutral)
        sess.execute(
            insert(ProductAspect),
            [
                {
                    "asin": SUBJECT,
                    "aspect": "durability",
                    "mention_count": 3,
                    "negative_count": 0,
                    "neutral_count": 1,
                    "positive_count": 2,
                    "negative_share": 0.0,
                    "neutral_share": 0.333333,
                    "positive_share": 0.666667,
                    "net_sentiment": 0.666667,
                    "product_labeled_reviews": 4,
                    "product_negative_share": 0.25,
                    "negativity_lift": None,
                    "classification": "strength",
                    "examples": ["Solid build quality."],
                    "computed_at": datetime(2026, 10, 11),
                },
                {
                    "asin": SUBJECT,
                    "aspect": "installation",
                    "mention_count": 2,
                    "negative_count": 2,
                    "neutral_count": 0,
                    "positive_count": 0,
                    "negative_share": 1.0,
                    "neutral_share": 0.0,
                    "positive_share": 0.0,
                    "net_sentiment": -1.0,
                    "product_labeled_reviews": 4,
                    "product_negative_share": 0.25,
                    "negativity_lift": 4.0,
                    "classification": "pain_point",
                    "examples": ["Difficult to install."],
                    "computed_at": datetime(2026, 10, 11),
                },
                {
                    "asin": SUBJECT,
                    "aspect": "size",
                    "mention_count": 1,
                    "negative_count": 0,
                    "neutral_count": 1,
                    "positive_count": 0,
                    "negative_share": 0.0,
                    "neutral_share": 1.0,
                    "positive_share": 0.0,
                    "net_sentiment": 0.0,
                    "product_labeled_reviews": 4,
                    "product_negative_share": 0.25,
                    "negativity_lift": None,
                    "classification": "neutral",
                    "examples": ["Average size."],
                    "computed_at": datetime(2026, 10, 11),
                },
            ],
        )

        # 6. Product Evidence
        sess.execute(
            insert(ProductEvidence),
            [
                {
                    "asin": SUBJECT,
                    "n_reviews": 4,
                    "evidence_strength": 0.5875,
                    "indicators": {
                        "split": "train",
                        "indicators": {
                            "verified_purchase_ratio": {
                                "raw_value": 0.666667,
                                "normalized": 0.666667,
                                "direction": "benefit",
                                "description": "2 of 3 reviews are verified.",
                                "missing_reason": None,
                            },
                            "helpfulness_coverage": {
                                "raw_value": 0.5,
                                "normalized": 0.5,
                                "direction": "benefit",
                                "description": "2 of 4 reviews have helpful votes.",
                                "missing_reason": None,
                            },
                            "rating_agreement": {
                                "raw_value": 0.25,
                                "normalized": 0.25,
                                "direction": "benefit",
                                "description": "Rating agreement 0.25.",
                                "missing_reason": None,
                            },
                            "staleness": {
                                "raw_value": 7.0,
                                "normalized": 0.63,
                                "direction": "cost",
                                "description": "Latest review is 7 months old.",
                                "missing_reason": None,
                            },
                            "recent_negative_shift": {
                                "raw_value": None,
                                "normalized": None,
                                "direction": "cost",
                                "description": "Undefined due to small window.",
                                "missing_reason": "recent_window_below_min_reviews",
                            },
                        },
                    },
                    "reference_date": datetime(2018, 10, 4),
                    "computed_at": datetime(2026, 10, 11),
                },
                {
                    "asin": ALT_ONE,
                    "n_reviews": 2,
                    "evidence_strength": 0.85,
                    "indicators": {"split": "train", "indicators": {}},
                    "reference_date": datetime(2018, 10, 4),
                    "computed_at": datetime(2026, 10, 11),
                },
                {
                    "asin": NULL_STRENGTH_PROD,
                    "n_reviews": 1,
                    "evidence_strength": None,
                    "indicators": {
                        "split": "train",
                        "indicators": {
                            "rating_agreement": {
                                "raw_value": 1.0,
                                "normalized": 1.0,
                                "direction": "benefit",
                                "description": "Single rated review.",
                                "missing_reason": None,
                            }
                        },
                    },
                    "reference_date": datetime(2018, 10, 4),
                    "computed_at": datetime(2026, 10, 11),
                },
            ],
        )

        # 7. Product Risk Indicators
        sess.execute(
            insert(ProductRiskIndicator),
            [
                {
                    "asin": SUBJECT,
                    "indicators": {
                        "split": "train",
                        "criteria": {
                            "product_quality": {
                                "value": 0.62,
                                "direction": "benefit",
                                "description": "Product quality description.",
                            },
                            "evidence_strength": {
                                "value": 0.5875,
                                "direction": "benefit",
                                "description": "Stored evidence strength 0.5875.",
                            },
                            "rating_reliability": {
                                "value": 0.75,
                                "direction": "benefit",
                                "description": "Rating reliability description.",
                            },
                            "negative_feedback_concentration": {
                                "value": 0.35,
                                "direction": "cost",
                                "description": "Negative concentration description.",
                            },
                            "price_position": {
                                "value": 0.45,
                                "direction": "cost",
                                "description": "Price position description.",
                            },
                        },
                    },
                    "risk_factors": [],
                    "priors": {},
                    "reference_date": datetime(2018, 10, 4),
                    "computed_at": datetime(2026, 10, 11),
                },
                {
                    "asin": NO_ASPECTS_PROD,
                    "indicators": {
                        "split": "train",
                        "criteria": {
                            "product_quality": {
                                "value": 0.5,
                                "direction": "benefit",
                                "description": "Product quality description.",
                            },
                            "evidence_strength": {
                                "value": 0.4,
                                "direction": "benefit",
                                "description": "Evidence strength description.",
                            },
                            "rating_reliability": {
                                "value": 0.6,
                                "direction": "benefit",
                                "description": "Rating reliability description.",
                            },
                            "negative_feedback_concentration": {
                                "value": 0.2,
                                "direction": "cost",
                                "description": "Negative concentration description.",
                            },
                            "price_position": {
                                "value": None,
                                "direction": "cost",
                                "description": (
                                    "The product has no usable price percentile, so the "
                                    "price position is unknown and is not imputed."
                                ),
                                "missing_reason": "price_missing",
                            },
                        },
                    },
                    "risk_factors": [],
                    "priors": {},
                    "reference_date": datetime(2018, 10, 4),
                    "computed_at": datetime(2026, 10, 11),
                }
            ],
        )
        sess.flush()
        yield sess


def test_hybrid_analysis_validates_and_stamps_version(
    session: Session, test_settings: Settings
) -> None:
    """The hybrid service output parses into AnalysisResponse with hybrid-0 pipeline version."""
    service = HybridAnalysisService(session, test_settings)
    analysis = service.analyze(SUBJECT)

    assert isinstance(analysis, AnalysisResponse)
    assert analysis.pipeline_version == HYBRID_PIPELINE_VERSION
    assert analysis.asin == SUBJECT
    assert any(w.startswith(HYBRID_WARNING_PREFIX) for w in analysis.warnings)
    # Round-trip through the wire format: every field validates and survives.
    payload = analysis.model_dump()
    assert AnalysisResponse.model_validate(payload).model_dump() == payload


def test_hybrid_service_real_fields_match_stored_data(
    session: Session, test_settings: Settings
) -> None:
    """Real parts (aspects, evidence, predicted sentiment, rating distribution) match DB values."""
    service = HybridAnalysisService(session, test_settings)
    analysis = service.analyze(SUBJECT)

    # 1. Rating distribution
    rd = analysis.review_analysis.rating_distribution
    assert rd.total == 4
    assert rd.mean_rating == 3.25
    assert rd.counts == {1: 1, 2: 0, 3: 1, 4: 1, 5: 1}

    # 2. Sentiment breakdown (from predicted labels: 2 positive, 1 neutral, 1 negative)
    sb = analysis.review_analysis.sentiment
    assert sb.total == 4
    assert sb.negative_count == 1
    assert sb.neutral_count == 1
    assert sb.positive_count == 2
    assert sb.unlabelled_count == 0
    assert sb.negative_share == 0.25
    assert sb.neutral_share == 0.25
    assert sb.positive_share == 0.50
    # Star-derived weak shares are also exposed
    assert sb.star_negative_share == 0.25
    assert sb.star_neutral_share == 0.25
    assert sb.star_positive_share == 0.50

    # 3. Aspects: pain points first, then strengths, then neutral
    aspects = analysis.review_analysis.aspects
    assert len(aspects) == 3
    assert aspects[0].aspect == "installation"
    assert aspects[0].polarity == AspectPolarity.NEGATIVE
    assert aspects[0].mention_count == 2
    assert aspects[0].examples == ["Difficult to install."]

    assert aspects[1].aspect == "durability"
    assert aspects[1].polarity == AspectPolarity.POSITIVE
    assert aspects[1].mention_count == 3
    assert aspects[1].examples == ["Solid build quality."]

    assert aspects[2].aspect == "size"
    assert aspects[2].polarity == AspectPolarity.MIXED
    assert aspects[2].mention_count == 1

    # 4. Evidence: composite strength and indicators from product_evidence
    ev = analysis.evidence
    assert ev.evidence_strength == 0.5875
    assert ev.verified_purchase.value == 0.6667
    assert ev.verified_purchase.direction == IndicatorDirection.BENEFIT
    assert ev.helpfulness.value == 0.5
    assert ev.review_depth.name == "review_depth"

    # Details check: extra indicators with missing_reason
    details_map = {ind.name: ind for ind in ev.details}
    assert "rating_agreement" in details_map
    assert details_map["rating_agreement"].value == 0.25
    assert "staleness" in details_map
    assert details_map["staleness"].value == 0.63
    assert "recent_negative_shift" in details_map
    assert details_map["recent_negative_shift"].value is None
    assert details_map["recent_negative_shift"].missing_reason == "recent_window_below_min_reviews"

    # 5. Risk indicators: 5 criteria populated from product_risk_indicators
    assert len(analysis.risk_indicators) == 5
    risk_map = {crit.name: crit for crit in analysis.risk_indicators}
    assert risk_map["product_quality"].value == 0.62
    assert risk_map["evidence_strength"].value == 0.5875
    assert risk_map["rating_reliability"].value == 0.75
    assert risk_map["negative_feedback_concentration"].value == 0.35
    assert risk_map["price_position"].value == 0.45


def test_hybrid_service_placeholders_match_stub_identically(
    session: Session, test_settings: Settings
) -> None:
    """Placeholder fields (fuzzy, TOPSIS, SHAP, risk score) are identical to StubAnalysisService."""
    hybrid = HybridAnalysisService(session, test_settings)
    stub = StubAnalysisService(session, test_settings)

    h_res = hybrid.analyze(SUBJECT)
    s_res = stub.analyze(SUBJECT)

    assert h_res.decision is not None and s_res.decision is not None
    assert h_res.decision.fuzzy == s_res.decision.fuzzy
    assert h_res.decision.topsis.closeness == s_res.decision.topsis.closeness
    assert h_res.decision.topsis.rank == s_res.decision.topsis.rank

    assert h_res.explanation is not None and s_res.explanation is not None
    assert h_res.explanation.base_value == s_res.explanation.base_value
    # The feature values differ (predicted vs weak shares); the seeded
    # contributions themselves must be identical.
    assert [item.contribution for item in h_res.explanation.contributions] == [
        item.contribution for item in s_res.explanation.contributions
    ]

    assert h_res.risk_score == s_res.risk_score
    assert h_res.verdict is not None and s_res.verdict is not None
    assert h_res.verdict.risk_score == s_res.verdict.risk_score
    assert len(h_res.alternatives.items) == len(s_res.alternatives.items)
    if h_res.alternatives.items:
        assert h_res.alternatives.items[0].risk_score == s_res.alternatives.items[0].risk_score


def test_hybrid_service_graceful_degradation_missing_aspects_or_evidence(
    session: Session, test_settings: Settings
) -> None:
    """When aspects or evidence rows are missing, service degrades gracefully with explicit reasons."""
    service = HybridAnalysisService(session, test_settings)

    # 1. Product without aspects
    no_aspects = service.analyze(NO_ASPECTS_PROD)
    assert no_aspects.review_analysis.aspects == []
    assert any("No stored aspect rows exist" in w for w in no_aspects.warnings)

    # 2. Product without stored evidence
    no_evidence = service.analyze(NO_EVIDENCE_PROD)
    assert no_evidence.evidence is not None
    assert no_evidence.evidence.evidence_strength is not None
    assert any("No stored product_evidence row exists" in w for w in no_evidence.warnings)
    # Risk indicators empty when no row exists
    assert no_evidence.risk_indicators == []
    # Its review has no stored prediction, so the sentiment is explicitly unlabelled.
    assert no_evidence.review_analysis.sentiment.unlabelled_count == 1
    assert any("No stored sentiment predictions" in w for w in no_evidence.warnings)


def test_hybrid_service_keeps_uncomputable_stored_criterion(
    session: Session, test_settings: Settings
) -> None:
    """A stored criterion with a null value keeps it, plus the stored reason."""
    analysis = HybridAnalysisService(session, test_settings).analyze(NO_ASPECTS_PROD)
    assert len(analysis.risk_indicators) == 5
    price = next(item for item in analysis.risk_indicators if item.name == "price_position")
    assert price.value is None
    assert price.raw_value is None
    assert price.missing_reason == "price_missing"
    assert price.direction == IndicatorDirection.COST


def test_hybrid_service_falls_back_when_stored_strength_is_null(
    session: Session, test_settings: Settings
) -> None:
    """A null stored composite is recomputed from the rows, never reported as 0.0."""
    analysis = HybridAnalysisService(session, test_settings).analyze(NULL_STRENGTH_PROD)
    assert analysis.evidence.evidence_strength is not None
    assert analysis.evidence_strength == analysis.evidence.evidence_strength
    assert any("null composite strength" in w for w in analysis.warnings)
    # The stored rating_agreement detail still travels in details.
    assert "rating_agreement" in {item.name for item in analysis.evidence.details}


def test_hybrid_service_details_never_shadow_fixed_names(
    session: Session, test_settings: Settings
) -> None:
    """The stored keys mapped onto fixed fields never reappear in details."""
    analysis = HybridAnalysisService(session, test_settings).analyze(SUBJECT)
    detail_names = {item.name for item in analysis.evidence.details}
    assert "verified_purchase_ratio" not in detail_names
    assert "helpfulness_coverage" not in detail_names
    assert not detail_names & {"verified_purchase", "helpfulness_ratio", "review_depth"}


def test_hybrid_service_is_deterministic(
    session: Session, test_settings: Settings
) -> None:
    """Multiple calls on the same ASIN yield byte-identical serialized JSON."""
    service = HybridAnalysisService(session, test_settings)
    call1 = service.analyze(SUBJECT).model_dump_json()
    call2 = service.analyze(SUBJECT).model_dump_json()
    assert call1 == call2


@pytest.mark.parametrize(
    ("classification", "net_sentiment", "expected"),
    [
        ("pain_point", -0.5, AspectPolarity.NEGATIVE),
        ("pain_point", 0.0, AspectPolarity.NEGATIVE),
        ("strength", 0.8, AspectPolarity.POSITIVE),
        ("strength", 0.0, AspectPolarity.POSITIVE),
        ("neutral", 0.4, AspectPolarity.POSITIVE),
        ("neutral", -0.4, AspectPolarity.NEGATIVE),
        ("neutral", 0.0, AspectPolarity.MIXED),
    ],
)
def test_hybrid_service_aspect_polarity_mapping(
    classification: str,
    net_sentiment: float,
    expected: AspectPolarity,
) -> None:
    """The stored classification wins; a neutral aspect follows net sentiment.

    Only the three schema polarities are ever produced, and a neutral aspect with
    exactly zero net sentiment maps to ``mixed`` (the schema has no ``neutral``
    polarity).
    """
    row = ProductAspect(
        asin="B0POLARITY",
        aspect="test_aspect",
        mention_count=10,
        negative_count=5,
        neutral_count=0,
        positive_count=5,
        negative_share=0.5,
        neutral_share=0.0,
        positive_share=0.5,
        net_sentiment=net_sentiment,
        product_labeled_reviews=20,
        product_negative_share=0.25,
        negativity_lift=None,
        classification=classification,
        examples=["example text"],
        computed_at=datetime(2026, 10, 11),
    )

    insights, reason = _hybrid_aspects([row])
    assert reason is None
    assert len(insights) == 1
    assert insights[0].polarity == expected


def test_hybrid_service_raises_on_unknown_product(
    session: Session, test_settings: Settings
) -> None:
    """Unknown ASIN raises ProductNotFoundError."""
    service = HybridAnalysisService(session, test_settings)
    with pytest.raises(ProductNotFoundError):
        service.analyze("NONEXISTENT_ASIN")

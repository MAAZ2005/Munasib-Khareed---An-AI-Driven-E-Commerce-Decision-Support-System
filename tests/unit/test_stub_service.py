"""Tests for the stub analysis service and its deterministic seeding.

Seeds a tiny in-memory serving cache (4 products, 8 reviews, 5 related edges)
plus a temporary ``meta.json``, so every number asserted below is real and the
tests never touch ``data/``. The real configs (criteria weights, fuzzy rules,
verdict thresholds, evidence weights) are deliberately *not* stubbed: the point
is to prove the stub reads names, weights, directions, rule ids and bands from
config rather than hard-coding them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime

import pytest
from sqlalchemy import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from munasib.core.config import Settings, get_settings
from munasib.core.constants import (
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABEL_POSITIVE,
    SPLIT_TRAIN,
    VERDICT_LABELS,
)
from munasib.core.exceptions import ProductNotFoundError
from munasib.db.models import Product, RelatedProduct, Review
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)
from munasib.pipeline.service import (
    STUB_PIPELINE_VERSION,
    STUB_WARNING_PREFIX,
    AnalysisService,
    StubAnalysisService,
    _rating_distribution,
    _seed_int,
    _sentiment_breakdown,
    _temporal_trend,
    load_criteria_spec,
    load_dataset_meta,
    load_fuzzy_spec,
    load_verdict_thresholds,
)

SUBJECT = "B000000001"
ALT_ONE = "B000000002"
ALT_TWO = "B000000003"
LONELY = "B000000004"
EDGES_ONLY_OUT = "B000000005"
SELFREF = "B000000007"


def _product_row(asin: str, *, n_reviews: int, mean_rating: float | None) -> dict:
    """Build one products-table insert payload.

    Args:
        asin: Product ASIN.
        n_reviews: Stored review count.
        mean_rating: Stored mean rating (None when unrated).

    Returns:
        Insert payload with proper NULLs for the missing brand/price/rank.
    """
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
        "description": None,
        "features": None,
        "n_reviews": n_reviews,
        "mean_rating": mean_rating,
        "first_review_time": datetime(2018, 1, 1),
        "last_review_time": datetime(2018, 3, 1),
    }


PRODUCTS = [
    _product_row(SUBJECT, n_reviews=4, mean_rating=3.25),
    _product_row(ALT_ONE, n_reviews=2, mean_rating=4.5),
    _product_row(ALT_TWO, n_reviews=2, mean_rating=2.0),
    _product_row(LONELY, n_reviews=1, mean_rating=5.0),
    _product_row(EDGES_ONLY_OUT, n_reviews=1, mean_rating=3.0),
    _product_row(SELFREF, n_reviews=1, mean_rating=4.0),
]


def _review_row(
    asin: str,
    review_id: str,
    when: datetime,
    *,
    rating: float | None,
    label: int | None,
    verified: bool | None,
    helpful: int,
    text_length: int,
) -> dict:
    """Build one reviews-table insert payload.

    Args:
        asin: Owning product ASIN.
        review_id: Stable 16-hex review key.
        when: Naive UTC review time.
        rating: Star rating (None means unrated).
        label: Weak label code or None.
        verified: Verified flag; None means the source did not say.
        helpful: Helpful-vote count.
        text_length: Cleaned text length.

    Returns:
        Insert payload.
    """
    return {
        "review_id": review_id,
        "asin": asin,
        "split": SPLIT_TRAIN,
        "reviewer_id": f"user-{review_id[-1]}",
        "rating": rating,
        "review_text_raw": "body text",
        "review_text_clean": "body text",
        "summary_raw": "Headline",
        "summary_clean": "headline",
        "has_text": text_length > 0,
        "text_length": text_length,
        "helpful_votes": helpful,
        "verified": verified,
        "review_time": when,
        "year": when.year,
        "month": when.month,
        "sentiment_label_weak": label,
    }


REVIEWS = [
    _review_row(
        SUBJECT, "00000000000000a1", datetime(2018, 1, 1), rating=5.0,
        label=SENTIMENT_LABEL_POSITIVE, verified=True, helpful=2, text_length=100,
    ),
    _review_row(
        SUBJECT, "00000000000000a2", datetime(2018, 2, 1), rating=1.0,
        label=SENTIMENT_LABEL_NEGATIVE, verified=False, helpful=0, text_length=50,
    ),
    _review_row(
        SUBJECT, "00000000000000a3", datetime(2018, 3, 1), rating=3.0,
        label=SENTIMENT_LABEL_NEUTRAL, verified=None, helpful=1, text_length=200,
    ),
    _review_row(
        SUBJECT, "00000000000000a4", datetime(2018, 3, 1), rating=4.0,
        label=SENTIMENT_LABEL_POSITIVE, verified=True, helpful=0, text_length=50,
    ),
    _review_row(
        ALT_ONE, "00000000000000b1", datetime(2018, 1, 5), rating=5.0,
        label=SENTIMENT_LABEL_POSITIVE, verified=True, helpful=1, text_length=200,
    ),
    _review_row(
        ALT_ONE, "00000000000000b2", datetime(2018, 2, 5), rating=4.0,
        label=SENTIMENT_LABEL_POSITIVE, verified=True, helpful=0, text_length=200,
    ),
    _review_row(
        ALT_TWO, "00000000000000c1", datetime(2018, 1, 7), rating=1.0,
        label=SENTIMENT_LABEL_NEGATIVE, verified=False, helpful=0, text_length=20,
    ),
    _review_row(
        ALT_TWO, "00000000000000c2", datetime(2018, 2, 7), rating=3.0,
        label=SENTIMENT_LABEL_NEUTRAL, verified=False, helpful=0, text_length=20,
    ),
    _review_row(
        LONELY, "00000000000000d1", datetime(2018, 3, 9), rating=5.0,
        label=SENTIMENT_LABEL_POSITIVE, verified=True, helpful=0, text_length=200,
    ),
    _review_row(
        EDGES_ONLY_OUT, "00000000000000e1", datetime(2018, 3, 8), rating=3.0,
        label=SENTIMENT_LABEL_NEUTRAL, verified=True, helpful=0, text_length=200,
    ),
    _review_row(
        SELFREF, "00000000000000g1", datetime(2018, 3, 10), rating=4.0,
        label=SENTIMENT_LABEL_POSITIVE, verified=True, helpful=0, text_length=200,
    ),
]

EDGES = [
    # Subject: 4 edges over 3 distinct related products (ALT_TWO appears under
    # both edge types), of which 2 are usable alternatives.
    {"asin": SUBJECT, "related_asin": ALT_ONE, "relation_type": "also_viewed", "in_subset": True},
    {"asin": SUBJECT, "related_asin": ALT_TWO, "relation_type": "also_bought", "in_subset": True},
    {"asin": SUBJECT, "related_asin": ALT_TWO, "relation_type": "also_viewed", "in_subset": True},
    {"asin": SUBJECT, "related_asin": "OUTSIDE999", "relation_type": "also_bought", "in_subset": False},
    # Edges exist but every target is outside the processed subset.
    {"asin": EDGES_ONLY_OUT, "related_asin": "OUTSIDE999", "relation_type": "also_viewed", "in_subset": False},
    # A product linked to itself must never be its own alternative: 2 edges but
    # only 1 distinct related product (ALT_ONE).
    {"asin": SELFREF, "related_asin": SELFREF, "relation_type": "also_bought", "in_subset": True},
    {"asin": SELFREF, "related_asin": ALT_ONE, "relation_type": "also_viewed", "in_subset": True},
]

META = {
    "category": "TestCategory",
    "built_at": "2026-10-11T10:45:51+00:00",
    "review_date_range": {"max": "2018-10-04T00:00:00+00:00"},
}


def _settings_for(tmp_path: object) -> Settings:
    """Build settings pointing at the temp meta.json directory.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Settings copy whose ``processed_data_dir`` maps the fixture's meta.json.
    """
    base = get_settings()
    return base.model_copy(
        update={"paths": base.paths.model_copy(update={"processed_data_dir": tmp_path})}
    )


@pytest.fixture()
def meta_json(tmp_path: object) -> object:
    """Write the fixture meta.json into the temp directory.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The temp directory holding ``meta.json``.
    """
    from pathlib import Path

    Path(tmp_path).mkdir(parents=True, exist_ok=True)  # type: ignore[arg-type]
    (Path(tmp_path) / "meta.json").write_text(json.dumps(META), encoding="utf-8")  # type: ignore[arg-type]
    return tmp_path


@pytest.fixture()
def engine() -> Iterator[Engine]:
    """Seeded in-memory engine.

    Yields:
        Engine with the fixture products, reviews and edges loaded.
    """
    engine = create_engine_from_settings(db_url="sqlite://")
    init_db(engine)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        session.execute(insert(Product), PRODUCTS)
        session.execute(insert(Review), REVIEWS)
        session.execute(insert(RelatedProduct), EDGES)
    yield engine
    engine.dispose()


@pytest.fixture()
def session(engine: Engine) -> Iterator[Session]:
    """Open session over the seeded engine.

    Args:
        engine: The seeded engine fixture.

    Yields:
        An open session, rolled back and closed afterwards.
    """
    session = create_session_factory(engine)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def service(session: Session, meta_json: object) -> StubAnalysisService:
    """A stub service over the fixture cache and fixture meta.json.

    Args:
        session: The seeded session fixture.
        meta_json: Directory holding the fixture meta.json.

    Returns:
        The stub service under test.
    """
    return StubAnalysisService(session, _settings_for(meta_json))


# ------------------------------------------------------------------------------
# Seeding
# ------------------------------------------------------------------------------


def test_seed_is_stable_and_hash_based() -> None:
    """The seed comes from sha256, so PYTHONHASHSEED cannot change it."""
    assert _seed_int(SUBJECT, "risk") == _seed_int(SUBJECT, "risk")
    assert _seed_int(SUBJECT, "risk") != _seed_int(ALT_ONE, "risk")
    assert _seed_int(SUBJECT, "risk") != _seed_int(SUBJECT, "closeness")


def test_analysis_is_deterministic(service: StubAnalysisService) -> None:
    """Two calls for the same ASIN are byte-identical."""
    first = service.analyze(SUBJECT).model_dump_json()
    second = service.analyze(SUBJECT).model_dump_json()
    assert first == second


def test_different_asins_get_different_placeholders(service: StubAnalysisService) -> None:
    """Different ASINS produce different placeholder values."""
    subject = service.analyze(SUBJECT)
    alternative = service.analyze(ALT_TWO)
    assert subject.decision is not None and alternative.decision is not None
    assert subject.decision.topsis.risk_score != alternative.decision.topsis.risk_score
    assert subject.decision.fuzzy.defuzzified != alternative.decision.fuzzy.defuzzified


def test_analysis_service_is_the_declared_contract(
    service: StubAnalysisService,
) -> None:
    """The stub satisfies the abstract contract the API depends on."""
    assert isinstance(service, AnalysisService)


# ------------------------------------------------------------------------------
# Real evidence numbers
# ------------------------------------------------------------------------------


def test_review_analysis_numbers_are_real(service: StubAnalysisService) -> None:
    """Counts, shares and monthly buckets come from the fixture rows."""
    analysis = service.analyze(SUBJECT)
    distribution = analysis.review_analysis.rating_distribution
    assert distribution.counts == {1: 1, 2: 0, 3: 1, 4: 1, 5: 1}
    assert distribution.total == 4
    assert distribution.mean_rating == pytest.approx(3.25)

    sentiment = analysis.review_analysis.sentiment
    assert (sentiment.negative_count, sentiment.neutral_count, sentiment.positive_count) == (1, 1, 2)
    assert sentiment.unlabelled_count == 0
    assert sentiment.negative_share == pytest.approx(0.25)
    assert sentiment.positive_share == pytest.approx(0.5)

    trend = analysis.review_analysis.temporal_trend
    assert [bucket.n_reviews for bucket in trend.buckets] == [1, 1, 2]
    assert trend.positive_share_slope == pytest.approx(-0.25)
    assert trend.n_reviews_used == 4


def test_evidence_indicators_match_config_weights(service: StubAnalysisService) -> None:
    """evidence_strength is the configured weighted mean of the three indicators."""
    analysis = service.analyze(SUBJECT)
    evidence = analysis.evidence
    assert evidence.verified_purchase.value == pytest.approx(0.5)
    assert evidence.verified_purchase.raw_value == pytest.approx(0.5)
    assert evidence.helpfulness.value == pytest.approx(0.5)
    assert evidence.helpfulness.raw_value == pytest.approx(0.75)
    assert evidence.review_depth.value == pytest.approx(0.5)
    assert evidence.review_depth.raw_value == pytest.approx(100.0)
    assert evidence.evidence_strength == pytest.approx(0.5)

    details = {detail.name: detail for detail in evidence.details}
    assert details["rating_distribution_entropy"].value == pytest.approx(0.8614)
    assert details["helpfulness_weighted_sentiment"].value == pytest.approx(0.8333)
    assert details["negative_severity_share"].value == pytest.approx(1.0)
    assert details["negative_severity_share"].raw_value == 1.0


def test_helpfulness_sentiment_narrative_matches_numbers(
    service: StubAnalysisService,
) -> None:
    """The 'above/below' sentence must compare like with like (0-1 vs 0-1)."""
    details = {
        detail.name: detail for detail in service.analyze(SUBJECT).evidence.details
    }
    indicator = details["helpfulness_weighted_sentiment"]
    # The fixture: labels [2,0,1,2] with votes [2,0,1,0] -> weighted mean 1.6667
    # (0.8333 on 0-1); unweighted mean 1.25 (0.625 on 0-1), so it IS above.
    assert indicator.value == pytest.approx(0.8333, abs=1e-3)
    assert "above the 0.6250 unweighted value" in indicator.description
    assert "read better than average" in indicator.description


def test_unweighted_reference_uses_the_same_scale(service: StubAnalysisService) -> None:
    """The unweighted reference is rescaled 0-1, never the raw 0-2 mean."""
    sentiment = service.analyze(SUBJECT).review_analysis.sentiment
    raw_mean = (sentiment.neutral_count + 2 * sentiment.positive_count) / sentiment.total
    description = next(
        detail.description
        for detail in service.analyze(SUBJECT).evidence.details
        if detail.name == "helpfulness_weighted_sentiment"
    )
    assert f"{raw_mean:.4f}" not in description, "the raw 0-2 mean must not be cited as the comparison"

def test_real_helpfulness_narrative_is_correct() -> None:
    """On real data the sentence names the right direction, not a hardcoded 'above'."""
    from pathlib import Path

    from munasib.db.session import create_engine_from_settings, create_session_factory

    db_path = Path(get_settings().paths.db_path)
    if not db_path.exists():
        pytest.skip("SQLite serving cache not built (python scripts/ingest_db.py)")
    engine = create_engine_from_settings(get_settings())
    try:
        with create_session_factory(engine)() as session:
            service = StubAnalysisService(session, get_settings())
            analysis = service.analyze("B00009V3UF")
    finally:
        engine.dispose()
    details = {detail.name: detail for detail in analysis.evidence.details}
    indicator = details["helpfulness_weighted_sentiment"]
    # Real numbers: weighted 0.5517 vs unweighted rescaled 0.4231 -> genuinely above.
    assert indicator.value == pytest.approx(0.5517, abs=1e-4)
    assert "above the 0.4231 unweighted value" in indicator.description
    assert "read better than average" in indicator.description


def test_evidence_descriptions_cite_real_numbers(service: StubAnalysisService) -> None:
    """Every indicator description quotes the count it came from."""
    evidence = service.analyze(SUBJECT).evidence
    assert "2 of 4" in evidence.verified_purchase.description
    assert "independent" not in evidence.verified_purchase.description
    assert "100.00 characters" in evidence.review_depth.description


def test_unrated_reviews_are_unlabelled_not_dropped(service: StubAnalysisService) -> None:
    """A review with no rating is counted as unlabelled, never fabricated."""
    session = service._session  # noqa: SLF001 - test reaches into the fixture
    session.add(
        Review(
            review_id="00000000000000f1",
            asin=SUBJECT,
            split=SPLIT_TRAIN,
            reviewer_id="user-f",
            rating=None,
            review_text_raw=None,
            review_text_clean=None,
            summary_raw=None,
            summary_clean=None,
            has_text=False,
            text_length=0,
            helpful_votes=0,
            verified=None,
            review_time=datetime(2018, 4, 1),
            year=2018,
            month=4,
            sentiment_label_weak=None,
        )
    )
    session.flush()
    analysis = service.analyze(SUBJECT)
    assert analysis.review_analysis.sentiment.unlabelled_count == 1
    assert analysis.review_analysis.sentiment.total == 5


# ------------------------------------------------------------------------------
# Config-driven placeholder blocks
# ------------------------------------------------------------------------------


def test_criteria_come_from_config(service: StubAnalysisService) -> None:
    """The six criteria carry the configured names, weights and directions."""
    analysis = service.analyze(SUBJECT)
    criteria = {criterion.name.value: criterion for criterion in analysis.decision.topsis.criteria}
    assert set(criteria) == {
        "product_quality",
        "evidence_strength",
        "rating_reliability",
        "negative_feedback_concentration",
        "price_position",
        "fuzzy_risk",
    }
    assert sum(criterion.weight for criterion in criteria.values()) == pytest.approx(1.0)
    assert criteria["product_quality"].direction == "benefit"
    assert criteria["fuzzy_risk"].direction == "cost"
    assert criteria["evidence_strength"].value == pytest.approx(analysis.evidence_strength)


def test_fuzzy_rule_ids_come_from_config(service: StubAnalysisService) -> None:
    """Fired rules are real configured rules; only their strengths are seeded."""
    spec = load_fuzzy_spec(service._settings)  # noqa: SLF001 - inspecting the config load
    configured = {int(rule["id"]) for rule in spec.rules}
    analysis = service.analyze(SUBJECT)
    fuzzy = analysis.decision.fuzzy
    assert fuzzy.defuzzification_method == "centroid"
    assert 1 <= len(fuzzy.fired_rules) <= 3
    for rule in fuzzy.fired_rules:
        assert rule.rule_id in configured
        assert rule.consequent in ("low", "moderate", "high", "critical")
        assert 0.0 <= rule.strength <= 1.0
    assert set(fuzzy.output_degrees) == set(spec.output_terms)
    assert set(fuzzy.input_memberships) == set(spec.inputs)


def test_stub_marks_itself_and_warns(service: StubAnalysisService) -> None:
    """The response never lets a placeholder pass as a result."""
    analysis = service.analyze(SUBJECT)
    assert analysis.pipeline_version == STUB_PIPELINE_VERSION
    assert analysis.warnings[0].startswith(STUB_WARNING_PREFIX)
    assert any("weak supervision" in warning for warning in analysis.warnings)


def test_risk_score_is_in_bounds_and_verdict_matches_bands(
    service: StubAnalysisService,
) -> None:
    """The placeholder risk stays 0-100 and the label follows the thresholds."""
    thresholds = load_verdict_thresholds(service._settings)  # noqa: SLF001
    for asin in (SUBJECT, ALT_ONE, ALT_TWO, LONELY):
        analysis = service.analyze(asin)
        assert analysis.verdict is not None
        assert 0.0 <= analysis.verdict.risk_score <= 100.0
        assert analysis.verdict.label.value in VERDICT_LABELS
        assert analysis.risk_score == analysis.verdict.risk_score
        if analysis.verdict.risk_score >= thresholds.risk_min_risk:
            assert analysis.verdict.label.value == "CONSUMER RISK"


def test_explanation_reports_no_surrogate(service: StubAnalysisService) -> None:
    """The surrogate fidelity says plainly that nothing was fitted."""
    explanation = service.analyze(SUBJECT).explanation
    assert explanation is not None
    assert explanation.surrogate_fidelity.n_training_points == 0
    assert explanation.surrogate_fidelity.meets_minimum is False
    assert explanation.reasons, "a reason must cite the real negative share"
    assert "4" in explanation.reasons[0].text


def test_data_basis_uses_dataset_not_today(service: StubAnalysisService) -> None:
    """reference_date and generated_at come from meta.json, never the clock."""
    meta = load_dataset_meta(service._settings)  # noqa: SLF001
    basis = service.analyze(SUBJECT).data_basis
    assert basis.category == "TestCategory"
    assert basis.n_reviews == 4
    assert basis.reference_date == meta.reference_date
    assert basis.date_range_start == datetime(2018, 1, 1, tzinfo=meta.reference_date.tzinfo)
    assert service.analyze(SUBJECT).generated_at == meta.built_at


def test_verdict_rationale_names_itself_as_a_stub(service: StubAnalysisService) -> None:
    """The rationale states the risk it summarises is seeded, not computed."""
    rationale = service.analyze(SUBJECT).verdict.rationale
    assert "stub" in rationale.lower()
    assert "placeholder risk" in rationale


def test_risk_indicators_cover_the_five_unit_criteria(
    service: StubAnalysisService,
) -> None:
    """risk_indicators exposes the five 0-1 criteria, not fuzzy_risk."""
    indicators = service.analyze(SUBJECT).risk_indicators
    names = [indicator.name for indicator in indicators]
    assert names == [
        "product_quality",
        "evidence_strength",
        "rating_reliability",
        "negative_feedback_concentration",
        "price_position",
    ]
    by_name = {indicator.name: indicator for indicator in indicators}
    assert by_name["evidence_strength"].value == pytest.approx(
        service.analyze(SUBJECT).evidence_strength
    )
    for indicator in indicators:
        assert 0.0 <= indicator.value <= 1.0
        assert str(indicator.direction) in ("benefit", "cost")


# ------------------------------------------------------------------------------
# Alternatives
# ------------------------------------------------------------------------------


def test_alternatives_count_distinct_related_products_not_edges(
    service: StubAnalysisService,
) -> None:
    """total_related is distinct related products; usable a subset of them."""
    alternatives = service.analyze(SUBJECT).alternatives
    # The fixture stores 4 edges for the subject, over 3 distinct targets.
    edges = service._related.get_related(SUBJECT, only_in_subset=False)  # noqa: SLF001
    assert len(edges) == 4
    assert alternatives.total_related == 3
    assert alternatives.usable == 2
    assert len(alternatives.items) == 2
    assert alternatives.empty_reason is None
    assert alternatives.usable <= alternatives.total_related


def test_target_under_both_relation_types_counts_once(
    service: StubAnalysisService,
) -> None:
    """ALT_TWO is linked by both edge types but counts once in both totals."""
    alternatives = service.analyze(SUBJECT).alternatives
    twice = [item for item in alternatives.items if item.summary.asin == ALT_TWO]
    assert len(twice) == 1
    assert sorted(twice[0].relation_types) == ["also_bought", "also_viewed"]
    # 3 distinct targets minus the doubled one being counted twice.
    assert alternatives.total_related == 3
    assert [item.summary.asin for item in alternatives.items] == [ALT_ONE, ALT_TWO]


def test_self_reference_is_excluded(service: StubAnalysisService) -> None:
    """A product linking to itself is never its own alternative."""
    edges = service._related.get_related(SELFREF, only_in_subset=False)  # noqa: SLF001
    assert len(edges) == 2  # one self edge + one real target

    alternatives = service.alternatives(SELFREF)
    assert alternatives.total_related == 1
    assert alternatives.usable == 1
    assert alternatives.empty_reason is None
    assert [item.summary.asin for item in alternatives.items] == [ALT_ONE]
    assert SELFREF not in {item.summary.asin for item in alternatives.items}


def test_usable_never_exceeds_total_related(service: StubAnalysisService) -> None:
    """Whatever the edge shape, usable is a subset of the distinct products."""
    for asin in (SUBJECT, ALT_ONE, ALT_TWO, LONELY, EDGES_ONLY_OUT, SELFREF):
        alternatives = service.alternatives(asin)
        assert alternatives.usable <= alternatives.total_related, asin
        assert len(alternatives.items) <= alternatives.usable, asin
        assert alternatives.total_related >= 0


def test_alternative_deltas_are_alternative_minus_subject(
    service: StubAnalysisService,
) -> None:
    """Deltas are alternative - subject in the documented direction."""
    analysis = service.analyze(SUBJECT)
    for item in analysis.alternatives.items:
        assert item.deltas.risk_score_delta == pytest.approx(
            item.risk_score - analysis.risk_score
        )
        assert item.deltas.mean_rating_delta is not None


def test_product_without_edges_returns_empty_with_reason(
    service: StubAnalysisService,
) -> None:
    """A product with no links at all gets the zero-target reason."""
    alternatives = service.alternatives(LONELY)
    assert alternatives.items == []
    assert alternatives.total_related == 0
    assert alternatives.usable == 0
    assert "0 distinct targets" in alternatives.empty_reason


def test_product_with_no_usable_edges_returns_empty_with_reason(
    service: StubAnalysisService,
) -> None:
    """Distinct related products outside the subset still yield a clear reason."""
    alternatives = service.alternatives(EDGES_ONLY_OUT)
    assert alternatives.items == []
    assert alternatives.total_related == 1
    assert alternatives.usable == 0
    assert "1 distinct related product(s) found" in alternatives.empty_reason
    assert "none of them are in the processed subset" in alternatives.empty_reason


def test_real_b00009v3uf_counts_distinct_related_products() -> None:
    """B00009V3UF reports 60 distinct related products, 11 of them usable.

    The serving cache holds 62 edges for this ASIN and 12 in-subset edges; the
    edge counts must not leak into either reported number. Skipped when the
    SQLite cache has not been built.
    """
    from pathlib import Path

    from munasib.db.session import create_engine_from_settings, create_session_factory

    db_path = Path(get_settings().paths.db_path)
    if not db_path.exists():
        pytest.skip("SQLite serving cache not built (python scripts/ingest_db.py)")
    engine = create_engine_from_settings(get_settings())
    try:
        with create_session_factory(engine)() as session:
            alternatives = StubAnalysisService(session, get_settings()).alternatives(
                "B00009V3UF"
            )
    finally:
        engine.dispose()
    assert alternatives.total_related == 60
    assert alternatives.usable == 11
    assert alternatives.usable <= alternatives.total_related
    assert len(alternatives.items) <= alternatives.usable
    assert "B00009V3UF" not in {item.summary.asin for item in alternatives.items}


def test_alternatives_are_deterministic(service: StubAnalysisService) -> None:
    """Repeated alternatives calls are identical."""
    assert service.alternatives(SUBJECT).model_dump_json() == service.alternatives(
        SUBJECT
    ).model_dump_json()


def test_unknown_asin_raises_domain_error(service: StubAnalysisService) -> None:
    """Both methods raise the domain error, which the API maps to 404."""
    with pytest.raises(ProductNotFoundError):
        service.analyze("MISSING1")
    with pytest.raises(ProductNotFoundError):
        service.alternatives("MISSING1")


# ------------------------------------------------------------------------------
# Config loading failures
# ------------------------------------------------------------------------------


def test_missing_meta_json_is_a_data_error(session: Session, tmp_path: object) -> None:
    """A missing meta.json is an explicit DataError, not a fabricated date."""
    from munasib.core.exceptions import DataError

    with pytest.raises(DataError):
        StubAnalysisService(session, _settings_for(tmp_path))


def test_criteria_weights_must_sum_to_one(session: Session, meta_json: object) -> None:
    """A weights file that does not sum to 1 is rejected, not silently used."""
    from pathlib import Path

    from munasib.core.exceptions import DataError

    broken_file = Path(str(meta_json)) / "criteria_weights.yaml"  # type: ignore[arg-type]
    broken_file.write_text(
        "criteria:\n"
        "  product_quality:\n    weight: 0.30\n    type: benefit\n"
        "  evidence_strength:\n    weight: 0.30\n    type: benefit\n"
        "  rating_reliability:\n    weight: 0.30\n    type: benefit\n"
        "  negative_feedback_concentration:\n    weight: 0.10\n    type: cost\n"
        "  price_position:\n    weight: 0.10\n    type: cost\n"
        "  fuzzy_risk:\n    weight: 0.10\n    type: cost\n"
        "validation:\n  sum_tolerance: 0.001\n",
        encoding="utf-8",
    )
    base = _settings_for(meta_json)
    broken = base.model_copy(
        update={"decision": base.decision.model_copy(update={"criteria_weights_file": broken_file})}
    )
    with pytest.raises(DataError, match="sum to 1.0"):
        StubAnalysisService(session, broken)


def test_criteria_missing_a_name_is_rejected(session: Session, meta_json: object) -> None:
    """A criterion entry without a name/weight/direction is rejected."""
    from pathlib import Path

    from munasib.core.exceptions import DataError

    broken_file = Path(str(meta_json)) / "criteria_weights.yaml"  # type: ignore[arg-type]
    broken_file.write_text(
        "criteria:\n"
        "  product_quality:\n    weight: 0.25\n    type: benefit\n"
        "validation:\n  sum_tolerance: 0.001\n",
        encoding="utf-8",
    )
    base = _settings_for(meta_json)
    broken = base.model_copy(
        update={"decision": base.decision.model_copy(update={"criteria_weights_file": broken_file})}
    )
    with pytest.raises(DataError):
        StubAnalysisService(session, broken)


def test_load_criteria_spec_matches_schema_enum() -> None:
    """The config and the schema enum declare the same six criteria."""
    spec = load_criteria_spec(get_settings())
    from munasib.api.schemas.decision import CRITERION_NAMES

    assert tuple(criterion.name for criterion in spec.criteria) == CRITERION_NAMES


# ------------------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------------------


def test_sentiment_breakdown_uses_shared_codes(session: Session) -> None:
    """The breakdown maps codes through the shared constants, not bare ints."""
    rows = [
        _review_row(SUBJECT, "0000000000000101", datetime(2018, 1, 1), rating=1.0,
                    label=SENTIMENT_LABEL_NEGATIVE, verified=True, helpful=0, text_length=10),
        _review_row(SUBJECT, "0000000000000102", datetime(2018, 1, 2), rating=5.0,
                    label=SENTIMENT_LABEL_POSITIVE, verified=True, helpful=0, text_length=10),
    ]
    breakdown = _sentiment_breakdown(_FakeReviews(rows))
    assert breakdown.negative_count == 1
    assert breakdown.positive_count == 1
    assert breakdown.total == 2


def test_temporal_trend_needs_two_periods(session: Session) -> None:
    """A single period has no slope, which stays None rather than 0."""
    rows = [
        _review_row(SUBJECT, "0000000000000201", datetime(2018, 1, 1), rating=5.0,
                    label=SENTIMENT_LABEL_POSITIVE, verified=True, helpful=0, text_length=10),
        _review_row(SUBJECT, "0000000000000202", datetime(2018, 1, 2), rating=5.0,
                    label=SENTIMENT_LABEL_POSITIVE, verified=True, helpful=0, text_length=10),
    ]
    trend = _temporal_trend(_FakeReviews(rows))
    assert len(trend.buckets) == 1
    assert trend.positive_share_slope is None


def test_rating_distribution_handles_no_reviews(session: Session) -> None:
    """An empty review list yields zeros, never a division error."""
    distribution = _rating_distribution([])
    assert distribution.total == 0
    assert distribution.counts == {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    assert distribution.mean_rating is None


class _FakeReviews:
    """Minimal sequence wrapper so pure helpers can take plain dict rows."""

    def __init__(self, rows: list[dict]) -> None:
        """Store the rows.

        Args:
            rows: Insert-payload dicts shaped like the reviews table.
        """
        self._rows = rows

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Yield lightweight attribute holders for each row.

        Returns:
            Iterator over row objects exposing the fields the helpers read.
        """
        return (_Row(row) for row in self._rows)

    def __len__(self) -> int:
        """Return the number of rows.

        Returns:
            The row count.
        """
        return len(self._rows)


class _Row:
    """Attribute view over one review row dict."""

    def __init__(self, row: dict) -> None:
        """Store the row.

        Args:
            row: Insert-payload dict.
        """
        self._row = row

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        """Proxy attribute access to the underlying dict.

        Args:
            name: Field name.

        Returns:
            The stored value.

        Raises:
            AttributeError: If the dict has no such key.
        """
        try:
            return self._row[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

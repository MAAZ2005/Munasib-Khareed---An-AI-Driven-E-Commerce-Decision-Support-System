"""Tests for the API contract schemas (``src/munasib/api/schemas``).

Covers: every declared example parses; out-of-range values are rejected at the
boundary (risk scores, indicator values, unknown verdict labels and weak-label
codes); the analysis example round-trips through JSON; the empty-alternatives
case validates; every JSON block in ``docs/api.md`` validates against the schema
named by its marker comment; and - when the SQLite serving cache exists - a
schema is built from a real ORM product row and the example's data-derived
numbers are cross-checked against the database (skipped when the cache is
missing). Hermetic apart from that optional check: no network, no GPU.
"""

from __future__ import annotations

import importlib.util
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from munasib.api import schemas
from munasib.api.schemas import (
    ANALYSIS_EXAMPLE,
    AlternativesResult,
    AnalysisResponse,
    CategoryMetaResponse,
    ErrorResponse,
    EvidenceIndicator,
    EvidenceIndicators,
    FuzzyAssessment,
    HealthResponse,
    Page,
    ProductDetail,
    ProductPage,
    RatingDistribution,
    ReviewPage,
    ReviewSnippet,
    SentimentBreakdown,
    SentimentLabel,
    TopsisCriterion,
    TopsisResult,
    VerdictLabel,
)
from munasib.api.schemas.alternatives import ALTERNATIVES_EMPTY_EXAMPLE
from munasib.api.schemas.reviews import SENTIMENT_LABEL_CODE_FOR_NAME

DOCS_API = Path(__file__).resolve().parents[2] / "docs" / "api.md"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_schemas.py"

#: Models whose declared example this suite validates.
EXAMPLE_MODELS = (
    AnalysisResponse,
    ErrorResponse,
    HealthResponse,
    CategoryMetaResponse,
    ProductPage,
    ProductDetail,
    ReviewPage,
    AlternativesResult,
)


def _module_example(model: type) -> dict:
    """Return a model's declared ``json_schema_extra`` example.

    Args:
        model: Schema class.

    Returns:
        The example payload as a dict.
    """
    extra = model.model_config.get("json_schema_extra")
    assert isinstance(extra, dict), f"{model.__name__} declares no json_schema_extra"
    example = extra["example"]
    assert isinstance(example, dict), f"{model.__name__} example must be an object"
    return example


# ------------------------------------------------------------------------------
# Valid examples parse
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("model", EXAMPLE_MODELS, ids=lambda model: model.__name__)
def test_declared_examples_parse(model: type) -> None:
    """Every top-level response's declared example is accepted by its schema."""
    parsed = model.model_validate(_module_example(model))
    assert parsed is not None


def test_example_analysis_has_real_data_basis() -> None:
    """The analysis example is anchored on the real dataset end date."""
    response = AnalysisResponse.model_validate(ANALYSIS_EXAMPLE)
    assert response.data_basis.reference_date == datetime(2018, 10, 4, tzinfo=UTC)
    assert response.data_basis.category == "Appliances"
    assert response.data_basis.n_reviews == response.product.n_reviews
    assert response.product.first_review_time <= response.data_basis.date_range_start


def test_analysis_example_round_trips_through_json() -> None:
    """The analysis example survives JSON encode/decode unchanged."""
    response = AnalysisResponse.model_validate(ANALYSIS_EXAMPLE)
    dumped = response.model_dump_json()
    again = AnalysisResponse.model_validate_json(dumped)
    assert again.model_dump() == response.model_dump()
    assert again.verdict is not None and again.verdict.label == VerdictLabel.CONSIDER_WITH_CAUTION


def test_examples_serialise_to_json_without_error() -> None:
    """Every example must be JSON-serialisable (the docs and the frontend rely on it)."""
    for model in EXAMPLE_MODELS:
        payload = model.model_validate(_module_example(model)).model_dump(mode="json")
        assert isinstance(payload, dict)


# ------------------------------------------------------------------------------
# Bounds enforcement
# ------------------------------------------------------------------------------


def _minimal_analysis(**overrides: object) -> dict:
    """Build a minimal valid AnalysisResponse payload for mutation tests.

    Args:
        overrides: Field overrides to apply.

    Returns:
        The payload dict.
    """
    payload = dict(ANALYSIS_EXAMPLE)
    payload.update(overrides)
    return payload


def test_risk_score_above_hundred_is_rejected() -> None:
    """Risk scores are 0-100; 150 is not clamped, it is rejected."""
    payload = _minimal_analysis(risk_score=150.0)
    with pytest.raises(ValidationError) as excinfo:
        AnalysisResponse.model_validate(payload)
    assert "risk_score" in str(excinfo.value)


def test_verdict_risk_score_above_hundred_is_rejected() -> None:
    """The verdict block enforces the same 0-100 bound."""
    with pytest.raises(ValidationError):
        schemas.ConsumerVerdict(
            label=VerdictLabel.CONSUMER_RISK,
            risk_score=150.0,
            evidence_strength=0.5,
            major_concerns=[],
            rationale="x",
        )


def test_unknown_verdict_label_is_rejected() -> None:
    """Only the three locked verdict labels exist."""
    with pytest.raises(ValidationError):
        schemas.ConsumerVerdict(
            label="BUY IT",
            risk_score=10.0,
            evidence_strength=0.5,
            major_concerns=[],
            rationale="x",
        )


def test_verdict_labels_are_exactly_three_locked_labels() -> None:
    """The enum mirrors the locked constants, nothing more."""
    assert [label.value for label in VerdictLabel] == [
        "CONSIDER",
        "CONSIDER WITH CAUTION",
        "CONSUMER RISK",
    ]


@pytest.mark.parametrize("bad_value", [-0.1, 1.2])
def test_indicator_outside_unit_interval_is_rejected(bad_value: float) -> None:
    """Indicator values are 0-1 and neither negative nor above one."""
    with pytest.raises(ValidationError):
        EvidenceIndicator(
            name="verified_purchase",
            value=bad_value,
            raw_value=bad_value,
            direction="benefit",
            description="x",
        )


def test_indicator_unit_bounds_are_inclusive() -> None:
    """0.0 and 1.0 are valid indicator values."""
    for value in (0.0, 1.0):
        indicator = EvidenceIndicator(
            name="verified_purchase",
            value=value,
            raw_value=value,
            direction="cost",
            description="x",
        )
        assert indicator.value == value


def test_evidence_strength_above_one_is_rejected() -> None:
    """Evidence strength is a unit-interval composite."""
    with pytest.raises(ValidationError):
        EvidenceIndicators(
            verified_purchase={
                "name": "verified_purchase",
                "value": 0.5,
                "direction": "benefit",
                "description": "x",
            },
            helpfulness={
                "name": "helpfulness_ratio",
                "value": 0.5,
                "direction": "benefit",
                "description": "x",
            },
            review_depth={
                "name": "review_depth",
                "value": 0.5,
                "direction": "benefit",
                "description": "x",
            },
            evidence_strength=1.2,
        )


def test_rating_distribution_requires_every_star() -> None:
    """Dropping a star from the histogram is a validation error, not a silent zero."""
    with pytest.raises(ValidationError):
        RatingDistribution(
            counts={"1": 1, "2": 2, "3": 3, "4": 4},
            shares={"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4},
            mean_rating=3.0,
            total=10,
        )


def test_rating_distribution_rejects_unknown_star() -> None:
    """A star outside 1-5 is rejected."""
    with pytest.raises(ValidationError):
        RatingDistribution(
            counts={"1": 1, "2": 2, "3": 3, "4": 4, "6": 5},
            shares={"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4, "6": 0.5},
            mean_rating=3.0,
            total=15,
        )


def test_fuzzy_output_term_must_be_configured() -> None:
    """Fuzzy output degrees are keyed by the configured output terms only."""
    with pytest.raises(ValidationError):
        FuzzyAssessment(
            input_memberships={"neg": {"term_degrees": {"low": 0.1}}},
            fired_rules=[],
            output_degrees={"dangerous": 1.0},
            defuzzified=10.0,
            defuzzification_method="centroid",
        )


def test_topsis_weights_must_sum_to_one() -> None:
    """A TOPSIS problem whose criteria weights do not sum to 1 is rejected."""

    def criterion(weight: float) -> TopsisCriterion:
        return TopsisCriterion(
            name=f"criterion_{weight}",
            description="x",
            weight=weight,
            direction="benefit",
            value=0.5,
            normalized_value=0.5,
            contribution=0.1,
        )

    with pytest.raises(ValidationError):
        TopsisResult(
            criteria=[criterion(0.3), criterion(0.3), criterion(0.3), criterion(0.3)],
            closeness=0.5,
            rank=1,
            ranked_count=4,
            peer_count=3,
            percentile=100.0,
            risk_score=10.0,
        )


def test_unknown_sentiment_label_code_is_rejected() -> None:
    """The API never accepts a label code outside the locked vocabulary."""
    with pytest.raises(ValidationError):
        ReviewSnippet(review_id="x", asin="y", review_time="2018-01-01T00:00:00Z", sentiment_label=7)


def test_sentiment_label_code_maps_to_shared_constant() -> None:
    """The enum values are the shared names, keyed by the shared codes."""
    assert SentimentLabel.from_code(0) is SentimentLabel.NEGATIVE
    assert SentimentLabel.from_code(1) is SentimentLabel.NEUTRAL
    assert SentimentLabel.from_code(2) is SentimentLabel.POSITIVE
    assert SentimentLabel.from_code(None) is None
    assert SENTIMENT_LABEL_CODE_FOR_NAME[SentimentLabel.NEGATIVE] == 0
    assert SENTIMENT_LABEL_CODE_FOR_NAME[SentimentLabel.POSITIVE] == 2


def test_page_requires_pagination_fields() -> None:
    """A page without total/limit/offset is invalid."""
    with pytest.raises(ValidationError):
        Page(items=[])  # type: ignore[call-arg]

    parsed = ProductPage.model_validate({"total": 0, "limit": 20, "offset": 0, "items": []})
    assert parsed.items == []


def test_utc_datetime_assumes_naive_input_is_utc() -> None:
    """Naive timestamps are interpreted as UTC, never shifted."""
    parsed = CategoryMetaResponse.model_validate(
        {
            "category": "Appliances",
            "n_products": 1,
            "n_reviews": 1,
            "n_related_edges": 0,
            "n_related_edges_in_subset": 0,
            "date_range_start": "2001-11-08T00:00:00",
            "date_range_end": "2018-10-04T00:00:00+00:00",
            "reference_date": "2018-10-04T00:00:00Z",
            "split_sizes": {"train": {"products": 1, "reviews": 1}},
        }
    )
    assert parsed.date_range_start == datetime(2001, 11, 8, tzinfo=UTC)
    assert parsed.date_range_end == parsed.reference_date


def test_unknown_split_name_in_split_sizes_is_rejected() -> None:
    """Split keys must come from the shared split vocabulary."""
    with pytest.raises(ValidationError):
        CategoryMetaResponse.model_validate(
            {
                "category": "Appliances",
                "n_products": 1,
                "n_reviews": 1,
                "n_related_edges": 0,
                "n_related_edges_in_subset": 0,
                "date_range_start": "2001-11-08T00:00:00Z",
                "date_range_end": "2018-10-04T00:00:00Z",
                "reference_date": "2018-10-04T00:00:00Z",
                "split_sizes": {"dev": {"products": 1, "reviews": 1}},
            }
        )


# ------------------------------------------------------------------------------
# Alternatives
# ------------------------------------------------------------------------------


def test_empty_alternatives_validate_with_a_reason() -> None:
    """An empty alternatives list is valid when it explains itself."""
    result = AlternativesResult.model_validate(ALTERNATIVES_EMPTY_EXAMPLE)
    assert result.items == []
    assert "no related-product edges" in (result.empty_reason or "")


def test_empty_alternatives_require_a_reason() -> None:
    """Silence about why there are no alternatives is not acceptable."""
    with pytest.raises(ValidationError):
        AlternativesResult.model_validate(
            {"items": [], "empty_reason": None, "total_related": 0, "usable": 0}
        )


def test_non_empty_alternatives_must_not_carry_a_reason() -> None:
    """A reason alongside results would be contradictory."""
    payload = dict(ANALYSIS_EXAMPLE["alternatives"])  # type: ignore[index]
    payload["empty_reason"] = "no alternatives"
    with pytest.raises(ValidationError):
        AlternativesResult.model_validate(payload)


def test_usable_cannot_exceed_total_related() -> None:
    """Sanity on the two edge counts."""
    with pytest.raises(ValidationError):
        AlternativesResult.model_validate(
            {
                "items": [],
                "empty_reason": "none",
                "total_related": 1,
                "usable": 2,
            }
        )


# ------------------------------------------------------------------------------
# docs/api.md examples are generated from the schemas
# ------------------------------------------------------------------------------


def _doc_blocks() -> list[tuple[str, str]]:
    """Extract every ``(schema_name, json_body)`` block from docs/api.md.

    Returns:
        One tuple per marker comment introduced JSON block.
    """
    text = DOCS_API.read_text(encoding="utf-8")
    pattern = re.compile(r"<!--\s*schema:\s*(\w+)\s*-->\s*```json\s*(?P<body>.*?)```", re.DOTALL)
    return [(match.group(1), match.group("body")) for match in pattern.finditer(text)]


def test_docs_api_md_exists_and_has_examples() -> None:
    """The generated doc exists and contains marked-up examples."""
    assert DOCS_API.exists(), "run python scripts/export_schemas.py"
    blocks = _doc_blocks()
    assert len(blocks) >= 8, f"expected examples for every endpoint, found {len(blocks)}"
    names = {name for name, _ in blocks}
    assert {
        "HealthResponse",
        "CategoryMetaResponse",
        "ProductPage",
        "ProductDetail",
        "ReviewPage",
        "AnalysisResponse",
        "AlternativesResult",
        "ErrorResponse",
    } <= names


@pytest.mark.parametrize(
    "model_name",
    ["HealthResponse", "CategoryMetaResponse", "ProductPage", "ProductDetail", "ReviewPage",
     "AnalysisResponse", "AlternativesResult", "ErrorResponse"],
)
def test_docs_api_md_examples_validate(model_name: str) -> None:
    """Every documented example parses against the schema its marker names."""
    model = getattr(schemas, model_name)
    blocks = [body for name, body in _doc_blocks() if name == model_name]
    assert blocks, f"no documented example for {model_name}"
    for body in blocks:
        model.model_validate_json(body)


def test_docs_api_md_documents_every_documented_error_code() -> None:
    """The error table and payloads cover 404, 422 and 500."""
    text = DOCS_API.read_text(encoding="utf-8")
    for code in ("PRODUCT_NOT_FOUND", "VALIDATION_ERROR", "INTERNAL_ERROR"):
        assert code in text
    for status in ("404", "422", "500"):
        assert f"| {status} |" in text


def test_export_schemas_reports_current_docs() -> None:
    """Re-running the exporter with --check finds the committed outputs current."""
    spec = importlib.util.spec_from_file_location("munasib_export_schemas_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


# ------------------------------------------------------------------------------
# Building a schema from a real ORM row
# ------------------------------------------------------------------------------


def _orm_product_schema() -> ProductDetail | None:
    """Build a ProductDetail from a real products row when the cache exists.

    Returns:
        The schema, or None when the SQLite serving cache is missing.
    """
    from sqlalchemy import select

    from munasib.core.config import Settings, get_settings
    from munasib.db.models import Product
    from munasib.db.session import create_engine_from_settings, create_session_factory, init_db

    settings: Settings = get_settings()
    db_path = Path(settings.paths.db_path)
    if not db_path.exists():
        return None
    engine = create_engine_from_settings(settings)
    try:
        init_db(engine)
        with create_session_factory(engine)() as session:
            row = session.scalars(select(Product).where(Product.asin == ANALYSIS_EXAMPLE["asin"])).one()
            return ProductDetail.model_validate(row)
    finally:
        engine.dispose()


def test_product_detail_builds_from_orm_row() -> None:
    """A real products row becomes a ProductDetail without re-deriving values."""
    product = _orm_product_schema()
    if product is None:
        pytest.skip("SQLite serving cache not built (python scripts/ingest_db.py)")
    example_product = ANALYSIS_EXAMPLE["product"]
    assert product.asin == example_product["asin"]
    assert product.title == example_product["title"]
    assert product.brand == example_product["brand"]
    assert product.price == example_product["price"]
    assert product.n_reviews == example_product["n_reviews"]
    assert product.rank_value == example_product["rank_value"]
    assert product.mean_rating == pytest.approx(example_product["mean_rating"], abs=1e-4)


def _orm_review_schema() -> ReviewSnippet | None:
    """Build a ReviewSnippet from a real reviews row when the cache exists.

    Returns:
        The schema, or None when the SQLite serving cache is missing.
    """
    from sqlalchemy import select

    from munasib.core.config import Settings, get_settings
    from munasib.db.models import Review
    from munasib.db.session import create_engine_from_settings, create_session_factory, init_db

    settings: Settings = get_settings()
    db_path = Path(settings.paths.db_path)
    if not db_path.exists():
        return None
    engine = create_engine_from_settings(settings)
    try:
        init_db(engine)
        with create_session_factory(engine)() as session:
            row = session.scalars(
                select(Review).where(Review.asin == ANALYSIS_EXAMPLE["asin"]).limit(1)
            ).first()
            if row is None:
                return None
            return ReviewSnippet.model_validate(row)
    finally:
        engine.dispose()


def test_review_snippet_builds_from_orm_row() -> None:
    """A real reviews row becomes a ReviewSnippet; codes become label names."""
    review = _orm_review_schema()
    if review is None:
        pytest.skip("SQLite serving cache not built (python scripts/ingest_db.py)")
    assert review.asin == ANALYSIS_EXAMPLE["asin"]
    assert review.rating in (1, 2, 3, 4, 5)
    if review.sentiment_label is not None:
        assert review.sentiment_label in set(SentimentLabel)
        assert SENTIMENT_LABEL_CODE_FOR_NAME[review.sentiment_label] in (0, 1, 2)


def test_review_snippet_reads_orm_aliases_without_db() -> None:
    """The alias contract works on any object with the ORM's field names."""
    row = SimpleNamespace(
        review_id="abc",
        asin="B00009V3UF",
        reviewer_id="A1",
        rating=4.0,
        verified=None,
        helpful_votes=2,
        review_time=datetime(2017, 5, 4),
        review_text_raw="raw body",
        has_text=True,
        text_length=8,
        summary_clean="fine",
        sentiment_label_weak=2,
    )
    snippet = ReviewSnippet.model_validate(row)
    assert snippet.review_text == "raw body"
    assert snippet.sentiment_label is SentimentLabel.POSITIVE
    assert snippet.verified is None  # unknown stays unknown, never False
    assert snippet.review_time == datetime(2017, 5, 4, tzinfo=UTC)


def test_product_summary_from_attributes_requires_no_aliases() -> None:
    """ProductSummary maps straight onto the ORM row."""
    row = SimpleNamespace(
        asin="B00009V3UF",
        title="Range Kleen 8234 Gas Range Knobs",
        brand="Range Kleen",
        price=12.66,
        n_reviews=52,
        mean_rating=2.634615,
        split="train",
    )
    summary = schemas.ProductSummary.model_validate(row)
    assert summary.asin == "B00009V3UF"
    assert summary.mean_rating == 2.634615


def test_sentiment_breakdown_counts_are_consistent_in_example() -> None:
    """The example's sentiment counts add up to its total."""
    breakdown = SentimentBreakdown.model_validate(
        ANALYSIS_EXAMPLE["review_analysis"]["sentiment"]
    )
    assert breakdown.total == (
        breakdown.negative_count + breakdown.neutral_count + breakdown.positive_count + breakdown.unlabelled_count
    )
    assert (
        breakdown.negative_share + breakdown.neutral_share + breakdown.positive_share
    ) == pytest.approx(1.0, abs=1e-3)

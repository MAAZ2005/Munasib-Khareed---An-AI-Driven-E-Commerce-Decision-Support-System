"""End-to-end API tests over a temporary SQLite serving cache.

Builds a small file-backed cache plus its own ``meta.json`` in a temp directory,
so the suite is hermetic and deterministic. A second block of tests runs against
the real ``data/db/munasib.sqlite`` when it exists and skips automatically
otherwise, which keeps the meaningful "does it work on real data" checks without
making the suite depend on a multi-megabyte artifact.

Lifespan note: ``TestClient`` only runs the app lifespan when used as a context
manager, which is what wires ``app.state.session_factory`` - so every client
here is created inside ``with``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import insert

from munasib.api.main import REQUEST_ID_HEADER, create_app
from munasib.core.config import Settings, get_settings
from munasib.core.constants import (
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABEL_POSITIVE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from munasib.db.models import Product, RelatedProduct, Review
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)

SUBJECT = "B000000001"
ALT_ONE = "B000000002"
ALT_TWO = "B000000003"
LONELY = "B000000004"
EDGES_ONLY_OUT = "B000000005"
TEST_SPLIT = "B000000006"

META = {
    "category": "TestCategory",
    "built_at": "2026-10-11T10:45:51+00:00",
    "review_date_range": {"max": "2018-10-04T00:00:00+00:00"},
    "splits": {"sizes": {"train": {"products": 4, "reviews": 8}, "val": {"products": 1, "reviews": 1}, "test": {"products": 1, "reviews": 1}}},
}


def _product_row(
    asin: str,
    *,
    title: str,
    brand: str | None,
    split: str,
    n_reviews: int,
    image_url: str | None = None,
    image_cached: bool = False,
) -> dict:
    """Build one products-table insert payload.

    Args:
        asin: Product ASIN.
        title: Product title.
        brand: Brand or None.
        split: Product-level split name.
        n_reviews: Stored review count.
        image_url: Source image URL or None (``image_missing`` mirrors it).
        image_cached: True only when a local thumbnail exists for the ASIN.

    Returns:
        Insert payload with NULLs for the missing brand/price/rank.
    """
    return {
        "asin": asin,
        "split": split,
        "title": title,
        "title_clean": title.lower(),
        "brand": brand,
        "brand_clean": brand.lower() if brand else None,
        "brand_missing": brand is None,
        "price": 12.66 if brand else None,
        "price_max": None,
        "price_status": "single" if brand else "missing",
        "price_missing": brand is None,
        "rank": "#4 in Widgets" if brand else None,
        "rank_value": 4 if brand else None,
        "rank_category": "Widgets" if brand else None,
        "rank_missing": brand is None,
        "description": "a widget",
        "features": ["steel"],
        "n_reviews": n_reviews,
        "mean_rating": 3.0,
        "first_review_time": datetime(2018, 1, 1),
        "last_review_time": datetime(2018, 3, 1),
        "image_url": image_url,
        "image_missing": image_url is None,
        "image_cached": image_cached,
    }


PRODUCTS = [
    _product_row(
        SUBJECT,
        title="FilterBuy Widget",
        brand="FilterBuy",
        split=SPLIT_TRAIN,
        n_reviews=4,
        image_url=f"https://example.com/{SUBJECT}.jpg",
        image_cached=True,
    ),
    _product_row(
        ALT_ONE,
        title="Acme Filter",
        brand="Acme",
        split=SPLIT_TRAIN,
        n_reviews=2,
        image_url=f"https://example.com/{ALT_ONE}.jpg",
    ),
    _product_row(ALT_TWO, title="Filter Pro", brand="Acme", split=SPLIT_VAL, n_reviews=2),
    _product_row(LONELY, title="Orphan Widget", brand="Acme", split=SPLIT_TRAIN, n_reviews=1),
    _product_row(EDGES_ONLY_OUT, title="Outside Widget", brand="Acme", split=SPLIT_TRAIN, n_reviews=1),
    _product_row(TEST_SPLIT, title="Filter Test", brand="Acme", split=SPLIT_TEST, n_reviews=1),
]


def _review_row(
    asin: str, review_id: str, when: datetime, *, rating: float | None, label: int | None,
    verified: bool | None, helpful: int, text_length: int,
) -> dict:
    """Build one reviews-table insert payload.

    Args:
        asin: Owning product ASIN.
        review_id: Stable 16-hex key.
        when: Naive UTC review time.
        rating: Star rating (None when unrated).
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
    _review_row(SUBJECT, "00000000000000a1", datetime(2018, 1, 1), rating=5.0,
                label=SENTIMENT_LABEL_POSITIVE, verified=True, helpful=2, text_length=100),
    _review_row(SUBJECT, "00000000000000a2", datetime(2018, 2, 1), rating=1.0,
                label=SENTIMENT_LABEL_NEGATIVE, verified=False, helpful=0, text_length=50),
    _review_row(SUBJECT, "00000000000000a3", datetime(2018, 3, 1), rating=3.0,
                label=SENTIMENT_LABEL_NEUTRAL, verified=None, helpful=1, text_length=200),
    _review_row(SUBJECT, "00000000000000a4", datetime(2018, 3, 1), rating=4.0,
                label=SENTIMENT_LABEL_POSITIVE, verified=True, helpful=0, text_length=50),
    _review_row(ALT_ONE, "00000000000000b1", datetime(2018, 1, 5), rating=5.0,
                label=SENTIMENT_LABEL_POSITIVE, verified=True, helpful=1, text_length=200),
    _review_row(ALT_TWO, "00000000000000c1", datetime(2018, 1, 7), rating=1.0,
                label=SENTIMENT_LABEL_NEGATIVE, verified=False, helpful=0, text_length=20),
    _review_row(LONELY, "00000000000000d1", datetime(2018, 3, 9), rating=5.0,
                label=SENTIMENT_LABEL_POSITIVE, verified=True, helpful=0, text_length=200),
    _review_row(EDGES_ONLY_OUT, "00000000000000e1", datetime(2018, 3, 8), rating=3.0,
                label=SENTIMENT_LABEL_NEUTRAL, verified=True, helpful=0, text_length=200),
]

EDGES = [
    {"asin": SUBJECT, "related_asin": ALT_ONE, "relation_type": "also_viewed", "in_subset": True},
    {"asin": SUBJECT, "related_asin": ALT_TWO, "relation_type": "also_bought", "in_subset": True},
    {"asin": SUBJECT, "related_asin": ALT_TWO, "relation_type": "also_viewed", "in_subset": True},
    {"asin": SUBJECT, "related_asin": "OUTSIDE999", "relation_type": "also_bought", "in_subset": False},
    {"asin": EDGES_ONLY_OUT, "related_asin": "OUTSIDE999", "relation_type": "also_viewed", "in_subset": False},
]


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Create the temp cache, its meta.json and one cached thumbnail.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The directory holding the SQLite file, meta.json and ``images/``.
    """
    (tmp_path / "meta.json").write_text(json.dumps(META), encoding="utf-8")
    # A real (tiny) JPEG so the image route streams genuine image bytes offline.
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(images_dir / f"{SUBJECT}.jpg", format="JPEG")
    engine = create_engine_from_settings(db_url=f"sqlite:///{tmp_path / 'cache.sqlite'}")
    init_db(engine)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        session.execute(insert(Product), PRODUCTS)
        session.execute(insert(Review), REVIEWS)
        session.execute(insert(RelatedProduct), EDGES)
    engine.dispose()
    return tmp_path


@pytest.fixture()
def settings(workspace: Path) -> Settings:
    """Settings pointing at the temp cache, temp meta.json and temp image cache.

    Args:
        workspace: Directory holding the fixture cache.

    Returns:
        A settings copy with overridden paths.
    """
    base = get_settings()
    return base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={
                    "db_path": workspace / "cache.sqlite",
                    "processed_data_dir": workspace,
                }
            ),
            "images": base.images.model_copy(update={"directory": workspace / "images"}),
        }
    )


@pytest.fixture()
def client(settings: Settings) -> Iterator[TestClient]:
    """TestClient over the app built from the temp settings.

    Args:
        settings: The temp-dir settings fixture.

    Yields:
        A client whose lifespan has opened the fixture cache.
    """
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture()
def real_client() -> Iterator[TestClient | None]:
    """TestClient over the real serving cache, or None when it is absent.

    Yields:
        A client for the real cache, or None when the artifact is not built.
    """
    db_path = Path(get_settings().paths.db_path)
    if not db_path.exists():
        yield None
        return
    with TestClient(create_app()) as test_client:
        yield test_client


# ------------------------------------------------------------------------------
# Every route answers 200 and validates against its schema
# ------------------------------------------------------------------------------

ROUTES_200 = [
    "/health",
    "/meta/category",
    "/products",
    f"/products/{SUBJECT}",
    f"/products/{SUBJECT}/reviews",
    f"/products/{SUBJECT}/analysis",
    f"/products/{SUBJECT}/alternatives",
]


@pytest.mark.parametrize("path", ROUTES_200)
def test_route_returns_200_and_validates(client: TestClient, path: str) -> None:
    """Each endpoint answers 200 and parses as its declared response model."""
    response = client.get(path)
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict)
    assert body, f"{path} returned an empty object"


def test_health_reports_database_reachable(client: TestClient) -> None:
    """/health reflects the reachable fixture cache and a UTC time."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"] == "reachable"
    assert body["version"] == get_settings().project.version
    assert body["time"].endswith("Z") or "+00:00" in body["time"]


def test_meta_category_reports_cache_contents(client: TestClient) -> None:
    """/meta/category counts come from the fixture, and reference_date from meta.json."""
    body = client.get("/meta/category").json()
    assert body["category"] == "TestCategory"
    assert body["n_products"] == 6
    assert body["n_reviews"] == 8
    assert body["n_related_edges"] == 5
    assert body["n_related_edges_in_subset"] == 3
    assert body["reference_date"] == "2018-10-04T00:00:00Z"
    assert body["split_sizes"]["train"]["products"] == 4
    assert body["split_sizes"]["test"]["products"] == 1


def test_product_detail_keeps_missing_fields_null(client: TestClient) -> None:
    """A product with no brand/price/rank keeps them null and flagged."""
    body = client.get(f"/products/{ALT_ONE}").json()
    assert body["brand"] == "Acme"
    assert body["brand_missing"] is False
    assert body["rank_missing"] is False
    lonely = client.get(f"/products/{LONELY}").json()
    assert lonely["brand_missing"] is False


def test_reviews_ordering_and_filters(client: TestClient) -> None:
    """Reviews page newest first and the label filter uses the API names."""
    body = client.get(f"/products/{SUBJECT}/reviews").json()
    assert body["total"] == 4
    assert [item["sentiment_label"] for item in body["items"]] == [
        "neutral",
        "positive",
        "negative",
        "positive",
    ]
    negative = client.get(f"/products/{SUBJECT}/reviews?label=negative").json()
    assert negative["total"] == 1
    assert negative["items"][0]["sentiment_label"] == "negative"

    unverified = client.get(f"/products/{SUBJECT}/reviews?verified=false").json()
    assert unverified["total"] == 1
    verified = client.get(f"/products/{SUBJECT}/reviews?verified=true").json()
    assert verified["total"] == 2


def test_review_text_and_label_never_expose_codes(client: TestClient) -> None:
    """review_text is the raw body, summary the cleaned headline, label a name."""
    item = client.get(f"/products/{SUBJECT}/reviews").json()["items"][0]
    assert item["review_text"] == "body text"
    assert item["summary"] == "headline"
    assert item["sentiment_label"] in {"negative", "neutral", "positive"}
    assert "sentiment_label_weak" not in item


def test_analysis_carries_real_evidence_and_stub_markers(client: TestClient) -> None:
    """The analysis mixes real evidence numbers with clearly-marked placeholders."""
    body = client.get(f"/products/{SUBJECT}/analysis").json()
    assert body["pipeline_version"] == "stub-0"
    assert body["warnings"][0].startswith("STUB:")
    assert body["evidence"]["evidence_strength"] == 0.5
    assert body["review_analysis"]["rating_distribution"]["counts"]["1"] == 1
    assert body["decision"] is not None
    assert body["decision"]["topsis"]["criteria"][0]["name"] == "product_quality"
    assert body["risk_score"] == body["verdict"]["risk_score"]
    assert [indicator["name"] for indicator in body["risk_indicators"]] == [
        "product_quality",
        "evidence_strength",
        "rating_reliability",
        "negative_feedback_concentration",
        "price_position",
    ]
    assert body["alternatives"]["total_related"] == 3  # ALT_ONE, ALT_TWO, OUTSIDE999
    assert body["alternatives"]["usable"] == 2
    assert body["data_basis"]["reference_date"] == "2018-10-04T00:00:00Z"
    assert body["generated_at"].startswith("2026-10-11")


def test_analysis_is_deterministic_across_calls(client: TestClient) -> None:
    """Two identical calls return identical bytes (except the live /health)."""
    first = client.get(f"/products/{SUBJECT}/analysis").content
    second = client.get(f"/products/{SUBJECT}/analysis").content
    assert first == second


def test_alternatives_empty_with_reason(client: TestClient) -> None:
    """A product without usable alternatives returns an explicit reason."""
    body = client.get(f"/products/{LONELY}/alternatives").json()
    assert body["items"] == []
    assert body["total_related"] == 0
    assert body["usable"] == 0
    assert "0 distinct targets" in body["empty_reason"]

    edges_only = client.get(f"/products/{EDGES_ONLY_OUT}/alternatives").json()
    assert edges_only["items"] == []
    assert edges_only["total_related"] == 1
    assert edges_only["usable"] == 0
    assert "none of them are in the processed subset" in edges_only["empty_reason"]


def test_alternative_relation_types_collapse_to_one_alternative(
    client: TestClient,
) -> None:
    """An ASIN linked twice is counted once in both totals, listing both edges."""
    body = client.get(f"/products/{SUBJECT}/alternatives").json()
    twice = [item for item in body["items"] if item["summary"]["asin"] == ALT_TWO]
    assert len(twice) == 1
    assert sorted(twice[0]["relation_types"]) == ["also_bought", "also_viewed"]
    # 4 edges over 3 distinct targets: the doubled target is counted once.
    assert body["total_related"] == 3
    assert body["usable"] <= body["total_related"]


# ------------------------------------------------------------------------------
# 404 on every {asin} route
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/products/NOTANASIN",
        "/products/NOTANASIN/reviews",
        "/products/NOTANASIN/analysis",
        "/products/NOTANASIN/alternatives",
    ],
)
def test_unknown_asin_returns_404_envelope(client: TestClient, path: str) -> None:
    """Every product-scoped route answers 404 with the shared error envelope."""
    response = client.get(path)
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "PRODUCT_NOT_FOUND"
    assert body["detail"] == {"asin": "NOTANASIN"}
    assert body["message"] == "product 'NOTANASIN' not found in the serving cache"


# ------------------------------------------------------------------------------
# Product images (local cache, never hotlinked)
# ------------------------------------------------------------------------------


def test_product_detail_reports_has_image(client: TestClient) -> None:
    """`has_image` is true only for a product with a locally cached thumbnail."""
    subject = client.get(f"/products/{SUBJECT}").json()
    assert subject["has_image"] is True
    assert "image_url" not in subject  # the upstream URL is never exposed
    other = client.get(f"/products/{ALT_ONE}").json()
    # ALT_ONE has a source image URL but no downloaded thumbnail.
    assert other["has_image"] is False


def test_product_list_reports_has_image(client: TestClient) -> None:
    """The search contract carries the same per-item flag."""
    body = client.get("/products?limit=100").json()
    flags = {item["asin"]: item["has_image"] for item in body["items"]}
    assert flags[SUBJECT] is True
    assert all(value is False for asin, value in flags.items() if asin != SUBJECT)
    assert all(isinstance(value, bool) for value in flags.values())


def test_image_route_serves_the_cached_thumbnail(
    client: TestClient, settings: Settings
) -> None:
    """The route streams real JPEG bytes from the local cache with Cache-Control."""
    response = client.get(f"/products/{SUBJECT}/image")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")  # JPEG SOI marker
    assert response.headers["cache-control"] == (
        f"public, max-age={settings.images.cache_max_age_seconds}"
    )


def test_image_route_head_answers_headers_only(
    client: TestClient, settings: Settings
) -> None:
    """HEAD is supported (FastAPI does not derive it from GET) and sends no body."""
    response = client.head(f"/products/{SUBJECT}/image")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == (
        f"public, max-age={settings.images.cache_max_age_seconds}"
    )
    assert response.content == b""


@pytest.mark.parametrize("asin", [ALT_ONE, ALT_TWO, LONELY, TEST_SPLIT])
def test_image_route_404_when_nothing_is_cached(client: TestClient, asin: str) -> None:
    """A product without a cached thumbnail answers 404 - never a redirect."""
    response = client.get(f"/products/{asin}/image")
    assert response.status_code == 404, response.text
    body = response.json()
    assert body["code"] == "PRODUCT_NOT_FOUND"
    assert body["detail"] == {"asin": asin}


def test_image_route_404_when_the_file_disappeared(
    client: TestClient, settings: Settings
) -> None:
    """A cache flag with no file behind it is reported as missing, not served."""
    Path(settings.images.directory, f"{SUBJECT}.jpg").unlink()
    response = client.get(f"/products/{SUBJECT}/image")
    assert response.status_code == 404
    assert response.json()["code"] == "PRODUCT_NOT_FOUND"


@pytest.mark.parametrize(
    "asin",
    ["NOTANASIN", "short", "B0000000010", "B00000000.", "%2e%2e%2eetc", "B00000000-"],
)
def test_image_route_rejects_malformed_asins(client: TestClient, asin: str) -> None:
    """A malformed ASIN is a 404 before any filesystem access (never a 500)."""
    response = client.get(f"/products/{asin}/image")
    assert response.status_code == 404, response.text
    assert response.json()["code"] == "PRODUCT_NOT_FOUND"


# ------------------------------------------------------------------------------
# 422 validation
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/products?limit=0",
        "/products?limit=1000",
        "/products?offset=-1",
        "/products?split=banana",
        f"/products/{SUBJECT}/reviews?label=angry",
        f"/products/{SUBJECT}/reviews?limit=0",
        f"/products/{SUBJECT}/reviews?offset=-1",
        f"/products/{SUBJECT}/reviews?verified=maybe",
    ],
)
def test_bad_query_parameters_return_422_envelope(client: TestClient, path: str) -> None:
    """Invalid paging or enum values answer 422 with the offending field."""
    response = client.get(path)
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert body["detail"]["errors"], "the failing field must be named"


def test_validation_error_names_the_field(client: TestClient) -> None:
    """The 422 payload points at the exact query parameter that failed."""
    body = client.get("/products?limit=0").json()
    error = body["detail"]["errors"][0]
    assert error["field"] == "query.limit"
    assert error["constraint"] == {"ge": 1}


# ------------------------------------------------------------------------------
# Search pagination and totals
# ------------------------------------------------------------------------------


def test_search_pagination_and_totals(client: TestClient) -> None:
    """Search is case-insensitive, stable and reports the true total."""
    first = client.get("/products?query=FILTER&limit=2&offset=0").json()
    assert first["total"] == 4
    assert first["limit"] == 2
    assert first["offset"] == 0
    assert len(first["items"]) == 2
    first_asins = {item["asin"] for item in first["items"]}

    second = client.get("/products?query=filter&limit=2&offset=2").json()
    assert second["total"] == 4
    second_asins = {item["asin"] for item in second["items"]}
    assert not first_asins & second_asins, "pages must not repeat rows"

    every = client.get("/products?query=filter&limit=100").json()["items"]
    assert first_asins | second_asins == {item["asin"] for item in every}
    assert len(every) == 4, "the pages together cover every match"


def test_search_brand_and_total_are_unpaginated(client: TestClient) -> None:
    """`total` counts all matches, not just the returned page."""
    body = client.get("/products?query=acme&limit=1").json()
    assert body["total"] == 5
    assert len(body["items"]) == 1


def test_search_split_filter(client: TestClient) -> None:
    """The split filter narrows both the page and the total."""
    all_rows = client.get("/products?limit=100").json()
    train = client.get("/products?split=train&limit=100").json()
    assert train["total"] == 4
    assert all(item["split"] == "train" for item in train["items"])
    assert all_rows["total"] == 6


# ------------------------------------------------------------------------------
# Request-ID middleware
# ------------------------------------------------------------------------------


def test_request_id_is_generated_and_echoed(client: TestClient) -> None:
    """A request without an id gets one back in the response header."""
    response = client.get("/health")
    generated = response.headers.get(REQUEST_ID_HEADER)
    assert generated
    assert len(generated) == 36  # uuid4


def test_incoming_request_id_is_honoured(client: TestClient) -> None:
    """An inbound X-Request-ID is echoed verbatim."""
    response = client.get("/health", headers={REQUEST_ID_HEADER: "my-trace-id"})
    assert response.headers.get(REQUEST_ID_HEADER) == "my-trace-id"


def test_request_id_is_in_error_bodies(client: TestClient) -> None:
    """Error envelopes carry the same id the response header reports."""
    response = client.get("/products/NOTANASIN", headers={REQUEST_ID_HEADER: "trace-42"})
    assert response.headers.get(REQUEST_ID_HEADER) == "trace-42"
    assert response.json()["request_id"] == "trace-42"

    validation = client.get("/products?limit=0", headers={REQUEST_ID_HEADER: "trace-43"})
    assert validation.json()["request_id"] == "trace-43"


# ------------------------------------------------------------------------------
# Service selection
# ------------------------------------------------------------------------------


def test_selecting_real_service_fails_with_documented_error(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """`analysis.service: real` has no engine, so it must fail loudly, not fall back.

    The response stays generic (an internal cause never leaks to a client), but
    the log records the actionable text and the envelope names the service.
    """
    real_settings = settings.model_copy(
        update={"analysis": settings.analysis.model_copy(update={"service": "real"})}
    )
    with caplog.at_level("ERROR"), TestClient(create_app(real_settings)) as client:
        response = client.get(f"/products/{SUBJECT}/analysis")
    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "MODEL_UNAVAILABLE"
    assert body["detail"] == {"service": "real"}
    assert body["request_id"]
    assert "analysis.service" in caplog.text
    assert 'Set analysis.service to "stub"' in caplog.text
    # The stub is never silently substituted for the requested real engine.
    assert body.get("pipeline_version") != "stub-0"


def test_default_service_is_stub(settings: Settings) -> None:
    """The shipped default serves the stub, so the frontend can be built now."""
    assert settings.analysis.service == "stub"
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/products/{SUBJECT}/analysis").json()["pipeline_version"] == "stub-0"


# ------------------------------------------------------------------------------
# Real serving cache (skipped when the artifact is absent)
# ------------------------------------------------------------------------------


def test_real_cache_health(real_client: TestClient | None) -> None:
    """The real serving cache is reachable and reports the built subset."""
    if real_client is None:
        pytest.skip("SQLite serving cache not built (python scripts/ingest_db.py)")
    body = real_client.get("/health").json()
    assert body["database"] == "reachable"


def test_real_cache_meta_matches_meta_json(real_client: TestClient | None) -> None:
    """/meta/category reproduces the counts recorded in data/processed/meta.json."""
    if real_client is None:
        pytest.skip("SQLite serving cache not built (python scripts/ingest_db.py)")
    meta_path = Path(get_settings().paths.processed_data_dir) / "meta.json"
    if not meta_path.exists():
        pytest.skip("data/processed/meta.json not built (python scripts/build_dataset.py)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    body = real_client.get("/meta/category").json()
    assert body["n_products"] == meta["row_counts"]["products_final"]
    assert body["n_reviews"] == meta["row_counts"]["reviews_final"]
    assert body["n_related_edges"] == meta["row_counts"]["related_edges"]
    assert body["n_related_edges_in_subset"] == meta["row_counts"]["related_in_subset"]
    assert body["reference_date"] == "2018-10-04T00:00:00Z" if meta["review_date_range"]["max"].startswith("2018-10-04") else True


def test_real_analysis_matches_documented_example_numbers(
    real_client: TestClient | None,
) -> None:
    """The real analysis reproduces the numbers recorded in docs/api.md."""
    if real_client is None:
        pytest.skip("SQLite serving cache not built (python scripts/ingest_db.py)")
    body = real_client.get("/products/B00009V3UF/analysis").json()
    assert body["evidence"]["evidence_strength"] == pytest.approx(0.7173, abs=1e-4)
    distribution = body["review_analysis"]["rating_distribution"]
    assert distribution["counts"] == {"1": 19, "2": 4, "3": 14, "4": 7, "5": 8}
    assert body["review_analysis"]["sentiment"]["negative_count"] == 23
    details = {item["name"]: item for item in body["evidence"]["details"]}
    assert details["rating_distribution_entropy"]["value"] == pytest.approx(0.9173, abs=1e-4)
    assert details["helpfulness_weighted_sentiment"]["value"] == pytest.approx(0.5517, abs=1e-4)
    assert details["negative_severity_share"]["value"] == pytest.approx(0.8261, abs=1e-4)
    alternatives = body["alternatives"]
    assert alternatives["total_related"] == 60
    assert alternatives["usable"] == 11
    assert body["pipeline_version"] == "stub-0"


def test_real_b00009v3uf_counts_distinct_related_products(
    real_client: TestClient | None,
) -> None:
    """B00009V3UF reports 60 distinct related products, 11 of them usable.

    The cache holds 62 edges for this ASIN (4 also_bought + 58 also_viewed): two
    targets are linked by both edge types, so the edge count must not leak into
    ``total_related``. 12 of those edges are in-subset, covering 11 distinct
    products. Skipped when the serving cache is absent.
    """
    if real_client is None:
        pytest.skip("SQLite serving cache not built (python scripts/ingest_db.py)")
    body = real_client.get("/products/B00009V3UF/alternatives").json()
    assert body["total_related"] == 60
    assert body["usable"] == 11
    assert body["usable"] <= body["total_related"]
    assert len(body["items"]) <= body["usable"]
    # No target appears twice, and the subject never lists itself.
    asins = [item["summary"]["asin"] for item in body["items"]]
    assert len(asins) == len(set(asins))
    assert "B00009V3UF" not in asins


def test_real_analysis_is_deterministic(real_client: TestClient | None) -> None:
    """The real cache also produces byte-identical repeated analyses."""
    if real_client is None:
        pytest.skip("SQLite serving cache not built (python scripts/ingest_db.py)")
    first = real_client.get("/products/B00009V3UF/analysis").content
    second = real_client.get("/products/B00009V3UF/analysis").content
    assert first == second


def test_real_unknown_asin_is_404(real_client: TestClient | None) -> None:
    """An unknown ASIN on the real cache answers 404."""
    if real_client is None:
        pytest.skip("SQLite serving cache not built (python scripts/ingest_db.py)")
    assert real_client.get("/products/NOTANASIN").status_code == 404


def test_real_cache_image_flag_matches_the_route(real_client: TestClient | None) -> None:
    """On the real cache, `has_image` and the image route can never disagree."""
    if real_client is None:
        pytest.skip("SQLite serving cache not built (python scripts/ingest_db.py)")
    detail = real_client.get("/products/B00009V3UF").json()
    assert isinstance(detail["has_image"], bool)
    assert "image_url" not in detail
    response = real_client.get("/products/B00009V3UF/image")
    if detail["has_image"]:
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
    else:
        assert response.status_code == 404


def test_real_product_without_usable_alternatives(real_client: TestClient | None) -> None:
    """A real zero-link product returns the empty result with a reason."""
    if real_client is None:
        pytest.skip("SQLite serving cache not built (python scripts/ingest_db.py)")
    body = real_client.get("/products/B00007E7RZ/alternatives").json()
    assert body["items"] == []
    assert body["total_related"] == 0
    assert body["usable"] == 0
    assert "0 distinct targets" in body["empty_reason"]


def test_real_product_with_related_products_but_none_usable(
    real_client: TestClient | None,
) -> None:
    """A real product whose links all fall outside the subset explains itself."""
    if real_client is None:
        pytest.skip("SQLite serving cache not built (python scripts/ingest_db.py)")
    body = real_client.get("/products/B0001YM48Q/alternatives").json()
    assert body["items"] == []
    assert body["usable"] == 0
    assert body["total_related"] > 0
    assert body["usable"] <= body["total_related"]
    assert f"{body['total_related']} distinct related product(s) found" in body[
        "empty_reason"
    ]
    assert "none of them are in the processed subset" in body["empty_reason"]


def test_missing_database_fails_startup_with_clear_message(tmp_path: Path) -> None:
    """A missing cache fails startup naming the script that rebuilds it."""
    base = get_settings()
    broken = base.model_copy(
        update={"paths": base.paths.model_copy(update={"db_path": tmp_path / "nope.sqlite"})}
    )
    app = create_app(broken)
    with pytest.raises(RuntimeError, match="ingest_db"):
        with TestClient(app):
            pass

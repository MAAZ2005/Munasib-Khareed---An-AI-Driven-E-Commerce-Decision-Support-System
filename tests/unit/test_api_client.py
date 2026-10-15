"""Unit tests for the Munasib Khareed API client using httpx.MockTransport."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.api_client import (
    ApiClient,
    ApiNotFound,
    ApiServerError,
    ApiUnavailable,
    ApiValidationError,
)
from munasib.api.schemas import (
    AlternativesResult,
    AnalysisResponse,
    CategoryMetaResponse,
    HealthResponse,
    ProductDetail,
    ProductPage,
    ReviewPage,
)
from munasib.api.schemas.analysis import ANALYSIS_EXAMPLE


def _mock_transport(handler: Any) -> httpx.MockTransport:
    """Helper to create a MockTransport with the given handler."""
    return httpx.MockTransport(handler)


def test_health_success() -> None:
    """Verify health() returns a typed HealthResponse and extracts headers."""
    payload = {
        "status": "ok",
        "service": "Munasib Khareed",
        "version": "0.1.0",
        "database": "reachable",
        "time": "2026-10-15T10:00:00Z",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        assert request.method == "GET"
        return httpx.Response(
            200,
            json=payload,
            headers={"X-Request-ID": "req-health-1"},
        )

    client = ApiClient(transport=_mock_transport(handler))
    resp = client.health()
    assert isinstance(resp, HealthResponse)
    assert resp.status == "ok"
    assert resp.database == "reachable"
    assert resp.version == "0.1.0"


def test_get_category_meta_success() -> None:
    """Verify get_category_meta() deserializes into CategoryMetaResponse."""
    payload = {
        "category": "Appliances",
        "n_products": 100,
        "n_reviews": 500,
        "n_related_edges": 120,
        "n_related_edges_in_subset": 80,
        "date_range_start": "2015-01-01T00:00:00Z",
        "date_range_end": "2018-10-04T00:00:00Z",
        "reference_date": "2018-10-04",
        "split_sizes": {
            "train": {"products": 70, "reviews": 350},
            "val": {"products": 15, "reviews": 75},
            "test": {"products": 15, "reviews": 75},
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/meta/category"
        return httpx.Response(200, json=payload)

    client = ApiClient(transport=_mock_transport(handler))
    meta = client.get_category_meta()
    assert isinstance(meta, CategoryMetaResponse)
    assert meta.category == "Appliances"
    assert meta.n_products == 100


def test_search_products_success() -> None:
    """Verify search_products() serializes parameters and parses ProductPage."""
    payload = {
        "total": 1,
        "limit": 10,
        "offset": 0,
        "items": [
            {
                "asin": "B00009V3UF",
                "title": "Range Kleen Knob Set",
                "brand": "Range Kleen",
                "price": 12.99,
                "n_reviews": 52,
                "mean_rating": 2.63,
                "split": "train",
                "brand_missing": False,
                "price_status": "single",
                "price_missing": False,
                "price_max": None,
                "rank": "4165 in Appliances",
                "rank_value": 4165,
                "rank_category": "Appliances",
                "rank_missing": False,
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/products"
        assert request.url.params["query"] == "knob"
        assert request.url.params["limit"] == "10"
        assert request.url.params["offset"] == "0"
        return httpx.Response(200, json=payload)

    client = ApiClient(transport=_mock_transport(handler))
    page = client.search_products(query="knob", limit=10, offset=0)
    assert isinstance(page, ProductPage)
    assert page.total == 1
    assert len(page.items) == 1
    assert page.items[0].asin == "B00009V3UF"


def test_get_product_detail_success() -> None:
    """Verify get_product() returns a ProductDetail model."""
    payload = {
        "asin": "B00009V3UF",
        "title": "Range Kleen Knob Set",
        "brand": "Range Kleen",
        "price": 12.99,
        "n_reviews": 52,
        "mean_rating": 2.63,
        "split": "train",
        "brand_missing": False,
        "price_status": "single",
        "price_missing": False,
        "price_max": None,
        "rank": "4165 in Appliances",
        "rank_value": 4165,
        "rank_category": "Appliances",
        "rank_missing": False,
        "description": "Replacement knobs.",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/products/B00009V3UF"
        return httpx.Response(200, json=payload)

    client = ApiClient(transport=_mock_transport(handler))
    prod = client.get_product("B00009V3UF")
    assert isinstance(prod, ProductDetail)
    assert prod.asin == "B00009V3UF"


def test_get_reviews_success() -> None:
    """Verify get_reviews() passes query parameters and parses ReviewPage."""
    payload = {
        "total": 1,
        "limit": 20,
        "offset": 0,
        "items": [
            {
                "review_id": "rev-1",
                "asin": "B00009V3UF",
                "reviewer_id": "user-1",
                "rating": 5.0,
                "verified": True,
                "helpful_votes": 3,
                "review_time": "2018-05-01T12:00:00Z",
                "review_text": "Great product!",
                "has_text": True,
                "text_length": 14,
                "summary": "Great",
                "sentiment_label": "positive",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/products/B00009V3UF/reviews"
        assert request.url.params["verified"] == "true"
        return httpx.Response(200, json=payload)

    client = ApiClient(transport=_mock_transport(handler))
    page = client.get_reviews("B00009V3UF", verified=True)
    assert isinstance(page, ReviewPage)
    assert page.total == 1
    assert page.items[0].rating == 5.0


def test_get_analysis_success() -> None:
    """Verify get_analysis() parses AnalysisResponse."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/products/{ANALYSIS_EXAMPLE['asin']}/analysis"
        return httpx.Response(200, json=ANALYSIS_EXAMPLE)

    client = ApiClient(transport=_mock_transport(handler))
    analysis = client.get_analysis(ANALYSIS_EXAMPLE["asin"])
    assert isinstance(analysis, AnalysisResponse)
    assert analysis.asin == ANALYSIS_EXAMPLE["asin"]


def test_get_alternatives_success() -> None:
    """Verify get_alternatives() parses AlternativesResult."""
    payload = {
        "asin": "B00009V3UF",
        "total_related": 10,
        "usable": 2,
        "items": [],
        "empty_reason": "No superior alternative found",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/products/B00009V3UF/alternatives"
        return httpx.Response(200, json=payload)

    client = ApiClient(transport=_mock_transport(handler))
    alts = client.get_alternatives("B00009V3UF")
    assert isinstance(alts, AlternativesResult)
    assert alts.total_related == 10
    assert alts.usable == 2


def test_404_raises_api_not_found_with_request_id() -> None:
    """Verify 404 maps to ApiNotFound with request_id attached."""
    body = {
        "code": "PRODUCT_NOT_FOUND",
        "message": "Product 'B000UNKNOWN' not found in serving cache",
        "request_id": "trace-404-body",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json=body,
            headers={"X-Request-ID": "trace-404-header"},
        )

    client = ApiClient(transport=_mock_transport(handler))
    with pytest.raises(ApiNotFound) as exc_info:
        client.get_product("B000UNKNOWN")

    err = exc_info.value
    assert err.status_code == 404
    assert "not found" in err.message
    # Header takes precedence or body is present
    assert err.request_id in ("trace-404-header", "trace-404-body")


def test_422_raises_api_validation_error() -> None:
    """Verify 422 maps to ApiValidationError with detail preserved."""
    body = {
        "code": "VALIDATION_ERROR",
        "message": "Out-of-range query parameter value",
        "detail": {
            "errors": [
                {
                    "field": "limit",
                    "type": "greater_than_equal",
                    "message": "Input should be greater than or equal to 1",
                }
            ]
        },
        "request_id": "trace-422",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json=body)

    client = ApiClient(transport=_mock_transport(handler))
    with pytest.raises(ApiValidationError) as exc_info:
        client.search_products(limit=0)

    err = exc_info.value
    assert err.status_code == 422
    assert err.request_id == "trace-422"
    assert err.detail is not None
    assert "errors" in err.detail


def test_500_retried_then_raises_api_server_error() -> None:
    """Verify 500 on GET is retried up to max_retries before raising ApiServerError."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            500,
            json={"message": "Internal error", "request_id": "trace-500"},
            headers={"X-Request-ID": "trace-500"},
        )

    # max_retries=2 means 3 total attempts
    client = ApiClient(
        max_retries=2,
        backoff_factor=0.01,
        transport=_mock_transport(handler),
    )

    with pytest.raises(ApiServerError) as exc_info:
        client.health()

    assert calls == 3
    err = exc_info.value
    assert err.status_code == 500
    assert err.request_id == "trace-500"


def test_connection_error_retried_then_raises_api_unavailable() -> None:
    """Verify network connection failures are retried and raise ApiUnavailable."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("Failed to reach server")

    client = ApiClient(
        max_retries=2,
        backoff_factor=0.01,
        transport=_mock_transport(handler),
    )

    with pytest.raises(ApiUnavailable) as exc_info:
        client.health()

    assert calls == 3
    assert "Unable to connect" in exc_info.value.message

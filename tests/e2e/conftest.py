"""Shared fixtures for the Streamlit end-to-end tests.

Every test here runs a real page through ``streamlit.testing.v1.AppTest``, which
executes the whole script - sidebar included. The sidebar reads ``GET /health``,
so an unpatched test would open a socket to ``127.0.0.1:8000`` and pay the API
client's retry/backoff (about 2.4s) whenever no server is listening, which is
enough to blow the AppTest 3s run timeout.

The whole API client is therefore patched for every test in this package, and the
fixtures are built from the schema examples (``ANALYSIS_EXAMPLE`` and friends) -
so nothing here is hand-typed and nothing here is a result. These tests never touch
the network, and ``streamlit``'s ``cache_data`` is cleared between tests so one
test's fixture can never satisfy another test's assertion.

Individual tests override one method (``mock.patch`` inside the test body wins over
the package-wide patch) to exercise a failure or an empty payload.
"""

from __future__ import annotations

import contextlib
import copy
import unittest.mock as mock
from datetime import UTC, datetime
from typing import Any

import pytest

from munasib.api.schemas import (
    AlternativesResult,
    AnalysisResponse,
    CategoryMetaResponse,
    HealthResponse,
    ProductDetail,
    ProductPage,
    ProductSummary,
    ReviewPage,
    ReviewSnippet,
    SplitSizes,
)
from munasib.api.schemas.alternatives import ALTERNATIVES_EMPTY_EXAMPLE, ALTERNATIVES_EXAMPLE
from munasib.api.schemas.analysis import ANALYSIS_EXAMPLE
from munasib.api.schemas.reviews import REVIEW_SNIPPET_EXAMPLE, REVIEW_SNIPPET_EXAMPLE_2

#: ASIN every fixture is built around (the schema examples' subject).
DEFAULT_ASIN = "B00009V3UF"

#: Version reported by the stub analysis fixture, so the stub banner must show.
STUB_PIPELINE_VERSION = "stub-0"

#: Second ASIN used to build a two-item search page.
SECOND_ASIN = "B00QO4ZX30"


def build_health_response() -> HealthResponse:
    """Build the ``GET /health`` payload returned by the patched client.

    Returns:
        A healthy response for the project version with a reachable database.
    """
    return HealthResponse(
        status="ok",
        service="Munasib Khareed",
        version="0.1.0",
        database="reachable",
        time=datetime.now(UTC),
    )


def build_category_meta() -> CategoryMetaResponse:
    """Build the ``GET /meta/category`` payload from the schema example.

    Returns:
        The category metadata returned by the patched client.
    """
    return CategoryMetaResponse(
        category="Appliances",
        n_products=2066,
        n_reviews=425993,
        n_related_edges=84885,
        n_related_edges_in_subset=22349,
        date_range_start="2001-11-08T00:00:00Z",
        date_range_end="2018-10-04T00:00:00Z",
        reference_date="2018-10-04T00:00:00Z",
        split_sizes={
            "train": SplitSizes(products=1444, reviews=303317),
            "val": SplitSizes(products=310, reviews=62961),
            "test": SplitSizes(products=312, reviews=59715),
        },
    )


def build_product_summary(
    asin: str = DEFAULT_ASIN,
    has_image: bool = True,
    price: float | None = 12.66,
) -> ProductSummary:
    """Build one ``ProductSummary`` for a search page.

    Args:
        asin: The ASIN to set.
        has_image: Value of the ``has_image`` flag.
        price: The price, or None to exercise the "Price not listed" absence.

    Returns:
        A validated product summary.
    """
    return ProductSummary(
        asin=asin,
        title=f"Fixture product {asin}",
        brand="Fixture Brand",
        price=price,
        n_reviews=52,
        mean_rating=2.634615,
        split="train",
        has_image=has_image,
    )


def build_product_page() -> ProductPage:
    """Build a two-item ``GET /products`` page (one with an image, one without).

    The second item also has no price, so the rendering of an explicit price
    absence is covered by every test that uses the default search page.

    Returns:
        A validated product page.
    """
    return ProductPage(
        total=2,
        limit=10,
        offset=0,
        items=[
            build_product_summary(DEFAULT_ASIN, has_image=True, price=12.66),
            build_product_summary(SECOND_ASIN, has_image=False, price=None),
        ],
    )


def build_product_detail(asin: str = DEFAULT_ASIN, has_image: bool = True) -> ProductDetail:
    """Build a ``ProductDetail`` from the schema example.

    Args:
        asin: The ASIN to set.
        has_image: Value of the ``has_image`` flag.

    Returns:
        A validated product detail.
    """
    return ProductDetail(
        asin=asin,
        title="Range Kleen 8234 Gas Range Knobs",
        brand="Range Kleen",
        price=12.66,
        n_reviews=52,
        mean_rating=2.634615,
        split="train",
        has_image=has_image,
        brand_missing=False,
        price_status="single",
        price_missing=False,
        price_max=None,
        rank=">#4,165 in Appliances (See top 100)",
        rank_value=4165,
        rank_category="Appliances",
        rank_missing=False,
        description="Replacement knobs for gas ranges.",
        features=["Safe replacement knobs.", "Complete set of four."],
        first_review_time="2009-07-20T00:00:00Z",
        last_review_time="2018-02-03T00:00:00Z",
    )


def build_review_page() -> ReviewPage:
    """Build a two-item ``GET /products/{asin}/reviews`` page.

    Returns:
        A validated review page built from the review schema examples.
    """
    return ReviewPage(
        total=2,
        limit=20,
        offset=0,
        items=[
            ReviewSnippet.model_validate(REVIEW_SNIPPET_EXAMPLE),
            ReviewSnippet.model_validate(REVIEW_SNIPPET_EXAMPLE_2),
        ],
    )


def build_alternatives(*, empty: bool = False) -> AlternativesResult:
    """Build an ``AlternativesResult``, populated or explicitly empty.

    Args:
        empty: When True, build the documented empty case (with its reason).

    Returns:
        A validated alternatives result.
    """
    source = ALTERNATIVES_EMPTY_EXAMPLE if empty else ALTERNATIVES_EXAMPLE
    return AlternativesResult.model_validate(copy.deepcopy(source))


def build_analysis(asin: str = DEFAULT_ASIN, alternatives: AlternativesResult | None = None) -> AnalysisResponse:
    """Build a stub ``AnalysisResponse`` from ``ANALYSIS_EXAMPLE``.

    The result is flagged as stub (``pipeline_version`` and a ``STUB:`` warning),
    so every page that shows analysis data must render the stub banner. Only the
    ASIN and (optionally) the alternatives block are overridden; every number stays
    the schema example's.

    Args:
        asin: The analysed ASIN.
        alternatives: Alternatives block to use instead of the example's.

    Returns:
        A validated stub analysis.
    """
    payload = copy.deepcopy(ANALYSIS_EXAMPLE)
    payload["asin"] = asin
    payload["product"]["asin"] = asin
    payload["pipeline_version"] = STUB_PIPELINE_VERSION
    payload["warnings"] = ["STUB: placeholder analysis, not a real result.", *payload["warnings"]]
    if alternatives is not None:
        payload["alternatives"] = alternatives.model_dump()
    return AnalysisResponse.model_validate(payload)


#: ``SplitSizes`` re-exported for tests that build their own meta payload.
__all__ = [
    "DEFAULT_ASIN",
    "SECOND_ASIN",
    "STUB_PIPELINE_VERSION",
    "SplitSizes",
    "build_alternatives",
    "build_analysis",
    "build_category_meta",
    "build_health_response",
    "build_product_detail",
    "build_product_page",
    "build_product_summary",
    "build_review_page",
]


@pytest.fixture(autouse=True)
def api_is_reachable() -> Any:
    """Answer every API call with a fixture instead of opening a socket.

    Yields:
        Control back to the test with the patches active.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch("app.api_client.ApiClient.health", return_value=build_health_response())
        )
        stack.enter_context(
            mock.patch(
                "app.api_client.ApiClient.get_category_meta", return_value=build_category_meta()
            )
        )
        stack.enter_context(
            mock.patch("app.api_client.ApiClient.search_products", return_value=build_product_page())
        )
        stack.enter_context(
            mock.patch("app.api_client.ApiClient.get_product", side_effect=build_product_detail)
        )
        stack.enter_context(
            mock.patch("app.api_client.ApiClient.get_reviews", return_value=build_review_page())
        )
        stack.enter_context(
            mock.patch("app.api_client.ApiClient.get_analysis", side_effect=build_analysis)
        )
        stack.enter_context(
            mock.patch(
                "app.api_client.ApiClient.get_alternatives",
                return_value=build_alternatives(),
            )
        )
        stack.enter_context(
            mock.patch("app.api_client.ApiClient.get_product_image", return_value=None)
        )
        yield


@pytest.fixture(autouse=True)
def _clear_streamlit_cache() -> None:
    """Drop cached API payloads so no test can read another test's fixture."""
    import streamlit as st

    try:
        st.cache_data.clear()
    except Exception:
        pass

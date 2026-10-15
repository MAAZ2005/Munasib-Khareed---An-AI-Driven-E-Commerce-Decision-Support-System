"""Cached API wrappers using Streamlit's cache_data.

Provides memoized wrappers around ApiClient calls to minimize round-trips and
accelerate page navigation. Implements cache TTLs according to project specifications:
- Metadata / Category: 10 minutes (600 seconds)
- Product Search & Reviews: 60 seconds
- Full Analysis & Alternatives: 5 minutes (300 seconds)
- Health check: 15 seconds

All cached functions accept ONLY hashable scalar arguments (strings, integers,
booleans, None) so that hashing never fails.
"""

from __future__ import annotations

import streamlit as st

from app.api_client import get_api_client
from munasib.api.schemas import (
    AlternativesResult,
    AnalysisResponse,
    CategoryMetaResponse,
    HealthResponse,
    ProductDetail,
    ProductPage,
    ReviewPage,
)

TTL_META = 600  # 10 minutes
TTL_SEARCH = 60  # 60 seconds
TTL_ANALYSIS = 300  # 5 minutes
TTL_PRODUCT = 300  # 5 minutes
TTL_REVIEWS = 60  # 60 seconds
TTL_ALTERNATIVES = 300  # 5 minutes
TTL_HEALTH = 15  # 15 seconds


@st.cache_data(ttl=TTL_HEALTH, show_spinner=False)
def get_cached_health() -> HealthResponse:
    """Fetch backend liveness and database reachability with a short TTL."""
    client = get_api_client()
    return client.health()


@st.cache_data(ttl=TTL_META, show_spinner="Loading dataset metadata...")
def get_cached_category_meta() -> CategoryMetaResponse:
    """Fetch category details, dataset counts, and split distributions."""
    client = get_api_client()
    return client.get_category_meta()


@st.cache_data(ttl=TTL_SEARCH, show_spinner="Searching products...")
def get_cached_search_products(
    query: str = "",
    limit: int = 20,
    offset: int = 0,
    split: str | None = None,
) -> ProductPage:
    """Search products with hashable parameters and 60-second TTL."""
    client = get_api_client()
    return client.search_products(query=query, limit=limit, offset=offset, split=split)


@st.cache_data(ttl=TTL_PRODUCT, show_spinner="Loading product details...")
def get_cached_product(asin: str) -> ProductDetail:
    """Fetch profile of a single product by ASIN."""
    client = get_api_client()
    return client.get_product(asin=asin)


@st.cache_data(ttl=TTL_REVIEWS, show_spinner="Loading reviews...")
def get_cached_reviews(
    asin: str,
    limit: int = 20,
    offset: int = 0,
    label: str | None = None,
    verified: bool | None = None,
) -> ReviewPage:
    """Fetch reviews of one product sorted newest first."""
    client = get_api_client()
    return client.get_reviews(
        asin=asin, limit=limit, offset=offset, label=label, verified=verified
    )


@st.cache_data(ttl=TTL_ANALYSIS, show_spinner="Computing decision-support analysis...")
def get_cached_analysis(asin: str) -> AnalysisResponse:
    """Fetch full composed decision-support analysis with 5-minute TTL."""
    client = get_api_client()
    return client.get_analysis(asin=asin)


@st.cache_data(ttl=TTL_ALTERNATIVES, show_spinner="Loading alternatives...")
def get_cached_alternatives(asin: str) -> AlternativesResult:
    """Fetch evidence-based alternatives for a product."""
    client = get_api_client()
    return client.get_alternatives(asin=asin)


def clear_all_cached_data() -> None:
    """Invalidate all Streamlit cached data across the application."""
    st.cache_data.clear()

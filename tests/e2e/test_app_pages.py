"""End-to-end tests for the Streamlit frontend pages using streamlit.testing.v1.AppTest.

Tests verify page rendering, navigation guards, state management, search interaction,
and stub alert banners without requiring a running API or live network connection.
"""

from __future__ import annotations

import unittest.mock as mock
from datetime import UTC, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from munasib.api.schemas import (
    AnalysisResponse,
    CategoryMetaResponse,
    HealthResponse,
    ProductPage,
    ProductSummary,
    SplitSizes,
)
from munasib.api.schemas.analysis import ANALYSIS_EXAMPLE

ROOT_DIR = Path(__file__).resolve().parents[2]


def _app_path(rel: str) -> str:
    """Return absolute path string for an app file."""
    return str(ROOT_DIR / rel)


PAGES_REQUIRING_PRODUCT = [
    "app/pages/2_Product_Overview.py",
    "app/pages/3_Review_Analysis.py",
    "app/pages/4_Risk_and_Evidence.py",
    "app/pages/5_Explanation_SHAP.py",
    "app/pages/6_Alternatives.py",
    "app/pages/7_Consumer_Verdict.py",
]


@pytest.fixture(autouse=True)
def _clear_streamlit_cache() -> None:
    """Clear Streamlit cache_data between test runs to avoid state leakage."""
    import streamlit as st

    try:
        st.cache_data.clear()
    except Exception:
        pass


def _fixture_category_meta() -> CategoryMetaResponse:
    """Provide realistic CategoryMetaResponse for testing Home."""
    return CategoryMetaResponse(
        category="Appliances",
        n_products=500,
        n_reviews=2500,
        n_related_edges=600,
        n_related_edges_in_subset=400,
        date_range_start=datetime(2015, 1, 1, tzinfo=UTC),
        date_range_end=datetime(2018, 10, 4, tzinfo=UTC),
        reference_date="2018-10-04",
        split_sizes={
            "train": SplitSizes(products=350, reviews=1750),
            "val": SplitSizes(products=75, reviews=375),
            "test": SplitSizes(products=75, reviews=375),
        },
    )


def _fixture_health() -> HealthResponse:
    """Provide realistic HealthResponse for testing sidebar."""
    return HealthResponse(
        status="ok",
        service="Munasib Khareed",
        version="0.1.0",
        database="reachable",
        time=datetime.now(UTC),
    )


def _fixture_product_page() -> ProductPage:
    """Provide single-item ProductPage for testing selection."""
    return ProductPage(
        total=1,
        limit=10,
        offset=0,
        items=[
            ProductSummary(
                asin="B00009V3UF",
                title="Range Kleen 8234 Gas Range Knobs",
                brand="Range Kleen",
                price=12.66,
                n_reviews=52,
                mean_rating=2.63,
                split="train",
                brand_missing=False,
                price_status="single",
                price_missing=False,
                price_max=None,
                rank="4165 in Appliances",
                rank_value=4165,
                rank_category="Appliances",
                rank_missing=False,
            )
        ],
    )


def test_home_page_renders_with_backend_online() -> None:
    """Verify Home.py renders dataset stats when backend API answers."""
    with (
        mock.patch(
            "app.api_client.ApiClient.get_category_meta",
            return_value=_fixture_category_meta(),
        ),
        mock.patch("app.api_client.ApiClient.health", return_value=_fixture_health()),
    ):
        at = AppTest.from_file(_app_path("app/Home.py"))
        at.run()

        assert not at.exception, at.exception
        assert len(at.title) > 0
        assert "Munasib Khareed" in at.title[0].value
        assert any("Connected to backend serving cache" in s.value for s in at.success)


def test_home_page_handles_backend_offline() -> None:
    """Verify Home.py provides clear recovery instructions when API is offline."""
    with (
        mock.patch(
            "app.api_client.ApiClient.get_category_meta",
            side_effect=Exception("Connection refused"),
        ),
        mock.patch(
            "app.api_client.ApiClient.health",
            side_effect=Exception("Connection refused"),
        ),
    ):
        at = AppTest.from_file(_app_path("app/Home.py"))
        at.run()

        assert not at.exception, at.exception
        assert any("FastAPI backend is currently unavailable" in e.value for e in at.error)
        assert any("uvicorn munasib.api.main:app" in info.value for info in at.info)


def test_product_selection_page_renders_and_selects_asin() -> None:
    """Verify Page 1 renders search items and clicking Select stores selected_asin."""
    with (
        mock.patch(
            "app.api_client.ApiClient.search_products",
            return_value=_fixture_product_page(),
        ),
        mock.patch("app.api_client.ApiClient.health", return_value=_fixture_health()),
    ):
        at = AppTest.from_file(_app_path("app/pages/1_Product_Selection.py"))
        at.run()

        assert not at.exception, at.exception
        assert len(at.title) > 0
        assert "Product Selection" in at.title[0].value

        # Click the 'Select' button for B00009V3UF
        select_buttons = [b for b in at.button if b.key == "select_B00009V3UF"]
        assert len(select_buttons) == 1
        select_buttons[0].click().run()

        assert not at.exception, at.exception
        assert at.session_state["selected_asin"] == "B00009V3UF"


@pytest.mark.parametrize("rel_path", PAGES_REQUIRING_PRODUCT)
def test_pages_guard_when_no_product_selected(rel_path: str) -> None:
    """Verify pages 2-7 display guard warning and halt when no ASIN is selected."""
    at = AppTest.from_file(_app_path(rel_path))
    at.session_state["selected_asin"] = None
    at.run()

    assert not at.exception, at.exception
    assert any("No product selected" in w.value for w in at.warning)


@pytest.mark.parametrize("rel_path", PAGES_REQUIRING_PRODUCT)
def test_pages_render_title_with_selected_product(rel_path: str) -> None:
    """Verify pages 2-7 render successfully with title when a product is chosen."""
    at = AppTest.from_file(_app_path(rel_path))
    at.session_state["selected_asin"] = "B00009V3UF"
    at.run()

    assert not at.exception, at.exception
    assert len(at.title) > 0
    assert not any("No product selected" in w.value for w in at.warning)


def test_stub_banner_appears_for_stub_analysis() -> None:
    """Verify prominent stub warning appears when pipeline_version starts with 'stub'."""
    stub_data = dict(ANALYSIS_EXAMPLE)
    stub_data["pipeline_version"] = "stub-0"
    stub_data["warnings"] = ["STUB: Deterministic placeholder assessment."]
    analysis = AnalysisResponse.model_validate(stub_data)

    at = AppTest.from_file(_app_path("app/pages/4_Risk_and_Evidence.py"))
    at.session_state["selected_asin"] = "B00009V3UF"
    at.session_state["last_analysis"] = analysis
    at.run()

    assert not at.exception, at.exception
    assert any("Placeholder analysis from the stub engine" in w.value for w in at.warning)


def test_stub_banner_does_not_appear_for_non_stub_analysis() -> None:
    """Verify stub banner is absent for real/non-stub analysis responses."""
    real_data = dict(ANALYSIS_EXAMPLE)
    real_data["pipeline_version"] = "0.1.0"
    real_data["warnings"] = []
    analysis = AnalysisResponse.model_validate(real_data)

    at = AppTest.from_file(_app_path("app/pages/4_Risk_and_Evidence.py"))
    at.session_state["selected_asin"] = "B00009V3UF"
    at.session_state["last_analysis"] = analysis
    at.run()

    assert not at.exception, at.exception
    assert not any("Placeholder analysis from the stub engine" in w.value for w in at.warning)

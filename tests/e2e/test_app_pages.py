"""End-to-end tests for the Streamlit frontend pages using streamlit.testing.v1.AppTest.

Tests verify page rendering, navigation guards, state management, search interaction,
and stub alert banners without requiring a running API or live network connection.
"""

from __future__ import annotations

import re
import unittest.mock as mock
from datetime import UTC, datetime
from pathlib import Path

import pytest
from streamlit.runtime.memory_media_file_storage import _calculate_file_id
from streamlit.testing.v1 import AppTest

from app.api_client import ApiNotFound, ApiUnavailable
from app.formatting import (
    analysis_json,
    analysis_json_filename,
    build_summary_csv,
    summary_csv_filename,
)
from app.theme import IMAGE_ATTRIBUTION
from munasib.api.schemas import (
    AnalysisResponse,
    CategoryMetaResponse,
    HealthResponse,
    ProductPage,
    ProductSummary,
    SplitSizes,
)
from munasib.api.schemas.analysis import ANALYSIS_EXAMPLE
from tests.e2e import conftest as e2e

ROOT_DIR = Path(__file__).resolve().parents[2]

#: The committed logo, asserted to be a loadable local asset (no network).
LOGO_PATH = ROOT_DIR / "app" / "assets" / "logo.png"


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


# ---------------------------------------------------------------------------
# Dashboard pages on the stub API (p10)
# ---------------------------------------------------------------------------

STUB_BANNER_TEXT = "Placeholder analysis from the stub engine"

ANALYSIS_PAGES = [
    "app/pages/3_Review_Analysis.py",
    "app/pages/4_Risk_and_Evidence.py",
    "app/pages/5_Explanation_SHAP.py",
    "app/pages/6_Alternatives.py",
    "app/pages/7_Consumer_Verdict.py",
]

IMAGE_PAGES = [
    "app/pages/1_Product_Selection.py",
    "app/pages/2_Product_Overview.py",
    "app/pages/6_Alternatives.py",
    "app/pages/7_Consumer_Verdict.py",
]


def _jpeg_bytes() -> bytes:
    """Return a minimal valid JPEG so ``st.image`` renders real bytes."""
    import io

    from PIL import Image as PILImage

    buffer = io.BytesIO()
    PILImage.new("RGB", (8, 8), (200, 30, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _run_page(
    rel_path: str,
    asin: str | None = None,
    analysis: AnalysisResponse | None = None,
) -> AppTest:
    """Run one page under AppTest with an optional product and analysis.

    Args:
        rel_path: Page path relative to the repository root.
        asin: Value to put in ``selected_asin`` (skipped when None).
        analysis: Value to put in ``last_analysis`` (skipped when None).

    Returns:
        The run AppTest instance.
    """
    at = AppTest.from_file(_app_path(rel_path))
    if asin is not None:
        at.session_state["selected_asin"] = asin
    if analysis is not None:
        at.session_state["last_analysis"] = analysis
    at.run()
    return at


@pytest.mark.parametrize("rel_path", ANALYSIS_PAGES)
def test_analysis_pages_show_the_stub_banner(rel_path: str) -> None:
    """Every page that shows analysis data warns that it is stub output."""
    at = _run_page(rel_path, asin=e2e.DEFAULT_ASIN)

    assert not at.exception, at.exception
    assert any(STUB_BANNER_TEXT in w.value for w in at.warning)


@pytest.mark.parametrize("rel_path", IMAGE_PAGES)
def test_image_pages_show_the_attribution_caption(rel_path: str) -> None:
    """Every page that can display an image carries the required attribution."""
    at = _run_page(rel_path, asin=e2e.DEFAULT_ASIN)

    assert not at.exception, at.exception
    assert any(IMAGE_ATTRIBUTION in c.value for c in at.caption)


def test_product_selection_renders_cards_and_price_absence() -> None:
    """Page 1 renders one card per result, with the price absence shown."""
    # The header logo is disabled so the only images left would be product images.
    with mock.patch("app.components.header.load_logo_bytes", return_value=None):
        at = AppTest.from_file(_app_path("app/pages/1_Product_Selection.py"))
        at.run()

    assert not at.exception, at.exception
    assert any("Price not listed" in m.value for m in at.markdown)
    # The patched image call returns None, so both cards show the placeholder.
    assert len(at.image) == 0
    assert any("No image available" in m.value for m in at.markdown)


def test_product_overview_renders_profile_features_and_data_basis() -> None:
    """Page 2 renders the profile fields, an expander and the data-basis caption."""
    at = _run_page("app/pages/2_Product_Overview.py", asin=e2e.DEFAULT_ASIN)

    assert not at.exception, at.exception
    assert any("Product Overview" in t.value for t in at.title)
    assert len(at.metric) >= 4
    assert len(at.expander) >= 1
    assert any(c.value.startswith("Data basis") for c in at.caption)


def test_product_overview_shows_the_image_when_cached() -> None:
    """A product whose thumbnail is cached renders a real image element."""
    with mock.patch("app.api_client.ApiClient.get_product_image", return_value=_jpeg_bytes()):
        at = _run_page("app/pages/2_Product_Overview.py", asin=e2e.DEFAULT_ASIN)

    assert not at.exception, at.exception
    assert len(at.image) >= 1
    assert any(IMAGE_ATTRIBUTION in c.value for c in at.caption)


def test_product_overview_placeholder_makes_no_image_request() -> None:
    """A product without an image shows the placeholder and never calls the route."""
    spy = mock.Mock(return_value=None)
    with (
        mock.patch(
            "app.api_client.ApiClient.get_product",
            side_effect=lambda **kw: e2e.build_product_detail(kw["asin"], has_image=False),
        ),
        mock.patch("app.api_client.ApiClient.get_product_image", spy),
        mock.patch("app.components.header.load_logo_bytes", return_value=None),
    ):
        at = _run_page("app/pages/2_Product_Overview.py", asin=e2e.DEFAULT_ASIN)

    assert not at.exception, at.exception
    assert len(at.image) == 0
    assert any("No image available" in m.value for m in at.markdown)
    spy.assert_not_called()


def test_review_analysis_renders_charts_metrics_and_table() -> None:
    """Page 3 renders the review charts, the evidence metrics and a review table."""
    at = _run_page("app/pages/3_Review_Analysis.py", asin=e2e.DEFAULT_ASIN)

    assert not at.exception, at.exception
    assert len(at.get("plotly_chart")) >= 4
    assert len(at.dataframe) >= 1
    labels = [m.label for m in at.metric]
    assert "Verified-purchase share" in labels
    assert "Helpfulness ratio" in labels
    assert any(s.label == "Sentiment label (weak, from stars)" for s in at.selectbox)


def test_risk_and_evidence_renders_gauge_table_and_decision_charts() -> None:
    """Page 4 renders the gauge, the indicator table and the decision charts."""
    at = _run_page("app/pages/4_Risk_and_Evidence.py", asin=e2e.DEFAULT_ASIN)

    assert not at.exception, at.exception
    assert len(at.get("plotly_chart")) >= 4
    assert len(at.dataframe) >= 1
    assert any(m.label == "Composite evidence strength" for m in at.metric)


def test_explanation_renders_waterfall_and_fidelity() -> None:
    """Page 5 renders the waterfall and the surrogate-fidelity metrics."""
    at = _run_page("app/pages/5_Explanation_SHAP.py", asin=e2e.DEFAULT_ASIN)

    assert not at.exception, at.exception
    assert len(at.get("plotly_chart")) >= 1
    labels = [m.label for m in at.metric]
    assert "Surrogate R²" in labels
    assert "Surrogate MAE" in labels
    assert "Meets configured minimum" in labels


def test_alternatives_renders_cards_and_switch_updates_selection() -> None:
    """Page 6 renders a card per alternative; switching re-points the selection."""
    analysis = e2e.build_analysis(e2e.DEFAULT_ASIN)
    at = _run_page("app/pages/6_Alternatives.py", asin=e2e.DEFAULT_ASIN, analysis=analysis)

    assert not at.exception, at.exception
    assert any(b.label == "Switch to this product" for b in at.button)

    switch = [b for b in at.button if b.key == "switch_B0050O1N2U"]
    assert len(switch) == 1
    switch[0].click().run()

    assert not at.exception, at.exception
    assert at.session_state["selected_asin"] == "B0050O1N2U"
    # The previous product's analysis was dropped; the new one was re-fetched.
    assert at.session_state["last_analysis"].asin == "B0050O1N2U"


def test_alternatives_empty_shows_reason_and_counts() -> None:
    """Page 6 with no usable alternative shows why, and how many there were."""
    analysis = e2e.build_analysis(
        e2e.DEFAULT_ASIN, alternatives=e2e.build_alternatives(empty=True)
    )
    at = _run_page("app/pages/6_Alternatives.py", asin=e2e.DEFAULT_ASIN, analysis=analysis)

    assert not at.exception, at.exception
    assert any("no related products found" in info.value for info in at.info)
    assert not any((b.key or "").startswith("switch_") for b in at.button)


def test_consumer_verdict_renders_downloads_and_json_round_trips() -> None:
    """Page 7 exposes both downloads and the JSON payload is the real analysis."""
    analysis = e2e.build_analysis(e2e.DEFAULT_ASIN)
    at = _run_page("app/pages/7_Consumer_Verdict.py", asin=e2e.DEFAULT_ASIN, analysis=analysis)

    assert not at.exception, at.exception
    labels = [b.label for b in at.download_button]
    assert any("JSON" in label for label in labels)
    assert any("CSV" in label for label in labels)

    json_button = next(b for b in at.download_button if "JSON" in b.label)
    expected_json_id = _calculate_file_id(
        analysis_json(analysis).encode(),
        "application/json",
        analysis_json_filename(e2e.DEFAULT_ASIN),
    )
    assert json_button.proto.url.endswith(f"{expected_json_id}.json")

    csv_button = next(b for b in at.download_button if "CSV" in b.label)
    expected_csv_id = _calculate_file_id(
        build_summary_csv(analysis).encode(),
        "text/csv",
        summary_csv_filename(e2e.DEFAULT_ASIN),
    )
    assert csv_button.proto.url.endswith(f"{expected_csv_id}.csv")

    # The payload the button serves parses back into the contract.
    restored = AnalysisResponse.model_validate_json(analysis_json(analysis))
    assert restored.asin == e2e.DEFAULT_ASIN


def test_api_not_found_shows_friendly_message_and_clears_selection() -> None:
    """A 404 on the analysis maps to a friendly message and drops the selection."""
    with mock.patch(
        "app.api_client.ApiClient.get_analysis",
        side_effect=ApiNotFound("Product B00009V3UF is not in the serving cache."),
    ):
        at = _run_page("app/pages/3_Review_Analysis.py", asin=e2e.DEFAULT_ASIN)

    assert not at.exception, at.exception
    assert any("Product not found in the serving cache" in e.value for e in at.error)
    assert at.session_state["selected_asin"] is None


def test_api_unavailable_shows_startup_instructions() -> None:
    """An unreachable API maps to a friendly message naming the uvicorn command."""
    with mock.patch(
        "app.api_client.ApiClient.get_analysis",
        side_effect=ApiUnavailable("connection refused"),
    ):
        at = _run_page("app/pages/4_Risk_and_Evidence.py", asin=e2e.DEFAULT_ASIN)

    assert not at.exception, at.exception
    assert any("not reachable" in e.value for e in at.error)
    assert any("uvicorn munasib.api.main:app --port 8000" in info.value for info in at.info)


# ---------------------------------------------------------------------------
# Marketplace chrome and the About page (p10c)
# ---------------------------------------------------------------------------

#: Every page, with the ASIN it needs (None when it does not require a product).
ALL_PAGE_RUNS: list[tuple[str, str | None]] = [
    ("app/Home.py", None),
    ("app/pages/1_Product_Selection.py", None),
    ("app/pages/2_Product_Overview.py", e2e.DEFAULT_ASIN),
    ("app/pages/3_Review_Analysis.py", e2e.DEFAULT_ASIN),
    ("app/pages/4_Risk_and_Evidence.py", e2e.DEFAULT_ASIN),
    ("app/pages/5_Explanation_SHAP.py", e2e.DEFAULT_ASIN),
    ("app/pages/6_Alternatives.py", e2e.DEFAULT_ASIN),
    ("app/pages/7_Consumer_Verdict.py", e2e.DEFAULT_ASIN),
    ("app/pages/About.py", None),
]

_PAGE_IDS = [rel for rel, _ in ALL_PAGE_RUNS]


def _page_text(at: AppTest) -> str:
    """Collect the visible text of every widget so a page can be searched as text.

    Args:
        at: The run AppTest context.

    Returns:
        Newline-joined ``value``/``label`` strings of every element kind that
        carries text, including the header, the sidebar and the footer.
    """
    chunks: list[str] = []
    for kind in (
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "text",
        "warning",
        "info",
        "error",
        "success",
        "metric",
        "code",
        "checkbox",
        "selectbox",
        "button",
    ):
        try:
            elements = getattr(at, kind)
        except Exception:  # pragma: no cover - kind not exposed by AppTest
            continue
        for element in elements:
            for attribute in ("value", "label"):
                value = getattr(element, attribute, None)
                if value:
                    chunks.append(str(value))
    return "\n".join(chunks)


def test_logo_asset_exists_and_loads_with_pillow() -> None:
    """The committed logo is a real local PNG (no network, no CDN)."""
    from PIL import Image as PILImage

    assert LOGO_PATH.is_file(), f"missing committed logo at {LOGO_PATH}"
    with PILImage.open(LOGO_PATH) as image:
        assert image.format == "PNG"
        assert image.size[0] > 100 and image.size[1] > 40


@pytest.mark.parametrize(("rel_path", "asin"), ALL_PAGE_RUNS, ids=_PAGE_IDS)
def test_header_renders_the_logo_on_every_page(rel_path: str, asin: str | None) -> None:
    """The header (with the logo) appears at the top of every page."""
    at = _run_page(rel_path, asin=asin)

    assert not at.exception, at.exception
    assert len(at.image) >= 1, "the header should render the logo on every page"


@pytest.mark.parametrize(("rel_path", "asin"), ALL_PAGE_RUNS, ids=_PAGE_IDS)
def test_header_falls_back_to_app_name_when_logo_is_missing(
    rel_path: str, asin: str | None
) -> None:
    """A missing logo degrades to the app name as text and never crashes."""
    with mock.patch("app.components.header.load_logo_bytes", return_value=None):
        at = _run_page(rel_path, asin=asin)

    assert not at.exception, at.exception
    assert any("Munasib Khareed" in m.value for m in at.markdown)


def test_header_search_stores_query_resets_offset_and_routes_to_selection() -> None:
    """Submitting the header search stores the query, resets paging and routes."""
    at = AppTest.from_file(_app_path("app/pages/1_Product_Selection.py"))
    at.session_state["search_offset"] = 20
    at.run()
    assert not at.exception, at.exception

    at.text_input[0].input("gas range knobs")
    with mock.patch("streamlit.switch_page") as switch:
        submit = next(b for b in at.button if b.label == "Search")
        submit.click().run()

    assert not at.exception, at.exception
    assert at.session_state["search_query"] == "gas range knobs"
    assert at.session_state["search_offset"] == 0
    switch.assert_called_once()
    assert switch.call_args.args[0] == "pages/1_Product_Selection.py"


def test_product_grid_renders_cards_escapes_titles_and_shows_price_absence() -> None:
    """The selection grid shows one card per item, escapes HTML and hides no price."""
    malicious = e2e.build_product_summary("B000AAA003", has_image=False, price=None).model_copy(
        update={"title": '<script>alert(1)</script> & "quoted"', "brand": "Brand <b>X</b>"}
    )
    page = ProductPage(
        total=3,
        limit=8,
        offset=0,
        items=[
            e2e.build_product_summary("B000AAA001", has_image=True, price=12.66),
            e2e.build_product_summary("B000AAA002", has_image=False, price=None),
            malicious,
        ],
    )

    with mock.patch("app.api_client.ApiClient.search_products", return_value=page):
        at = AppTest.from_file(_app_path("app/pages/1_Product_Selection.py"))
        at.run()

    assert not at.exception, at.exception
    select_buttons = [b for b in at.button if (b.key or "").startswith("select_")]
    assert len(select_buttons) == 3

    joined = "\n".join(m.value for m in at.markdown)
    assert "Price not listed" in joined
    assert "&lt;script&gt;" in joined
    assert "<script>" not in joined
    assert "&lt;b&gt;X&lt;/b&gt;" in joined


def test_overview_advocate_panel_is_labelled_placeholder_under_stub() -> None:
    """Page 2's Consumer Advocate panel shows the stub banner and a placeholder note."""
    analysis = e2e.build_analysis(e2e.DEFAULT_ASIN)
    at = _run_page("app/pages/2_Product_Overview.py", asin=e2e.DEFAULT_ASIN, analysis=analysis)

    assert not at.exception, at.exception
    assert any(STUB_BANNER_TEXT in w.value for w in at.warning)
    assert any("Consumer Advocate" in m.value for m in at.markdown)
    assert any("Placeholder" in c.value for c in at.caption)
    assert any(b.label == "View full analysis" for b in at.button)


def test_about_page_renders_the_team_and_publishes_no_contact_details() -> None:
    """The About page shows the team and contains no email or long digit run."""
    at = AppTest.from_file(_app_path("app/pages/About.py")).run()

    assert not at.exception, at.exception
    text = _page_text(at)

    assert "Mr. Ubaid Aftab Chawla" in text
    assert "Muhammad Maaz Nizami" in text
    assert "23K-2052" in text
    assert "Muhammad Shehryar Choudhary" in text
    assert "23K-0503" in text
    assert "Abuzar Ali" in text
    assert "23K-0819" in text
    assert "Leader" in text
    assert "FAST National University" in text

    assert "@" not in text, "the About page must publish no email address"
    assert re.search(r"\d{8,}", text) is None, "the About page must contain no long digit run"

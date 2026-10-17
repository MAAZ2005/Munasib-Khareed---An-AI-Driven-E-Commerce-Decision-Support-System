"""Page 2: Product Overview.

A marketplace product page with three zones:

* the product image on the left,
* the title, brand, star row, price, sales rank and an "About this item" bullet
  list from the published features in the centre,
* a "Consumer Advocate" panel on the right (a buy-box style summary) with the
  verdict badge, a compact risk gauge, the evidence strength and a button through
  to the full analysis.

While the analysis still comes from the stub engine the panel shows the stub
banner and is explicitly labelled a placeholder. The long description lives in an
expander, and the data-basis provenance block closes the page. Missing price, rank,
brand or date is always shown as its explicit absence, never a fabricated value.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path so 'app.*' imports resolve
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from app.cache import get_cached_product  # noqa: E402
from app.components import (  # noqa: E402
    card_price_html,
    card_rating_html,
    metric_card,
    product_image,
    render_footer,
    render_header,
    render_risk_gauge,
    verdict_badge,
)
from app.components.sidebar import is_stub_analysis, render_sidebar, show_stub_banner  # noqa: E402
from app.errors import render_api_error  # noqa: E402
from app.formatting import (  # noqa: E402
    BRAND_NOT_LISTED,
    NOT_AVAILABLE,
    format_count,
    format_date,
    format_rank,
    format_rating,
    format_unit,
)
from app.state import get_or_load_analysis, require_product  # noqa: E402
from app.theme import apply_theme  # noqa: E402
from munasib.api.schemas import AnalysisResponse, ProductDetail  # noqa: E402

#: Larger thumbnail edge for the profile view.
IMAGE_WIDTH = 240

#: Path of the full-analysis page the panel links to.
VERDICT_PAGE = "pages/7_Consumer_Verdict.py"

#: Caption shown under the stub banner inside the advocate panel.
PLACEHOLDER_LABEL = (
    "Placeholder: this Consumer Advocate summary is illustrative while the stub "
    "engine is active and is not a real recommendation."
)


def _render_advocate_panel(analysis: AnalysisResponse | None) -> None:
    """Render the Consumer Advocate buy box (verdict, risk, evidence, link).

    Args:
        analysis: The composed analysis, or None when it could not be fetched
            (the panel then explains the absence instead of inventing a verdict).
    """
    with st.container(border=True):
        st.markdown("#### 🛡️ Consumer Advocate")

        if analysis is None:
            st.info(
                "The decision-support analysis is not available right now; the "
                "product profile above is still valid."
            )
            return

        show_stub_banner(analysis)

        verdict = analysis.verdict
        verdict_badge(verdict.label if verdict is not None else None)

        if verdict is not None:
            st.caption(verdict.rationale)

        if is_stub_analysis(analysis):
            st.caption(PLACEHOLDER_LABEL)

        render_risk_gauge(analysis.risk_score)
        metric_card(
            "Evidence strength",
            format_unit(analysis.evidence_strength),
            help="Weighted composite of the evidence indicators, 0 to 1.",
        )

        if st.button("View full analysis", type="primary", use_container_width=True):
            try:
                st.switch_page(VERDICT_PAGE)
            except Exception:
                # Standalone page run (no navigation context): not an error.
                pass


def _render_zones(product: ProductDetail, analysis: AnalysisResponse | None) -> None:
    """Render the image / centre-details / advocate-panel three-zone layout.

    Args:
        product: The loaded product profile.
        analysis: The composed analysis, or None.
    """
    col_image, col_center, col_advocate = st.columns([1, 2.1, 1.2], gap="large")

    with col_image:
        product_image(
            product.asin, product.has_image, width=IMAGE_WIDTH, title=product.title
        )

    with col_center:
        st.subheader(product.title or NOT_AVAILABLE)
        st.markdown(f"**Brand:** {product.brand or BRAND_NOT_LISTED}")
        st.markdown(
            card_rating_html(product.mean_rating, product.n_reviews), unsafe_allow_html=True
        )
        st.markdown(
            card_price_html(product.price, product.price_max, product.price_status),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Sales rank:** "
            f"{format_rank(product.rank_missing, product.rank_value, product.rank_category)}"
        )
        st.markdown("#### About this item")
        if product.features:
            st.markdown("\n".join(f"- {feature}" for feature in product.features))
        else:
            st.info("No feature bullets were published for this product.")

    with col_advocate:
        _render_advocate_panel(analysis)


def _render_metrics(product: ProductDetail) -> None:
    """Render the four profile metrics (reviews, rating, first/last review).

    Args:
        product: The loaded product profile.
    """
    col_reviews, col_rating, col_first, col_last = st.columns(4)
    with col_reviews:
        metric_card("Reviews in subset", product.n_reviews)
    with col_rating:
        metric_card("Mean rating", format_rating(product.mean_rating))
    with col_first:
        metric_card("First review", format_date(product.first_review_time))
    with col_last:
        metric_card("Last review", format_date(product.last_review_time))


def _render_description(product: ProductDetail) -> None:
    """Render the long description inside an expander.

    Args:
        product: The loaded product profile.
    """
    with st.expander("Full product description"):
        st.write(product.description or "No description was published for this product.")


def _render_basis(analysis: AnalysisResponse | None) -> None:
    """Render the related-product counts and the data-basis provenance block.

    Args:
        analysis: The composed analysis, or None when it was not available.
    """
    if analysis is None:
        st.info("The related-product counts and the data basis are not available right now.")
        return

    alternatives = analysis.alternatives
    st.subheader("Related products")
    st.markdown(
        f"- **Related products in the catalogue:** {format_count(alternatives.total_related)}\n"
        f"- **Usable in-subset alternatives (with an analysis):** {format_count(alternatives.usable)}"
    )

    basis = analysis.data_basis
    st.caption(
        "Data basis · "
        f"category `{basis.category}` · "
        f"{format_count(basis.n_reviews)} reviews · "
        f"{format_date(basis.date_range_start)} to {format_date(basis.date_range_end)} · "
        f"reference date {format_date(basis.reference_date)} · "
        f"pipeline `{analysis.pipeline_version}`"
    )


def _render_body() -> None:
    """Render the profile, the three zones and the trailing blocks."""
    asin = require_product()

    st.title("📦 Product Overview")
    st.caption(f"Active subject ASIN: `{asin}`")

    try:
        product = get_cached_product(asin)
    except Exception as exc:
        render_api_error(exc)
        return

    try:
        analysis: AnalysisResponse | None = get_or_load_analysis(asin)
    except Exception as exc:
        # The profile is still valid; only the analysis-backed zones are missing.
        render_api_error(exc)
        analysis = None

    _render_zones(product, analysis)
    st.divider()
    _render_metrics(product)
    _render_description(product)
    st.divider()
    _render_basis(analysis)


def main() -> None:
    """Render the Product Overview page."""
    apply_theme("Product Overview")
    render_header()
    render_sidebar()

    _render_body()
    render_footer()


if __name__ == "__main__":
    main()

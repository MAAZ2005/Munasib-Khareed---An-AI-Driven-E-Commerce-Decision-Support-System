"""Entry point and landing page for the Munasib Khareed Streamlit frontend.

This file is the application's main script and does two things:

1. it registers every page with :func:`streamlit.navigation` - giving the sidebar
   clean titles (no numeric prefixes), explicit order and explicit icons, and
   keeping dev-only pages out of the way; and
2. it is the landing page itself, explaining the virtual consumer advocate
   philosophy and showing the analysed dataset's metadata from the FastAPI backend.

The header band (brand, catalogue search, Search/About links) and the footer are
rendered by every page through :mod:`app.components.header`, so this landing page
renders them too.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path so 'app.*' imports resolve
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from app.branding import APP_NAME, logo_path  # noqa: E402
from app.cache import get_cached_category_meta  # noqa: E402
from app.components import render_footer, render_header  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402
from app.theme import apply_theme  # noqa: E402


def _page_icon() -> str:
    """Return the browser page icon: the local logo, or a neutral fallback.

    Returns:
        The logo path when the asset exists, otherwise a shopping-cart emoji.
    """
    return str(logo_path()) if logo_path().is_file() else "🛒"


def _render_dataset_status() -> None:
    """Render the analysed dataset's metadata and the backend status."""
    try:
        meta = get_cached_category_meta()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Category", meta.category)
        with col2:
            st.metric("Products in Cache", f"{meta.n_products:,}")
        with col3:
            st.metric("Total Reviews", f"{meta.n_reviews:,}")
        with col4:
            st.metric("Usable Product Edges", f"{meta.n_related_edges_in_subset:,}")

        st.caption(
            f"Review date coverage: **{meta.date_range_start.strftime('%Y-%m-%d')}** to "
            f"**{meta.date_range_end.strftime('%Y-%m-%d')}** (reference anchor: {meta.reference_date})."
        )

        with st.expander("Dataset Partitioning (Train / Val / Test)", expanded=False):
            if meta.split_sizes:
                for split_name, sizes in meta.split_sizes.items():
                    st.markdown(
                        f"- **{split_name.title()}:** {sizes.products:,} products "
                        f"({sizes.reviews:,} reviews)"
                    )
            else:
                st.markdown("Split details not available.")

        st.success("✅ Connected to backend serving cache.")

    except Exception:
        st.error("⚠️ **FastAPI backend is currently unavailable.**")
        st.info(
            "To start the API server, activate the project virtual environment and run:  \n\n"
            "```powershell\n"
            "uvicorn munasib.api.main:app --port 8000\n"
            "```\n\n"
            "Or run via make:  \n\n"
            "```powershell\n"
            "make run-api\n"
            "```"
        )


def _landing() -> None:
    """Render the landing page (the app's default page)."""
    apply_theme("Home")
    render_header()
    render_sidebar()

    st.title("🛒 Munasib Khareed")
    st.subheader("AI-Driven E-Commerce Decision-Support System (Virtual Consumer Advocate)")

    st.markdown(
        """
**Munasib Khareed** ("Appropriate / Fair Purchase") is an intelligent decision-support system
designed to defend online consumers from misleading star ratings, low-credibility reviews,
and unvetted product claims. Conventional e-commerce algorithms optimize for transaction volume
and seller conversion; Munasib Khareed acts strictly as a **virtual consumer advocate**,
prioritizing consumer protection and purchase transparency.

### How It Works
1. **Evidence-Based Credibility:** Reviews are not treated equally. Ratings are weighted
   by empirical evidence signals—verified purchase status, community helpfulness ratios,
   and descriptive review depth.
2. **Aspect & Sentiment Analysis:** Natural language processing identifies specific
   functional product aspects and their associated sentiment polarity using weak-supervision
   methods anchored in real consumer feedback.
3. **Fuzzy Logic & TOPSIS Risk Modeling:** Qualitative concerns and quantitative review
   distributions are mapped into a standardized **0–100 Consumer Risk Score**, ranking
   the product against in-category peers and verified alternatives.
4. **Transparent XAI (SHAP):** Every risk score is accompanied by local feature attribution,
   explaining precisely which attributes contributed positively or negatively to the final assessment.
5. **Decisive Verdicts:** Culminates in one of three clear verdicts:
   **CONSIDER**, **CONSIDER WITH CAUTION**, or **CONSUMER RISK**. Products lacking adequate
   review evidence are never granted a clean endorsement.
"""
    )

    st.divider()
    st.subheader("📊 Analyzed Dataset & Backend Status")
    _render_dataset_status()

    st.divider()
    st.markdown("### Ready to evaluate a product?")
    try:
        st.page_link(
            "pages/1_Product_Selection.py",
            label="🔍 Launch Product Selection",
            icon="👉",
        )
    except Exception:
        try:
            st.page_link(
                "1_Product_Selection.py",
                label="🔍 Launch Product Selection",
                icon="👉",
            )
        except Exception:
            pass

    render_footer()


def _pages() -> list[st.Page]:
    """Build the application's page list (clean titles, explicit order).

    Returns:
        The pages registered with :func:`streamlit.navigation`, default first.
    """
    return [
        st.Page(_landing, title="Home", icon="🏠", url_path="home", default=True),
        st.Page(
            "pages/1_Product_Selection.py",
            title="Product Selection",
            icon="🔍",
            url_path="product-selection",
        ),
        st.Page(
            "pages/2_Product_Overview.py",
            title="Product Overview",
            icon="📦",
            url_path="product-overview",
        ),
        st.Page(
            "pages/3_Review_Analysis.py",
            title="Review Analysis",
            icon="💬",
            url_path="review-analysis",
        ),
        st.Page(
            "pages/4_Risk_and_Evidence.py",
            title="Risk and Evidence",
            icon="⚖️",
            url_path="risk-and-evidence",
        ),
        st.Page(
            "pages/5_Explanation_SHAP.py",
            title="Explanation (SHAP)",
            icon="🔬",
            url_path="explanation-shap",
        ),
        st.Page(
            "pages/6_Alternatives.py",
            title="Alternatives",
            icon="🔄",
            url_path="alternatives",
        ),
        st.Page(
            "pages/7_Consumer_Verdict.py",
            title="Consumer Verdict",
            icon="🏷️",
            url_path="consumer-verdict",
        ),
        st.Page("pages/About.py", title="About", icon="ℹ️", url_path="about"),
        st.Page(
            "pages/_component_gallery.py",
            title="Component Gallery",
            icon="🧪",
            url_path="component-gallery",
        ),
    ]


def main() -> None:
    """Configure the page and run the navigation-selected page."""
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=_page_icon(),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.navigation(_pages(), position="sidebar").run()


if __name__ == "__main__":
    main()

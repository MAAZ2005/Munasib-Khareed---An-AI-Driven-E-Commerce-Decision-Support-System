"""Landing page for the Munasib Khareed Streamlit frontend.

Explains the virtual consumer advocate philosophy, displays category metadata and
dataset statistics from the FastAPI backend, and provides clear recovery instructions
if the API server is offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path so 'app.*' imports resolve
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from app.cache import get_cached_category_meta  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402
from app.theme import apply_theme  # noqa: E402


def main() -> None:
    """Render the landing home page."""
    apply_theme("Home")
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


if __name__ == "__main__":
    main()

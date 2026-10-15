"""Page 2: Product Overview.

Displays the complete product profile, verified metadata, and rating summary.
Requires an active product selection. Real content visualizations arrive in Prompts 9-10.
"""

from __future__ import annotations

import streamlit as st

from app.components.sidebar import render_sidebar
from app.state import require_product
from app.theme import apply_theme


def main() -> None:
    """Render the Product Overview skeleton page."""
    apply_theme("Product Overview")
    render_sidebar()

    asin = require_product()

    st.title("📦 2. Product Overview")
    st.caption(f"Active Subject ASIN: `{asin}`")

    st.info(
        "ℹ️ **Page Under Construction (Prompts 9–10)**  \n\n"
        "This view will render the product's comprehensive profile:\n"
        "- Full metadata breakdown: title, brand, sales category rank, and price range.\n"
        "- Transparent missing data indicators (e.g. unlisted brand, missing rank).\n"
        "- Star rating distribution histogram and review volume statistics.\n"
        "- Product description and technical specifications."
    )


if __name__ == "__main__":
    main()

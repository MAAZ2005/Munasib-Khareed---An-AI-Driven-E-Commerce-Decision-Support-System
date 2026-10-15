"""Page 4: Risk and Evidence.

Evaluates empirical review evidence strength and details the fuzzy-logic + TOPSIS
consumer risk calculation. Requires an active product selection.
Real content visualizations arrive in Prompts 9-10.
"""

from __future__ import annotations

import streamlit as st

from app.components.sidebar import render_sidebar, show_stub_banner
from app.state import get_last_analysis, require_product
from app.theme import apply_theme


def main() -> None:
    """Render the Risk and Evidence skeleton page."""
    apply_theme("Risk and Evidence")
    render_sidebar()

    asin = require_product()

    st.title("⚖️ 4. Risk and Evidence")
    st.caption(f"Active Subject ASIN: `{asin}`")

    analysis = get_last_analysis()
    show_stub_banner(analysis)

    st.info(
        "ℹ️ **Page Under Construction (Prompts 9–10)**  \n\n"
        "This view will render the risk and evidence analysis:\n"
        "- **Evidence Strength Indicators:** Verified purchase ratio, review depth, and helpfulness metrics.\n"
        "- **Fuzzy Logic Assessment:** Qualitative linguistic risk mapping, membership curves, and fired fuzzy rules.\n"
        "- **TOPSIS Ranking:** Multi-criteria decision ranking comparing this product against sampled category peers.\n"
        "- **Composite Risk Index:** Defuzzified 0–100 consumer risk score."
    )


if __name__ == "__main__":
    main()

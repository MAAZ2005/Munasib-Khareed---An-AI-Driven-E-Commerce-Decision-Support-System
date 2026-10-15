"""Page 7: Consumer Verdict.

Displays the virtual consumer advocate's final purchase verdict (CONSIDER,
CONSIDER WITH CAUTION, or CONSUMER RISK) alongside key concerns and empirical rationale.
Requires an active product selection. Real content visualizations arrive in Prompts 9-10.
"""

from __future__ import annotations

import streamlit as st

from app.components.sidebar import render_sidebar, show_stub_banner
from app.state import get_last_analysis, require_product
from app.theme import apply_theme


def main() -> None:
    """Render the Consumer Verdict skeleton page."""
    apply_theme("Consumer Verdict")
    render_sidebar()

    asin = require_product()

    st.title("🏷️ 7. Consumer Verdict")
    st.caption(f"Active Subject ASIN: `{asin}`")

    analysis = get_last_analysis()
    show_stub_banner(analysis)

    st.info(
        "ℹ️ **Page Under Construction (Prompts 9–10)**  \n\n"
        "This view will render the final consumer advocate verdict:\n"
        "- **Three-Tier Verdict:** Clear classification into `CONSIDER`, `CONSIDER WITH CAUTION`, or `CONSUMER RISK`.\n"
        "- **Evidence Safeguard:** Transparent enforcement ensuring products with weak evidence never receive a clean `CONSIDER`.\n"
        "- **Empirical Consumer Concerns:** Ranked list of identified risks citing exact computed metrics.\n"
        "- **Executive Rationale:** Evidence-backed narrative guiding the buyer's ultimate decision."
    )


if __name__ == "__main__":
    main()

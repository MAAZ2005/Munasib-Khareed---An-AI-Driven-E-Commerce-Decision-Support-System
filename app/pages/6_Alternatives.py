"""Page 6: Alternatives.

Displays evidence-ranked alternative products connected via also-bought and also-viewed
relationships within the category. Requires an active product selection.
Real content visualizations arrive in Prompts 9-10.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path so 'app.*' imports resolve
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from app.components.sidebar import render_sidebar, show_stub_banner  # noqa: E402
from app.state import get_last_analysis, require_product  # noqa: E402
from app.theme import apply_theme  # noqa: E402


def main() -> None:
    """Render the Alternatives skeleton page."""
    apply_theme("Alternatives")
    render_sidebar()

    asin = require_product()

    st.title("🔄 6. Alternatives")
    st.caption(f"Active Subject ASIN: `{asin}`")

    analysis = get_last_analysis()
    show_stub_banner(analysis)

    st.info(
        "ℹ️ **Page Under Construction (Prompts 9–10)**  \n\n"
        "This view will render evidence-based alternative recommendations:\n"
        "- **Empirical Graph Links:** Distinct products identified from `also_bought` and `also_viewed` co-occurrences.\n"
        "- **Comparative Deltas:** Side-by-side comparison on price, rating, review volume, and assessed risk.\n"
        "- **Superiority Classifications:** Clear indicators highlighting safer, higher-evidence alternatives.\n"
        "- **Coverage Transparency:** Explicit reporting of total related products vs. usable items in serving cache."
    )


if __name__ == "__main__":
    main()

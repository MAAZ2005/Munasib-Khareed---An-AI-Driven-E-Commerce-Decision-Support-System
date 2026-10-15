"""Page 3: Review Analysis.

Displays weak-supervision sentiment shares, aspect-level polarity breakdowns,
and temporal review volume trends. Requires an active product selection.
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

from app.components.sidebar import render_sidebar  # noqa: E402
from app.state import require_product  # noqa: E402
from app.theme import apply_theme  # noqa: E402


def main() -> None:
    """Render the Review Analysis skeleton page."""
    apply_theme("Review Analysis")
    render_sidebar()

    asin = require_product()

    st.title("💬 3. Review Analysis")
    st.caption(f"Active Subject ASIN: `{asin}`")

    st.info(
        "ℹ️ **Page Under Construction (Prompts 9–10)**  \n\n"
        "This view will render granular review intelligence:\n"
        "- **Weak Supervision Sentiment:** Negative (1–2⭐), Neutral (3⭐), Positive (4–5⭐) distributions.\n"
        "- **Aspect Extraction:** Top functional attributes (e.g. durability, installation, noise) with polarity.\n"
        "- **Temporal Review Trends:** Monthly review frequency and sentiment trajectory over time.\n"
        "- **Filtered Review Explorer:** Interactive review reader filtered by sentiment and verified purchase status."
    )


if __name__ == "__main__":
    main()

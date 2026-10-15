"""Page 5: Explanation (SHAP).

Explains the surrogate regressor predictions using SHAP feature contributions,
highlighting fidelity and driving risk factors. Requires an active product selection.
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
    """Render the SHAP Explanation skeleton page."""
    apply_theme("Explanation (SHAP)")
    render_sidebar()

    asin = require_product()

    st.title("🔬 5. Explanation (SHAP)")
    st.caption(f"Active Subject ASIN: `{asin}`")

    analysis = get_last_analysis()
    show_stub_banner(analysis)

    st.info(
        "ℹ️ **Page Under Construction (Prompts 9–10)**  \n\n"
        "This view will render interpretable AI explanations:\n"
        "- **SHAP Feature Attribution:** Force/waterfall plots showing factors increasing or decreasing risk.\n"
        "- **Surrogate Fidelity Metrics:** Strict R² score reporting between surrogate regressor and the decision engine.\n"
        "- **Key Explanatory Drivers:** Plain-language summary of primary features driving the assessment.\n"
        "- **Baseline Comparison:** How this product deviates from typical category baseline features."
    )


if __name__ == "__main__":
    main()

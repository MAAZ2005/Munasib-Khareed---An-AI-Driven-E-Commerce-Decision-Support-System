"""Page 5: Explanation (SHAP).

Explains the risk score with the surrogate regressor's SHAP contributions: the
waterfall chart, the plain-language reasons (each citing a real computed number)
and the surrogate's fidelity (R², MAE and whether it meets the configured
minimum), with a note on how to read it. Locked design decision #3: SHAP explains
a surrogate trained on the pipeline's risk scores, never the fuzzy/TOPSIS stages
directly. The stub banner is shown whenever the analysis still comes from the stub
engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path so 'app.*' imports resolve
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    metric_card,
    render_footer,
    render_header,
    render_shap_waterfall_chart,
)
from app.components.sidebar import render_sidebar, show_stub_banner  # noqa: E402
from app.errors import render_api_error  # noqa: E402
from app.state import get_or_load_analysis, require_product  # noqa: E402
from app.theme import apply_theme  # noqa: E402

#: Plain-language note on how to read a surrogate explanation.
FIDELITY_NOTE = (
    "How to read this: the explanation describes a **surrogate regressor** fitted on the "
    "pipeline's consumer risk scores, not the fuzzy or TOPSIS machinery directly (locked "
    "design decision). R² is the share of the risk-score variation the surrogate reproduces "
    "and MAE is its average error in risk points on the 0-100 scale, so the feature "
    "contributions are only as trustworthy as that fidelity."
)


def _render_body() -> None:
    """Render the surrogate-fidelity metrics and the SHAP waterfall."""
    asin = require_product()

    st.title("🔬 Explanation (SHAP)")
    st.caption(f"Active subject ASIN: `{asin}`")

    try:
        analysis = get_or_load_analysis(asin)
    except Exception as exc:
        render_api_error(exc)
        return

    show_stub_banner(analysis)

    explanation = analysis.explanation
    if explanation is not None:
        fidelity = explanation.surrogate_fidelity
        col_r2, col_mae, col_meets = st.columns(3)
        with col_r2:
            metric_card(
                "Surrogate R²",
                f"{fidelity.r2:.3f}",
                help="Share of the risk-score variation the surrogate reproduces.",
            )
        with col_mae:
            metric_card(
                "Surrogate MAE",
                f"{fidelity.mae:.2f}",
                help="Mean absolute error of the surrogate, in risk-score points (0-100).",
            )
        with col_meets:
            metric_card(
                "Meets configured minimum",
                "yes" if fidelity.meets_minimum else "no",
                help="Whether R² reaches xai.min_fidelity_r2 in configs/config.yaml.",
            )
        st.caption(FIDELITY_NOTE)

    render_shap_waterfall_chart(explanation)


def main() -> None:
    """Render the SHAP Explanation page."""
    apply_theme("Explanation (SHAP)")
    render_header()
    render_sidebar()

    _render_body()
    render_footer()


if __name__ == "__main__":
    main()

"""Page 4: Risk and Evidence.

Shows the fuzzy-logic + TOPSIS consumer risk assessment of the selected product:
the 0-100 risk gauge banded by the verdict thresholds, the composite evidence
strength, a table of every evidence and risk indicator (value, raw value, direction
and plain-language description), the evidence radar, the fuzzy membership chart and
the TOPSIS criterion contributions. The stub banner is shown whenever the analysis
still comes from the stub engine.
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
    render_evidence_indicator_radar,
    render_footer,
    render_fuzzy_membership_chart,
    render_header,
    render_risk_gauge,
    render_topsis_ranking_chart,
)
from app.components.sidebar import render_sidebar, show_stub_banner  # noqa: E402
from app.errors import render_api_error  # noqa: E402
from app.formatting import INDICATOR_COLUMNS, build_indicator_rows, format_unit  # noqa: E402
from app.state import get_or_load_analysis, require_product  # noqa: E402
from app.theme import apply_theme  # noqa: E402


def _render_body() -> None:
    """Render the risk gauge, evidence table and decision charts."""
    asin = require_product()

    st.title("⚖️ Risk and Evidence")
    st.caption(f"Active subject ASIN: `{asin}`")

    try:
        analysis = get_or_load_analysis(asin)
    except Exception as exc:
        render_api_error(exc)
        return

    show_stub_banner(analysis)

    col_gauge, col_strength = st.columns([1.2, 1])
    with col_gauge:
        render_risk_gauge(analysis.risk_score)
    with col_strength:
        metric_card(
            "Composite evidence strength",
            format_unit(analysis.evidence_strength),
            help="Weighted mean of the configured evidence indicators, 0 to 1; a low "
            "value means the evidence base is too thin to trust.",
        )

    st.subheader("Evidence and risk indicators")
    st.caption(
        "Every indicator carries its normalised 0-1 value, the raw value it came "
        "from, its direction (benefit = higher is better) and a plain-language "
        "description."
    )
    st.dataframe(
        build_indicator_rows(analysis.evidence, analysis.risk_indicators),
        column_order=list(INDICATOR_COLUMNS),
        use_container_width=True,
        hide_index=True,
    )

    render_evidence_indicator_radar(analysis.evidence, analysis.risk_indicators)

    decision = analysis.decision
    render_fuzzy_membership_chart(decision.fuzzy if decision is not None else None)
    render_topsis_ranking_chart(decision.topsis if decision is not None else None)


def main() -> None:
    """Render the Risk and Evidence page."""
    apply_theme("Risk and Evidence")
    render_header()
    render_sidebar()

    _render_body()
    render_footer()


if __name__ == "__main__":
    main()

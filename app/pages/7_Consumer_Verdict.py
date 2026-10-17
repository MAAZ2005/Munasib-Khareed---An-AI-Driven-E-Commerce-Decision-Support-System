"""Page 7: Consumer Verdict.

The end of the pipeline: the locked verdict badge (CONSIDER, CONSIDER WITH
CAUTION or CONSUMER RISK), the risk gauge and evidence strength, the major concerns
behind the verdict, its rationale, the data-basis block (category, review count,
date range, reference date and pipeline version), the image attribution, and two
downloads - the full :class:`~munasib.api.schemas.AnalysisResponse` as JSON and a
flat summary as CSV. The stub banner is shown whenever the analysis still comes
from the stub engine (PDF export arrives in a later prompt).
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
    concern_list,
    metric_card,
    product_image,
    render_footer,
    render_header,
    render_risk_gauge,
    verdict_badge,
)
from app.components.sidebar import render_sidebar, show_stub_banner  # noqa: E402
from app.errors import render_api_error  # noqa: E402
from app.formatting import (  # noqa: E402
    NOT_AVAILABLE,
    analysis_json,
    analysis_json_filename,
    build_summary_csv,
    format_count,
    format_date,
    format_unit,
    summary_csv_filename,
)
from app.state import get_or_load_analysis, require_product  # noqa: E402
from app.theme import IMAGE_ATTRIBUTION, apply_theme  # noqa: E402

#: Thumbnail edge, in pixels, for the verdict header.
IMAGE_WIDTH = 180


def _render_header(asin: str) -> None:
    """Render the product image, title, verdict badge and rationale.

    Args:
        asin: The selected product ASIN.
    """
    analysis = get_or_load_analysis(asin)
    verdict = analysis.verdict

    col_image, col_head = st.columns([1, 3])
    with col_image:
        product_image(asin, analysis.product.has_image, width=IMAGE_WIDTH, title=analysis.product.title)

    with col_head:
        st.subheader(analysis.product.title or NOT_AVAILABLE)
        verdict_badge(verdict.label if verdict is not None else None)
        if verdict is not None:
            st.markdown(f"**Rationale.** {verdict.rationale}")


def _render_downloads(asin: str) -> None:
    """Render the JSON and CSV download buttons.

    Args:
        asin: The selected product ASIN.
    """
    analysis = get_or_load_analysis(asin)

    st.subheader("Download this analysis")
    col_json, col_csv = st.columns(2)
    with col_json:
        st.download_button(
            "⬇️ Full analysis (JSON)",
            data=analysis_json(analysis),
            file_name=analysis_json_filename(asin),
            mime="application/json",
            use_container_width=True,
        )
    with col_csv:
        st.download_button(
            "⬇️ Flat summary (CSV)",
            data=build_summary_csv(analysis),
            file_name=summary_csv_filename(asin),
            mime="text/csv",
            use_container_width=True,
        )
    st.caption(
        f"Generated at {format_date(analysis.generated_at)} (UTC); the analysis itself is "
        f"anchored on the reference date {format_date(analysis.data_basis.reference_date)}."
    )


def _render_body() -> None:
    """Render the verdict, gauge, concerns, data basis and the downloads."""
    asin = require_product()

    st.title("🏷️ Consumer Verdict")
    st.caption(f"Active subject ASIN: `{asin}`")

    try:
        analysis = get_or_load_analysis(asin)
    except Exception as exc:
        render_api_error(exc)
        return

    show_stub_banner(analysis)

    _render_header(asin)

    st.divider()
    col_gauge, col_strength = st.columns([1.2, 1])
    with col_gauge:
        render_risk_gauge(analysis.risk_score)
    with col_strength:
        metric_card(
            "Evidence strength",
            format_unit(analysis.evidence_strength),
            help="Weighted composite of the evidence indicators, 0 to 1.",
        )
        metric_card(
            "Risk score",
            format_unit(analysis.risk_score, 1),
            help="Consumer risk on a 0-100 scale; higher is riskier.",
        )

    st.subheader("Major concerns")
    verdict = analysis.verdict
    if verdict is None:
        st.info("Concerns appear once a verdict has been computed for this product.")
    else:
        concern_list(verdict.major_concerns)

    st.subheader("Data basis")
    basis = analysis.data_basis
    st.markdown(
        f"- **Category:** `{basis.category}`\n"
        f"- **Reviews in the subset:** {format_count(basis.n_reviews)}\n"
        f"- **Review date range:** {format_date(basis.date_range_start)} to "
        f"{format_date(basis.date_range_end)}\n"
        f"- **Reference date (recency anchor):** {format_date(basis.reference_date)}\n"
        f"- **Pipeline version:** `{analysis.pipeline_version}`"
    )
    if analysis.warnings:
        with st.expander("Warnings reported with this analysis"):
            for warning in analysis.warnings:
                st.markdown(f"- {warning}")

    st.caption(IMAGE_ATTRIBUTION)

    st.divider()
    _render_downloads(asin)


def main() -> None:
    """Render the Consumer Verdict page."""
    apply_theme("Consumer Verdict")
    render_header()
    render_sidebar()

    _render_body()
    render_footer()


if __name__ == "__main__":
    main()

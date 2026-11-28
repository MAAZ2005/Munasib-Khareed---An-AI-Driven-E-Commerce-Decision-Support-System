"""Sidebar component and global alert banners for Munasib Khareed frontend.

Renders:
- Backend health indicator (green/red dot and DB reachability from GET /health).
- Currently selected product indicator.
- Concise navigation hints guiding the consumer through the 7-step analysis.
- "Clear cached data" action button calling st.cache_data.clear().
- `is_stub_analysis` / `is_hybrid_analysis`: the one place the "which engine
  produced this?" rule lives.
- `show_stub_banner(analysis)`: prominent warning whenever the analysis carries
  placeholder decision numbers - the full stub banner for `stub`, a narrower
  "placeholder decision assessment" banner for `hybrid`.
"""

from __future__ import annotations

import streamlit as st

from app.cache import clear_all_cached_data, get_cached_health
from app.state import get_selected_asin
from munasib.api.schemas import AnalysisResponse


def render_sidebar() -> None:
    """Render the application sidebar with health, active product, navigation, and cache tools."""
    with st.sidebar:
        st.markdown("### 🛒 Munasib Khareed")
        st.caption("AI-Driven Consumer Decision Support")
        st.divider()

        # 1. Backend Health Indicator
        st.markdown("**Backend Status**")
        try:
            health_info = get_cached_health()
            if health_info.database == "reachable":
                st.success(f"🟢 **Backend Online**  \nv{health_info.version} (DB ready)")
            else:
                st.warning(f"🟡 **Backend Degraded**  \nv{health_info.version} (DB unreachable)")
        except Exception:
            st.error("🔴 **Backend Offline**")
            st.caption(
                "Run API server:  \n`uvicorn munasib.api.main:app --port 8000`"
            )

        st.divider()

        # 2. Currently Selected Product
        st.markdown("**Active Product**")
        selected_asin = get_selected_asin()
        if selected_asin:
            st.info(f"📌 **Selected ASIN:** `{selected_asin}`")
        else:
            st.info("ℹ️ No product selected yet")

        st.divider()

        # 3. Short Navigation Hints
        st.markdown("**Workflow Steps**")
        st.markdown(
            """
1. **Product Selection** (search & pick)
2. **Product Overview** (specs & rating)
3. **Review Analysis** (aspects & sentiment)
4. **Risk & Evidence** (fuzzy + TOPSIS)
5. **Explanation (SHAP)** (feature impact)
6. **Alternatives** (evidence-ranked)
7. **Consumer Verdict** (final recommendation)
"""
        )

        st.divider()

        # 4. Clear Cached Data Button
        if st.button("🧹 Clear cached data", use_container_width=True):
            clear_all_cached_data()
            st.toast("Cached data cleared successfully!", icon="🧹")
            st.rerun()


def is_stub_analysis(analysis: AnalysisResponse | None) -> bool:
    """Report whether an analysis still comes from the stub engine.

    Pure, so the rule lives in one place and can be unit-tested: an analysis is a
    stub when its ``pipeline_version`` starts with ``stub`` or any warning starts
    with ``STUB:``.

    Args:
        analysis: AnalysisResponse instance or None.

    Returns:
        True when the analysis is stub output, False otherwise (including None).
    """
    if analysis is None:
        return False
    if analysis.pipeline_version and analysis.pipeline_version.lower().startswith("stub"):
        return True
    return any(warning.startswith("STUB:") for warning in (analysis.warnings or []))


def is_hybrid_analysis(analysis: AnalysisResponse | None) -> bool:
    """Report whether an analysis comes from the hybrid service.

    An analysis is hybrid when its ``pipeline_version`` starts with ``hybrid``
    or any warning starts with ``PARTIAL:``.

    Args:
        analysis: AnalysisResponse instance or None.

    Returns:
        True when the analysis is hybrid output, False otherwise (including None).
    """
    if analysis is None:
        return False
    if analysis.pipeline_version and analysis.pipeline_version.lower().startswith("hybrid"):
        return True
    return any(warning.startswith("PARTIAL:") for warning in (analysis.warnings or []))


def show_stub_banner(analysis: AnalysisResponse | None) -> None:
    """Display the placeholder warning for stub or hybrid output.

    Triggers when `analysis.pipeline_version` starts with 'stub' or 'hybrid', or
    any warning in `analysis.warnings` starts with 'STUB:' or 'PARTIAL:'. The
    stub gets the full-engine warning; the hybrid gets the narrower
    "placeholder decision assessment" warning because its review analysis,
    aspects and evidence are real stored data.

    Args:
        analysis: AnalysisResponse instance or None.
    """
    if is_stub_analysis(analysis):
        st.warning(
            "⚠️ **Placeholder analysis from the stub engine. Risk scores, verdicts and explanations are NOT real results.**"
        )
    elif is_hybrid_analysis(analysis):
        st.warning(
            "⚠️ **Placeholder decision assessment. Risk scores, verdicts, and explanations are placeholders.**"
        )

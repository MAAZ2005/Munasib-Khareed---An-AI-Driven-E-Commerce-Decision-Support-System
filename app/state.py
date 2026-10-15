"""Session state helpers and navigation guards for Munasib Khareed frontend.

Manages application state across pages in Streamlit's multipage framework:
selected product ASIN, latest fetched analysis, active search query, and pagination
offset. Provides ``require_product()`` as a guard to stop execution on product-dependent
pages when no ASIN has been chosen.
"""

from __future__ import annotations

import streamlit as st

from munasib.api.schemas import AnalysisResponse

KEY_SELECTED_ASIN = "selected_asin"
KEY_LAST_ANALYSIS = "last_analysis"
KEY_SEARCH_QUERY = "search_query"
KEY_SEARCH_OFFSET = "search_offset"


def init_session_state() -> None:
    """Ensure all expected session state variables are initialized with safe defaults."""
    if KEY_SELECTED_ASIN not in st.session_state:
        st.session_state[KEY_SELECTED_ASIN] = None
    if KEY_LAST_ANALYSIS not in st.session_state:
        st.session_state[KEY_LAST_ANALYSIS] = None
    if KEY_SEARCH_QUERY not in st.session_state:
        st.session_state[KEY_SEARCH_QUERY] = ""
    if KEY_SEARCH_OFFSET not in st.session_state:
        st.session_state[KEY_SEARCH_OFFSET] = 0


def get_selected_asin() -> str | None:
    """Retrieve the currently selected product ASIN, or None if none chosen.

    Returns:
        The selected ASIN string or None.
    """
    init_session_state()
    val = st.session_state.get(KEY_SELECTED_ASIN)
    return str(val) if val else None


def set_selected_asin(asin: str | None) -> None:
    """Update the selected product ASIN in session state.

    Clears the stored analysis if the ASIN has changed.

    Args:
        asin: Product ASIN or None to deselect.
    """
    init_session_state()
    old_asin = st.session_state.get(KEY_SELECTED_ASIN)
    st.session_state[KEY_SELECTED_ASIN] = asin
    if asin != old_asin:
        st.session_state[KEY_LAST_ANALYSIS] = None


def get_last_analysis() -> AnalysisResponse | None:
    """Retrieve the cached AnalysisResponse from session state.

    Returns:
        AnalysisResponse if previously stored, else None.
    """
    init_session_state()
    val = st.session_state.get(KEY_LAST_ANALYSIS)
    if isinstance(val, AnalysisResponse):
        return val
    return None


def set_last_analysis(analysis: AnalysisResponse | None) -> None:
    """Store the latest AnalysisResponse in session state.

    Args:
        analysis: AnalysisResponse or None.
    """
    init_session_state()
    st.session_state[KEY_LAST_ANALYSIS] = analysis


def get_search_query() -> str:
    """Retrieve the current search query string from session state.

    Returns:
        Search query string (defaults to empty string).
    """
    init_session_state()
    val = st.session_state.get(KEY_SEARCH_QUERY, "")
    return str(val) if val is not None else ""


def set_search_query(query: str) -> None:
    """Update the search query string in session state.

    Args:
        query: New search string.
    """
    init_session_state()
    st.session_state[KEY_SEARCH_QUERY] = query


def get_search_offset() -> int:
    """Retrieve the current product search pagination offset.

    Returns:
        Pagination offset integer (non-negative).
    """
    init_session_state()
    val = st.session_state.get(KEY_SEARCH_OFFSET, 0)
    try:
        return max(0, int(val))
    except (TypeError, ValueError):
        return 0


def set_search_offset(offset: int) -> None:
    """Update the product search pagination offset in session state.

    Args:
        offset: New non-negative offset integer.
    """
    init_session_state()
    st.session_state[KEY_SEARCH_OFFSET] = max(0, offset)


def require_product() -> str:
    """Guard product-dependent pages.

    If no product ASIN is selected, displays an informative warning message with a
    navigation link to the Product Selection page, and stops execution immediately.

    Returns:
        The valid selected ASIN string.
    """
    asin = get_selected_asin()
    if not asin:
        st.warning(
            "⚠️ **No product selected.** Please select a product from the catalogue to view this page."
        )
        try:
            st.page_link(
                "pages/1_Product_Selection.py",
                label="Go to Product Selection",
                icon="🔍",
            )
        except Exception:
            try:
                st.page_link(
                    "1_Product_Selection.py",
                    label="Go to Product Selection",
                    icon="🔍",
                )
            except Exception:
                pass
        st.stop()
    return asin

"""Shared marketplace-style header and footer, rendered by every page.

The dashboard borrows the *layout conventions* of large online shops (a slim top
bar with the brand on the left, a search box in the middle and links on the right,
a fixed footer) without using any third-party name, logo or asset: the brand is
"Munasib Khareed" and the only logo is the committed local file.

Two rules hold for the whole module:

* **Offline only.** The logo is read from ``app/assets/logo.png``; the band is
  styled by :data:`app.theme.CUSTOM_CSS`; there is no CDN, web font or remote
  image anywhere.
* **Native first, and any HTML is escaped.** The band is a ``st.container`` styled
  by a key-based CSS class, the search box is a native ``st.form`` and the links
  are native ``st.page_link`` elements. The only HTML written here is the wordmark
  fallback, whose text is a module constant (never product data).

The search box is the single entry point for the catalogue: submitting it stores
the query in session state, resets the pagination offset and routes to Product
Selection, exactly as page 1 expects.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from app.branding import APP_NAME, APP_TAGLINE, load_logo_bytes
from app.state import set_search_offset, set_search_query
from app.theme import DATASET_CITATION, FOOTER_CREDIT, IMAGE_ATTRIBUTION

#: Key of the header container; ``st.container(key=...)`` adds the
#: ``.st-key-mk_header`` class that :data:`app.theme.CUSTOM_CSS` styles.
HEADER_KEY = "mk_header"

#: Key and element ids of the header search form.
SEARCH_FORM_KEY = "mk_header_search"
SEARCH_INPUT_KEY = "mk_header_search_query"

#: Placeholder for the header search box.
SEARCH_PLACEHOLDER = "Search products by title or brand…"

#: Logo render width, in pixels.
LOGO_WIDTH = 150

#: Page paths the header links to (relative to the entry script's directory).
SELECTION_PAGE = "pages/1_Product_Selection.py"
ABOUT_PAGE = "pages/About.py"


def _render_brand() -> None:
    """Render the local logo, or the application name as a text fallback.

    The committed logo is a wordmark that already carries the product name, so it
    is shown on its own; only a missing/unreadable asset falls back to the name as
    text (never an exception, never a broken image).
    """
    data = load_logo_bytes()
    if data is not None:
        st.image(data, width=LOGO_WIDTH, alt=APP_NAME)
        return
    # Text fallback: the strings are module constants, never product data.
    st.markdown(
        f'<div class="mk-wordmark">{escape(APP_NAME)}</div>'
        f'<div class="mk-tagline">{escape(APP_TAGLINE)}</div>',
        unsafe_allow_html=True,
    )


def _submit_search(query: str) -> None:
    """Persist a submitted query and route to Product Selection.

    Storing the query and resetting the offset happen first, so the target page
    reads the new search even if the navigation is not available (for example when
    a page script is run on its own in a test).

    Args:
        query: The text the user typed into the header search box.
    """
    set_search_query(query)
    set_search_offset(0)
    try:
        st.switch_page(SELECTION_PAGE)
    except Exception:
        # No navigation context (e.g. a page run standalone): the session-state
        # update above is still the right outcome, so this is not an error.
        pass


def _render_search() -> None:
    """Render the header search box (native form; Enter or the button submits)."""
    with st.form(SEARCH_FORM_KEY, clear_on_submit=True, border=False):
        col_input, col_button = st.columns([4, 1], vertical_alignment="center")
        with col_input:
            query = st.text_input(
                "Search products",
                key=SEARCH_INPUT_KEY,
                placeholder=SEARCH_PLACEHOLDER,
                label_visibility="collapsed",
            )
        with col_button:
            submitted = st.form_submit_button(
                "Search", type="primary", use_container_width=True
            )
    if submitted:
        _submit_search(query)


def _render_links() -> None:
    """Render the Search and About links (best effort; never raises)."""
    col_search, col_about = st.columns(2)
    for column, label, target, icon in (
        (col_search, "Search", SELECTION_PAGE, "🔍"),
        (col_about, "About", ABOUT_PAGE, "ℹ️"),
    ):
        with column:
            try:
                st.page_link(target, label=label, icon=icon)
            except Exception:
                # page_link needs a navigation context; on a standalone page run
                # it is unavailable, which must not break the page.
                pass


def render_header() -> None:
    """Render the shared header band on the current page.

    Called at the top of every page's ``main`` so the brand, the catalogue search
    and the Search/About links appear consistently, before any page content.
    """
    with st.container(key=HEADER_KEY):
        col_brand, col_search, col_links = st.columns(
            [1.7, 3.0, 1.6], vertical_alignment="center"
        )
        with col_brand:
            _render_brand()
        with col_search:
            _render_search()
        with col_links:
            _render_links()


def render_footer() -> None:
    """Render the shared footer on the current page.

    Carries the image attribution, the dataset citation and the project credit, so
    the sources and the academic context are visible from every screen.
    """
    st.divider()
    st.caption(IMAGE_ATTRIBUTION)
    st.caption(DATASET_CITATION)
    st.caption(FOOTER_CREDIT)


__all__ = [
    "ABOUT_PAGE",
    "HEADER_KEY",
    "LOGO_WIDTH",
    "SEARCH_FORM_KEY",
    "SEARCH_INPUT_KEY",
    "SEARCH_PLACEHOLDER",
    "SELECTION_PAGE",
    "render_footer",
    "render_header",
]

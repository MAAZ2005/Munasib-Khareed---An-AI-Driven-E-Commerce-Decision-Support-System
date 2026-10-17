"""Page 1: Product Selection.

A marketplace-style results grid over ``GET /products`` with server-side
pagination. The search text is typed in the shared header (which stores it in
session state and routes here); this page reads that query and the offset from
session state, so both survive navigation.

Each result is a card with a fixed-height image box (the cached thumbnail or the
neutral placeholder), a title clamped to two lines, the brand, a star row that
always carries the numeric rating and the review count, the price in large type
(or "Price not listed") and an accent **Select** button that makes the product the
active subject of pages 2-7.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path so 'app.*' imports resolve
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from app.cache import get_cached_search_products  # noqa: E402
from app.components import (  # noqa: E402
    card_brand_html,
    card_price_html,
    card_rating_html,
    card_title_html,
    product_image,
    render_footer,
    render_header,
)
from app.components.product_card import CARD_WIDTH, IMAGE_BOX_HEIGHT, IMAGE_WIDTH  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402
from app.errors import render_api_error  # noqa: E402
from app.state import (  # noqa: E402
    get_search_offset,
    get_search_query,
    get_selected_asin,
    set_search_offset,
    set_search_query,
    set_selected_asin,
)
from app.theme import IMAGE_ATTRIBUTION, apply_theme  # noqa: E402
from munasib.api.schemas import ProductSummary  # noqa: E402

#: Results per page (four cards per row, two rows on a wide screen).
PAGE_SIZE = 8


def _render_active_banner(asin: str) -> None:
    """Show the currently selected product and a link to the next page.

    Args:
        asin: The active ASIN.
    """
    st.success(f"✅ Selected product: **`{asin}`** — it anchors pages 2 through 7.")
    try:
        st.page_link("pages/2_Product_Overview.py", label="Open Product Overview", icon="👉")
    except Exception:
        try:
            st.page_link("2_Product_Overview.py", label="Open Product Overview", icon="👉")
        except Exception:
            pass


def _render_card(item: ProductSummary, current_asin: str | None) -> None:
    """Render one search result as a marketplace card with a Select button.

    Args:
        item: A validated :class:`~munasib.api.schemas.ProductSummary`.
        current_asin: The active ASIN, so the current product is marked selected.
    """
    with st.container(border=True, width=CARD_WIDTH):
        with st.container(height=IMAGE_BOX_HEIGHT, horizontal_alignment="center"):
            product_image(
                item.asin,
                item.has_image,
                width=IMAGE_WIDTH,
                title=item.title,
                show_attribution=False,
            )
        st.markdown(card_title_html(item.title), unsafe_allow_html=True)
        st.markdown(card_brand_html(item.brand), unsafe_allow_html=True)
        st.markdown(card_rating_html(item.mean_rating, item.n_reviews), unsafe_allow_html=True)
        st.markdown(card_price_html(item.price), unsafe_allow_html=True)

        is_active = current_asin == item.asin
        if st.button(
            "Selected ✅" if is_active else "Select",
            key=f"select_{item.asin}",
            disabled=is_active,
            type="primary",
            use_container_width=True,
        ):
            # set_selected_asin clears last_analysis when the ASIN changes.
            set_selected_asin(item.asin)
            st.toast(f"Selected product {item.asin}", icon="✅")
            st.rerun()


def _render_body() -> None:
    """Render the search summary, the results grid and the pagination controls."""
    st.title("🔍 Product Selection")
    st.markdown(
        "Search the serving cache from the box above, then select the product to "
        "anchor the full consumer decision-support pipeline."
    )

    query = get_search_query()
    if query:
        col_query, col_clear = st.columns([5, 1], vertical_alignment="center")
        with col_query:
            st.caption(f'Showing matches for **"{query}"**.')
        with col_clear:
            if st.button("Clear search", use_container_width=True):
                set_search_query("")
                set_search_offset(0)
                st.rerun()

    current_asin = get_selected_asin()
    if current_asin:
        _render_active_banner(current_asin)
        st.divider()

    offset = get_search_offset()
    try:
        page = get_cached_search_products(query=query, limit=PAGE_SIZE, offset=offset)
    except Exception as exc:
        render_api_error(exc)
        return

    if page.total == 0:
        st.info("No matching products found. Try clearing or broadening your search terms.")
        return

    shown_from = offset + 1
    shown_to = min(offset + PAGE_SIZE, page.total)
    st.caption(
        f"Displaying **{shown_from}**–**{shown_to}** of **{page.total:,}** matching products"
    )

    with st.container(horizontal=True, wrap=True, gap="medium"):
        for item in page.items:
            _render_card(item, current_asin)

    st.caption(IMAGE_ATTRIBUTION)

    col_prev, col_pages, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("⬅️ Previous", disabled=(offset == 0), use_container_width=True):
            set_search_offset(max(0, offset - PAGE_SIZE))
            st.rerun()

    with col_pages:
        total_pages = max(1, (page.total + PAGE_SIZE - 1) // PAGE_SIZE)
        current_page_num = (offset // PAGE_SIZE) + 1
        st.markdown(
            "<div style='text-align: center; padding-top: 8px;'>"
            f"Page <b>{current_page_num}</b> of <b>{total_pages}</b>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_next:
        has_next = offset + PAGE_SIZE < page.total
        if st.button("Next ➡️", disabled=not has_next, use_container_width=True):
            set_search_offset(offset + PAGE_SIZE)
            st.rerun()


def main() -> None:
    """Render the Product Selection page."""
    apply_theme("Product Selection")
    render_header()
    render_sidebar()

    _render_body()
    render_footer()


if __name__ == "__main__":
    main()

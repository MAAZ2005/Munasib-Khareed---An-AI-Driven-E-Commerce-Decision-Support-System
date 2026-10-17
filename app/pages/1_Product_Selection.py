"""Page 1: Product Selection.

Interactive search against ``GET /products`` with server-side pagination. Each
result is rendered as a card with its cached image (or a neutral placeholder),
title, brand, price, review count and mean rating, plus a **Select** button that
stores the ASIN as the active product, clears any stale analysis and confirms the
choice. The search query and offset live in session state so paging and search
survive navigation between pages.
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
from app.components import product_image  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402
from app.errors import render_api_error  # noqa: E402
from app.formatting import NOT_AVAILABLE, format_price, format_rating  # noqa: E402
from app.state import (  # noqa: E402
    get_search_offset,
    get_search_query,
    get_selected_asin,
    set_search_offset,
    set_search_query,
    set_selected_asin,
)
from app.theme import apply_theme  # noqa: E402
from munasib.api.schemas import ProductSummary  # noqa: E402

#: Results per page (within the API's 1-100 bound).
PAGE_SIZE = 10

#: Thumbnail edge, in pixels, for the card image/placeholder.
IMAGE_WIDTH = 110


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
    """Render one search result as a card with an image and a Select button.

    Args:
        item: A validated :class:`~munasib.api.schemas.ProductSummary`.
        current_asin: The active ASIN, so the current product is marked selected.
    """
    with st.container(border=True):
        col_image, col_info, col_action = st.columns([1, 4, 1.3])

        with col_image:
            product_image(
                item.asin, item.has_image, width=IMAGE_WIDTH, title=item.title
            )

        with col_info:
            st.markdown(f"**{item.title or NOT_AVAILABLE}**")
            st.markdown(
                f"**Brand:** {item.brand or NOT_AVAILABLE} "
                f"&nbsp;·&nbsp; **Price:** {format_price(item.price)} "
                f"&nbsp;·&nbsp; **Reviews:** {item.n_reviews:,} "
                f"&nbsp;·&nbsp; **Mean rating:** {format_rating(item.mean_rating)}"
            )
            st.caption(f"ASIN `{item.asin}` · split `{item.split}`")

        with col_action:
            is_active = current_asin == item.asin
            if st.button(
                "Selected ✅" if is_active else "Select",
                key=f"select_{item.asin}",
                disabled=is_active,
                use_container_width=True,
            ):
                # set_selected_asin clears last_analysis when the ASIN changes.
                set_selected_asin(item.asin)
                st.toast(f"Selected product {item.asin}", icon="✅")
                st.rerun()


def main() -> None:
    """Render the Product Selection interface."""
    apply_theme("Product Selection")
    render_sidebar()

    st.title("🔍 1. Product Selection")
    st.markdown(
        "Search the serving cache by title or brand, then select the product to "
        "anchor the full consumer decision-support pipeline."
    )

    current_asin = get_selected_asin()
    if current_asin:
        _render_active_banner(current_asin)
        st.divider()

    saved_query = get_search_query()
    query_input = st.text_input(
        "Search by title or brand keyword:",
        value=saved_query,
        placeholder="e.g. Range Kleen, filter, knob, refrigerator...",
    )

    current_offset = get_search_offset()

    # Changing the query resets paging so offset can never point past the results.
    if query_input != saved_query:
        set_search_query(query_input)
        set_search_offset(0)
        current_offset = 0

    try:
        page = get_cached_search_products(
            query=query_input,
            limit=PAGE_SIZE,
            offset=current_offset,
        )
    except Exception as exc:
        render_api_error(exc)
        return

    if page.total == 0:
        st.info("No matching products found. Try clearing or broadening your search terms.")
        return

    shown_from = current_offset + 1
    shown_to = min(current_offset + PAGE_SIZE, page.total)
    st.caption(f"Displaying **{shown_from}**–**{shown_to}** of **{page.total:,}** matching products")

    for item in page.items:
        _render_card(item, current_asin)

    col_prev, col_pages, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("⬅️ Previous", disabled=(current_offset == 0), use_container_width=True):
            set_search_offset(max(0, current_offset - PAGE_SIZE))
            st.rerun()

    with col_pages:
        total_pages = max(1, (page.total + PAGE_SIZE - 1) // PAGE_SIZE)
        current_page_num = (current_offset // PAGE_SIZE) + 1
        st.markdown(
            "<div style='text-align: center; padding-top: 8px;'>"
            f"Page <b>{current_page_num}</b> of <b>{total_pages}</b>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_next:
        has_next = current_offset + PAGE_SIZE < page.total
        if st.button("Next ➡️", disabled=not has_next, use_container_width=True):
            set_search_offset(current_offset + PAGE_SIZE)
            st.rerun()


if __name__ == "__main__":
    main()

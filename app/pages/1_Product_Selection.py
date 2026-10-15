"""Page 1: Product Selection.

Provides an interactive search interface against GET /products with server-side pagination,
allowing the user to search by brand or title, browse candidate items, and select an
active ASIN for subsequent deep analysis across pages 2 through 7.
"""

from __future__ import annotations

import streamlit as st

from app.cache import get_cached_search_products
from app.components.sidebar import render_sidebar
from app.state import (
    get_search_offset,
    get_search_query,
    get_selected_asin,
    set_search_offset,
    set_search_query,
    set_selected_asin,
)
from app.theme import apply_theme

PAGE_SIZE = 10


def main() -> None:
    """Render the Product Selection interface."""
    apply_theme("Product Selection")
    render_sidebar()

    st.title("🔍 1. Product Selection")
    st.markdown(
        "Search products in the serving cache by title or brand. "
        "Select a product to anchor the full consumer decision-support pipeline."
    )

    current_asin = get_selected_asin()
    if current_asin:
        st.success(f"📌 Currently Active Product: **`{current_asin}`**")
        try:
            st.page_link(
                "pages/2_Product_Overview.py",
                label="Proceed to Product Overview",
                icon="👉",
            )
        except Exception:
            try:
                st.page_link(
                    "2_Product_Overview.py",
                    label="Proceed to Product Overview",
                    icon="👉",
                )
            except Exception:
                pass
        st.divider()

    # Search bar
    saved_query = get_search_query()
    query_input = st.text_input(
        "Search by title or brand keyword:",
        value=saved_query,
        placeholder="e.g. Range Kleen, filter, knob, refrigerator...",
    )

    current_offset = get_search_offset()

    # Reset offset if search term changed
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

        st.caption(
            f"Displaying **{current_offset + 1}**–**{min(current_offset + PAGE_SIZE, page.total)}** "
            f"of **{page.total:,}** matching products"
        )

        if page.total == 0:
            st.info("No matching products found. Try clearing or broadening your search terms.")
            return

        for item in page.items:
            with st.container():
                col_info, col_action = st.columns([5, 1])

                with col_info:
                    st.markdown(f"**{item.title}**")
                    rating_str = f"⭐ {item.mean_rating:.2f}" if item.mean_rating is not None else "⭐ N/A"
                    meta_parts: list[str] = [
                        f"**ASIN:** `{item.asin}`",
                        f"**Brand:** {item.brand or 'N/A'}",
                        f"**Rating:** {rating_str} ({item.n_reviews:,} reviews)",
                        f"**Split:** `{item.split}`",
                    ]
                    if item.price is not None:
                        meta_parts.append(f"**Price:** ${item.price:.2f}")
                    else:
                        meta_parts.append("**Price:** N/A")

                    st.markdown(" • ".join(meta_parts))

                with col_action:
                    is_active = current_asin == item.asin
                    button_label = "Selected ✅" if is_active else "Select"
                    if st.button(
                        button_label,
                        key=f"select_{item.asin}",
                        disabled=is_active,
                        use_container_width=True,
                    ):
                        set_selected_asin(item.asin)
                        st.toast(f"Selected product {item.asin}", icon="✅")
                        st.rerun()

                st.divider()

        # Pagination controls
        col_prev, col_pages, col_next = st.columns([1, 4, 1])
        with col_prev:
            if st.button("⬅️ Previous", disabled=(current_offset == 0), use_container_width=True):
                set_search_offset(max(0, current_offset - PAGE_SIZE))
                st.rerun()

        with col_pages:
            total_pages = max(1, (page.total + PAGE_SIZE - 1) // PAGE_SIZE)
            current_page_num = (current_offset // PAGE_SIZE) + 1
            st.markdown(
                f"<div style='text-align: center; padding-top: 8px;'>"
                f"Page <b>{current_page_num}</b> of <b>{total_pages}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_next:
            has_next = current_offset + PAGE_SIZE < page.total
            if st.button("Next ➡️", disabled=not has_next, use_container_width=True):
                set_search_offset(current_offset + PAGE_SIZE)
                st.rerun()

    except Exception as exc:
        st.error(f"⚠️ Unable to fetch products from backend API: {exc}")
        st.info(
            "Ensure the FastAPI server is running:  \n"
            "`uvicorn munasib.api.main:app --port 8000`"
        )


if __name__ == "__main__":
    main()

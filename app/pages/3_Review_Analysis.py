"""Page 3: Review Analysis.

Shows the weak-supervision review picture of the selected product - star
distribution, sentiment breakdown, sentiment over time and dominant aspects - with
metric cards for the verified-purchase and helpfulness evidence indicators, and a
filterable, paged sample of the underlying reviews from
``GET /products/{asin}/reviews``. Sentiment labels are weak supervision from star
ratings, never model sentiment, and the stub banner is shown whenever the analysis
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

from app.cache import get_cached_reviews  # noqa: E402
from app.components import (  # noqa: E402
    metric_card,
    render_aspect_bar_chart,
    render_rating_distribution_chart,
    render_sentiment_breakdown_chart,
    render_sentiment_over_time_chart,
)
from app.components.sidebar import render_sidebar, show_stub_banner  # noqa: E402
from app.errors import render_api_error  # noqa: E402
from app.formatting import build_review_rows, format_unit  # noqa: E402
from app.state import get_or_load_analysis, require_product  # noqa: E402
from app.theme import apply_theme  # noqa: E402

PAGE_SIZE = 15

LABEL_OPTIONS = ("all", "negative", "neutral", "positive")
VERIFIED_OPTIONS = ("all", "verified only", "unverified only")

#: Session keys for the review-table filters and offset.
_KEY_LABEL = "review_label_filter"
_KEY_VERIFIED = "review_verified_filter"
_KEY_SIGNATURE = "review_filter_signature"
_KEY_OFFSET = "review_offset"


def _render_review_table(asin: str) -> None:
    """Render the filtered, paged review sample table.

    Args:
        asin: The selected product ASIN.
    """
    st.subheader("Review sample")

    col_label, col_verified = st.columns(2)
    label_choice = col_label.selectbox(
        "Sentiment label (weak, from stars)", LABEL_OPTIONS, key=_KEY_LABEL
    )
    verified_choice = col_verified.selectbox(
        "Verified purchase", VERIFIED_OPTIONS, key=_KEY_VERIFIED
    )

    # Changing a filter resets paging: the old offset may be past the new total.
    signature = (label_choice, verified_choice)
    if st.session_state.get(_KEY_SIGNATURE) != signature:
        st.session_state[_KEY_SIGNATURE] = signature
        st.session_state[_KEY_OFFSET] = 0

    offset = int(st.session_state.get(_KEY_OFFSET, 0))
    label = None if label_choice == "all" else label_choice
    verified = None if verified_choice == "all" else verified_choice == "verified only"

    try:
        reviews = get_cached_reviews(
            asin=asin, limit=PAGE_SIZE, offset=offset, label=label, verified=verified
        )
    except Exception as exc:
        render_api_error(exc)
        return

    st.caption(f"**{reviews.total:,}** reviews match the current filters.")

    if reviews.total == 0:
        st.info("No reviews match the current filters. Try widening them.")
        return

    st.dataframe(build_review_rows(reviews.items), use_container_width=True, hide_index=True)

    col_prev, col_pages, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("⬅️ Newer", disabled=(offset == 0), use_container_width=True):
            st.session_state[_KEY_OFFSET] = max(0, offset - PAGE_SIZE)
            st.rerun()
    with col_pages:
        total_pages = max(1, (reviews.total + PAGE_SIZE - 1) // PAGE_SIZE)
        st.markdown(
            "<div style='text-align: center; padding-top: 8px;'>"
            f"Page <b>{offset // PAGE_SIZE + 1}</b> of <b>{total_pages}</b>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button(
            "Older ➡️",
            disabled=(offset + PAGE_SIZE >= reviews.total),
            use_container_width=True,
        ):
            st.session_state[_KEY_OFFSET] = offset + PAGE_SIZE
            st.rerun()


def main() -> None:
    """Render the Review Analysis page."""
    apply_theme("Review Analysis")
    render_sidebar()

    asin = require_product()

    st.title("💬 3. Review Analysis")
    st.caption(f"Active subject ASIN: `{asin}`")

    try:
        analysis = get_or_load_analysis(asin)
    except Exception as exc:
        render_api_error(exc)
        return

    show_stub_banner(analysis)

    review_analysis = analysis.review_analysis
    col_rating, col_sentiment = st.columns(2)
    with col_rating:
        render_rating_distribution_chart(review_analysis.rating_distribution)
    with col_sentiment:
        render_sentiment_breakdown_chart(review_analysis.sentiment)

    render_sentiment_over_time_chart(review_analysis.temporal_trend)
    render_aspect_bar_chart(review_analysis.aspects)

    st.subheader("Evidence indicators")
    col_verified, col_helpful = st.columns(2)
    with col_verified:
        indicator = analysis.evidence.verified_purchase
        metric_card("Verified-purchase share", format_unit(indicator.value), help=indicator.description)
    with col_helpful:
        indicator = analysis.evidence.helpfulness
        metric_card("Helpfulness ratio", format_unit(indicator.value), help=indicator.description)

    st.divider()
    _render_review_table(asin)


if __name__ == "__main__":
    main()

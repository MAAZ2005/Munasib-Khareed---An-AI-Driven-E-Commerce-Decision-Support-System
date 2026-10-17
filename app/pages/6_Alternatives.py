"""Page 6: Alternatives.

Presents the evidence-ranked alternatives as a "Customers also considered"-style
grid of cards: each card shows the alternative's cached image (or the neutral
placeholder), a clamped title, its own verdict badge, its risk score (with the
comparison delta against the subject) and a **Switch to this product** button that
re-points the selection. A compact comparison table follows.

When there is no usable alternative the page shows why and how many related
products existed in total. The stub banner is shown whenever the analysis still
comes from the stub engine.
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

# Ensure repository root is on sys.path so 'app.*' imports resolve
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    alternatives_table,
    card_price_html,
    card_rating_html,
    card_title_html,
    product_image,
    render_footer,
    render_header,
    verdict_badge,
)
from app.components.alternatives_table import RELATION_ICONS  # noqa: E402
from app.components.product_card import CARD_WIDTH, IMAGE_BOX_HEIGHT, IMAGE_WIDTH  # noqa: E402
from app.components.sidebar import render_sidebar, show_stub_banner  # noqa: E402
from app.errors import render_api_error  # noqa: E402
from app.formatting import format_count, format_signed  # noqa: E402
from app.state import get_or_load_analysis, require_product, set_selected_asin  # noqa: E402
from app.theme import IMAGE_ATTRIBUTION, apply_theme  # noqa: E402
from munasib.api.schemas import AlternativeProduct  # noqa: E402


def _render_card(alternative: AlternativeProduct) -> None:
    """Render one alternative as a marketplace card with a switch button.

    Args:
        alternative: The alternative to render.
    """
    summary = alternative.summary
    relation = alternative.relation.value
    icon = RELATION_ICONS.get(relation, "ℹ️")

    with st.container(border=True, width=CARD_WIDTH):
        with st.container(height=IMAGE_BOX_HEIGHT, horizontal_alignment="center"):
            product_image(
                summary.asin,
                summary.has_image,
                width=IMAGE_WIDTH,
                title=summary.title,
                show_attribution=False,
            )
        st.markdown(card_title_html(summary.title), unsafe_allow_html=True)
        verdict_badge(alternative.verdict)
        st.markdown(
            card_rating_html(summary.mean_rating, summary.n_reviews), unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="mk-review-count">{escape(icon)} {escape(relation)} than the '
            "selected product</div>",
            unsafe_allow_html=True,
        )
        st.markdown(card_price_html(summary.price), unsafe_allow_html=True)
        st.markdown(
            f"**Risk score:** {alternative.risk_score:.1f} / 100  \n"
            f"**Δ Risk vs subject:** {escape(format_signed(alternative.deltas.risk_score_delta))}"
        )

        if st.button(
            "Switch to this product",
            key=f"switch_{summary.asin}",
            type="primary",
            use_container_width=True,
        ):
            # set_selected_asin clears last_analysis when the ASIN changes.
            set_selected_asin(summary.asin)
            st.toast(f"Switched to {summary.asin}", icon="🔄")
            st.rerun()


def _render_body() -> None:
    """Render the alternatives grid, the empty state and the comparison table."""
    asin = require_product()

    st.title("🔄 Alternatives")
    st.caption(f"Active subject ASIN: `{asin}`")

    try:
        analysis = get_or_load_analysis(asin)
    except Exception as exc:
        render_api_error(exc)
        return

    show_stub_banner(analysis)

    alternatives = analysis.alternatives
    st.markdown("#### Customers also considered")
    st.caption(
        f"**{format_count(alternatives.usable)}** of "
        f"**{format_count(alternatives.total_related)}** related products are usable "
        "alternatives (a related product counts only when it is itself in the "
        "processed subset with its own analysis)."
    )

    if not alternatives.items:
        reason = alternatives.empty_reason or "the reason was not provided"
        st.info(f"No usable alternatives for this product: {reason}.")
        return

    with st.container(horizontal=True, wrap=True, gap="medium"):
        for alternative in alternatives.items:
            _render_card(alternative)

    st.caption(IMAGE_ATTRIBUTION)

    st.divider()
    st.subheader("Comparison table")
    alternatives_table(alternatives)


def main() -> None:
    """Render the Alternatives page."""
    apply_theme("Alternatives")
    render_header()
    render_sidebar()

    _render_body()
    render_footer()


if __name__ == "__main__":
    main()

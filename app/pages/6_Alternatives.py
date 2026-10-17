"""Page 6: Alternatives.

Renders one card per evidence-ranked alternative (its cached image or a neutral
placeholder, title, its own verdict badge, risk score and comparison deltas against
the subject, plus the better/comparable/worse relation as text and icon) with a
**Switch to this product** button, followed by the compact comparison table. When
there is no usable alternative the page shows why, and how many related products
existed in total. The stub banner is shown whenever the analysis still comes from
the stub engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path so 'app.*' imports resolve
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from app.components import alternatives_table, metric_card, product_image, verdict_badge  # noqa: E402
from app.components.alternatives_table import RELATION_ICONS  # noqa: E402
from app.components.sidebar import render_sidebar, show_stub_banner  # noqa: E402
from app.errors import render_api_error  # noqa: E402
from app.formatting import NOT_AVAILABLE, format_count, format_price, format_signed  # noqa: E402
from app.state import get_or_load_analysis, require_product, set_selected_asin  # noqa: E402
from app.theme import apply_theme  # noqa: E402
from munasib.api.schemas import AlternativeProduct  # noqa: E402

#: Thumbnail edge, in pixels, for the alternative cards.
IMAGE_WIDTH = 100


def _render_card(alternative: AlternativeProduct) -> None:
    """Render one alternative as a card with its deltas and a switch button.

    Args:
        alternative: The alternative to render.
    """
    summary = alternative.summary
    relation = alternative.relation.value
    icon = RELATION_ICONS.get(relation, "ℹ️")

    with st.container(border=True):
        col_image, col_info, col_action = st.columns([1, 4, 1.3])

        with col_image:
            product_image(summary.asin, summary.has_image, width=IMAGE_WIDTH, title=summary.title)

        with col_info:
            st.markdown(f"**{summary.title or NOT_AVAILABLE}**")
            st.markdown(
                f"{icon} **{relation}** than the selected product &nbsp;·&nbsp; "
                f"**Price:** {format_price(summary.price)} &nbsp;·&nbsp; "
                f"**Reviews:** {format_count(summary.n_reviews)}"
            )
            col_risk, col_drisk, col_dev, col_drating = st.columns(4)
            with col_risk:
                metric_card("Risk score", f"{alternative.risk_score:.1f}")
            with col_drisk:
                metric_card(
                    "Δ Risk",
                    format_signed(alternative.deltas.risk_score_delta),
                    help="Alternative minus subject; negative means the alternative is safer.",
                )
            with col_dev:
                metric_card("Δ Evidence", format_signed(alternative.deltas.evidence_strength_delta))
            with col_drating:
                metric_card("Δ Rating", format_signed(alternative.deltas.mean_rating_delta))

        with col_action:
            verdict_badge(alternative.verdict)
            if st.button(
                "Switch to this product",
                key=f"switch_{summary.asin}",
                use_container_width=True,
            ):
                # set_selected_asin clears last_analysis when the ASIN changes.
                set_selected_asin(summary.asin)
                st.toast(f"Switched to {summary.asin}", icon="🔄")
                st.rerun()


def main() -> None:
    """Render the Alternatives page."""
    apply_theme("Alternatives")
    render_sidebar()

    asin = require_product()

    st.title("🔄 6. Alternatives")
    st.caption(f"Active subject ASIN: `{asin}`")

    try:
        analysis = get_or_load_analysis(asin)
    except Exception as exc:
        render_api_error(exc)
        return

    show_stub_banner(analysis)

    alternatives = analysis.alternatives
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

    for alternative in alternatives.items:
        _render_card(alternative)

    st.divider()
    st.subheader("Comparison table")
    alternatives_table(alternatives)


if __name__ == "__main__":
    main()

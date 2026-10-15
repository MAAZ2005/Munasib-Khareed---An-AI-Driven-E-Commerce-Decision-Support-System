"""Verdict badge component: the locked three-label verdict shown as icon + text."""

from __future__ import annotations

from app.theme import get_verdict_style, render_verdict_badge
from munasib.api.schemas import VerdictLabel

#: Rendered when no verdict was computed. Explicit absence, not a verdict.
NO_VERDICT_MESSAGE = "No purchase verdict has been computed for this product yet."


def verdict_badge(label: str | VerdictLabel | None) -> None:
    """Render the consumer verdict badge.

    Supports exactly the three locked labels - CONSIDER, CONSIDER WITH CAUTION
    and CONSUMER RISK - using the theme's styling, which always pairs the color
    with an icon and the full label text so the verdict never depends on color
    alone.

    Args:
        label: Verdict string or VerdictLabel enum, or None when the pipeline
            has not produced a verdict yet.
    """
    import streamlit as st

    if label is None:
        st.info(NO_VERDICT_MESSAGE)
        return

    st.markdown(render_verdict_badge(label), unsafe_allow_html=True)
    st.caption(get_verdict_style(label)["description"])

"""Sentiment breakdown component: labelled weak-supervision counts per class."""

from __future__ import annotations

from plotly import graph_objects as go

from app.components._common import empty_figure, finalize_figure, format_number, placeholder_message
from app.theme import CHART_PALETTE
from munasib.api.schemas import SentimentBreakdown

#: Rendered when no reviews carry a sentiment label.
NO_DATA_MESSAGE = "No classified reviews yet, so there is no sentiment breakdown to show."

#: Weak-supervision mapping, documented here and in every caption that shows
#: these numbers (locked design decision #2): the labels come from the star
#: rating, not from a model prediction.
SUPERVISION_NOTE = "Weak supervision from star ratings: 1-2 stars negative, 3 stars neutral, 4-5 stars positive."


def build_sentiment_breakdown_figure(sentiment: SentimentBreakdown | None) -> go.Figure:
    """Build the horizontal sentiment count bars.

    Args:
        sentiment: Weak-supervision counts and shares, or None when nothing was
            classified.

    Returns:
        A figure with one bar per non-empty class, or the trace-less
        placeholder when there is nothing to plot.
    """
    if sentiment is None or sentiment.total <= 0:
        return empty_figure(NO_DATA_MESSAGE)

    rows: list[tuple[str, int, float | None, str, str]] = []
    if sentiment.unlabelled_count > 0:
        # No share is published for this class, so none is drawn or quoted.
        rows.append(
            (
                "Unlabelled (no rating)",
                sentiment.unlabelled_count,
                None,
                CHART_PALETTE["unlabelled"],
                "no weak label",
            )
        )
    rows.extend(
        [
            ("Negative (1-2 stars)", sentiment.negative_count, sentiment.negative_share, CHART_PALETTE["negative"], "weak label: negative"),
            ("Neutral (3 stars)", sentiment.neutral_count, sentiment.neutral_share, CHART_PALETTE["neutral"], "weak label: neutral"),
            ("Positive (4-5 stars)", sentiment.positive_count, sentiment.positive_share, CHART_PALETTE["positive"], "weak label: positive"),
        ]
    )
    rows = [row for row in rows if row[1] > 0]
    if not rows:
        return empty_figure(NO_DATA_MESSAGE)

    figure = go.Figure(
        go.Bar(
            orientation="h",
            y=[row[0] for row in rows],
            x=[row[1] for row in rows],
            marker_color=[row[3] for row in rows],
            text=[f"{row[1]:,}" if row[2] is None else f"{row[1]:,} ({row[2]:.1%})" for row in rows],
            textposition="outside",
            customdata=[(row[4], row[2]) for row in rows],
            hovertemplate="%{y}: %{x:,} reviews · %{customdata[0]}<extra></extra>",
        )
    )
    finalize_figure(
        figure,
        title="Sentiment breakdown",
        x_title="Reviews",
        height=320,
    )
    return figure


def render_sentiment_breakdown_chart(sentiment: SentimentBreakdown | None) -> None:
    """Render the sentiment bars with the weak-supervision caveat.

    Args:
        sentiment: Weak-supervision counts and shares, or None when nothing was
            classified.
    """
    import streamlit as st

    if sentiment is None:
        st.info(NO_DATA_MESSAGE)
        return

    figure = build_sentiment_breakdown_figure(sentiment)
    if not figure.data:
        st.info(placeholder_message(figure))
        return

    st.plotly_chart(figure, width="stretch")
    st.caption(SUPERVISION_NOTE)
    st.caption(
        f"{format_number(sentiment.total)} reviews counted · "
        f"{format_number(sentiment.unlabelled_count)} carry no star rating and therefore no label."
    )

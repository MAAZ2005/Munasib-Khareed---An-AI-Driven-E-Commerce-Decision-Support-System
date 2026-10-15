"""Star rating distribution component: a five-bar histogram of 1-5 star counts."""

from __future__ import annotations

from plotly import graph_objects as go

from app.components._common import empty_figure, finalize_figure, format_number, placeholder_message
from app.theme import CHART_PALETTE
from munasib.api.schemas import RatingDistribution

#: Rendered when there is no rating evidence. Explicit absence, never zeros.
NO_DATA_MESSAGE = "No star-rating evidence has been recorded for this product yet."

#: The star scale; the schema guarantees a key for every one of them.
STAR_SCALE: tuple[int, ...] = (1, 2, 3, 4, 5)


def _star_color(star: int) -> str:
    """Map a star to its sentiment color, matching the weak-supervision labels.

    Args:
        star: Star value 1-5.

    Returns:
        Palette color: 1-2 stars negative, 3 stars neutral, 4-5 stars positive.
    """
    if star <= 2:
        return CHART_PALETTE["negative"]
    if star == 3:
        return CHART_PALETTE["neutral"]
    return CHART_PALETTE["positive"]


def build_rating_distribution_figure(distribution: RatingDistribution | None) -> go.Figure:
    """Build the star rating histogram.

    Args:
        distribution: Rating histogram, or None when no reviews were rated.

    Returns:
        A figure with one five-point ``bar`` trace, or the trace-less
        placeholder when there is nothing to plot.
    """
    if distribution is None or distribution.total <= 0:
        return empty_figure(NO_DATA_MESSAGE)

    labels = [f"{star} star" if star == 1 else f"{star} stars" for star in STAR_SCALE]
    counts = [distribution.counts.get(star, 0) for star in STAR_SCALE]
    shares = [distribution.shares.get(star, 0.0) for star in STAR_SCALE]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            marker_color=[_star_color(star) for star in STAR_SCALE],
            text=[f"{share:.1%}" for share in shares],
            textposition="outside",
            customdata=shares,
            hovertemplate="%{x}: %{y:,} reviews (%{customdata:.1%} of rated reviews)<extra></extra>",
        )
    )
    finalize_figure(
        figure,
        title="Star rating distribution",
        x_title="Star rating",
        y_title="Rated reviews",
        height=340,
    )
    figure.update_yaxes(tickformat=",.0f")
    return figure


def render_rating_distribution_chart(distribution: RatingDistribution | None) -> None:
    """Render the star histogram with a real-numbers caption.

    Args:
        distribution: Rating histogram, or None when no reviews were rated.
    """
    import streamlit as st

    if distribution is None:
        st.info(NO_DATA_MESSAGE)
        return

    figure = build_rating_distribution_figure(distribution)
    if not figure.data:
        st.info(placeholder_message(figure))
        return

    st.plotly_chart(figure, width="stretch")
    mean_note = f"{distribution.mean_rating:.2f}" if distribution.mean_rating is not None else "not available (no ratings)"
    st.caption(f"{format_number(distribution.total)} rated reviews · mean rating {mean_note}")

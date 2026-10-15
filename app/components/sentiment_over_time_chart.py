"""Sentiment over time component: monthly positive/neutral/negative share lines."""

from __future__ import annotations

from plotly import graph_objects as go

from app.components._common import (
    add_note,
    empty_figure,
    finalize_figure,
    format_number,
    placeholder_message,
)
from app.theme import CHART_PALETTE
from munasib.api.schemas import TemporalTrend

#: Rendered when the product has no period with at least one classified review.
NO_DATA_MESSAGE = "This product has no period with classified reviews, so no sentiment trend can be drawn."

#: Order of the plotted series; the legend labels carry the same text.
_SERIES: tuple[tuple[str, str], ...] = (
    ("positive_share", "Positive share"),
    ("neutral_share", "Neutral share"),
    ("negative_share", "Negative share"),
)

_COLOR_KEYS = {"positive_share": "positive", "neutral_share": "neutral", "negative_share": "negative"}


def build_sentiment_over_time_figure(trend: TemporalTrend | None) -> go.Figure:
    """Build the per-period sentiment share lines.

    Args:
        trend: Period buckets plus the slope, or None when there are no
            classified reviews at all.

    Returns:
        A figure with one trace per plotted share, or the trace-less
        placeholder when there are no buckets.
    """
    if trend is None or not trend.buckets:
        return empty_figure(NO_DATA_MESSAGE)

    periods = [bucket.period_start for bucket in trend.buckets]
    figure = go.Figure(
        [
            go.Scatter(
                x=periods,
                y=[getattr(bucket, field) for bucket in trend.buckets],
                mode="lines+markers",
                name=label,
                line={"color": CHART_PALETTE[_COLOR_KEYS[field]], "width": 2},
                marker={"size": 5},
                hovertemplate="%{x|%b %Y}: %{y:.1%} of classified reviews<extra>" + label + "</extra>",
            )
            for field, label in _SERIES
        ]
    )
    finalize_figure(
        figure,
        title=f"Sentiment share by {trend.period}",
        x_title="Period start (UTC)",
        y_title="Share of classified reviews",
        height=360,
        show_legend=True,
    )
    figure.update_yaxes(range=[0.0, 1.0], tickformat=".0%")

    slope = trend.positive_share_slope
    if slope is None:
        note = (
            f"Positive-share slope: not computed, only {len(trend.buckets)} period(s) "
            "contain classified reviews."
        )
    else:
        direction = "improved" if slope > 0 else "declined" if slope < 0 else "held steady"
        note = (
            f"Least-squares slope of the positive share: {slope:+.6f} per {trend.period} "
            f"(positive sentiment {direction} over {len(trend.buckets)} periods)."
        )
    add_note(figure, note)
    return figure

def render_sentiment_over_time_chart(trend: TemporalTrend | None) -> None:
    """Render the trend lines; the slope is drawn as figure text.

    Args:
        trend: Period buckets plus the slope, or None when there are no
            classified reviews at all.
    """
    import streamlit as st

    if trend is None:
        st.info(NO_DATA_MESSAGE)
        return

    figure = build_sentiment_over_time_figure(trend)
    if not figure.data:
        st.info(placeholder_message(figure))
        return

    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"{format_number(trend.n_reviews_used)} classified reviews across "
        f"{len(trend.buckets)} {trend.period} period(s); periods without reviews are omitted, not interpolated."
    )

"""Sentiment breakdown component: labelled counts per class.

The headline counts can come from two sources, and the chart labels say which:
stored model **predictions** (the hybrid service reads
``review_features.predicted_label``) or star-derived **weak labels** (the pure
stub, and every payload produced before predictions existed). Locked design
decision #2 documents the weak mapping; the hybrid payload additionally carries
the star-derived shares as ``star_*_share`` so the two views can be compared.
"""

from __future__ import annotations

from typing import Literal

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

#: Shown when the headline counts are the stored model predictions.
PREDICTION_NOTE = (
    "Headline counts are the stored model's predicted labels; the star-derived "
    "weak shares are shown alongside for comparison."
)

#: Which label family the headline counts come from.
SentimentSource = Literal["weak", "predicted"]


def build_sentiment_breakdown_figure(
    sentiment: SentimentBreakdown | None,
    *,
    source: SentimentSource = "weak",
) -> go.Figure:
    """Build the horizontal sentiment count bars.

    Args:
        sentiment: Sentiment counts and shares, or None when nothing was
            classified.
        source: ``"weak"`` when the counts are star-derived weak labels,
            ``"predicted"`` when they are stored model predictions. Only the
            labels and hover text change; the numbers are the payload's.

    Returns:
        A figure with one bar per non-empty class, or the trace-less
        placeholder when there is nothing to plot.
    """
    if sentiment is None or sentiment.total <= 0:
        return empty_figure(NO_DATA_MESSAGE)

    if source == "predicted":
        unlabelled_label = "Unlabelled (no prediction)"
        negative_label, neutral_label, positive_label = "Negative", "Neutral", "Positive"
        unlabelled_hover = "no predicted label"
        label_word = "predicted label"
    else:
        unlabelled_label = "Unlabelled (no rating)"
        negative_label = "Negative (1-2 stars)"
        neutral_label = "Neutral (3 stars)"
        positive_label = "Positive (4-5 stars)"
        unlabelled_hover = "no weak label"
        label_word = "weak label"

    rows: list[tuple[str, int, float | None, str, str]] = []
    if sentiment.unlabelled_count > 0:
        # No share is published for this class, so none is drawn or quoted.
        rows.append(
            (
                unlabelled_label,
                sentiment.unlabelled_count,
                None,
                CHART_PALETTE["unlabelled"],
                unlabelled_hover,
            )
        )
    rows.extend(
        [
            (
                negative_label,
                sentiment.negative_count,
                sentiment.negative_share,
                CHART_PALETTE["negative"],
                f"{label_word}: negative",
            ),
            (
                neutral_label,
                sentiment.neutral_count,
                sentiment.neutral_share,
                CHART_PALETTE["neutral"],
                f"{label_word}: neutral",
            ),
            (
                positive_label,
                sentiment.positive_count,
                sentiment.positive_share,
                CHART_PALETTE["positive"],
                f"{label_word}: positive",
            ),
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


def render_sentiment_breakdown_chart(
    sentiment: SentimentBreakdown | None,
    *,
    source: SentimentSource = "weak",
) -> None:
    """Render the sentiment bars with the caption that matches their source.

    Args:
        sentiment: Sentiment counts and shares, or None when nothing was
            classified.
        source: ``"weak"`` for star-derived weak labels (the stub),
            ``"predicted"`` for stored model predictions (the hybrid service).
    """
    import streamlit as st

    if sentiment is None:
        st.info(NO_DATA_MESSAGE)
        return

    figure = build_sentiment_breakdown_figure(sentiment, source=source)
    if not figure.data:
        st.info(placeholder_message(figure))
        return

    st.plotly_chart(figure, width="stretch")
    if source == "predicted":
        st.caption(PREDICTION_NOTE)
        star_shares = (
            sentiment.star_negative_share,
            sentiment.star_neutral_share,
            sentiment.star_positive_share,
        )
        if all(share is not None for share in star_shares):
            st.caption(
                "Star-derived weak shares for comparison: "
                f"{sentiment.star_negative_share:.1%} negative, "
                f"{sentiment.star_neutral_share:.1%} neutral, "
                f"{sentiment.star_positive_share:.1%} positive."
            )
        st.caption(
            f"{format_number(sentiment.total)} reviews counted · "
            f"{format_number(sentiment.unlabelled_count)} have no stored prediction."
        )
        return

    st.caption(SUPERVISION_NOTE)
    st.caption(
        f"{format_number(sentiment.total)} reviews counted · "
        f"{format_number(sentiment.unlabelled_count)} carry no star rating and therefore no label."
    )

"""Aspect bar component: positive versus negative aspects with mention counts."""

from __future__ import annotations

from collections.abc import Sequence

from plotly import graph_objects as go

from app.components._common import empty_figure, finalize_figure, placeholder_message
from app.theme import CHART_PALETTE
from munasib.api.schemas import AspectInsight, AspectPolarity

#: Rendered when the aspect extractor found nothing worth showing.
NO_DATA_MESSAGE = "No dominant review aspects were detected for this product."

#: Polarity legend labels; the bar color repeats what the legend already says,
#: so the chart never depends on color alone.
_POLARITY_LABELS: dict[str, str] = {
    AspectPolarity.POSITIVE.value: "Positive aspect",
    AspectPolarity.NEGATIVE.value: "Negative aspect",
    AspectPolarity.MIXED.value: "Mixed aspect",
}

#: Draw order of the polarity traces.
_POLARITY_ORDER: tuple[str, ...] = (
    AspectPolarity.POSITIVE.value,
    AspectPolarity.NEGATIVE.value,
    AspectPolarity.MIXED.value,
)


def build_aspect_bar_figure(aspects: Sequence[AspectInsight] | None) -> go.Figure:
    """Build the horizontal aspect bars, most-mentioned first.

    The y axis carries the aspect names and the legend carries the polarity, so
    the polarity is readable as text; the bar length is the mention count.

    Args:
        aspects: Top aspects, or None / empty when none were extracted.

    Returns:
        A figure with one trace per polarity present, or the trace-less
        placeholder when there are no aspects.
    """
    if not aspects:
        return empty_figure(NO_DATA_MESSAGE)

    ordered = sorted(aspects, key=lambda item: (-item.mention_count, item.aspect))
    traces: list[go.Bar] = []
    for polarity in _POLARITY_ORDER:
        group = [item for item in ordered if item.polarity.value == polarity]
        if not group:
            continue
        traces.append(
            go.Bar(
                orientation="h",
                y=[item.aspect for item in group],
                x=[item.mention_count for item in group],
                name=_POLARITY_LABELS[polarity],
                marker_color=CHART_PALETTE[polarity],
                text=[f"{item.mention_count:,} reviews" for item in group],
                textposition="outside",
                customdata=[[item.aspect, item.polarity.value, len(item.examples)] for item in group],
                hovertemplate=(
                    "%{y} · %{customdata[1]} · %{x:,} reviews mention it · "
                    "%{customdata[2]} verbatim excerpt(s) available<extra></extra>"
                ),
            )
        )

    figure = go.Figure(traces)
    finalize_figure(
        figure,
        title="Top review aspects by mention count",
        x_title="Reviews mentioning the aspect",
        height=max(320, 60 + 34 * len(ordered)),
        show_legend=len(traces) > 1,
    )
    # Draw the most-mentioned aspect at the top of the horizontal axis.
    figure.update_layout(barmode="overlay")
    figure.update_yaxes(categoryorder="array", categoryarray=list(reversed([item.aspect for item in ordered])))
    return figure


def render_aspect_bar_chart(aspects: Sequence[AspectInsight] | None) -> None:
    """Render the aspect bars.

    Args:
        aspects: Top aspects, or None / empty when none were extracted.
    """
    import streamlit as st

    figure = build_aspect_bar_figure(aspects)
    if not figure.data:
        st.info(placeholder_message(figure))
        return

    st.plotly_chart(figure, width="stretch")
    st.caption(f"{len(aspects or ())} aspects shown, ordered by how many reviews mention them.")

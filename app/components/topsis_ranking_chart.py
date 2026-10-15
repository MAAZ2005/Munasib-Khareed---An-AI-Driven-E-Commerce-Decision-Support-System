"""TOPSIS ranking component: per-criterion contributions with closeness and rank."""

from __future__ import annotations

from plotly import graph_objects as go

from app.components._common import add_note, empty_figure, finalize_figure, placeholder_message
from app.theme import CHART_PALETTE
from munasib.api.schemas import TopsisCriterion, TopsisResult

#: Rendered when the decision stage produced no ranking.
NO_DATA_MESSAGE = "No TOPSIS ranking was produced for this product yet."

#: ``TopsisCriterion.direction`` is one of these two strings.
_DIRECTION_BENEFIT = "benefit"
_DIRECTION_COST = "cost"


def _trace(criteria: list[TopsisCriterion], direction: str, name: str) -> go.Bar | None:
    """Build one contribution bar trace for a single criterion direction.

    Args:
        criteria: The criteria of this direction, in configuration order.
        direction: ``benefit`` or ``cost``; selects the palette color.
        name: Legend label explaining what the direction means.

    Returns:
        The trace, or None when no criterion carries this direction, so that no
        empty legend entry is created.
    """
    if not criteria:
        return None
    return go.Bar(
        x=[criterion.name.value for criterion in criteria],
        y=[criterion.contribution for criterion in criteria],
        name=name,
        marker_color=CHART_PALETTE["benefit" if direction == _DIRECTION_BENEFIT else "cost"],
        text=[f"{criterion.contribution:.3f}" for criterion in criteria],
        textposition="outside",
        customdata=[[criterion.value, criterion.normalized_value, criterion.weight] for criterion in criteria],
        hovertemplate=(
            "%{x}<br>contribution %{y:.3f} · subject value %{customdata[0]:.3f} · "
            "normalised %{customdata[1]:.3f} · weight %{customdata[2]:.3f}<extra></extra>"
        ),
    )


def build_topsis_ranking_figure(topsis: TopsisResult | None) -> go.Figure:
    """Build the per-criterion TOPSIS contribution bars.

    Args:
        topsis: The TOPSIS result, or None when no ranking was produced.

    Returns:
        A figure with up to two traces (benefit and cost criteria), or the
        trace-less placeholder when there is no ranking.
    """
    if topsis is None or not topsis.criteria:
        return empty_figure(NO_DATA_MESSAGE)

    traces = [
        trace
        for trace in (
            _trace(
                [c for c in topsis.criteria if c.direction == _DIRECTION_BENEFIT],
                _DIRECTION_BENEFIT,
                "Benefit criterion (higher is better)",
            ),
            _trace(
                [c for c in topsis.criteria if c.direction == _DIRECTION_COST],
                _DIRECTION_COST,
                "Cost criterion (higher is worse)",
            ),
        )
        if trace is not None
    ]
    if not traces:
        return empty_figure(NO_DATA_MESSAGE)

    figure = go.Figure(traces)
    finalize_figure(
        figure,
        title="TOPSIS contribution per criterion",
        x_title="Criterion (subject product)",
        y_title="Weighted contribution to closeness (0-1)",
        height=400,
        show_legend=len(traces) > 1,
    )
    figure.update_yaxes(range=[0.0, 1.0])
    add_note(
        figure,
        f"Closeness {topsis.closeness:.3f} of 1.000 · "
        f"rank {topsis.rank} of {topsis.ranked_count} · "
        f"safer than {topsis.percentile:.1f}% of the ranked set",
    )
    return figure


def render_topsis_ranking_chart(topsis: TopsisResult | None) -> None:
    """Render the contribution bars; closeness, rank and percentile are in the
    figure text and repeated as a caption.

    Args:
        topsis: The TOPSIS result, or None when no ranking was produced.
    """
    import streamlit as st

    if topsis is None:
        st.info(NO_DATA_MESSAGE)
        return

    figure = build_topsis_ranking_figure(topsis)
    if not figure.data:
        st.info(placeholder_message(figure))
        return

    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"Each bar is one criterion's weighted contribution to the closeness coefficient "
        f"({topsis.closeness:.3f}); the subject ranks {topsis.rank} of {topsis.ranked_count} "
        f"products ({topsis.peer_count} sampled category peer(s) included)."
    )

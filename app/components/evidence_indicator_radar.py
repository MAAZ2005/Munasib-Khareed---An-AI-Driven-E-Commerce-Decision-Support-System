"""Evidence indicator radar: normalised evidence strengths on a 0-1 polar axis."""

from __future__ import annotations

from collections.abc import Sequence

from plotly import graph_objects as go

from app.components._common import empty_figure, finalize_figure, placeholder_message
from app.theme import CHART_PALETTE
from munasib.api.schemas import EvidenceIndicator, EvidenceIndicators, IndicatorDirection

#: Rendered when the product carries no evidence indicators at all.
NO_DATA_MESSAGE = "No evidence-strength indicators were computed for this product."

#: Suffix appended to the axis label of a cost-direction indicator, so a reader
#: cannot mistake "high is good" for "high is bad" on the same ring.
COST_SUFFIX = " (cost)"

#: Legend label of the risk-indicator series.
RISK_LEGEND = "Risk indicators"


def _label(indicator: EvidenceIndicator) -> str:
    """Return the axis label of one indicator, marking cost direction in text.

    Args:
        indicator: Indicator to label.

    Returns:
        The indicator name, with " (cost)" appended when a higher raw value is
        worse for the consumer.
    """
    if indicator.direction == IndicatorDirection.COST:
        return f"{indicator.name}{COST_SUFFIX}"
    return indicator.name


def _trace(indicators: Sequence[EvidenceIndicator], name: str) -> go.Scatterpolar | None:
    """Build one polar trace for a group of indicators.

    Args:
        indicators: The indicators of this series, already in display order.
        name: Legend label for the series.

    Returns:
        The trace, or None when the group is empty, so that no empty legend
        entry is created.
    """
    if not indicators:
        return None
    return go.Scatterpolar(
        r=[item.value for item in indicators],
        theta=[_label(item) for item in indicators],
        mode="lines+markers",
        fill="toself",
        name=name,
        line={"color": CHART_PALETTE["benefit"], "width": 2},
        marker={
            "size": 8,
            "color": [
                CHART_PALETTE["cost"] if item.direction == IndicatorDirection.COST else CHART_PALETTE["benefit"]
                for item in indicators
            ],
        },
        text=[
            f"{item.name} · direction {item.direction.value} · "
            + ("no raw value" if item.raw_value is None else f"raw {item.raw_value:g}")
            + f" · {item.description}"
            for item in indicators
        ],
        hovertemplate="%{text}<br>normalised value %{r:.2f} of 1.00<extra></extra>",
    )


def build_evidence_indicator_radar_figure(
    evidence: EvidenceIndicators | None,
    risk_indicators: Sequence[EvidenceIndicator] = (),
) -> go.Figure:
    """Build the evidence strength radar.

    Args:
        evidence: Evidence indicators, or None when none were computed.
        risk_indicators: Optional risk indicators plotted on the same 0-1 ring;
            they get their own legend entry.

    Returns:
        A figure with one trace per non-empty series, or the trace-less
        placeholder when there is nothing to plot.
    """
    fixed: list[EvidenceIndicator] = []
    details: list[EvidenceIndicator] = []
    if evidence is not None:
        fixed = [evidence.verified_purchase, evidence.helpfulness, evidence.review_depth]
        details = list(evidence.details)

    traces = [
        trace
        for trace in (
            _trace(fixed + details, "Evidence strength"),
            _trace(list(risk_indicators), RISK_LEGEND),
        )
        if trace is not None
    ]
    if not traces:
        return empty_figure(NO_DATA_MESSAGE)

    figure = go.Figure(traces)
    finalize_figure(
        figure,
        title="Evidence strength indicators (0 = none, 1 = strongest)",
        height=420,
        show_legend=len(traces) > 1,
    )
    figure.update_layout(
        polar={
            "radialaxis": {
                "range": [0.0, 1.0],
                "tick0": 0.0,
                "dtick": 0.25,
                "tickformat": ".2f",
                "gridcolor": "#e2e8f0",
                "linecolor": CHART_PALETTE["grey"],
            },
            "angularaxis": {"gridcolor": "#e2e8f0", "linecolor": CHART_PALETTE["grey"]},
        },
    )
    return figure


def render_evidence_indicator_radar(
    evidence: EvidenceIndicators | None,
    risk_indicators: Sequence[EvidenceIndicator] = (),
) -> None:
    """Render the radar and explain the cost-direction marker.

    Args:
        evidence: Evidence indicators, or None when none were computed.
        risk_indicators: Optional risk indicators plotted on the same ring.
    """
    import streamlit as st

    figure = build_evidence_indicator_radar_figure(evidence, risk_indicators)
    if not figure.data:
        st.info(placeholder_message(figure))
        return

    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"Indicators marked '{COST_SUFFIX.strip()}' are cost-direction: a higher raw value is worse "
        "for the consumer, so a longer bar there means more risk, not more evidence."
    )
    if evidence is not None:
        st.caption(
            f"Weighted composite evidence strength: {evidence.evidence_strength:.2f} of 1.00, "
            "from the three fixed indicators plus the extra signals listed on the ring."
        )

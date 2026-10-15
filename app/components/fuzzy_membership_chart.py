"""Fuzzy membership component: input memberships beside the aggregated output."""

from __future__ import annotations

from plotly import graph_objects as go
from plotly.subplots import make_subplots

from app.components._common import (
    CHART_MARGIN,
    add_note,
    empty_figure,
    finalize_figure,
    placeholder_message,
)
from app.theme import CHART_PALETTE
from munasib.api.schemas.decision import FUZZY_OUTPUT_TERMS, FuzzyAssessment, MembershipProfile

#: Rendered when the fuzzy stage produced neither memberships nor output degrees.
NO_DATA_MESSAGE = "The fuzzy risk assessment has no membership degrees to display."

#: Color per linguistic term, mapped so that "more risk" always reads warmer.
#: Any term outside this mapping falls back to the neutral palette gray.
_TERM_COLORS: dict[str, str] = {
    "low": "positive",
    "medium": "cost",
    "moderate": "cost",
    "high": "negative",
    "critical": "purple",
}


def _term_color(term: str) -> str:
    """Return the palette color of a linguistic term.

    Args:
        term: Linguistic term, e.g. ``low`` or ``critical``.

    Returns:
        Palette color for the term; gray when the term is unknown to the UI.
    """
    return CHART_PALETTE[_TERM_COLORS.get(term, "grey")]


def _input_terms(memberships: dict[str, MembershipProfile]) -> list[str]:
    """Collect the linguistic terms used by the input variables, in first-seen order.

    Args:
        memberships: Input variable name to its membership profile.

    Returns:
        Terms in the order they first appear, so the legend order is stable for
        a given payload.
    """
    terms: list[str] = []
    for profile in memberships.values():
        for term in profile.term_degrees:
            if term not in terms:
                terms.append(term)
    return terms


def build_fuzzy_membership_figure(fuzzy: FuzzyAssessment | None) -> go.Figure:
    """Build the two-panel fuzzy chart.

    The left panel groups the input variables by linguistic term, one trace per
    term; the right panel plots the aggregated output degree per configured
    output term. Both share a 0-1 degree axis.

    Args:
        fuzzy: The fuzzy stage, or None when it has not run.

    Returns:
        A figure with one trace per input term plus the output trace, or the
        trace-less placeholder when there is nothing to plot.
    """
    if fuzzy is None or (not fuzzy.input_memberships and not fuzzy.output_degrees):
        return empty_figure(NO_DATA_MESSAGE)

    variables = list(fuzzy.input_memberships)
    terms = _input_terms(fuzzy.input_memberships)
    output_terms = [term for term in FUZZY_OUTPUT_TERMS if term in fuzzy.output_degrees]

    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("Input memberships", "Output degrees"),
        column_widths=[0.6, 0.4],
    )
    for term in terms:
        figure.add_trace(
            go.Bar(
                x=variables,
                y=[fuzzy.input_memberships[var].term_degrees.get(term, 0.0) for var in variables],
                name=term,
                marker_color=_term_color(term),
                hovertemplate="%{x}: %{y:.2f} degree of " + term + "<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if output_terms:
        figure.add_trace(
            go.Bar(
                x=output_terms,
                y=[fuzzy.output_degrees[term] for term in output_terms],
                name="Output degree",
                marker={"color": [_term_color(term) for term in output_terms]},
                text=[f"{fuzzy.output_degrees[term]:.2f}" for term in output_terms],
                textposition="outside",
                hovertemplate="%{x}: aggregated degree %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    finalize_figure(figure, title="Fuzzy risk memberships", height=400, show_legend=True)
    figure.update_layout(margin=CHART_MARGIN)
    # finalize_figure styles the first subplot's axes; extend the same grid and
    # axis line to every axis of the multi-panel figure.
    figure.update_xaxes(gridcolor="#e2e8f0", linecolor=CHART_PALETTE["grey"])
    figure.update_yaxes(gridcolor="#e2e8f0", linecolor=CHART_PALETTE["grey"])
    figure.update_xaxes(title_text="Input variable", tickangle=20, row=1, col=1)
    figure.update_xaxes(title_text="Fuzzy output term", row=1, col=2)
    figure.update_yaxes(title_text="Degree (0-1)", range=[0.0, 1.0], tickformat=".0%", row=1, col=1)
    figure.update_yaxes(range=[0.0, 1.0], tickformat=".0%", row=1, col=2)
    add_note(
        figure,
        f"Defuzzified risk: {fuzzy.defuzzified:.2f} of 100 "
        f"({fuzzy.defuzzification_method} method) · {len(fuzzy.fired_rules)} rule(s) fired",
    )
    return figure


def render_fuzzy_membership_chart(fuzzy: FuzzyAssessment | None) -> None:
    """Render the fuzzy memberships chart.

    Args:
        fuzzy: The fuzzy stage, or None when it has not run.
    """
    import streamlit as st

    if fuzzy is None:
        st.info(NO_DATA_MESSAGE)
        return

    figure = build_fuzzy_membership_figure(fuzzy)
    if not figure.data:
        st.info(placeholder_message(figure))
        return

    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Degrees are how strongly each linguistic term applies (0-1); the defuzzified value "
        "is the crisp 0-100 risk the fuzzy stage hands to TOPSIS."
    )

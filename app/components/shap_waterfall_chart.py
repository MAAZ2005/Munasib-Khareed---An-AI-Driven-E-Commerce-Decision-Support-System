"""SHAP waterfall component: base value, per-feature contributions, prediction."""

from __future__ import annotations

from plotly import graph_objects as go

from app.components._common import empty_figure, finalize_figure, format_number, placeholder_message
from app.theme import CHART_PALETTE
from munasib.api.schemas import ShapContribution, ShapExplanation

#: Rendered when the surrogate produced no contributions.
NO_DATA_MESSAGE = "No SHAP contributions were produced for this product yet."

#: Labels of the two bookend bars.
BASE_LABEL = "Base value"
PREDICTION_LABEL = "Prediction (risk score)"


def _prediction(explanation: ShapExplanation) -> float:
    """Return the prediction the base value and the contributions add up to.

    The surrogate is additive, so the last bar is the base value plus every
    contribution; it is derived, never typed in.

    Args:
        explanation: The SHAP explanation of the product.

    Returns:
        The base value plus the sum of all contributions, in risk-score units.
    """
    return explanation.base_value + sum(item.contribution for item in explanation.contributions)


def build_shap_waterfall_figure(explanation: ShapExplanation | None) -> go.Figure:
    """Build the SHAP waterfall.

    Args:
        explanation: SHAP explanation, or None when the surrogate explained
            nothing.

    Returns:
        A figure with one ``waterfall`` trace, or the trace-less placeholder
        when there are no contributions to draw.
    """
    if explanation is None or not explanation.contributions:
        return empty_figure(NO_DATA_MESSAGE)

    contributions: list[ShapContribution] = explanation.contributions
    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            x=[BASE_LABEL, *[item.feature for item in contributions], PREDICTION_LABEL],
            y=[explanation.base_value, *[item.contribution for item in contributions], _prediction(explanation)],
            measure=["absolute", *["relative"] * len(contributions), "total"],
            text=[
                f"{explanation.base_value:+.2f}",
                *[f"{item.contribution:+.2f}" for item in contributions],
                f"{_prediction(explanation):+.2f}",
            ],
            textposition="outside",
            # A contribution that raises risk is bad news, so it reads vermillion;
            # one that lowers risk reads green. The signed text repeats it.
            increasing={"marker": {"color": CHART_PALETTE["negative"]}},
            decreasing={"marker": {"color": CHART_PALETTE["positive"]}},
            totals={"marker": {"color": CHART_PALETTE["grey"], "line": {"color": CHART_PALETTE["grey"], "width": 1}}},
            connector={"line": {"color": CHART_PALETTE["grey"], "dash": "dot"}},
            customdata=[[item.feature, item.value, item.contribution] for item in contributions],
            hovertemplate=(
                "%{x}<br>contribution %{y:+.2f} risk points<extra></extra>"
            ),
        )
    )
    finalize_figure(
        figure,
        title="Why the surrogate predicted this risk score",
        x_title="Feature (subject product)",
        y_title="Risk-score points",
        height=420,
    )
    figure.update_yaxes(zeroline=True, zerolinecolor=CHART_PALETTE["grey"])
    return figure


def render_shap_waterfall_chart(explanation: ShapExplanation | None) -> None:
    """Render the waterfall, the reasons and the surrogate fidelity.

    Args:
        explanation: SHAP explanation, or None when the surrogate explained
            nothing.
    """
    import streamlit as st

    if explanation is None:
        st.info(NO_DATA_MESSAGE)
        return

    figure = build_shap_waterfall_figure(explanation)
    if not figure.data:
        st.info(placeholder_message(figure))
        return

    st.plotly_chart(figure, width="stretch")

    fidelity = explanation.surrogate_fidelity
    meets = "meets" if fidelity.meets_minimum else "does not meet"
    st.caption(
        f"Base value {explanation.base_value:.2f} plus the contributions above gives "
        f"{_prediction(explanation):+.2f}. Surrogate fidelity: R² {fidelity.r2:.3f}, "
        f"MAE {fidelity.mae:.2f} on {format_number(fidelity.n_training_points)} held-out points; "
        f"it {meets} the configured minimum."
    )

    if not explanation.reasons:
        st.info("The pipeline produced no written reasons for this explanation.")
        return

    st.markdown("**Reasons, each citing its own feature and value**")
    for reason in explanation.reasons:
        value = "" if reason.raw_value is None else f" = {format_number(reason.raw_value)}"
        st.markdown(f"- `{reason.feature}`{value} — {reason.text}")

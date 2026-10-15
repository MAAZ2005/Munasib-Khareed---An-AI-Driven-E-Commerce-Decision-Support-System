"""Consumer risk gauge: a 0-100 Plotly indicator banded by the verdict thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from plotly import graph_objects as go

from app.components._common import CHART_MARGIN, empty_figure, finalize_figure, placeholder_message
from app.theme import CHART_PALETTE
from munasib.core.config import ConfigError, get_settings

#: Rendered when the pipeline has not produced a score. Explicit absence.
NO_RISK_MESSAGE = "No consumer risk score has been produced for this product yet."

#: Rendered when the thresholds file cannot be read. The gauge never falls back
#: to invented bands, so the score is simply not shown.
NO_THRESHOLDS_MESSAGE = (
    "Verdict thresholds could not be read from configs/verdict_thresholds.yaml, "
    "so the risk gauge cannot be banded."
)

#: The risk scale is 0.0-100.0 (see munasib.api.schemas.common.RiskScore).
SCALE_MIN = 0.0
SCALE_MAX = 100.0


@dataclass(frozen=True)
class RiskBands:
    """The two boundaries that split the risk scale into the three verdicts.

    Attributes:
        consider_max: Upper bound of the CONSIDER band (inclusive).
        risk_min: Lower bound of the CONSUMER RISK band (inclusive). The open
            interval between the two bounds is CONSIDER WITH CAUTION.
        source: Path of the YAML file the bounds were read from.
    """

    consider_max: float
    risk_min: float
    source: str


def _thresholds_path() -> Path:
    """Resolve configs/verdict_thresholds.yaml through the existing settings.

    The band boundaries are never hardcoded in the UI: they come from the same
    ``configs/verdict_thresholds.yaml`` the backend decision module uses, read
    via :func:`munasib.core.config.get_settings`, so a config change moves the
    gauge and the verdict logic together.

    Returns:
        Path of the thresholds file. A relative configured path is resolved
        against the repository root, because Streamlit's working directory is
        not guaranteed to be the directory ``configs/`` lives in.
    """
    repo_root = Path(__file__).resolve().parents[2]
    try:
        settings = get_settings()
    except ConfigError:
        # No (or unrunnable) configs/config.yaml: still read the thresholds file
        # from the checkout rather than substituting a default.
        return repo_root / "configs" / "verdict_thresholds.yaml"
    configured = Path(settings.verdict.thresholds_file)
    return configured if configured.is_absolute() else repo_root / configured


@lru_cache(maxsize=1)
def load_risk_bands() -> RiskBands:
    """Read the verdict band boundaries from configs/verdict_thresholds.yaml.

    Returns:
        The two band boundaries.

    Raises:
        ValueError: If the file is missing or does not carry both bounds, so
            that a broken config is reported instead of silently banded.
    """
    path = _thresholds_path()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    verdicts = raw.get("verdicts", {})
    try:
        consider_max = float(verdicts["consider"]["max_risk_score"])
        risk_min = float(verdicts["consumer_risk"]["min_risk_score"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Verdict thresholds are missing or malformed in {path}") from exc
    if not SCALE_MIN <= consider_max < risk_min <= SCALE_MAX:
        raise ValueError(
            f"Verdict thresholds in {path} do not order as {SCALE_MIN} <= consider_max "
            f"< risk_min <= {SCALE_MAX}: got {consider_max} and {risk_min}"
        )
    return RiskBands(consider_max=consider_max, risk_min=risk_min, source=str(path))


def build_risk_gauge_figure(risk_score: float | None) -> go.Figure:
    """Build the 0-100 consumer risk gauge.

    The band boundaries are read from ``configs/verdict_thresholds.yaml`` and
    the score is clamped to the scale, so a value produced by an older model
    still renders instead of breaking the axis.

    Args:
        risk_score: Consumer risk score, 0 (safest) - 100 (riskiest), or None.

    Returns:
        A figure with one ``indicator`` trace, or the trace-less placeholder
        when the score or the thresholds are unavailable.
    """
    if risk_score is None:
        return empty_figure(NO_RISK_MESSAGE)

    try:
        bands = load_risk_bands()
    except (OSError, ValueError):
        return empty_figure(NO_THRESHOLDS_MESSAGE)

    value = max(SCALE_MIN, min(SCALE_MAX, float(risk_score)))
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " / 100", "valueformat": ".1f", "font": {"size": 32}},
            gauge={
                "axis": {
                    "range": [SCALE_MIN, SCALE_MAX],
                    "tickwidth": 1,
                    "tickcolor": CHART_PALETTE["grey"],
                    "tickvals": [SCALE_MIN, bands.consider_max, bands.risk_min, SCALE_MAX],
                },
                "bar": {"color": CHART_PALETTE["grey"], "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": CHART_PALETTE["grey"],
                "steps": [
                    {
                        "range": [SCALE_MIN, bands.consider_max],
                        "color": CHART_PALETTE["band_consider_fill"],
                    },
                    {
                        "range": [bands.consider_max, bands.risk_min],
                        "color": CHART_PALETTE["band_caution_fill"],
                    },
                    {
                        "range": [bands.risk_min, SCALE_MAX],
                        "color": CHART_PALETTE["band_risk_fill"],
                    },
                ],
            },
        )
    )
    finalize_figure(figure, title="Consumer risk score (higher is riskier)", height=270)
    figure.update_layout(margin={**CHART_MARGIN, "t": 56, "b": 32})
    return figure


def render_risk_gauge(risk_score: float | None) -> None:
    """Render the risk gauge plus the caption and numeric band legend.

    Args:
        risk_score: Consumer risk score 0-100, or None when not computed.
    """
    import streamlit as st

    figure = build_risk_gauge_figure(risk_score)
    if not figure.data:
        st.info(placeholder_message(figure))
        return

    st.plotly_chart(figure, width="stretch")
    st.caption("Higher means riskier.")
    bands = load_risk_bands()
    st.caption(
        "Verdict bands from configs/verdict_thresholds.yaml: "
        f"CONSIDER up to {bands.consider_max:g}, "
        f"CONSIDER WITH CAUTION {bands.consider_max:g}-{bands.risk_min:g}, "
        f"CONSUMER RISK from {bands.risk_min:g} to {SCALE_MAX:g}."
    )

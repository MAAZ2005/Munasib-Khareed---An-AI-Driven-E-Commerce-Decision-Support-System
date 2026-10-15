"""Concern list component: severity-tagged concerns with their supporting numbers."""

from __future__ import annotations

from collections.abc import Sequence

from app.components._common import format_number
from app.theme import CHART_PALETTE
from munasib.api.schemas import Concern, ConcernSeverity

#: Rendered when the verdict carries no concerns. Explicit absence.
NO_CONCERNS_MESSAGE = "No major concerns were recorded for this product."

#: An icon per severity, always shown next to the severity word so the level is
#: never signalled by color alone.
SEVERITY_ICONS: dict[str, str] = {
    ConcernSeverity.HIGH.value: "🔴",
    ConcernSeverity.MEDIUM.value: "🟠",
    ConcernSeverity.LOW.value: "🔵",
}

_FALLBACK_ICON = "ℹ️"


def _severity_chip(severity: ConcernSeverity) -> str:
    """Build the colored severity chip: icon plus the severity word.

    Args:
        severity: Severity of the concern.

    Returns:
        Inline HTML carrying the icon, the upper-cased severity word and its
        palette color.
    """
    key = severity.value
    icon = SEVERITY_ICONS.get(key, _FALLBACK_ICON)
    color = CHART_PALETTE.get(f"severity_{key}", CHART_PALETTE["grey"])
    return f'<span style="color:{color};font-weight:700;">{icon} {key.upper()}</span>'


def _supporting_numbers(concern: Concern) -> str:
    """Format the supporting numbers of one concern, keys in stable order.

    Args:
        concern: Concern whose numbers are quoted.

    Returns:
        A comma-separated ``name = value`` string, or an empty string when the
        concern quotes no numbers.
    """
    return ", ".join(
        f"{name} = {format_number(value)}" for name, value in sorted(concern.supporting_numbers.items())
    )


def concern_list(concerns: Sequence[Concern] | None) -> None:
    """Render the list of major concerns behind a verdict.

    Each concern shows its severity as icon + word + color, its stable code and
    the sentence that quotes the computed numbers, followed by those numbers.

    Args:
        concerns: Concerns in the order the pipeline ranked them (most severe
            first), or None / empty when the verdict is clean.
    """
    import streamlit as st

    if not concerns:
        st.info(NO_CONCERNS_MESSAGE)
        return

    st.markdown(f"**{len(concerns)} major concern(s) behind this verdict**")
    for concern in concerns:
        st.markdown(f"{_severity_chip(concern.severity)} &nbsp; `{concern.code}`", unsafe_allow_html=True)
        st.markdown(f"{concern.text}")
        numbers = _supporting_numbers(concern)
        if numbers:
            st.caption(f"Supporting numbers: {numbers}")

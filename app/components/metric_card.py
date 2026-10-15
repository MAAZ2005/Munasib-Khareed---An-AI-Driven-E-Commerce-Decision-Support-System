"""Metric card component: one labelled number with optional help text and delta."""

from __future__ import annotations

from app.components._common import format_number

#: Rendered when the value is missing. Explicit absence, never a fabricated 0.
NOT_AVAILABLE = "n/a"


def metric_card(
    label: str,
    value: str | int | float | None,
    help: str | None = None,
    delta: str | float | None = None,
) -> None:
    """Render one headline number as a Streamlit metric.

    A ``None`` value renders as "n/a" so a product with no computed metric still
    shows the metric's name and its absence, instead of the card silently
    disappearing or claiming a zero.

    Args:
        label: Metric name shown above the value.
        value: The number (or short string) to display, or None when missing.
        help: Optional longer explanation shown on hover.
        delta: Optional change indicator; a float is formatted the same way as
            the value, None renders no delta.
    """
    import streamlit as st

    if value is None:
        display: str | int | float = NOT_AVAILABLE
    elif isinstance(value, str):
        display = value
    else:
        display = format_number(value)

    delta_display = delta if delta is None or isinstance(delta, str) else format_number(delta)

    st.metric(label=label, value=display, help=help, delta=delta_display)

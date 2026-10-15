"""Private scaffolding shared by the component library.

This is **not** a component: it holds the two pieces every chart reuses - the
placeholder figure a builder returns when there is nothing to plot, and the
title/axis/legend layout that makes the library look uniform - plus the number
formatting shared by the non-chart components.

Two conventions live here:

* **Nothing to show == zero traces.** A builder that receives ``None`` or an
  empty payload returns ``empty_figure(message)``. The renderer then tests
  ``if not figure.data`` and shows ``st.info`` with the very same message,
  which it reads back with :func:`placeholder_message`. The builder and its
  renderer therefore cannot disagree about what counts as empty, and the
  placeholder message exists exactly once per component.

* **No streamlit import.** Builders are pure functions that must stay
  unit-testable, so this module imports only Plotly and the theme.
"""

from __future__ import annotations

from plotly import graph_objects as go

from app.theme import CHART_PALETTE, FONT_FAMILY

#: Layout shared by every chart: room for the title and a long category label
#: without clipping, on a canvas that stays readable in a narrow column.
CHART_MARGIN = {"l": 56, "r": 24, "t": 64, "b": 48, "pad": 4}


def empty_figure(message: str) -> go.Figure:
    """Build the placeholder a builder returns when there is nothing to plot.

    Args:
        message: Human-readable reason, also shown by the renderer as
            ``st.info``. It must state what is missing, not invent a number.

    Returns:
        A figure with **no traces**, which is the signal renderers use to show
        the ``st.info`` fallback instead of a chart.
    """
    figure = go.Figure()
    figure.update_layout(
        annotations=[
            {
                "text": message,
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "xanchor": "center",
                "font": {"size": 13, "color": CHART_PALETTE["grey"]},
            }
        ],
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=200,
        margin=CHART_MARGIN,
    )
    return figure


def placeholder_message(figure: go.Figure) -> str:
    """Read back the reason a builder attached to its placeholder figure.

    Args:
        figure: Figure returned by a builder, presumably with no traces.

    Returns:
        The message carried by the figure's first annotation, or a neutral
        fallback when the figure carries none.
    """
    annotations = getattr(figure.layout, "annotations", None)
    if annotations:
        return str(annotations[0].text)
    return "No data is available for this component yet."


def finalize_figure(
    figure: go.Figure,
    *,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int = 380,
    show_legend: bool = False,
) -> go.Figure:
    """Apply the library-wide title, axis labels, font and margins.

    Axis dictionaries are **merged** into whatever the builder already set, so a
    component can still customize ``range``, ``categoryarray`` or ``radialaxis``
    without repeating the shared styling.

    Args:
        figure: The figure the builder already populated with traces.
        title: Chart title shown top-left.
        x_title: Horizontal axis label, or None when the axis is self-evident.
        y_title: Vertical axis label, or None when the axis is self-evident.
        height: Figure height in pixels; horizontal bar charts raise it to fit
            every category.
        show_legend: Whether to show the series legend.

    Returns:
        The same figure, for chaining.
    """
    axis_line = {"color": CHART_PALETTE["grey"], "width": 1}
    grid = {"color": "#e2e8f0"}
    figure.update_layout(
        title={
            "text": title,
            "x": 0,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "font": {"size": 16},
        },
        font={"family": FONT_FAMILY, "size": 13},
        height=height,
        margin=CHART_MARGIN,
        showlegend=show_legend,
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": x_title, "gridcolor": grid["color"], "linecolor": axis_line["color"], "automargin": True},
        yaxis={"title": y_title, "gridcolor": grid["color"], "linecolor": axis_line["color"], "automargin": True},
    )
    return figure


def add_note(figure: go.Figure, text: str) -> go.Figure:
    """Add a one-line note in the band between the title and the plot area.

    Used for the figure-level numbers a reader needs without scrolling to a
    caption - the trend slope, the TOPSIS closeness and rank, the defuzzified
    fuzzy output. The note sits at the top-left of that band, which the shared
    top margin leaves free at every chart height this library uses.

    Args:
        figure: The figure to annotate.
        text: The note; it must quote real computed numbers only.

    Returns:
        The same figure, for chaining.
    """
    figure.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0.0,
        y=0.92,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font={"size": 12, "color": CHART_PALETTE["grey"]},
    )
    return figure


def format_number(value: float | int | None) -> str:
    """Format a real number for display without inventing precision.

    Whole numbers keep no decimals, other values keep up to four decimals with
    trailing zeros trimmed, and a non-zero value smaller than ``0.0001`` is
    shown in scientific notation rather than being rounded down to ``0``.

    Args:
        value: The number to format, or None when the backend did not compute
            one.

    Returns:
        A display string; ``"n/a"`` when the value is missing.
    """
    if value is None:
        return "n/a"
    number = float(value)
    if number == 0:
        return "0"
    if number.is_integer():
        return f"{int(number):,}"
    if abs(number) < 0.0001:
        return f"{number:,.2e}"
    return f"{number:,.4f}".rstrip("0").rstrip(".")

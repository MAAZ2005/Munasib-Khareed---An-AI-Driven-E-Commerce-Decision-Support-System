"""End-to-end tests for the component gallery page.

The gallery is the one page that renders every component, so running it under
``AppTest`` is the cheapest check that a component still works when it is fed a
real payload - and that every empty state degrades to ``st.info`` instead of an
exception. The API client is patched package-wide by ``tests/e2e/conftest.py``,
so these runs never open a socket.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

#: The gallery lives in ``app/pages`` so it is reachable from the sidebar.
GALLERY_PATH = Path(__file__).resolve().parents[2] / "app" / "pages" / "_component_gallery.py"

#: One Plotly chart per chart component: gauge, rating, sentiment, trend,
#: aspects, radar, fuzzy, TOPSIS and SHAP.
SAMPLE_CHART_COUNT = 9

#: Empty-mode ``st.info`` calls: verdict, gauge, rating, sentiment, trend,
#: aspects, radar, fuzzy, TOPSIS, SHAP, concerns and the alternatives
#: ``empty_reason`` - plus the sidebar's own "no product selected yet".
EMPTY_INFO_COUNT = 13


def run_gallery() -> AppTest:
    """Run the gallery page with its default (sample-data) toggle.

    The generous timeout only covers the first run, which pays for importing
    Plotly and validating the schema example; later runs are far quicker.

    Returns:
        The executed AppTest context.
    """
    app = AppTest.from_file(str(GALLERY_PATH))
    app.run(timeout=60)
    return app


def toggle_empty_data(app: AppTest) -> AppTest:
    """Switch the gallery's single toggle to the empty-data mode and re-run.

    Args:
        app: The gallery AppTest context.

    Returns:
        The re-run context.
    """
    app.checkbox[0].set_value(True).run(timeout=60)
    return app


def test_gallery_renders_every_component_without_an_exception() -> None:
    """Sample mode draws all nine charts and flags the data as illustrative."""
    app = run_gallery()

    assert not app.exception
    assert any("Sample data: component gallery (illustrative values, not results)" in w.value for w in app.warning)
    assert len(app.checkbox) == 1
    assert len(app.get("plotly_chart")) == SAMPLE_CHART_COUNT


def test_gallery_toggle_switches_every_component_to_empty_data() -> None:
    """Empty mode replaces every chart with an info message, not a crash."""
    app = toggle_empty_data(run_gallery())

    assert not app.exception
    assert len(app.get("plotly_chart")) == 0
    assert len(app.info) == EMPTY_INFO_COUNT


def test_gallery_empty_mode_reports_missing_metrics_as_not_available() -> None:
    """The metric card shows n/a rather than a fabricated zero."""
    app = toggle_empty_data(run_gallery())

    assert [metric.value for metric in app.metric] == ["n/a", "n/a"]


@pytest.mark.parametrize("empty", [False, True], ids=["sample-data", "empty-data"])
def test_gallery_keeps_its_banner_in_both_modes(empty: bool) -> None:
    """The illustrative-data banner never disappears behind the toggle."""
    app = run_gallery()
    if empty:
        app = toggle_empty_data(app)

    assert not app.exception
    assert any("Sample data: component gallery" in w.value for w in app.warning)

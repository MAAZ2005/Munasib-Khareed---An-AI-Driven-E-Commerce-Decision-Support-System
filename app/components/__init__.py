"""Reusable UI components for the Munasib Khareed Streamlit frontend.

Every chart component follows the same design rule:

* ``build_<name>_figure(data)`` is a **pure** builder: it imports no
  ``streamlit``, calls no API, reads no session state, and returns a
  ``plotly.graph_objects.Figure``. The same input always produces the same
  figure, so the builders are unit-tested directly.
* ``render_<name>(data)`` is the thin wrapper that adds titles, axis labels and
  captions and hands the figure to ``st.plotly_chart``.

The non-chart components (``metric_card``, ``verdict_badge``, ``concern_list``
and ``alternatives_table``) expose only their ``render`` function.

Empty or partial data never crashes a component: a builder returns a figure
with no traces and its renderer shows ``st.info`` with that figure's message,
so a component never invents a value it was not given.
"""

from app.components.alternatives_table import alternatives_table, build_alternatives_rows
from app.components.aspect_bar_chart import (
    NO_DATA_MESSAGE as ASPECT_NO_DATA_MESSAGE,
)
from app.components.aspect_bar_chart import (
    build_aspect_bar_figure,
    render_aspect_bar_chart,
)
from app.components.concern_list import concern_list
from app.components.evidence_indicator_radar import (
    COST_SUFFIX as EVIDENCE_COST_SUFFIX,
)
from app.components.evidence_indicator_radar import (
    build_evidence_indicator_radar_figure,
    render_evidence_indicator_radar,
)
from app.components.fuzzy_membership_chart import (
    build_fuzzy_membership_figure,
    render_fuzzy_membership_chart,
)
from app.components.metric_card import metric_card
from app.components.product_image import (
    IMAGE_ATTRIBUTION as PRODUCT_IMAGE_ATTRIBUTION,
)
from app.components.product_image import (
    NO_IMAGE_LABEL,
    has_renderable_image,
    placeholder_initial,
    product_image,
)
from app.components.rating_distribution_chart import (
    NO_DATA_MESSAGE as RATING_NO_DATA_MESSAGE,
)
from app.components.rating_distribution_chart import (
    build_rating_distribution_figure,
    render_rating_distribution_chart,
)
from app.components.risk_gauge import (
    RiskBands,
    build_risk_gauge_figure,
    load_risk_bands,
    render_risk_gauge,
)
from app.components.sentiment_breakdown_chart import (
    SUPERVISION_NOTE,
    build_sentiment_breakdown_figure,
    render_sentiment_breakdown_chart,
)
from app.components.sentiment_over_time_chart import (
    build_sentiment_over_time_figure,
    render_sentiment_over_time_chart,
)
from app.components.shap_waterfall_chart import (
    build_shap_waterfall_figure,
    render_shap_waterfall_chart,
)
from app.components.topsis_ranking_chart import (
    build_topsis_ranking_figure,
    render_topsis_ranking_chart,
)
from app.components.verdict_badge import verdict_badge

__all__ = [
    "ASPECT_NO_DATA_MESSAGE",
    "EVIDENCE_COST_SUFFIX",
    "NO_IMAGE_LABEL",
    "PRODUCT_IMAGE_ATTRIBUTION",
    "RATING_NO_DATA_MESSAGE",
    "SUPERVISION_NOTE",
    "RiskBands",
    "alternatives_table",
    "build_alternatives_rows",
    "build_aspect_bar_figure",
    "build_evidence_indicator_radar_figure",
    "build_fuzzy_membership_figure",
    "build_rating_distribution_figure",
    "build_risk_gauge_figure",
    "build_shap_waterfall_figure",
    "build_sentiment_breakdown_figure",
    "build_sentiment_over_time_figure",
    "build_topsis_ranking_figure",
    "concern_list",
    "has_renderable_image",
    "load_risk_bands",
    "metric_card",
    "placeholder_initial",
    "product_image",
    "render_aspect_bar_chart",
    "render_evidence_indicator_radar",
    "render_fuzzy_membership_chart",
    "render_rating_distribution_chart",
    "render_risk_gauge",
    "render_shap_waterfall_chart",
    "render_sentiment_breakdown_chart",
    "render_sentiment_over_time_chart",
    "render_topsis_ranking_chart",
    "verdict_badge",
]

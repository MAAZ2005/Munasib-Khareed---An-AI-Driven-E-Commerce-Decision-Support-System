"""Component gallery: every reusable component rendered on one page.

This page exists so the component library can be reviewed in isolation before
the dashboard pages are assembled. It calls no endpoint: every number it shows
comes from the example payloads already shipped with the API schemas (the same
``ANALYSIS_EXAMPLE`` / ``ALTERNATIVES_EMPTY_EXAMPLE`` dictionaries that
``scripts/export_schemas.py`` publishes), so nothing here is hand-typed and
nothing here is a result.

The "empty data" toggle re-renders the page with missing payloads - None, empty
lists and zero-count schemas - to check each component's empty state.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path so 'app.*' imports resolve
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    alternatives_table,
    concern_list,
    metric_card,
    render_aspect_bar_chart,
    render_evidence_indicator_radar,
    render_fuzzy_membership_chart,
    render_rating_distribution_chart,
    render_risk_gauge,
    render_sentiment_breakdown_chart,
    render_sentiment_over_time_chart,
    render_shap_waterfall_chart,
    render_topsis_ranking_chart,
    verdict_badge,
)
from app.components.sidebar import render_sidebar  # noqa: E402
from app.theme import apply_theme  # noqa: E402
from munasib.api.schemas import (  # noqa: E402
    AlternativesResult,
    AnalysisResponse,
    FuzzyAssessment,
    RatingDistribution,
    SentimentBreakdown,
    TemporalTrend,
)
from munasib.api.schemas.alternatives import ALTERNATIVES_EMPTY_EXAMPLE  # noqa: E402
from munasib.api.schemas.analysis import ANALYSIS_EXAMPLE  # noqa: E402

#: Banner shown above every component, in both data modes.
SAMPLE_BANNER = "Sample data: component gallery (illustrative values, not results)"

#: Toggle label; the same switch drives every component on the page.
EMPTY_MODE_LABEL = "Switch every component to empty data"

#: Stars present in an empty rating distribution; the schema requires a key for
#: every star, so an empty distribution is explicit zeros, not a missing key.
_STARS = (1, 2, 3, 4, 5)


def sample_analysis() -> AnalysisResponse:
    """Return the schema example analysis used by the sample-data mode.

    Returns:
        The ``ANALYSIS_EXAMPLE`` payload, validated so the gallery exercises
        exactly the objects the API serves.
    """
    return AnalysisResponse.model_validate(ANALYSIS_EXAMPLE)


def empty_rating_distribution() -> RatingDistribution:
    """Return a rating distribution with no rated reviews.

    Returns:
        A schema-valid distribution counting zero reviews per star.
    """
    return RatingDistribution(
        counts=dict.fromkeys(_STARS, 0),
        shares=dict.fromkeys(_STARS, 0.0),
        mean_rating=None,
        total=0,
    )


def empty_sentiment() -> SentimentBreakdown:
    """Return a sentiment breakdown counting no reviews.

    Returns:
        A schema-valid breakdown with every count and share at zero.
    """
    return SentimentBreakdown(
        total=0,
        negative_count=0,
        neutral_count=0,
        positive_count=0,
        unlabelled_count=0,
        negative_share=0.0,
        neutral_share=0.0,
        positive_share=0.0,
    )


def empty_trend() -> TemporalTrend:
    """Return a temporal trend with no periods.

    Returns:
        A schema-valid trend whose bucket list is empty and whose slope is None
        (fewer than two periods), which is how the backend reports it.
    """
    return TemporalTrend(period="month", buckets=[], positive_share_slope=None, n_reviews_used=0)


def empty_fuzzy() -> FuzzyAssessment:
    """Return a fuzzy assessment with no memberships and no output degrees.

    Returns:
        A schema-valid assessment of a scale minimum (0.0) with no fired rules.
    """
    return FuzzyAssessment(
        input_memberships={},
        fired_rules=[],
        output_degrees={},
        defuzzified=0.0,
        defuzzification_method="centroid",
    )


def empty_alternatives() -> AlternativesResult:
    """Return the schema's real empty alternatives result.

    Returns:
        ``ALTERNATIVES_EMPTY_EXAMPLE``, a documented case of a product with no
        related products at all.
    """
    return AlternativesResult.model_validate(ALTERNATIVES_EMPTY_EXAMPLE)


def render_components(analysis: AnalysisResponse, empty: bool) -> None:
    """Render every component, in either sample or empty-data mode.

    Args:
        analysis: The validated sample analysis supplying the sample values.
        empty: True to feed every component missing data instead.
    """
    review = analysis.review_analysis

    st.subheader("1 · Metric card")
    if empty:
        metric_card("Consumer risk score", None, help="0-100; higher is riskier.")
        metric_card("Evidence strength", None, delta=None)
    else:
        metric_card("Consumer risk score", analysis.risk_score, help="0-100; higher is riskier.")
        metric_card("Evidence strength", analysis.evidence_strength, delta=None)

    st.subheader("2 · Verdict badge")
    verdict_badge(None if empty else (analysis.verdict.label if analysis.verdict else None))

    st.subheader("3 · Risk gauge")
    render_risk_gauge(None if empty else analysis.risk_score)

    st.subheader("4 · Rating distribution")
    render_rating_distribution_chart(empty_rating_distribution() if empty else review.rating_distribution)

    st.subheader("5 · Sentiment breakdown")
    render_sentiment_breakdown_chart(empty_sentiment() if empty else review.sentiment)

    st.subheader("6 · Sentiment over time")
    render_sentiment_over_time_chart(empty_trend() if empty else review.temporal_trend)

    st.subheader("7 · Aspects")
    render_aspect_bar_chart([] if empty else review.aspects)

    st.subheader("8 · Evidence indicator radar")
    render_evidence_indicator_radar(
        None if empty else analysis.evidence,
        () if empty else analysis.risk_indicators,
    )

    st.subheader("9 · Fuzzy memberships")
    render_fuzzy_membership_chart(empty_fuzzy() if empty else (analysis.decision.fuzzy if analysis.decision else None))

    st.subheader("10 · TOPSIS ranking")
    render_topsis_ranking_chart(None if empty else (analysis.decision.topsis if analysis.decision else None))

    st.subheader("11 · SHAP waterfall")
    render_shap_waterfall_chart(None if empty else analysis.explanation)

    st.subheader("12 · Concern list")
    concern_list([] if empty else (analysis.verdict.major_concerns if analysis.verdict else []))

    st.subheader("13 · Alternatives table")
    alternatives_table(empty_alternatives() if empty else analysis.alternatives)


def main() -> None:
    """Render the component gallery page."""
    apply_theme("Component Gallery")
    render_sidebar()

    st.title("🎨 Component gallery")
    st.warning(SAMPLE_BANNER)

    empty = st.checkbox(
        EMPTY_MODE_LABEL,
        value=False,
        help="Feeds each component None, an empty list or a zero-count schema so the "
        "empty states can be checked without breaking anything.",
    )
    render_components(sample_analysis(), empty)


if __name__ == "__main__":
    main()

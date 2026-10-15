"""Unit tests for the pure chart builders in ``app/components``.

The builders are the testable half of the component library: they return a
Plotly figure, import no Streamlit, call no API and read no session state, so
they can be driven straight from the API schema examples. Every number below
comes from ``ANALYSIS_EXAMPLE`` - the same payload ``scripts/export_schemas.py``
publishes - or from a schema-valid object built for an edge case; nothing is
hand-typed as a result.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from plotly import graph_objects as go

from app.components import (
    build_alternatives_rows,
    build_aspect_bar_figure,
    build_evidence_indicator_radar_figure,
    build_fuzzy_membership_figure,
    build_rating_distribution_figure,
    build_risk_gauge_figure,
    build_sentiment_breakdown_figure,
    build_sentiment_over_time_figure,
    build_shap_waterfall_figure,
    build_topsis_ranking_figure,
)
from app.components._common import format_number, placeholder_message
from app.components.alternatives_table import RELATION_ICONS
from app.components.evidence_indicator_radar import COST_SUFFIX
from app.components.risk_gauge import SCALE_MAX, SCALE_MIN, load_risk_bands
from app.theme import CHART_PALETTE
from munasib.api.schemas import (
    AlternativesResult,
    AnalysisResponse,
    SentimentBreakdown,
    TemporalBucket,
    TemporalTrend,
)
from munasib.api.schemas.alternatives import ALTERNATIVES_EMPTY_EXAMPLE
from munasib.api.schemas.analysis import ANALYSIS_EXAMPLE


@pytest.fixture()
def analysis() -> AnalysisResponse:
    """Validate the schema example the whole suite draws its numbers from.

    Returns:
        The ``ANALYSIS_EXAMPLE`` payload as a validated response.
    """
    return AnalysisResponse.model_validate(ANALYSIS_EXAMPLE)


def _fig_json(figure: go.Figure) -> str:
    """Serialise a figure so two builds can be compared for equality.

    Args:
        figure: Figure to serialise.

    Returns:
        Canonical JSON; ``default=str`` keeps datetime axes comparable.
    """
    return json.dumps(figure.to_plotly_json(), sort_keys=True, default=str)


def _annotation_texts(figure: go.Figure) -> list[str]:
    """Return the text of every annotation of a figure.

    Args:
        figure: Figure to read.

    Returns:
        Annotation texts, in declaration order.
    """
    return [str(annotation.text) for annotation in (figure.layout.annotations or ())]


def test_risk_gauge_is_a_single_indicator_inside_the_scale() -> None:
    """The gauge is one indicator trace whose value stays on the 0-100 scale."""
    figure = build_risk_gauge_figure(62.5)

    assert [trace.type for trace in figure.data] == ["indicator"]
    assert figure.data[0].mode == "gauge+number"
    assert SCALE_MIN <= figure.data[0].value <= SCALE_MAX


def test_risk_gauge_clamps_scores_outside_the_scale() -> None:
    """A score outside 0-100 renders clamped instead of stretching the axis."""
    assert build_risk_gauge_figure(150.0).data[0].value == SCALE_MAX
    assert build_risk_gauge_figure(-5.0).data[0].value == SCALE_MIN


def test_risk_gauge_bands_come_from_the_verdict_config() -> None:
    """The band steps are exactly the config thresholds, not UI constants."""
    bands = load_risk_bands()
    figure = build_risk_gauge_figure(62.5)
    steps = figure.data[0].gauge.steps

    assert [step.range[1] for step in steps] == [bands.consider_max, bands.risk_min, SCALE_MAX]
    assert [step.range[0] for step in steps] == [SCALE_MIN, bands.consider_max, bands.risk_min]
    assert SCALE_MIN <= bands.consider_max < bands.risk_min <= SCALE_MAX
    assert steps[0].color == CHART_PALETTE["band_consider_fill"]
    assert steps[1].color == CHART_PALETTE["band_caution_fill"]
    assert steps[2].color == CHART_PALETTE["band_risk_fill"]


def test_risk_gauge_without_a_score_has_no_traces() -> None:
    """A missing score produces the placeholder, never a zeroed gauge."""
    figure = build_risk_gauge_figure(None)

    assert len(figure.data) == 0
    assert "risk score" in placeholder_message(figure).lower()


def test_rating_distribution_has_one_bar_per_star(analysis: AnalysisResponse) -> None:
    """Five bars, counting the stars of the example product."""
    distribution = analysis.review_analysis.rating_distribution
    figure = build_rating_distribution_figure(distribution)

    assert [trace.type for trace in figure.data] == ["bar"]
    assert len(figure.data[0].x) == 5
    assert list(figure.data[0].y) == [distribution.counts[star] for star in (1, 2, 3, 4, 5)]
    assert sum(figure.data[0].y) == distribution.total
    assert all(count >= 0 for count in figure.data[0].y)


def test_rating_distribution_shares_stay_within_zero_and_one(analysis: AnalysisResponse) -> None:
    """Every plotted share is a share: 0.0-1.0."""
    figure = build_rating_distribution_figure(analysis.review_analysis.rating_distribution)

    assert all(0.0 <= share <= 1.0 for share in figure.data[0].customdata)


def test_rating_distribution_with_no_reviews_is_a_placeholder() -> None:
    """A zero-total distribution renders the placeholder, not empty bars."""
    figure = build_rating_distribution_figure(None)

    assert len(figure.data) == 0
    assert "star-rating evidence" in placeholder_message(figure)


def test_sentiment_breakdown_counts_match_the_example(analysis: AnalysisResponse) -> None:
    """The bars carry the example's class counts and never negative counts."""
    sentiment = analysis.review_analysis.sentiment
    figure = build_sentiment_breakdown_figure(sentiment)

    assert [trace.type for trace in figure.data] == ["bar"]
    assert all(count >= 0 for count in figure.data[0].x)
    assert sum(figure.data[0].x) == sentiment.total - sentiment.unlabelled_count


def test_sentiment_breakdown_never_fabricates_a_share_for_unlabelled() -> None:
    """Unlabelled reviews are counted but carry no share, so none is drawn.

    Built by hand because the schema example has no unlabelled reviews; the
    numbers are internally consistent counts, not product results.
    """
    sentiment = SentimentBreakdown(
        total=10,
        negative_count=4,
        neutral_count=3,
        positive_count=2,
        unlabelled_count=1,
        negative_share=0.4,
        neutral_share=0.3,
        positive_share=0.2,
    )
    figure = build_sentiment_breakdown_figure(sentiment)

    labels = list(figure.data[0].y)
    shares = [entry[1] for entry in figure.data[0].customdata]
    unlabelled_index = labels.index("Unlabelled (no rating)")
    assert shares[unlabelled_index] is None
    assert figure.data[0].text[unlabelled_index] == "1"


def test_sentiment_breakdown_without_reviews_is_a_placeholder() -> None:
    """Nothing classified means no chart and an explanatory message."""
    figure = build_sentiment_breakdown_figure(None)

    assert len(figure.data) == 0
    assert "classified reviews" in placeholder_message(figure)


def test_sentiment_over_time_plots_every_bucket_for_every_series(analysis: AnalysisResponse) -> None:
    """Three share lines, each with one point per bucket."""
    trend = analysis.review_analysis.temporal_trend
    figure = build_sentiment_over_time_figure(trend)

    assert [trace.type for trace in figure.data] == ["scatter"] * 3
    assert all(len(trace.y) == len(trend.buckets) for trace in figure.data)


def test_sentiment_over_time_shares_stay_on_the_unit_interval(analysis: AnalysisResponse) -> None:
    """Every plotted share is inside 0.0-1.0, the axis range of the chart."""
    figure = build_sentiment_over_time_figure(analysis.review_analysis.temporal_trend)

    assert all(0.0 <= share <= 1.0 for trace in figure.data for share in trace.y)
    assert list(figure.layout.yaxis.range) == [0.0, 1.0]


def test_sentiment_over_time_writes_the_slope_as_text(analysis: AnalysisResponse) -> None:
    """The slope is drawn in the figure, quoting the schema's own slope."""
    trend = analysis.review_analysis.temporal_trend
    figure = build_sentiment_over_time_figure(trend)
    notes = _annotation_texts(figure)

    assert any(f"{trend.positive_share_slope:+.6f}" in note for note in notes)


def test_sentiment_over_time_reports_a_missing_slope() -> None:
    """A slope that was not computed is stated as such, not as zero."""
    trend = TemporalTrend(
        period="month",
        buckets=[
            TemporalBucket(
                period_start=datetime(2018, 1, 1, tzinfo=UTC),
                n_reviews=1,
                negative_share=0.0,
                neutral_share=1.0,
                positive_share=0.0,
            )
        ],
        positive_share_slope=None,
        n_reviews_used=1,
    )
    figure = build_sentiment_over_time_figure(trend)

    assert any("not computed" in note for note in _annotation_texts(figure))


def test_sentiment_over_time_without_buckets_is_a_placeholder() -> None:
    """No period with reviews means no trend line at all."""
    figure = build_sentiment_over_time_figure(None)

    assert len(figure.data) == 0
    assert "sentiment trend" in placeholder_message(figure)


def test_aspect_bar_has_one_trace_per_polarity_and_one_point_per_aspect(
    analysis: AnalysisResponse,
) -> None:
    """Polarity becomes the legend; the bar length is the mention count."""
    aspects = analysis.review_analysis.aspects
    figure = build_aspect_bar_figure(aspects)

    points = sum(len(trace.y) for trace in figure.data)
    assert points == len(aspects)
    assert {trace.name for trace in figure.data} == {"Positive aspect", "Negative aspect"}
    assert sorted(value for trace in figure.data for value in trace.x) == sorted(
        aspect.mention_count for aspect in aspects
    )


def test_aspect_bar_orders_the_most_mentioned_aspect_first(analysis: AnalysisResponse) -> None:
    """The y axis is ordered by mention count, most mentioned at the top."""
    figure = build_aspect_bar_figure(analysis.review_analysis.aspects)
    plotted = list(figure.layout.yaxis.categoryarray)
    counts = {aspect.aspect: aspect.mention_count for aspect in analysis.review_analysis.aspects}

    assert [counts[name] for name in plotted] == sorted(counts.values())
    assert len(plotted) == len(counts)


def test_aspect_bar_without_aspects_is_a_placeholder() -> None:
    """No extracted aspect means the placeholder, not an empty axis."""
    figure = build_aspect_bar_figure(None)

    assert len(figure.data) == 0
    assert "aspects" in placeholder_message(figure)


def test_evidence_radar_values_stay_on_the_zero_to_one_ring(analysis: AnalysisResponse) -> None:
    """Every plotted radius is a normalised 0-1 indicator value."""
    figure = build_evidence_indicator_radar_figure(analysis.evidence, analysis.risk_indicators)

    assert [trace.type for trace in figure.data] == ["scatterpolar"]
    assert all(0.0 <= radius <= 1.0 for radius in figure.data[0].r)
    assert list(figure.layout.polar.radialaxis.range) == [0.0, 1.0]


def test_evidence_radar_marks_cost_direction_in_the_label(analysis: AnalysisResponse) -> None:
    """Cost-direction indicators carry a text marker, not just a color."""
    figure = build_evidence_indicator_radar_figure(analysis.evidence)
    labels = list(figure.data[0].theta)
    cost_labels = [label for label in labels if label.endswith(COST_SUFFIX)]

    assert cost_labels == [f"rating_distribution_entropy{COST_SUFFIX}", f"negative_severity_share{COST_SUFFIX}"]
    assert len(cost_labels) == sum(
        1 for item in analysis.evidence.details if item.direction.value == "cost"
    )


def test_evidence_radar_gives_risk_indicators_their_own_trace(analysis: AnalysisResponse) -> None:
    """Risk indicators plot as a second series with their own legend entry.

    The optional risk series is fed from the schema's own detail indicators,
    because the example product's ``risk_indicators`` list is empty.
    """
    figure = build_evidence_indicator_radar_figure(analysis.evidence, analysis.evidence.details)

    assert len(figure.data) == 2
    assert [trace.name for trace in figure.data] == ["Evidence strength", "Risk indicators"]
    assert len(figure.data[1].theta) == len(analysis.evidence.details)


def test_evidence_radar_without_indicators_is_a_placeholder() -> None:
    """No indicators at all means the placeholder, never an empty ring."""
    figure = build_evidence_indicator_radar_figure(None)

    assert len(figure.data) == 0
    assert "evidence-strength indicators" in placeholder_message(figure)


def test_fuzzy_membership_plots_input_terms_and_output_degrees(analysis: AnalysisResponse) -> None:
    """One trace per input term plus the output trace, all on a 0-1 degree axis."""
    fuzzy = analysis.decision.fuzzy
    figure = build_fuzzy_membership_figure(fuzzy)

    input_terms = {term for profile in fuzzy.input_memberships.values() for term in profile.term_degrees}
    assert len(figure.data) == len(input_terms) + 1
    assert {trace.name for trace in figure.data[:-1]} == input_terms
    assert figure.data[-1].name == "Output degree"
    assert len(figure.data[-1].x) == len(fuzzy.output_degrees)
    assert list(figure.layout.yaxis.range) == [0.0, 1.0]


def test_fuzzy_membership_degrees_stay_within_zero_and_one(analysis: AnalysisResponse) -> None:
    """Every membership and output degree is a 0-1 degree."""
    figure = build_fuzzy_membership_figure(analysis.decision.fuzzy)

    assert all(0.0 <= degree <= 1.0 for trace in figure.data for degree in trace.y)


def test_fuzzy_membership_without_assessment_is_a_placeholder() -> None:
    """No memberships and no output degrees means the placeholder."""
    figure = build_fuzzy_membership_figure(None)

    assert len(figure.data) == 0
    assert "membership degrees" in placeholder_message(figure)


def test_topsis_plots_one_trace_per_criterion_direction(analysis: AnalysisResponse) -> None:
    """Benefit and cost criteria are separated so the legend means something."""
    topsis = analysis.decision.topsis
    figure = build_topsis_ranking_figure(topsis)

    assert len(figure.data) == 2
    assert {trace.name for trace in figure.data} == {
        "Benefit criterion (higher is better)",
        "Cost criterion (higher is worse)",
    }
    assert sum(len(trace.x) for trace in figure.data) == len(topsis.criteria)


def test_topsis_contributions_closeness_and_percentile_stay_in_bounds(
    analysis: AnalysisResponse,
) -> None:
    """Contributions and closeness are 0-1; percentile is 0-100."""
    topsis = analysis.decision.topsis
    figure = build_topsis_ranking_figure(topsis)

    assert all(0.0 <= contribution <= 1.0 for trace in figure.data for contribution in trace.y)
    assert 0.0 <= topsis.closeness <= 1.0
    assert 0.0 <= topsis.percentile <= 100.0
    notes = _annotation_texts(figure)
    assert any(f"{topsis.closeness:.3f}" in note and f"rank {topsis.rank}" in note for note in notes)


def test_topsis_without_a_ranking_is_a_placeholder() -> None:
    """No TOPSIS result means the placeholder, not an empty bar chart."""
    figure = build_topsis_ranking_figure(None)

    assert len(figure.data) == 0
    assert "TOPSIS ranking" in placeholder_message(figure)


def test_shap_waterfall_runs_from_base_value_to_prediction(analysis: AnalysisResponse) -> None:
    """Base value + every contribution lands on the final prediction bar."""
    explanation = analysis.explanation
    figure = build_shap_waterfall_figure(explanation)
    trace = figure.data[0]

    assert [traceItem.type for traceItem in figure.data] == ["waterfall"]
    assert len(trace.x) == len(explanation.contributions) + 2
    assert list(trace.measure) == ["absolute"] + ["relative"] * len(explanation.contributions) + ["total"]
    assert trace.y[-1] == pytest.approx(explanation.base_value + sum(c.contribution for c in explanation.contributions))


def test_shap_waterfall_without_contributions_is_a_placeholder() -> None:
    """No contributions means the placeholder, not a lone base-value bar."""
    figure = build_shap_waterfall_figure(None)

    assert len(figure.data) == 0
    assert "SHAP contributions" in placeholder_message(figure)


def test_alternatives_rows_carry_real_deltas(analysis: AnalysisResponse) -> None:
    """Rows quote the schema deltas, formatting missing values as n/a."""
    rows = build_alternatives_rows(analysis.alternatives)

    assert [row["ASIN"] for row in rows] == [item.summary.asin for item in analysis.alternatives.items]
    assert rows[0]["Relation"] == f"{RELATION_ICONS['better']} better"
    assert rows[0]["Δ Risk"] == f"{analysis.alternatives.items[0].deltas.risk_score_delta:+.2f}"
    assert rows[0]["Δ Rating"] == f"{analysis.alternatives.items[0].deltas.mean_rating_delta:+.2f}"


def test_alternatives_rows_from_the_empty_example_are_empty() -> None:
    """The documented empty result yields no rows at all."""
    empty = AlternativesResult.model_validate(ALTERNATIVES_EMPTY_EXAMPLE)

    assert build_alternatives_rows(empty) == []
    assert build_alternatives_rows(None) == []


def test_format_number_keeps_missing_and_tiny_values_honest() -> None:
    """None is n/a, and a non-zero value is never rounded down to zero."""
    assert format_number(None) == "n/a"
    assert format_number(52.0) == "52"
    assert format_number(2.634615) == "2.6346"
    assert format_number(0.00001) == "1.00e-05"


DETERMINISM_CASES: dict[str, Callable[[AnalysisResponse], go.Figure]] = {
    "risk_gauge": lambda analysis: build_risk_gauge_figure(analysis.risk_score),
    "rating_distribution": lambda analysis: build_rating_distribution_figure(
        analysis.review_analysis.rating_distribution
    ),
    "sentiment_breakdown": lambda analysis: build_sentiment_breakdown_figure(
        analysis.review_analysis.sentiment
    ),
    "sentiment_over_time": lambda analysis: build_sentiment_over_time_figure(
        analysis.review_analysis.temporal_trend
    ),
    "aspect_bar": lambda analysis: build_aspect_bar_figure(analysis.review_analysis.aspects),
    "evidence_radar": lambda analysis: build_evidence_indicator_radar_figure(analysis.evidence),
    "fuzzy_membership": lambda analysis: build_fuzzy_membership_figure(analysis.decision.fuzzy),
    "topsis_ranking": lambda analysis: build_topsis_ranking_figure(analysis.decision.topsis),
    "shap_waterfall": lambda analysis: build_shap_waterfall_figure(analysis.explanation),
}


@pytest.mark.parametrize("name", sorted(DETERMINISM_CASES))
def test_builders_are_deterministic(name: str, analysis: AnalysisResponse) -> None:
    """The same input always produces the same figure."""
    build = DETERMINISM_CASES[name]

    assert _fig_json(build(analysis)) == _fig_json(build(analysis))


@pytest.mark.parametrize("name", sorted(DETERMINISM_CASES))
def test_builders_return_a_figure(name: str, analysis: AnalysisResponse) -> None:
    """Every builder returns a Plotly figure with at least one trace."""
    figure = DETERMINISM_CASES[name](analysis)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) >= 1
    assert figure.layout.title.text


@pytest.mark.parametrize(
    ("build", "expected_key"),
    [
        (lambda: build_risk_gauge_figure(None), "risk score"),
        (lambda: build_rating_distribution_figure(None), "star-rating evidence"),
        (lambda: build_sentiment_breakdown_figure(None), "classified reviews"),
        (lambda: build_sentiment_over_time_figure(None), "sentiment trend"),
        (lambda: build_aspect_bar_figure(None), "aspects"),
        (lambda: build_evidence_indicator_radar_figure(None), "evidence-strength indicators"),
        (lambda: build_fuzzy_membership_figure(None), "membership degrees"),
        (lambda: build_topsis_ranking_figure(None), "TOPSIS ranking"),
        (lambda: build_shap_waterfall_figure(None), "SHAP contributions"),
    ],
)
def test_empty_payloads_never_raise(build: Callable[[], go.Figure], expected_key: str) -> None:
    """Every builder answers missing data with a message, not an exception."""
    figure = build()

    assert len(figure.data) == 0
    assert expected_key in placeholder_message(figure)


def test_placeholder_message_is_the_figure_annotation() -> None:
    """A populated figure reports no placeholder reason."""
    figure = build_aspect_bar_figure([])

    assert isinstance(placeholder_message(figure), str)
    assert placeholder_message(figure)


def test_palette_covers_every_semantic_key_used_by_the_components() -> None:
    """The theme palette carries the colors the components ask for by name."""
    required = {
        "positive",
        "negative",
        "neutral",
        "unlabelled",
        "benefit",
        "cost",
        "better",
        "comparable",
        "worse",
        "severity_high",
        "severity_medium",
        "severity_low",
        "band_consider",
        "band_caution",
        "band_risk",
        "band_consider_fill",
        "band_caution_fill",
        "band_risk_fill",
        "grey",
    }

    assert required <= set(CHART_PALETTE)
    assert all(value.startswith("#") for value in CHART_PALETTE.values())


def test_placeholder_figure_json_round_trips() -> None:
    """Placeholder figures serialize like any other figure."""
    figure = build_shap_waterfall_figure(None)
    payload: Any = figure.to_plotly_json()

    assert payload["data"] == []
    assert _fig_json(figure) == _fig_json(build_shap_waterfall_figure(None))

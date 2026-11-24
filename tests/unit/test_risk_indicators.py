"""Hand-computed tests for the pure risk-indicator functions.

Every expected value here is derived by hand from the documented formula (see
``docs/risk_indicators.md``) - no cache, no network, no clock.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from munasib.core.config import RiskCriteriaConfig
from munasib.core.exceptions import DataError
from munasib.risk.indicators import (
    CRITERION_ORDER,
    DIRECTION_BENEFIT,
    DIRECTION_COST,
    EXPECTED_DIRECTIONS,
    Criterion,
    category_priors,
    criteria_vector,
    evidence_strength_criterion,
    negative_feedback_concentration_criterion,
    price_position_criterion,
    product_quality_criterion,
    product_sentiment_totals,
    rating_reliability_criterion,
    read_criteria_directions,
    shrink,
)

CONFIG = RiskCriteriaConfig()
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "criteria_weights.yaml"


def test_shrink_mixes_the_prior_by_the_pseudo_count() -> None:
    """``(n * value + m * prior) / (n + m)``, and more data wins over the prior."""
    assert shrink(5.0, 3, 4.0, 20) == pytest.approx(95 / 23)
    thin = shrink(5.0, 3, 4.0, 20)
    thick = shrink(5.0, 300, 4.0, 20)
    assert thin < thick < 5.0


def test_shrink_rejects_invalid_inputs() -> None:
    """A negative count or a non-positive pseudo-count is a usage error."""
    with pytest.raises(ValueError, match="non-negative"):
        shrink(4.0, -1, 4.0, 20)
    with pytest.raises(ValueError, match="positive"):
        shrink(4.0, 10, 4.0, 0)


def test_product_quality_hand_computed() -> None:
    """The three components are hand-computable for a tiny product."""
    criterion = product_quality_criterion(
        mean_rating=5.0,
        n_reviews=3,
        mean_net=1.0,
        n_predictions=3,
        recent_negative_shift=0.9,
        recent_negative_shift_reason=None,
        prior_rating=4.0,
        prior_net=0.0,
        config=CONFIG,
    )
    # rating: (95/23 - 1) / 4 = 18/23; net: (3/23 + 1) / 2 = 13/23
    expected = 0.5 * (18 / 23) + 0.3 * (13 / 23) + 0.2 * 0.9
    assert criterion.value == pytest.approx(expected)
    assert criterion.value == pytest.approx(0.740870, abs=1e-6)
    assert criterion.direction == DIRECTION_BENEFIT
    assert criterion.confidence == pytest.approx(1.0)
    assert criterion.components_used == (
        "shrunk_rating",
        "shrunk_sentiment",
        "recent_sentiment",
    )
    assert criterion.missing_components == {}
    assert criterion.extras["shrunk_rating"] == pytest.approx(95 / 23, abs=1e-6)


def test_product_quality_renormalizes_around_missing_components() -> None:
    """A missing rating is dropped and the rest share its weight."""
    criterion = product_quality_criterion(
        mean_rating=None,
        n_reviews=0,
        mean_net=0.0,
        n_predictions=10,
        recent_negative_shift=None,
        recent_negative_shift_reason="recent_window_below_min_reviews",
        prior_rating=4.0,
        prior_net=0.0,
        config=CONFIG,
    )
    # Only the shrunk sentiment (weight 0.30) is available -> renormalized to 1.
    assert criterion.value == pytest.approx(0.5)
    assert criterion.components_used == ("shrunk_sentiment",)
    assert criterion.weights == {"shrunk_sentiment": pytest.approx(1.0)}
    assert criterion.missing_components == {
        "shrunk_rating": "no_ratings",
        "recent_sentiment": "recent_window_below_min_reviews",
    }
    assert criterion.confidence == pytest.approx(0.30)


def test_product_quality_is_none_without_any_component() -> None:
    """No component -> None with a reason, never a fabricated zero."""
    criterion = product_quality_criterion(
        mean_rating=None,
        n_reviews=0,
        mean_net=None,
        n_predictions=0,
        recent_negative_shift=None,
        recent_negative_shift_reason="no_evidence_row",
        prior_rating=4.0,
        prior_net=0.0,
        config=CONFIG,
    )
    assert criterion.value is None
    assert criterion.confidence == 0.0
    assert criterion.missing_reason == "all_components_missing"
    assert criterion.to_payload()["value"] is None


def test_evidence_strength_passes_the_stored_composite_through() -> None:
    """The criterion is the stored composite; absence stays absent."""
    present = evidence_strength_criterion(0.68, low_evidence=False, missing_reason=None)
    assert present.value == pytest.approx(0.68)
    assert present.confidence == 1.0
    assert present.extras == {"low_evidence": False}

    absent = evidence_strength_criterion(None, low_evidence=None, missing_reason="no_evidence_row")
    assert absent.value is None
    assert absent.missing_reason == "no_evidence_row"
    assert absent.confidence == 0.0


def test_rating_reliability_helpfulness_component_saturates_at_the_scale() -> None:
    """``1 - min(1, gap / scale)`` and its renormalized weight."""
    half = rating_reliability_criterion(
        rating_agreement=None,
        polarized=None,
        rating_text_consistency=None,
        helpfulness_gap=0.05,
        config=CONFIG,
    )
    assert half.value == pytest.approx(0.5)
    assert half.components_used == ("helpfulness_agreement",)
    assert half.weights == {"helpfulness_agreement": pytest.approx(1.0)}
    assert half.confidence == pytest.approx(0.20)
    assert half.extras["helpfulness_agreement"] == pytest.approx(0.5)

    saturated = rating_reliability_criterion(
        rating_agreement=None,
        polarized=None,
        rating_text_consistency=None,
        helpfulness_gap=0.20,
        config=CONFIG,
    )
    assert saturated.value == pytest.approx(0.0)


def test_rating_reliability_non_polarization_is_gated_on_agreement() -> None:
    """An unrated product never gets a fabricated non-polarization of 1.0."""
    criterion = rating_reliability_criterion(
        rating_agreement=None,
        polarized=False,
        rating_text_consistency=None,
        helpfulness_gap=None,
        config=CONFIG,
    )
    assert criterion.value is None
    assert criterion.missing_components["non_polarization"] == "no_ratings"
    assert criterion.missing_reason == "all_components_missing"


def test_rating_reliability_uses_the_chosen_scale() -> None:
    """The run-selected scale overrides the configured one and is recorded."""
    criterion = rating_reliability_criterion(
        rating_agreement=None,
        polarized=None,
        rating_text_consistency=None,
        helpfulness_gap=0.10,
        config=CONFIG,
        agreement_scale=0.20,
    )
    assert criterion.value == pytest.approx(0.5)
    assert criterion.extras["agreement_scale"] == pytest.approx(0.20)


def test_negative_feedback_concentration_hand_computed() -> None:
    """Prevalence x (focus_floor + (1 - focus_floor) x focus), by hand."""
    criterion = negative_feedback_concentration_criterion(
        n_predictions=100,
        negative_share=0.2,
        prior_negative_share=0.1,
        top_pain_point_share=0.5,
        aspect_concentration=0.25,
        temporal_burst=0.75,
        config=CONFIG,
    )
    # shrunk = (100*0.2 + 20*0.1) / 120 = 11/60; prevalence = (11/60)/0.4
    prevalence = (11 / 60) / 0.4
    assert criterion.value == pytest.approx(prevalence * 0.75)
    assert criterion.value == pytest.approx(0.34375)
    assert criterion.direction == DIRECTION_COST
    assert criterion.extras["focus"] == pytest.approx(0.5)
    assert criterion.extras["focus_unknown"] is False
    assert criterion.extras["focus_components"] == [
        "aspect_concentration",
        "temporal_burst",
        "top_pain_point_share",
    ]


def test_negative_feedback_concentration_records_unknown_focus() -> None:
    """All focus inputs missing -> focus 0, flagged, value still real."""
    criterion = negative_feedback_concentration_criterion(
        n_predictions=100,
        negative_share=0.2,
        prior_negative_share=0.1,
        top_pain_point_share=None,
        aspect_concentration=None,
        temporal_burst=None,
        config=CONFIG,
    )
    prevalence = (11 / 60) / 0.4
    assert criterion.value == pytest.approx(prevalence * 0.5)
    assert criterion.extras["focus_unknown"] is True
    assert criterion.extras["focus"] == pytest.approx(0.0)


def test_negative_feedback_concentration_is_none_without_predictions() -> None:
    """No prediction -> None, never zero concentration."""
    criterion = negative_feedback_concentration_criterion(
        n_predictions=0,
        negative_share=None,
        prior_negative_share=0.1,
        top_pain_point_share=0.5,
        aspect_concentration=0.5,
        temporal_burst=0.5,
        config=CONFIG,
    )
    assert criterion.value is None
    assert criterion.missing_reason == "no_predictions"
    assert criterion.confidence == 0.0


def test_price_position_carries_the_fallback_confidence() -> None:
    """A fallback peer group halves the confidence but not the value."""
    fallback = price_position_criterion(
        0.8,
        price_missing=False,
        peer_group="Home & Kitchen > Appliances",
        peer_count=120,
        peer_fallback=True,
        config=CONFIG,
    )
    assert fallback.value == pytest.approx(0.8)
    assert fallback.confidence == pytest.approx(0.5)
    assert fallback.direction == DIRECTION_COST

    exact = price_position_criterion(
        0.8,
        price_missing=False,
        peer_group="group",
        peer_count=120,
        peer_fallback=False,
        config=CONFIG,
    )
    assert exact.confidence == pytest.approx(1.0)


def test_price_position_missing_stays_missing() -> None:
    """No price -> None with the stored reason, never 0 or 0.5."""
    criterion = price_position_criterion(
        None,
        price_missing=True,
        peer_group=None,
        peer_count=None,
        peer_fallback=None,
        config=CONFIG,
    )
    assert criterion.value is None
    assert criterion.missing_reason == "price_missing"
    assert criterion.confidence == 0.0
    assert criterion.extras["price_missing"] is True


def test_criteria_vector_keeps_the_fixed_order_and_nulls() -> None:
    """The vector is positional and preserves a missing criterion as None."""
    criteria = {
        "price_position": Criterion(
            name="price_position",
            value=0.4,
            direction=DIRECTION_COST,
            description="",
            confidence=1.0,
        ),
        "product_quality": Criterion(
            name="product_quality",
            value=0.9,
            direction=DIRECTION_BENEFIT,
            description="",
            confidence=1.0,
        ),
    }
    assert criteria_vector(criteria) == (0.9, None, None, None, 0.4)
    assert CRITERION_ORDER == (
        "product_quality",
        "evidence_strength",
        "rating_reliability",
        "negative_feedback_concentration",
        "price_position",
    )


def test_read_criteria_directions_matches_the_repository_config() -> None:
    """The declared directions are read from the YAML and match the contract."""
    directions = read_criteria_directions(CONFIG_PATH)
    assert directions == dict(EXPECTED_DIRECTIONS)


def test_read_criteria_directions_rejects_a_flipped_direction(tmp_path: Path) -> None:
    """A file that flips a criterion aborts instead of silently inverting it."""
    path = tmp_path / "criteria.yaml"
    path.write_text(
        "criteria:\n"
        "  product_quality: {type: benefit, weight: 0.25}\n"
        "  evidence_strength: {type: benefit, weight: 0.20}\n"
        "  rating_reliability: {type: benefit, weight: 0.15}\n"
        "  negative_feedback_concentration: {type: cost, weight: 0.20}\n"
        "  price_position: {type: benefit, weight: 0.20}\n",
        encoding="utf-8",
    )
    with pytest.raises(DataError, match="price_position"):
        read_criteria_directions(path)


def test_read_criteria_directions_rejects_a_missing_file(tmp_path: Path) -> None:
    """A missing weights file is a hard error."""
    with pytest.raises(DataError, match="not found"):
        read_criteria_directions(tmp_path / "nope.yaml")


def test_read_criteria_directions_rejects_a_missing_criterion(tmp_path: Path) -> None:
    """Every one of the five criteria must be declared."""
    path = tmp_path / "criteria.yaml"
    path.write_text(
        "criteria:\n"
        "  product_quality: {type: benefit, weight: 1.0}\n",
        encoding="utf-8",
    )
    with pytest.raises(DataError, match="missing"):
        read_criteria_directions(path)


def test_product_sentiment_totals_ignores_reviews_without_a_prediction() -> None:
    """Only reviews with a label and both probabilities enter the totals."""
    reviews = pd.DataFrame(
        {
            "asin": ["A", "A", "A", "B"],
            "predicted_label": [0.0, 2.0, None, 0.0],
            "prob_negative": [0.8, 0.1, 0.5, 0.7],
            "prob_positive": [0.1, 0.8, 0.5, 0.2],
        }
    )
    totals = product_sentiment_totals(reviews).set_index("asin")
    assert totals.loc["A", "n_predictions"] == 2
    assert totals.loc["A", "mean_net"] == pytest.approx(0.0)
    assert totals.loc["A", "negative_share"] == pytest.approx(0.5)
    assert totals.loc["B", "n_predictions"] == 1
    assert totals.loc["B", "mean_net"] == pytest.approx(-0.5)


def test_product_sentiment_totals_requires_the_prediction_columns() -> None:
    """A malformed frame fails loudly."""
    with pytest.raises(ValueError, match="lacks column"):
        product_sentiment_totals(pd.DataFrame({"asin": ["A"]}))


def test_category_priors_use_the_train_split_only() -> None:
    """Validation products never move a prior; each prior is a macro mean."""
    products = pd.DataFrame(
        {
            "asin": ["A", "B", "C"],
            "split": ["train", "train", "val"],
            "mean_rating": [5.0, 3.0, 1.0],
        }
    )
    sentiment = pd.DataFrame(
        {
            "asin": ["A", "B", "C"],
            "n_predictions": [10, 10, 10],
            "mean_net": [0.5, 0.1, -1.0],
            "negative_share": [0.1, 0.3, 1.0],
        }
    )
    priors = category_priors(products, sentiment)
    assert priors["mean_rating"] == pytest.approx(4.0)
    assert priors["net_sentiment"] == pytest.approx(0.3)
    assert priors["negative_share"] == pytest.approx(0.2)
    assert priors["n_train_products"] == 2
    assert priors["n_train_with_rating"] == 2
    assert priors["n_train_with_predictions"] == 2


def test_category_priors_refuse_to_fabricate_without_train_predictions() -> None:
    """No train predictions -> DataError, never a made-up prior."""
    products = pd.DataFrame(
        {"asin": ["A"], "split": ["train"], "mean_rating": [4.0]}
    )
    sentiment = pd.DataFrame(
        {
            "asin": ["A"],
            "n_predictions": [0],
            "mean_net": [float("nan")],
            "negative_share": [float("nan")],
        }
    )
    with pytest.raises(DataError, match="priors"):
        category_priors(products, sentiment)

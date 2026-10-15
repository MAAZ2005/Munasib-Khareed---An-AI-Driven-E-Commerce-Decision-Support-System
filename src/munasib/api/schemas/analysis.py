"""Analysis contract: the single composed response the dashboard consumes.

One ``AnalysisResponse`` carries everything the pipeline produced for a product
- the profile, the review analysis, the evidence indicators, the fuzzy + TOPSIS
decision, the surrogate explanation, the alternatives and the verdict - plus the
provenance a consumer needs to judge it: ``pipeline_version``,
``generated_at`` and the ``data_basis`` block (which category, how many reviews,
the real date range of the data, and the ``reference_date`` every recency feature
was measured from - the dataset ends 2018-10-04, so nothing here ever uses
today's date).

``warnings`` is where degraded or thin results announce themselves (for example
an evidence base far below the category median), so a frontend never has to
guess whether an analysis was trustworthy. When the fuzzy/TOPSIS stages cannot
run, ``decision``, ``explanation`` and ``verdict`` are ``None`` and the reason is
listed in ``warnings`` - the API never fabricates scores.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .alternatives import AlternativesResult
from .common import RiskScore, UnitFloat, UtcDatetime
from .decision import DecisionAssessment
from .evidence import EvidenceIndicator, EvidenceIndicators
from .explanation import ShapExplanation
from .products import ProductDetail
from .review_analysis import ReviewAnalysis
from .verdict import ConsumerVerdict

#: Pipeline version reported with an analysis (project version from configs).
PIPELINE_VERSION = "0.1.0"

#: Timestamp used by the examples: the real processing build of the serving cache
#: for this category (recorded in ``data/processed/meta.json``), NOT today.
EXAMPLE_GENERATED_AT = "2026-10-10T11:31:52Z"

#: Subject of every example in this module: a real product of the processed
#: ``Appliances`` subset (52 reviews, 12 related-product edges).
EXAMPLE_ASIN = "B00009V3UF"

ANALYSIS_EXAMPLE: dict[str, Any] = {
    "asin": EXAMPLE_ASIN,
    "product": {
        "asin": EXAMPLE_ASIN,
        "title": "Range Kleen 8234 Gas Range Knobs &ndash; 4 White Gas Range Safety Knobs "
        "Gas Stove Accessories",
        "brand": "Range Kleen",
        "price": 12.66,
        "n_reviews": 52,
        "mean_rating": 2.634615,
        "split": "train",
        "brand_missing": False,
        "price_status": "single",
        "price_missing": False,
        "price_max": None,
        "rank": ">#4,165 in Appliances (See top 100)",
        "rank_value": 4165,
        "rank_category": "Appliances",
        "rank_missing": False,
        "description": "Want a fresh start with your stove burner knobs? Replace them with "
        "the Range Kleen 8234 Pack of 4 white gas range replacement knobs! Use these gas "
        "stove accessoriesto replace broken gas burner knobsthat have become a nuisance and "
        "a safety hazard to you and your loved ones. This pack of 4 gas range safety knobs "
        "includes four gas range knobs, eight adapters, and four overlays. The set of gas "
        "range knobs replacements are neutral white in color,making them compatible with "
        "any kitchen style or motif. No need to do the heavy lifting when it comes to "
        "replacing an appliance; just give your stove a facelift with new burner knobs! "
        "NOTE: Some gas range knobs replacements may require a longer shaft for "
        "replacement. Item 82001 Knob Shaft Extender may be required to replace thestove "
        "burner knobs on your gas stove.",
        "features": [
            "SAFE GAS RANGE REPLACEMENT KNOBS: Gas range knobs give you piece of mind; "
            "stove burner knobs protect and help gauge stove settings.",
            "PRISTINE WHITE BURNER KNOB: Range Kleen 8234 gas burner knobs are immaculate "
            "white color; providing maximum design compatibility.",
            "COMPLETE BURNER KNOBS SET: 4 knobs, 8 adapters and 4 overlays included in this "
            "burner knobs for gas stove set; update all of your gas burner knobs with Range "
            "Kleen.",
            "REPLACEMENT BURNER KNOB FOR GAS STOVE: No need to buy a new stove, just "
            "freshen up the burner knob covers for gas range stoves for a fresh look.",
            "EASY-TURN BURNER KNOB: Gas range knob replacements reset your stoves look; "
            "these brand-new burner gas stove accessories restore your stove to that "
            "factory setting feel.",
        ],
        "first_review_time": "2009-07-20T00:00:00Z",
        "last_review_time": "2018-02-03T00:00:00Z",
    },
    "review_analysis": {
        "rating_distribution": {
            "counts": {"1": 19, "2": 4, "3": 14, "4": 7, "5": 8},
            "shares": {"1": 0.3654, "2": 0.0769, "3": 0.2692, "4": 0.1346, "5": 0.1538},
            "mean_rating": 2.634615,
            "total": 52,
        },
        "sentiment": {
            "total": 52,
            "negative_count": 23,
            "neutral_count": 14,
            "positive_count": 15,
            "unlabelled_count": 0,
            "negative_share": 0.4423,
            "neutral_share": 0.2692,
            "positive_share": 0.2885,
        },
        "temporal_trend": {
            "period": "month",
            "buckets": [
                {"period_start": "2009-07-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2010-02-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 1.0, "positive_share": 0.0},
                {"period_start": "2011-11-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 0.0, "positive_share": 1.0},
                {"period_start": "2011-12-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 0.0, "positive_share": 1.0},
                {"period_start": "2012-01-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 1.0, "positive_share": 0.0},
                {"period_start": "2012-08-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2012-10-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2012-11-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 0.0, "positive_share": 1.0},
                {"period_start": "2013-01-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2013-02-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 1.0, "positive_share": 0.0},
                {"period_start": "2013-04-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 0.0, "positive_share": 1.0},
                {"period_start": "2013-05-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 1.0, "positive_share": 0.0},
                {"period_start": "2013-06-01T00:00:00Z", "n_reviews": 3, "negative_share": 0.0, "neutral_share": 0.6667, "positive_share": 0.3333},
                {"period_start": "2013-08-01T00:00:00Z", "n_reviews": 2, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2013-09-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 1.0, "positive_share": 0.0},
                {"period_start": "2013-10-01T00:00:00Z", "n_reviews": 2, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2013-11-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 0.0, "positive_share": 1.0},
                {"period_start": "2014-01-01T00:00:00Z", "n_reviews": 2, "negative_share": 0.0, "neutral_share": 1.0, "positive_share": 0.0},
                {"period_start": "2014-03-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 1.0, "positive_share": 0.0},
                {"period_start": "2014-04-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 1.0, "positive_share": 0.0},
                {"period_start": "2014-06-01T00:00:00Z", "n_reviews": 2, "negative_share": 0.0, "neutral_share": 0.0, "positive_share": 1.0},
                {"period_start": "2014-08-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2014-09-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2014-10-01T00:00:00Z", "n_reviews": 2, "negative_share": 0.5, "neutral_share": 0.0, "positive_share": 0.5},
                {"period_start": "2014-12-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 1.0, "positive_share": 0.0},
                {"period_start": "2015-01-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 0.0, "positive_share": 1.0},
                {"period_start": "2015-02-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 1.0, "positive_share": 0.0},
                {"period_start": "2015-03-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2015-05-01T00:00:00Z", "n_reviews": 2, "negative_share": 0.5, "neutral_share": 0.5, "positive_share": 0.0},
                {"period_start": "2015-06-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 0.0, "positive_share": 1.0},
                {"period_start": "2015-08-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2015-10-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2015-11-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2016-02-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2016-03-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2016-04-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
                {"period_start": "2016-08-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 0.0, "positive_share": 1.0},
                {"period_start": "2017-01-01T00:00:00Z", "n_reviews": 1, "negative_share": 0.0, "neutral_share": 0.0, "positive_share": 1.0},
                {"period_start": "2017-11-01T00:00:00Z", "n_reviews": 3, "negative_share": 0.6667, "neutral_share": 0.0, "positive_share": 0.3333},
                {"period_start": "2017-12-01T00:00:00Z", "n_reviews": 2, "negative_share": 0.5, "neutral_share": 0.0, "positive_share": 0.5},
                {"period_start": "2018-02-01T00:00:00Z", "n_reviews": 1, "negative_share": 1.0, "neutral_share": 0.0, "positive_share": 0.0},
            ],
            "positive_share_slope": -0.0001,
            "n_reviews_used": 52,
        },
        "aspects": [
            {
                "aspect": "fit",
                "polarity": "negative",
                "mention_count": 6,
                "examples": [
                    "These do not fit the Maytag Performa.",
                    "Have a cheap crapy stove and didn't fit",
                ],
            },
            {
                "aspect": "value",
                "polarity": "positive",
                "mention_count": 4,
                "examples": ["good buy", "quick and easy"],
            },
        ],
    },
    "evidence": {
        "verified_purchase": {
            "name": "verified_purchase",
            "value": 0.9423,
            "raw_value": 0.9423,
            "direction": "benefit",
            "description": "Share of the product's reviews that are verified purchases "
            "(49 of 52 here).",
        },
        "helpfulness": {
            "name": "helpfulness_ratio",
            "value": 0.1538,
            "raw_value": 0.56,
            "direction": "benefit",
            "description": "Share of reviews that at least one other shopper found helpful "
            "(8 of 52 here); the raw value is the mean helpful-vote count per review.",
        },
        "review_depth": {
            "name": "review_depth",
            "value": 1.0,
            "raw_value": 206.31,
            "direction": "benefit",
            "description": "How substantive the review text is, from the mean character "
            "length of the reviews (206.31 characters here).",
        },
        "evidence_strength": 0.7173,
        "details": [
            {
                "name": "rating_distribution_entropy",
                "value": 0.9173,
                "raw_value": 1.4764,
                "direction": "cost",
                "description": "Shannon entropy of the star distribution normalised "
                "by log(5) (0 = one shared star, 1 = perfectly spread): 0.9173 here "
                "over the 19/4/14/7/8 star split, so opinions span all five stars and "
                "the 2.63 mean hides a divided audience.",
            },
            {
                "name": "helpfulness_weighted_sentiment",
                "value": 0.5517,
                "raw_value": 1.1034,
                "direction": "benefit",
                "description": "Weak labels (0 negative, 1 neutral, 2 positive) "
                "averaged with each review's helpful votes as weight and rescaled to "
                "0-1 (raw 0-2 mean label 1.1034): 0.5517 here from 29 helpful votes "
                "across 52 reviews, above the 0.4231 unweighted value - the reviews "
                "shoppers voted useful read better than average.",
            },
            {
                "name": "negative_severity_share",
                "value": 0.8261,
                "raw_value": 19.0,
                "direction": "cost",
                "description": "Share of the weak-negative reviews that are one-star "
                "reviews (19 of 23 here): negative feedback packed into the harshest "
                "bucket rather than spread over 2-3 stars.",
            },
        ],
    },
    "risk_indicators": [],
    "decision": {
        "fuzzy": {
            "input_memberships": {
                "product_quality": {
                    "term_degrees": {"low": 0.02, "medium": 0.62, "high": 0.31}
                },
                "evidence_strength": {
                    "term_degrees": {"low": 0.11, "medium": 0.55, "high": 0.42}
                },
                "rating_reliability": {
                    "term_degrees": {"low": 0.14, "medium": 0.88, "high": 0.0}
                },
                "negative_feedback_concentration": {
                    "term_degrees": {"low": 0.08, "medium": 0.91, "high": 0.25}
                },
                "price_position": {
                    "term_degrees": {"low": 0.44, "medium": 0.52, "high": 0.12}
                },
            },
            "fired_rules": [
                {
                    "rule_id": 7,
                    "antecedents": {
                        "negative_feedback_concentration": "medium",
                        "product_quality": "medium",
                        "rating_reliability": "medium",
                    },
                    "consequent": "high",
                    "strength": 0.62,
                },
                {
                    "rule_id": 8,
                    "antecedents": {
                        "negative_feedback_concentration": "medium",
                        "product_quality": "low",
                        "rating_reliability": "medium",
                    },
                    "consequent": "high",
                    "strength": 0.25,
                },
                {
                    "rule_id": 13,
                    "antecedents": {
                        "negative_feedback_concentration": "low",
                        "product_quality": "medium",
                        "rating_reliability": "low",
                    },
                    "consequent": "moderate",
                    "strength": 0.14,
                },
            ],
            "output_degrees": {"low": 0.05, "moderate": 0.28, "high": 0.62, "critical": 0.11},
            "defuzzified": 62.5,
            "defuzzification_method": "centroid",
        },
        "topsis": {
            "criteria": [
                {
                    "name": "product_quality",
                    "description": "Product quality from review evidence, normalised "
                    "0-1 (higher is better). Illustrative value: 0.4087 scales the "
                    "example product's mean rating 2.634615 from 1-5 stars to 0-1; "
                    "the risk engine's quality criterion is not built yet.",
                    "weight": 0.25,
                    "direction": "benefit",
                    "value": 0.4087,
                    "normalized_value": 0.44,
                    "contribution": 0.11,
                },
                {
                    "name": "evidence_strength",
                    "description": "Review depth, length, and credibility indicator "
                    "(0-1 scale); mirrors evidence.evidence_strength (0.7173).",
                    "weight": 0.2,
                    "direction": "benefit",
                    "value": 0.7173,
                    "normalized_value": 0.78,
                    "contribution": 0.12,
                },
                {
                    "name": "rating_reliability",
                    "description": "Consistency of the rating signal, 0-1 (how far the "
                    "ratings agree). Illustrative value: the rating-reliability "
                    "criterion is not computed yet.",
                    "weight": 0.15,
                    "direction": "benefit",
                    "value": 0.42,
                    "normalized_value": 0.46,
                    "contribution": 0.07,
                },
                {
                    "name": "negative_feedback_concentration",
                    "description": "Concentration of negative feedback, 0-1 (higher "
                    "means more negative). Illustrative value: the risk engine's "
                    "concentration criterion is not computed yet; 0.4423 here is only "
                    "the example's weak-negative share (23 of 52 reviews).",
                    "weight": 0.15,
                    "direction": "cost",
                    "value": 0.4423,
                    "normalized_value": 0.51,
                    "contribution": 0.08,
                },
                {
                    "name": "price_position",
                    "description": "Price relative to peer products, 0-1 where 0 means "
                    "the cheapest price relative to peers. Illustrative value: no peer "
                    "price comparison has run yet; a missing price is handled "
                    "explicitly later, never imputed.",
                    "weight": 0.1,
                    "direction": "cost",
                    "value": 0.35,
                    "normalized_value": 0.4,
                    "contribution": 0.04,
                },
                {
                    "name": "fuzzy_risk",
                    "description": "Output of the fuzzy-logic consumer-risk stage "
                    "(0-100, lower is safer), fed to TOPSIS as an extra cost criterion. "
                    "Illustrative: mirrors the illustrative decision.fuzzy.defuzzified "
                    "value.",
                    "weight": 0.15,
                    "direction": "cost",
                    "value": 62.5,
                    "normalized_value": 0.71,
                    "contribution": 0.09,
                },
            ],
            "closeness": 0.38,
            "rank": 19,
            "ranked_count": 21,
            "peer_count": 20,
            "percentile": 9.5,
            "risk_score": 62.5,
        },
    },
    "explanation": {
        "base_value": 41.3,
        "contributions": [
            {"feature": "negative_sentiment_share", "value": 0.4423, "contribution": 16.8},
            {"feature": "aspect_complaint_intensity", "value": 0.37, "contribution": 14.9},
            {"feature": "verified_purchase_ratio", "value": 0.9423, "contribution": -8.4},
            {"feature": "evidence_strength", "value": 0.7173, "contribution": -5.2},
            {"feature": "review_depth", "value": 206.31, "contribution": 3.1},
            {"feature": "mean_rating", "value": 2.634615, "contribution": 0.4},
            {"feature": "rating_trend_slope", "value": -0.0001, "contribution": -0.4},
        ],
        "surrogate_fidelity": {
            "r2": 0.91,
            "mae": 2.34,
            "n_training_points": 2066,
            "meets_minimum": True,
        },
        "reasons": [
            {
                "text": "23 of 52 reviews (44.2%) carry the weak negative label, the "
                "largest single driver of the risk score.",
                "feature": "negative_sentiment_share",
                "raw_value": 0.4423,
            },
            {
                "text": "49 of 52 reviews (94.2%) are verified purchases, which lowers "
                "the risk score.",
                "feature": "verified_purchase_ratio",
                "raw_value": 0.9423,
            },
            {
                "text": "The mean rating is 2.63 out of 5 with 19 of 52 reviews rated one "
                "star, which raises the risk score.",
                "feature": "mean_rating",
                "raw_value": 2.634615,
            },
        ],
    },
    "alternatives": {
        "items": [
            {
                "summary": {
                    "asin": "B0050O1N2U",
                    "title": "Whirlpool 8273104 Knob for Range",
                    "brand": "Whirlpool",
                    "price": 13.99,
                    "n_reviews": 108,
                    "mean_rating": 4.787037,
                    "split": "train",
                },
                "risk_score": 27.9,
                "verdict": "CONSIDER",
                "deltas": {
                    "risk_score_delta": -34.6,
                    "evidence_strength_delta": 0.06,
                    "mean_rating_delta": 2.152422,
                },
                "relation": "better",
                "relation_types": ["also_viewed"],
            },
            {
                "summary": {
                    "asin": "B004W7UWIM",
                    "title": "AquaPlumb RKG Not Applicable Gas Range Knob- 5PC/Card",
                    "brand": "Aqua Plumb",
                    "price": 10.43,
                    "n_reviews": 318,
                    "mean_rating": 2.216981,
                    "split": "train",
                },
                "risk_score": 71.9,
                "verdict": "CONSUMER RISK",
                "deltas": {
                    "risk_score_delta": 9.4,
                    "evidence_strength_delta": -0.11,
                    "mean_rating_delta": -0.417634,
                },
                "relation": "worse",
                "relation_types": ["also_viewed"],
            },
        ],
        "empty_reason": None,
        "total_related": 62,
        "usable": 11,
    },
    "verdict": {
        "label": "CONSIDER WITH CAUTION",
        "risk_score": 62.5,
        "evidence_strength": 0.7173,
        "major_concerns": [
            {
                "code": "high_negative_sentiment",
                "text": "23 of 52 reviews (44.2%) carry the weak negative label, well above "
                "the 25% caution band.",
                "severity": "high",
                "supporting_numbers": {
                    "negative_share": 0.4423,
                    "negative_count": 23.0,
                    "total_reviews": 52.0,
                },
            },
            {
                "code": "low_mean_rating",
                "text": "The mean rating is 2.63 out of 5, with 19 of 52 reviews rated one "
                "star.",
                "severity": "medium",
                "supporting_numbers": {
                    "mean_rating": 2.634615,
                    "one_star_count": 19.0,
                    "one_star_share": 0.3654,
                },
            },
        ],
        "rationale": "Consider with caution: 23 of 52 reviews (44.2%) are weakly negative "
        "and the mean rating is 2.63 out of 5, but the evidence base is comparatively "
        "strong (49 of 52 verified purchases) and a safer alternative exists.",
    },
    "risk_score": 62.5,
    "evidence_strength": 0.7173,
    "pipeline_version": PIPELINE_VERSION,
    "generated_at": EXAMPLE_GENERATED_AT,
    "data_basis": {
        "category": "Appliances",
        "n_reviews": 52,
        "date_range_start": "2009-07-20T00:00:00Z",
        "date_range_end": "2018-02-03T00:00:00Z",
        "reference_date": "2018-10-04T00:00:00Z",
    },
    "warnings": [
        "Sentiment labels are weak supervision from star ratings (1-2 negative, 3 neutral, "
        "4-5 positive), not model sentiment.",
        "Evidence base of 52 reviews is below the category median of 111 reviews.",
    ],
}


class DataBasis(BaseModel):
    """What the analysis was computed from (so "recent" is always anchored)."""

    model_config = ConfigDict(from_attributes=True)

    category: str = Field(description="UCSD Amazon Review Data (2018) category analysed.")
    n_reviews: int = Field(
        ..., ge=0, description="Reviews for this product in the processed subset."
    )
    date_range_start: UtcDatetime = Field(
        description="Timestamp of the product's oldest stored review (UTC)."
    )
    date_range_end: UtcDatetime = Field(
        description="Timestamp of the product's newest stored review (UTC)."
    )
    reference_date: UtcDatetime = Field(
        description="The date recency was measured from: the latest review in the "
        "processed category (2018-10-04 for this dataset), never today.",
    )


class AnalysisResponse(BaseModel):
    """The full analysis of one product, from evidence to verdict."""

    model_config = ConfigDict(from_attributes=True, json_schema_extra={"example": ANALYSIS_EXAMPLE})

    asin: str = Field(description="Analysed product ASIN.")
    product: ProductDetail = Field(description="Product profile behind the analysis.")
    review_analysis: ReviewAnalysis
    evidence: EvidenceIndicators
    risk_indicators: list[EvidenceIndicator] = Field(
        default_factory=list,
        description="The risk stage's five criteria as 0-1 indicators with direction "
        "and description: product_quality, evidence_strength, rating_reliability, "
        "negative_feedback_concentration and price_position. Empty until the risk "
        "stage computes them; exposed at the top level so clients do not have to "
        "reach into the TOPSIS block for risk inputs.",
    )
    decision: DecisionAssessment | None = Field(
        default=None,
        description="Fuzzy + TOPSIS decision; None when the risk stages could not run "
        "(see warnings).",
    )
    explanation: ShapExplanation | None = Field(
        default=None,
        description="Surrogate-based SHAP explanation; None when no surrogate was fitted.",
    )
    alternatives: AlternativesResult = Field(
        description="Comparable alternatives, or an explicit reason there are none."
    )
    verdict: ConsumerVerdict | None = Field(
        default=None,
        description="Final verdict; None when the decision stages could not run "
        "(see warnings).",
    )
    risk_score: RiskScore | None = Field(
        default=None,
        description="Headline consumer risk 0-100 (mirrors verdict.risk_score); None "
        "when the risk stage did not run.",
    )
    evidence_strength: UnitFloat | None = Field(
        default=None,
        description="Headline evidence strength 0-1 (mirrors verdict.evidence_strength); "
        "None when it was not computed.",
    )
    pipeline_version: str = Field(description="Version of the pipeline that produced this.")
    generated_at: UtcDatetime = Field(
        description="When this analysis was generated (UTC); data-derived fields stay "
        "anchored on data_basis.reference_date instead.",
    )
    data_basis: DataBasis
    warnings: list[str] = Field(
        default_factory=list,
        description="Degraded or low-evidence conditions detected while producing this "
        "analysis; empty when everything was computed normally.",
    )


__all__ = ["ANALYSIS_EXAMPLE", "AnalysisResponse", "DataBasis", "PIPELINE_VERSION"]

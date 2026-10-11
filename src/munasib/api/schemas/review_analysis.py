"""Review-analysis contract: rating distribution, sentiment, temporal trend, aspects.

Every share is a ratio of the product's stored reviews (unit interval), and every
number comes from the processed subset - the "recent" windows used by the
pipeline are anchored on ``reference_date`` from
:class:`~munasib.api.schemas.analysis.DataBasis` (the latest review in the
dataset, 2018-10-04), never on today's date. Sentiment numbers are weak labels
derived from star ratings, not model sentiment; the model's own predictions
arrive in a later stage.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import UnitFloat, UtcDatetime


class AspectPolarity(StrEnum):
    """Whether an aspect is mentioned favourably, unfavourably, or both."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"


class RatingDistribution(BaseModel):
    """Star histogram for one product (counts and shares for every star 1-5)."""

    model_config = ConfigDict(
        from_attributes=True,
        strict=False,
        json_schema_extra={
            "example": {
                "counts": {"1": 19, "2": 4, "3": 14, "4": 7, "5": 8},
                "shares": {"1": 0.3654, "2": 0.0769, "3": 0.2692, "4": 0.1346, "5": 0.1538},
                "mean_rating": 2.634615,
                "total": 52,
            }
        },
    )

    counts: dict[int, int] = Field(
        description="Reviews per star (1-5); every star is present, 0 when absent."
    )
    shares: dict[int, UnitFloat] = Field(
        description="Count divided by `total` for every star 1-5 (shares sum to 1)."
    )
    mean_rating: float | None = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="Average star rating of these reviews; None when unrated.",
    )
    total: int = Field(..., ge=0, description="Reviews in this distribution.")

    @field_validator("counts", "shares")
    @classmethod
    def _validate_star_keys(cls, value: dict[int, object]) -> dict[int, object]:
        """Require keys for every star 1-5 (never silently drop an absent star).

        Args:
            value: Parsed star-keyed mapping.

        Returns:
            The unchanged mapping.

        Raises:
            ValueError: If a key is outside 1-5 or a star is missing.
        """
        expected = {1, 2, 3, 4, 5}
        keys = set(value)
        unknown = sorted(keys - expected)
        if unknown:
            raise ValueError(f"unexpected star key(s) {unknown}; expected {sorted(expected)}")
        missing = sorted(expected - keys)
        if missing:
            raise ValueError(f"missing star key(s) {missing}; every star 1-5 must be present")
        return value


class SentimentBreakdown(BaseModel):
    """Weak-supervision sentiment counts and shares for one product.

    ``*_share`` values are counts divided by ``total``. ``unlabelled_count`` counts
    reviews with no rating, which therefore carry no weak label at all.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "total": 52,
                "negative_count": 23,
                "neutral_count": 14,
                "positive_count": 15,
                "unlabelled_count": 0,
                "negative_share": 0.4423,
                "neutral_share": 0.2692,
                "positive_share": 0.2885,
            }
        },
    )

    total: int = Field(..., ge=0, description="Reviews counted.")
    negative_count: int = Field(..., ge=0, description="Reviews with the negative weak label.")
    neutral_count: int = Field(..., ge=0, description="Reviews with the neutral weak label.")
    positive_count: int = Field(..., ge=0, description="Reviews with the positive weak label.")
    unlabelled_count: int = Field(
        ..., ge=0, description="Reviews with no rating, hence no weak label."
    )
    negative_share: UnitFloat
    neutral_share: UnitFloat
    positive_share: UnitFloat

    @field_validator("unlabelled_count")
    @classmethod
    def _check_totals(cls, value: int) -> int:
        """Keep the unlabelled count within the total (guard against bad payloads).

        Args:
            value: Parsed unlabelled count.

        Returns:
            The unchanged value.

        Raises:
            ValueError: If the count is negative.
        """
        if value < 0:
            raise ValueError("unlabelled_count must be non-negative")
        return value


class TemporalBucket(BaseModel):
    """Sentiment shares for one period of a product's review timeline."""

    model_config = ConfigDict(from_attributes=True)

    period_start: UtcDatetime = Field(
        description="First instant of the period (month granularity), UTC."
    )
    n_reviews: int = Field(..., ge=0, description="Reviews in this period.")
    negative_share: UnitFloat
    neutral_share: UnitFloat
    positive_share: UnitFloat


class TemporalTrend(BaseModel):
    """Per-period sentiment shares plus the trend slope.

    ``buckets`` covers periods that contain at least one review (empty months are
    not fabricated). ``positive_share_slope`` is the ordinary-least-squares slope
    of ``positive_share`` against the bucket index: positive means sentiment
    improved over time. It is None when fewer than two periods exist.
    """

    model_config = ConfigDict(from_attributes=True)

    period: str = Field(description="Bucket granularity; 'month' for this contract.")
    buckets: list[TemporalBucket] = Field(description="Non-empty periods in ascending order.")
    positive_share_slope: float | None = Field(
        default=None,
        description="OLS slope of positive_share per period (None with < 2 periods).",
    )
    n_reviews_used: int = Field(..., ge=0, description="Reviews covered by these buckets.")


class AspectInsight(BaseModel):
    """One product aspect, its polarity and the reviews that mention it."""

    model_config = ConfigDict(from_attributes=True)

    aspect: str = Field(description="Short aspect noun phrase, e.g. 'fit' or 'value'.")
    polarity: AspectPolarity = Field(
        description="Overall leaning of the mentions for this aspect."
    )
    mention_count: int = Field(..., ge=0, description="Reviews that mention the aspect.")
    examples: list[str] = Field(
        default_factory=list,
        description="Up to a few verbatim review excerpts that mention the aspect.",
    )


class ReviewAnalysis(BaseModel):
    """The whole review-analysis block attached to a product analysis."""

    model_config = ConfigDict(from_attributes=True)

    rating_distribution: RatingDistribution
    sentiment: SentimentBreakdown
    temporal_trend: TemporalTrend
    aspects: list[AspectInsight] = Field(description="Top aspects by mention count.")


__all__ = [
    "AspectInsight",
    "AspectPolarity",
    "RatingDistribution",
    "ReviewAnalysis",
    "SentimentBreakdown",
    "TemporalBucket",
    "TemporalTrend",
]

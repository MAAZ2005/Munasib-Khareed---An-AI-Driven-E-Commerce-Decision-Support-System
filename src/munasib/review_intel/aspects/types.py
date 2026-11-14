"""Shared value objects for aspect extraction, vocabulary and aggregation.

The aspect stage has three data shapes:

* :class:`AspectExample` - one verbatim review excerpt selected for the
  dashboard, with the review-level predicted label and helpfulness that chose
  it.
* :class:`AspectLabelCounts` - the per-product label distribution of one
  aspect. Counts are **reviews**, not occurrences: a review that mentions an
  aspect twice still contributes one mention.
* :class:`ProductAspectStats` - the complete row persisted in
  ``product_aspects`` (counts, shares, lift, classification and examples).

Nothing here fabricates a value. A missing lift is ``None``; shares are
computed only from stored counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

#: Stored classification labels. A pain point must satisfy every configured
#: rule; a strength is the mutually exclusive positive rule; everything else is
#: neutral and is still stored so the dashboard can show it.
ASPECT_PAIN_POINT = "pain_point"
ASPECT_STRENGTH = "strength"
ASPECT_NEUTRAL = "neutral"

#: All valid stored classifications, in display order (problems first).
ASPECT_CLASSIFICATIONS: tuple[str, ...] = (
    ASPECT_PAIN_POINT,
    ASPECT_STRENGTH,
    ASPECT_NEUTRAL,
)


@dataclass(frozen=True)
class AspectExample:
    """One verbatim review excerpt selected for an aspect.

    Attributes:
        review_id: Stable review identifier (tie-breaker for helpfulness).
        label: Review-level predicted sentiment label (0/1/2) used for polarity.
        helpful_votes: Stored helpful-vote count (missing votes are stored as 0).
        text: Exact prefix of the review text, at most ``example_max_chars``.
    """

    review_id: str
    label: int
    helpful_votes: int
    text: str

    def sort_key(self) -> tuple[int, str]:
        """Return the deterministic example ranking key.

        Returns:
            ``(-helpful_votes, review_id)``: most helpful first, then the stable
            review id so two runs over the same cache select the same excerpt.
        """
        return (-self.helpful_votes, self.review_id)


@dataclass
class AspectLabelCounts:
    """Review-level predicted-label counts for one product aspect.

    The counts always sum to :attr:`mentions`; reviews without a predicted
    label are excluded before aggregation rather than assigned a polarity.
    """

    negative: int = 0
    neutral: int = 0
    positive: int = 0

    @property
    def mentions(self) -> int:
        """Return the number of reviews that mention the aspect.

        Returns:
            Sum of the three label counts.
        """
        return self.negative + self.neutral + self.positive

    def add(self, label: int) -> None:
        """Record one mention with a predicted label.

        Args:
            label: Predicted sentiment code (0 negative, 1 neutral, 2 positive).

        Raises:
            ValueError: If ``label`` is not one of the three sentiment codes.
        """
        if label == 0:
            self.negative += 1
        elif label == 1:
            self.neutral += 1
        elif label == 2:
            self.positive += 1
        else:
            raise ValueError(f"unknown predicted sentiment label {label!r}")

    @property
    def negative_share(self) -> float:
        """Return the share of mentions whose review was predicted negative.

        Returns:
            ``negative / mentions``, or 0.0 when there are no mentions.
        """
        return self.negative / self.mentions if self.mentions else 0.0

    @property
    def neutral_share(self) -> float:
        """Return the share of mentions whose review was predicted neutral.

        Returns:
            ``neutral / mentions``, or 0.0 when there are no mentions.
        """
        return self.neutral / self.mentions if self.mentions else 0.0

    @property
    def positive_share(self) -> float:
        """Return the share of mentions whose review was predicted positive.

        Returns:
            ``positive / mentions``, or 0.0 when there are no mentions.
        """
        return self.positive / self.mentions if self.mentions else 0.0

    @property
    def net_sentiment(self) -> float:
        """Return the polarity balance in ``[-1, 1]``.

        Returns:
            ``(positive - negative) / mentions``, or 0.0 when there are no
            mentions. A value near -1 means almost every mention came from a
            negative review; near +1 means almost every mention was positive.
        """
        return (self.positive - self.negative) / self.mentions if self.mentions else 0.0

    def dominant_label(self) -> int:
        """Return the most frequent label, with the documented tie order.

        Ties resolve in the order negative, neutral, positive so a polarised
        aspect never silently becomes "positive" just because of iteration
        order.

        Returns:
            One of the three sentiment label codes.
        """
        ranked = (
            (self.negative, 0),
            (self.neutral, 1),
            (self.positive, 2),
        )
        return max(ranked, key=lambda pair: (pair[0], -pair[1]))[1]


@dataclass(frozen=True)
class ProductAspectStats:
    """One persisted ``product_aspects`` row."""

    asin: str
    aspect: str
    counts: AspectLabelCounts
    product_labeled_reviews: int
    product_negative_share: float
    negativity_lift: float | None
    classification: str
    examples: list[AspectExample] = field(default_factory=list)

    @property
    def examples_text(self) -> list[str]:
        """Return the dashboard-ready example strings.

        Returns:
            The selected excerpts in their stored order.
        """
        return [example.text for example in self.examples]

    def to_row(self, computed_at: datetime) -> dict[str, object]:
        """Convert this value object into a table insert/upsert payload.

        Args:
            computed_at: Dataset build timestamp (naive UTC, from
                ``data/processed/meta.json``).

        Returns:
            Mapping with every ``product_aspects`` column.
        """
        if computed_at.tzinfo is not None:
            raise ValueError(
                "computed_at must be naive UTC (see munasib.db.models); "
                f"got tzinfo={computed_at.tzinfo!r}"
            )
        return {
            "asin": self.asin,
            "aspect": self.aspect,
            "mention_count": self.counts.mentions,
            "negative_count": self.counts.negative,
            "neutral_count": self.counts.neutral,
            "positive_count": self.counts.positive,
            "negative_share": self.counts.negative_share,
            "neutral_share": self.counts.neutral_share,
            "positive_share": self.counts.positive_share,
            "net_sentiment": self.counts.net_sentiment,
            "product_labeled_reviews": self.product_labeled_reviews,
            "product_negative_share": self.product_negative_share,
            "negativity_lift": self.negativity_lift,
            "classification": self.classification,
            "examples": self.examples_text,
            "computed_at": computed_at,
        }


@dataclass(frozen=True)
class VocabularyEntry:
    """One aspect that passed the category-level minimum review count."""

    aspect: str
    review_count: int
    product_count: int


@dataclass(frozen=True)
class VocabularyBuildResult:
    """Vocabulary plus the raw counts needed for the inspection printout."""

    entries: list[VocabularyEntry]
    all_review_counts: dict[str, int]
    all_product_counts: dict[str, int]
    min_review_count: int

    def nearest_cutoff(self, limit: int) -> list[tuple[str, int]]:
        """Return aspects nearest the vocabulary cutoff on either side.

        Args:
            limit: Maximum number of aspects to return.

        Returns:
            ``(aspect, review_count)`` pairs ordered by absolute distance to
            ``min_review_count``, then by count descending, then by aspect.
        """
        ranked = sorted(
            self.all_review_counts.items(),
            key=lambda item: (
                abs(item[1] - self.min_review_count),
                -item[1],
                item[0],
            ),
        )
        return ranked[:limit]

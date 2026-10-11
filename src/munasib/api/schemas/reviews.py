"""Review contract: the weak-supervision sentiment vocabulary and :class:`ReviewSnippet`.

The API exposes sentiment **names**, never the raw integer codes: the codes live
in :mod:`munasib.core.constants` (locked design decision #2 - weak supervision
from star ratings), and :class:`SentimentLabel` is the single place that maps a
code to the name shown to users. :class:`SentimentLabel.from_code` performs the
conversion, so a database row can be handed to the model directly.

Text is served verbatim from the cache: ``review_text`` is the raw review body
kept for display, and ``summary`` is the cleaned headline. Reviews whose source
body was empty are still returned with ``has_text=False`` (they were kept, never
dropped).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from munasib.core.constants import (
    SENTIMENT_LABEL_NAMES,
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABEL_POSITIVE,
)

from .common import Page, StarRating, UtcDatetime


class SentimentLabel(StrEnum):
    """Weak-supervision sentiment label names (the API never exposes 0/1/2)."""

    NEGATIVE = SENTIMENT_LABEL_NAMES[SENTIMENT_LABEL_NEGATIVE]
    NEUTRAL = SENTIMENT_LABEL_NAMES[SENTIMENT_LABEL_NEUTRAL]
    POSITIVE = SENTIMENT_LABEL_NAMES[SENTIMENT_LABEL_POSITIVE]

    @classmethod
    def from_code(cls, code: int | None) -> SentimentLabel | None:
        """Map a weak-label code from the database to its API name.

        Args:
            code: Integer code (0 negative, 1 neutral, 2 positive) or None.

        Returns:
            The matching label, or None when the code is None (unlabelled).

        Raises:
            ValueError: If the code is not one of the shared label codes.
        """
        if code is None:
            return None
        name = SENTIMENT_LABEL_NAMES.get(int(code))
        if name is None:
            raise ValueError(f"unknown weak-supervision label code {code!r}")
        return cls(name)


def _coerce_sentiment_label(value: Any) -> Any:
    """Coerce enum names or integer codes into :class:`SentimentLabel`.

    Args:
        value: Enum member, code, name, or None.

    Returns:
        The coerced value (unchanged for anything pydantic already consumed).
    """
    if value is None or isinstance(value, SentimentLabel):
        return value
    if isinstance(value, bool):
        raise ValueError("sentiment label must be an integer code or a name, not a bool")
    if isinstance(value, int) or (isinstance(value, str) and value.lstrip("-").isdigit()):
        coerced = SentimentLabel.from_code(int(value))
        if coerced is None:
            raise ValueError("sentiment label code must not be empty")
        return coerced
    return value


#: Reverse of :meth:`SentimentLabel.from_code`: the label name used by the API back
#: to the integer code stored in the cache (derived from the shared constants).
SENTIMENT_LABEL_CODE_FOR_NAME: dict[SentimentLabel, int] = {
    SentimentLabel(name): code for code, name in SENTIMENT_LABEL_NAMES.items()
}


REVIEW_SNIPPET_EXAMPLE: dict[str, Any] = {
    "review_id": "1b5dbcd03044cba1",
    "asin": "B00009V3UF",
    "reviewer_id": "A2MBG73LAWGRDZ",
    "rating": 1.0,
    "verified": True,
    "helpful_votes": 0,
    "review_time": "2018-02-03T00:00:00Z",
    "review_text": "These knobs are supposed to have rings you can place on them with "
    "temperatures so you know where to turn the knob. They were missing the rings, or as I "
    "can see from other reviews - they simply don't come with them even though the packaging "
    "states that they do. Don't waste your money on these unless the cooktop itself has the "
    "temperature markings.",
    "has_text": True,
    "text_length": 348,
    "summary": "Packaging incorrect - no temperature markings are included.",
    "sentiment_label": "negative",
}

REVIEW_SNIPPET_EXAMPLE_2: dict[str, Any] = {
    "review_id": "9ee8b475b1af2971",
    "asin": "B00009V3UF",
    "reviewer_id": "A1W1EHTEKYJIKJ",
    "rating": 5.0,
    "verified": True,
    "helpful_votes": 0,
    "review_time": "2017-12-15T00:00:00Z",
    "review_text": "Exactly as expected",
    "has_text": True,
    "text_length": 19,
    "summary": "Five Stars",
    "sentiment_label": "positive",
}


class ReviewSnippet(BaseModel):
    """One review as shown in product pages (identity, rating and text)."""

    model_config = ConfigDict(
        from_attributes=True, populate_by_name=True, json_schema_extra={"example": REVIEW_SNIPPET_EXAMPLE}
    )

    review_id: str = Field(
        description="Deterministic 16-hex id assigned by the preprocessing pipeline."
    )
    asin: str = Field(description="Product the review belongs to.")
    reviewer_id: str = Field(description="Reviewer id exactly as published in the source.")
    rating: StarRating | None = Field(
        default=None,
        description="Star rating 1-5; None only when the source row had no rating "
        "(the sentiment label would be unlabelled too).",
    )
    verified: bool | None = Field(
        default=None,
        description="True/False verified-purchase flag; None means the source did not say, "
        "which is NOT the same as False.",
    )
    helpful_votes: int = Field(
        ge=0,
        description="Helpful votes recorded in the source; a missing vote count is stored as 0.",
    )
    review_time: UtcDatetime = Field(description="UTC timestamp of the review.")
    review_text: str | None = Field(
        default=None,
        validation_alias=AliasChoices("review_text", "review_text_raw"),
        description="Review body exactly as stored (kept for display); None when the "
        "source had no text.",
    )
    has_text: bool = Field(
        description="False for rows kept without usable review text (never dropped)."
    )
    text_length: int = Field(
        ge=0, description="Character length of the cleaned review text."
    )
    summary: str | None = Field(
        default=None,
        description="Review headline as published; None when the source had none.",
    )
    sentiment_label: SentimentLabel | None = Field(
        default=None,
        validation_alias=AliasChoices("sentiment_label", "sentiment_label_weak"),
        description="Weak label derived from the star rating (1-2 negative, 3 neutral, "
        "4-5 positive) - a weak signal, not true sentiment; None when unrated.",
    )

    @field_validator("sentiment_label", mode="before")
    @classmethod
    def _coerce_label(cls, value: Any) -> Any:
        """Accept either the API name or the stored integer code.

        Args:
            value: Raw input value.

        Returns:
            The coerced :class:`SentimentLabel` or None.
        """
        return _coerce_sentiment_label(value)


class ReviewPage(Page[ReviewSnippet]):
    """Paginated reviews for one product (newest first, stable by review id)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 52,
                "limit": 2,
                "offset": 0,
                "items": [REVIEW_SNIPPET_EXAMPLE, REVIEW_SNIPPET_EXAMPLE_2],
            }
        }
    )


__all__ = [
    "REVIEW_SNIPPET_EXAMPLE",
    "REVIEW_SNIPPET_EXAMPLE_2",
    "ReviewPage",
    "ReviewSnippet",
    "SENTIMENT_LABEL_CODE_FOR_NAME",
    "SentimentLabel",
]

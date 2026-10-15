"""Alternatives contract: safer/approved substitutes for a product.

An alternative is only offered when it is itself a processed subset product with
an analysis; edges pointing outside the subset are counted in ``total_related``
but cannot be compared, and a product with no usable alternative is a valid
answer (about 18% of products in this category have no usable alternative) - so
``empty_reason`` exists precisely for that case and is required whenever
``items`` is empty.

``relation`` is the consumer-facing summary of the comparison deltas: ``better``,
``comparable`` or ``worse`` than the subject. All deltas are ``alternative minus
subject`` (negative risk delta = the alternative is safer).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .products import ProductSummary
from .verdict import VerdictLabel


class AlternativeRelation(StrEnum):
    """How an alternative compares with the subject product."""

    BETTER = "better"
    COMPARABLE = "comparable"
    WORSE = "worse"


class ComparisonDeltas(BaseModel):
    """Signed differences between an alternative and the subject product."""

    model_config = ConfigDict(from_attributes=True)

    risk_score_delta: float = Field(
        description="Alternative risk minus subject risk; negative means safer."
    )
    evidence_strength_delta: float = Field(
        description="Alternative evidence strength minus subject evidence strength."
    )
    mean_rating_delta: float | None = Field(
        default=None,
        description="Alternative mean rating minus subject mean rating; None if either "
        "has no rating.",
    )


class AlternativeProduct(BaseModel):
    """One alternative with its own verdict and its deltas against the subject."""

    model_config = ConfigDict(from_attributes=True)

    summary: ProductSummary = Field(description="The alternative's product summary.")
    risk_score: float = Field(description="Alternative's consumer risk, 0-100.")
    verdict: VerdictLabel = Field(description="Alternative's own verdict label.")
    deltas: ComparisonDeltas = Field(description="Differences versus the subject product.")
    relation: AlternativeRelation = Field(
        description="better / comparable / worse than the subject."
    )
    relation_types: list[str] = Field(
        default_factory=list,
        description="Edge types that link the two products (also_bought / also_viewed).",
    )


ALTERNATIVES_EXAMPLE: dict[str, Any] = {
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
}

#: Real empty case: this product has no related-product edges at all in the
#: processed subset (checked against the serving cache).
ALTERNATIVES_EMPTY_EXAMPLE: dict[str, Any] = {
    "items": [],
    "empty_reason": "no related-product edges in the processed subset "
    "(0 also_bought / also_viewed links)",
    "total_related": 0,
    "usable": 0,
}

#: What the example numbers above were verified against: the serving cache for
#: B00009V3UF has 62 related-product edges in total (4 ``also_bought`` +
#: 58 ``also_viewed``), of which 12 edges point back into the processed subset and
#: cover 11 distinct usable alternatives. ``total_related`` therefore counts
#: every edge while ``usable`` counts distinct in-subset alternatives, which is
#: why one ASIN linked by both edge types is counted twice in the first number
#: and once in the second.
EXAMPLE_VERIFIED_COUNTS: dict[str, int] = {
    "total_related": 62,
    "in_subset_edges": 12,
    "usable": 11,
}


class AlternativesResult(BaseModel):
    """Alternatives for one product, or an explicit reason there are none."""

    model_config = ConfigDict(
        from_attributes=True, json_schema_extra={"example": ALTERNATIVES_EXAMPLE}
    )

    items: list[AlternativeProduct] = Field(
        default_factory=list,
        description="Comparable alternatives; empty when nothing usable was found.",
    )
    empty_reason: str | None = Field(
        default=None,
        description="Why items is empty (e.g. no related-product edges in the subset, or "
        "no edge target is itself a processed product); None when items is non-empty.",
    )
    total_related: int = Field(
        ...,
        ge=0,
        description="Related-product edges found for the product (counting both edge "
        "types per target).",
    )
    usable: int = Field(
        ...,
        ge=0,
        description="Distinct alternatives that could actually be compared "
        "(subset products with an analysis).",
    )

    @model_validator(mode="after")
    def _check_empty_reason(self) -> AlternativesResult:
        """Keep ``empty_reason`` in sync with ``items`` and usable counts sane.

        Returns:
            The validated model.

        Raises:
            ValueError: If ``items`` is empty without a reason, non-empty with a
                reason, or if ``usable`` exceeds the distinct alternatives.
        """
        if not self.items:
            if not self.empty_reason:
                raise ValueError("empty_reason is required when items is empty")
        elif self.empty_reason:
            raise ValueError("empty_reason must be None when items is not empty")
        if self.usable > self.total_related:
            raise ValueError("usable cannot exceed total_related")
        if len(self.items) > self.usable:
            raise ValueError("items cannot exceed usable")
        return self


__all__ = [
    "ALTERNATIVES_EMPTY_EXAMPLE",
    "ALTERNATIVES_EXAMPLE",
    "AlternativeProduct",
    "AlternativeRelation",
    "AlternativesResult",
    "ComparisonDeltas",
    "EXAMPLE_VERIFIED_COUNTS",
]

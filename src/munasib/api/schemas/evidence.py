"""Evidence contract: strength indicators with their raw values.

Every indicator carries the normalised 0-1 ``value`` **and** the ``raw_value``
it was derived from, plus ``direction`` (benefit/cost) and a plain-language
``description`` - a "verified purchases are rare" flag is useless without the
ratio and the review count it came from. Indicator names match the weight keys
of ``review_intel.evidence`` in ``configs/config.yaml``; the composite
``evidence_strength`` is the weighted mean defined by that config block.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .common import UnitFloat

#: Indicator names as configured in `configs/config.yaml` (review_intel.evidence).
INDICATOR_VERIFIED_PURCHASE = "verified_purchase"
INDICATOR_HELPFULNESS = "helpfulness_ratio"
INDICATOR_REVIEW_DEPTH = "review_depth"


class IndicatorDirection(StrEnum):
    """Whether more of the raw quantity helps (benefit) or hurts (cost)."""

    BENEFIT = "benefit"
    COST = "cost"


class EvidenceIndicator(BaseModel):
    """One indicator: the 0-1 value plus everything needed to audit it."""

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(
        description=f"Indicator key; one of {INDICATOR_VERIFIED_PURCHASE!r}, "
        f"{INDICATOR_HELPFULNESS!r}, {INDICATOR_REVIEW_DEPTH!r}."
    )
    value: UnitFloat = Field(description="Normalised indicator value, 0.0-1.0 (1.0 = best).")
    raw_value: float | None = Field(
        default=None,
        description="Value in its natural unit before normalisation (e.g. mean review "
        "length in characters); None when the raw input was unavailable.",
    )
    direction: IndicatorDirection = Field(
        description="benefit = a higher raw value is better for the consumer; "
        "cost = a higher raw value is worse.",
    )
    description: str = Field(
        description="One sentence explaining what this indicator measures in plain language."
    )


class EvidenceIndicators(BaseModel):
    """The three evidence indicators plus the composite 0-1 evidence strength."""

    model_config = ConfigDict(from_attributes=True)

    verified_purchase: EvidenceIndicator = Field(
        description="Share of the product's reviews that are verified purchases."
    )
    helpfulness: EvidenceIndicator = Field(
        description="How helpful other shoppers found the reviews (helpful-vote signal)."
    )
    review_depth: EvidenceIndicator = Field(
        description="How substantive the review text is (length-based depth)."
    )
    evidence_strength: UnitFloat = Field(
        description="Weighted composite of the indicators, 0.0-1.0; low values mean the "
        "product's evidence base is too thin to trust."
    )


__all__ = [
    "EvidenceIndicator",
    "EvidenceIndicators",
    "IndicatorDirection",
]

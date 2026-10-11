"""Evidence contract: strength indicators with their raw values.

Every indicator carries the normalised 0-1 ``value`` **and** the ``raw_value``
it was derived from, plus ``direction`` (benefit/cost) and a plain-language
``description`` - a "verified purchases are rare" flag is useless without the
ratio and the review count it came from. Indicator names match the weight keys
of ``review_intel.evidence`` in ``configs/config.yaml``; the composite
``evidence_strength`` is the weighted mean defined by that config block.

The contract is extensible without breaking clients: signals computed beyond
those three configured indicators (rating-distribution entropy,
helpfulness-weighted sentiment, ...) travel in
``EvidenceIndicators.details`` as full ``EvidenceIndicator`` entries, each
with its own 0-1 value, raw value, direction and description. Names stay a
reliable key: a name repeated inside ``details`` or reused from a fixed
indicator is rejected by a validator instead of silently colliding.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import UnitFloat

#: Indicator names as configured in `configs/config.yaml` (review_intel.evidence).
INDICATOR_VERIFIED_PURCHASE = "verified_purchase"
INDICATOR_HELPFULNESS = "helpfulness_ratio"
INDICATOR_REVIEW_DEPTH = "review_depth"

#: Keys of the three fixed indicators. They already travel as dedicated fields,
#: so the extensible `details` list may not reuse them (a duplicate key would
#: leave clients guessing which indicator a name refers to).
FIXED_INDICATOR_NAMES: tuple[str, ...] = (
    INDICATOR_VERIFIED_PURCHASE,
    INDICATOR_HELPFULNESS,
    INDICATOR_REVIEW_DEPTH,
)


class IndicatorDirection(StrEnum):
    """Whether more of the raw quantity helps (benefit) or hurts (cost)."""

    BENEFIT = "benefit"
    COST = "cost"


class EvidenceIndicator(BaseModel):
    """One indicator: the 0-1 value plus everything needed to audit it."""

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(
        description="Indicator key. The fixed evidence indicators are "
        f"{INDICATOR_VERIFIED_PURCHASE!r}, {INDICATOR_HELPFULNESS!r} and "
        f"{INDICATOR_REVIEW_DEPTH!r}; `details` entries carry additional unique "
        "keys (e.g. 'rating_distribution_entropy')."
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
    """The three evidence indicators, the composite strength, and any extras.

    ``details`` extends the contract without changing the four original fields:
    each entry is a full :class:`EvidenceIndicator` for a signal computed
    beyond ``configs/config.yaml`` (rating-distribution entropy,
    helpfulness-weighted sentiment, ...), so new signals need no schema change.
    A name may appear only once in ``details`` and may not shadow
    :data:`FIXED_INDICATOR_NAMES`; both violations are rejected because
    clients key indicators by name.
    """

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
    details: list[EvidenceIndicator] = Field(
        default_factory=list,
        description="Additional evidence signals computed beyond the three fixed "
        "indicators, each with a 0-1 value, its raw value, direction and a "
        "plain-language description; empty when no extra signals were computed. "
        "Names are unique within this list and never reuse the fixed keys "
        f"{FIXED_INDICATOR_NAMES!r}.",
    )

    @field_validator("details")
    @classmethod
    def _validate_detail_names(cls, value: list[EvidenceIndicator]) -> list[EvidenceIndicator]:
        """Reject duplicate indicator names in the extensible ``details`` list.

        Args:
            value: Parsed details list.

        Returns:
            The unchanged list.

        Raises:
            ValueError: If a name repeats inside ``details`` or shadows one of
                the three fixed indicator names.
        """
        seen: set[str] = set()
        duplicates: set[str] = set()
        shadowed: set[str] = set()
        for indicator in value:
            if indicator.name in seen:
                duplicates.add(indicator.name)
            seen.add(indicator.name)
            if indicator.name in FIXED_INDICATOR_NAMES:
                shadowed.add(indicator.name)
        problems: list[str] = []
        if duplicates:
            problems.append(f"duplicate detail name(s) {sorted(duplicates)}")
        if shadowed:
            problems.append(
                f"detail name(s) {sorted(shadowed)} shadow the fixed indicators "
                f"{list(FIXED_INDICATOR_NAMES)}"
            )
        if problems:
            raise ValueError("; ".join(problems))
        return value


__all__ = [
    "EvidenceIndicator",
    "EvidenceIndicators",
    "IndicatorDirection",
]

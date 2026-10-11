"""Verdict contract: the consumer-facing decision and the concerns behind it.

The label is one of exactly three values (locked design decision #5), expressed
as a string enum whose values are the shared constants from
:mod:`munasib.core.constants`. Every concern carries the numbers that support
it, and the rationale is required to cite real computed numbers - the pipeline
is never allowed to narrate without evidence.

Guardrail (from ``configs/verdict_thresholds.yaml``): a product with very weak
evidence can never receive a clean ``CONSIDER``; the verdict layer enforces it
and the API exposes the consequence in ``warnings`` of
:class:`~munasib.api.schemas.analysis.AnalysisResponse`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from munasib.core.constants import VERDICT_CAUTION, VERDICT_CONSIDER, VERDICT_RISK

from .common import RiskScore, UnitFloat


class VerdictLabel(StrEnum):
    """The three (and only three) verdict labels, increasing in severity."""

    CONSIDER = VERDICT_CONSIDER
    CONSIDER_WITH_CAUTION = VERDICT_CAUTION
    CONSUMER_RISK = VERDICT_RISK


class ConcernSeverity(StrEnum):
    """How much a single concern weighs on the verdict."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Concern(BaseModel):
    """One specific consumer concern with the numbers that back it."""

    model_config = ConfigDict(from_attributes=True)

    code: str = Field(
        description="Stable machine token for the concern type, e.g. "
        "'high_negative_sentiment'.",
    )
    text: str = Field(
        description="Consumer-readable explanation that quotes the supporting numbers."
    )
    severity: ConcernSeverity = Field(description="How much this concern weighs.")
    supporting_numbers: dict[str, float] = Field(
        default_factory=dict,
        description="The computed values quoted by the text (e.g. negative_share: 0.4423).",
    )


class ConsumerVerdict(BaseModel):
    """The final verdict for one product."""

    model_config = ConfigDict(from_attributes=True)

    label: VerdictLabel = Field(description="One of CONSIDER, CONSIDER WITH CAUTION, CONSUMER RISK.")
    risk_score: RiskScore = Field(description="Risk score on 0-100; higher is riskier.")
    evidence_strength: UnitFloat = Field(
        description="Strength of the evidence base, 0.0-1.0; weak evidence blocks CONSIDER."
    )
    major_concerns: list[Concern] = Field(
        default_factory=list,
        description="The concerns that shaped the verdict, most severe first; empty for a "
        "clean CONSIDER.",
    )
    rationale: str = Field(
        description="Consumer-readable summary citing the real numbers behind the verdict."
    )


__all__ = ["Concern", "ConcernSeverity", "ConsumerVerdict", "VerdictLabel"]

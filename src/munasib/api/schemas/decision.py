"""Decision contract: the fuzzy assessment, the TOPSIS ranking that consumes it.

Pipeline order (locked design decision): the fuzzy assessment produces a
defuzzified risk value, that value is handed to TOPSIS as an **extra cost
criterion** (``risk_score`` in ``configs/criteria_weights.yaml``), and the final
consumer risk score is the TOPSIS result combined with the fuzzy uncertainty.
This module exposes both halves so a consumer - or a marker - can audit each
step: which rules fired, with which weights, and where the product landed
among its alternatives plus the sampled category peer set.

All degrees, weights, memberships and sub-contributions are on 0-1; the only
0-100 quantities in this module are the fuzzy output scale and the final
consumer risk score. ``percentile`` is reported on 0-100 because a percentile is
already a percentage: it is the share of the ranked set at or below the subject,
times 100.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import RiskScore, UnitFloat

#: Weights in a TOPSIS problem must sum to 1.0; mirrors the tolerance documented
#: in `validation.sum_tolerance` of configs/criteria_weights.yaml.
WEIGHT_SUM_TOLERANCE = 0.001

#: Fuzzy output terms as configured in `variables.output.consumer_risk.terms` of
#: configs/fuzzy_rules.yaml (low, moderate, high, critical).
FUZZY_OUTPUT_TERMS = ("low", "moderate", "high", "critical")


class MembershipProfile(BaseModel):
    """Membership degrees of one fuzzy input variable across its linguistic terms."""

    model_config = ConfigDict(from_attributes=True)

    term_degrees: dict[str, UnitFloat] = Field(
        description="Degree (0-1) with which the variable's crisp value belongs to "
        "each linguistic term defined in configs/fuzzy_rules.yaml."
    )


class FiredRule(BaseModel):
    """One fuzzy rule that contributed to the output, with its firing strength."""

    model_config = ConfigDict(from_attributes=True)

    rule_id: int = Field(..., ge=1, description="Rule index from configs/fuzzy_rules.yaml.")
    antecedents: dict[str, str] = Field(
        description="Input variable -> fired term, exactly as configured for the rule."
    )
    consequent: str = Field(description="Output term the rule implies.")
    strength: UnitFloat = Field(description="Firing strength (0-1) of this rule.")


class FuzzyAssessment(BaseModel):
    """The fuzzy stage: memberships, fired rules, output degrees, crisp output.

    ``output_degrees`` is keyed by the fuzzy output terms from
    ``configs/fuzzy_rules.yaml`` - currently ``low``, ``moderate``, ``high`` and
    ``critical`` - and each value is that term's aggregated firing degree (0-1).
    ``defuzzified`` is the crisp value on the fuzzy output scale (0-100), which is
    what TOPSIS receives as its extra cost criterion.
    """

    model_config = ConfigDict(from_attributes=True)

    input_memberships: dict[str, MembershipProfile] = Field(
        description="Per input variable (negative_sentiment, aspect_complaint_intensity, "
        "evidence_volatility): membership degrees for each linguistic term."
    )
    fired_rules: list[FiredRule] = Field(
        description="Rules that fired, strongest first."
    )
    output_degrees: dict[str, UnitFloat] = Field(
        description="Aggregated firing degree per fuzzy output term ("
        + ", ".join(FUZZY_OUTPUT_TERMS)
        + ")."
    )
    defuzzified: RiskScore = Field(
        description="Defuzzified fuzzy risk on the 0-100 output scale; the extra cost "
        "criterion fed to TOPSIS.",
    )
    defuzzification_method: str = Field(
        description="Defuzzification method from the config (default 'centroid')."
    )

    @field_validator("output_degrees")
    @classmethod
    def _validate_terms(cls, value: dict[str, float]) -> dict[str, float]:
        """Reject keys outside the configured fuzzy output vocabulary.

        Args:
            value: Parsed output degrees.

        Returns:
            The unchanged mapping.

        Raises:
            ValueError: If a term is not one of the configured output terms.
        """
        unknown = sorted(set(value) - set(FUZZY_OUTPUT_TERMS))
        if unknown:
            raise ValueError(
                f"unexpected fuzzy output term(s) {unknown}; expected {list(FUZZY_OUTPUT_TERMS)}"
            )
        return value


class TopsisCriterion(BaseModel):
    """One TOPSIS criterion and the subject product's contribution under it."""

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(
        description="Criterion key from configs/criteria_weights.yaml; `risk_score` is "
        "the fuzzy criterion (cost orientation)."
    )
    description: str = Field(description="What the criterion measures, in plain language.")
    weight: UnitFloat = Field(description="Criterion weight (all criteria sum to 1.0).")
    direction: str = Field(
        description="'benefit' (higher is better) or 'cost' (lower is better)."
    )
    value: float = Field(
        description="The subject product's value for this criterion in its natural unit "
        "(0-1 ratios; 0-100 for risk_score)."
    )
    normalized_value: UnitFloat = Field(
        description="Vector-normalised value used inside the TOPSIS matrix."
    )
    contribution: UnitFloat = Field(
        description="This criterion's weighted contribution to the subject's separation "
        "from the negative-ideal solution (weight x distance, 0-1)."
    )


class TopsisResult(BaseModel):
    """The TOPSIS ranking of the subject against alternatives plus category peers."""

    model_config = ConfigDict(from_attributes=True)

    criteria: list[TopsisCriterion] = Field(
        description="All criteria with the subject's values and contributions; the list "
        "includes the fuzzy risk criterion as a cost term."
    )
    closeness: UnitFloat = Field(
        description="TOPSIS closeness coefficient of the subject (1 = ideal, 0 = anti-ideal)."
    )
    rank: int = Field(
        ..., ge=1, description="1-based rank of the subject within the ranked set."
    )
    ranked_count: int = Field(
        ..., ge=1, description="Number of products in the ranked set (subject included)."
    )
    peer_count: int = Field(
        ..., ge=0, description="Sampled category peers (alternatives are included here)."
    )
    percentile: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentile of the ranked set the subject outranks (is safer than): "
        "computed as 100 * (ranked_count - rank) / ranked_count, so rank 1 = 100.0 "
        "(safest) and last = almost 0.0 (riskiest). Reported on 0-100 because it is "
        "a percentage.",
    )
    risk_score: RiskScore = Field(
        description="Final consumer risk: the TOPSIS result combined with the fuzzy "
        "uncertainty (higher is riskier).",
    )

    @field_validator("criteria")
    @classmethod
    def _validate_weight_sum(cls, value: list[TopsisCriterion]) -> list[TopsisCriterion]:
        """Check that the criteria weights form a valid TOPSIS problem.

        Args:
            value: Parsed criteria list.

        Returns:
            The unchanged list.

        Raises:
            ValueError: If the criteria list is empty or the weights do not sum to 1.
        """
        if not value:
            raise ValueError("criteria must not be empty")
        total = sum(criterion.weight for criterion in value)
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"criterion weights must sum to 1.0 (got {total:.4f} "
                f"+- {WEIGHT_SUM_TOLERANCE})"
            )
        return value


class DecisionAssessment(BaseModel):
    """Both decision stages for one product: fuzzy assessment then TOPSIS."""

    model_config = ConfigDict(from_attributes=True)

    fuzzy: FuzzyAssessment
    topsis: TopsisResult


__all__ = [
    "DecisionAssessment",
    "FiredRule",
    "FuzzyAssessment",
    "MembershipProfile",
    "TopsisCriterion",
    "TopsisResult",
]

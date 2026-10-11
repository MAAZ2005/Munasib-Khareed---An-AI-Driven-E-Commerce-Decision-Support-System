"""Explanation contract: SHAP-style contributions plus human-readable reasons.

Locked design decision #3: SHAP explains a **surrogate regressor** trained on
the pipeline's risk scores - not the fuzzy or TOPSIS machinery directly - so the
surrogate's fidelity (R2 and MAE) is always reported alongside the explanation.
A consumer who sees a high risk score must also see how faithful the surrogate
is before trusting the feature ranking.

``contributions`` are additive in risk-score units: ``base_value`` is the
surrogate's mean prediction and each contribution is the feature's push away
from it. Every reason in :class:`ShapExplanation` cites exactly one feature and
the real computed value it is based on (narrative text without a number behind
it is not allowed to ship).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SurrogateFidelity(BaseModel):
    """How well the surrogate regressor reproduces the pipeline's risk scores."""

    model_config = ConfigDict(from_attributes=True)

    r2: float = Field(description="R2 of the surrogate regressor against pipeline risk scores.")
    mae: float = Field(
        ge=0.0, description="Mean absolute error in risk-score units (0-100 scale)."
    )
    n_training_points: int = Field(
        ge=0, description="Samples the surrogate was fitted on."
    )
    meets_minimum: bool = Field(
        description="True when r2 reaches xai.min_fidelity_r2 from configs/config.yaml."
    )


class ShapContribution(BaseModel):
    """One feature's contribution to the surrogate's risk prediction."""

    model_config = ConfigDict(from_attributes=True)

    feature: str = Field(description="Feature name in the surrogate's input space.")
    value: float = Field(description="The product's value for this feature.")
    contribution: float = Field(
        description="Signed contribution in risk-score units (positive raises risk)."
    )


class ShapReason(BaseModel):
    """A consumer-readable sentence that cites one real computed value."""

    model_config = ConfigDict(from_attributes=True)

    text: str = Field(
        description="Plain-language explanation; must quote the real number it cites."
    )
    feature: str = Field(description="Feature this reason is about.")
    raw_value: float | None = Field(
        default=None, description="The computed value the text quotes; None if not numeric."
    )


class ShapExplanation(BaseModel):
    """Surrogate-based explanation of one product's risk score."""

    model_config = ConfigDict(from_attributes=True)

    base_value: float = Field(
        description="Surrogate's mean prediction (risk-score units) before contributions."
    )
    contributions: list[ShapContribution] = Field(
        description="Per-feature contributions, largest absolute contribution first."
    )
    surrogate_fidelity: SurrogateFidelity
    reasons: list[ShapReason] = Field(
        description="Consumer-readable reasons, each citing a real computed number."
    )


__all__ = ["ShapContribution", "ShapExplanation", "ShapReason", "SurrogateFidelity"]

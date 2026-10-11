"""Pydantic v2 models for request and response models.

One module per API area, all re-exported here so callers import from a single
place:

* :mod:`~munasib.api.schemas.common` - shared bounds (unit interval, risk score,
  star rating, UTC datetimes), pagination, the error envelope, service meta.
* :mod:`~munasib.api.schemas.products` - product summary/detail and their page.
* :mod:`~munasib.api.schemas.reviews` - the weak-supervision sentiment vocabulary
  and a review snippet.
* :mod:`~munasib.api.schemas.review_analysis` - rating distribution, sentiment,
  temporal trend, aspects.
* :mod:`~munasib.api.schemas.evidence` - evidence indicators and strength.
* :mod:`~munasib.api.schemas.decision` - fuzzy assessment and TOPSIS result.
* :mod:`~munasib.api.schemas.explanation` - SHAP contributions and fidelity.
* :mod:`~munasib.api.schemas.alternatives` - alternatives with comparison deltas.
* :mod:`~munasib.api.schemas.verdict` - the consumer verdict and its concerns.
* :mod:`~munasib.api.schemas.analysis` - the composed analysis response.

Every model is Pydantic v2, uses ``from_attributes`` where it is built from SQL
rows, and carries field descriptions so the generated OpenAPI documentation is
readable. ``scripts/export_schemas.py`` exports the JSON Schemas of the two
top-level responses (analysis and error) to ``docs/schemas/``.
"""

from munasib.api.schemas.alternatives import (
    AlternativeProduct,
    AlternativeRelation,
    AlternativesResult,
    ComparisonDeltas,
)
from munasib.api.schemas.analysis import (
    ANALYSIS_EXAMPLE,
    PIPELINE_VERSION,
    AnalysisResponse,
    DataBasis,
)
from munasib.api.schemas.common import (
    CategoryMetaResponse,
    ErrorResponse,
    HealthResponse,
    Page,
    RiskScore,
    SplitSizes,
    StarRating,
    UnitFloat,
    UtcDatetime,
)
from munasib.api.schemas.decision import (
    CRITERION_NAMES,
    CriterionName,
    DecisionAssessment,
    FiredRule,
    FuzzyAssessment,
    MembershipProfile,
    TopsisCriterion,
    TopsisResult,
)
from munasib.api.schemas.evidence import (
    EvidenceIndicator,
    EvidenceIndicators,
    IndicatorDirection,
)
from munasib.api.schemas.explanation import (
    ShapContribution,
    ShapExplanation,
    ShapReason,
    SurrogateFidelity,
)
from munasib.api.schemas.products import ProductDetail, ProductPage, ProductSummary
from munasib.api.schemas.review_analysis import (
    AspectInsight,
    AspectPolarity,
    RatingDistribution,
    ReviewAnalysis,
    SentimentBreakdown,
    TemporalBucket,
    TemporalTrend,
)
from munasib.api.schemas.reviews import ReviewPage, ReviewSnippet, SentimentLabel
from munasib.api.schemas.verdict import (
    Concern,
    ConcernSeverity,
    ConsumerVerdict,
    VerdictLabel,
)

__all__ = [
    "ANALYSIS_EXAMPLE",
    "AlternativeProduct",
    "AlternativeRelation",
    "AlternativesResult",
    "AnalysisResponse",
    "AspectInsight",
    "AspectPolarity",
    "CRITERION_NAMES",
    "CategoryMetaResponse",
    "ComparisonDeltas",
    "Concern",
    "ConcernSeverity",
    "ConsumerVerdict",
    "CriterionName",
    "DataBasis",
    "DecisionAssessment",
    "ErrorResponse",
    "EvidenceIndicator",
    "EvidenceIndicators",
    "FiredRule",
    "FuzzyAssessment",
    "HealthResponse",
    "IndicatorDirection",
    "MembershipProfile",
    "PIPELINE_VERSION",
    "Page",
    "ProductDetail",
    "ProductPage",
    "ProductSummary",
    "RatingDistribution",
    "ReviewAnalysis",
    "ReviewPage",
    "ReviewSnippet",
    "RiskScore",
    "SentimentBreakdown",
    "SentimentLabel",
    "ShapContribution",
    "ShapExplanation",
    "ShapReason",
    "SplitSizes",
    "StarRating",
    "SurrogateFidelity",
    "TemporalBucket",
    "TemporalTrend",
    "TopsisCriterion",
    "TopsisResult",
    "UnitFloat",
    "UtcDatetime",
    "VerdictLabel",
]

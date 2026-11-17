"""Per-product evidence indicators and the composite evidence strength.

The public entry points are intentionally small: the CLI script orchestrates the
stage, these modules hold the tested logic.

* :mod:`~munasib.review_intel.evidence.indicators` - pure indicator arithmetic
  (no database, no clock; every formula is in ``docs/evidence.md``).
* :mod:`~munasib.review_intel.evidence.compute` - assembly of one row per
  product plus the run summary and the ``reports/`` rendering.
* :mod:`~munasib.review_intel.evidence.store` - serving-cache reads, the
  dataset reference date and persistence.

The three-indicator composite served live by the API (``review_intel.evidence``
``.weights``) is a separate, smaller contract; the stored six-component
composite in this package is documented and compared in ``docs/evidence.md``.
"""

from munasib.review_intel.evidence.compute import (
    JSON_FILENAME,
    MARKDOWN_FILENAME,
    SUMMARY_FLAGS,
    EvidenceRow,
    EvidenceSummary,
    compute_evidence,
    render_summary_markdown,
    summarize,
    write_summary_reports,
)
from munasib.review_intel.evidence.indicators import (
    COMPONENT_INDICATOR,
    DIGITS,
    DIRECTION_BENEFIT,
    DIRECTION_COST,
    STRENGTH_COMPONENT_KEYS,
    Indicator,
    StrengthResult,
    burst_indicator,
    composite_strength,
    concentration_indicators,
    helpfulness_indicators,
    normalized_entropy,
    rating_agreement_indicator,
    rating_text_agreement_indicator,
    recency_indicators,
    review_volume_sufficiency,
    sentiment_shares_indicator,
    star_counts,
    verified_purchase_indicator,
    volume_indicator,
)
from munasib.review_intel.evidence.store import (
    count_evidence,
    load_aspects,
    load_products,
    load_reviews,
    read_evidence,
    read_reference_date,
    write_evidence,
)

__all__ = [
    "COMPONENT_INDICATOR",
    "DIGITS",
    "DIRECTION_BENEFIT",
    "DIRECTION_COST",
    "JSON_FILENAME",
    "MARKDOWN_FILENAME",
    "STRENGTH_COMPONENT_KEYS",
    "SUMMARY_FLAGS",
    "EvidenceRow",
    "EvidenceSummary",
    "Indicator",
    "StrengthResult",
    "burst_indicator",
    "composite_strength",
    "compute_evidence",
    "concentration_indicators",
    "count_evidence",
    "helpfulness_indicators",
    "load_aspects",
    "load_products",
    "load_reviews",
    "normalized_entropy",
    "rating_agreement_indicator",
    "rating_text_agreement_indicator",
    "read_evidence",
    "read_reference_date",
    "recency_indicators",
    "render_summary_markdown",
    "review_volume_sufficiency",
    "sentiment_shares_indicator",
    "star_counts",
    "summarize",
    "verified_purchase_indicator",
    "volume_indicator",
    "write_evidence",
    "write_summary_reports",
]

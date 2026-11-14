"""Aspect extraction, vocabulary, per-product aggregation and audit helpers.

The public entry points are intentionally small: the CLI scripts orchestrate the
stages, while these modules hold the tested logic.

* :mod:`~munasib.review_intel.aspects.extraction` - spaCy noun-chunk extraction.
* :mod:`~munasib.review_intel.aspects.vocabulary` - category vocabulary.
* :mod:`~munasib.review_intel.aspects.aggregation` - per-product statistics.
* :mod:`~munasib.review_intel.aspects.cache` - resumable candidate parts.
* :mod:`~munasib.review_intel.aspects.storage` - database reads/parquet writes.
"""

from munasib.review_intel.aspects.aggregation import (
    AspectPredictionRow,
    aggregate_product,
    classify_aspect,
    negativity_lift,
)
from munasib.review_intel.aspects.extraction import (
    canonical_aspect,
    extract_doc_aspects,
    filter_aspect_phrase,
    load_nlp,
)
from munasib.review_intel.aspects.types import (
    ASPECT_CLASSIFICATIONS,
    ASPECT_NEUTRAL,
    ASPECT_PAIN_POINT,
    ASPECT_STRENGTH,
    AspectExample,
    AspectLabelCounts,
    ProductAspectStats,
    VocabularyEntry,
)
from munasib.review_intel.aspects.vocabulary import (
    build_vocabulary,
    load_vocabulary,
    vocabulary_preview,
)

__all__ = [
    "ASPECT_CLASSIFICATIONS",
    "ASPECT_NEUTRAL",
    "ASPECT_PAIN_POINT",
    "ASPECT_STRENGTH",
    "AspectExample",
    "AspectLabelCounts",
    "AspectPredictionRow",
    "ProductAspectStats",
    "VocabularyEntry",
    "aggregate_product",
    "build_vocabulary",
    "canonical_aspect",
    "classify_aspect",
    "extract_doc_aspects",
    "filter_aspect_phrase",
    "load_nlp",
    "load_vocabulary",
    "negativity_lift",
    "vocabulary_preview",
]

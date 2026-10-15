"""``GET /products/{asin}/analysis`` - the composed analysis for one product.

The route is a thin pass-through to the configured
:class:`~munasib.pipeline.service.AnalysisService`. While that service is the
stub, the response carries ``pipeline_version="stub-0"`` and a ``warnings``
entry beginning with ``"STUB:"``; swapping in the real engine needs no change
here.
"""

from __future__ import annotations

from fastapi import APIRouter

from munasib.api.deps import AnalysisServiceDep
from munasib.api.schemas.analysis import AnalysisResponse

router = APIRouter(tags=["analysis"])


@router.get("/products/{asin}/analysis", response_model=AnalysisResponse)
def get_analysis(service: AnalysisServiceDep, asin: str) -> AnalysisResponse:
    """Return the full analysis of one product.

    Args:
        service: The configured analysis implementation.
        asin: Product ASIN.

    Returns:
        The composed analysis: product profile, review analysis, evidence, the
        fuzzy + TOPSIS decision, the explanation, alternatives and verdict.

    Raises:
        ProductNotFoundError: If the ASIN is not in the serving cache (``404``).
        ModelError: If ``analysis.service`` is ``real`` and the engine is not
            implemented (``503``).
    """
    return service.analyze(asin)

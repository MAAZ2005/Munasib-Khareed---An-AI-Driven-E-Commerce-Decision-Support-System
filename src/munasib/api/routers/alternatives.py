"""``GET /products/{asin}/alternatives`` - evidence-based alternatives.

``total_related`` counts every related edge of the product (an ASIN linked by
both ``also_bought`` and ``also_viewed`` counts twice) while ``usable`` counts
the distinct alternatives that are themselves processed subset products - the
two numbers are deliberately different, which is why a product can report 62
edges and 11 usable alternatives. A product with nothing comparable is a valid
answer: ``items`` is empty and ``empty_reason`` says why (about 18% of this
category's products).
"""

from __future__ import annotations

from fastapi import APIRouter

from munasib.api.deps import AnalysisServiceDep
from munasib.api.schemas.alternatives import AlternativesResult

router = APIRouter(tags=["analysis"])


@router.get("/products/{asin}/alternatives", response_model=AlternativesResult)
def get_alternatives(service: AnalysisServiceDep, asin: str) -> AlternativesResult:
    """Return the usable alternatives for one product.

    Args:
        service: The configured analysis implementation.
        asin: Product ASIN.

    Returns:
        The alternatives with their deltas versus the subject, or an empty
        result with an explicit ``empty_reason``.

    Raises:
        ProductNotFoundError: If the ASIN is not in the serving cache (``404``).
        ModelError: If ``analysis.service`` is ``real`` and the engine is not
            implemented (``503``).
    """
    return service.alternatives(asin)

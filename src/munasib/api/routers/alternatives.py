"""``GET /products/{asin}/alternatives`` - evidence-based alternatives.

``total_related`` counts DISTINCT related products of the product, merged across
``also_bought`` and ``also_viewed`` and excluding the product itself, while
``usable`` counts how many of those distinct products are themselves processed
subset products - so ``usable`` is never greater than ``total_related``. The two
numbers differ because links can fall outside the subset: B00009V3UF has 62
edges collapsing to 60 distinct related products, 11 of them in the subset. A
product with nothing comparable is a valid answer: ``items`` is empty and
``empty_reason`` says why (about 18% of this category's products).
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

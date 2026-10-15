"""Route controllers for every Munasib Khareed endpoint.

One module per area so a router can be read without the rest of the surface:

* :mod:`~munasib.api.routers.health` - ``GET /health``.
* :mod:`~munasib.api.routers.meta` - ``GET /meta/category``.
* :mod:`~munasib.api.routers.products` - ``GET /products`` and ``GET /products/{asin}``.
* :mod:`~munasib.api.routers.reviews` - ``GET /products/{asin}/reviews``.
* :mod:`~munasib.api.routers.analysis` - ``GET /products/{asin}/analysis``.
* :mod:`~munasib.api.routers.alternatives` - ``GET /products/{asin}/alternatives``.

Every route declares ``response_model`` from :mod:`munasib.api.schemas`, so an
out-of-bounds or shape-invalid payload fails at the boundary instead of reaching
the dashboard.
"""

from munasib.api.routers.alternatives import router as alternatives_router
from munasib.api.routers.analysis import router as analysis_router
from munasib.api.routers.health import router as health_router
from munasib.api.routers.meta import router as meta_router
from munasib.api.routers.products import router as product_router
from munasib.api.routers.reviews import router as review_router

__all__ = [
    "alternatives_router",
    "analysis_router",
    "health_router",
    "meta_router",
    "product_router",
    "review_router",
]

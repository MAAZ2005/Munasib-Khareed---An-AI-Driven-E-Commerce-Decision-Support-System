"""Product endpoints: search, the full profile of one ASIN, and its cached thumbnail.

Paging bounds are enforced by the query annotations, so ``limit=0``,
``limit=1000``, ``offset=-1`` and an unknown ``split`` all come back as ``422``
with the standard error envelope before a repository is ever touched.

``GET /products/{asin}/image`` streams the locally cached thumbnail (images are
display-only and never hotlinked): the ASIN is validated against the shared
:data:`~munasib.core.constants.ASIN_PATTERN`, the resolved file is confirmed to
sit inside the cache directory, and a product with no cached image is a plain
``404`` - never a redirect to the upstream URL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
from pydantic import AfterValidator

from munasib.api.deps import ProductRepoDep, SettingsDep
from munasib.api.schemas.common import ErrorResponse
from munasib.api.schemas.products import ProductDetail, ProductPage, ProductSummary
from munasib.core.constants import ASIN_PATTERN, SPLIT_ORDER
from munasib.core.exceptions import ProductNotFoundError

router = APIRouter(tags=["products"])

#: Filename suffix of the cached thumbnails written by ``scripts/download_images.py``.
IMAGE_FILENAME_SUFFIX = ".jpg"

#: Default page size, shared with the repositories so the two cannot drift.
DEFAULT_LIMIT = 20

#: Largest page the serving cache will hand out in one request.
MAX_LIMIT = 100


def _validate_split(value: str | None) -> str | None:
    """Reject a split outside the shared split vocabulary.

    Args:
        value: Candidate split name or None (no filter).

    Returns:
        The unchanged value.

    Raises:
        ValueError: If the split is not one of :data:`SPLIT_ORDER`.
    """
    if value is not None and value not in SPLIT_ORDER:
        raise ValueError(f"unknown split {value!r}; expected one of {list(SPLIT_ORDER)}")
    return value


#: Optional split filter validated against the shared constants.
SplitQuery = Annotated[str | None, AfterValidator(_validate_split), Query()]

#: Page size, bounded so a client cannot ask for the whole category at once.
LimitQuery = Annotated[int, Query(ge=1, le=MAX_LIMIT)]

#: Rows to skip; negative offsets are rejected.
OffsetQuery = Annotated[int, Query(ge=0)]


@router.get("/products", response_model=ProductPage)
def search_products(
    repo: ProductRepoDep,
    query: str = "",
    limit: LimitQuery = DEFAULT_LIMIT,
    offset: OffsetQuery = 0,
    split: SplitQuery = None,
) -> ProductPage:
    """Search products by title or brand, case-insensitively.

    Args:
        repo: Product repository bound to the request session.
        query: Literal text matched against title or brand; empty matches all.
        limit: Page size (1-100).
        offset: Rows to skip before the page.
        split: Optional ``train``/``val``/``test`` filter.

    Returns:
        The page of matching products plus the total match count. Ordering is by
        ASIN, so paging with ``limit``/``offset`` is stable.
    """
    page = repo.search_products(query=query, limit=limit, offset=offset, split=split)
    return ProductPage(
        total=page.total,
        limit=page.limit,
        offset=page.offset,
        items=[ProductSummary.model_validate(row) for row in page.items],
    )


@router.get("/products/{asin}", response_model=ProductDetail)
def get_product(repo: ProductRepoDep, asin: str) -> ProductDetail:
    """Return the full profile of one product.

    Args:
        repo: Product repository bound to the request session.
        asin: Product ASIN.

    Returns:
        Every stored field of the product, with missing price/rank/brand kept
        as ``null`` and flagged by the ``*_missing`` booleans.

    Raises:
        ProductNotFoundError: If the ASIN is not in the serving cache (answered
            as ``404`` with the standard error envelope).
    """
    return ProductDetail.model_validate(repo.get_product(asin))


@router.api_route(
    "/products/{asin}/image",
    # HEAD is registered explicitly: FastAPI does not derive it from GET, and
    # Starlette's FileResponse already answers a HEAD with headers only.
    methods=["GET", "HEAD"],
    response_class=FileResponse,
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "The cached JPEG thumbnail of this product.",
        },
        404: {
            "model": ErrorResponse,
            "description": "No cached thumbnail exists for this ASIN.",
        },
    },
)
def get_product_image(
    repo: ProductRepoDep, settings: SettingsDep, asin: str
) -> FileResponse:
    """Serve the locally cached product thumbnail.

    Offline by design: the file is read from ``images.directory`` (filled once by
    ``scripts/download_images.py``); the upstream image URL is never revealed and
    no request ever leaves this server.

    Args:
        repo: Product repository bound to the request session.
        settings: Project settings (cache directory and ``Cache-Control`` lifetime).
        asin: Product ASIN.

    Returns:
        The cached ``image/jpeg`` thumbnail with a long-lived ``Cache-Control``
        header (cached files are immutable for a given ASIN).

    Raises:
        ProductNotFoundError: When the ASIN is malformed or unknown, when the
            product has no cached image, or when the cache file is missing
            (answered as ``404`` with the standard error envelope).
    """
    if not ASIN_PATTERN.match(asin):
        raise ProductNotFoundError(asin, f"'{asin}' is not a valid ASIN.")
    product = repo.get_product(asin)
    if not product.image_cached:
        raise ProductNotFoundError(
            asin, f"No cached image exists for product '{asin}'."
        )
    cache_dir = Path(settings.images.directory)
    resolved = (cache_dir / f"{asin}{IMAGE_FILENAME_SUFFIX}").resolve()
    # ASIN_PATTERN already forbids separators; this keeps the guarantee explicit.
    if resolved.parent != cache_dir.resolve() or not resolved.is_file():
        raise ProductNotFoundError(
            asin, f"The cached image file for product '{asin}' is missing."
        )
    return FileResponse(
        resolved,
        media_type="image/jpeg",
        headers={
            "Cache-Control": f"public, max-age={settings.images.cache_max_age_seconds}"
        },
    )

"""``GET /meta/category`` - what the serving cache actually contains.

Every count is read from the SQLite cache, so the endpoint reports the data the
API is really serving rather than a number copied from a report. The only value
that does **not** come from the cache is ``reference_date``: it is read from
``data/processed/meta.json``, which records the dataset-wide latest review date
that every recency feature is anchored on. Nothing here uses today's date.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter
from sqlalchemy import func, select

from munasib.api.deps import SessionDep, SettingsDep
from munasib.api.schemas.common import CategoryMetaResponse, SplitSizes
from munasib.core.constants import SPLIT_ORDER
from munasib.db.models import Product, RelatedProduct, Review
from munasib.pipeline.service import load_dataset_meta

router = APIRouter(tags=["service"])


@router.get("/meta/category", response_model=CategoryMetaResponse)
def category_meta(session: SessionDep, settings: SettingsDep) -> CategoryMetaResponse:
    """Describe the analysed category as stored in the serving cache.

    Args:
        session: Request-scoped session over the serving cache.
        settings: Project settings (``paths.processed_data_dir``).

    Returns:
        Category name, product/review/edge counts, split sizes, the real review
        date range and the dataset ``reference_date``.
    """
    n_products = session.scalar(select(func.count()).select_from(Product)) or 0
    n_reviews = session.scalar(select(func.count()).select_from(Review)) or 0
    n_edges = session.scalar(select(func.count()).select_from(RelatedProduct)) or 0
    n_edges_in_subset = session.scalar(
        select(func.count()).select_from(RelatedProduct).where(RelatedProduct.in_subset.is_(True))
    ) or 0

    earliest = session.scalar(select(func.min(Review.review_time)))
    latest = session.scalar(select(func.max(Review.review_time)))

    split_sizes: dict[str, SplitSizes] = {}
    for split in SPLIT_ORDER:
        products = session.scalar(
            select(func.count()).select_from(Product).where(Product.split == split)
        ) or 0
        reviews = session.scalar(
            select(func.count()).select_from(Review).where(Review.split == split)
        ) or 0
        split_sizes[split] = SplitSizes(products=int(products), reviews=int(reviews))

    return CategoryMetaResponse(
        category=load_dataset_meta(settings).category,
        n_products=int(n_products),
        n_reviews=int(n_reviews),
        n_related_edges=int(n_edges),
        n_related_edges_in_subset=int(n_edges_in_subset),
        date_range_start=_naive_to_utc(earliest),
        date_range_end=_naive_to_utc(latest),
        reference_date=load_dataset_meta(settings).reference_date,
        split_sizes=split_sizes,
    )


def _naive_to_utc(value: datetime | None) -> datetime:
    """Interpret a naive SQLite timestamp as UTC.

    Args:
        value: Timestamp as stored (naive) or None.

    Returns:
        An aware UTC datetime; the epoch is substituted only when the cache
        holds no review at all, which cannot happen for a built subset.
    """
    if value is None:
        return datetime(1970, 1, 1, tzinfo=UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)

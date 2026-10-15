"""Review endpoints for one product.

``label`` accepts the API sentiment names (``negative``, ``neutral``,
``positive``) and maps them onto the shared integer codes from
:mod:`munasib.core.constants` before they reach the repository - the codes are
never exposed. An unknown label is rejected as ``422`` by the enum itself.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from munasib.api.deps import ProductRepoDep, ReviewRepoDep
from munasib.api.schemas.reviews import (
    SENTIMENT_LABEL_CODE_FOR_NAME,
    ReviewPage,
    ReviewSnippet,
    SentimentLabel,
)
from munasib.db.models import Review

router = APIRouter(tags=["reviews"])

#: Weak-label filter; the enum is what turns a bad value into a ``422``.
LabelQuery = Annotated[SentimentLabel | None, Query()]

#: Tri-state verified filter: ``true``/``false`` filter, absence means "no filter".
VerifiedQuery = Annotated[bool | None, Query()]


def _to_snippet(row: Review) -> ReviewSnippet:
    """Map one review row onto the API snippet.

    ``review_text`` is the stored raw body kept for display and ``summary`` is
    the cleaned headline; the weak-label code is converted to its API name.

    Args:
        row: A ``reviews`` ORM row.

    Returns:
        The review snippet.
    """
    return ReviewSnippet(
        review_id=row.review_id,
        asin=row.asin,
        reviewer_id=row.reviewer_id,
        rating=row.rating,
        verified=row.verified,
        helpful_votes=row.helpful_votes,
        review_time=row.review_time,
        review_text=row.review_text_raw,
        has_text=row.has_text,
        text_length=row.text_length,
        summary=row.summary_clean,
        sentiment_label=SentimentLabel.from_code(row.sentiment_label_weak),
    )


@router.get("/products/{asin}/reviews", response_model=ReviewPage)
def get_reviews(
    products: ProductRepoDep,
    repo: ReviewRepoDep,
    asin: str,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    label: LabelQuery = None,
    verified: VerifiedQuery = None,
) -> ReviewPage:
    """Return the reviews of one product, newest first.

    Args:
        products: Product repository, used to answer 404 for an unknown ASIN.
        repo: Review repository bound to the request session.
        asin: Product ASIN.
        limit: Page size (1-100).
        offset: Rows to skip.
        label: Optional weak-label filter (``negative``/``neutral``/``positive``).
        verified: Optional verified-purchase filter; ``false`` excludes rows
            whose source did not say, because NULL is not False.

    Returns:
        The page plus the total matching count, ordered by ``review_time``
        descending then review id, so paging is stable.

    Raises:
        ProductNotFoundError: If the ASIN is not in the serving cache.
    """
    products.get_product(asin)
    code = None if label is None else SENTIMENT_LABEL_CODE_FOR_NAME[label]
    page = repo.get_reviews(
        asin=asin, limit=limit, offset=offset, label=code, verified=verified
    )
    return ReviewPage(
        total=page.total,
        limit=page.limit,
        offset=page.offset,
        items=[_to_snippet(row) for row in page.items],
    )

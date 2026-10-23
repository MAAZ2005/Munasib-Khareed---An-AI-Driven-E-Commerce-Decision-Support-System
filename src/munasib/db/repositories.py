"""Typed repository classes over the SQLite serving cache.

Each repository wraps one table family and owns its query contract (filters,
ordering, paging) so API handlers never hand-write SQL. Conventions:

* Ordering is always **stable** (a unique tiebreaker follows any user filter)
  so pagination cannot skip or repeat rows.
* ``Page.total`` is the unpaginated count for the same filter, so callers can
  render "showing X of Y" without a second query.
* Missing products raise :class:`~munasib.core.exceptions.ProductNotFoundError`
  (domain error), while empty collections are returned as empty lists.
* Split and label arguments are validated against the shared constants in
  :mod:`munasib.core.constants` - no bare ``"train"`` or ``0/1/2`` literals.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Generic, TypeVar

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import Session

from munasib.core.constants import (
    SENTIMENT_LABELS,
    SPLIT_ORDER,
)
from munasib.core.exceptions import ProductNotFoundError
from munasib.db.models import (
    RELATION_ALSO_BOUGHT,
    RELATION_ALSO_VIEWED,
    AnalysisResult,
    Product,
    RelatedProduct,
    Review,
)

T = TypeVar("T")

#: Default page size for product/review listings (serving-cache sized).
DEFAULT_PAGE_SIZE = 20


@dataclass(frozen=True)
class Page(Generic[T]):
    """One page of results plus the total count matching the same filter."""

    items: list[T]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True)
class RelatedCounts:
    """Distinct related-product counts for one subject product.

    Every count is per distinct **related product**, never per edge, matching the
    alternatives contract in :mod:`munasib.pipeline.service`: a target linked by
    both ``also_bought`` and ``also_viewed`` counts once in each of those two
    fields, a product never counts itself (self-referencing edges are ignored),
    and ``usable_alternatives`` counts only targets that are themselves in the
    processed subset (so it can never exceed the other two).
    """

    also_bought: int = 0
    also_viewed: int = 0
    usable_alternatives: int = 0


def _validate_split(split: str | None) -> None:
    """Raise ``ValueError`` for a split outside the shared split vocabulary.

    Args:
        split: Candidate split name or None (None = no filter).

    Raises:
        ValueError: If ``split`` is not one of ``train``/``val``/``test``.
    """
    if split is not None and split not in SPLIT_ORDER:
        raise ValueError(f"unknown split {split!r}; expected one of {list(SPLIT_ORDER)}")


def _like_pattern(query: str) -> str:
    """Escape a user query for a case-insensitive literal ``LIKE`` match.

    ``%``, ``_`` and the escape character itself are escaped so a query like
    ``"50%"`` matches literally instead of acting as a wildcard.

    Args:
        query: Raw user query.

    Returns:
        Pattern with ``%...%`` around the escaped query.
    """
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class ProductRepository:
    """Product lookups: point get, text search, peer sampling, candidates."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_product(self, asin: str) -> Product:
        """Fetch one product by ASIN.

        Args:
            asin: Product ASIN.

        Returns:
            The product row.

        Raises:
            ProductNotFoundError: If no product with this ASIN exists.
        """
        product = self._session.get(Product, asin)
        if product is None:
            raise ProductNotFoundError(
                asin, f"product {asin!r} not found in the serving cache"
            )
        return product

    def search_products(
        self,
        query: str,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        split: str | None = None,
    ) -> Page[Product]:
        """Case-insensitive title/brand search with pagination and total count.

        Ordering is by ``asin`` (unique, so stable across pages).

        Args:
            query: Text matched against title or brand; empty query matches all.
            limit: Page size (maximum rows to return).
            offset: Rows to skip.
            split: Optional split filter (``train``/``val``/``test``).

        Returns:
            Page with the matching products and the total match count.

        Raises:
            ValueError: If ``split`` is outside the split vocabulary.
        """
        _validate_split(split)
        if limit < 0 or offset < 0:
            raise ValueError("limit and offset must be non-negative")

        pattern = _like_pattern(query)
        conditions = [
            or_(
                func.lower(Product.title).like(pattern, escape="\\"),
                func.lower(Product.brand).like(pattern, escape="\\"),
            )
        ]
        if split is not None:
            conditions.append(Product.split == split)

        total = self._session.scalar(
            select(func.count()).select_from(Product).where(*conditions)
        )
        assert total is not None  # COUNT(*) over a filtered FROM is never NULL
        rows = list(
            self._session.scalars(
                select(Product).where(*conditions).order_by(Product.asin).limit(limit).offset(offset)
            )
        )
        return Page(items=rows, total=int(total), limit=limit, offset=offset)

    def list_candidate_asins(self, split: str | None = None) -> list[str]:
        """List every candidate ASIN ordered by ASIN (stable).

        Args:
            split: Optional split filter (``train``/``val``/``test``).

        Returns:
            Sorted candidate ASINs.

        Raises:
            ValueError: If ``split`` is outside the split vocabulary.
        """
        _validate_split(split)
        statement = select(Product.asin).order_by(Product.asin)
        if split is not None:
            statement = statement.where(Product.split == split)
        return list(self._session.scalars(statement))

    def list_all_products(self) -> list[Product]:
        """Fetch every product row, ordered by ASIN (stable).

        Used by offline batch jobs (``scripts/compute_product_features.py``)
        that need the whole candidate set in memory to build peer
        distributions; the paged API paths keep using
        :meth:`search_products` instead of loading everything.

        Returns:
            Every stored product, ascending by ASIN (empty list when the cache
            has no products).
        """
        return list(self._session.scalars(select(Product).order_by(Product.asin)))

    def get_products(self, asins: Sequence[str]) -> list[Product]:
        """Fetch several products by ASIN, ordered by ASIN (stable).

        Unlike :meth:`get_product` this never raises for an unknown ASIN - the
        caller decides whether a missing row is an error (batch jobs compare
        the requested set against the returned one).

        Args:
            asins: ASINs to fetch (duplicates collapse to one row each).

        Returns:
            The products that exist, ascending by ASIN.
        """
        return self._fetch_by_asins(sorted(set(asins)))

    def list_peer_products(self, asin: str, n: int, seed: int) -> list[Product]:
        """Deterministic sample of ``n`` other products from the same category.

        Candidates are every product except ``asin``, sorted by ``asin`` (so
        the result does not depend on insertion or query order), then shuffled
        with ``random.Random(seed)``. The same ``(asin, n, seed)`` therefore
        always yields the same sample; results are returned in ``asin`` order.

        Args:
            asin: Product to exclude (the subject of the analysis).
            n: Maximum number of peers.
            seed: Seed from config (``project.random_seed`` or split seed).

        Returns:
            Sampled peer products ordered by ASIN.

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        candidate_asins = list(
            self._session.scalars(
                select(Product.asin).where(Product.asin != asin).order_by(Product.asin)
            )
        )
        if n == 0 or not candidate_asins:
            return []
        sampled = random.Random(seed).sample(candidate_asins, min(n, len(candidate_asins)))
        return self._fetch_by_asins(sampled)

    def _fetch_by_asins(self, asins: list[str]) -> list[Product]:
        """Fetch products for ``asins`` in a deterministic order.

        Args:
            asins: ASINs to fetch (duplicates collapse to one row each).

        Returns:
            Products ordered by ASIN.
        """
        rows = self._session.scalars(
            select(Product).where(Product.asin.in_(asins)).order_by(Product.asin)
        )
        return list(rows)


class ReviewRepository:
    """Review lookups for a product (pagination, label/verified filters)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_reviews(
        self,
        asin: str,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        label: int | None = None,
        verified: bool | None = None,
    ) -> Page[Review]:
        """Fetch reviews for one product, newest first.

        Ordering is ``review_time DESC, id ASC`` (unique tiebreaker), so
        pagination is stable even for same-timestamp bursts.

        ``verified`` is tri-state: ``True``/``False`` filter to that value and
        NULL rows (unknown) match **neither** - filtering ``verified=False``
        does not silently include unknown rows.

        Args:
            asin: Product ASIN.
            limit: Page size.
            offset: Rows to skip.
            label: Optional weak-label filter using the shared codes from
                :mod:`munasib.core.constants`.
            verified: Optional verified-purchase filter (None = no filter).

        Returns:
            Page with matching reviews and the total count.

        Raises:
            ValueError: If ``label`` is not a known weak label, or paging is
                negative.
        """
        if label is not None and label not in SENTIMENT_LABELS:
            raise ValueError(
                f"unknown sentiment label {label!r}; expected one of {list(SENTIMENT_LABELS)}"
            )
        if limit < 0 or offset < 0:
            raise ValueError("limit and offset must be non-negative")

        conditions = [Review.asin == asin]
        if label is not None:
            conditions.append(Review.sentiment_label_weak == label)
        if verified is not None:
            conditions.append(Review.verified.is_(True) if verified else Review.verified.is_(False))

        total = self._session.scalar(
            select(func.count()).select_from(Review).where(*conditions)
        )
        assert total is not None
        rows = list(
            self._session.scalars(
                select(Review)
                .where(*conditions)
                .order_by(Review.review_time.desc(), Review.id.asc())
                .limit(limit)
                .offset(offset)
            )
        )
        return Page(items=rows, total=int(total), limit=limit, offset=offset)

    def get_review_count(self, asin: str) -> int:
        """Count all reviews for one product (no filters).

        Args:
            asin: Product ASIN.

        Returns:
            Number of reviews stored for the product.
        """
        count = self._session.scalar(
            select(func.count()).select_from(Review).where(Review.asin == asin)
        )
        assert count is not None
        return int(count)


class RelatedRepository:
    """Related-product edge lookups."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_related(self, asin: str, only_in_subset: bool = True) -> list[RelatedProduct]:
        """Fetch related-product edges for one product.

        Args:
            asin: Subject product ASIN.
            only_in_subset: When True (default), keep only edges whose target
                is itself a processed candidate product (``in_subset=True``).

        Returns:
            Edges ordered by ``(related_asin, relation_type)`` (stable).
        """
        conditions = [RelatedProduct.asin == asin]
        if only_in_subset:
            conditions.append(RelatedProduct.in_subset.is_(True))
        rows = self._session.scalars(
            select(RelatedProduct)
            .where(*conditions)
            .order_by(RelatedProduct.related_asin, RelatedProduct.relation_type)
        )
        return list(rows)

    def count_related(self, asins: Sequence[str] | None = None) -> dict[str, RelatedCounts]:
        """Aggregate the distinct related-product counts for many products.

        One grouped query replaces a per-product edge scan, so profiling the
        whole candidate set stays a single pass. The three counts follow the
        :class:`RelatedCounts` contract: per distinct target product, never per
        edge, self-references excluded, and ``usable_alternatives`` restricted
        to in-subset targets.

        Args:
            asins: Subject ASINs to restrict the aggregation to, or None for
                every product that has at least one edge.

        Returns:
            Mapping ``asin -> RelatedCounts``; a product with no edges is
            absent from the mapping (callers treat that as the zero counts,
            never as an error). Ordered by ASIN.
        """
        bought = case(
            (
                and_(
                    RelatedProduct.relation_type == RELATION_ALSO_BOUGHT,
                    RelatedProduct.related_asin != RelatedProduct.asin,
                ),
                RelatedProduct.related_asin,
            ),
            else_=None,
        )
        viewed = case(
            (
                and_(
                    RelatedProduct.relation_type == RELATION_ALSO_VIEWED,
                    RelatedProduct.related_asin != RelatedProduct.asin,
                ),
                RelatedProduct.related_asin,
            ),
            else_=None,
        )
        usable = case(
            (
                and_(
                    RelatedProduct.in_subset.is_(True),
                    RelatedProduct.related_asin != RelatedProduct.asin,
                ),
                RelatedProduct.related_asin,
            ),
            else_=None,
        )
        statement = (
            select(
                RelatedProduct.asin.label("asin"),
                func.count(func.distinct(bought)).label("also_bought"),
                func.count(func.distinct(viewed)).label("also_viewed"),
                func.count(func.distinct(usable)).label("usable"),
            )
            .group_by(RelatedProduct.asin)
            .order_by(RelatedProduct.asin)
        )
        if asins is not None:
            statement = statement.where(RelatedProduct.asin.in_(asins))
        return {
            str(asin): RelatedCounts(
                also_bought=int(also_bought),
                also_viewed=int(also_viewed),
                usable_alternatives=int(usable_count),
            )
            for asin, also_bought, also_viewed, usable_count in self._session.execute(statement)
        }


class AnalysisRepository:
    """Persisted pipeline outputs keyed by (asin, pipeline_version)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_analysis(self, asin: str, pipeline_version: str) -> AnalysisResult | None:
        """Fetch one analysis result if it exists.

        Args:
            asin: Product ASIN.
            pipeline_version: Pipeline version string that produced the row.

        Returns:
            The stored row, or None when this (product, version) was never run.
        """
        return self._session.get(AnalysisResult, (asin, pipeline_version))

    def upsert_analysis(
        self,
        asin: str,
        pipeline_version: str,
        risk_score: float | None,
        evidence_strength: float | None,
        verdict: str | None,
        payload: dict | None,
        created_at: datetime | None = None,
    ) -> AnalysisResult:
        """Insert or update one analysis result (idempotent per key).

        The composite primary key ``(asin, pipeline_version)`` means re-running
        the pipeline overwrites the stored row instead of duplicating it, and
        ``created_at`` is refreshed to record when the row was last written
        (UTC; defaults to now).

        Args:
            asin: Product ASIN (must already exist in ``products``).
            pipeline_version: Pipeline version that produced the result.
            risk_score: Fuzzy+TOPSIS consumer risk (None until computed).
            evidence_strength: Evidence-strength indicator (None until computed).
            verdict: Verdict label from :mod:`munasib.core.constants`.
            payload: Full structured explanation blob (SHAP, aspects, ...).
            created_at: UTC timestamp to store; now() when omitted.

        Returns:
            The persisted row (re-read after the upsert).

        Raises:
            IntegrityError: If the product does not exist (foreign key) - the
                caller's transaction is rolled back by ``session_scope``.
        """
        timestamp = created_at or datetime.now(UTC)
        if timestamp.tzinfo is not None:
            # Contract: every stored datetime is naive UTC (see db.models).
            timestamp = timestamp.astimezone(UTC).replace(tzinfo=None)
        # merge() = SELECT then INSERT/UPDATE on the composite key; an integrity
        # failure (e.g. unknown product) propagates and session_scope rolls the
        # whole transaction back.
        self._session.merge(
            AnalysisResult(
                asin=asin,
                pipeline_version=pipeline_version,
                risk_score=risk_score,
                evidence_strength=evidence_strength,
                verdict=verdict,
                payload=payload,
                created_at=timestamp,
            )
        )
        self._session.flush()
        stored = self._session.get(AnalysisResult, (asin, pipeline_version))
        assert stored is not None  # just flushed above
        return stored


__all__ = [
    "AnalysisRepository",
    "Page",
    "ProductRepository",
    "RelatedCounts",
    "RelatedRepository",
    "ReviewRepository",
]

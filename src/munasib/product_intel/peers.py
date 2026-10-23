"""Peer distributions that make product-level percentiles cheap and exact.

A percentile is only meaningful against a reference set, so this module builds
that reference **once** per run (:meth:`PeerStats.from_products`) and keeps it
as sorted arrays: ``bisect_right`` then answers every lookup in ``O(log n)``
instead of re-scanning the category per product. Nothing here queries on its
own - the caller decides where the rows come from (the serving cache in
``scripts/compute_product_features.py``, plain lists in tests).

**Tie rule (used by every percentile below).** A percentile is the fraction of
*other* products in the distribution whose value is ``<=`` the subject's value::

    percentile = (# peers with value <= subject value - 1) / (# peers - 1)

The ``- 1`` removes the subject's own row from both sides, because a product is
never its own peer. Ties therefore resolve *towards* the subject: the cheapest
of three equally cheap products scores ``1/2``, not ``0``. The rule is identical
for price and rank, so in both cases **0 is the favourable end** (cheapest /
best-selling) and 1 the unfavourable one - exactly the orientation
``price_position`` expects in ``configs/criteria_weights.yaml``.

**Missing data is never imputed.** A product without a price simply is not in
the price distribution and its price percentiles are ``None``; a product whose
brand has fewer than ``product_intel.min_brand_peers`` priced peers gets ``None``
for the brand percentile, and the same holds for a rank category with fewer than
``product_intel.min_rank_peers`` ranked peers. Every ``None`` is paired with a
``*_missing`` flag on the profile (see :mod:`munasib.product_intel.profile`).
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from munasib.core.config import ProductIntelConfig
from munasib.data.cleaning import normalize_brand as _normalize_raw_brand


class ProductLike(Protocol):
    """Product columns the profiler and the peer stats read.

    Structurally satisfied by :class:`munasib.db.models.Product`, so the ORM row
    can be passed straight in, while tests can use any lightweight stand-in.
    """

    asin: str
    title: str | None
    title_clean: str | None
    brand: str | None
    brand_clean: str | None
    brand_missing: bool
    price: float | None
    price_max: float | None
    price_status: str | None
    price_missing: bool
    rank_value: int | None
    rank_category: str | None
    rank_missing: bool
    description: str | None
    features: list[str] | None


@dataclass(frozen=True)
class PeerStats:
    """Pre-computed peer distributions for one candidate set.

    Attributes:
        prices_by_asin: Price of every **priced** product, keyed by ASIN.
        sorted_prices: All those prices, ascending (bisect target).
        brand_by_asin: Normalized brand of every branded product, keyed by ASIN.
        sorted_prices_by_brand: Per-brand ascending price arrays (priced only).
        brand_counts: Number of products per normalized brand (priced or not).
        rank_by_asin: Sales rank of every ranked product, keyed by ASIN.
        rank_category_by_asin: Rank category of every ranked product.
        sorted_ranks_by_category: Per-category ascending rank arrays.
        min_brand_peers: Minimum other priced products a brand needs for a
            brand price percentile (``product_intel.min_brand_peers``).
        min_rank_peers: Minimum other ranked products a category needs for a
            rank percentile (``product_intel.min_rank_peers``).
    """

    prices_by_asin: Mapping[str, float]
    sorted_prices: tuple[float, ...]
    brand_by_asin: Mapping[str, str]
    sorted_prices_by_brand: Mapping[str, tuple[float, ...]]
    brand_counts: Mapping[str, int]
    rank_by_asin: Mapping[str, int]
    rank_category_by_asin: Mapping[str, str]
    sorted_ranks_by_category: Mapping[str, tuple[int, ...]]
    min_brand_peers: int
    min_rank_peers: int

    @classmethod
    def from_products(
        cls, products: Iterable[ProductLike], config: ProductIntelConfig
    ) -> PeerStats:
        """Build every distribution from one candidate set.

        Args:
            products: Candidate products (the whole subset, not a sample).
            config: Product-intelligence thresholds (peer minima).

        Returns:
            The distributions; identical input rows always produce identical
            output (rows are processed in ASIN order).
        """
        ordered = sorted(products, key=lambda product: product.asin)

        prices_by_asin: dict[str, float] = {}
        brand_by_asin: dict[str, str] = {}
        brand_counts: Counter[str] = Counter()
        rank_by_asin: dict[str, int] = {}
        rank_category_by_asin: dict[str, str] = {}
        brand_prices: dict[str, list[float]] = {}
        ranks_by_category: dict[str, list[int]] = {}

        for product in ordered:
            brand = normalized_brand(product)
            if brand is not None:
                brand_by_asin[product.asin] = brand
                brand_counts[brand] += 1
            if product.price is not None:
                prices_by_asin[product.asin] = float(product.price)
                if brand is not None:
                    brand_prices.setdefault(brand, []).append(float(product.price))
            if product.rank_value is not None and product.rank_category is not None:
                rank_by_asin[product.asin] = int(product.rank_value)
                rank_category_by_asin[product.asin] = product.rank_category
                ranks_by_category.setdefault(product.rank_category, []).append(
                    int(product.rank_value)
                )

        return cls(
            prices_by_asin=prices_by_asin,
            sorted_prices=tuple(sorted(prices_by_asin.values())),
            brand_by_asin=brand_by_asin,
            sorted_prices_by_brand={
                brand: tuple(sorted(values)) for brand, values in sorted(brand_prices.items())
            },
            brand_counts=dict(sorted(brand_counts.items())),
            rank_by_asin=rank_by_asin,
            rank_category_by_asin=rank_category_by_asin,
            sorted_ranks_by_category={
                category: tuple(sorted(values))
                for category, values in sorted(ranks_by_category.items())
            },
            min_brand_peers=config.min_brand_peers,
            min_rank_peers=config.min_rank_peers,
        )

    # ------------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------------

    @property
    def n_products(self) -> int:
        """Number of distinct ASINs observed across the distributions."""
        return len(
            set(self.prices_by_asin)
            | set(self.brand_by_asin)
            | set(self.rank_by_asin)
        )

    @property
    def n_priced(self) -> int:
        """Number of priced products (the ``price_percentile_category`` base)."""
        return len(self.sorted_prices)

    def price_percentile_category(self, asin: str) -> float | None:
        """Price percentile over every priced product in the candidate set.

        Args:
            asin: Subject product ASIN (must belong to the candidate set).

        Returns:
            Fraction of other priced products at or below this price
            (0 = cheapest, 1 = most expensive), or None when the product has no
            price or it is the only priced product (no peers to compare with).
        """
        price = self.prices_by_asin.get(asin)
        if price is None:
            return None
        return _percentile(self.sorted_prices, price)

    def price_percentile_brand(self, asin: str) -> float | None:
        """Price percentile among the subject's own brand.

        Args:
            asin: Subject product ASIN.

        Returns:
            Fraction of other **priced** products of the same brand at or below
            this price, or None when the product has no price, no brand, or the
            brand has fewer than ``min_brand_peers`` other priced products.
        """
        price = self.prices_by_asin.get(asin)
        brand = self.brand_by_asin.get(asin)
        if price is None or brand is None:
            return None
        values = self.sorted_prices_by_brand.get(brand, ())
        if len(values) - 1 < self.min_brand_peers:
            return None
        return _percentile(values, price)

    def rank_percentile(self, asin: str) -> float | None:
        """Sales-rank percentile inside the subject's rank category.

        A **lower** sales rank means a better seller, and the shared tie rule
        counts ``<=``, so 0 is the best-selling position and 1 the worst.

        Args:
            asin: Subject product ASIN.

        Returns:
            Fraction of other ranked products of the same category whose rank
            is at or below (worse than or equal to) this one, or None when the
            product has no rank or the category has fewer than ``min_rank_peers``
            other ranked products.
        """
        rank = self.rank_by_asin.get(asin)
        category = self.rank_category_by_asin.get(asin)
        if rank is None or category is None:
            return None
        values = self.sorted_ranks_by_category.get(category, ())
        if len(values) - 1 < self.min_rank_peers:
            return None
        return _percentile(values, rank)

    def brand_peer_count(self, asin: str) -> int:
        """How many **other** products share the subject's normalized brand.

        Args:
            asin: Subject product ASIN.

        Returns:
            Number of other candidate products with the same normalized brand;
            0 when the product has no brand or is the only one carrying it.
        """
        brand = self.brand_by_asin.get(asin)
        if brand is None:
            return 0
        return max(0, self.brand_counts.get(brand, 0) - 1)


def normalized_brand(product: ProductLike) -> str | None:
    """Normalized brand key of a product, or None when it has no usable brand.

    Prefers the pipeline's stored comparison key (``brand_clean``, produced by
    :func:`munasib.data.cleaning.normalize_brand`) and falls back to
    normalizing the display ``brand`` on the spot, so a row whose clean key is
    absent still groups by its real brand instead of silently counting as
    unbranded. An empty raw brand is **not** a brand (the dataset stores ``''``
    with ``brand_missing=True``).

    Args:
        product: Product row.

    Returns:
        Lower-cased, whitespace-collapsed brand key, or None.
    """
    if product.brand_clean is not None and product.brand_clean.strip():
        return product.brand_clean
    if product.brand is not None and product.brand.strip():
        return _normalize_raw_brand(product.brand)
    return None


def _percentile(sorted_values: tuple[float, ...], value: float) -> float | None:
    """Tie-ruled percentile of ``value`` inside a sorted array (self excluded).

    Args:
        sorted_values: Ascending array that contains the subject's own value.
        value: The subject's value.

    Returns:
        ``(count of entries <= value - 1) / (len - 1)`` rounded into ``[0, 1]``,
        or None when the array holds no peer besides the subject itself.
    """
    peers = len(sorted_values) - 1
    if peers < 1:
        return None
    at_or_below = bisect_right(sorted_values, value) - 1
    return min(1.0, max(0.0, at_or_below / peers))


__all__ = [
    "PeerStats",
    "ProductLike",
    "normalized_brand",
]

"""Product profile: metadata turned into typed, honest product indicators.

A :class:`ProductProfile` is the product-level view the risk stage consumes: how
complete the listing is, where its price sits against its category peer group
and against its own brand, where its sales rank sits inside its rank category,
and how much related-product evidence exists to compare it with.

Two rules govern every field here:

* **Nothing is invented.** A missing price, rank, brand, description or feature
  list yields ``None`` (for a value) or ``0.0`` (for a completeness that has
  nothing to measure) **plus** the matching ``*_missing`` flag. No field is ever
  back-filled from a peer, a median or a default.
* **Every threshold comes from config** (``product_intel`` in
  ``configs/config.yaml``): the description/feature completeness targets, the
  brand, rank and price peer minima, and the metadata field list.

Percentiles are delegated to :class:`munasib.product_intel.peers.PeerStats`,
which owns the tie rule (fraction of *other* products at or below the subject's
value) and the peer-minimum guards, so both stay defined in exactly one place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from munasib.core.config import ProductIntelConfig
from munasib.data.loaders import is_present
from munasib.db.repositories import RelatedCounts
from munasib.product_intel.peers import PeerStats, ProductLike, normalized_brand

#: Indicator keys stored in ``product_features.indicators`` (stable contract).
PROFILE_INDICATOR_KEYS: tuple[str, ...] = (
    "brand",
    "brand_missing",
    "brand_peer_count",
    "rank_category",
    "rank_value",
    "rank_missing",
    "rank_percentile",
    "rank_percentile_missing",
    "description_completeness",
    "description_missing",
    "feature_completeness",
    "features_missing",
    "metadata_completeness",
    "metadata_missing_fields",
    "price",
    "price_max",
    "price_status",
    "price_missing",
    "price_percentile_category",
    "price_percentile_category_missing",
    "price_peer_group",
    "price_peer_count",
    "price_peer_fallback",
    "price_percentile_brand",
    "price_percentile_brand_missing",
    "n_also_bought",
    "n_also_viewed",
    "n_usable_alternatives",
)


@dataclass(frozen=True)
class ProductProfile:
    """Typed product-level indicators derived from one product's metadata.

    See the module docstring for the two governing rules. ``*_missing`` flags
    always accompany the value they explain, so a consumer can tell "no data"
    from "data that scores 0".
    """

    asin: str

    #: Normalized brand comparison key (lower-cased), None when unbranded.
    brand: str | None
    brand_missing: bool
    #: Other candidate products sharing the normalized brand.
    brand_peer_count: int

    #: Sales-rank category string (``products.rank_category``), None when absent.
    rank_category: str | None
    rank_value: int | None
    rank_missing: bool
    #: 0 = best seller in the category, 1 = worst (None when not comparable).
    rank_percentile: float | None
    rank_percentile_missing: bool

    #: min(1, description words / target), 0.0 when the description is absent.
    description_completeness: float
    description_missing: bool
    #: min(1, feature bullets / target), 0.0 when there are no bullets.
    feature_completeness: float
    features_missing: bool
    #: Share of the configured metadata fields that are present (0-1).
    metadata_completeness: float
    #: Configured metadata fields that were absent, in config order.
    metadata_missing_fields: tuple[str, ...]

    price: float | None
    price_max: float | None
    price_status: str | None
    price_missing: bool
    #: 0 = cheapest priced product of the peer group, 1 = most expensive.
    price_percentile_category: float | None
    price_percentile_category_missing: bool
    #: Peer group the category percentile was measured in: the " > "-joined
    #: breadcrumb prefix used, or "all products" for the whole candidate set.
    price_peer_group: str
    #: Priced products in that group (the subject counts when it has a price).
    price_peer_count: int
    #: True when the product's own full breadcrumb was not the peer group.
    price_peer_fallback: bool
    #: Same, but among the brand's priced products only.
    price_percentile_brand: float | None
    price_percentile_brand_missing: bool

    n_also_bought: int
    n_also_viewed: int
    n_usable_alternatives: int

    # ------------------------------------------------------------------------
    # Serialization (the stored indicator contract)
    # ------------------------------------------------------------------------

    def to_indicators(self) -> dict[str, Any]:
        """Render the profile as a JSON-ready indicator mapping.

        Returns:
            Dict with exactly the keys in :data:`PROFILE_INDICATOR_KEYS`, in
            that order (a tuple field becomes a JSON list). Stable key order
            plus sorted JSON keys makes two runs byte-identical.
        """
        payload: dict[str, Any] = {}
        for key in PROFILE_INDICATOR_KEYS:
            value = getattr(self, key)
            payload[key] = list(value) if isinstance(value, tuple) else value
        return payload

    @classmethod
    def from_indicators(cls, asin: str, payload: dict[str, Any]) -> ProductProfile:
        """Rebuild a profile from a stored indicator mapping.

        Args:
            asin: Product ASIN (the mapping itself carries no identity).
            payload: Indicators previously produced by :meth:`to_indicators`.

        Returns:
            The reconstructed profile.

        Raises:
            ValueError: If the payload misses an indicator key, so a corrupt or
                truncated row fails loudly instead of silently defaulting.
        """
        missing = [key for key in PROFILE_INDICATOR_KEYS if key not in payload]
        if missing:
            raise ValueError(
                f"indicator payload for {asin} is missing keys: {sorted(missing)}"
            )
        values = {key: payload[key] for key in PROFILE_INDICATOR_KEYS}
        values["metadata_missing_fields"] = tuple(values["metadata_missing_fields"])
        return cls(asin=asin, **values)  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, Any]:
        """Full dataclass dump including ``asin`` (reports and tests only).

        Returns:
            Every field of the profile as a plain dict, ASIN first.
        """
        data = asdict(self)
        data["metadata_missing_fields"] = list(self.metadata_missing_fields)
        return data


def build_profile(
    product: ProductLike,
    peers: PeerStats,
    config: ProductIntelConfig,
    related: RelatedCounts | None = None,
) -> ProductProfile:
    """Derive the product indicators for one product.

    Args:
        product: Product row (an ORM :class:`~munasib.db.models.Product` or any
            stand-in exposing the :class:`~munasib.product_intel.peers.ProductLike`
            columns).
        peers: Peer distributions built from the **whole** candidate set (the
            subject must be part of that set for its percentiles to be defined).
        config: ``product_intel`` thresholds from ``configs/config.yaml``.
        related: Distinct related-product counts; None means "no related rows",
            which is recorded as zeros rather than guessed.

    Returns:
        The typed profile. Every ``None`` value carries its ``*_missing`` flag.
    """
    brand = normalized_brand(product)
    description_missing = not is_present(product.description)
    features_missing = not _has_feature_bullets(product.features)

    percentiles = {
        "price_percentile_category": peers.price_percentile_category(product.asin),
        "price_percentile_brand": peers.price_percentile_brand(product.asin),
        "rank_percentile": peers.rank_percentile(product.asin),
    }
    peer_group = peers.price_peer_selection(product.asin)
    missing_fields = _missing_metadata_fields(product, config)

    return ProductProfile(
        asin=product.asin,
        brand=brand,
        brand_missing=brand is None,
        brand_peer_count=peers.brand_peer_count(product.asin),
        rank_category=product.rank_category,
        rank_value=product.rank_value,
        rank_missing=product.rank_value is None,
        rank_percentile=percentiles["rank_percentile"],
        rank_percentile_missing=percentiles["rank_percentile"] is None,
        description_completeness=_description_completeness(
            product.description, config.target_description_words
        ),
        description_missing=description_missing,
        feature_completeness=_feature_completeness(
            product.features, config.target_feature_count
        ),
        features_missing=features_missing,
        metadata_completeness=_metadata_completeness(product, config),
        metadata_missing_fields=missing_fields,
        price=float(product.price) if product.price is not None else None,
        price_max=float(product.price_max) if product.price_max is not None else None,
        price_status=product.price_status,
        price_missing=product.price is None,
        price_percentile_category=percentiles["price_percentile_category"],
        price_percentile_category_missing=percentiles["price_percentile_category"] is None,
        price_peer_group=peer_group.group,
        price_peer_count=peer_group.priced_count,
        price_peer_fallback=peer_group.fallback,
        price_percentile_brand=percentiles["price_percentile_brand"],
        price_percentile_brand_missing=percentiles["price_percentile_brand"] is None,
        n_also_bought=related.also_bought if related is not None else 0,
        n_also_viewed=related.also_viewed if related is not None else 0,
        n_usable_alternatives=related.usable_alternatives if related is not None else 0,
    )


def build_profiles(
    products: list[ProductLike],
    peers: PeerStats,
    config: ProductIntelConfig,
    related_by_asin: dict[str, RelatedCounts] | None = None,
) -> list[ProductProfile]:
    """Build profiles for a batch, ordered by ASIN.

    Args:
        products: Candidate products (order does not matter; output is sorted).
        peers: Peer distributions for the same candidate set.
        config: ``product_intel`` thresholds.
        related_by_asin: Optional ASIN -> :class:`RelatedCounts` map; products
            absent from it get zero counts.

    Returns:
        One profile per product, ascending by ASIN (deterministic).
    """
    related_by_asin = related_by_asin or {}
    return [
        build_profile(product, peers, config, related_by_asin.get(product.asin))
        for product in sorted(products, key=lambda item: item.asin)
    ]


# ------------------------------------------------------------------------------
# Completeness helpers (pure, unit-tested at the boundaries)
# ------------------------------------------------------------------------------


def description_word_count(description: str | None) -> int:
    """Count the whitespace-separated words of a description.

    Args:
        description: Cleaned description text or None.

    Returns:
        Word count; 0 for a missing or blank description.
    """
    if not is_present(description):
        return 0
    return len(str(description).split())


def _description_completeness(description: str | None, target_words: int) -> float:
    """Completeness of a description against the configured word target.

    Formula: ``min(1.0, description_word_count(description) / target_words)``.
    A missing description scores 0.0 (never a guessed partial score) and its
    ``description_missing`` flag is set alongside.

    Args:
        description: Cleaned description text or None.
        target_words: ``product_intel.target_description_words`` (>= 1).

    Returns:
        Completeness in [0, 1].
    """
    if target_words < 1:
        raise ValueError(f"target_words must be >= 1, got {target_words}")
    return min(1.0, description_word_count(description) / target_words)


def feature_bullet_count(features: list[str] | None) -> int:
    """Count the usable bullets of a feature list.

    Args:
        features: Feature bullets (``products.features`` JSON array) or None.

    Returns:
        Number of present, non-blank bullets; 0 when the list is missing or
        holds nothing usable.
    """
    if features is None:
        return 0
    return sum(1 for item in features if is_present(item))


def _has_feature_bullets(features: list[str] | None) -> bool:
    """Whether a feature list carries at least one usable bullet.

    Args:
        features: Feature bullets or None.

    Returns:
        True when at least one bullet is present.
    """
    return feature_bullet_count(features) > 0


def _feature_completeness(features: list[str] | None, target_count: int) -> float:
    """Completeness of a feature list against the configured bullet target.

    Formula: ``min(1.0, feature_bullet_count(features) / target_count)``.
    A missing or empty list scores 0.0 with ``features_missing`` set.

    Args:
        features: Feature bullets or None.
        target_count: ``product_intel.target_feature_count`` (>= 1).

    Returns:
        Completeness in [0, 1].
    """
    if target_count < 1:
        raise ValueError(f"target_count must be >= 1, got {target_count}")
    return min(1.0, feature_bullet_count(features) / target_count)


def _field_present(product: ProductLike, field_name: str) -> bool:
    """Presence test for one configured metadata field.

    Args:
        product: Product row.
        field_name: One of :data:`munasib.core.config.METADATA_FIELD_VOCABULARY`.

    Returns:
        True when the field carries information. ``rank`` counts as present only
        when a parsed ``rank_value`` exists (an unparsable rank string carries
        no rank to reason about).

    Raises:
        ValueError: If ``field_name`` is outside the configured vocabulary, so
            a typo in config fails instead of quietly scoring 0.
        KeyError: If the product row lacks the column the field maps to.
    """
    if field_name == "title":
        return is_present(product.title_clean) or is_present(product.title)
    if field_name == "brand":
        return normalized_brand(product) is not None
    if field_name == "price":
        return product.price is not None
    if field_name == "rank":
        return product.rank_value is not None
    if field_name == "description":
        return is_present(product.description)
    if field_name == "features":
        return _has_feature_bullets(product.features)
    raise ValueError(f"unknown metadata field {field_name!r}")


def _missing_metadata_fields(product: ProductLike, config: ProductIntelConfig) -> tuple[str, ...]:
    """Configured metadata fields that are absent, in config order.

    Args:
        product: Product row.
        config: ``product_intel`` settings (``metadata_fields``).

    Returns:
        Tuple of absent field names (empty when the listing is complete).
    """
    return tuple(name for name in config.metadata_fields if not _field_present(product, name))


def _metadata_completeness(product: ProductLike, config: ProductIntelConfig) -> float:
    """Share of the configured metadata fields that are present.

    Formula: ``present_fields / len(config.metadata_fields)`` - an unweighted
    mean over exactly the fields the config names, never over a hard-coded
    list. An empty field list is rejected by the config validator, so the
    denominator is always >= 1.

    Args:
        product: Product row.
        config: ``product_intel`` settings.

    Returns:
        Completeness in [0, 1].
    """
    fields_list = config.metadata_fields
    present = sum(1 for name in fields_list if _field_present(product, name))
    return present / len(fields_list)


def indicator_field_names() -> tuple[str, ...]:
    """Names of every dataclass field of :class:`ProductProfile`.

    Returns:
        Field names in declaration order (``asin`` first).
    """
    return tuple(field.name for field in fields(ProductProfile))


__all__ = [
    "PROFILE_INDICATOR_KEYS",
    "ProductProfile",
    "RelatedCounts",
    "build_profile",
    "build_profiles",
    "description_word_count",
    "feature_bullet_count",
    "indicator_field_names",
]

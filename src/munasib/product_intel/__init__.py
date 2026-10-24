"""Product profile extraction and metadata intelligence.

Turns a product's stored metadata into typed indicators for the risk stage:
metadata completeness, price position against the category and the brand,
sales-rank position inside its rank category, and related-product counts.

Layers (all thresholds come from ``product_intel`` in ``configs/config.yaml``):

* :mod:`munasib.product_intel.peers` - peer distributions and the tie-ruled
  percentile lookups, computed once per candidate set.
* :mod:`munasib.product_intel.profile` - the :class:`ProductProfile` dataclass
  and :func:`build_profile`, which never imputes a missing value.
* :mod:`munasib.product_intel.features` - persistence into ``product_features``
  with the dataset build timestamp (deterministic, idempotent).

``scripts/compute_product_features.py`` is the CLI entry point; every formula
is documented in ``docs/product_intelligence.md``.
"""

from munasib.product_intel.features import (
    list_profiled_asins,
    read_build_timestamp,
    read_indicators,
    read_profile,
    read_profiles,
    write_product_features,
)
from munasib.product_intel.peers import (
    ALL_PRODUCTS_GROUP,
    PeerStats,
    PricePeerSelection,
    ProductLike,
    category_prefixes,
    normalized_brand,
)
from munasib.product_intel.profile import (
    PROFILE_INDICATOR_KEYS,
    ProductProfile,
    build_profile,
    build_profiles,
    description_word_count,
    feature_bullet_count,
)

__all__ = [
    "ALL_PRODUCTS_GROUP",
    "PROFILE_INDICATOR_KEYS",
    "PeerStats",
    "PricePeerSelection",
    "ProductLike",
    "ProductProfile",
    "build_profile",
    "build_profiles",
    "category_prefixes",
    "description_word_count",
    "feature_bullet_count",
    "list_profiled_asins",
    "normalized_brand",
    "read_build_timestamp",
    "read_indicators",
    "read_profile",
    "read_profiles",
    "write_product_features",
]

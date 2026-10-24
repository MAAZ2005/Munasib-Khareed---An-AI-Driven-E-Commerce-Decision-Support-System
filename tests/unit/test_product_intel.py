"""Tests for the Product Intelligence module (p11).

Two layers are covered here:

* **Pure** - :class:`~munasib.product_intel.peers.PeerStats` and
  :func:`~munasib.product_intel.profile.build_profile` against a tiny
  hand-checked candidate set: the tie rule, percentile bounds, the peer
  minima, the completeness boundaries, the one-product / one-brand edge cases
  and determinism across two runs.
* **Persistence** - :mod:`munasib.product_intel.features` against an in-memory
  serving cache: upsert idempotence, the dataset-build `computed_at`, the
  JSON round-trip and the zero-count handling for a product with no related
  rows.

A final test runs against the **real** ``data/db/munasib.sqlite`` and skips
when it is absent, asserting only invariants (bounds, missing-value rules) -
never a hard-coded statistic. Hermetic otherwise: no network, no GPU.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import insert
from sqlalchemy.orm import Session

from munasib.core.config import ProductIntelConfig, Settings, get_settings
from munasib.core.exceptions import DataError
from munasib.db.models import Product, ProductFeature, RelatedProduct
from munasib.db.repositories import RelatedCounts
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)
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

#: Five products: three priced (two tied at the low end), one unpriced, one
#: unbranded. Every expectation below is computed by hand from these rows.
CHEAP_A = "B0000000CA"
CHEAP_B = "B0000000CB"
DEAR = "B0000000DE"
UNPRICED = "B0000000NP"
UNBRANDED = "B0000000NB"

CHEAP_PRICE = 10.0
DEAR_PRICE = 30.0

#: Sentinel so ``_product(features=None)`` can mean "no feature list" while an
#: omitted ``features`` argument still gets the default two bullets.
_NO_FEATURES = object()


@dataclass
class StubProduct:
    """Lightweight :class:`ProductLike` stand-in for pure (DB-free) tests."""

    asin: str
    title: str | None = None
    title_clean: str | None = None
    brand: str | None = None
    brand_clean: str | None = None
    brand_missing: bool = False
    price: float | None = None
    price_max: float | None = None
    price_status: str | None = None
    price_missing: bool = True
    rank_value: int | None = None
    rank_category: str | None = None
    rank_missing: bool = True
    category_path: str | None = None
    description: str | None = None
    features: list[str] | None = None


def _product(
    asin: str,
    *,
    brand: str | None = "acme",
    price: float | None = CHEAP_PRICE,
    rank_value: int | None = 1,
    rank_category: str | None = "Widgets",
    category_path: str | None = None,
    description: str | None = "A widget.",
    features: list[str] | None | object = _NO_FEATURES,
) -> StubProduct:
    """Build one StubProduct with sensible defaults for the fixture set.

    Args:
        asin: Product ASIN.
        brand: Raw brand (None = unbranded).
        price: Price (None = unpriced).
        rank_value: Sales rank (None = unranked).
        rank_category: Rank category (None = unranked).
        category_path: Browse breadcrumb (None = none published, which sends
            the price percentile to the whole-candidate-set peer group).
        description: Cleaned description.
        features: Feature bullets; ``None`` stays ``None`` (an absent list),
            and omitting the argument yields the default two bullets.

    Returns:
        The stub product row.
    """
    bullets: list[str] | None = (
        ["bullet one", "bullet two"] if features is _NO_FEATURES else features  # type: ignore[assignment]
    )
    return StubProduct(
        asin=asin,
        title=f"Product {asin}",
        title_clean=f"product {asin}",
        brand=brand,
        brand_clean=brand,
        brand_missing=brand is None,
        price=price,
        price_status="missing" if price is None else "single",
        price_missing=price is None,
        rank_value=rank_value,
        rank_category=rank_category,
        rank_missing=rank_value is None,
        category_path=category_path,
        description=description,
        features=bullets,
    )


#: The five-product candidate set (two products tied at CHEAP_PRICE). Brands
#: are deliberate: "acme" has two members, "deluxe" and "ghost" one each, and
#: UNBRANDED has none - so the brand peer-count and brand-percentile guards can
#: all be exercised.
FIXTURE_PRODUCTS: list[StubProduct] = [
    _product(CHEAP_A, brand="acme", price=CHEAP_PRICE, rank_value=1),
    _product(CHEAP_B, brand="acme", price=CHEAP_PRICE, rank_value=2),
    _product(DEAR, brand="deluxe", price=DEAR_PRICE, rank_value=9),
    _product(UNPRICED, brand="ghost", price=None, rank_value=None, rank_category=None),
    _product(UNBRANDED, brand=None, price=20.0, rank_value=3),
]

CONFIG = ProductIntelConfig(target_description_words=10, target_feature_count=2)


def min_price_config(min_price_peers: int) -> ProductIntelConfig:
    """The fixture config with only the price peer minimum overridden.

    Args:
        min_price_peers: ``product_intel.min_price_peers`` for the test.

    Returns:
        A config whose completeness targets stay the fixture defaults.
    """
    return ProductIntelConfig(
        target_description_words=CONFIG.target_description_words,
        target_feature_count=CONFIG.target_feature_count,
        min_price_peers=min_price_peers,
    )


def _peers(products: list[StubProduct] | None = None, **overrides: object) -> PeerStats:
    """Build PeerStats over a candidate set with optional config overrides.

    Args:
        products: Candidate set (defaults to :data:`FIXTURE_PRODUCTS`).
        **overrides: ``ProductIntelConfig`` field overrides.

    Returns:
        The peer distributions.
    """
    return PeerStats.from_products(
        products if products is not None else FIXTURE_PRODUCTS,
        ProductIntelConfig(**overrides) if overrides else CONFIG,  # type: ignore[arg-type]
    )


# ------------------------------------------------------------------------------
# PeerStats: tie rule and bounds
# ------------------------------------------------------------------------------


def test_price_percentile_category_applies_the_tie_rule() -> None:
    """Two products tied at the low price each score 1/3, not 0.

    Priced products are CHEAP_A (10), CHEAP_B (10), UNBRANDED (20) and DEAR
    (30); the unpriced product is not in the price distribution at all. For
    CHEAP_A the three peers are CHEAP_B (10), UNBRANDED (20) and DEAR (30), and
    only CHEAP_B is at or below 10, so the tie rule gives 1/3 - the tie is
    counted, but the subject's own row is not.
    """
    stats = _peers()
    assert stats.price_percentile_category(CHEAP_A) == pytest.approx(1 / 3)
    # Symmetry: the tied twin scores exactly the same.
    assert stats.price_percentile_category(CHEAP_B) == pytest.approx(1 / 3)
    # UNBRANDED (20) has two cheaper peers (the tied pair) -> 2/3.
    assert stats.price_percentile_category(UNBRANDED) == pytest.approx(2 / 3)


def test_price_percentile_category_extremes_are_zero_and_one() -> None:
    """Cheapest scores 0, most expensive scores 1, over the priced subset."""
    stats = _peers()
    # UNBRANDED (20) is the middle price; DEAR (30) is the maximum.
    assert stats.price_percentile_category(DEAR) == 1.0
    # The minimum is shared by CHEAP_A/CHEAP_B, so neither reaches 0 - the
    # strictly-cheapest case is checked with a unique minimum below.
    stats_unique = _peers([_product("A", price=5.0), _product("B", price=10.0), _product("C", price=15.0)])
    assert stats_unique.price_percentile_category("A") == 0.0
    assert stats_unique.price_percentile_category("C") == 1.0
    assert stats_unique.price_percentile_category("B") == pytest.approx(0.5)


def test_price_percentile_category_is_none_without_a_price() -> None:
    """An unpriced product gets None, never a fabricated percentile."""
    stats = _peers()
    assert stats.price_percentile_category(UNPRICED) is None


def test_price_percentile_category_is_none_with_no_peers() -> None:
    """A single priced product has no peers, so the percentile is undefined."""
    stats = _peers([_product("ONLY", price=42.0)])
    assert stats.n_priced == 1
    assert stats.price_percentile_category("ONLY") is None


def test_percentiles_stay_within_zero_and_one_for_every_product() -> None:
    """No product ever produces a percentile outside [0, 1]."""
    stats = _peers()
    for product in FIXTURE_PRODUCTS:
        for value in (
            stats.price_percentile_category(product.asin),
            stats.price_percentile_brand(product.asin),
            stats.rank_percentile(product.asin),
        ):
            if value is not None:
                assert 0.0 <= value <= 1.0


def test_rank_percentile_is_zero_for_the_best_seller() -> None:
    """A lower rank_value is better, so the top rank scores 0.

    Ranked products are CHEAP_A (1), CHEAP_B (2), UNBRANDED (3) and DEAR (9) -
    four products, three peers each. CHEAP_A's rank is at or below all three
    others, so 0/3. ``min_rank_peers`` is dropped to 1 here so the four-row
    fixture is comparable at all.
    """
    stats = _peers(min_rank_peers=1)
    assert stats.rank_percentile(CHEAP_A) == 0.0
    assert stats.rank_percentile(DEAR) == 1.0
    # Mid rank: peers at or below rank 3 are ranks 1 and 2 -> 2/3.
    assert stats.rank_percentile(UNBRANDED) == pytest.approx(2 / 3)


def test_rank_percentile_is_none_below_the_default_peer_minimum() -> None:
    """The default ``min_rank_peers=5`` makes a four-product category None."""
    stats = _peers()
    assert stats.rank_percentile(CHEAP_A) is None
    assert stats.rank_percentile(DEAR) is None


def test_rank_percentile_is_none_without_a_rank() -> None:
    """An unranked product gets None."""
    assert _peers().rank_percentile(UNPRICED) is None


def test_brand_percentile_requires_min_brand_peers() -> None:
    """A brand below the peer minimum reports None, not a thin percentile.

    The fixture brand "acme" has only CHEAP_A and CHEAP_B priced, so with
    ``min_brand_peers=5`` there is nothing to compare against.
    """
    stats = _peers()
    assert stats.price_percentile_brand(CHEAP_A) is None


def test_brand_percentile_appears_once_the_minimum_is_met() -> None:
    """Raising the fixture to the minimum yields a real brand percentile.

    Six same-brand priced products (the subject plus five peers) satisfies
    ``min_brand_peers=5``.
    """
    products = [_product(f"B{i:09d}", price=float(i + 1)) for i in range(6)]
    stats = _peers(products, min_brand_peers=5)
    # Subject is the cheapest of six, so 0 of its 5 peers are <= it.
    assert stats.price_percentile_brand("B000000000") == 0.0
    # Subject is the most expensive, so all 5 peers are <= it.
    assert stats.price_percentile_brand("B000000005") == 1.0


def test_brand_percentile_is_none_without_a_brand() -> None:
    """An unbranded product gets None for the brand percentile."""
    stats = _peers()
    # UNBRANDED is priced, but no brand -> no brand percentile.
    assert stats.brand_by_asin.get(UNBRANDED) is None
    assert stats.price_percentile_brand(UNBRANDED) is None


def test_rank_percentile_is_none_for_a_category_that_is_too_small() -> None:
    """A rank category below ``min_rank_peers`` reports None."""
    products = [
        _product("C1", rank_value=1, rank_category="Tiny"),
        _product("C2", rank_value=2, rank_category="Tiny"),
    ]
    stats = _peers(products, min_rank_peers=5)
    assert stats.rank_percentile("C1") is None
    assert stats.rank_percentile("C2") is None


def test_rank_percentile_is_scoped_to_the_own_category() -> None:
    """Products in different rank categories never see each other."""
    products = [
        _product("X1", rank_value=1, rank_category="Alpha"),
        _product("X2", rank_value=100, rank_category="Alpha"),
        _product("Y1", rank_value=50, rank_category="Beta"),
        _product("Y2", rank_value=60, rank_category="Beta"),
    ]
    stats = _peers(products, min_rank_peers=1)
    # Alpha ranks are [1, 100]; X1 is at or below its only peer -> 0.
    assert stats.rank_percentile("X1") == 0.0
    # Beta ranks are [50, 60]; Y1 is at or below its only peer -> 0. Y1 never
    # sees Alpha's rank 100, which would otherwise have made it 0/3.
    assert stats.rank_percentile("Y1") == 0.0


def test_brand_peer_count_excludes_the_subject() -> None:
    """The subject is not its own peer, so a two-brand-member count is 1."""
    stats = _peers()
    # "acme" = CHEAP_A + CHEAP_B, so each sees 1 peer.
    assert stats.brand_peer_count(CHEAP_A) == 1
    assert stats.brand_peer_count(CHEAP_B) == 1


def test_brand_peer_count_is_zero_for_a_unique_or_missing_brand() -> None:
    """A unique brand or no brand at all both report 0 peers."""
    stats = _peers()
    assert stats.brand_peer_count(DEAR) == 0  # "deluxe" is unique to DEAR
    assert stats.brand_peer_count(UNBRANDED) == 0


# ------------------------------------------------------------------------------
# Price peer groups: the breadcrumb walk that picks the comparison set
# ------------------------------------------------------------------------------

#: Two sibling browse nodes under one root, used by the peer-group tests below.
WATER_FILTERS = "Appliances > Water Filters"
AIR_PURIFIERS = "Appliances > Air Purifiers"

#: ASINs of the hand-checked grouped fixture (prices chosen so the expected
#: percentiles below are computable by hand).
WF1, WF2, WF3, WF4 = "B0000000WF1", "B0000000WF2", "B0000000WF3", "B0000000WF4"
AP1, AP2, AP3, AP4 = "B0000000AP1", "B0000000AP2", "B0000000AP3", "B0000000AP4"


def _grouped_fixture() -> list[StubProduct]:
    """Eight priced products, four per browse node, prices 10..400.

    Returns:
        The candidate set; Water Filters holds 10/20/30/40 and Air Purifiers
        100/200/300/400, so a product's in-group and whole-set percentiles
        always differ.
    """
    return [
        _product(WF1, category_path=WATER_FILTERS, price=10.0),
        _product(WF2, category_path=WATER_FILTERS, price=20.0),
        _product(WF3, category_path=WATER_FILTERS, price=30.0),
        _product(WF4, category_path=WATER_FILTERS, price=40.0),
        _product(AP1, category_path=AIR_PURIFIERS, price=100.0),
        _product(AP2, category_path=AIR_PURIFIERS, price=200.0),
        _product(AP3, category_path=AIR_PURIFIERS, price=300.0),
        _product(AP4, category_path=AIR_PURIFIERS, price=400.0),
    ]


def test_category_prefixes_run_most_specific_first() -> None:
    """Every leading level is a prefix, from the leaf up to the root."""
    assert category_prefixes(WATER_FILTERS) == [WATER_FILTERS, "Appliances"]
    assert category_prefixes("Appliances") == ["Appliances"]


def test_price_peer_group_uses_the_leaf_when_it_is_large_enough() -> None:
    """Four priced products beats ``min_price_peers=3``, so no fallback."""
    stats = _peers(_grouped_fixture(), min_price_peers=3)
    selection = stats.price_peer_selection(WF1)
    assert selection.group == WATER_FILTERS
    assert selection.priced_count == 4
    assert selection.fallback is False


def test_price_peer_group_falls_back_one_level_when_the_leaf_is_too_small() -> None:
    """A 2-product leaf falls back to the root, which has 6 priced products."""
    products = [
        _product(WF1, category_path=WATER_FILTERS, price=10.0),
        _product(WF2, category_path=WATER_FILTERS, price=20.0),
        _product(AP1, category_path=AIR_PURIFIERS, price=100.0),
        _product(AP2, category_path=AIR_PURIFIERS, price=200.0),
        _product(AP3, category_path=AIR_PURIFIERS, price=300.0),
        _product(AP4, category_path=AIR_PURIFIERS, price=400.0),
    ]
    stats = _peers(products, min_price_peers=3)
    selection = stats.price_peer_selection(WF1)
    assert selection.group == "Appliances"
    assert selection.priced_count == 6
    assert selection.fallback is True
    # Hand check against that 6-product group: prices 10, 20, 100, 200, 300,
    # 400; at or below 10 is just WF1 -> (1-1)/(6-1) = 0.0, and at or below 20
    # is WF1+WF2 -> (2-1)/(6-1) = 0.2.
    assert stats.price_percentile_category(WF1) == 0.0
    assert stats.price_percentile_category(WF2) == pytest.approx(0.2)


def test_price_peer_group_falls_back_to_all_products_when_no_prefix_qualifies() -> None:
    """Neither the leaf nor the root clears the minimum -> the whole set."""
    products = [
        _product(WF1, category_path=WATER_FILTERS, price=10.0),
        _product(WF2, category_path=WATER_FILTERS, price=20.0),
    ]
    stats = _peers(products, min_price_peers=3)
    selection = stats.price_peer_selection(WF1)
    assert selection.group == ALL_PRODUCTS_GROUP
    assert selection.priced_count == 2
    assert selection.fallback is True


def test_price_peer_group_is_all_products_without_a_breadcrumb() -> None:
    """A product with no breadcrumb compares against the whole candidate set."""
    products = [*_grouped_fixture(), _product("B0000000NC", category_path=None, price=250.0)]
    stats = _peers(products, min_price_peers=3)
    selection = stats.price_peer_selection("B0000000NC")
    assert selection.group == ALL_PRODUCTS_GROUP
    assert selection.priced_count == 9  # 8 grouped + the ungrouped subject
    assert selection.fallback is True
    # Hand check over all nine prices (10, 20, 30, 40, 100, 200, 250, 300,
    # 400): at or below 250 is seven products -> (7-1)/(9-1) = 0.75.
    assert stats.price_percentile_category("B0000000NC") == pytest.approx(0.75)


def test_price_percentile_is_measured_inside_the_group_not_the_whole_set() -> None:
    """The point of the peer group: 40 is the priciest water filter, not a mid-price."""
    stats = _peers(_grouped_fixture(), min_price_peers=3)
    # Inside "Appliances > Water Filters" (10, 20, 30, 40) WF4 is the maximum.
    assert stats.price_percentile_category(WF4) == 1.0
    # Against all eight it would have been (4-1)/(8-1) = 3/7 - the group is
    # what changed, not the arithmetic.
    assert stats.price_percentile_category(WF4) != pytest.approx(3 / 7)


def test_price_peer_count_includes_the_subject_when_it_is_priced() -> None:
    """A priced subject is one of its own group's priced products."""
    stats = _peers(_grouped_fixture(), min_price_peers=3)
    assert stats.price_peer_selection(WF2).priced_count == 4  # WF1..WF4


def test_price_peer_group_is_reported_for_an_unpriced_product_too() -> None:
    """The group is where the price *would* be compared; the percentile stays None."""
    products = [
        _product(WF1, category_path=WATER_FILTERS, price=10.0),
        _product(WF2, category_path=WATER_FILTERS, price=20.0),
        _product(WF3, category_path=WATER_FILTERS, price=None),
    ]
    stats = _peers(products, min_price_peers=2)
    selection = stats.price_peer_selection(WF3)
    assert selection.group == WATER_FILTERS
    # Only the priced rows count: the subject has no price, so it is not one.
    assert selection.priced_count == 2
    assert stats.price_percentile_category(WF3) is None


def test_breadcrumb_prefixes_match_on_level_boundaries_only() -> None:
    """"Parts" is not a prefix of "Parts & Accessories" - the text is a coincidence."""
    products = [
        _product("PARTS001", category_path="Appliances > Parts", price=10.0),
        _product("PARTS002", category_path="Appliances > Parts", price=20.0),
        _product("ACCES001", category_path="Appliances > Parts & Accessories", price=30.0),
        _product("ACCES002", category_path="Appliances > Parts & Accessories", price=40.0),
    ]
    stats = _peers(products, min_price_peers=2)
    selection = stats.price_peer_selection("PARTS001")
    assert selection.group == "Appliances > Parts"
    assert selection.priced_count == 2  # not 4: the sibling node did not leak in


def test_price_peer_selection_is_deterministic_across_input_order() -> None:
    """ASIN-ordered building makes the chosen group independent of row order."""
    products = _grouped_fixture()
    forward = _peers(products, min_price_peers=3)
    backward = _peers(list(reversed(products)), min_price_peers=3)
    for product in products:
        assert forward.price_peer_selection(product.asin) == backward.price_peer_selection(
            product.asin
        )
        assert forward.price_percentile_category(product.asin) == (
            backward.price_percentile_category(product.asin)
        )


def test_price_percentiles_stay_in_bounds_inside_peer_groups() -> None:
    """Grouped percentiles obey the same [0, 1] bounds as the whole-set ones."""
    stats = _peers(_grouped_fixture(), min_price_peers=3)
    for product in _grouped_fixture():
        value = stats.price_percentile_category(product.asin)
        assert value is not None
        assert 0.0 <= value <= 1.0
        assert stats.price_peer_selection(product.asin).priced_count >= 2


def test_single_product_candidate_set_is_handled() -> None:
    """One product, no peers: every percentile is None, counts still valid."""
    stats = _peers([_product("SOLO", price=7.5, rank_value=4)])
    assert stats.n_products == 1
    assert stats.n_priced == 1
    assert stats.price_percentile_category("SOLO") is None
    assert stats.rank_percentile("SOLO") is None
    assert stats.brand_peer_count("SOLO") == 0


def test_single_brand_candidate_set_is_handled() -> None:
    """One brand carrying every product: peer counts are still consistent."""
    products = [_product(f"S{i:09d}", price=float(i + 1)) for i in range(3)]
    stats = _peers(products, min_brand_peers=2)
    assert stats.brand_peer_count("S000000000") == 2
    assert stats.price_percentile_brand("S000000000") == 0.0


def test_normalized_brand_falls_back_to_the_raw_brand() -> None:
    """A row without ``brand_clean`` still groups by its normalized brand."""
    product = StubProduct(asin="F1", brand="  Acme, ", brand_clean=None)
    assert normalized_brand(product) == "acme"


def test_normalized_brand_is_none_for_an_empty_brand() -> None:
    """An empty-string brand (stored with brand_missing) is not a brand."""
    product = StubProduct(asin="F2", brand="", brand_clean=None, brand_missing=True)
    assert normalized_brand(product) is None


# ------------------------------------------------------------------------------
# build_profile: completeness boundaries and flags
# ------------------------------------------------------------------------------


def test_missing_price_sets_none_and_the_flag() -> None:
    """No price -> price None, flag True, and both price percentiles None."""
    profile = build_profile(_product(UNPRICED, price=None), _peers(), CONFIG)
    assert profile.price is None
    assert profile.price_missing is True
    assert profile.price_percentile_category is None
    assert profile.price_percentile_category_missing is True


def test_profile_carries_the_price_peer_group_fields() -> None:
    """The peer group a percentile was measured in is reported, not hidden."""
    products = _grouped_fixture()
    stats = _peers(products, min_price_peers=3)
    dearest_water_filter = next(p for p in products if p.asin == WF4)
    profile = build_profile(dearest_water_filter, stats, min_price_config(3))
    assert profile.price_peer_group == WATER_FILTERS
    assert profile.price_peer_count == 4
    assert profile.price_peer_fallback is False
    assert profile.price_percentile_category == 1.0
    assert profile.price_percentile_category_missing is False


def test_profile_flags_a_fallback_peer_group() -> None:
    """A fallback group is reported as such so the comparison stays honest."""
    products = [
        _product(WF1, category_path=WATER_FILTERS, price=10.0),
        _product(WF2, category_path=WATER_FILTERS, price=20.0),
    ]
    stats = _peers(products, min_price_peers=3)
    profile = build_profile(products[0], stats, min_price_config(3))
    assert profile.price_peer_group == ALL_PRODUCTS_GROUP
    assert profile.price_peer_count == 2
    assert profile.price_peer_fallback is True


def test_profile_peer_group_is_all_products_when_the_breadcrumb_is_missing() -> None:
    """No breadcrumb at all still yields a group, never an empty string."""
    stats = _peers(FIXTURE_PRODUCTS)
    for product in FIXTURE_PRODUCTS:
        profile = build_profile(product, stats, CONFIG)
        assert profile.price_peer_group == ALL_PRODUCTS_GROUP
        assert profile.price_peer_fallback is True
        assert profile.price_peer_count >= 0


def test_description_completeness_at_the_boundaries() -> None:
    """Empty -> 0.0 + flag; exactly the target -> 1.0; over the target -> 1.0."""
    stats = _peers()
    empty = build_profile(_product("D0", description=None), stats, CONFIG)
    assert empty.description_completeness == 0.0
    assert empty.description_missing is True

    exactly = build_profile(
        _product("D1", description="one two three four five six seven eight nine ten"),
        stats,
        CONFIG,
    )
    assert exactly.description_completeness == 1.0
    assert exactly.description_missing is False

    over = build_profile(
        _product("D2", description=" ".join(["w"] * 25)), stats, CONFIG
    )
    assert over.description_completeness == 1.0

    half = build_profile(_product("D3", description="one two three four five"), stats, CONFIG)
    assert half.description_completeness == pytest.approx(0.5)


def test_feature_completeness_at_the_boundaries() -> None:
    """Empty -> 0.0 + flag; at/over the target -> 1.0; blank bullets ignored."""
    stats = _peers()
    none = build_profile(_product("F0", features=None), stats, CONFIG)
    assert none.feature_completeness == 0.0
    assert none.features_missing is True

    empty_list = build_profile(_product("F1", features=[]), stats, CONFIG)
    assert empty_list.feature_completeness == 0.0
    assert empty_list.features_missing is True

    exactly = build_profile(_product("F2", features=["a", "b"]), stats, CONFIG)
    assert exactly.feature_completeness == 1.0
    assert exactly.features_missing is False

    # Blank entries are not bullets: 3 slots but only 1 usable -> 0.5.
    blank = build_profile(_product("F3", features=["a", "", "  "]), stats, CONFIG)
    assert blank.feature_completeness == pytest.approx(0.5)
    assert blank.features_missing is False


def test_metadata_completeness_is_the_share_of_present_fields() -> None:
    """All six fields present -> 1.0; the unpriced fixture -> 5/6."""
    stats = _peers()
    full = build_profile(_product("M1"), stats, CONFIG)
    assert full.metadata_completeness == 1.0
    assert full.metadata_missing_fields == ()

    partial = build_profile(_product(UNPRICED, price=None, rank_value=None, rank_category=None), stats, CONFIG)
    # title, brand, description, features present; price and rank absent.
    assert partial.metadata_completeness == pytest.approx(4 / 6)
    assert partial.metadata_missing_fields == ("price", "rank")


def test_metadata_completeness_honours_the_configured_field_list() -> None:
    """A narrower field list scores only over the fields it names."""
    narrow = ProductIntelConfig(metadata_fields=["title", "brand"])
    profile = build_profile(_product("N1", brand=None), _peers(), narrow)
    assert profile.metadata_completeness == pytest.approx(0.5)
    assert profile.metadata_missing_fields == ("brand",)


def test_related_counts_default_to_zero_when_absent() -> None:
    """No RelatedCounts mapping -> zeros, never an exception."""
    profile = build_profile(_product("R1"), _peers(), CONFIG, related=None)
    assert (profile.n_also_bought, profile.n_also_viewed, profile.n_usable_alternatives) == (0, 0, 0)


def test_related_counts_are_recorded_when_supplied() -> None:
    """Supplied counts land verbatim on the profile."""
    counts = RelatedCounts(also_bought=4, also_viewed=7, usable_alternatives=3)
    profile = build_profile(_product("R2"), _peers(), CONFIG, related=counts)
    assert profile.n_also_bought == 4
    assert profile.n_also_viewed == 7
    assert profile.n_usable_alternatives == 3


def test_indicator_keys_cover_every_profile_field_except_asin() -> None:
    """The stored indicator contract is exactly the dataclass minus ``asin``."""
    profile = build_profile(_product("K1"), _peers(), CONFIG)
    assert set(PROFILE_INDICATOR_KEYS) == set(profile.as_dict()) - {"asin"}


def test_to_indicators_round_trips_through_from_indicators() -> None:
    """A profile survives a JSON round-trip unchanged."""
    profile = build_profile(_product("T1"), _peers(), CONFIG)
    payload = json.loads(json.dumps(profile.to_indicators()))
    restored = ProductProfile.from_indicators(profile.asin, payload)
    assert restored == profile


def test_from_indicators_rejects_a_partial_payload() -> None:
    """A truncated blob fails loudly instead of defaulting the gaps."""
    profile = build_profile(_product("T2"), _peers(), CONFIG)
    payload = profile.to_indicators()
    del payload["price_percentile_category"]
    with pytest.raises(ValueError, match="missing keys"):
        ProductProfile.from_indicators("T2", payload)


def test_description_word_count_and_bullet_count_helpers() -> None:
    """The two counters agree with their documented definitions."""
    assert description_word_count(None) == 0
    assert description_word_count("   ") == 0
    assert description_word_count("a b  c") == 3
    assert feature_bullet_count(None) == 0
    assert feature_bullet_count([]) == 0
    assert feature_bullet_count(["x", "", "y"]) == 2


# ------------------------------------------------------------------------------
# Determinism
# ------------------------------------------------------------------------------


def test_two_runs_produce_identical_profiles() -> None:
    """Building twice over shuffled input yields byte-identical JSON."""
    first = build_profiles(FIXTURE_PRODUCTS, _peers(), CONFIG)
    shuffled = list(reversed(FIXTURE_PRODUCTS))
    second = build_profiles(shuffled, _peers(shuffled), CONFIG)
    assert first == second
    assert json.dumps([p.to_indicators() for p in first], sort_keys=True) == json.dumps(
        [p.to_indicators() for p in second], sort_keys=True
    )


def test_profiles_are_ordered_by_asin() -> None:
    """``build_profiles`` always returns ascending ASIN order."""
    profiles = build_profiles(FIXTURE_PRODUCTS, _peers(), CONFIG)
    assert [p.asin for p in profiles] == sorted(p.asin for p in profiles)


# ------------------------------------------------------------------------------
# Persistence (in-memory cache)
# ------------------------------------------------------------------------------


def _row(product: StubProduct) -> dict:
    """Render a StubProduct as a ``products`` insert payload.

    Args:
        product: Stub product.

    Returns:
        Insert payload with every NOT NULL column filled.
    """
    return {
        "asin": product.asin,
        "split": "train",
        "title": product.title,
        "title_clean": product.title_clean,
        "brand": product.brand,
        "brand_clean": product.brand_clean,
        "brand_missing": product.brand_missing,
        "price": product.price,
        "price_max": product.price_max,
        "price_status": product.price_status,
        "price_missing": product.price_missing,
        "rank": None,
        "rank_value": product.rank_value,
        "rank_category": product.rank_category,
        "rank_missing": product.rank_missing,
        "description": product.description,
        "features": product.features,
        "n_reviews": 0,
        "mean_rating": None,
        "first_review_time": None,
        "last_review_time": None,
        "image_url": None,
        "image_missing": True,
        "image_cached": False,
    }


@pytest.fixture()
def session() -> Iterator[Session]:
    """Seeded in-memory session over the five-product fixture set.

    Yields:
        Session with products and related edges loaded.
    """
    engine = create_engine_from_settings(db_url="sqlite://")
    init_db(engine)
    factory = create_session_factory(engine)
    with session_scope(factory) as session_obj:
        session_obj.execute(insert(Product), [_row(p) for p in FIXTURE_PRODUCTS])
        session_obj.execute(
            insert(RelatedProduct),
            [
                # CHEAP_A: two distinct targets, one linked twice -> 2 bought,
                # 1 viewed, 2 usable (both in subset); plus a self-edge that
                # must be ignored.
                {
                    "asin": CHEAP_A,
                    "related_asin": CHEAP_B,
                    "relation_type": "also_bought",
                    "in_subset": True,
                },
                {
                    "asin": CHEAP_A,
                    "related_asin": DEAR,
                    "relation_type": "also_bought",
                    "in_subset": True,
                },
                {
                    "asin": CHEAP_A,
                    "related_asin": DEAR,
                    "relation_type": "also_viewed",
                    "in_subset": True,
                },
                {
                    "asin": CHEAP_A,
                    "related_asin": CHEAP_A,
                    "relation_type": "also_bought",
                    "in_subset": True,
                },
                # DEAR links only outside the subset -> 1 distinct target, 0 usable.
                {
                    "asin": DEAR,
                    "related_asin": "OUTSIDE001",
                    "relation_type": "also_viewed",
                    "in_subset": False,
                },
            ],
        )
    open_session = create_session_factory(engine)()
    try:
        yield open_session
    finally:
        open_session.rollback()
        open_session.close()
        engine.dispose()


@pytest.fixture()
def meta_json(tmp_path: Path) -> Path:
    """Write a temp ``meta.json`` with a fixed build timestamp.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The directory holding ``meta.json``.
    """
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "category": "TestCategory",
                "built_at": "2026-10-17T11:11:15+00:00",
                "review_date_range": {"max": "2018-10-04T00:00:00+00:00"},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _settings_for(processed_dir: Path) -> Settings:
    """Build settings pointing ``processed_data_dir`` at a temp directory.

    Args:
        processed_dir: Directory holding ``meta.json``.

    Returns:
        A copy of the global settings with the path overridden.
    """
    base = get_settings()
    return base.model_copy(
        update={"paths": base.paths.model_copy(update={"processed_data_dir": processed_dir})}
    )


def test_read_build_timestamp_is_naive_utc_from_meta(session: Session, meta_json: Path) -> None:
    """The build timestamp is parsed from meta.json as naive UTC."""
    timestamp = read_build_timestamp(_settings_for(meta_json))
    assert timestamp == datetime(2026, 10, 17, 11, 11, 15)
    assert timestamp.tzinfo is None


def test_read_build_timestamp_refuses_to_fall_back_to_the_clock(tmp_path: Path) -> None:
    """A missing meta.json raises instead of silently stamping 'now'."""
    with pytest.raises(DataError, match="meta.json"):
        read_build_timestamp(_settings_for(tmp_path))


def test_write_is_idempotent_and_stamps_the_build_time(session: Session, meta_json: Path) -> None:
    """Re-writing the same profiles leaves exactly one identical row each."""
    peers = _peers()
    profiles = build_profiles(FIXTURE_PRODUCTS, peers, CONFIG)
    timestamp = read_build_timestamp(_settings_for(meta_json))

    assert write_product_features(session, profiles, timestamp) == 5
    session.commit()
    first = {asin: read_indicators(session, asin) for asin in list_profiled_asins(session)}

    assert write_product_features(session, profiles, timestamp) == 5
    session.commit()
    second = {asin: read_indicators(session, asin) for asin in list_profiled_asins(session)}

    assert list_profiled_asins(session) == sorted(p.asin for p in FIXTURE_PRODUCTS)
    assert first == second
    row = session.get(ProductFeature, CHEAP_A)
    assert row is not None
    assert row.computed_at == timestamp


def test_write_rejects_a_timezone_aware_timestamp(session: Session) -> None:
    """A tz-aware timestamp is refused so the storage contract stays naive."""
    profiles = build_profiles(FIXTURE_PRODUCTS, _peers(), CONFIG)
    with pytest.raises(ValueError, match="naive UTC"):
        write_product_features(session, profiles, datetime.now(UTC))


def test_related_counts_are_computed_per_distinct_target(session: Session) -> None:
    """The grouped query matches the documented distinct-target contract."""
    from munasib.db.repositories import RelatedRepository

    counts = RelatedRepository(session).count_related()
    assert counts[CHEAP_A] == RelatedCounts(also_bought=2, also_viewed=1, usable_alternatives=2)
    assert counts[DEAR] == RelatedCounts(also_bought=0, also_viewed=1, usable_alternatives=0)
    # UNPRICED / UNBRANDED / CHEAP_B have no edges and are simply absent.
    assert CHEAP_B not in counts


def test_product_without_related_rows_stores_zero_counts(session: Session, meta_json: Path) -> None:
    """A product with no edges at all gets zeros, and profiles fine."""
    from munasib.db.repositories import RelatedRepository

    peers = _peers()
    counts = RelatedRepository(session).count_related([UNPRICED])
    profiles = build_profiles([p for p in FIXTURE_PRODUCTS if p.asin == UNPRICED], peers, CONFIG, counts)
    assert profiles[0].n_usable_alternatives == 0
    write_product_features(session, profiles, read_build_timestamp(_settings_for(meta_json)))
    session.commit()
    stored = read_profile(session, UNPRICED)
    assert stored is not None
    assert stored.n_also_bought == 0


def test_read_profile_returns_none_for_an_unknown_asin(session: Session) -> None:
    """An unprofiled product reads back as None, not as a default profile."""
    assert read_profile(session, "NOSUCH0000") is None
    assert read_indicators(session, "NOSUCH0000") is None


def test_read_profiles_can_be_restricted_to_asins(session: Session, meta_json: Path) -> None:
    """``read_profiles`` honours an ASIN filter and stays ASIN-ordered."""
    profiles = build_profiles(FIXTURE_PRODUCTS, _peers(), CONFIG)
    write_product_features(session, profiles, read_build_timestamp(_settings_for(meta_json)))
    session.commit()
    subset = read_profiles(session, [DEAR, CHEAP_A])
    assert [p.asin for p in subset] == [CHEAP_A, DEAR]
    assert len(read_profiles(session)) == 5


# ------------------------------------------------------------------------------
# Real database (skips when absent) - invariants only, no hard-coded numbers
# ------------------------------------------------------------------------------


def test_real_database_percentiles_are_in_bounds_and_flagged() -> None:
    """Against the real cache: bounds hold and missing prices stay None."""
    db_path = Path("data/db/munasib.sqlite")
    if not db_path.exists():
        pytest.skip(f"real serving cache not present at {db_path}")

    from munasib.core.config import get_settings
    from munasib.db.repositories import ProductRepository
    from munasib.db.session import init_db

    engine = create_engine_from_settings(get_settings())
    init_db(engine)
    factory = create_session_factory(engine)
    try:
        with session_scope(factory) as session:
            products = ProductRepository(session).list_all_products()
            assert products, "expected a non-empty candidate set"
            source_paths = {
                row.asin: row.category_path
                for row in products
                if row.category_path is not None
            }
            peers = PeerStats.from_products(products, get_settings().product_intel)
            profiles = build_profiles(products, peers, get_settings().product_intel)
    finally:
        engine.dispose()

    priced = [p for p in profiles if not p.price_missing]
    assert priced, "expected at least one priced product"
    priced_by_asin = {p.asin: p for p in priced}
    for profile in profiles:
        for value in (
            profile.price_percentile_category,
            profile.price_percentile_brand,
            profile.rank_percentile,
        ):
            if value is not None:
                assert 0.0 <= value <= 1.0
        if profile.price_missing:
            assert profile.price_percentile_category is None
            assert profile.price_percentile_category_missing is True
        if profile.rank_missing:
            assert profile.rank_percentile is None
            assert profile.rank_percentile_missing is True
        if profile.brand_missing:
            assert profile.brand is None
            assert profile.price_percentile_brand is None
        # A completeness score is never out of range.
        assert 0.0 <= profile.description_completeness <= 1.0
        assert 0.0 <= profile.feature_completeness <= 1.0
        assert 0.0 <= profile.metadata_completeness <= 1.0
        assert set(profile.metadata_missing_fields) <= set(CONFIG.metadata_fields)
        # Peer-group invariants: the group is always named and self-describing.
        assert profile.price_peer_group
        assert profile.price_peer_count >= 0
        source = source_paths.get(profile.asin)
        if source is None:
            # No breadcrumb published -> the whole candidate set is the group.
            assert profile.price_peer_group == ALL_PRODUCTS_GROUP
            assert profile.price_peer_fallback is True
        elif not profile.price_peer_fallback:
            assert profile.price_peer_group == source  # the leaf itself
        else:
            # A fallback group is always a shorter prefix of the product's own
            # breadcrumb (never a sibling node, never an unrelated string).
            assert profile.price_peer_group in category_prefixes(source)[1:]
        if profile.asin in priced_by_asin:
            # A priced product's group always contains it, and the group size
            # matches the peer array the percentile was measured against.
            assert profile.price_peer_count >= 1

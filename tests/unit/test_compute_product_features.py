"""Tests for ``scripts/compute_product_features.py`` (product profiles CLI).

Hermetic: a temporary SQLite cache holds a handful of products and related
edges, and a temporary ``meta.json`` fixes the build timestamp, so every count
and percentile asserted here is computed by the code under test - no network,
no real ``data/`` artifact.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import delete as sa_delete
from sqlalchemy import func, insert
from sqlalchemy import select as sa_select
from sqlalchemy.orm import Session

from munasib.core.config import Settings, get_settings
from munasib.db.models import Product, ProductFeature, RelatedProduct
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)
from munasib.product_intel.peers import PeerStats
from munasib.product_intel.profile import PROFILE_INDICATOR_KEYS, build_profiles

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compute_product_features.py"

#: Four priced/ranked products across two brands plus one unpriced, unbranded.
P_A = "B0000000PA"
P_B = "B0000000PB"
P_C = "B0000000PC"
P_D = "B0000000PD"
P_E = "B0000000PE"

BUILD_TIMESTAMP = "2026-10-17T11:11:15+00:00"


def _load_module():
    """Import ``scripts/compute_product_features.py`` as a module.

    The module is registered in ``sys.modules`` before execution, because the
    script defines a ``@dataclass`` and the dataclass machinery resolves type
    annotations through ``sys.modules[cls.__module__]``.

    Returns:
        The loaded script module.
    """
    spec = importlib.util.spec_from_file_location("munasib_compute_features_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _product_row(
    asin: str,
    *,
    brand: str | None,
    price: float | None,
    rank_value: int | None,
    rank_category: str | None,
    description: str | None = "A perfectly ordinary description.",
    features: list[str] | None = None,
) -> dict:
    """Build a minimal products insert payload.

    Args:
        asin: Product ASIN.
        brand: Brand (None = unbranded).
        price: Price (None = unpriced).
        rank_value: Sales rank (None = unranked).
        rank_category: Rank category.
        description: Cleaned description.
        features: Feature bullets.

    Returns:
        Insert payload with every NOT NULL column filled.
    """
    return {
        "asin": asin,
        "split": "train",
        "title": f"Product {asin}",
        "title_clean": f"product {asin}",
        "brand": brand,
        "brand_clean": brand.lower() if brand else None,
        "brand_missing": brand is None,
        "price": price,
        "price_max": None,
        "price_status": "single" if price is not None else "missing",
        "price_missing": price is None,
        "rank": None,
        "rank_value": rank_value,
        "rank_category": rank_category,
        "rank_missing": rank_value is None,
        "description": description,
        "features": features if features is not None else ["bullet one", "bullet two"],
        "n_reviews": 0,
        "mean_rating": None,
        "first_review_time": None,
        "last_review_time": None,
        "image_url": None,
        "image_missing": True,
        "image_cached": False,
    }


def _stub(asin: str, **overrides: object) -> Product:
    """Build an in-memory ORM ``Product`` stand-in (no session needed).

    Args:
        asin: Product ASIN.
        **overrides: Column values overriding the defaults.

    Returns:
        A detached Product instance usable as a ``ProductLike``.
    """
    defaults: dict = {
        "asin": asin,
        "split": "train",
        "title": f"Product {asin}",
        "title_clean": f"product {asin}",
        "brand": "Acme",
        "brand_clean": "acme",
        "brand_missing": False,
        "price": 10.0,
        "price_max": None,
        "price_status": "single",
        "price_missing": False,
        "rank_value": 1,
        "rank_category": "Widgets",
        "rank_missing": False,
        "description": "A perfectly ordinary description.",
        "features": ["bullet one", "bullet two"],
        "n_reviews": 0,
        "image_missing": True,
    }
    defaults.update(overrides)
    return Product(**defaults)


PRODUCTS = [
    _product_row(P_A, brand="Acme", price=10.0, rank_value=1, rank_category="Widgets"),
    _product_row(P_B, brand="Acme", price=20.0, rank_value=2, rank_category="Widgets"),
    _product_row(P_C, brand="Acme", price=30.0, rank_value=3, rank_category="Widgets"),
    _product_row(P_D, brand="Beta", price=40.0, rank_value=4, rank_category="Widgets"),
    _product_row(P_E, brand=None, price=None, rank_value=None, rank_category=None),
]

EDGES = [
    # P_A: two distinct targets (P_B linked twice) -> 1 usable alternative.
    {"asin": P_A, "related_asin": P_B, "relation_type": "also_bought", "in_subset": True},
    {"asin": P_A, "related_asin": P_B, "relation_type": "also_viewed", "in_subset": True},
]


@pytest.fixture()
def cache(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """Create a temp serving cache and a temp processed dir with meta.json.

    Args:
        tmp_path: Pytest temporary directory.

    Yields:
        Tuple ``(db_path, processed_dir)``.
    """
    db_path = tmp_path / "cache.sqlite"
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    init_db(engine)
    with session_scope(create_session_factory(engine)) as session:
        session.execute(insert(Product), PRODUCTS)
        session.execute(insert(RelatedProduct), EDGES)
    engine.dispose()

    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "meta.json").write_text(
        json.dumps(
            {
                "category": "TestCategory",
                "built_at": BUILD_TIMESTAMP,
                "review_date_range": {"max": "2018-10-04T00:00:00+00:00"},
            }
        ),
        encoding="utf-8",
    )
    yield db_path, processed


def _config(tmp_path: Path, db_path: Path, processed: Path) -> Path:
    """Write a minimal config pointing at the temp cache and processed dir.

    Args:
        tmp_path: Pytest temporary directory.
        db_path: SQLite cache to use.
        processed: Processed data dir holding meta.json.

    Returns:
        The config path.
    """
    config = tmp_path / "config.yaml"
    config.write_text(
        f'paths:\n  db_path: "{db_path.as_posix()}"\n'
        f'  processed_data_dir: "{processed.as_posix()}"\n',
        encoding="utf-8",
    )
    return config


def _stored(settings: Settings, asin: str) -> tuple[dict, object]:
    """Read one stored indicator blob plus its computed_at from the cache.

    Args:
        settings: Settings pointing at the temp cache.
        asin: Product ASIN.

    Returns:
        Tuple ``(indicators, computed_at)``.
    """
    engine = create_engine_from_settings(settings)
    factory = create_session_factory(engine)
    try:
        with session_scope(factory) as session:
            row = session.get(ProductFeature, asin)
            assert row is not None, f"no product_features row for {asin}"
            assert row.indicators is not None
            return dict(row.indicators), row.computed_at
    finally:
        engine.dispose()


# ------------------------------------------------------------------------------
# Summary aggregation (pure)
# ------------------------------------------------------------------------------


def test_summarize_counts_missing_values_and_percentiles() -> None:
    """The summary counts the ``*_missing`` flags and omits all-None percentiles."""
    module = _load_module()
    products = [
        _stub("X1", brand="Acme", brand_clean="acme", price=1.0, rank_value=1, rank_category="C"),
        _stub(
            "X2",
            brand="Acme",
            brand_clean="acme",
            price=None,
            price_missing=True,
            rank_value=None,
            rank_category=None,
            rank_missing=True,
        ),
    ]
    peers = PeerStats.from_products(products, get_settings().product_intel)
    profiles = build_profiles(products, peers, get_settings().product_intel)
    summary = module.summarize(profiles)
    assert summary.written == 2
    assert summary.without_price == 1
    assert summary.without_rank == 1
    assert summary.without_brand == 0
    # Only X1 is priced, and it is the only member of its price set, so its
    # category percentile is None (no peers) and the stat is omitted.
    assert "price_percentile_category" not in summary.percentile_stats


def test_summarize_on_an_empty_batch_is_all_zero() -> None:
    """An empty profile list summarises to zeros, not to an error."""
    module = _load_module()
    summary = module.summarize([])
    assert summary.written == 0
    assert summary.percentile_stats == {}
    assert any("profiles written" in line for line in summary.lines())


# ------------------------------------------------------------------------------
# select_targets
# ------------------------------------------------------------------------------


@contextmanager
def _session_for(db_path: Path) -> Iterator[Session]:
    """Open a session over a temp cache.

    Args:
        db_path: SQLite cache path.

    Yields:
        Open session.
    """
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    session = create_session_factory(engine)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


def test_select_targets_applies_the_limit_in_asin_order(cache: tuple[Path, Path]) -> None:
    """``--limit N`` takes the first N products by ASIN."""
    module = _load_module()
    db_path, _ = cache
    with _session_for(db_path) as session:
        selected = module.select_targets(session, 2, None)
        assert [row.asin for row in selected] == [P_A, P_B]


def test_select_targets_honours_explicit_asins(cache: tuple[Path, Path]) -> None:
    """``--asins`` profiles exactly those products, in ASIN order."""
    module = _load_module()
    db_path, _ = cache
    with _session_for(db_path) as session:
        selected = module.select_targets(session, None, [P_D, P_A])
        assert [row.asin for row in selected] == [P_A, P_D]


def test_select_targets_rejects_an_unknown_asin(cache: tuple[Path, Path]) -> None:
    """An ASIN missing from the cache fails loudly instead of shrinking the run."""
    module = _load_module()
    db_path, _ = cache
    with _session_for(db_path) as session:
        with pytest.raises(Exception, match="NOTINTHERE"):
            module.select_targets(session, None, ["NOTINTHERE"])


def test_select_targets_rejects_a_negative_limit(cache: tuple[Path, Path]) -> None:
    """A negative limit is a usage error, not an empty run."""
    module = _load_module()
    db_path, _ = cache
    with _session_for(db_path) as session:
        with pytest.raises(ValueError, match="non-negative"):
            module.select_targets(session, -1, None)


# ------------------------------------------------------------------------------
# CLI end-to-end
# ------------------------------------------------------------------------------


def test_main_writes_every_profile_with_the_build_timestamp(
    tmp_path: Path, cache: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """A full run stores all five profiles stamped with meta.json's built_at."""
    module = _load_module()
    db_path, processed = cache
    config = _config(tmp_path, db_path, processed)
    assert module.main(["--config", str(config)]) == 0

    out = capsys.readouterr().out
    assert "profiles written            : 5" in out
    assert "without price               : 1" in out
    assert "without rank                : 1" in out
    assert "without brand               : 1" in out

    settings = Settings.model_validate(
        {"paths": {"db_path": str(db_path), "processed_data_dir": str(processed)}}
    )
    indicators, computed_at = _stored(settings, P_A)
    assert set(indicators) == set(PROFILE_INDICATOR_KEYS)
    assert computed_at.isoformat() == "2026-10-17T11:11:15"


def test_main_is_deterministic_across_two_runs(
    tmp_path: Path, cache: tuple[Path, Path]
) -> None:
    """Two runs over the same cache write byte-identical indicator blobs."""
    module = _load_module()
    db_path, processed = cache
    config = _config(tmp_path, db_path, processed)
    settings = Settings.model_validate(
        {"paths": {"db_path": str(db_path), "processed_data_dir": str(processed)}}
    )

    assert module.main(["--config", str(config)]) == 0
    first = {asin: _stored(settings, asin) for asin in (P_A, P_E)}
    assert module.main(["--config", str(config)]) == 0
    second = {asin: _stored(settings, asin) for asin in (P_A, P_E)}
    assert first == second


def test_main_records_the_real_related_counts(tmp_path: Path, cache: tuple[Path, Path]) -> None:
    """The stored profile carries the distinct-target counts from the edges."""
    module = _load_module()
    db_path, processed = cache
    config = _config(tmp_path, db_path, processed)
    assert module.main(["--config", str(config)]) == 0

    settings = Settings.model_validate(
        {"paths": {"db_path": str(db_path), "processed_data_dir": str(processed)}}
    )
    indicators, _ = _stored(settings, P_A)
    # P_B is linked by both edge types -> one distinct target, one usable.
    assert indicators["n_also_bought"] == 1
    assert indicators["n_also_viewed"] == 1
    assert indicators["n_usable_alternatives"] == 1
    # P_E has no edges at all -> zeros, never an error.
    lonely, _ = _stored(settings, P_E)
    assert lonely["n_usable_alternatives"] == 0
    assert lonely["price"] is None
    assert lonely["price_missing"] is True


def test_main_with_limit_only_stores_that_slice(
    tmp_path: Path, cache: tuple[Path, Path]
) -> None:
    """``--limit`` stores only the selected profiles, but keeps full peer sets."""
    module = _load_module()
    db_path, processed = cache
    config = _config(tmp_path, db_path, processed)
    assert module.main(["--config", str(config), "--limit", "2"]) == 0

    settings = Settings.model_validate(
        {"paths": {"db_path": str(db_path), "processed_data_dir": str(processed)}}
    )
    engine = create_engine_from_settings(settings)
    factory = create_session_factory(engine)
    try:
        with session_scope(factory) as session:
            stored = sorted(session.scalars(sa_select(ProductFeature.asin)))
    finally:
        engine.dispose()
    assert stored == [P_A, P_B]


def test_main_with_asins_stores_only_those(
    tmp_path: Path, cache: tuple[Path, Path]
) -> None:
    """``--asins`` stores exactly the requested profiles."""
    module = _load_module()
    db_path, processed = cache
    config = _config(tmp_path, db_path, processed)
    assert module.main(["--config", str(config), "--asins", P_E, P_A]) == 0

    settings = Settings.model_validate(
        {"paths": {"db_path": str(db_path), "processed_data_dir": str(processed)}}
    )
    engine = create_engine_from_settings(settings)
    factory = create_session_factory(engine)
    try:
        with session_scope(factory) as session:
            stored = sorted(session.scalars(sa_select(ProductFeature.asin)))
    finally:
        engine.dispose()
    assert stored == [P_A, P_E]


def test_main_returns_one_on_an_unknown_asin(
    tmp_path: Path, cache: tuple[Path, Path]
) -> None:
    """A typo in ``--asins`` exits 1 and stores nothing."""
    module = _load_module()
    db_path, processed = cache
    config = _config(tmp_path, db_path, processed)
    assert module.main(["--config", str(config), "--asins", "NOTINTHERE"]) == 1

    settings = Settings.model_validate(
        {"paths": {"db_path": str(db_path), "processed_data_dir": str(processed)}}
    )
    engine = create_engine_from_settings(settings)
    factory = create_session_factory(engine)
    try:
        with session_scope(factory) as session:
            assert session.scalar(sa_select(func.count()).select_from(ProductFeature)) == 0
    finally:
        engine.dispose()


def test_main_returns_one_without_meta_json(
    tmp_path: Path, cache: tuple[Path, Path]
) -> None:
    """A missing meta.json exits 1 - the CLI never falls back to the wall clock."""
    module = _load_module()
    db_path, processed = cache
    (processed / "meta.json").unlink()
    config = _config(tmp_path, db_path, processed)
    assert module.main(["--config", str(config)]) == 1


def test_main_returns_one_on_an_empty_cache(
    tmp_path: Path, cache: tuple[Path, Path]
) -> None:
    """An empty product table exits 1 naming the build commands."""
    module = _load_module()
    db_path, processed = cache
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    with session_scope(create_session_factory(engine)) as session:
        session.execute(sa_delete(RelatedProduct))
        session.execute(sa_delete(Product))
    engine.dispose()
    config = _config(tmp_path, db_path, processed)
    assert module.main(["--config", str(config)]) == 1


def test_main_returns_one_on_a_bad_config(tmp_path: Path) -> None:
    """A missing config file exits 1 without touching any cache."""
    module = _load_module()
    assert module.main(["--config", str(tmp_path / "nope.yaml")]) == 1


def test_parse_args_defaults() -> None:
    """The default config path is the master config and no filter is applied."""
    module = _load_module()
    args = module.parse_args([])
    assert args.config == "configs/config.yaml"
    assert args.limit is None
    assert args.asins is None


def test_parse_args_reads_limit_and_asins() -> None:
    """``--limit`` and ``--asins`` parse into the documented shapes."""
    module = _load_module()
    args = module.parse_args(["--limit", "3", "--asins", "B0000000PA", "B0000000PB"])
    assert args.limit == 3
    assert args.asins == ["B0000000PA", "B0000000PB"]

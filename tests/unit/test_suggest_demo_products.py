"""Tests for ``scripts/suggest_demo_products.py`` (demo ASIN candidates).

Hermetic: a temporary SQLite cache holds a handful of products, reviews and
related edges, so every share and count asserted here is computed by the code
under test - no network, no real ``data/`` artifact.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import insert

from munasib.core.config import DemoProductsConfig
from munasib.db.models import Product, RelatedProduct, Review
from munasib.db.session import create_engine_from_settings, create_session_factory, init_db

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "suggest_demo_products.py"

#: A product with an image URL and enough alternatives for every group.
CANDIDATE_W1 = "B0000000W1"
CANDIDATE_W2 = "B0000000W2"
CANDIDATE_P1 = "B0000000P1"
CANDIDATE_P2 = "B0000000P2"
CANDIDATE_L1 = "B0000000L1"
FEW_ALTERNATIVES = "B0000000F1"
NO_IMAGE = "B0000000N1"


def _load_module():
    """Import ``scripts/suggest_demo_products.py`` as a module.

    Returns:
        The loaded script module.
    """
    spec = importlib.util.spec_from_file_location("munasib_suggest_demo_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(
    asin: str,
    *,
    n_reviews: int,
    mean_rating: float | None,
    five_star_share: float,
    low_star_share: float,
    usable_alternatives: int,
) -> dict[str, object]:
    """Build one ``load_candidates``-shaped row for :func:`select_groups`.

    Args:
        asin: Product ASIN.
        n_reviews: Stored review count.
        mean_rating: Stored mean rating or None.
        five_star_share: Share of 5-star reviews.
        low_star_share: Share of 1-2 star reviews.
        usable_alternatives: Distinct in-subset alternatives.

    Returns:
        Candidate dict.
    """
    return {
        "asin": asin,
        "title": f"Product {asin}",
        "n_reviews": n_reviews,
        "mean_rating": mean_rating,
        "five_star_share": five_star_share,
        "low_star_share": low_star_share,
        "usable_alternatives": usable_alternatives,
    }


def _product_row(
    asin: str, *, n_reviews: int, mean_rating: float | None, image_url: str | None
) -> dict:
    """Build a minimal products insert payload.

    Args:
        asin: Product ASIN.
        n_reviews: Stored review count.
        mean_rating: Stored mean rating or None.
        image_url: Source image URL or None.

    Returns:
        Insert payload with the NOT NULL columns filled.
    """
    return {
        "asin": asin,
        "split": "train",
        "brand_missing": False,
        "price_missing": False,
        "rank_missing": True,
        "n_reviews": n_reviews,
        "mean_rating": mean_rating,
        "image_url": image_url,
        "image_missing": image_url is None,
        "image_cached": False,
    }


def _review_row(asin: str, index: int, rating: float) -> dict:
    """Build one reviews insert payload.

    Args:
        asin: Owning product ASIN.
        index: Row index (makes the review id unique).
        rating: Star rating.

    Returns:
        Insert payload with the NOT NULL columns filled.
    """
    review_id = f"{index:016d}"
    return {
        "review_id": review_id,
        "asin": asin,
        "split": "train",
        "reviewer_id": f"user-{index}",
        "rating": rating,
        "review_text_raw": "body",
        "review_text_clean": "body",
        "summary_raw": "head",
        "summary_clean": "head",
        "has_text": True,
        "text_length": 4,
        "helpful_votes": 0,
        "verified": True,
        "review_time": datetime(2018, 1, 1, tzinfo=None),
        "year": 2018,
        "month": 1,
        "sentiment_label_weak": None,
    }


def _cache(tmp_path: Path, products: list[dict], reviews: list[dict], edges: list[dict]) -> Path:
    """Create a temporary serving cache and fill it.

    Args:
        tmp_path: Pytest temporary directory.
        products: Product insert payloads.
        reviews: Review insert payloads.
        edges: Related-product insert payloads.

    Returns:
        The database path.
    """
    db_path = tmp_path / "cache.sqlite"
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    init_db(engine)
    with create_session_factory(engine)() as session:
        if products:
            session.execute(insert(Product), products)
        if reviews:
            session.execute(insert(Review), reviews)
        if edges:
            session.execute(insert(RelatedProduct), edges)
        session.commit()
    engine.dispose()
    return db_path


def _config(tmp_path: Path, db_path: Path) -> Path:
    """Write a minimal config pointing at the temp cache.

    Args:
        tmp_path: Pytest temporary directory.
        db_path: SQLite cache to use.

    Returns:
        The config path.
    """
    config = tmp_path / "config.yaml"
    config.write_text(f'paths:\n  db_path: "{db_path.as_posix()}"\n', encoding="utf-8")
    return config


# ------------------------------------------------------------------------------
# Group selection (pure)
# ------------------------------------------------------------------------------

CANDIDATES = [
    _candidate(CANDIDATE_W1, n_reviews=120, mean_rating=4.8, five_star_share=0.80,
               low_star_share=0.05, usable_alternatives=3),
    _candidate(CANDIDATE_W2, n_reviews=300, mean_rating=4.6, five_star_share=0.90,
               low_star_share=0.02, usable_alternatives=5),
    _candidate(CANDIDATE_P1, n_reviews=100, mean_rating=3.2, five_star_share=0.30,
               low_star_share=0.30, usable_alternatives=4),
    _candidate(CANDIDATE_P2, n_reviews=200, mean_rating=3.0, five_star_share=0.26,
               low_star_share=0.26, usable_alternatives=3),
    _candidate(CANDIDATE_L1, n_reviews=40, mean_rating=2.5, five_star_share=0.10,
               low_star_share=0.70, usable_alternatives=3),
    _candidate(FEW_ALTERNATIVES, n_reviews=500, mean_rating=4.9, five_star_share=0.95,
               low_star_share=0.01, usable_alternatives=2),
    _candidate(NO_IMAGE, n_reviews=90, mean_rating=None, five_star_share=0.50,
               low_star_share=0.50, usable_alternatives=9),
]


def test_select_groups_applies_the_configured_thresholds() -> None:
    """Each group uses its configured thresholds; ordering is by review count."""
    module = _load_module()
    groups = module.select_groups(CANDIDATES, DemoProductsConfig())
    assert [item["asin"] for item in groups["well-reviewed"]] == [CANDIDATE_W2, CANDIDATE_W1]
    # FEW_ALTERNATIVES has the best rating but too few usable alternatives.
    assert FEW_ALTERNATIVES not in {item["asin"] for item in groups["well-reviewed"]}
    assert [item["asin"] for item in groups["polarized"]] == [
        CANDIDATE_P2,
        CANDIDATE_P1,
        NO_IMAGE,
    ]
    # Group membership is not exclusive: P2 is both polarized and low-rated.
    assert [item["asin"] for item in groups["low-rated"]] == [CANDIDATE_P2, CANDIDATE_L1]
    # A share-based group does not need a mean rating (NO_IMAGE has none) ...
    assert NO_IMAGE in {item["asin"] for item in groups["polarized"]}
    # ... but the rating-based groups do exclude an unrated product.
    assert NO_IMAGE not in {item["asin"] for item in groups["well-reviewed"]}
    assert NO_IMAGE not in {item["asin"] for item in groups["low-rated"]}


def test_select_groups_truncates_to_per_group() -> None:
    """``per_group`` caps every group."""
    module = _load_module()
    groups = module.select_groups(CANDIDATES, DemoProductsConfig(per_group=1))
    assert all(len(members) == 1 for members in groups.values())
    assert groups["well-reviewed"][0]["asin"] == CANDIDATE_W2


def test_select_groups_can_return_empty_groups() -> None:
    """A group with no match prints as empty rather than being dropped."""
    module = _load_module()
    groups = module.select_groups([], DemoProductsConfig())
    assert set(groups) == {"well-reviewed", "polarized", "low-rated"}
    assert all(members == [] for members in groups.values())


def test_select_groups_requires_alternatives() -> None:
    """The usable-alternatives floor comes from config and is enforced everywhere."""
    module = _load_module()
    strict = DemoProductsConfig(min_usable_alternatives=10)
    groups = module.select_groups(CANDIDATES, strict)
    assert all(members == [] for members in groups.values())


# ------------------------------------------------------------------------------
# Loading candidates from the cache
# ------------------------------------------------------------------------------


def test_load_candidates_computes_shares_and_alternative_counts(tmp_path: Path) -> None:
    """Shares come from the stored reviews; only in-subset distinct targets count."""
    module = _load_module()
    products = [
        _product_row(CANDIDATE_W1, n_reviews=4, mean_rating=4.0,
                     image_url="https://example.com/w1.jpg"),
        _product_row(NO_IMAGE, n_reviews=1, mean_rating=1.0, image_url=None),
    ]
    reviews = [
        _review_row(CANDIDATE_W1, 1, 5.0),
        _review_row(CANDIDATE_W1, 2, 5.0),
        _review_row(CANDIDATE_W1, 3, 1.0),
        _review_row(CANDIDATE_W1, 4, 3.0),
        _review_row(NO_IMAGE, 5, 1.0),
    ]
    edges = [
        {"asin": CANDIDATE_W1, "related_asin": "B0000000R1", "relation_type": "also_bought",
         "in_subset": True},
        {"asin": CANDIDATE_W1, "related_asin": "B0000000R2", "relation_type": "also_bought",
         "in_subset": True},
        {"asin": CANDIDATE_W1, "related_asin": "B0000000R2", "relation_type": "also_viewed",
         "in_subset": True},  # duplicate pair, another relation type -> one alternative
        {"asin": CANDIDATE_W1, "related_asin": "OUTSIDE99", "relation_type": "also_viewed",
         "in_subset": False},  # out of subset -> not usable
    ]
    db_path = _cache(tmp_path, products, reviews, edges)
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    try:
        candidates = module.load_candidates(engine)
    finally:
        engine.dispose()

    # NO_IMAGE has no image URL, so it is not a candidate at all.
    assert [candidate["asin"] for candidate in candidates] == [CANDIDATE_W1]
    only = candidates[0]
    assert only["n_reviews"] == 4
    assert only["mean_rating"] == pytest.approx(4.0)
    assert only["five_star_share"] == pytest.approx(0.5)  # two of four
    assert only["low_star_share"] == pytest.approx(0.25)  # one of four
    assert only["usable_alternatives"] == 2  # R1 + R2, deduplicated; OUTSIDE99 excluded


def test_load_candidates_zero_alternatives_is_zero_not_missing(tmp_path: Path) -> None:
    """A product with no related edges reports 0 usable alternatives."""
    module = _load_module()
    db_path = _cache(
        tmp_path,
        [_product_row(CANDIDATE_L1, n_reviews=1, mean_rating=2.0,
                      image_url="https://example.com/l1.jpg")],
        [_review_row(CANDIDATE_L1, 1, 2.0)],
        [],
    )
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    try:
        candidates = module.load_candidates(engine)
    finally:
        engine.dispose()
    assert candidates[0]["usable_alternatives"] == 0
    assert candidates[0]["low_star_share"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def test_main_prints_every_group(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI prints all three groups, the candidate count and the suggested ASINs."""
    module = _load_module()
    products = [
        _product_row(asin, n_reviews=n, mean_rating=m, image_url=f"https://example.com/{asin}.jpg")
        for asin, n, m in [
            (CANDIDATE_W1, 120, 4.8),
            (CANDIDATE_P1, 100, 3.2),
            (CANDIDATE_L1, 40, 2.5),
        ]
    ]
    reviews: list[dict] = []
    edges: list[dict] = []
    index = 0
    for asin, five, low, neutral in [
        (CANDIDATE_W1, 10, 1, 1),
        (CANDIDATE_P1, 30, 30, 30),
        (CANDIDATE_L1, 1, 30, 0),
    ]:
        for _ in range(five):
            index += 1
            reviews.append(_review_row(asin, index, 5.0))
        for _ in range(low):
            index += 1
            reviews.append(_review_row(asin, index, 1.0))
        for _ in range(neutral):
            index += 1
            reviews.append(_review_row(asin, index, 3.0))
        for target in ("B0000000R1", "B0000000R2", "B0000000R3", "B0000000R4"):
            edges.append(
                {
                    "asin": asin,
                    "related_asin": target,
                    "relation_type": "also_bought",
                    "in_subset": True,
                }
            )
    db_path = _cache(tmp_path, products, reviews, edges)
    config = _config(tmp_path, db_path)
    assert module.main(["--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "## well-reviewed" in out and "## polarized" in out and "## low-rated" in out
    assert out.count(CANDIDATE_W1) >= 1  # appears in the table and in the log line
    assert "none match the configured thresholds" not in out


def test_main_returns_one_on_an_empty_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No candidate at all is an error naming the build commands, exit code 1."""
    module = _load_module()
    db_path = _cache(tmp_path, [], [], [])
    config = _config(tmp_path, db_path)
    assert module.main(["--config", str(config)]) == 1


def test_main_returns_one_on_a_bad_config(tmp_path: Path) -> None:
    """A missing config file exits 1 without touching the cache."""
    module = _load_module()
    assert module.main(["--config", str(tmp_path / "nope.yaml")]) == 1


def test_parse_args_defaults() -> None:
    """The default config path is the master config."""
    module = _load_module()
    assert module.parse_args([]).config == "configs/config.yaml"

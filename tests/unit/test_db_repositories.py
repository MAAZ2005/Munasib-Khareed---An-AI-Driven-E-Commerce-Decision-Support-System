"""Tests for the typed repositories over the serving cache.

Seeds a tiny in-memory database (six products, four reviews, three related
edges) and exercises every repository contract: point gets, the
``ProductNotFoundError`` domain error, case-insensitive search with stable
pagination and totals, review filters (including ``verified`` NULL versus
False), deterministic peer sampling, related-edge filtering and analysis
upserts. Hermetic: no network, no GPU, no real parquet files.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, insert, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from munasib.core.constants import (
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABEL_POSITIVE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from munasib.core.exceptions import ProductNotFoundError
from munasib.db.models import AnalysisResult, Product, RelatedProduct, Review
from munasib.db.repositories import (
    AnalysisRepository,
    ProductRepository,
    RelatedRepository,
    ReviewRepository,
)
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)

PRODUCTS = [
    ("B000000001", "Acme Widget Pro", "Acme", SPLIT_TRAIN),
    ("B000000002", "Beta Widget", "BetaCorp", SPLIT_TRAIN),
    ("B000000003", "Gamma Gadget", "Acme", SPLIT_VAL),
    ("B000000004", "delta 100% widget", None, SPLIT_TEST),
    ("B000000005", "Omega Widget", "acme industries", SPLIT_TRAIN),
    ("B000000006", "Super 100 machine", None, SPLIT_TRAIN),
]


def _product_row(asin: str, title: str, brand: str | None, split: str) -> dict:
    """Build one products-table insert payload.

    Args:
        asin: Product ASIN.
        title: Product title.
        brand: Brand or None (brand_missing tracks it).
        split: Product-level split name.

    Returns:
        Insert payload with proper NULLs for missing brand/price/rank.
    """
    return {
        "asin": asin,
        "split": split,
        "title": title,
        "title_clean": title.lower() if title else None,
        "brand": brand,
        "brand_clean": brand.lower() if brand else None,
        "brand_missing": brand is None,
        "price": 19.99,
        "price_max": None,
        "price_status": "ok" if brand is not None else "missing",
        "price_missing": False,
        "rank": None,
        "rank_value": None,
        "rank_category": None,
        "rank_missing": True,
        "description": None,
        "features": None,
        "n_reviews": 0,
        "mean_rating": 4.0,
        "first_review_time": datetime(2018, 1, 1),
        "last_review_time": datetime(2018, 1, 4),
    }


def _review_row(
    asin: str,
    review_id: str,
    when: datetime,
    *,
    verified: bool | None,
    label: int,
) -> dict:
    """Build one reviews-table insert payload.

    Args:
        asin: Owning product ASIN.
        review_id: Stable 16-hex review key.
        when: Naive UTC review time.
        verified: Verified flag; None means "unknown" in the source.
        label: Weak sentiment label code.

    Returns:
        Insert payload.
    """
    return {
        "review_id": review_id,
        "asin": asin,
        "split": SPLIT_TRAIN,
        "reviewer_id": f"user-{review_id[-1]}",
        "rating": 4.0 if label == SENTIMENT_LABEL_POSITIVE else 2.0,
        "review_text_raw": "text",
        "review_text_clean": "text",
        "summary_raw": "summary",
        "summary_clean": "summary",
        "has_text": True,
        "text_length": 4,
        "helpful_votes": 0,
        "verified": verified,
        "review_time": when,
        "year": when.year,
        "month": when.month,
        "sentiment_label_weak": label,
    }


SEED_REVIEWS = [
    _review_row(
        "B000000001", "00000000000000a1", datetime(2018, 1, 1), verified=True, label=SENTIMENT_LABEL_POSITIVE
    ),
    _review_row(
        "B000000001", "00000000000000a2", datetime(2018, 1, 2), verified=False, label=SENTIMENT_LABEL_NEGATIVE
    ),
    _review_row(
        "B000000001", "00000000000000a3", datetime(2018, 1, 3), verified=None, label=SENTIMENT_LABEL_NEUTRAL
    ),
    _review_row(
        "B000000002", "00000000000000b1", datetime(2018, 1, 4), verified=True, label=SENTIMENT_LABEL_POSITIVE
    ),
]

SEED_EDGES = [
    {"asin": "B000000001", "related_asin": "B000000002", "relation_type": "also_bought", "in_subset": True},
    {"asin": "B000000001", "related_asin": "B000000003", "relation_type": "also_viewed", "in_subset": True},
    {"asin": "B000000001", "related_asin": "OUTSIDE999", "relation_type": "also_bought", "in_subset": False},
]


@pytest.fixture()
def engine() -> Iterator[Engine]:
    """Seeded in-memory engine.

    Yields:
        Engine with products, reviews and related edges loaded.
    """
    engine = create_engine_from_settings(db_url="sqlite://")
    init_db(engine)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        session.execute(
            insert(Product),
            [_product_row(asin, title, brand, split) for asin, title, brand, split in PRODUCTS],
        )
        session.execute(insert(Review), SEED_REVIEWS)
        session.execute(insert(RelatedProduct), SEED_EDGES)
    yield engine
    engine.dispose()


@pytest.fixture()
def session(engine: Engine) -> Iterator[Session]:
    """Open session over the seeded engine (rolled back and closed after)."""
    session = create_session_factory(engine)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# ------------------------------------------------------------------------------
# ProductRepository
# ------------------------------------------------------------------------------


def test_get_product_found(session: Session) -> None:
    """get_product returns the stored row."""
    product = ProductRepository(session).get_product("B000000001")
    assert product.title == "Acme Widget Pro"
    assert product.split == SPLIT_TRAIN


def test_get_product_missing_raises(session: Session) -> None:
    """An unknown ASIN raises the domain error, not an ORM miss."""
    with pytest.raises(ProductNotFoundError) as excinfo:
        ProductRepository(session).get_product("MISSING1")
    assert excinfo.value.details["asin"] == "MISSING1"


def test_search_matches_title_or_brand_case_insensitively(session: Session) -> None:
    """A lower-case query matches titles and brands regardless of casing."""
    page = ProductRepository(session).search_products("acme")
    matched = [product.asin for product in page.items]
    # B1 (title+brand), B3 (brand), B5 (brand "acme industries").
    assert matched == ["B000000001", "B000000003", "B000000005"]
    assert page.total == 3


def test_search_is_case_insensitive_on_uppercase(session: Session) -> None:
    """Upper-case query returns the same rows as lower-case."""
    lower = ProductRepository(session).search_products("widget")
    upper = ProductRepository(session).search_products("WIDGET")
    assert [product.asin for product in lower.items] == [product.asin for product in upper.items]
    assert lower.total == upper.total == 4  # Gamma Gadget has no "widget"


def test_search_escapes_like_wildcards(session: Session) -> None:
    """A literal '%' in the query matches only the product containing '%'."""
    page = ProductRepository(session).search_products("100%")
    assert [product.asin for product in page.items] == ["B000000004"]
    assert page.total == 1


def test_search_pagination_is_stable_with_total(session: Session) -> None:
    """Pages partition the result set in asin order; total is page-independent."""
    repository = ProductRepository(session)
    first = repository.search_products("widget", limit=2, offset=0)
    second = repository.search_products("widget", limit=2, offset=2)
    assert [product.asin for product in first.items] == ["B000000001", "B000000002"]
    assert [product.asin for product in second.items] == ["B000000004", "B000000005"]
    assert first.total == second.total == 4


def test_search_split_filter(session: Session) -> None:
    """A split filter restricts results to that split."""
    page = ProductRepository(session).search_products("widget", split=SPLIT_TEST)
    assert [product.asin for product in page.items] == ["B000000004"]
    assert page.total == 1


def test_search_unknown_split_raises(session: Session) -> None:
    """A split outside the vocabulary is rejected loudly."""
    with pytest.raises(ValueError, match="unknown split"):
        ProductRepository(session).search_products("widget", split="dev")


def test_search_negative_paging_raises(session: Session) -> None:
    """Negative limit/offset are rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        ProductRepository(session).search_products("widget", limit=-1)


def test_search_empty_query_matches_all(session: Session) -> None:
    """An empty query lists every product (useful for browse views)."""
    page = ProductRepository(session).search_products("", limit=10)
    assert page.total == len(PRODUCTS)


def test_list_candidate_asins_sorted(session: Session) -> None:
    """Candidates come back in stable asin order, optionally per split."""
    asins = ProductRepository(session).list_candidate_asins()
    assert asins == sorted(asin for asin, *_ in PRODUCTS)
    train = ProductRepository(session).list_candidate_asins(split=SPLIT_TRAIN)
    assert train == ["B000000001", "B000000002", "B000000005", "B000000006"]


def test_peer_sampling_is_deterministic(session: Session) -> None:
    """The same seed yields the same peer set; the subject is excluded."""
    repository = ProductRepository(session)
    first = [product.asin for product in repository.list_peer_products("B000000001", n=3, seed=42)]
    again = [product.asin for product in repository.list_peer_products("B000000001", n=3, seed=42)]
    assert first == again
    assert "B000000001" not in first
    assert len(first) == 3


def test_peer_sampling_returns_products_in_asin_order(session: Session) -> None:
    """Peers are returned as full rows in asin order regardless of the draw."""
    peers = ProductRepository(session).list_peer_products("B000000001", n=100, seed=7)
    assert [product.asin for product in peers] == [
        asin for asin, *_ in PRODUCTS if asin != "B000000001"
    ]


def test_peer_sampling_zero_and_single_candidate(session: Session) -> None:
    """n=0 returns no peers; the only product can have no peers at all."""
    repository = ProductRepository(session)
    assert repository.list_peer_products("B000000001", n=0, seed=1) == []


# ------------------------------------------------------------------------------
# ReviewRepository
# ------------------------------------------------------------------------------


def test_get_reviews_newest_first(session: Session) -> None:
    """Reviews are returned newest first with the total count."""
    page = ReviewRepository(session).get_reviews("B000000001")
    assert [review.review_id for review in page.items] == [
        "00000000000000a3",
        "00000000000000a2",
        "00000000000000a1",
    ]
    assert page.total == 3


def test_get_reviews_verified_false_excludes_unknown(session: Session) -> None:
    """verified=False matches only explicit False - never NULL (unknown)."""
    page = ReviewRepository(session).get_reviews("B000000001", verified=False)
    assert [review.review_id for review in page.items] == ["00000000000000a2"]
    assert page.total == 1


def test_get_reviews_verified_true_and_unfiltered(session: Session) -> None:
    """verified=True matches only True; no filter returns every review."""
    repository = ReviewRepository(session)
    true_page = repository.get_reviews("B000000001", verified=True)
    assert [review.review_id for review in true_page.items] == ["00000000000000a1"]
    all_page = repository.get_reviews("B000000001")
    assert all_page.total == 3
    null_row = next(review for review in all_page.items if review.review_id == "00000000000000a3")
    assert null_row.verified is None  # unknown, preserved as NULL


def test_get_reviews_label_filter(session: Session) -> None:
    """Weak-label filtering uses the shared label constants."""
    page = ReviewRepository(session).get_reviews("B000000001", label=SENTIMENT_LABEL_NEUTRAL)
    assert [review.review_id for review in page.items] == ["00000000000000a3"]


def test_get_reviews_unknown_label_raises(session: Session) -> None:
    """A label outside the weak-label vocabulary is rejected."""
    with pytest.raises(ValueError, match="unknown sentiment label"):
        ReviewRepository(session).get_reviews("B000000001", label=7)


def test_get_reviews_pagination(session: Session) -> None:
    """limit/offset page over the stable newest-first order."""
    page = ReviewRepository(session).get_reviews("B000000001", limit=2, offset=2)
    assert [review.review_id for review in page.items] == ["00000000000000a1"]
    assert page.total == 3


def test_get_review_count(session: Session) -> None:
    """Counts are per product; unknown products simply have zero reviews."""
    repository = ReviewRepository(session)
    assert repository.get_review_count("B000000001") == 3
    assert repository.get_review_count("B000000009") == 0


# ------------------------------------------------------------------------------
# RelatedRepository
# ------------------------------------------------------------------------------


def test_get_related_only_in_subset(session: Session) -> None:
    """The default keeps only edges whose target is in the subset."""
    edges = RelatedRepository(session).get_related("B000000001")
    assert [(edge.related_asin, edge.relation_type) for edge in edges] == [
        ("B000000002", "also_bought"),
        ("B000000003", "also_viewed"),
    ]


def test_get_related_all_edges(session: Session) -> None:
    """only_in_subset=False includes edges pointing outside the subset."""
    edges = RelatedRepository(session).get_related("B000000001", only_in_subset=False)
    assert [edge.related_asin for edge in edges] == ["B000000002", "B000000003", "OUTSIDE999"]


# ------------------------------------------------------------------------------
# AnalysisRepository
# ------------------------------------------------------------------------------


def test_get_analysis_missing_returns_none(session: Session) -> None:
    """No stored analysis means None (the caller decides how to react)."""
    assert AnalysisRepository(session).get_analysis("B000000001", "v1") is None


def test_upsert_analysis_inserts_then_updates(session: Session) -> None:
    """The first upsert inserts; the second overwrites the same key."""
    repository = AnalysisRepository(session)
    created = repository.upsert_analysis(
        "B000000001",
        "v1",
        risk_score=42.5,
        evidence_strength=0.8,
        verdict="CONSIDER",
        payload={"note": "first"},
        created_at=datetime(2018, 10, 4, tzinfo=UTC),
    )
    assert created.risk_score == 42.5
    assert created.created_at == datetime(2018, 10, 4)

    updated = repository.upsert_analysis(
        "B000000001",
        "v1",
        risk_score=63.0,
        evidence_strength=0.9,
        verdict="CONSIDER WITH CAUTION",
        payload={"note": "second"},
        created_at=datetime(2018, 10, 5, tzinfo=UTC),
    )
    assert updated.risk_score == 63.0
    assert updated.created_at == datetime(2018, 10, 5)

    stored = repository.get_analysis("B000000001", "v1")
    assert stored is not None
    assert stored.verdict == "CONSIDER WITH CAUTION"
    assert stored.payload == {"note": "second"}
    # One row only - the composite key prevents duplicates.
    total = session.scalar(select(func.count()).select_from(AnalysisResult))
    assert total == 1


def test_upsert_analysis_separate_versions_are_distinct(session: Session) -> None:
    """Different pipeline versions of the same product coexist."""
    repository = AnalysisRepository(session)
    repository.upsert_analysis("B000000001", "v1", 10.0, 0.1, "CONSIDER", {}, None)
    repository.upsert_analysis("B000000001", "v2", 20.0, 0.2, "CONSUMER RISK", {}, None)
    assert repository.get_analysis("B000000001", "v1").risk_score == 10.0
    assert repository.get_analysis("B000000001", "v2").risk_score == 20.0


def test_upsert_analysis_unknown_product_raises(session: Session) -> None:
    """A foreign-key violation surfaces for an unknown product."""
    with pytest.raises(IntegrityError):
        AnalysisRepository(session).upsert_analysis("MISSING1", "v1", 1.0, 1.0, "CONSIDER", {}, None)

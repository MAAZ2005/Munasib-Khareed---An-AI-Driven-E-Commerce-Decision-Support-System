"""Unit tests for :mod:`munasib.data.features` (feature derivation).

Covers de-duplication identity, weak-label derivation from star ratings (config
boundaries), time features anchored on the data, per-product aggregates and the
related-products edge table. Hermetic: no network, no GPU, no raw data files.
"""

from __future__ import annotations

import pandas as pd
import pytest

from munasib.core.config import RatingMappingConfig
from munasib.data.features import (
    REVIEW_ID_HASH_CHARS,
    add_review_ids,
    add_review_time_features,
    build_related_products_table,
    compute_product_review_aggregates,
    derive_weak_sentiment_labels,
    drop_duplicate_reviews,
    normalize_dedup_text,
    normalize_helpful_votes,
)


def _review_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a minimal review frame with the columns the feature ops expect.

    Args:
        rows: Dicts with reviewer_id/asin/unixReviewTime/review_text_clean.

    Returns:
        DataFrame with those columns.
    """
    return pd.DataFrame(rows)


class TestNormalizeDedupText:
    """De-duplication text key."""

    def test_lowercases_and_collapses(self) -> None:
        """Case and whitespace differences normalise to the same key."""
        assert normalize_dedup_text("Hello   World") == "hello world"

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_missing_is_empty_string(self, value: object) -> None:
        """Missing text yields an empty key, not an error."""
        assert normalize_dedup_text(value) == ""


class TestReviewIdsAndDedup:
    """Stable review identity and duplicate removal."""

    def _frame(self) -> pd.DataFrame:
        return _review_frame(
            [
                {
                    "reviewer_id": "U1",
                    "asin": "B000000001",
                    "unixReviewTime": 1385510400.0,
                    "review_text_clean": "Great item",
                },
                {
                    "reviewer_id": "U1",
                    "asin": "B000000001",
                    "unixReviewTime": 1385510400.0,
                    "review_text_clean": "Great item",
                },
                {
                    "reviewer_id": "U1",
                    "asin": "B000000001",
                    "unixReviewTime": 1385510400.0,
                    "review_text_clean": "Different text",
                },
            ]
        )

    def test_ids_are_deterministic_and_bounded(self) -> None:
        """Same key -> same id; id length is the documented 16 hex chars."""
        frame = self._frame()
        first = add_review_ids(frame.copy())
        second = add_review_ids(frame.copy())
        assert list(first["review_id"]) == list(second["review_id"])
        assert all(len(value) == REVIEW_ID_HASH_CHARS for value in first["review_id"])

    def test_identical_rows_share_id_and_text_change_differs(self) -> None:
        """Duplicates hash identically; a text change produces a new id."""
        frame = add_review_ids(self._frame())
        assert frame["review_id"].iloc[0] == frame["review_id"].iloc[1]
        assert frame["review_id"].iloc[0] != frame["review_id"].iloc[2]

    def test_dedup_drops_exact_duplicates_only(self) -> None:
        """Only the full-key duplicate is dropped; similar rows survive."""
        frame = add_review_ids(self._frame())
        deduplicated, dropped = drop_duplicate_reviews(frame)
        assert dropped == 1
        assert len(deduplicated) == 2

    def test_dedup_with_missing_reviewer_and_timestamp(self) -> None:
        """Missing key parts still produce a working key (no crash)."""
        frame = _review_frame(
            [
                {"asin": "B1", "unixReviewTime": None, "review_text_clean": None},
                {"asin": "B1", "unixReviewTime": None, "review_text_clean": None},
                {"asin": "B1", "unixReviewTime": None, "review_text_clean": "other"},
            ]
        )
        with_ids = add_review_ids(frame)
        deduplicated, dropped = drop_duplicate_reviews(with_ids)
        assert dropped == 1
        assert len(deduplicated) == 2


class TestHelpfulVotes:
    """Documented zero-fill for missing votes."""

    def test_missing_becomes_zero(self) -> None:
        """Missing vote field means 0 recorded helpful votes (documented)."""
        votes = pd.Series([2, None, 4], dtype="Int64")
        result = normalize_helpful_votes(votes)
        assert list(result) == [2, 0, 4]
        assert result.dtype == "int32"

    def test_all_missing_all_zero(self) -> None:
        """A file with no vote column at all yields all zeros, not nulls."""
        result = normalize_helpful_votes(pd.Series([None, None], dtype="Int64"))
        assert list(result) == [0, 0]


class TestWeakSentimentLabels:
    """Weak supervision mapping (1-2 negative / 3 neutral / 4-5 positive)."""

    def test_default_mapping(self) -> None:
        """Ratings 1..5 map to labels 0/0/1/2/2."""
        labels = derive_weak_sentiment_labels(
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), RatingMappingConfig()
        )
        assert list(labels) == [0, 0, 1, 2, 2]
        assert labels.dtype == "Int8"

    def test_missing_rating_stays_null(self) -> None:
        """A missing rating never gets a fabricated label."""
        labels = derive_weak_sentiment_labels(
            pd.Series([1.0, None, float("nan")]), RatingMappingConfig()
        )
        assert labels.iloc[0] == 0
        assert pd.isna(labels.iloc[1])
        assert pd.isna(labels.iloc[2])

    def test_config_boundaries_are_respected(self) -> None:
        """Different config boundaries change the labels (no hardcoding)."""
        mapping = RatingMappingConfig(negative_max=1, neutral_val=2, positive_min=3)
        labels = derive_weak_sentiment_labels(pd.Series([1.0, 2.0, 3.0, 5.0]), mapping)
        assert list(labels) == [0, 1, 2, 2]

    def test_non_numeric_rating_is_null(self) -> None:
        """Garbage ratings coerce to null labels."""
        labels = derive_weak_sentiment_labels(pd.Series(["bad", "3"]), RatingMappingConfig())
        assert pd.isna(labels.iloc[0])
        assert labels.iloc[1] == 1


class TestTimeFeatures:
    """Time features derived from the data (never today's date)."""

    def test_unix_to_utc_parts(self) -> None:
        """1385510400 -> 2013-11-27 UTC with correct year/month parts."""
        frame = pd.DataFrame({"unixReviewTime": [1385510400.0, 1514764800.0]})
        result = add_review_time_features(frame)
        assert str(result["review_datetime"].iloc[0]) == "2013-11-27 00:00:00+00:00"
        assert str(result["review_datetime"].iloc[1]) == "2018-01-01 00:00:00+00:00"
        assert list(result["review_year"]) == [2013, 2018]
        assert list(result["review_month"]) == [11, 1]
        assert "UTC" in str(result["review_datetime"].dtype)

    def test_unparsable_timestamp_is_nat(self) -> None:
        """Missing timestamps become NaT/NA - never a guessed date."""
        frame = pd.DataFrame({"unixReviewTime": [float("nan")]})
        result = add_review_time_features(frame)
        assert pd.isna(result["review_datetime"].iloc[0])
        assert pd.isna(result["review_year"].iloc[0])


class TestProductAggregates:
    """Per-product review statistics from the kept reviews."""

    def test_counts_and_bounds(self) -> None:
        """n_reviews/first/last are computed per asin from review_datetime."""
        reviews = pd.DataFrame(
            {
                "asin": ["A", "A", "B"],
                "review_datetime": pd.to_datetime(
                    ["2018-01-01", "2018-06-01", "2017-05-05"], utc=True
                ),
            }
        )
        aggregates = compute_product_review_aggregates(reviews)
        row_a = aggregates[aggregates["asin"] == "A"].iloc[0]
        row_b = aggregates[aggregates["asin"] == "B"].iloc[0]
        assert int(row_a["n_reviews"]) == 2
        assert str(row_a["first_review_time"]) == "2018-01-01 00:00:00+00:00"
        assert str(row_a["last_review_time"]) == "2018-06-01 00:00:00+00:00"
        assert int(row_b["n_reviews"]) == 1
        assert aggregates["n_reviews"].dtype == "int32"


class TestRelatedProductsTable:
    """Related-products edge table construction."""

    def _products(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "asin": ["A1", "A2"],
                "also_buy": [["A2", "A1", "X9"], ["Y8"]],
                "also_view": [["A2", "A2"], None],
            }
        )

    def test_edges_self_refs_and_duplicates(self) -> None:
        """Self-references and duplicate triples drop; distinct relations stay."""
        table = build_related_products_table(self._products(), kept_asins={"A1", "A2"})
        assert len(table) == 4  # A1->X9 buy, A2->Y8 buy, A1->A2 view x2 deduped
        assert not (table["asin"] == table["related_asin"]).any()
        assert set(table["relation_type"]) == {"also_bought", "also_viewed"}
        # A pair present in both lists keeps both relations.
        both = table[(table["asin"] == "A1") & (table["related_asin"] == "A2")]
        assert set(both["relation_type"]) == {"also_bought", "also_viewed"}

    def test_in_subset_flag(self) -> None:
        """in_subset marks related ASINs that survived candidate selection."""
        table = build_related_products_table(self._products(), kept_asins={"A1", "A2"})
        flags = {(row["related_asin"], row["in_subset"]) for _, row in table.iterrows()}
        assert ("X9", False) in flags
        assert ("A2", True) in flags

    def test_duplicate_triples_collapse(self) -> None:
        """Repeated (asin, related_asin, relation_type) rows collapse to one."""
        table = build_related_products_table(self._products(), kept_asins=set())
        triples = table[["asin", "related_asin", "relation_type"]]
        assert triples.duplicated().sum() == 0

    def test_empty_inputs_yield_empty_table(self) -> None:
        """No links at all still yields a well-formed empty table."""
        products = pd.DataFrame({"asin": ["A1"], "also_buy": [None], "also_view": [None]})
        table = build_related_products_table(products, kept_asins={"A1"})
        assert list(table.columns) == ["asin", "related_asin", "relation_type", "in_subset"]
        assert table.empty

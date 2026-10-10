"""Unit tests for the product-level stratified train/val/test split.

Covers the leakage guarantee (a product appears in exactly one split and every
review inherits it), determinism for a fixed seed, binning edges and the config
validation. Hermetic: in-memory frames only - no files, no network, no GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from munasib.core.config import SplitConfig
from munasib.data.splits import (
    SPLIT_ORDER,
    UNKNOWN_BIN,
    assign_product_splits,
    assign_review_splits,
    bin_labels,
    bin_values,
    split_sizes,
)


def _products(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Deterministic product frame spanning several rating/count strata.

    Args:
        n: Number of products.
        seed: RNG seed for the synthetic values (not the split seed).

    Returns:
        Frame with ``asin``, ``mean_rating``, ``n_reviews``.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "asin": [f"B{i:05d}" for i in range(n)],
            "mean_rating": rng.uniform(2.5, 5.0, n),
            "n_reviews": rng.integers(40, 600, n),
        }
    )


class TestSplitConfigValidation:
    """Config validation: fractions sum to 1.0, edges ascend strictly."""

    def test_defaults_are_valid_and_sum_to_one(self) -> None:
        config = SplitConfig()
        assert config.train_fraction + config.val_fraction + config.test_fraction == pytest.approx(1.0)
        assert config.seed == 42

    @pytest.mark.parametrize(
        "overrides",
        [
            {"train_fraction": 0.5, "val_fraction": 0.5, "test_fraction": 0.5},
            {"train_fraction": 0.8, "val_fraction": 0.1, "test_fraction": 0.05},
            {"train_fraction": 0.0, "val_fraction": 0.5, "test_fraction": 0.5},
        ],
    )
    def test_fractions_must_be_positive_and_sum_to_one(self, overrides: dict) -> None:
        with pytest.raises(ValidationError):
            SplitConfig(**overrides)

    @pytest.mark.parametrize(
        "edges",
        [[4.5, 3.5], [3.5, 3.5], [75.0, 50.0]],
    )
    def test_bin_edges_must_ascend_strictly(self, edges: list[float]) -> None:
        field = "rating_bin_edges" if edges[0] < 10 else "review_count_bin_edges"
        with pytest.raises(ValidationError):
            SplitConfig(**{field: edges})


class TestBinning:
    """Right-closed bins from the configured interior cut points."""

    def test_bin_labels(self) -> None:
        assert bin_labels([3.5, 4.5]) == ["<=3.5", "3.5-4.5", ">4.5"]
        assert bin_labels([]) == ["all"]

    def test_edges_are_right_closed(self) -> None:
        values = pd.Series([1.0, 3.5, 3.5001, 4.5, 4.5001])
        binned = bin_values(values, [3.5, 4.5]).tolist()
        assert binned == ["<=3.5", "<=3.5", "3.5-4.5", "3.5-4.5", ">4.5"]

    def test_missing_value_is_unknown_bin(self) -> None:
        binned = bin_values(pd.Series([3.0, np.nan]), [3.5]).tolist()
        assert binned[0] == "<=3.5"
        assert binned[1] == UNKNOWN_BIN

    def test_no_edges_single_bin(self) -> None:
        assert set(bin_values(pd.Series([1.0, 9.0]), [])) == {"all"}


class TestAssignProductSplits:
    """Every product lands in exactly one split, deterministically."""

    def test_no_product_in_more_than_one_split(self) -> None:
        assigned = assign_product_splits(_products(), SplitConfig())
        assert not assigned["asin"].duplicated().any()
        assert set(assigned["split"].astype(str)) <= set(SPLIT_ORDER)
        assert assigned["split"].isna().sum() == 0

    def test_determinism_with_fixed_seed(self) -> None:
        products = _products()
        config = SplitConfig()
        first = assign_product_splits(products, config)
        second = assign_product_splits(products, config)
        mapping_first = dict(zip(first["asin"], first["split"], strict=True))
        mapping_second = dict(zip(second["asin"], second["split"], strict=True))
        assert mapping_first == mapping_second

    def test_row_order_does_not_change_assignment(self) -> None:
        products = _products()
        shuffled = products.sample(frac=1.0, random_state=7).reset_index(drop=True)
        first = assign_product_splits(products, SplitConfig())
        second = assign_product_splits(shuffled, SplitConfig())
        assert (
            first.set_index("asin")["split"].sort_index()
            == second.set_index("asin")["split"].sort_index()
        ).all()

    def test_different_seed_changes_assignment(self) -> None:
        products = _products()
        base = assign_product_splits(products, SplitConfig())
        other = assign_product_splits(products, SplitConfig(seed=99))
        assert not (
            base.set_index("asin")["split"].sort_index()
            == other.set_index("asin")["split"].sort_index()
        ).all()

    def test_fractions_are_respected_globally(self) -> None:
        assigned = assign_product_splits(_products(400), SplitConfig())
        counts = assigned["split"].astype(str).value_counts()
        assert counts["train"] == pytest.approx(280, abs=20)  # 70% of 400, strata rounding
        assert counts["val"] == pytest.approx(60, abs=15)  # 15%
        assert counts["test"] == pytest.approx(60, abs=15)  # 15%
        assert int(counts.sum()) == 400

    def test_stratum_columns_reflect_config_bins(self) -> None:
        assigned = assign_product_splits(_products(50), SplitConfig())
        assert set(assigned["rating_bin"].astype(str)) <= {"<=3.5", "3.5-4.5", ">4.5"}
        assert set(assigned["review_count_bin"].astype(str)) <= {
            "<=75",
            "75-150",
            "150-400",
            ">400",
        }

    def test_duplicate_asin_raises(self) -> None:
        products = pd.DataFrame(
            {"asin": ["A", "A"], "mean_rating": [4.0, 4.0], "n_reviews": [1, 1]}
        )
        with pytest.raises(ValueError, match="duplicate asin"):
            assign_product_splits(products, SplitConfig())

    def test_missing_asin_column_raises(self) -> None:
        products = pd.DataFrame({"mean_rating": [4.0], "n_reviews": [1]})
        with pytest.raises(ValueError, match="asin"):
            assign_product_splits(products, SplitConfig())


class TestAssignReviewSplits:
    """Reviews inherit their product's split; orphans raise instead of leaking."""

    def _assigned(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        products = assign_product_splits(_products(60), SplitConfig())
        reviews = pd.DataFrame(
            {
                "asin": np.repeat(products["asin"].to_numpy(), 3),
                "reviewer_id": ["U"] * (60 * 3),
            }
        )
        return products, reviews

    def test_reviews_inherit_product_split(self) -> None:
        products, reviews = self._assigned()
        merged = assign_review_splits(reviews, products)
        expected = merged.merge(
            products[["asin", "split"]], on="asin", suffixes=("_review", "_product")
        )
        assert (expected["split_review"] == expected["split_product"]).all()
        assert merged["split"].isna().sum() == 0

    def test_no_product_spans_multiple_review_splits(self) -> None:
        products, reviews = self._assigned()
        merged = assign_review_splits(reviews, products)
        per_product = merged.groupby("asin", observed=True)["split"].nunique()
        assert per_product.max() == 1

    def test_orphan_review_raises(self) -> None:
        products, reviews = self._assigned()
        reviews = pd.concat(
            [reviews, pd.DataFrame({"asin": ["GHOST"], "reviewer_id": ["X"]})],
            ignore_index=True,
        )
        with pytest.raises(ValueError, match="no product split"):
            assign_review_splits(reviews, products)

    def test_duplicate_mapping_raises(self) -> None:
        products = _products(4)
        products = assign_product_splits(products, SplitConfig())
        bad = pd.concat([products, products.iloc[[0]]], ignore_index=True)
        reviews = pd.DataFrame({"asin": ["B00000"], "reviewer_id": ["U"]})
        with pytest.raises(ValueError, match="duplicate asin"):
            assign_review_splits(reviews, bad)


class TestSplitSizes:
    """Reporting helper: per-split product/review counts, zero-filled."""

    def test_counts_products_and_reviews(self) -> None:
        products = pd.DataFrame({"split": ["train", "train", "test"]})
        reviews = pd.DataFrame({"split": ["train"] * 10 + ["val"] * 4 + ["test"] * 2})
        sizes = split_sizes(products, reviews)
        assert sizes == {
            "train": {"products": 2, "reviews": 10},
            "val": {"products": 0, "reviews": 4},
            "test": {"products": 1, "reviews": 2},
        }

    def test_split_order_is_canonical(self) -> None:
        assert SPLIT_ORDER == ("train", "val", "test")

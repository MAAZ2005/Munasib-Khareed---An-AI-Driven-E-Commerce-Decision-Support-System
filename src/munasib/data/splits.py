"""Product-level train/val/test split assignment (leakage-safe).

The split is assigned at the **product** level: every product is placed in
exactly one split, and every review of that product inherits the product's
split. This guarantees no product's reviews are spread across train/val/test,
which would leak per-product evidence into evaluation.

Stratification (project rule: all thresholds/seeds come from config):

1. Each product's average rating and review count are binned with the interior
   cut points from ``dataset.split`` in ``configs/config.yaml``. Bins are
   right-closed: ``(-inf, e0], (e0, e1], ..., (en, inf)``.
2. The stratum key is ``rating=<bin> | reviews=<bin>``.
3. Strata are visited in sorted order and products inside a stratum are sorted
   by ``asin`` (stable, independent of input order), shuffled with
   ``numpy.random.default_rng(split.seed)``, then allocated to train/val/test by
   rounding the configured fractions (test absorbs the rounding remainder).

Both steps are deterministic for a fixed seed, so re-running the build produces
byte-identical split assignments. Only aggregate stratum sizes are reported
never per-product assignments, so nothing outside the parquet tables can grow
stale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from munasib.core.config import SplitConfig
from munasib.core.constants import SPLIT_ORDER, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL

#: Label used when a stratification value is missing (never binned silently).
UNKNOWN_BIN = "unknown"


def bin_labels(edges: list[float]) -> list[str]:
    """Readable labels for right-closed bins defined by interior cut points.

    Args:
        edges: Strictly ascending interior cut points (may be empty).

    Returns:
        One label per bin: ``["<=e0", "e0-e1", ..., ">en"]`` (or ``["all"]``
        when there are no edges).
    """
    if not edges:
        return ["all"]
    labels = [f"<={edges[0]:g}"]
    labels.extend(f"{low:g}-{high:g}" for low, high in zip(edges, edges[1:], strict=False))
    labels.append(f">{edges[-1]:g}")
    return labels


def bin_values(values: pd.Series, edges: list[float]) -> pd.Series:
    """Bin numeric values into right-closed bins ``(-inf, e0], (e0, e1], ...``.

    Args:
        values: Numeric values (NaN is kept as its own ``"unknown"`` bin).
        edges: Strictly ascending interior cut points; empty -> single bin.

    Returns:
        String labels aligned to ``values.index``.
    """
    labels = bin_labels(edges)
    out = pd.Series(UNKNOWN_BIN, index=values.index, dtype="object")
    numeric = pd.to_numeric(values, errors="coerce")
    present = numeric.notna()
    if not present.any():
        return out.astype("string")
    if not edges:
        out.loc[present] = labels[0]
        return out.astype("string")
    # right=True -> bin index i means edges[i-1] < v <= edges[i]; index 0 is v <= edges[0].
    indices = np.digitize(numeric.loc[present].to_numpy(dtype=float), edges, right=True)
    out.loc[present] = np.asarray(labels, dtype=object)[indices]
    return out.astype("string")


def _allocate(n: int, config: SplitConfig) -> tuple[int, int, int]:
    """Allocate ``n`` products to (train, val, test) by the configured fractions.

    Args:
        n: Number of products in the stratum (n >= 0).
        config: Split fractions.

    Returns:
        (train_count, val_count, test_count); test absorbs any rounding
        remainder and is clamped to be non-negative.
    """
    if n == 0:
        return (0, 0, 0)
    train_n = int(round(n * config.train_fraction))
    val_n = int(round(n * config.val_fraction))
    train_n = min(train_n, n)
    val_n = min(val_n, n - train_n)
    test_n = n - train_n - val_n
    if test_n < 0:  # defensive: only reachable for extreme fraction roundings
        take = min(-test_n, train_n)
        train_n -= take
        test_n += take
    return (train_n, val_n, test_n)


def assign_product_splits(products: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Assign every product to exactly one split, stratified and deterministic.

    Args:
        products: Product frame with ``asin``, ``mean_rating`` and ``n_reviews``
            (each ``asin`` must be unique; a missing mean rating is fine).
        config: Split fractions, seed and bin edges.

    Returns:
        ``products`` plus the columns ``split`` (``train``/``val``/``test``),
        ``rating_bin`` and ``review_count_bin`` (stratum labels; reporting-only).

    Raises:
        ValueError: When ``asin`` values are duplicated (splits would be
            ambiguous) or the ``asin`` column is missing.
    """
    if "asin" not in products.columns:
        raise ValueError("products frame must contain an 'asin' column")
    if products["asin"].duplicated().any():
        raise ValueError("products frame has duplicate asin values; split would be ambiguous")

    frame = products.copy()
    frame["rating_bin"] = bin_values(frame["mean_rating"], config.rating_bin_edges)
    frame["review_count_bin"] = bin_values(
        frame["n_reviews"].astype("float64"), config.review_count_bin_edges
    )
    stratum = frame["rating_bin"].astype("str") + " | " + frame["review_count_bin"].astype("str")

    asin_array = frame["asin"].astype("str").to_numpy()
    split_col = pd.Series(pd.NA, index=frame.index, dtype="string")
    rng = np.random.default_rng(config.seed)
    # Sorted stratum order + asin ordering inside each stratum make the RNG draw
    # order independent of input row order (identical splits for the same seed).
    for stratum_key in sorted(stratum.unique()):
        mask = (stratum == stratum_key).to_numpy()
        members = np.sort(asin_array[mask])
        shuffled = members[rng.permutation(len(members))]
        n_train, n_val, n_test = _allocate(len(shuffled), config)
        assigned = np.concatenate(
            [
                np.full(n_train, SPLIT_TRAIN, dtype=object),
                np.full(n_val, SPLIT_VAL, dtype=object),
                np.full(n_test, SPLIT_TEST, dtype=object),
            ]
        )
        positions = pd.Series(assigned, index=pd.Index(shuffled, name="asin"))
        mapped = pd.Series(asin_array[mask]).map(positions)
        split_col.loc[mask] = mapped.to_numpy()

    frame["split"] = split_col.astype("string")
    if frame["split"].isna().any():
        missing = frame.loc[frame["split"].isna(), "asin"].tolist()
        raise ValueError(f"internal error: {len(missing)} product(s) unassigned, e.g. {missing[:5]}")
    return frame


def assign_review_splits(reviews: pd.DataFrame, product_splits: pd.DataFrame) -> pd.DataFrame:
    """Propagate the product-level split to every review of that product.

    Args:
        reviews: Review frame with an ``asin`` column.
        product_splits: Frame with ``asin`` and ``split`` (one row per product).

    Returns:
        ``reviews`` plus a non-null ``split`` column.

    Raises:
        ValueError: When the mapping frame has duplicate ``asin`` values (the
            split would be ambiguous) or when a review's product has no split
            (would silently leak).
    """
    mapping = product_splits[["asin", "split"]]
    if mapping["asin"].duplicated().any():
        raise ValueError("product split mapping has duplicate asin values")
    merged = reviews.merge(mapping, on="asin", how="left", validate="many_to_one")
    orphan = merged["split"].isna().sum()
    if orphan:
        sample = merged.loc[merged["split"].isna(), "asin"].unique().tolist()[:5]
        raise ValueError(f"{orphan} review(s) have no product split; e.g. asin {sample}")
    merged["split"] = merged["split"].astype("string")
    return merged


def split_sizes(products: pd.DataFrame, reviews: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Count products and reviews per split for reporting.

    Args:
        products: Product frame with a ``split`` column.
        reviews: Review frame with a ``split`` column.

    Returns:
        ``{"train": {"products": n, "reviews": m}, "val": ..., "test": ...}``
        with zero counts for any split that happens to be empty.
    """
    out: dict[str, dict[str, int]] = {}
    product_counts = products["split"].astype("str").value_counts()
    review_counts = reviews["split"].astype("str").value_counts()
    for split in SPLIT_ORDER:
        out[split] = {
            "products": int(product_counts.get(split, 0)),
            "reviews": int(review_counts.get(split, 0)),
        }
    return out

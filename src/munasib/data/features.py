"""Feature derivation for the processed Munasib Khareed dataset.

Transforms the cleaned review/product frames into analysis-ready columns:

* review identity + de-duplication (key: reviewer + asin + timestamp + normalised
  text),
* time features derived **from the data** (never from today's date - downstream
  "recent" features must anchor on ``meta.json:review_date_range.max``),
* weak sentiment labels derived from star ratings (locked design decision: this is
  *weak supervision*, not true sentiment - see :func:`derive_weak_sentiment_labels`),
* per-product review aggregates (``n_reviews``, ``first_review_time``,
  ``last_review_time``),
* the related-products edge table with an ``in_subset`` flag.
"""

from __future__ import annotations

import hashlib
import logging

import pandas as pd

from munasib.core.config import RatingMappingConfig
from munasib.data.loaders import is_present

logger = logging.getLogger("munasib.data.features")

REVIEW_ID_HASH_CHARS = 16

# Label ids for weak supervision (documented in configs/config.yaml comments).
SENTIMENT_LABEL_NEGATIVE = 0
SENTIMENT_LABEL_NEUTRAL = 1
SENTIMENT_LABEL_POSITIVE = 2


def normalize_dedup_text(value: object) -> str:
    """Normalise review text for de-duplication comparison only.

    Lower-cases and collapses whitespace so trivially re-posted duplicates match.
    This key is *not* the model input (``review_text_clean`` keeps natural casing).

    Args:
        value: Cleaned review text or None.

    Returns:
        Normalised comparison string; empty string for missing text.
    """
    if not is_present(value):
        return ""
    return " ".join(str(value).lower().split())


def add_review_ids(df: pd.DataFrame, text_col: str = "review_text_clean") -> pd.DataFrame:
    """Add a deterministic ``review_id`` derived from the de-duplication key.

    ``review_id = sha1(reviewer_id | asin | unixReviewTime | normalised_text)[:16]``
    - identical rows hash identically, which is exactly what de-duplication needs,
    and downstream layers get a stable primary key for free.

    Args:
        df: Review frame with ``reviewer_id``, ``asin``, ``unixReviewTime`` and the
            text column.
        text_col: Name of the cleaned-text column used in the key.

    Returns:
        The same frame with a ``review_id`` string column added.
    """
    def _key_part(series: pd.Series) -> pd.Series:
        # dtype-agnostic: str() of the present value, "" when missing (a
        # fillna("") on e.g. the float timestamp column would not be legal).
        return series.map(lambda value: str(value) if is_present(value) else "")

    reviewer = (
        _key_part(df["reviewer_id"])
        if "reviewer_id" in df.columns
        else pd.Series("", index=df.index, dtype="str")
    )
    keys = (
        reviewer
        + "|"
        + _key_part(df["asin"])
        + "|"
        + _key_part(df["unixReviewTime"])
        + "|"
        + df[text_col].map(normalize_dedup_text)
    )
    df["review_id"] = [
        hashlib.sha1(key.encode("utf-8", errors="replace")).hexdigest()[:REVIEW_ID_HASH_CHARS]
        for key in keys
    ]
    return df


def drop_duplicate_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate reviews and report how many were removed.

    Duplicates share the full de-duplication key: ``reviewer_id``, ``asin``,
    ``unixReviewTime`` and the normalised review text.

    Args:
        df: Review frame (must carry ``review_id`` from :func:`add_review_ids`).

    Returns:
        Tuple ``(deduplicated_frame, n_dropped)``.
    """
    before = len(df)
    deduplicated = df.drop_duplicates(subset=["review_id"], keep="first")
    return deduplicated, before - len(deduplicated)


def normalize_helpful_votes(votes: pd.Series) -> pd.Series:
    """Convert loader ``vote`` (nullable) to ``helpful_votes`` with 0-fill.

    Documented convention: a *missing* vote field means "no recorded helpful votes"
    and becomes 0. This is the one place where missing data is replaced, by design -
    the loader's nullable ``vote`` column is preserved upstream, and the audit shows
    only 10.8% of Appliances reviews carry the field at all.

    Args:
        votes: Loader-parsed vote column (pandas nullable ``Int64``).

    Returns:
        Non-null ``int32`` series.
    """
    return votes.fillna(0).astype("int32")


def derive_weak_sentiment_labels(
    ratings: pd.Series, mapping: RatingMappingConfig
) -> pd.Series:
    """Map star ratings to weak sentiment labels.

    Weak supervision (locked FYP design decision): the label *assumes* rating and
    sentiment agree - ratings ``<= mapping.negative_max`` (1-2) -> 0 negative,
    ``== mapping.neutral_val`` (3) -> 1 neutral, ``>= mapping.positive_min`` (4-5)
    -> 2 positive. These are **weak labels, not true sentiment**; every metric that
    uses them must say so.

    Args:
        ratings: Star ratings (float or missing).
        mapping: Rating-to-label boundaries from ``review_intel.sentiment`` config.

    Returns:
        Nullable ``int8`` series (null when the rating is missing or out of range).
    """
    numeric = pd.to_numeric(ratings, errors="coerce")
    labels = pd.Series(pd.NA, index=ratings.index, dtype="Int8")
    # Comparisons against NaN are False, so missing ratings keep <NA>.
    labels[numeric <= mapping.negative_max] = SENTIMENT_LABEL_NEGATIVE
    labels[numeric == mapping.neutral_val] = SENTIMENT_LABEL_NEUTRAL
    labels[numeric >= mapping.positive_min] = SENTIMENT_LABEL_POSITIVE
    return labels


def add_review_time_features(df: pd.DataFrame, unix_col: str = "unixReviewTime") -> pd.DataFrame:
    """Convert the unix review timestamp to UTC time parts.

    Args:
        df: Review frame with a unix-seconds column.
        unix_col: Name of the unix-seconds column.

    Returns:
        The same frame with ``review_datetime`` (UTC), ``review_year`` and
        ``review_month`` added. Unparsable timestamps become NaT/NA - never a
        fabricated date.
    """
    df["review_datetime"] = pd.to_datetime(
        pd.to_numeric(df[unix_col], errors="coerce"), unit="s", utc=True, errors="coerce"
    )
    df["review_year"] = df["review_datetime"].dt.year.astype("Int16")
    df["review_month"] = df["review_datetime"].dt.month.astype("Int8")
    return df


def compute_product_review_aggregates(reviews: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-product review statistics from the *kept* reviews.

    Args:
        reviews: Final review frame (post-filtering, post-dedup) with ``asin``,
            ``rating``, ``review_datetime``.

    Returns:
        DataFrame with columns ``asin``, ``n_reviews`` (int32), ``mean_rating``
        (float64; NaN when a product has no numeric rating - never fabricated),
        ``first_review_time``, ``last_review_time`` (UTC timestamps). The mean
        rating feeds the split stratification, so it is stored alongside it.
    """
    aggregates = (
        reviews.groupby("asin", sort=False)
        .agg(
            n_reviews=("review_datetime", "size"),
            mean_rating=("rating", "mean"),
            first_review_time=("review_datetime", "min"),
            last_review_time=("review_datetime", "max"),
        )
        .reset_index()
    )
    aggregates["n_reviews"] = aggregates["n_reviews"].astype("int32")
    aggregates["mean_rating"] = aggregates["mean_rating"].astype("float64")
    return aggregates


def build_related_products_table(
    products: pd.DataFrame, kept_asins: set[str]
) -> pd.DataFrame:
    """Build the related-products edge table for the kept product subset.

    Edges come from the loader's ``also_buy``/``also_view`` list columns.
    Self-references (``asin == related_asin``) and duplicate triples
    ``(asin, related_asin, relation_type)`` are dropped; a related ASIN appearing in
    *both* lists keeps both rows because they are distinct relations.

    Args:
        products: Product frame with ``asin``, ``also_buy``, ``also_view``.
        kept_asins: ASINs of the final product subset (for ``in_subset``).

    Returns:
        DataFrame with columns ``asin``, ``related_asin``, ``relation_type``
        (``"also_bought"`` | ``"also_viewed"``), ``in_subset`` (bool).
    """
    records: list[dict[str, object]] = []
    dropped_self_references = 0
    for relation_type, column in (("also_bought", "also_buy"), ("also_viewed", "also_view")):
        if column not in products.columns:
            continue
        for asin, links in zip(products["asin"], products[column], strict=True):
            if not isinstance(links, list):
                continue
            for related_asin in links:
                if related_asin == asin:
                    dropped_self_references += 1
                    continue
                records.append(
                    {
                        "asin": asin,
                        "related_asin": related_asin,
                        "relation_type": relation_type,
                    }
                )

    table = pd.DataFrame(records, columns=["asin", "related_asin", "relation_type"])
    if table.empty:
        table["in_subset"] = pd.Series(dtype="bool")
        return table

    before = len(table)
    table = table.drop_duplicates(
        subset=["asin", "related_asin", "relation_type"], keep="first"
    )
    dropped_duplicates = before - len(table)
    table["in_subset"] = table["related_asin"].isin(kept_asins)
    logger.info(
        "related products: %d edges (%d self-references dropped, %d duplicates dropped, "
        "%d in-subset)",
        len(table),
        dropped_self_references,
        dropped_duplicates,
        int(table["in_subset"].sum()),
    )
    return table.reset_index(drop=True)

"""Orchestration of the dataset build: raw gzipped JSON-lines -> clean parquet tables.

The build streams the two raw files four times so memory stays bounded (only the
candidate subset - a few thousand rows - is ever held in full):

======  =========================  ==============================================
Pass    Input                      Output
======  =========================  ==============================================
M1      metadata (streamed)        product-ASIN set, row/duplicate accounting
1       reviews (streamed)         review count per ASIN
-       config threshold           candidate ASINs (count >= ``candidate_min``
                                   AND present in the metadata)
M2      metadata (streamed)        full rows for candidates -> product cleaning
2       reviews (streamed)         full rows for candidates -> review cleaning
======  =========================  ==============================================

After the passes: de-duplication, weak-label derivation, per-product review
aggregates, the seeded product-level train/val/test split (reviews inherit their
product's split - see :mod:`munasib.data.splits`), the related-products edge
table, then explicit-schema parquet writes plus a ``meta.json`` of dataset facts
(category, config values, row counts, split sizes, date range, build timestamp)
and the auto-generated ``reports/data_quality_report.{json,md}``
(:mod:`munasib.data.quality` - every number in it is computed from these frames).

Design contract (project rules: *never fabricate*, *faithful load*):

* Nothing is imputed. Missing/unparseable price, rank or brand stay null and get a
  ``*_missing`` boolean flag; review text empties to null with ``has_text=False``.
* The metadata ``category`` breadcrumb becomes ``category_path``/``category_leaf``
  (levels HTML-unescaped and whitespace-collapsed, joined by ``" > "`` broad ->
  specific, see :func:`munasib.data.cleaning.parse_category_path`); a product
  whose source published no breadcrumb keeps null in both columns.
* The display-only product image (``image_url`` + ``image_missing``) is the first
  high-resolution URL, else the first small URL, else null. It is resolved over
  every duplicate row of an ASIN *before* rows are dropped, so an image carried
  only by a dropped duplicate is kept (the kept row's own image always wins).
* The only zero-fill in the build is ``helpful_votes`` (missing vote = 0 recorded
  helpful votes) - documented in :func:`munasib.data.features.normalize_helpful_votes`.
* Weak sentiment labels come from star ratings (1-2 negative / 3 neutral / 4-5
  positive via ``review_intel.sentiment.rating_mapping``); they are **weak labels,
  not true sentiment** and are recorded as such in ``meta.json``.
* Downstream "recent" logic must anchor on ``meta.json``'s
  ``review_date_range.max`` - never on today's date.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from munasib.core.config import Settings
from munasib.core.exceptions import DataError
from munasib.data.cleaning import (
    clean_description,
    clean_features,
    clean_review_text,
    clean_title,
    normalize_brand,
    parse_category_path,
    parse_rank_fields,
)
from munasib.data.features import (
    add_review_ids,
    add_review_time_features,
    build_related_products_table,
    compute_product_review_aggregates,
    derive_weak_sentiment_labels,
    drop_duplicate_reviews,
    normalize_helpful_votes,
)
from munasib.data.loaders import (
    DEFAULT_CHUNK_SIZE,
    LoadStats,
    is_present,
    iter_metadata_chunks,
    iter_review_chunks,
)
from munasib.data.quality import build_quality_report, write_quality_report
from munasib.data.splits import (
    SPLIT_ORDER,
    assign_product_splits,
    assign_review_splits,
    split_sizes,
)

logger = logging.getLogger("munasib.data.preprocessing")

# ------------------------------------------------------------------------------
# Output schemas (explicit - parquet dtypes are part of the pipeline contract)
# ------------------------------------------------------------------------------

REVIEW_SCHEMA = pa.schema(
    [
        ("review_id", pa.string()),
        ("asin", pa.string()),
        ("split", pa.string()),
        ("reviewer_id", pa.string()),
        ("rating", pa.float64()),
        ("review_text_raw", pa.string()),
        ("review_text_clean", pa.string()),
        ("summary_raw", pa.string()),
        ("summary_clean", pa.string()),
        ("has_text", pa.bool_()),
        ("text_length", pa.int32()),
        ("helpful_votes", pa.int32()),
        ("verified", pa.bool_()),
        ("review_datetime", pa.timestamp("us", tz="UTC")),
        ("review_year", pa.int16()),
        ("review_month", pa.int8()),
        ("sentiment_label_weak", pa.int8()),
    ]
)

PRODUCT_SCHEMA = pa.schema(
    [
        ("asin", pa.string()),
        ("split", pa.string()),
        ("title", pa.string()),
        ("title_clean", pa.string()),
        ("brand", pa.string()),
        ("brand_clean", pa.string()),
        ("brand_missing", pa.bool_()),
        ("price", pa.float64()),
        ("price_max", pa.float64()),
        ("price_status", pa.string()),
        ("price_missing", pa.bool_()),
        ("rank", pa.string()),
        ("rank_value", pa.int64()),
        ("rank_category", pa.string()),
        ("rank_missing", pa.bool_()),
        # Browse breadcrumb from the raw metadata "category" field: the levels
        # joined by " > " (broad -> specific) plus the most specific level.
        # Both NULL when the source published no usable breadcrumb.
        ("category_path", pa.string()),
        ("category_leaf", pa.string()),
        ("description_clean", pa.string()),
        ("features_clean", pa.list_(pa.string())),
        ("n_reviews", pa.int32()),
        ("mean_rating", pa.float64()),
        ("first_review_time", pa.timestamp("us", tz="UTC")),
        ("last_review_time", pa.timestamp("us", tz="UTC")),
        # Image columns are appended last so dropping them restores the previous
        # products.parquet column order exactly (see the P9b parity proof).
        ("image_url", pa.string()),
        ("image_missing", pa.bool_()),
    ]
)

RELATED_SCHEMA = pa.schema(
    [
        ("asin", pa.string()),
        ("related_asin", pa.string()),
        ("relation_type", pa.string()),
        ("in_subset", pa.bool_()),
    ]
)

_META_FILENAME = "meta.json"


# ------------------------------------------------------------------------------
# Build report (source of truth for meta.json and the console summary)
# ------------------------------------------------------------------------------


def settings_block(
    candidate_min_reviews: int,
    chunk_size: int,
    rating_mapping: dict[str, int],
    split_settings: dict[str, Any],
) -> dict[str, Any]:
    """Canonical ``settings`` block recorded in ``meta.json``.

    Single source of truth shared by the build and the CLI freshness check, so
    the two can never drift apart.

    Args:
        candidate_min_reviews: Configured candidate threshold.
        chunk_size: Streaming chunk size used for this build.
        rating_mapping: Weak-label rating mapping (from config).
        split_settings: Product-level split config dump (fractions, seed, bins).

    Returns:
        JSON-serialisable settings dict.
    """
    return {
        "candidate_min_reviews": candidate_min_reviews,
        "chunk_size": chunk_size,
        "rating_mapping": dict(rating_mapping),
        "split": dict(split_settings),
    }


def current_settings_block(settings: Settings, chunk_size: int) -> dict[str, Any]:
    """Settings block a build with ``settings``/``chunk_size`` would record.

    Used by ``scripts/build_dataset.py`` to decide whether existing outputs are
    current (skip) or stale (rebuild, unless inputs changed).

    Args:
        settings: Loaded project settings.
        chunk_size: Streaming chunk size the run would use.

    Returns:
        Same structure as :func:`settings_block`.
    """
    mapping = settings.review_intel.sentiment.rating_mapping
    return settings_block(
        settings.dataset.audit.candidate_min_reviews,
        chunk_size,
        {
            "negative_max": mapping.negative_max,
            "neutral_val": mapping.neutral_val,
            "positive_min": mapping.positive_min,
        },
        settings.dataset.split.model_dump(),
    )


@dataclass
class BuildReport:
    """All accounting produced by one dataset build (mirrors ``meta.json``)."""

    category: str
    config_path: str
    candidate_min_reviews: int
    chunk_size: int
    rating_mapping: dict[str, int]
    split_settings: dict[str, Any]
    reviews_path: str
    metadata_path: str
    reviews_bytes: int
    metadata_bytes: int
    built_at: str = ""
    # Pass accounting
    metadata_rows: int = 0
    metadata_rows_without_asin: int = 0
    metadata_duplicate_asin_rows: int = 0
    review_rows_scanned: int = 0
    review_rows_without_asin: int = 0
    products_below_min_reviews: int = 0
    products_missing_metadata: int = 0
    products_selected: int = 0
    products_duplicate_rows_dropped: int = 0
    review_rows_selected: int = 0
    duplicate_reviews_dropped: int = 0
    # Final outputs
    products_final: int = 0
    reviews_final: int = 0
    related_edges: int = 0
    related_in_subset: int = 0
    products_below_min_after_dedup: int = 0
    review_date_min: str | None = None
    review_date_max: str | None = None
    output_files: list[str] = field(default_factory=list)
    # Product-level split (fractions/seed come from config, never hardcoded)
    split_sizes: dict[str, dict[str, int]] = field(default_factory=dict)
    # Data-quality report files (reports dir, separate from the processed dir)
    report_files: list[str] = field(default_factory=list)

    @property
    def reviews_dropped_not_candidate(self) -> int:
        """Review rows skipped in pass 2 because their product was not a candidate."""
        return self.review_rows_scanned - self.review_rows_selected

    def to_meta_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable dataset facts written to ``meta.json``."""
        return {
            "category": self.category,
            "built_at": self.built_at,
            "config_path": self.config_path,
            "settings": settings_block(
                self.candidate_min_reviews,
                self.chunk_size,
                dict(self.rating_mapping),
                dict(self.split_settings),
            ),
            "inputs": {
                "reviews": {"path": self.reviews_path, "bytes": self.reviews_bytes},
                "metadata": {"path": self.metadata_path, "bytes": self.metadata_bytes},
            },
            "row_counts": {
                "metadata_rows": self.metadata_rows,
                "review_rows_scanned": self.review_rows_scanned,
                "products_selected": self.products_selected,
                "products_final": self.products_final,
                "reviews_final": self.reviews_final,
                "related_edges": self.related_edges,
                "related_in_subset": self.related_in_subset,
            },
            "drops": {
                "products_below_min_reviews": self.products_below_min_reviews,
                "products_missing_metadata": self.products_missing_metadata,
                "products_duplicate_rows_dropped": self.products_duplicate_rows_dropped,
                "products_below_min_after_dedup": self.products_below_min_after_dedup,
                "reviews_not_candidate_product": self.reviews_dropped_not_candidate,
                "reviews_missing_asin": self.review_rows_without_asin,
                "duplicate_reviews": self.duplicate_reviews_dropped,
            },
            "review_date_range": {
                "min": self.review_date_min,
                "max": self.review_date_max,
                "note": (
                    "Anchored on the review data itself; downstream recency features "
                    "must use this max, never today's date."
                ),
            },
            "weak_labels": {
                "column": "sentiment_label_weak",
                "mapping": dict(self.rating_mapping),
                "note": (
                    "Weak supervision derived from star ratings (1-2 negative, 3 "
                    "neutral, 4-5 positive) - not true sentiment."
                ),
            },
            "splits": {
                "grain": "product",
                "sizes": {
                    split: dict(counts) for split, counts in self.split_sizes.items()
                },
                "note": (
                    "Reviews inherit their product's split (no product appears in "
                    "more than one split); assignment is stratified and seeded via "
                    "settings.split."
                ),
            },
            "files": list(self.output_files),
            "report_files": list(self.report_files),
        }


# ------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------


def _get_or_na(frame: pd.DataFrame, column: str, dtype: str) -> pd.Series:
    """Return a column or a typed NA series when the source file lacks it.

    Args:
        frame: Source chunk.
        column: Column name to fetch.
        dtype: pandas dtype for the all-NA fallback series.

    Returns:
        The column when present, otherwise an all-NA series of ``dtype``.
    """
    if column in frame.columns:
        return frame[column]
    return pd.Series(pd.NA, index=frame.index, dtype=dtype)


def _string_or_none(value: Any) -> str | None:
    """Preserve a raw value as a display string, or None when truly missing.

    Only null/NaN/``pd.NA`` become ``None``; an empty string in the source stays
    ``""`` so the raw column remains faithful to the file.

    Args:
        value: Raw cell value.

    Returns:
        ``str(value)`` when present, else ``None``.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return str(value)


def _clean_review_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Turn a filtered raw review chunk into the cleaned review columns.

    Keeps original text (``review_text_raw``/``summary_raw``) beside the cleaned
    text (encoding fixes, URL removal, whitespace collapse - casing and
    punctuation preserved for the transformer), plus nullable canonical columns.

    Args:
        chunk: Raw review rows whose ``asin`` is already a candidate.

    Returns:
        DataFrame with cleaning columns (``review_id``/time/label columns are
        added later by the orchestrator).
    """
    text_raw = _get_or_na(chunk, "reviewText", "str")
    summary_raw = _get_or_na(chunk, "summary", "str")

    out = pd.DataFrame(index=chunk.index)
    out["asin"] = chunk["asin"]
    out["reviewer_id"] = _get_or_na(chunk, "reviewerID", "str")
    out["rating"] = pd.to_numeric(
        _get_or_na(chunk, "rating", "float64"), errors="coerce"
    ).astype("float64")
    out["review_text_raw"] = [_string_or_none(v) for v in text_raw]
    out["review_text_clean"] = [clean_review_text(v) for v in text_raw]
    out["summary_raw"] = [_string_or_none(v) for v in summary_raw]
    out["summary_clean"] = [clean_review_text(v) for v in summary_raw]
    out["helpful_votes"] = normalize_helpful_votes(_get_or_na(chunk, "vote", "Int64"))
    out["verified"] = _get_or_na(chunk, "verified", "boolean")
    out["unixReviewTime"] = pd.to_numeric(
        _get_or_na(chunk, "unixReviewTime", "float64"), errors="coerce"
    )
    return out


def _first_image_url(highres: Any, small: Any) -> str | None:
    """Return the first usable image URL for a row, high-resolution preferred.

    The loader normalises ``imageURLHighRes``/``imageURL`` into list columns
    (``image_urls_highres``/``image_urls``), so a row carries the full preference
    order already: take the first high-res URL, else the first small URL, else
    nothing. Nothing is invented when both are absent/empty.

    Args:
        highres: ``image_urls_highres`` cell (list of URLs or missing).
        small: ``image_urls`` cell (list of URLs or missing).

    Returns:
        The chosen absolute URL, or None when the row has no usable image URL.
    """
    for value in (highres, small):
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return str(value[0])
    return None


def _first_image_by_asin(products_all: pd.DataFrame) -> dict[str, str]:
    """Map ASIN -> first usable image URL across *every* duplicate metadata row.

    The build keeps the first row per ASIN (file order) for every product field.
    A duplicate row of the same ASIN can still be the only one that carries an
    image URL, so the product's image is resolved over all rows of that ASIN
    *before* the duplicate rows are dropped. Because rows are scanned in file
    order and the kept row is the first one, the kept row's own image always wins
    when it has one; a dropped duplicate only ever fills a gap.

    Args:
        products_all: Candidate metadata rows *before* de-duplication.

    Returns:
        ``{asin: first usable image URL}`` for ASINs that have one anywhere.
    """
    asin_series = _get_or_na(products_all, "asin", "str")
    highres_series = _get_or_na(products_all, "image_urls_highres", "object")
    small_series = _get_or_na(products_all, "image_urls", "object")
    found: dict[str, str] = {}
    for asin, highres, small in zip(asin_series, highres_series, small_series, strict=True):
        if not isinstance(asin, str) or asin in found:
            continue
        image = _first_image_url(highres, small)
        if image is not None:
            found[asin] = image
    return found


def _clean_product_frame(
    products: pd.DataFrame, image_lookup: dict[str, str] | None = None
) -> pd.DataFrame:
    """Clean product fields for the candidate subset.

    Strips HTML from title/description/features, normalises brand into a
    comparison key, parses rank into ``(rank_value, rank_category)``, parses the
    ``category`` breadcrumb into ``(category_path, category_leaf)``, flags
    missing/unparseable price, rank and brand with booleans, and resolves the
    display-only product image (``image_url`` plus its ``image_missing`` flag).
    Raw ``title``, ``brand`` and ``rank`` are preserved for display; the source
    ``category`` list is read but not stored (``category_path`` *is* its cleaned
    form); nothing is imputed.

    Args:
        products: Raw metadata rows for candidates (``asin`` unique).
        image_lookup: Optional ``{asin: image URL}`` resolved across duplicate
            rows *before* de-duplication, used only to fill a kept row whose own
            image fields are empty (see :func:`_first_image_by_asin`).

    Returns:
        Cleaned product frame (without review aggregates, which the orchestrator
        merges afterwards).
    """
    title_raw = _get_or_na(products, "title", "str")
    brand_raw = _get_or_na(products, "brand", "str")
    rank_raw = _get_or_na(products, "rank", "str")

    out = pd.DataFrame(index=products.index)
    out["asin"] = products["asin"]
    out["title"] = [_string_or_none(v) for v in title_raw]
    out["title_clean"] = [clean_title(v) for v in title_raw]
    out["brand"] = [_string_or_none(v) for v in brand_raw]
    out["brand_clean"] = [normalize_brand(v) for v in brand_raw]
    out["brand_missing"] = out["brand_clean"].isna()

    out["price"] = pd.to_numeric(
        _get_or_na(products, "price", "float64"), errors="coerce"
    ).astype("float64")
    out["price_max"] = pd.to_numeric(
        _get_or_na(products, "price_max", "float64"), errors="coerce"
    ).astype("float64")
    out["price_status"] = [_string_or_none(v) for v in _get_or_na(products, "price_status", "str")]
    # price_missing covers both "absent" and "present but unparseable" - the
    # loader leaves price null in both cases (price_status records which).
    out["price_missing"] = out["price"].isna()

    out["rank"] = [_string_or_none(v) for v in rank_raw]
    parsed = [parse_rank_fields(v) for v in rank_raw]
    out["rank_value"] = pd.Series(
        [value for value, _ in parsed], index=products.index, dtype="Int64"
    )
    out["rank_category"] = pd.Series(
        [category for _, category in parsed], index=products.index, dtype="object"
    )
    # rank_missing means "no numeric rank could be derived" (unparsable counts).
    out["rank_missing"] = out["rank_value"].isna()

    # Browse breadcrumb: the raw `category` field (list of strings, single string
    # or missing) becomes the typed " > "-joined path plus its most specific
    # level; both stay None when the source published nothing usable.
    parsed_paths = [
        parse_category_path(value) for value in _get_or_na(products, "category", "object")
    ]
    out["category_path"] = pd.Series(
        [path for path, _ in parsed_paths], index=products.index, dtype="object"
    )
    out["category_leaf"] = pd.Series(
        [leaf for _, leaf in parsed_paths], index=products.index, dtype="object"
    )

    description_raw = _get_or_na(products, "description", "object")
    feature_raw = _get_or_na(products, "feature", "object")
    out["description_clean"] = [clean_description(v) for v in description_raw]
    # list[Any]: a column of lists-or-None is not expressible in the pandas-stubs
    # `__setitem__` union (None is neither a scalar nor a Sequence).
    features_clean: list[Any] = [clean_features(v) for v in feature_raw]
    out["features_clean"] = features_clean

    # Display-only image URL (high-res preferred, then the small thumbnail). The
    # kept row's own image wins; a duplicate row of the same ASIN only fills a
    # gap (``image_lookup`` was built before de-duplication).
    highres_series = _get_or_na(products, "image_urls_highres", "object")
    small_series = _get_or_na(products, "image_urls", "object")
    row_images = [
        _first_image_url(highres, small)
        for highres, small in zip(highres_series, small_series, strict=True)
    ]
    if image_lookup:
        row_images = [
            image if image is not None else image_lookup.get(str(asin))
            for image, asin in zip(row_images, products["asin"], strict=True)
        ]
    out["image_url"] = pd.Series(row_images, index=products.index, dtype="object")
    out["image_missing"] = out["image_url"].isna()
    return out


def _write_parquet(df: pd.DataFrame, schema: pa.Schema, path: Path) -> int:
    """Write a DataFrame to parquet with an explicit pyarrow schema.

    Args:
        df: Frame whose columns match ``schema`` names (order enforced here).
        schema: Target schema (dtypes are the pipeline contract).
        path: Destination ``.parquet`` file.

    Returns:
        Number of rows written.
    """
    ordered = df[list(schema.names)]
    table = pa.Table.from_pandas(ordered, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    return len(table)


def _iso_or_none(value: Any) -> str | None:
    """Format a timestamp as an ISO-8601 UTC string, or None for NaT/missing.

    Args:
        value: Timestamp-like value.

    Returns:
        ISO-8601 string with UTC offset, or ``None``.
    """
    if value is None or pd.isna(value):
        return None
    # `value` is a pandas Timestamp / datetime at runtime, but the Any parameter
    # hides that from mypy: narrow the already-string result instead of returning Any.
    return cast(str, value.isoformat())


# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------


def build_processed_dataset(
    settings: Settings,
    *,
    config_path: str = "configs/config.yaml",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> BuildReport:
    """Build the product/review/related parquet tables, ``meta.json`` and the report.

    Outputs: ``products.parquet`` (with a ``split`` column), ``reviews.parquet``
    (with a ``split`` column), ``related_products.parquet``, ``meta.json`` in
    ``processed_dir`` - plus ``reports/data_quality_report.{json,md}`` in
    ``reports_dir``. The split is assigned per *product* (stratified by
    average-rating bin and review-count bin, seeded from ``dataset.split``) and
    every review inherits its product's split, so no product leaks across splits.

    Args:
        settings: Loaded project settings (category, paths, thresholds, mapping).
        config_path: Config file path recorded in ``meta.json``.
        chunk_size: Records per streamed chunk.
        raw_dir: Override for ``paths.raw_data_dir`` (tests).
        processed_dir: Override for ``paths.processed_data_dir`` (tests).
        reports_dir: Override for ``paths.reports_dir`` (tests).

    Returns:
        :class:`BuildReport` with all drop/keep accounting, split sizes and
        output facts.

    Raises:
        DataError: When the category has no registered sources, raw files are
            missing, or no product meets the candidate threshold.
    """
    category = settings.dataset.category
    source = settings.dataset.sources.get(category)
    if source is None:
        registered = ", ".join(sorted(settings.dataset.sources)) or "<none>"
        raise DataError(
            f"No dataset.sources entry for category '{category}' in {config_path}. "
            f"Registered: {registered}."
        )

    raw_dir = raw_dir if raw_dir is not None else settings.paths.raw_data_dir
    processed_dir = (
        processed_dir if processed_dir is not None else settings.paths.processed_data_dir
    )
    reports_dir = reports_dir if reports_dir is not None else settings.paths.reports_dir
    reviews_path = raw_dir / Path(source.reviews_url).name
    metadata_path = raw_dir / Path(source.metadata_url).name
    missing = [str(p) for p in (reviews_path, metadata_path) if not p.exists()]
    if missing:
        raise DataError(
            "Raw file(s) not found: "
            + ", ".join(missing)
            + f". Download them first: `python scripts/download_data.py --category {category}`."
        )

    min_reviews = settings.dataset.audit.candidate_min_reviews
    rating_mapping = settings.review_intel.sentiment.rating_mapping
    split_config = settings.dataset.split
    report = BuildReport(
        category=category,
        config_path=config_path,
        candidate_min_reviews=min_reviews,
        chunk_size=chunk_size,
        rating_mapping={
            "negative_max": rating_mapping.negative_max,
            "neutral_val": rating_mapping.neutral_val,
            "positive_min": rating_mapping.positive_min,
        },
        split_settings=split_config.model_dump(),
        reviews_path=str(reviews_path),
        metadata_path=str(metadata_path),
        reviews_bytes=reviews_path.stat().st_size,
        metadata_bytes=metadata_path.stat().st_size,
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    logger.info(
        "building '%s' (min %d reviews/product, split seed %d, fractions %.2f/%.2f/%.2f)",
        category,
        min_reviews,
        split_config.seed,
        split_config.train_fraction,
        split_config.val_fraction,
        split_config.test_fraction,
    )

    # -- Pass M1: metadata ASIN set (constant memory) -------------------------
    metadata_asins: set[str] = set()
    for chunk in iter_metadata_chunks(
        metadata_path, chunk_size=chunk_size, category=category, stats=LoadStats()
    ):
        report.metadata_rows += len(chunk)
        present = [str(v) for v in _get_or_na(chunk, "asin", "str") if is_present(v)]
        report.metadata_rows_without_asin += len(chunk) - len(present)
        metadata_asins.update(present)
    report.metadata_duplicate_asin_rows = (
        report.metadata_rows - report.metadata_rows_without_asin - len(metadata_asins)
    )
    logger.info(
        "pass M1: %d metadata rows, %d distinct ASINs, %d rows without ASIN, "
        "%d duplicate-ASIN rows",
        report.metadata_rows,
        len(metadata_asins),
        report.metadata_rows_without_asin,
        report.metadata_duplicate_asin_rows,
    )

    # -- Pass 1: review count per product (constant memory) -------------------
    counts: pd.Series | None = None
    for chunk in iter_review_chunks(reviews_path, chunk_size=chunk_size, stats=LoadStats()):
        report.review_rows_scanned += len(chunk)
        present = [str(v) for v in _get_or_na(chunk, "asin", "str") if is_present(v)]
        report.review_rows_without_asin += len(chunk) - len(present)
        if not present:
            continue
        chunk_counts = pd.Series(present, dtype="str").value_counts()
        counts = chunk_counts if counts is None else counts.add(chunk_counts, fill_value=0)
    if counts is None or report.review_rows_scanned == 0:
        raise DataError(f"No parsable review rows in {reviews_path}.")
    counts = counts.fillna(0).astype("int64")

    # -- Candidate selection (config threshold, never hardcoded) --------------
    eligible = counts[counts >= min_reviews]
    report.products_below_min_reviews = int((counts < min_reviews).sum())
    eligible_asins = {str(asin) for asin in eligible.index}
    candidate_asins = eligible_asins & metadata_asins
    report.products_missing_metadata = len(eligible_asins - metadata_asins)
    report.products_selected = len(candidate_asins)
    logger.info(
        "candidate selection (>= %d reviews AND in metadata): kept %d, dropped "
        "%d below threshold, dropped %d missing metadata",
        min_reviews,
        report.products_selected,
        report.products_below_min_reviews,
        report.products_missing_metadata,
    )
    if not candidate_asins:
        raise DataError(
            f"No product meets dataset.audit.candidate_min_reviews={min_reviews} in "
            f"category '{category}' - nothing to build."
        )

    # -- Pass M2: full metadata rows for candidates ---------------------------
    product_frames: list[pd.DataFrame] = []
    for chunk in iter_metadata_chunks(
        metadata_path, chunk_size=chunk_size, category=category, stats=LoadStats()
    ):
        mask = _get_or_na(chunk, "asin", "str").isin(candidate_asins)
        if mask.any():
            product_frames.append(chunk.loc[mask])
    products_all = pd.concat(product_frames, ignore_index=True)
    before = len(products_all)
    # Resolve each product's image over ALL rows of its ASIN before duplicate rows
    # are dropped, so an image that only a dropped duplicate carries is not lost.
    image_by_asin = _first_image_by_asin(products_all)
    products_raw = products_all.drop_duplicates(subset=["asin"], keep="first")
    report.products_duplicate_rows_dropped = before - len(products_raw)

    # Related-products edges must be built from the raw rows (also_buy/also_view
    # are only kept in memory; they are represented by the edge table on disk).
    related = build_related_products_table(products_raw, kept_asins=candidate_asins)
    report.related_edges = len(related)
    report.related_in_subset = int(related["in_subset"].sum()) if len(related) else 0

    products = _clean_product_frame(products_raw, image_lookup=image_by_asin)
    logger.info(
        "pass M2: %d product rows collected (%d duplicate rows dropped)",
        len(products),
        report.products_duplicate_rows_dropped,
    )

    # -- Pass 2: full review rows for candidates ------------------------------
    review_frames: list[pd.DataFrame] = []
    for chunk in iter_review_chunks(reviews_path, chunk_size=chunk_size, stats=LoadStats()):
        mask = _get_or_na(chunk, "asin", "str").isin(candidate_asins)
        selected = int(mask.sum())
        if selected:
            review_frames.append(_clean_review_chunk(chunk.loc[mask]))
        report.review_rows_selected += selected
    if not review_frames:
        raise DataError(f"No reviews found for the {len(candidate_asins)} candidates.")
    reviews = pd.concat(review_frames, ignore_index=True)
    logger.info(
        "pass 2: %d/%d review rows kept for candidates (%d dropped: product not "
        "selected, incl. %d rows without ASIN)",
        report.review_rows_selected,
        report.review_rows_scanned,
        report.reviews_dropped_not_candidate,
        report.review_rows_without_asin,
    )

    # -- De-duplication (reviewer + asin + timestamp + normalised text) -------
    reviews = add_review_ids(reviews)
    reviews, report.duplicate_reviews_dropped = drop_duplicate_reviews(reviews)
    logger.info(
        "deduplication: %d duplicate review rows dropped", report.duplicate_reviews_dropped
    )

    # -- Review features ------------------------------------------------------
    reviews = add_review_time_features(reviews)
    reviews["has_text"] = reviews["review_text_clean"].notna()
    reviews["text_length"] = (
        reviews["review_text_clean"].str.len().fillna(0).astype("int32")
    )
    reviews["sentiment_label_weak"] = derive_weak_sentiment_labels(
        reviews["rating"], rating_mapping
    )
    report.reviews_final = len(reviews)

    # -- Product aggregates from the kept reviews -----------------------------
    aggregates = compute_product_review_aggregates(reviews)
    products = products.merge(aggregates, on="asin", how="left", validate="one_to_one")
    if products["n_reviews"].isna().any():
        raise DataError(
            "Internal inconsistency: some candidates have no kept reviews after "
            "cleaning/deduplication."
        )
    report.products_below_min_after_dedup = int(
        (products["n_reviews"] < min_reviews).sum()
    )
    report.products_final = len(products)

    # -- Product-level split assignment (leakage-safe, seeded) ----------------
    # Products are stratified by (average-rating bin, review-count bin) and
    # shuffled with dataset.split.seed; reviews inherit their product's split.
    products = assign_product_splits(products, split_config)
    reviews = assign_review_splits(reviews, products)
    report.split_sizes = split_sizes(products, reviews)
    logger.info(
        "split assignment (seed %d): products {%s}, reviews {%s}",
        split_config.seed,
        ", ".join(
            f"{split}: {report.split_sizes[split]['products']}" for split in SPLIT_ORDER
        ),
        ", ".join(
            f"{split}: {report.split_sizes[split]['reviews']}" for split in SPLIT_ORDER
        ),
    )

    final_reviews = reviews[[column.name for column in REVIEW_SCHEMA]]
    final_reviews = final_reviews.copy()
    final_reviews["rating"] = final_reviews["rating"].astype("float64")
    final_reviews["text_length"] = final_reviews["text_length"].astype("int32")
    final_reviews["helpful_votes"] = final_reviews["helpful_votes"].astype("int32")
    final_reviews["has_text"] = final_reviews["has_text"].astype(bool)
    final_reviews["verified"] = final_reviews["verified"].astype("boolean")
    final_reviews["review_datetime"] = final_reviews["review_datetime"].astype(
        "datetime64[us, UTC]"
    )
    final_reviews["review_year"] = final_reviews["review_year"].astype("Int16")
    final_reviews["review_month"] = final_reviews["review_month"].astype("Int8")
    final_reviews["sentiment_label_weak"] = final_reviews["sentiment_label_weak"].astype(
        "Int8"
    )
    final_reviews["split"] = final_reviews["split"].astype("string")

    final_products = products[[column.name for column in PRODUCT_SCHEMA]].copy()
    final_products["price"] = final_products["price"].astype("float64")
    final_products["price_max"] = final_products["price_max"].astype("float64")
    final_products["price_missing"] = final_products["price_missing"].astype(bool)
    final_products["brand_missing"] = final_products["brand_missing"].astype(bool)
    final_products["rank_value"] = final_products["rank_value"].astype("Int64")
    final_products["rank_missing"] = final_products["rank_missing"].astype(bool)
    final_products["image_missing"] = final_products["image_missing"].astype(bool)
    final_products["n_reviews"] = final_products["n_reviews"].astype("Int32")
    final_products["mean_rating"] = final_products["mean_rating"].astype("float64")
    final_products["first_review_time"] = final_products["first_review_time"].astype(
        "datetime64[us, UTC]"
    )
    final_products["last_review_time"] = final_products["last_review_time"].astype(
        "datetime64[us, UTC]"
    )
    final_products["split"] = final_products["split"].astype("string")

    # -- Dataset facts (dates come from the data, never from today) -----------
    review_dates = final_reviews["review_datetime"].dropna()
    if len(review_dates):
        report.review_date_min = _iso_or_none(review_dates.min())
        report.review_date_max = _iso_or_none(review_dates.max())

    # -- Write outputs --------------------------------------------------------
    processed_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[pd.DataFrame, pa.Schema, str]] = [
        (final_products, PRODUCT_SCHEMA, "products.parquet"),
        (final_reviews, REVIEW_SCHEMA, "reviews.parquet"),
        (related, RELATED_SCHEMA, "related_products.parquet"),
    ]
    for frame, schema, filename in written:
        rows = _write_parquet(frame, schema, processed_dir / filename)
        report.output_files.append(filename)
        logger.info("wrote %s (%d rows)", processed_dir / filename, rows)

    report.output_files.append(_META_FILENAME)
    meta_path = processed_dir / _META_FILENAME

    # -- Data-quality report (all values computed from these frames) ----------
    quality = build_quality_report(
        build=report,
        products_raw=products_raw,
        products=final_products,
        reviews=final_reviews,
        related=related,
        split_config=split_config,
    )
    report_paths = write_quality_report(quality, reports_dir)
    report.report_files = [path.name for path in report_paths]
    logger.info("wrote %s and %s", report_paths[0], report_paths[1])

    meta_path.write_text(
        json.dumps(report.to_meta_dict(), indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "build complete: %d products, %d reviews, %d related edges -> %s",
        report.products_final,
        report.reviews_final,
        report.related_edges,
        processed_dir,
    )
    return report

"""Streaming loaders for the UCSD Amazon Review Data (2018) gzip'd JSON-lines files.

Both file families are JSON Lines (one JSON object per line) inside a gzip container:
a per-category review file (``<Category>.json.gz``) and a per-category metadata file
(``meta_<Category>.json.gz``). Files are streamed line-by-line in chunks so that a full
category never has to fit in memory.

Design contract (project rule: *load faithfully*):
  * Nothing is dropped or filtered here. Malformed lines are skipped, logged and
    counted in :class:`LoadStats`; they never raise.
  * Known format variations are normalised into stable, typed columns while every raw
    field present in the file is preserved as-is on the DataFrame. Aliases applied:

    ================  =====================================================
    Canonical field   Accepted sources (first present wins)
    ================  =====================================================
    ``asin``          ``asin``, ``parentAsin`` (reviews), ``parent_asin`` (meta)
    ``rating``        ``rating``, ``overall``
    ``reviewText``    ``reviewText``, ``text``
    ``vote``          ``vote``, ``helpfulVote``
    ``verified``      ``verified``, ``verifiedPurchase``
    ``unixReviewTime``  ``unixReviewTime``, numeric ``timestamp``
    ``brand``         ``brand``, ``details["Brand"]`` (metadata)
    ``feature``       ``feature``, ``features`` (metadata)
    ``rank``          ``rank`` (string *or* list of strings), ``salesRank`` dict/str
                      (metadata; prefers the entry matching the active category)
    ``also_buy``      ``related.also_buy``, top-level ``also_buy`` (metadata)
    ``also_view``     ``related.also_view``, ``related.also_viewed``,
                      top-level ``also_viewed``/``also_view`` (metadata)
    ``image_urls_highres``  ``imageURLHighRes`` (metadata; list, single string or
                      missing - only absolute http(s) URLs are kept)
    ``image_urls``    ``imageURL`` (metadata; same normalisation as above)
    ================  =====================================================

  * Missing data stays missing (``None``/``<NA>``) - it is never replaced with a
    fabricated default. ``vote``/``verified`` use pandas nullable dtypes so "absent"
    and "false/zero" stay distinguishable.
  * ``price`` is parsed to ``(price, price_max, price_status)`` where ``price_status``
    is one of ``single``/``range``/``missing``/``unparseable``; the raw value is kept
    in ``price_raw`` so no information is lost.
  * ``rank`` accepts the metadata ``rank`` string (``">#1,234 in Appliances"``) or the
    ``salesRank`` dict (``{"Appliances": "1,234"}``) and is stored as ``rank``.
"""

from __future__ import annotations

import gzip
import json
import logging
import numbers
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import IO, Any, Literal

import numpy as np
import pandas as pd

from munasib.core.exceptions import DataError

logger = logging.getLogger("munasib.data.loaders")

DEFAULT_CHUNK_SIZE = 50_000
MAX_MALFORMED_EXAMPLES = 20

# Field names that must exist as columns once a file uses any of their aliases.
REVIEW_CANONICAL_COLUMNS = ("asin", "rating", "reviewText", "vote", "verified", "unixReviewTime")
METADATA_CANONICAL_COLUMNS = (
    "asin",
    "price",
    "price_max",
    "price_status",
    "price_raw",
    "rank",
    "brand",
    "description",
    "feature",
    "also_buy",
    "also_view",
    "image_urls_highres",
    "image_urls",
)

_LITERAL_TRUE = {"true", "t", "yes", "y", "1"}
_LITERAL_FALSE = {"false", "f", "no", "n", "0"}
_RANGE_SPLIT = re.compile(r"\s+-\s+|(?<=\d)-(?=\d)")
# A price fragment is only trusted when it is *exactly* a number after stripping
# currency symbols and thousands separators. Amazon's scraped "price" field can
# contain stray HTML/CSS that is full of digits - those must never be parsed into a
# fabricated amount, so anything else is "unparseable".
_AMOUNT_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")


@dataclass
class LoadStats:
    """Counters describing a streamed load (surfaced in audit reports)."""

    total_lines: int = 0
    parsed_records: int = 0
    empty_lines: int = 0
    malformed_lines: int = 0
    malformed_line_numbers: list[int] = field(default_factory=list)

    def record_malformed(self, line_no: int, reason: str) -> None:
        """Register one malformed line.

        Args:
            line_no: 1-based line number in the source file.
            reason: Human-readable parse failure reason (logged).
        """
        self.malformed_lines += 1
        if len(self.malformed_line_numbers) < MAX_MALFORMED_EXAMPLES:
            self.malformed_line_numbers.append(line_no)
            logger.warning("malformed line %d skipped: %s", line_no, reason)
        elif self.malformed_lines == MAX_MALFORMED_EXAMPLES + 1:
            logger.warning("further malformed-line examples suppressed (counting continues)")

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the counters."""
        return {
            "total_lines": self.total_lines,
            "parsed_records": self.parsed_records,
            "empty_lines": self.empty_lines,
            "malformed_lines": self.malformed_lines,
            "malformed_line_numbers": list(self.malformed_line_numbers),
        }


# ------------------------------------------------------------------------------
# Primitive parsers (format variations)
# ------------------------------------------------------------------------------


def parse_vote(value: Any) -> int | None:
    """Parse a helpful-vote count.

    Accepts ints (including numpy scalars), floats with no fraction, plain numeric
    strings and comma-formatted strings (``"1,234"``). Missing or unparsable values
    return ``None`` (never a fabricated 0).

    Args:
        value: Raw ``vote``/``helpfulVote`` value.

    Returns:
        Vote count, or None when the value is absent or unparseable.
    """
    if value is None or value is pd.NA or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        # `numbers.Real` is not typed as supporting int()/isna(), but every concrete
        # real type does: normalise through float first (the guard needs it anyway).
        as_float = float(value)
        return int(as_float) if as_float.is_integer() else None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def parse_bool(value: Any) -> bool | None:
    """Parse a boolean-ish value into ``True``/``False``/``None``.

    Args:
        value: Raw ``verified``/``verifiedPurchase`` value.

    Returns:
        Parsed boolean; None when absent or unparseable (unknown stays unknown).
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Integral):
        if int(value) == 1:
            return True
        if int(value) == 0:
            return False
        return None
    if isinstance(value, numbers.Real):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _LITERAL_TRUE:
            return True
        if text in _LITERAL_FALSE:
            return False
    return None


def parse_price(value: Any) -> tuple[float | None, float | None, str]:
    """Parse a metadata price field.

    Handles numeric prices, ``"$12.99"``, price ranges (``"$10.99 - $19.99"``),
    empty strings and unparsable text.

    Args:
        value: Raw price value (number, string or missing).

    Returns:
        Tuple ``(price, price_max, status)`` where ``status`` is one of
        ``"single"``, ``"range"``, ``"missing"`` or ``"unparseable"``. For ranges
        ``price`` is the first listed amount and ``price_max`` the second; for all
        other statuses ``price_max`` is None.
    """
    if value is None or value is pd.NA:
        return None, None, "missing"
    if isinstance(value, bool):
        return None, None, "unparseable"
    if isinstance(value, numbers.Real):
        # `pd.isna` is stubbed only for scalars pandas knows; float() is the value
        # that is actually returned below, so check NaN on it (same result).
        numeric_price = float(value)
        if pd.isna(numeric_price):  # NaN from a JSON null in a typed column
            return None, None, "missing"
        return numeric_price, None, "single"

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None, None, "missing"

    parts = [p.strip() for p in _RANGE_SPLIT.split(text) if p.strip()]
    if len(parts) == 2:
        first = _to_amount(parts[0])
        second = _to_amount(parts[1])
        if first is not None and second is not None:
            return first, second, "range"

    amount = _to_amount(text)
    if amount is not None:
        return amount, None, "single"
    return None, None, "unparseable"


def _to_amount(text: str) -> float | None:
    """Extract a monetary amount from a price fragment.

    Only ``"$1,099.00"``-style fragments are accepted: after stripping ``$``,
    thousands separators and surrounding whitespace the remainder must be purely
    numeric. Scraped HTML/CSS fragments therefore return None instead of a
    fabricated number.

    Args:
        text: One price fragment, e.g. ``"$12.99"`` or ``"1,099.00"``.

    Returns:
        Parsed float, or None when the fragment is not a clean amount.
    """
    cleaned = text.strip().replace("$", "").replace(",", "").strip()
    if _AMOUNT_PATTERN.fullmatch(cleaned):
        return float(cleaned)
    return None


def parse_rank(value: Any, sales_rank: Any = None, category: str | None = None) -> str | None:
    """Normalise the sales-rank field across file generations.

    The ``rank`` field appears as a plain string (``">#1,234 in Appliances"``) in some
    generations and as a *list* of rank strings in others; ``salesRank`` appears as a
    dict keyed by category (older files) or a string.

    Args:
        value: The ``rank`` field: string, list of strings, or missing.
        sales_rank: The legacy ``salesRank`` field: dict keyed by category, a plain
            string, a list, or missing/None.
        category: Category whose entry/string to prefer when several candidates exist;
            when omitted, the first non-empty candidate is used.

    Returns:
        Rank string, or None when no field carries one.
    """
    candidates: list[str] = []

    def _collect(raw: Any) -> None:
        if isinstance(raw, str) and raw.strip():
            candidates.append(raw.strip())
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    candidates.append(item.strip())
        elif isinstance(raw, dict):
            ordered: list[str] = []
            if category is not None:
                for key, entry in raw.items():
                    if str(key).strip().lower() == category.strip().lower() and entry:
                        ordered.append(str(entry).strip())
            for entry in raw.values():
                if entry is not None and str(entry).strip():
                    candidate = str(entry).strip()
                    if candidate not in ordered:
                        ordered.append(candidate)
            candidates.extend(ordered)

    _collect(value)
    _collect(sales_rank)
    if not candidates:
        return None
    if category is not None:
        lowered = category.strip().lower()
        for candidate in candidates:
            if lowered in candidate.lower():
                return candidate
    return candidates[0]


def to_str_list(value: Any) -> list[str] | None:
    """Normalise a related-product link field to a list of strings.

    Args:
        value: Raw list, string, or missing value.

    Returns:
        List of non-empty strings, an empty list when the source list is empty, or
        None when the field is absent/scalar (empty list and None both mean
        "no links"; None keeps "field absent" distinguishable for auditing).
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else None
    return None


def parse_image_urls(value: Any) -> list[str] | None:
    """Normalise an image-URL field to the list of usable absolute URLs.

    The UCSD metadata ``imageURL``/``imageURLHighRes`` fields are lists of image
    URLs, occasionally a single string and sometimes missing altogether. Only
    absolute ``http(s)`` URLs are kept, so a relative path, a ``data:`` URI or a
    placeholder is dropped rather than handed to a downloader.

    Args:
        value: Raw list, single string, or missing value.

    Returns:
        The trimmed absolute http(s) URLs in source order (empty list when the
        field is present but carries none), or None when the field is absent or
        of an unsupported shape - keeping "absent" distinguishable from "present
        but empty" for auditing.
    """
    candidates: list[str]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (list, tuple)):
        candidates = [str(v).strip() for v in value if v is not None and str(v).strip()]
    elif isinstance(value, str):
        text = value.strip()
        candidates = [text] if text else []
    else:
        return None
    return [url for url in candidates if url.startswith(("http://", "https://"))]


def is_present(value: Any) -> bool:
    """Return True when a value carries information (non-null and non-empty).

    Args:
        value: Any DataFrame cell value.

    Returns:
        True for non-empty strings/lists/dicts and any non-null scalar.
    """
    if value is None or value is pd.NA:
        return False
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip() != ""
    return True


# ------------------------------------------------------------------------------
# Canonicalisation (alias resolution; raw columns are preserved)
# ------------------------------------------------------------------------------


def _merge_alias(df: pd.DataFrame, canonical: str, sources: tuple[str, ...]) -> None:
    """Fill ``canonical`` per row from the first source that carries a present value.

    File generations mix formats *within the same file* (e.g. some rows carry ``asin``
    while others only carry ``parent_asin``), so alias resolution must work row by
    row rather than by whole-column presence.

    Args:
        df: Chunk DataFrame (modified in place).
        canonical: Target canonical column name.
        sources: Source column names, in order of preference.
    """
    available = [s for s in sources if s in df.columns]
    if canonical not in df.columns and not available:
        return
    n_rows = len(df)
    base = list(df[canonical]) if canonical in df.columns else [None] * n_rows
    source_cols = [list(df[s]) for s in available]
    merged: list[Any] = []
    for i in range(n_rows):
        value = base[i]
        if is_present(value):
            merged.append(value)
            continue
        fallback = value
        for col in source_cols:
            if is_present(col[i]):
                fallback = col[i]
                break
        merged.append(fallback)
    df[canonical] = merged


def canonicalize_review_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve review-file field aliases and normalise mixed-type columns.

    Args:
        df: Raw chunk of review records (all original fields preserved).

    Returns:
        The same chunk with canonical columns added/typed (``asin``, ``rating``,
        ``reviewText``, ``vote``, ``verified``, ``unixReviewTime``).
    """
    if df.empty:
        return df

    _merge_alias(df, "asin", ("asin", "parentAsin"))
    _merge_alias(df, "rating", ("rating", "overall"))
    _merge_alias(df, "reviewText", ("reviewText", "text"))
    _merge_alias(df, "vote", ("vote", "helpfulVote"))
    _merge_alias(df, "verified", ("verified", "verifiedPurchase"))

    # Numeric-only timestamp fallback: a date-string 'timestamp' column is never
    # reinterpreted as epoch seconds (values stay in their own column).
    if "timestamp" in df.columns:
        numeric_ts = pd.to_numeric(df["timestamp"], errors="coerce")
        base_ts = list(df["unixReviewTime"]) if "unixReviewTime" in df.columns else [None] * len(df)
        filled: list[Any] = []
        for i, base_value in enumerate(base_ts):
            if is_present(base_value):
                filled.append(base_value)
            elif not pd.isna(numeric_ts.iloc[i]):
                filled.append(float(numeric_ts.iloc[i]))
            else:
                filled.append(base_value)
        df["unixReviewTime"] = filled

    if "vote" in df.columns:
        df["vote"] = pd.Series(
            [parse_vote(v) for v in df["vote"]], index=df.index, dtype="Int64"
        )
    if "verified" in df.columns:
        df["verified"] = pd.Series(
            [parse_bool(v) for v in df["verified"]], index=df.index, dtype="boolean"
        )
    return df


def canonicalize_metadata_chunk(
    df: pd.DataFrame, category: str | None = None
) -> pd.DataFrame:
    """Resolve metadata-file field aliases and normalise mixed-type columns.

    Adds canonical columns: ``asin``, ``price``/``price_max``/``price_status``/
    ``price_raw``, ``rank``, ``brand``, ``feature``, ``also_buy``, ``also_view``,
    ``image_urls_highres`` (from ``imageURLHighRes``) and ``image_urls`` (from
    ``imageURL``). The raw ``related`` dict stays on the row untouched.

    Args:
        df: Raw chunk of metadata records (all original fields preserved).
        category: Category name used to pick the matching ``salesRank`` dict entry.

    Returns:
        The same chunk with canonical columns added/typed.
    """
    if df.empty:
        return df

    _merge_alias(df, "asin", ("asin", "parent_asin"))
    _merge_alias(df, "feature", ("feature", "features"))

    # brand fallback: per-row details["Brand"] when the brand field is missing/empty
    if "details" in df.columns:
        base_brand = list(df["brand"]) if "brand" in df.columns else [None] * len(df)
        details_list = list(df["details"])
        merged_brand: list[Any] = []
        for brand_value, details in zip(base_brand, details_list, strict=True):
            if is_present(brand_value):
                merged_brand.append(brand_value)
            elif isinstance(details, dict):
                fallback = details.get("Brand") or details.get("brand")
                merged_brand.append(fallback if fallback else brand_value)
            else:
                merged_brand.append(brand_value)
        df["brand"] = merged_brand

    # related-product links: per-row merge of the nested "related" dict with the
    # top-level also_buy / also_view / also_viewed variants
    top_also_buy = list(df["also_buy"]) if "also_buy" in df.columns else None
    top_also_view = None
    for name in ("also_view", "also_viewed"):
        if name in df.columns:
            top_also_view = list(df[name])
            break
    if "related" in df.columns:
        related = list(df["related"])
        merged_buy: list[Any] = []
        merged_view: list[Any] = []
        for i, rel in enumerate(related):
            buy = rel.get("also_buy") if isinstance(rel, dict) else None
            view = (
                rel.get("also_view") or rel.get("also_viewed")
                if isinstance(rel, dict)
                else None
            )
            if buy is None and top_also_buy is not None:
                buy = top_also_buy[i]
            if view is None and top_also_view is not None:
                view = top_also_view[i]
            merged_buy.append(buy)
            merged_view.append(view)
        df["also_buy"] = merged_buy
        df["also_view"] = merged_view
    else:
        df["also_buy"] = top_also_buy if top_also_buy is not None else [None] * len(df)
        df["also_view"] = top_also_view if top_also_view is not None else [None] * len(df)

    # Annotated as list[Any]: a column of lists-or-None is not expressible in the
    # pandas-stubs `__setitem__` union (None is neither a scalar nor a Sequence).
    also_buy_cells: list[Any] = [to_str_list(v) for v in df["also_buy"]]
    also_view_cells: list[Any] = [to_str_list(v) for v in df["also_view"]]
    df["also_buy"] = also_buy_cells
    df["also_view"] = also_view_cells

    # image URLs: normalise the raw camelCase fields into canonical list columns
    # (list, single string or missing) keeping only absolute http(s) URLs.
    image_fields = (
        ("image_urls_highres", "imageURLHighRes"),
        ("image_urls", "imageURL"),
    )
    for canonical, source in image_fields:
        raw_images: list[Any] = list(df[source]) if source in df.columns else [None] * len(df)
        # Annotated as list[Any] for the same pandas-stubs reason as above.
        image_cells: list[Any] = [parse_image_urls(v) for v in raw_images]
        df[canonical] = image_cells

    # price (price_raw is the canonical raw source once canonicalisation has run)
    if "price_raw" in df.columns:
        raw_values: list[Any] = list(df["price_raw"])
    elif "price" in df.columns:
        raw_values = list(df["price"])
    else:
        raw_values = [None] * len(df)
    parsed = [parse_price(v) for v in raw_values]
    df["price_raw"] = raw_values
    df["price"] = pd.Series([p[0] for p in parsed], index=df.index, dtype="float64")
    df["price_max"] = pd.Series([p[1] for p in parsed], index=df.index, dtype="float64")
    df["price_status"] = [p[2] for p in parsed]

    # sales rank (rank string field or salesRank dict)
    rank_values = df["rank"] if "rank" in df.columns else pd.Series([None] * len(df), index=df.index)
    sales_values = (
        df["salesRank"] if "salesRank" in df.columns else pd.Series([None] * len(df), index=df.index)
    )
    df["rank"] = [
        parse_rank(r, s, category=category)
        for r, s in zip(rank_values, sales_values, strict=False)
    ]
    return df


# ------------------------------------------------------------------------------
# Streaming readers
# ------------------------------------------------------------------------------


def open_maybe_gzip(path: Path | str) -> IO[str]:
    """Open a file for text reading, transparently handling ``.gz`` compression.

    Args:
        path: Path to a ``.gz`` or plain UTF-8 text file.

    Returns:
        Opened text file object.

    Raises:
        DataError: If the file is missing or the gzip stream is corrupt.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"Dataset file not found: {path}")
    try:
        if path.suffix == ".gz":
            return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
        return path.open("rt", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise DataError(f"Cannot open dataset file {path}: {exc}") from exc


def iter_jsonl_records(
    path: Path | str,
    stats: LoadStats | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream parsed JSON objects from a (optionally gzipped) JSON-lines file.

    Args:
        path: Path to the ``.json``/``.jsonl``/``.json.gz`` file.
        stats: Optional counters object, updated in place as lines are consumed.

    Yields:
        Parsed record dictionaries in file order.

    Raises:
        DataError: If the file is missing or the gzip stream is corrupt mid-read.
    """
    stats = stats if stats is not None else LoadStats()
    path = Path(path)
    try:
        with open_maybe_gzip(path) as handle:
            for line_no, line in enumerate(handle, start=1):
                stats.total_lines += 1
                text = line.strip()
                if not text:
                    stats.empty_lines += 1
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    stats.record_malformed(line_no, f"JSON decode error: {exc}")
                    continue
                if not isinstance(record, dict):
                    stats.record_malformed(
                        line_no, f"expected JSON object, got {type(record).__name__}"
                    )
                    continue
                stats.parsed_records += 1
                yield record
    except gzip.BadGzipFile as exc:
        raise DataError(f"Corrupt gzip stream while reading {path}: {exc}") from exc
    except EOFError as exc:
        raise DataError(f"Truncated gzip stream while reading {path}: {exc}") from exc


def iter_jsonl_chunks(
    path: Path | str,
    kind: Literal["review", "metadata"],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stats: LoadStats | None = None,
    category: str | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream a JSON-lines file into canonicalised pandas DataFrames.

    Args:
        path: Path to the gzipped (or plain) JSON-lines file.
        kind: ``"review"`` for the category review file, ``"metadata"`` for
            ``meta_<Category>``.
        chunk_size: Number of records per yielded DataFrame.
        stats: Optional counters object, updated in place.
        category: Category name passed to metadata rank normalisation.

    Yields:
        Canonicalised DataFrames of at most ``chunk_size`` rows (the final chunk may
        be smaller). Empty files yield nothing.

    Raises:
        DataError: If the file is missing or its gzip stream is corrupt.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    stats = stats if stats is not None else LoadStats()
    if kind == "review":
        canonicalize: Callable[[pd.DataFrame], pd.DataFrame] = canonicalize_review_chunk
    else:
        canonicalize = partial(canonicalize_metadata_chunk, category=category)

    buffer: list[dict[str, Any]] = []
    for record in iter_jsonl_records(path, stats=stats):
        buffer.append(record)
        if len(buffer) >= chunk_size:
            yield canonicalize(pd.DataFrame(buffer))
            buffer = []
    if buffer:
        yield canonicalize(pd.DataFrame(buffer))


def iter_review_chunks(
    path: Path | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stats: LoadStats | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream the per-category review file in chunks.

    Args:
        path: Path to ``<Category>.json.gz``.
        chunk_size: Records per chunk.
        stats: Optional counters object, updated in place.

    Yields:
        Canonicalised review DataFrames.
    """
    yield from iter_jsonl_chunks(path, kind="review", chunk_size=chunk_size, stats=stats)


def iter_metadata_chunks(
    path: Path | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stats: LoadStats | None = None,
    category: str | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream the per-category metadata file in chunks.

    Args:
        path: Path to ``meta_<Category>.json.gz``.
        chunk_size: Records per chunk.
        stats: Optional counters object, updated in place.
        category: Category used to select the matching ``salesRank`` dict entry.

    Yields:
        Canonicalised metadata DataFrames.
    """
    yield from iter_jsonl_chunks(
        path, kind="metadata", chunk_size=chunk_size, stats=stats, category=category
    )

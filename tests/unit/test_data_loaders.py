"""Unit tests for the streaming UCSD Amazon Review Data (2018) loaders.

Covers every format variation the loaders must tolerate (helpful-vote forms, price
forms, rank forms, related-product link forms, field aliases, boolean forms) plus
malformed-line handling, chunk boundaries, gzip transparency and error paths.
Tests run against tiny fixtures in ``tests/fixtures/`` and never touch the network.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from munasib.core.exceptions import DataError
from munasib.data.loaders import (
    LoadStats,
    canonicalize_metadata_chunk,
    canonicalize_review_chunk,
    is_present,
    iter_jsonl_chunks,
    iter_jsonl_records,
    iter_metadata_chunks,
    iter_review_chunks,
    parse_bool,
    parse_image_urls,
    parse_price,
    parse_rank,
    parse_vote,
    to_str_list,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REVARIATIONS = FIXTURES / "reviews_variations.jsonl"
MREVIATIONS = FIXTURES / "metadata_variations.jsonl"
MALFORMED = FIXTURES / "reviews_malformed.jsonl"
TS_ALIAS = FIXTURES / "reviews_timestamp_alias.jsonl"
TS_STRING = FIXTURES / "reviews_timestamp_string.jsonl"


# ------------------------------------------------------------------------------
# Primitive parsers
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (5, 5),
        ("2", 2),
        ("1,234", 1234),
        (" 42 ", 42),
        (None, None),
        ("", None),
        ("   ", None),
        ("abc", None),
        (2.0, 2),
        (2.7, None),
        (True, None),
        (False, None),
    ],
    ids=[
        "int", "numeric-string", "comma-string", "padded-string",
        "missing", "empty-string", "blank-string", "garbage",
        "integral-float", "fractional-float", "bool-true", "bool-false",
    ],
)
def test_parse_vote(raw: object, expected: int | None) -> None:
    """Vote may be absent, an int, a string or comma-formatted; booleans are unknown."""
    assert parse_vote(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("FALSE", False),
        ("yes", True),
        ("no", False),
        (1, True),
        (0, False),
        (None, None),
        ("maybe", None),
    ],
    ids=[
        "bool-true", "bool-false", "str-true", "str-false", "str-yes", "str-no",
        "int-one", "int-zero", "missing", "garbage",
    ],
)
def test_parse_bool(raw: object, expected: bool | None) -> None:
    """Verified flags may be bool, string or absent; unparseable stays unknown."""
    assert parse_bool(raw) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (12.99, (12.99, None, "single")),
        ("$12.99", (12.99, None, "single")),
        ("12.99", (12.99, None, "single")),
        ("$1,099.00", (1099.0, None, "single")),
        ("$10.99 - $19.99", (10.99, 19.99, "range")),
        ("10.99-19.99", (10.99, 19.99, "range")),
        ("", (None, None, "missing")),
        (None, (None, None, "missing")),
        ("nan", (None, None, "missing")),
        (float("nan"), (None, None, "missing")),
        (".a-box-inner{color:rgba(1,2,3,.5)}", (None, None, "unparseable")),
        ("N/A", (None, None, "unparseable")),
    ],
    ids=[
        "number", "dollar-string", "plain-string", "comma-string",
        "range-spaces", "range-tight",
        "empty-string", "missing", "nan-string", "nan-float",
        "css-garbage", "na-string",
    ],
)
def test_parse_price(raw: object, expected: tuple[float | None, float | None, str]) -> None:
    """Price covers numbers, $-strings, ranges, empties and scraped garbage."""
    assert parse_price(raw) == expected


@pytest.mark.parametrize(
    ("value", "sales_rank", "category", "expected"),
    [
        (">#1,234 in Appliances", None, None, ">#1,234 in Appliances"),
        (
            [">#1,234 in Appliances (See top 100)", "#56 in Parts & Accessories"],
            None,
            "Appliances",
            ">#1,234 in Appliances (See top 100)",
        ),
        ([], None, None, None),
        (None, {"Appliances": "2,345", "Home & Kitchen": "9,999"}, "Appliances", "2,345"),
        (None, {"Home & Kitchen": "9,999"}, "Appliances", "9,999"),
        (None, {"Home & Kitchen": "9,999"}, None, "9,999"),
        (None, None, None, None),
        (["", "  "], None, None, None),
    ],
    ids=["plain-str", "list-prefers-category", "empty-list", "dict-category",
         "dict-fallback-first", "dict-fallback-no-category", "nothing", "blank-list"],
)
def test_parse_rank(
    value: object, sales_rank: object, category: str | None, expected: str | None
) -> None:
    """Rank accepts a string, a list of strings or the legacy salesRank dict."""
    assert parse_rank(value, sales_rank, category=category) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["B000000001", "B000000002"], ["B000000001", "B000000002"]),
        ([], []),
        (None, None),
        ("B000000001", ["B000000001"]),
        ("  ", None),
        (["B000000001", None, ""], ["B000000001"]),
        (float("nan"), None),
    ],
    ids=["list", "empty-list", "missing", "scalar-string", "blank",
         "list-with-holes", "nan"],
)
def test_to_str_list(raw: object, expected: list[str] | None) -> None:
    """Related-product link fields normalise to lists; absent stays distinguishable."""
    assert to_str_list(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["https://a/1.jpg", "http://b/2.jpg"], ["https://a/1.jpg", "http://b/2.jpg"]),
        ("https://a/1.jpg", ["https://a/1.jpg"]),
        ("  https://a/1.jpg  ", ["https://a/1.jpg"]),
        ([], []),
        ("", []),
        (None, None),
        (float("nan"), None),
        (["not-a-url", "/relative/x.jpg"], []),
        (["data:image/png;base64,AAAA"], []),
        (["https://a/1.jpg", None, "", "ftp://x/y.jpg"], ["https://a/1.jpg"]),
        (123, None),
    ],
    ids=["list", "scalar-string", "padded-string", "empty-list", "blank", "missing",
         "nan", "no-absolute-url", "data-uri", "mixed", "scalar-number"],
)
def test_parse_image_urls(raw: object, expected: list[str] | None) -> None:
    """Image URL fields keep only absolute http(s) URLs; absent stays distinguishable."""
    assert parse_image_urls(raw) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("text", True),
        ("", False),
        ("   ", False),
        ([], False),
        (["x"], True),
        ({}, False),
        ({"a": 1}, True),
        (None, False),
        (pd.NA, False),
        (float("nan"), False),
        (0, True),
        (1, True),
    ],
    ids=["text", "empty", "blank", "empty-list", "list", "empty-dict", "dict",
         "none", "pd-na", "nan", "zero", "one"],
)
def test_is_present(value: object, expected: bool) -> None:
    """Presence means non-null and non-empty; zero counts as present."""
    assert is_present(value) is expected


# ------------------------------------------------------------------------------
# Review chunk canonicalisation
# ------------------------------------------------------------------------------


def _load_review_frame(path: Path) -> pd.DataFrame:
    """Load a fixture file as a single canonical review chunk.

    Args:
        path: Path to the JSON-lines fixture.

    Returns:
        Canonicalised DataFrame.
    """
    return next(iter(iter_review_chunks(path, chunk_size=100)))


def test_review_vote_forms() -> None:
    """Vote: missing key -> <NA>, plain string -> int, comma-formatted string -> int."""
    df = _load_review_frame(REVARIATIONS)
    assert str(df["vote"].dtype) == "Int64"
    assert df["vote"].iloc[0] == 2
    assert df["vote"].iloc[2] == 1234
    assert pd.isna(df["vote"].iloc[1])
    assert pd.isna(df["vote"].iloc[3])
    assert pd.isna(df["vote"].iloc[4])  # empty string is not zero votes


def test_review_verified_forms() -> None:
    """Verified: bool, alias string forms and absent -> nullable boolean with <NA>."""
    df = _load_review_frame(REVARIATIONS)
    assert str(df["verified"].dtype) == "boolean"
    assert df["verified"].iloc[0]  # True
    assert not df["verified"].iloc[1]  # False
    assert df["verified"].iloc[3]  # "true" via verifiedPurchase alias
    assert df["verified"].iloc[5]  # "yes"
    assert pd.isna(df["verified"].iloc[4])  # absent must stay unknown, not False


def test_review_aliases() -> None:
    """overall->rating, text->reviewText, verifiedPurchase->verified, parentAsin->asin."""
    df = _load_review_frame(REVARIATIONS)
    assert df["rating"].iloc[3] == 4.0
    assert df["reviewText"].iloc[3] == "Decent value for the price."
    assert df["asin"].iloc[3] == "B000000003"
    # raw alias columns are preserved, not dropped
    assert "text" in df.columns and "verifiedPurchase" in df.columns
    assert "parentAsin" in df.columns and "overall" in df.columns


def test_review_text_missing_and_empty_kept_faithful() -> None:
    """Missing reviewText stays missing; empty string stays empty (never dropped)."""
    df = _load_review_frame(REVARIATIONS)
    assert len(df) == 6
    assert pd.isna(df["reviewText"].iloc[2])
    assert df["reviewText"].iloc[4] == ""


def test_review_timestamp_numeric_alias() -> None:
    """Numeric-only 'timestamp' columns are aliased into unixReviewTime."""
    df = _load_review_frame(TS_ALIAS)
    assert df["unixReviewTime"].iloc[0] == 1609459200
    assert df["unixReviewTime"].iloc[1] == 1577836800


def test_review_timestamp_string_not_reinterpreted() -> None:
    """String timestamps are never reinterpreted; only numeric rows get the alias."""
    df = _load_review_frame(TS_STRING)
    assert pd.isna(df["unixReviewTime"].iloc[0])  # "December 1, 2020" stays put
    assert df["timestamp"].iloc[0] == "December 1, 2020"
    assert df["unixReviewTime"].iloc[1] == 1577836800  # numeric row still aliased


def test_canonicalize_review_chunk_on_raw_frame() -> None:
    """Direct canonicalisation of an already-materialised frame is idempotent."""
    raw = pd.DataFrame(
        [{"overall": 5.0, "vote": "3", "verified": True, "asin": "B00TEST"}]
    )
    once = canonicalize_review_chunk(raw.copy())
    twice = canonicalize_review_chunk(once.copy())
    pd.testing.assert_frame_equal(once, twice)


# ------------------------------------------------------------------------------
# Metadata chunk canonicalisation
# ------------------------------------------------------------------------------


def _load_metadata_frame(path: Path = MREVIATIONS) -> pd.DataFrame:
    """Load the metadata fixture as a single canonical chunk (category Appliances).

    Args:
        path: Path to the JSON-lines fixture.

    Returns:
        Canonicalised DataFrame.
    """
    return next(iter(iter_metadata_chunks(path, chunk_size=100, category="Appliances")))


def test_metadata_price_forms() -> None:
    """Price: $-string, number, range, empty, missing, scraped CSS garbage."""
    df = _load_metadata_frame()
    assert df["price"].iloc[0] == 12.99 and df["price_status"].iloc[0] == "single"
    assert df["price"].iloc[1] == 9.99 and df["price_status"].iloc[1] == "single"
    assert df["price"].iloc[2] == 10.99 and df["price_max"].iloc[2] == 19.99
    assert df["price_status"].iloc[2] == "range"
    assert df["price_status"].iloc[3] == "missing"  # empty string
    assert df["price_status"].iloc[4] == "unparseable"  # CSS garbage must NOT parse
    assert df["price"].iloc[4] != df["price"].iloc[4]  # NaN, never a fabricated number
    assert df["price"].iloc[5] == 1099.0
    # raw values are preserved next to the parsed column
    assert df["price_raw"].iloc[0] == "$12.99"
    assert df["price_raw"].iloc[1] == 9.99


def test_metadata_rank_forms() -> None:
    """Rank: list (category-matching entry preferred), string, salesRank dict, empty."""
    df = _load_metadata_frame()
    assert df["rank"].iloc[0] == ">#1,234 in Appliances (See top 100)"
    assert df["rank"].iloc[1] == "2,345"  # empty rank list -> salesRank dict entry
    assert df["rank"].iloc[2] == "#99 in Appliances"
    assert pd.isna(df["rank"].iloc[3])  # rank field absent, no salesRank
    assert pd.isna(df["rank"].iloc[4])
    assert df["rank"].iloc[5] == "#7 in Appliances"


def test_metadata_related_links_forms() -> None:
    """Related links: top-level lists, related.also_viewed alias, related.also_view,
    empty lists and absent fields."""
    df = _load_metadata_frame()
    assert df["also_buy"].iloc[0] == ["B000000009"]
    assert df["also_view"].iloc[0] == ["B000000008"]
    assert df["also_buy"].iloc[1] == []  # explicitly empty
    assert df["also_view"].iloc[1] == []
    assert df["also_buy"].iloc[2] == ["B000000007"]  # from related dict
    assert df["also_view"].iloc[2] == ["B000000006"]  # via also_viewed alias
    assert df["also_buy"].iloc[3] is None  # absent stays None (not [])
    assert df["also_view"].iloc[3] == ["B000000005"]  # via related.also_view


def test_metadata_brand_and_feature_forms() -> None:
    """Brand: field, details['Brand'] fallback, empty string; feature: features alias."""
    df = _load_metadata_frame()
    assert df["brand"].iloc[0] == "Acme"
    assert df["brand"].iloc[2] == "DetailsBrand"  # details fallback
    assert df["brand"].iloc[3] == "NoLinks Brand"
    assert not is_present(df["brand"].iloc[1])  # empty string counts as absent
    assert df["feature"].iloc[2] == ["F2", "F3"]  # via "features" alias
    assert "features" in df.columns  # raw column preserved


def test_metadata_asin_fallback() -> None:
    """parent_asin is used as the product key when asin is absent."""
    df = _load_metadata_frame()
    assert df["asin"].iloc[5] == "B000000006"


def test_metadata_image_forms() -> None:
    """Image fields: high-res list, empty lists, a single string, non-http dropped."""
    df = _load_metadata_frame()
    assert df["image_urls_highres"].iloc[0] == [
        "https://example.com/hi1.jpg",
        "https://example.com/hi2.jpg",
    ]
    assert df["image_urls"].iloc[0] == ["https://example.com/lo1.jpg"]
    assert df["image_urls_highres"].iloc[1] == []  # present but empty
    assert df["image_urls"].iloc[1] == []
    assert df["image_urls_highres"].iloc[2] is None  # absent key
    assert df["image_urls"].iloc[2] == ["https://example.com/c.jpg"]  # scalar string
    assert df["image_urls_highres"].iloc[3] == []  # only non-http URLs
    assert df["image_urls"].iloc[3] == []  # only a data: URI


def test_metadata_image_raw_columns_preserved() -> None:
    """The raw camelCase image columns are kept untouched beside the canonical ones."""
    df = _load_metadata_frame()
    assert "imageURLHighRes" in df.columns and "imageURL" in df.columns
    assert df["imageURL"].iloc[2] == "https://example.com/c.jpg"


def test_canonicalize_metadata_chunk_idempotent() -> None:
    """Re-canonicalising an already canonical metadata frame changes nothing."""
    once = _load_metadata_frame()
    twice = canonicalize_metadata_chunk(once.copy(), category="Appliances")
    pd.testing.assert_frame_equal(once, twice)


def test_metadata_description_forms() -> None:
    """Description keeps its native shape (list or string) without fabrication."""
    df = _load_metadata_frame()
    assert df["description"].iloc[0] == ["Desc one."]
    assert df["description"].iloc[3] == "plain string description"
    assert not is_present(df["description"].iloc[1])  # empty list


# ------------------------------------------------------------------------------
# Streaming: malformed lines, chunking, gzip transparency, error paths
# ------------------------------------------------------------------------------


def test_malformed_lines_are_counted_not_raised() -> None:
    """Invalid JSON, a JSON array and an empty line are skipped and counted."""
    stats = LoadStats()
    records = list(iter_jsonl_records(MALFORMED, stats=stats))
    assert stats.total_lines == 6
    assert stats.empty_lines == 1
    assert stats.parsed_records == 3
    assert stats.malformed_lines == 2
    assert stats.malformed_line_numbers == [2, 5]
    assert [r["asin"] for r in records] == ["B000000001", "B000000002", "B000000003"]


def test_malformed_lines_do_not_stop_chunk_stream() -> None:
    """A malformed middle line does not abort loading; good rows still arrive."""
    stats = LoadStats()
    frames = list(iter_review_chunks(MALFORMED, chunk_size=2, stats=stats))
    combined = pd.concat(frames, ignore_index=True)
    assert len(combined) == 3
    assert stats.malformed_lines == 2


def test_chunk_boundaries_and_shared_stats() -> None:
    """Chunk size is honoured and LoadStats accumulate across chunks."""
    stats = LoadStats()
    frames = list(iter_review_chunks(REVARIATIONS, chunk_size=2, stats=stats))
    assert [len(f) for f in frames] == [2, 2, 2]
    assert stats.parsed_records == 6
    assert stats.malformed_lines == 0


def test_no_rows_are_dropped() -> None:
    """Every parsed record becomes exactly one row (faithful loading)."""
    stats = LoadStats()
    df = pd.concat(iter_review_chunks(REVARIATIONS, chunk_size=3, stats=stats), ignore_index=True)
    assert len(df) == stats.parsed_records == 6


def test_gzip_transparency(tmp_path: Path) -> None:
    """A gzipped fixture streams identically to the plain-text fixture."""
    gz_path = tmp_path / "reviews_variations.jsonl.gz"
    with REVARIATIONS.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    stats_plain, stats_gz = LoadStats(), LoadStats()
    plain = pd.concat(iter_review_chunks(REVARIATIONS, stats=stats_plain), ignore_index=True)
    zipped = pd.concat(iter_jsonl_chunks(gz_path, kind="review", stats=stats_gz), ignore_index=True)
    pd.testing.assert_frame_equal(plain, zipped)
    assert stats_plain.summary() == stats_gz.summary()


def test_metadata_gzip_roundtrip(tmp_path: Path) -> None:
    """Metadata fixtures survive gzip round-trip with identical canonical output."""
    gz_path = tmp_path / "meta_variations.json.gz"
    with MREVIATIONS.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    plain = _load_metadata_frame()
    zipped = next(iter(iter_metadata_chunks(gz_path, chunk_size=100, category="Appliances")))
    pd.testing.assert_frame_equal(plain, zipped)


def test_missing_file_raises_data_error(tmp_path: Path) -> None:
    """A missing dataset file surfaces as DataError, not FileNotFoundError."""
    with pytest.raises(DataError):
        list(iter_review_chunks(tmp_path / "does_not_exist.json.gz"))


def test_corrupt_gzip_raises_data_error(tmp_path: Path) -> None:
    """A corrupt gzip container surfaces as DataError."""
    corrupt = tmp_path / "corrupt.json.gz"
    corrupt.write_bytes(b"this is definitely not gzip data")
    with pytest.raises(DataError):
        list(iter_review_chunks(corrupt))


def test_empty_file_yields_nothing(tmp_path: Path) -> None:
    """An empty (gzip) file yields no chunks and zeroed stats."""
    empty = tmp_path / "empty.json.gz"
    with gzip.open(empty, "wt", encoding="utf-8"):
        pass
    stats = LoadStats()
    frames = list(iter_review_chunks(empty, stats=stats))
    assert frames == []
    assert stats.parsed_records == 0
    assert stats.malformed_lines == 0


def test_invalid_chunk_size_rejected() -> None:
    """Chunk sizes below 1 are rejected explicitly."""
    with pytest.raises(ValueError):
        next(iter_jsonl_chunks(REVARIATIONS, kind="review", chunk_size=0))


def test_fixture_files_are_valid_jsonl() -> None:
    """Sanity: every fixture line parses (fixtures themselves must be clean JSON)."""
    for fixture in (REVARIATIONS, MREVIATIONS, TS_ALIAS, TS_STRING):
        for line in fixture.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)

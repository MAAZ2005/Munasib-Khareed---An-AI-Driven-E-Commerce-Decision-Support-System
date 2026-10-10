"""End-to-end tests for :mod:`munasib.data.preprocessing` (raw -> parquet build).

Runs the full four-pass build over gzipped fixture files in a temp directory and
verifies candidate selection (threshold read from config, never hardcoded), the
drop accounting, cleaning rules, explicit parquet schemas and ``meta.json``
facts. Hermetic: fixtures only - no network, no GPU, no real raw files.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

from munasib.core.config import (
    AuditRubricConfig,
    DatasetConfig,
    DatasetSourceConfig,
    Settings,
)
from munasib.core.exceptions import DataError
from munasib.data.preprocessing import (
    PRODUCT_SCHEMA,
    RELATED_SCHEMA,
    REVIEW_SCHEMA,
    build_processed_dataset,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _write_gzip_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as a gzipped JSON-lines file.

    Args:
        path: Destination path (``.gz`` suffix triggers gzip reading).
        records: One dict per line.
    """
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a plain JSON-lines fixture.

    Args:
        path: Fixture file.

    Returns:
        List of records.
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _settings(min_reviews: int, category: str = "Appliances") -> Settings:
    """Build settings with fixture-compatible sources and a test threshold.

    Args:
        min_reviews: ``dataset.audit.candidate_min_reviews`` for the test.
        category: Active category name.

    Returns:
        Settings instance.
    """
    return Settings(
        dataset=DatasetConfig(
            category=category,
            sources={
                "Appliances": DatasetSourceConfig(
                    reviews_url="https://example.invalid/Appliances.json.gz",
                    metadata_url="https://example.invalid/meta_Appliances.json.gz",
                )
            },
            audit=AuditRubricConfig(candidate_min_reviews=min_reviews),
        )
    )


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    """Raw directory containing the gzipped review/metadata fixtures."""
    target = tmp_path / "raw"
    target.mkdir()
    _write_gzip_jsonl(target / "Appliances.json.gz", _read_jsonl(FIXTURES / "reviews_variations.jsonl"))
    _write_gzip_jsonl(
        target / "meta_Appliances.json.gz", _read_jsonl(FIXTURES / "metadata_variations.jsonl")
    )
    return target


class TestBuildEndToEnd:
    """Full build over the fixtures with the production threshold semantics."""

    def test_candidate_selection_and_counts(self, raw_dir: Path, tmp_path: Path) -> None:
        """Threshold comes from config; only B000000001 has >= 2 reviews."""
        out = tmp_path / "processed"
        report = build_processed_dataset(
            _settings(min_reviews=2), raw_dir=raw_dir, processed_dir=out, chunk_size=2
        )
        assert report.candidate_min_reviews == 2
        assert report.products_selected == 1
        assert report.products_final == 1
        assert report.products_below_min_reviews == 4  # four products with 1 review
        assert report.products_missing_metadata == 0
        assert report.review_rows_scanned == 6
        assert report.review_rows_selected == 2
        assert report.reviews_dropped_not_candidate == 4
        assert report.reviews_final == 2
        assert report.duplicate_reviews_dropped == 0
        assert report.metadata_rows == 6
        assert report.metadata_duplicate_asin_rows == 0
        assert report.related_edges == 2
        assert report.related_in_subset == 0

    def test_products_table_content(self, raw_dir: Path, tmp_path: Path) -> None:
        """Cleaned product row: parsed price/rank/brand, aggregates merged."""
        out = tmp_path / "processed"
        build_processed_dataset(
            _settings(min_reviews=2), raw_dir=raw_dir, processed_dir=out, chunk_size=2
        )
        products = pd.read_parquet(out / "products.parquet")
        assert len(products) == 1
        row = products.iloc[0]
        assert row["asin"] == "B000000001"
        assert pd.isna(row["title"]) and pd.isna(row["title_clean"])  # fixture has no title
        assert row["brand"] == "Acme" and row["brand_clean"] == "acme"
        assert bool(row["brand_missing"]) is False
        assert float(row["price"]) == pytest.approx(12.99)
        assert row["price_status"] == "single"
        assert bool(row["price_missing"]) is False
        assert int(row["rank_value"]) == 1234
        assert row["rank_category"] == "Appliances"
        assert bool(row["rank_missing"]) is False
        assert row["description_clean"] == "Desc one."
        assert list(row["features_clean"]) == ["F1"]
        assert int(row["n_reviews"]) == 2
        assert str(row["first_review_time"]) == "2013-11-27 00:00:00+00:00"
        assert str(row["last_review_time"]) == "2018-01-01 00:00:00+00:00"

    def test_reviews_table_content(self, raw_dir: Path, tmp_path: Path) -> None:
        """Cleaned review rows: raw preserved, labels weak, votes 0-filled."""
        out = tmp_path / "processed"
        build_processed_dataset(
            _settings(min_reviews=2), raw_dir=raw_dir, processed_dir=out, chunk_size=2
        )
        reviews = pd.read_parquet(out / "reviews.parquet")
        assert len(reviews) == 2
        # Raw text preserved for display; cleaned text keeps casing.
        first = reviews[reviews["review_text_raw"] == "Works as advertised, no complaints."].iloc[0]
        assert first["review_text_clean"] == "Works as advertised, no complaints."
        assert bool(first["has_text"]) is True
        assert int(first["text_length"]) == len("Works as advertised, no complaints.")
        assert int(first["helpful_votes"]) == 2
        assert bool(first["verified"]) is True
        assert int(first["sentiment_label_weak"]) == 2  # rating 5 -> positive
        # Missing vote means 0 (documented zero-fill).
        second = reviews[reviews["review_text_raw"] == "Broke after one week, total waste of money."].iloc[0]
        assert int(second["helpful_votes"]) == 0
        assert int(second["sentiment_label_weak"]) == 0  # rating 1 -> negative
        assert bool(second["verified"]) is False
        assert "UTC" in str(reviews["review_datetime"].dtype)
        assert set(reviews["review_id"].str.len()) == {16}

    def test_related_table_content(self, raw_dir: Path, tmp_path: Path) -> None:
        """Edge table from also_buy/also_view with in_subset flags."""
        out = tmp_path / "processed"
        build_processed_dataset(
            _settings(min_reviews=2), raw_dir=raw_dir, processed_dir=out, chunk_size=2
        )
        related = pd.read_parquet(out / "related_products.parquet")
        pairs = set(zip(related["asin"], related["related_asin"], related["relation_type"], strict=True))
        assert pairs == {
            ("B000000001", "B000000009", "also_bought"),
            ("B000000001", "B000000008", "also_viewed"),
        }
        assert not related["in_subset"].any()  # neighbours are not candidates here

    def test_meta_json_facts(self, raw_dir: Path, tmp_path: Path) -> None:
        """meta.json records category, config, counts, dates, weak-label note."""
        out = tmp_path / "processed"
        report = build_processed_dataset(
            _settings(min_reviews=2), raw_dir=raw_dir, processed_dir=out, chunk_size=2
        )
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["category"] == "Appliances"
        assert meta["settings"]["candidate_min_reviews"] == 2
        assert meta["settings"]["rating_mapping"] == {
            "negative_max": 2,
            "neutral_val": 3,
            "positive_min": 4,
        }
        assert meta["row_counts"]["products_final"] == report.products_final
        assert meta["row_counts"]["reviews_final"] == report.reviews_final
        assert meta["row_counts"]["related_edges"] == report.related_edges
        assert meta["review_date_range"]["min"] == report.review_date_min
        assert meta["review_date_range"]["max"] == report.review_date_max
        assert meta["review_date_range"]["max"] == "2018-01-01T00:00:00+00:00"
        assert "never today" in meta["review_date_range"]["note"]
        assert "not true sentiment" in meta["weak_labels"]["note"]
        assert meta["drops"]["reviews_not_candidate_product"] == 4
        assert meta["inputs"]["reviews"]["bytes"] > 0
        assert set(meta["files"]) == {
            "products.parquet",
            "reviews.parquet",
            "related_products.parquet",
            "meta.json",
        }
        assert meta["built_at"].endswith("+00:00")


class TestBuildWithLowerThreshold:
    """Threshold flexibility: min=1 keeps every reviewed product in the fixtures."""

    def test_five_products_six_reviews(self, raw_dir: Path, tmp_path: Path) -> None:
        """All five reviewed products qualify at >= 1 review."""
        out = tmp_path / "processed"
        report = build_processed_dataset(
            _settings(min_reviews=1), raw_dir=raw_dir, processed_dir=out, chunk_size=4
        )
        assert report.products_selected == 5
        assert report.reviews_final == 6
        assert report.products_below_min_reviews == 0

    def test_missing_text_flagged_not_dropped(self, raw_dir: Path, tmp_path: Path) -> None:
        """Reviews with empty/absent text stay in the table with has_text=False."""
        out = tmp_path / "processed"
        build_processed_dataset(
            _settings(min_reviews=1), raw_dir=raw_dir, processed_dir=out, chunk_size=4
        )
        reviews = pd.read_parquet(out / "reviews.parquet")
        assert len(reviews) == 6  # nothing dropped for empty text
        flagged = reviews[~reviews["has_text"]]
        assert len(flagged) == 2  # absent reviewText (B02) and empty string (B04)
        assert set(flagged["text_length"]) == {0}
        assert flagged["review_text_raw"].isna().sum() == 1  # B02 never had text
        # Vote forms: "1,234" parsed, missing/empty -> 0.
        votes = dict(zip(reviews["asin"], reviews["helpful_votes"], strict=True))
        assert votes["B000000002"] == 1234
        assert votes["B000000004"] == 0
        # Weak labels: ratings 5/1/3/4/2/5 -> 2/0/1/2/0/2.
        assert sorted(int(v) for v in reviews["sentiment_label_weak"]) == [0, 0, 1, 2, 2, 2]

    def test_product_missing_flags(self, raw_dir: Path, tmp_path: Path) -> None:
        """Missing/unparseable price, rank and brand get boolean flags, no imputation."""
        out = tmp_path / "processed"
        build_processed_dataset(
            _settings(min_reviews=1), raw_dir=raw_dir, processed_dir=out, chunk_size=4
        )
        products = pd.read_parquet(out / "products.parquet").set_index("asin")
        # B000000004: empty price, no rank. B000000005: CSS-garbage price, no rank.
        for asin in ("B000000004", "B000000005"):
            assert bool(products.loc[asin, "price_missing"]) is True
            assert pd.isna(products.loc[asin, "price"])
            assert bool(products.loc[asin, "rank_missing"]) is True
        assert products.loc["B000000005", "price_status"] == "unparseable"
        # B000000002: empty brand, salesRank-derived numeric rank without category.
        assert bool(products.loc["B000000002", "brand_missing"]) is True
        assert int(products.loc["B000000002", "rank_value"]) == 2345
        assert pd.isna(products.loc["B000000002", "rank_category"])
        # Range price keeps both bounds; brand from details fallback is normalised.
        assert float(products.loc["B000000003", "price"]) == pytest.approx(10.99)
        assert float(products.loc["B000000003", "price_max"]) == pytest.approx(19.99)
        assert products.loc["B000000003", "price_status"] == "range"
        assert products.loc["B000000003", "brand_clean"] == "detailsbrand"
        assert products["price_missing"].sum() == 2
        assert products["brand_missing"].sum() == 1

    def test_related_edge_into_subset_flagged(self, raw_dir: Path, tmp_path: Path) -> None:
        """B000000004 -> B000000005 links a candidate: in_subset is True."""
        out = tmp_path / "processed"
        build_processed_dataset(
            _settings(min_reviews=1), raw_dir=raw_dir, processed_dir=out, chunk_size=4
        )
        related = pd.read_parquet(out / "related_products.parquet")
        edge = related[
            (related["asin"] == "B000000004") & (related["related_asin"] == "B000000005")
        ]
        assert len(edge) == 1
        assert bool(edge.iloc[0]["in_subset"]) is True
        assert edge.iloc[0]["relation_type"] == "also_viewed"
        assert related["in_subset"].sum() == 1
        assert len(related) == 5


class TestDuplicateHandling:
    """Duplicate products and reviews are dropped and reported."""

    def test_duplicates_dropped_and_counted(self, tmp_path: Path) -> None:
        """Exact duplicate review rows and duplicate product rows both drop."""
        raw = tmp_path / "raw"
        raw.mkdir()
        reviews = [
            {
                "overall": 5.0,
                "reviewText": "Good stuff",
                "asin": "DUP0000001",
                "unixReviewTime": 1500000000,
                "reviewerID": "U9",
            },
            {  # exact duplicate of the row above
                "overall": 5.0,
                "reviewText": "Good stuff",
                "asin": "DUP0000001",
                "unixReviewTime": 1500000000,
                "reviewerID": "U9",
            },
            {
                "overall": 1.0,
                "reviewText": "Bad stuff",
                "asin": "DUP0000001",
                "unixReviewTime": 1500001000,
                "reviewerID": "U9",
            },
        ]
        metadata = [
            {"asin": "DUP0000001", "price": "$5.00", "brand": "Dup"},
            {"asin": "DUP0000001", "price": "$6.00", "brand": "Dup"},
        ]
        _write_gzip_jsonl(raw / "Appliances.json.gz", reviews)
        _write_gzip_jsonl(raw / "meta_Appliances.json.gz", metadata)

        report = build_processed_dataset(
            _settings(min_reviews=2),
            raw_dir=raw,
            processed_dir=tmp_path / "processed",
            chunk_size=10,
        )
        assert report.duplicate_reviews_dropped == 1
        assert report.reviews_final == 2
        assert report.products_duplicate_rows_dropped == 1
        assert report.products_final == 1
        assert report.products_below_min_after_dedup == 0  # 2 kept >= threshold 2

        products = pd.read_parquet(tmp_path / "processed" / "products.parquet")
        assert float(products.iloc[0]["price"]) == pytest.approx(5.0)  # first row kept


class TestBuildErrors:
    """Honest failures instead of fabricated outputs."""

    def test_missing_raw_files_raise(self, tmp_path: Path) -> None:
        """No raw files -> DataError pointing at the downloader."""
        with pytest.raises(DataError, match="Raw file"):
            build_processed_dataset(
                _settings(min_reviews=1),
                raw_dir=tmp_path / "does_not_exist",
                processed_dir=tmp_path / "out",
            )

    def test_unknown_category_raises(self, tmp_path: Path) -> None:
        """Category without sources -> DataError listing registered ones."""
        with pytest.raises(DataError, match="dataset.sources"):
            build_processed_dataset(
                _settings(min_reviews=1, category="Not_A_Category"),
                raw_dir=tmp_path,
                processed_dir=tmp_path / "out",
            )

    def test_no_candidates_raises(self, raw_dir: Path, tmp_path: Path) -> None:
        """Threshold nobody meets -> DataError, no empty outputs written."""
        with pytest.raises(DataError, match="candidate_min_reviews"):
            build_processed_dataset(
                _settings(min_reviews=99),
                raw_dir=raw_dir,
                processed_dir=tmp_path / "out",
            )
        assert not (tmp_path / "out").exists()


class TestParquetSchemas:
    """Written files carry the explicit pipeline schemas."""

    @pytest.mark.parametrize(
        ("filename", "schema"),
        [
            ("products.parquet", PRODUCT_SCHEMA),
            ("reviews.parquet", REVIEW_SCHEMA),
            ("related_products.parquet", RELATED_SCHEMA),
        ],
    )
    def test_schema_matches_contract(
        self, raw_dir: Path, tmp_path: Path, filename: str, schema: Any
    ) -> None:
        """Round-tripped parquet schema equals the declared schema (sans metadata)."""
        out = tmp_path / "processed"
        build_processed_dataset(
            _settings(min_reviews=1), raw_dir=raw_dir, processed_dir=out, chunk_size=4
        )
        written = pq.read_schema(out / filename)
        assert written.remove_metadata() == schema.remove_metadata()

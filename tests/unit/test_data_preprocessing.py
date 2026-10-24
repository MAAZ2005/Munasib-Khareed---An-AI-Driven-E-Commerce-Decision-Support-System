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
            _settings(min_reviews=2),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=2,
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
            _settings(min_reviews=2),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=2,
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
            _settings(min_reviews=2),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=2,
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
            _settings(min_reviews=2),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=2,
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
            _settings(min_reviews=2),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=2,
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
            _settings(min_reviews=1),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=4,
        )
        assert report.products_selected == 5
        assert report.reviews_final == 6
        assert report.products_below_min_reviews == 0

    def test_missing_text_flagged_not_dropped(self, raw_dir: Path, tmp_path: Path) -> None:
        """Reviews with empty/absent text stay in the table with has_text=False."""
        out = tmp_path / "processed"
        build_processed_dataset(
            _settings(min_reviews=1),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=4,
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
            _settings(min_reviews=1),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=4,
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
            _settings(min_reviews=1),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=4,
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
            reports_dir=tmp_path / "reports",
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
            _settings(min_reviews=1),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=4,
        )
        written = pq.read_schema(out / filename)
        assert written.remove_metadata() == schema.remove_metadata()


class TestSplitColumns:
    """Product-level split written to both tables: no leakage, inherited by reviews."""

    def _build(self, raw_dir: Path, tmp_path: Path, out: Path) -> Any:
        return build_processed_dataset(
            _settings(min_reviews=1),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=4,
        )

    def test_split_column_written_to_both_tables(
        self, raw_dir: Path, tmp_path: Path
    ) -> None:
        """products.parquet and reviews.parquet both carry a populated split."""
        out = tmp_path / "processed"
        self._build(raw_dir, tmp_path, out)
        products = pd.read_parquet(out / "products.parquet")
        reviews = pd.read_parquet(out / "reviews.parquet")
        assert "split" in products.columns and "split" in reviews.columns
        assert set(products["split"].astype(str)) <= {"train", "val", "test"}
        assert set(reviews["split"].astype(str)) <= {"train", "val", "test"}
        assert products["split"].isna().sum() == 0
        assert reviews["split"].isna().sum() == 0
        assert "mean_rating" in products.columns

    def test_no_leakage_reviews_inherit_product_split(
        self, raw_dir: Path, tmp_path: Path
    ) -> None:
        """No product spans splits; every review matches its product's split."""
        out = tmp_path / "processed"
        self._build(raw_dir, tmp_path, out)
        products = pd.read_parquet(out / "products.parquet")
        reviews = pd.read_parquet(out / "reviews.parquet")
        assert products["asin"].duplicated().sum() == 0
        per_product = reviews.groupby("asin", observed=True)["split"].nunique()
        assert per_product.max() == 1
        joined = reviews.merge(
            products[["asin", "split"]], on="asin", suffixes=("_review", "_product")
        )
        assert len(joined) == len(reviews)  # every review found its product
        assert (joined["split_review"] == joined["split_product"]).all()

    def test_split_assignment_is_deterministic(
        self, raw_dir: Path, tmp_path: Path
    ) -> None:
        """Two builds with the same seed produce identical splits and counts."""
        first = tmp_path / "first"
        second = tmp_path / "second"
        report_a = self._build(raw_dir, tmp_path, first)
        report_b = self._build(raw_dir, tmp_path, second)
        products_a = pd.read_parquet(first / "products.parquet")
        products_b = pd.read_parquet(second / "products.parquet")
        assert (
            products_a.set_index("asin")["split"].sort_index()
            == products_b.set_index("asin")["split"].sort_index()
        ).all()
        reviews_a = pd.read_parquet(first / "reviews.parquet")
        reviews_b = pd.read_parquet(second / "reviews.parquet")
        assert (
            reviews_a.set_index("review_id")["split"].sort_index()
            == reviews_b.set_index("review_id")["split"].sort_index()
        ).all()
        assert report_a.reviews_final == report_b.reviews_final
        assert report_a.products_final == report_b.products_final
        assert report_a.split_sizes == report_b.split_sizes

    def test_meta_records_split_settings_and_sizes(
        self, raw_dir: Path, tmp_path: Path
    ) -> None:
        """meta.json carries the seeded split config and the per-split counts."""
        out = tmp_path / "processed"
        report = self._build(raw_dir, tmp_path, out)
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["settings"]["split"]["seed"] == 42
        assert meta["settings"]["split"]["train_fraction"] == 0.7
        assert meta["splits"]["grain"] == "product"
        assert meta["splits"]["sizes"] == report.split_sizes
        assert sum(v["products"] for v in meta["splits"]["sizes"].values()) == 5
        assert sum(v["reviews"] for v in meta["splits"]["sizes"].values()) == 6

    def test_quality_report_written(self, raw_dir: Path, tmp_path: Path) -> None:
        """The build writes reports/data_quality_report.{json,md} with facts."""
        out = tmp_path / "processed"
        report = self._build(raw_dir, tmp_path, out)
        reports_dir = tmp_path / "reports"
        payload = json.loads(
            (reports_dir / "data_quality_report.json").read_text(encoding="utf-8")
        )
        markdown = (reports_dir / "data_quality_report.md").read_text(encoding="utf-8")
        assert report.report_files == [
            "data_quality_report.json",
            "data_quality_report.md",
        ]
        assert payload["row_counts"]["products"] == 5
        assert payload["row_counts"]["reviews"] == 6
        assert payload["split_sizes"]["leakage_check"]["product_splits_unique"] is True
        assert payload["split_sizes"]["leakage_check"]["reviews_match_product_split"] is True
        assert payload["cleaning_steps"]["reviews"]["streamed_from_raw_file"] == 6
        assert payload["cleaning_steps"]["reviews"]["after_candidate_filter"] == 6
        assert payload["cleaning_steps"]["products"]["after_asin_dedup"] == 5
        assert "# Data Quality Report - Appliances" in markdown
        assert "## Split sizes" in markdown
        assert "6" in markdown  # computed review count appears in the render


class TestVerifiedNullVersusFalse:
    """``verified`` distinguishes an absent field (null) from an explicit False."""

    def test_null_and_false_are_distinct(self, raw_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "processed"
        build_processed_dataset(
            _settings(min_reviews=1),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=4,
        )
        reviews = pd.read_parquet(out / "reviews.parquet").set_index("asin")
        # B000000004 has no "verified" field in the fixture -> null, not False.
        assert pd.isna(reviews.loc["B000000004", "verified"])
        # B000000002 records an explicit false -> False, not null.
        assert bool(reviews.loc["B000000002", "verified"]) is False
        assert not pd.isna(reviews.loc["B000000002", "verified"])
        # B000000001 has both a true and a false row.
        b1 = reviews.loc["B000000001", "verified"]
        assert {bool(v) for v in b1} == {True, False}
        assert str(reviews["verified"].dtype) == "boolean"  # nullable dtype survives


class TestProductImages:
    """``image_url``/``image_missing`` derived from the loader's image columns."""

    def _products(self, raw_dir: Path, tmp_path: Path) -> pd.DataFrame:
        out = tmp_path / "processed"
        build_processed_dataset(
            _settings(min_reviews=1),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=4,
        )
        return pd.read_parquet(out / "products.parquet").set_index("asin")

    def test_highres_preferred_then_small(self, raw_dir: Path, tmp_path: Path) -> None:
        """The first high-res URL wins; the small URL is only a fallback."""
        products = self._products(raw_dir, tmp_path)
        assert products.loc["B000000001", "image_url"] == "https://example.com/hi1.jpg"
        assert bool(products.loc["B000000001", "image_missing"]) is False
        # B000000003 has only a single-string imageURL -> used as-is.
        assert products.loc["B000000003", "image_url"] == "https://example.com/c.jpg"
        assert bool(products.loc["B000000003", "image_missing"]) is False
    def test_missing_and_non_http_stay_missing(self, raw_dir: Path, tmp_path: Path) -> None:
        """Empty lists, non-http URLs and absent fields all mean "no image"."""
        products = self._products(raw_dir, tmp_path)
        for asin in ("B000000002", "B000000004", "B000000005"):
            assert pd.isna(products.loc[asin, "image_url"])
            assert bool(products.loc[asin, "image_missing"]) is True

    def test_flag_matches_null_url(self, raw_dir: Path, tmp_path: Path) -> None:
        """``image_missing`` is exactly the nullness of ``image_url``."""
        products = self._products(raw_dir, tmp_path)
        assert len(products) == 5  # B000000006 has metadata but no review
        assert (products["image_url"].isna() == products["image_missing"]).all()
        assert int(products["image_missing"].sum()) == 3  # 2 of 5 have an image


class TestDuplicateImageRecovery:
    """An image carried only by a dropped duplicate row is recovered (nothing else changes)."""

    def _raw_dir(self, tmp_path: Path, metadata: list[dict[str, Any]]) -> Path:
        target = tmp_path / "raw"
        target.mkdir()
        reviews = [
            {
                "asin": record["asin"],
                "rating": 5.0,
                "reviewText": f"text {index}",
                "unixReviewTime": 1_500_000_000 + index,
            }
            for index, record in enumerate(metadata)
        ]
        _write_gzip_jsonl(target / "Appliances.json.gz", reviews)
        _write_gzip_jsonl(target / "meta_Appliances.json.gz", metadata)
        return target

    def test_dropped_duplicate_fills_a_missing_image(self, tmp_path: Path) -> None:
        """Kept row wins; a duplicate row's image is used only when the kept row has none."""
        raw_dir = self._raw_dir(
            tmp_path,
            [
                # A1: first (kept) row has no image, the duplicate does -> recovered.
                {"asin": "B0000000A1", "title": "A1 first", "imageURLHighRes": [], "imageURL": []},
                {
                    "asin": "B0000000A1",
                    "title": "A1 duplicate",
                    "imageURLHighRes": ["https://example.com/a1-dup.jpg"],
                },
                # A2: no duplicate, no image -> stays missing.
                {"asin": "B0000000A2", "title": "A2"},
                # A3: the kept row has an image and the duplicate a different one -> kept wins.
                {
                    "asin": "B0000000A3",
                    "title": "A3 first",
                    "imageURLHighRes": ["https://example.com/a3-first.jpg"],
                },
                {
                    "asin": "B0000000A3",
                    "title": "A3 duplicate",
                    "imageURLHighRes": ["https://example.com/a3-dup.jpg"],
                },
            ],
        )
        out = tmp_path / "processed"
        report = build_processed_dataset(
            _settings(min_reviews=1),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=2,
        )
        products = pd.read_parquet(out / "products.parquet").set_index("asin")

        assert report.products_duplicate_rows_dropped == 2
        assert products.loc["B0000000A1", "image_url"] == "https://example.com/a1-dup.jpg"
        assert bool(products.loc["B0000000A1", "image_missing"]) is False
        # The keep rule itself is unchanged: the first row is still the kept one.
        assert products.loc["B0000000A1", "title"] == "A1 first"
        assert products.loc["B0000000A3", "image_url"] == "https://example.com/a3-first.jpg"
        assert pd.isna(products.loc["B0000000A2", "image_url"])
        assert bool(products.loc["B0000000A2", "image_missing"]) is True


class TestCategoryBreadcrumb:
    """``category_path``/``category_leaf`` from the three shapes of the raw field.

    The 2018 metadata publishes ``category`` as a list of browse nodes ordered
    broad -> specific, occasionally as a single string, and sometimes not at
    all. All three must be handled without ever inventing a breadcrumb.
    """

    def _products(self, tmp_path: Path, metadata: list[dict[str, Any]]) -> pd.DataFrame:
        """Build a dataset over one-metadata-record-per-product fixtures.

        Args:
            tmp_path: Temporary directory for the build.
            metadata: Raw metadata records, one review each so all are candidates.

        Returns:
            ``products.parquet`` content indexed by ASIN.
        """
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        reviews = [
            {
                "asin": record["asin"],
                "rating": 5.0,
                "reviewText": f"text {index}",
                "unixReviewTime": 1_500_000_000 + index,
            }
            for index, record in enumerate(metadata)
        ]
        _write_gzip_jsonl(raw_dir / "Appliances.json.gz", reviews)
        _write_gzip_jsonl(raw_dir / "meta_Appliances.json.gz", metadata)
        out = tmp_path / "processed"
        build_processed_dataset(
            _settings(min_reviews=1),
            raw_dir=raw_dir,
            processed_dir=out,
            reports_dir=tmp_path / "reports",
            chunk_size=2,
        )
        return pd.read_parquet(out / "products.parquet").set_index("asin")

    def test_list_string_and_missing_shapes(self, tmp_path: Path) -> None:
        """A list is joined, a scalar string is split, an absent field stays NULL."""
        products = self._products(
            tmp_path,
            [
                # List form, with an HTML entity that must be decoded so the two
                # spellings of "Parts & Accessories" cannot split one browse node.
                {
                    "asin": "B0000000C1",
                    "category": ["Appliances", "Parts &amp; Accessories"],
                },
                # Scalar string form, pre-joined with the separator.
                {
                    "asin": "B0000000C2",
                    "category": "Appliances > Water Filters",
                },
                # Scalar string naming a single node.
                {"asin": "B0000000C3", "category": "Appliances"},
                # Absent entirely (and an empty list, which means the same).
                {"asin": "B0000000C4"},
                {"asin": "B0000000C5", "category": []},
            ],
        )
        assert products.loc["B0000000C1", "category_path"] == "Appliances > Parts & Accessories"
        assert products.loc["B0000000C1", "category_leaf"] == "Parts & Accessories"
        assert products.loc["B0000000C2", "category_path"] == "Appliances > Water Filters"
        assert products.loc["B0000000C2", "category_leaf"] == "Water Filters"
        assert products.loc["B0000000C3", "category_path"] == "Appliances"
        assert products.loc["B0000000C3", "category_leaf"] == "Appliances"
        for asin in ("B0000000C4", "B0000000C5"):
            assert pd.isna(products.loc[asin, "category_path"])
            assert pd.isna(products.loc[asin, "category_leaf"])

    def test_leaf_is_the_last_level_not_a_reparse(self, tmp_path: Path) -> None:
        """``category_leaf`` is the final breadcrumb level, never a re-derived guess."""
        products = self._products(
            tmp_path,
            [
                {"asin": "B0000000C1", "category": ["Appliances", "Water Filters"]},
                {
                    "asin": "B0000000C2",
                    "category": ["Appliances", "Water Filters", "Replacement Cartridges"],
                },
            ],
        )
        # C2's breadcrumb sits one level below C1's, so C1's node is a genuine
        # ancestor prefix of C2's - which is what the peer-group walk relies on.
        assert products.loc["B0000000C1", "category_path"] == "Appliances > Water Filters"
        assert (
            products.loc["B0000000C2", "category_path"]
            == "Appliances > Water Filters > Replacement Cartridges"
        )
        assert products.loc["B0000000C1", "category_leaf"] == "Water Filters"
        assert products.loc["B0000000C2", "category_leaf"] == "Replacement Cartridges"

    def test_blank_levels_are_dropped_not_invented(self, tmp_path: Path) -> None:
        """Empty or whitespace-only levels disappear; an all-blank field is NULL."""
        products = self._products(
            tmp_path,
            [
                {
                    "asin": "B0000000C1",
                    "category": ["  Appliances  ", "", "   ", "Water Filters"],
                },
                # Nothing usable survives -> NULL rather than " > " or similar.
                {"asin": "B0000000C2", "category": ["", "   "]},
            ],
        )
        assert products.loc["B0000000C1", "category_path"] == "Appliances > Water Filters"
        assert products.loc["B0000000C1", "category_leaf"] == "Water Filters"
        assert pd.isna(products.loc["B0000000C2", "category_path"])
        assert pd.isna(products.loc["B0000000C2", "category_leaf"])


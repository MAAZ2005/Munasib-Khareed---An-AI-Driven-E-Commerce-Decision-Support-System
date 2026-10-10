"""Tests for the auto-generated data-quality report (``reports/data_quality_report.*``).

Everything in the report must be code-computed: these tests rebuild the report
over the Prompt 3 fixtures and cross-check its numbers against the parquet
tables it describes. Hermetic: fixtures only - no network, no GPU.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd

from munasib.core.config import (
    AuditRubricConfig,
    DatasetConfig,
    DatasetSourceConfig,
    Settings,
)
from munasib.data.preprocessing import build_processed_dataset
from munasib.data.quality import render_markdown, write_quality_report

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


def _settings(min_reviews: int) -> Settings:
    """Fixture-compatible settings with the given candidate threshold.

    Args:
        min_reviews: ``dataset.audit.candidate_min_reviews``.

    Returns:
        Settings instance.
    """
    return Settings(
        dataset=DatasetConfig(
            category="Appliances",
            sources={
                "Appliances": DatasetSourceConfig(
                    reviews_url="https://example.invalid/Appliances.json.gz",
                    metadata_url="https://example.invalid/meta_Appliances.json.gz",
                )
            },
            audit=AuditRubricConfig(candidate_min_reviews=min_reviews),
        )
    )


def _build(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    """Build the fixture dataset and load the written quality report.

    Args:
        tmp_path: Test scratch directory.

    Returns:
        ``(report dict, processed dir, reports dir)``.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_gzip_jsonl(
        raw / "Appliances.json.gz",
        [
            json.loads(line)
            for line in (FIXTURES / "reviews_variations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ],
    )
    _write_gzip_jsonl(
        raw / "meta_Appliances.json.gz",
        [
            json.loads(line)
            for line in (FIXTURES / "metadata_variations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ],
    )
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    build_processed_dataset(
        _settings(min_reviews=1),
        raw_dir=raw,
        processed_dir=processed,
        reports_dir=reports,
        chunk_size=4,
    )
    payload = json.loads((reports / "data_quality_report.json").read_text(encoding="utf-8"))
    return payload, processed, reports


class TestQualityReportJson:
    """Structure and internal consistency of ``data_quality_report.json``."""

    def test_required_sections_present(self, tmp_path: Path) -> None:
        payload, _, _ = _build(tmp_path)
        for key in (
            "generated_at",
            "category",
            "settings",
            "cleaning_steps",
            "field_presence",
            "missingness",
            "split_sizes",
            "rating_distribution_per_split",
            "label_distribution_per_split",
            "reviews_per_product",
            "review_date_range",
            "row_counts",
        ):
            assert key in payload, f"missing section: {key}"
        assert payload["settings"]["candidate_min_reviews"] == 1
        assert payload["settings"]["split"]["seed"] == 42

    def test_step_counts_are_monotonic_and_match_parquet(
        self, tmp_path: Path
    ) -> None:
        """Cleaning-step row counts chain correctly and end at the written rows."""
        payload, processed, _ = _build(tmp_path)
        reviews_steps = payload["cleaning_steps"]["reviews"]
        assert (
            reviews_steps["streamed_from_raw_file"]
            >= reviews_steps["after_candidate_filter"]
            >= reviews_steps["after_duplicate_removal"]
        )
        assert (
            reviews_steps["written_to_reviews_parquet"]
            == reviews_steps["after_duplicate_removal"]
        )
        written_reviews = len(pd.read_parquet(processed / "reviews.parquet"))
        written_products = len(pd.read_parquet(processed / "products.parquet"))
        assert reviews_steps["written_to_reviews_parquet"] == written_reviews
        assert (
            payload["cleaning_steps"]["products"]["written_to_products_parquet"]
            == written_products
        )

    def test_missingness_matches_parquet(self, tmp_path: Path) -> None:
        """Per-column null counts equal what a fresh parquet read shows."""
        payload, processed, _ = _build(tmp_path)
        for table in ("products.parquet", "reviews.parquet", "related_products.parquet"):
            frame = pd.read_parquet(processed / table)
            reported = payload["missingness"]["tables"][table]
            assert reported["rows"] == len(frame)
            assert set(reported["columns"]) == set(frame.columns)
            for column, values in reported["columns"].items():
                assert values["null"] == int(frame[column].isna().sum()), column

    def test_split_sizes_match_parquet_and_are_complete(
        self, tmp_path: Path
    ) -> None:
        """Reported split sizes equal value_counts on both written tables."""
        payload, processed, _ = _build(tmp_path)
        products = pd.read_parquet(processed / "products.parquet")
        reviews = pd.read_parquet(processed / "reviews.parquet")
        per_split = payload["split_sizes"]["per_split"]
        for split in ("train", "val", "test"):
            assert per_split[split]["products"] == int(
                (products["split"].astype(str) == split).sum()
            )
            assert per_split[split]["reviews"] == int(
                (reviews["split"].astype(str) == split).sum()
            )
        assert sum(v["products"] for v in per_split.values()) == len(products)
        assert sum(v["reviews"] for v in per_split.values()) == len(reviews)
        assert payload["split_sizes"]["leakage_check"]["product_splits_unique"] is True
        assert payload["split_sizes"]["leakage_check"]["reviews_match_product_split"] is True
        # Strata counts also sum to the product total.
        assert sum(row["products"] for row in payload["split_sizes"]["strata"]) == len(
            products
        )

    def test_distributions_sum_to_split_review_counts(self, tmp_path: Path) -> None:
        """Rating and weak-label buckets per split sum to that split's reviews."""
        payload, _, _ = _build(tmp_path)
        reviews_per_split = {
            split: values["reviews"]
            for split, values in payload["split_sizes"]["per_split"].items()
        }
        for dist_key in ("rating_distribution_per_split", "label_distribution_per_split"):
            for split, buckets in payload[dist_key].items():
                assert sum(bucket["count"] for bucket in buckets.values()) == (
                    reviews_per_split[split]
                )
                for bucket in buckets.values():
                    assert 0.0 <= bucket["share"] <= 1.0

    def test_rating_three_maps_to_neutral_label(self, tmp_path: Path) -> None:
        """Fixture rating-3 review lands in label 1 (locked mapping edge)."""
        payload, _, _ = _build(tmp_path)
        labels = payload["label_distribution_per_split"]
        total_1 = sum(buckets.get("1", {}).get("count", 0) for buckets in labels.values())
        assert total_1 == 1  # exactly one rating-3 review in the fixtures
        ratings = payload["rating_distribution_per_split"]
        total_3 = sum(buckets.get("3", {}).get("count", 0) for buckets in ratings.values())
        assert total_3 == 1

    def test_date_range_and_reviews_per_product(self, tmp_path: Path) -> None:
        """Date range spans the fixture reviews; product stats are positive."""
        payload, _, _ = _build(tmp_path)
        assert payload["review_date_range"]["min"] is not None
        assert payload["review_date_range"]["max"] is not None
        assert payload["review_date_range"]["min"] <= payload["review_date_range"]["max"]
        assert "never today" in payload["review_date_range"]["note"]
        per_product = payload["reviews_per_product"]
        assert per_product["min"] >= 1
        assert per_product["max"] >= per_product["p50"]
        assert per_product["mean"] >= per_product["min"]

    def test_json_is_strictly_parseable_without_nan_tokens(
        self, tmp_path: Path
    ) -> None:
        """No NaN/Infinity tokens: every value is a real JSON number."""
        _, _, reports = _build(tmp_path)
        text = (reports / "data_quality_report.json").read_text(encoding="utf-8")
        for token in ("NaN", "Infinity", "-Infinity"):
            assert token not in text
        json.loads(text)  # strict parse succeeds

    def test_field_presence_before_after(self, tmp_path: Path) -> None:
        """Raw-vs-clean presence counts reflect the fixture cleaning rules."""
        payload, _, _ = _build(tmp_path)
        products = payload["field_presence"]["products"]
        # Fixture: 5 products, one has no brand at all -> brand before < rows.
        assert products["brand"]["rows"] == 5
        assert products["brand"]["before_present"] <= 5
        assert products["brand"]["after_missing"] == 5 - products["brand"]["after_present"]
        reviews = payload["field_presence"]["reviews"]
        # Fixture: row 3 has no reviewText and row 5 has "" -> 4 usable raw texts;
        # cleaning preserves all four (no URL/whitespace-only texts in fixtures).
        assert reviews["review_text"]["before_present"] == 4
        assert reviews["review_text"]["after_present"] == 4
        assert reviews["summary"]["after_present"] == 0  # fixture has no summaries


class TestQualityReportMarkdown:
    """The Markdown render is a pure function of the report dict."""

    def test_markdown_written_and_mentions_computed_values(
        self, tmp_path: Path
    ) -> None:
        payload, _, reports = _build(tmp_path)
        markdown = (reports / "data_quality_report.md").read_text(encoding="utf-8")
        assert markdown.endswith("\n")
        for heading in (
            "# Data Quality Report - Appliances",
            "## Row counts before/after each cleaning step",
            "## Field presence before/after cleaning",
            "## Missingness per column",
            "## Split sizes (product-level split, no leakage)",
            "## Stratification strata",
            "## Rating distribution per split",
            "## Weak-label distribution per split",
            "## Reviews per product",
            "## Review date range",
        ):
            assert heading in markdown, heading
        # Rendered numbers come from the dict (1-row table of the products written).
        assert f"{payload['row_counts']['products']:,}" in markdown
        assert f"{payload['row_counts']['reviews']:,}" in markdown

    def test_render_is_deterministic(self, tmp_path: Path) -> None:
        payload, _, _ = _build(tmp_path)
        assert render_markdown(payload) == render_markdown(payload)

    def test_write_quality_report_roundtrip(self, tmp_path: Path) -> None:
        payload, _, _ = _build(tmp_path)
        target = tmp_path / "again"
        paths = write_quality_report(payload, target)
        assert [p.name for p in paths] == [
            "data_quality_report.json",
            "data_quality_report.md",
        ]
        assert json.loads(paths[0].read_text(encoding="utf-8")) == payload

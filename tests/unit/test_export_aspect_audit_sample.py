"""Tests for the audit sample export/evaluation CLI."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import insert
from sqlalchemy.engine import Engine

from munasib.db.models import Product, Review, ReviewFeature
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_aspect_audit_sample.py"


def _load_module():
    """Import the audit script once.

    Returns:
        Loaded module.
    """
    spec = importlib.util.spec_from_file_location("munasib_aspect_audit_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seed(engine: Engine) -> None:
    """Seed three products/reviews with two aspects.

    Args:
        engine: Target cache.
    """
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        session.execute(
            insert(Product),
            [
                {
                    "asin": f"B00000000{i}",
                    "split": "train",
                    "title": f"Product {i}",
                    "title_clean": f"product {i}",
                    "brand_missing": True,
                    "price_missing": True,
                    "rank_missing": True,
                    "n_reviews": 1,
                    "image_missing": True,
                }
                for i in range(1, 4)
            ],
        )
        session.execute(
            insert(Review),
            [
                {
                    "review_id": f"r{i}",
                    "asin": f"B00000000{i}",
                    "split": "train",
                    "reviewer_id": "u1",
                    "review_text_raw": f"text about filter {i}",
                    "review_text_clean": f"text about filter {i}",
                    "summary_raw": None,
                    "summary_clean": None,
                    "has_text": True,
                    "text_length": 20,
                    "helpful_votes": i,
                    "verified": None,
                    "review_time": datetime(2018, 1, 1),
                    "year": 2018,
                    "month": 1,
                    "sentiment_label_weak": 0,
                }
                for i in range(1, 4)
            ],
        )
        session.execute(
            insert(ReviewFeature),
            [
                {
                    "review_id": f"r{i}",
                    "predicted_label": 0,
                    "prob_negative": 0.8,
                    "prob_neutral": 0.1,
                    "prob_positive": 0.1,
                    "aspects_json": ["filter", "motor"],
                }
                for i in range(1, 4)
            ],
        )


@pytest.fixture()
def workspace(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """Create a seeded cache and config.

    Yields:
        ``(config_path, output_path)``.
    """
    db_path = tmp_path / "cache.sqlite"
    engine = create_engine_from_settings(db_url=f"sqlite:///{db_path}")
    init_db(engine)
    _seed(engine)
    engine.dispose()
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "meta.json").write_text(json.dumps({"built_at": "2026-01-01T00:00:00+00:00"}))
    config = tmp_path / "config.yaml"
    output = tmp_path / "audit.csv"
    config.write_text(
        f"""
project:
  random_seed: 42
paths:
  db_path: "{db_path.as_posix()}"
  processed_data_dir: "{processed.as_posix()}"
review_intel:
  aspects:
    audit_sample_size: 4
    audit_sample_path: "{output.as_posix()}"
""",
        encoding="utf-8",
    )
    yield config, output


def test_export_writes_blank_long_format_rows(workspace: tuple[Path, Path]) -> None:
    """The export is long format and leaves annotation columns blank."""
    module = _load_module()
    config, output = workspace
    assert module.main(["--config", str(config)]) == 0
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(row["is_correct"] == "" and row["notes"] == "" for row in rows)
    assert {row["aspect"] for row in rows} == {"filter", "motor"}
    assert all(row["review_id"] and row["asin"] and row["review_text"] for row in rows)


def test_evaluation_refuses_blank_annotations(workspace: tuple[Path, Path]) -> None:
    """No precision is printed while any sampled mention is unannotated."""
    module = _load_module()
    config, output = workspace
    assert module.main(["--config", str(config)]) == 0
    assert module.main(["--config", str(config), "--evaluate"]) == 1


def test_evaluation_reports_precision_after_annotation(
    workspace: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Filled yes/no annotations produce a precision computed by the script."""
    module = _load_module()
    config, output = workspace
    assert module.main(["--config", str(config)]) == 0
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["is_correct"] = "yes"
    rows[1]["is_correct"] = "no"
    rows[2]["is_correct"] = "yes"
    rows[3]["is_correct"] = "yes"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert module.main(["--config", str(config), "--evaluate"]) == 0
    output_text = capsys.readouterr().out
    assert "aspect precision            : 0.7500" in output_text


def test_invalid_annotation_is_rejected(workspace: tuple[Path, Path]) -> None:
    """A typo in is_correct is an error, never silently counted as correct."""
    module = _load_module()
    config, output = workspace
    assert module.main(["--config", str(config)]) == 0
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["is_correct"] = "maybe"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert module.main(["--config", str(config), "--evaluate"]) == 1

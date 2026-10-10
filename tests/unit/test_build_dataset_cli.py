"""Tests for the ``scripts/build_dataset.py`` CLI wrapper.

Covers argument parsing (``--category``, ``--force``, ``--chunk-size``), the
failure exit codes, and the idempotent re-run behaviour (skip when outputs are
current, rebuild with ``--force``) over the Prompt 3 fixtures with a temporary
config - never the real raw files. Hermetic: no network, no GPU.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_dataset.py"


def _load_module():
    """Import ``scripts/build_dataset.py`` as a module.

    Returns:
        The loaded script module.
    """
    spec = importlib.util.spec_from_file_location("munasib_build_dataset_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults() -> None:
    """Defaults point at the master config and the loader's chunk size."""
    module = _load_module()
    args = module.parse_args([])
    assert args.config == "configs/config.yaml"
    assert args.chunk_size == 50_000
    assert args.category is None  # default comes from dataset.category in config
    assert args.force is False


def test_parse_args_overrides() -> None:
    """Flags override config path, chunk size, category and force."""
    module = _load_module()
    args = module.parse_args(
        ["--config", "other.yaml", "--chunk-size", "10", "--category", "Luxury_Beauty", "--force"]
    )
    assert args.config == "other.yaml"
    assert args.chunk_size == 10
    assert args.category == "Luxury_Beauty"
    assert args.force is True


def test_main_rejects_invalid_chunk_size() -> None:
    """A non-positive chunk size exits 1 before any file is opened."""
    module = _load_module()
    assert module.main(["--chunk-size", "0"]) == 1
    assert module.main(["--chunk-size", "-5"]) == 1


def _write_config(path: Path, raw: Path, processed: Path, reports: Path) -> Path:
    """Write a minimal project config pointing at temporary directories.

    Args:
        path: Destination YAML path.
        raw: Directory holding the fixture raw files.
        processed: Output directory for the parquet tables.
        reports: Output directory for the data-quality report.

    Returns:
        The config path.
    """
    text = f"""
paths:
  raw_data_dir: "{raw.as_posix()}"
  processed_data_dir: "{processed.as_posix()}"
  reports_dir: "{reports.as_posix()}"
dataset:
  category: "Appliances"
  sources:
    Appliances:
      reviews_url: "https://example.invalid/Appliances.json.gz"
      metadata_url: "https://example.invalid/meta_Appliances.json.gz"
  audit:
    candidate_min_reviews: 1
"""
    path.write_text(text, encoding="utf-8")
    return path


def _write_fixture_raws(raw: Path) -> None:
    """Copy the Prompt 3 JSONL fixtures into gzipped raw files.

    Args:
        raw: Directory to create and fill.
    """
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    raw.mkdir(parents=True, exist_ok=True)
    for fixture, target in (
        ("reviews_variations.jsonl", "Appliances.json.gz"),
        ("metadata_variations.jsonl", "meta_Appliances.json.gz"),
    ):
        records = [
            json.loads(line)
            for line in (fixtures / fixture).read_text(encoding="utf-8").splitlines()
        ]
        with gzip.open(raw / target, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")


def test_main_missing_raw_files_exits_one(tmp_path: Path) -> None:
    """Missing raw files exit 1 with an actionable downloader message."""
    module = _load_module()
    raw = tmp_path / "raw"  # never created
    config = _write_config(tmp_path / "config.yaml", raw, tmp_path / "p", tmp_path / "r")
    assert module.main(["--config", str(config)]) == 1


def test_main_build_then_skip_then_force(tmp_path: Path) -> None:
    """Same seed/config re-runs skip (idempotent); --force rebuilds."""
    module = _load_module()
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    _write_fixture_raws(raw)
    config = _write_config(tmp_path / "config.yaml", raw, processed, reports)

    # First run: full build.
    assert module.main(["--config", str(config)]) == 0
    meta_path = processed / "meta.json"
    assert meta_path.exists()
    assert (reports / "data_quality_report.json").exists()
    assert (reports / "data_quality_report.md").exists()
    first_mtime = meta_path.stat().st_mtime_ns
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["settings"]["split"]["seed"] == 42
    assert meta["report_files"] == [
        "data_quality_report.json",
        "data_quality_report.md",
    ]

    # Second run: outputs current -> skipped, nothing rewritten.
    assert module.main(["--config", str(config)]) == 0
    assert meta_path.stat().st_mtime_ns == first_mtime

    # --force: rebuilt (meta rewritten), row counts identical.
    assert module.main(["--config", str(config), "--force"]) == 0
    assert meta_path.stat().st_mtime_ns != first_mtime
    rebuilt = json.loads(meta_path.read_text(encoding="utf-8"))
    assert rebuilt["row_counts"] == meta["row_counts"]
    assert rebuilt["splits"] == meta["splits"]


def test_main_unknown_category_exits_one(tmp_path: Path) -> None:
    """--category for an unregistered category exits 1 via the build error."""
    module = _load_module()
    raw = tmp_path / "raw"
    _write_fixture_raws(raw)
    config = _write_config(tmp_path / "config.yaml", raw, tmp_path / "p", tmp_path / "r")
    assert module.main(["--config", str(config), "--category", "Not_A_Category"]) == 1

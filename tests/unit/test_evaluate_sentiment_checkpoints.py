"""Tests for the model comparison script (``scripts/evaluate_sentiment_checkpoints.py``).

The comparison table must be a faithful reading of the JSON reports, so these
tests feed hand-written fixture reports with distinctive numbers and check that
those exact numbers (and no invented ones) reach the Markdown and CSV. The
test-split lock is checked separately; no model is ever run here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_sentiment_checkpoints.py"

NEUTRAL_F1 = 0.4012
MAJORITY_ACCURACY = 0.8339
MAJORITY_MACRO_F1 = 0.3031


@pytest.fixture(scope="module")
def script():
    """Import the comparison script as a module.

    Returns:
        The loaded script module.
    """
    spec = importlib.util.spec_from_file_location("munasib_evaluate_checkpoints", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metrics(macro_f1: float) -> dict[str, object]:
    """Build the metrics block shape both report families use."""
    return {
        "accuracy": 0.9,
        "macro_f1": macro_f1,
        "per_class": [
            {"name": "negative", "f1": 0.75},
            {"name": "neutral", "f1": NEUTRAL_F1},
            {"name": "positive", "f1": 0.93},
        ],
        "binary": {"f1": 0.78, "roc_auc": 0.97},
        "majority_baseline": {
            "name": "positive",
            "accuracy": MAJORITY_ACCURACY,
            "macro_f1": MAJORITY_MACRO_F1,
        },
    }


def _neural_payload(split: str, macro_f1: float) -> dict[str, object]:
    """Build a neural report payload with only the fields the table reads."""
    return {
        "split": split,
        "evaluated_rows": 100,
        "metrics": _metrics(macro_f1),
        "calibration": {
            "hard_predictions": {"mae": 0.02},
            "mean_negative_probability": {"mae": 0.016},
        },
    }


def _baseline_payload(split: str) -> dict[str, object]:
    """Build a baseline report payload with both class-weight variants."""
    variants = {
        variant: {
            "metrics": _metrics(macro),
            "calibration": {"mae": 0.03},
        }
        for variant, macro in (("balanced", 0.6622), ("none", 0.6166))
    }
    return {"evaluations": {split: {"split": split, "rows": 100, "variants": variants}}}


def _write_reports(directory: Path, *, split: str = "val", baseline: bool = True) -> Path:
    """Write the three validation reports into a temporary report directory."""
    directory.mkdir(parents=True, exist_ok=True)
    if baseline:
        (directory / "sentiment_baseline_val.json").write_text(
            json.dumps(_baseline_payload(split)), encoding="utf-8"
        )
    (directory / "sentiment_transformer_val.json").write_text(
        json.dumps(_neural_payload(split, 0.7314)), encoding="utf-8"
    )
    (directory / "sentiment_rnn_val.json").write_text(
        json.dumps(_neural_payload(split, 0.6709)), encoding="utf-8"
    )
    return directory


def test_validation_rows_come_from_the_reports_only(script, tmp_path: Path) -> None:
    """Every cell is the value stored in the report, including the baseline variants."""
    directory = _write_reports(tmp_path / "reports")
    rows, notes = script.collect_validation_rows(directory)

    assert notes == []
    by_label = {row.label: row for row in rows}
    assert set(by_label) == {
        "TF-IDF + logistic regression (balanced)",
        "TF-IDF + logistic regression (none)",
        "DistilBERT",
        "BiGRU",
        "Majority class (positive)",
    }
    assert by_label["TF-IDF + logistic regression (balanced)"].macro_f1 == pytest.approx(0.6622)
    assert by_label["TF-IDF + logistic regression (none)"].macro_f1 == pytest.approx(0.6166)
    assert by_label["DistilBERT"].macro_f1 == pytest.approx(0.7314)
    assert by_label["BiGRU"].macro_f1 == pytest.approx(0.6709)
    assert by_label["DistilBERT"].neutral_f1 == pytest.approx(NEUTRAL_F1)
    assert by_label["DistilBERT"].calibration_mae_hard == pytest.approx(0.02)
    assert by_label["DistilBERT"].calibration_mae_soft == pytest.approx(0.016)
    assert by_label["TF-IDF + logistic regression (balanced)"].calibration_mae_soft is None
    assert by_label["DistilBERT"].source == "sentiment_transformer_val.json"


def test_majority_reference_row_invents_nothing(script, tmp_path: Path) -> None:
    """The baseline row carries only the two numbers its report computes."""
    directory = _write_reports(tmp_path / "reports")
    rows, _ = script.collect_validation_rows(directory)
    majority = next(row for row in rows if row.model.startswith("Majority class"))
    assert majority.accuracy == pytest.approx(MAJORITY_ACCURACY)
    assert majority.macro_f1 == pytest.approx(MAJORITY_MACRO_F1)
    assert majority.neutral_f1 is None
    assert majority.binary_f1 is None
    assert majority.calibration_mae_hard is None


def test_missing_reports_are_noted_not_guessed(script, tmp_path: Path) -> None:
    """A model without a report is left out and the table says why."""
    directory = _write_reports(tmp_path / "reports", baseline=False)
    (directory / "sentiment_rnn_val.json").unlink()
    rows, notes = script.collect_validation_rows(directory)
    assert [row.model for row in rows] == ["DistilBERT", "Majority class (positive)"]
    assert any("sentiment_rnn_val.json" in note for note in notes)
    assert any("sentiment_baseline_val.json" in note for note in notes)


def test_comparison_markdown_and_csv_are_written(script, tmp_path: Path) -> None:
    """The rendered table shows the report numbers and `n/a` for absent ones."""
    directory = _write_reports(tmp_path / "reports")
    rows, notes = script.collect_validation_rows(directory)
    markdown, csv_path = script.write_comparison(
        directory, rows, notes, stem="sentiment_model_comparison", split="val"
    )

    text = markdown.read_text(encoding="utf-8")
    assert "| DistilBERT | 100 |" in text
    assert "0.7314" in text
    assert "0.4012" in text
    assert "n/a" in text
    assert "weak supervision from star ratings" in text
    assert "computes no metric of its own" in text

    frame = pd.read_csv(csv_path)
    assert list(frame.columns) == list(script.CSV_COLUMNS)
    distilbert = frame.loc[frame["model"] == "DistilBERT"].iloc[0]
    assert distilbert["macro_f1"] == pytest.approx(0.7314)
    assert not pd.isna(distilbert["calibration_mae_soft"])
    baseline = frame.loc[frame["variant"] == "balanced"].iloc[0]
    assert pd.isna(baseline["calibration_mae_soft"])


def test_missing_optional_metric_renders_as_na(script) -> None:
    """A metric the report never computed is printed as n/a, not as zero."""
    assert script._fmt(None) == "n/a"
    assert script._fmt(0.5) == "0.5000"


def test_final_lock_refuses_a_second_run(script, tmp_path: Path) -> None:
    """The lock records the date and hashes, then blocks a silent re-use."""
    directory = tmp_path / "reports"
    lock = script.final_lock_path(directory)
    script.check_final_lock(lock, reuse=False)  # no lock yet: allowed

    entries = [
        {
            "model": "transformer",
            "checkpoint": "models/sentiment_distilbert/best",
            "weights_sha256": "abc123",
            "model_meta_sha256": "def456",
            "weights_file": "model.safetensors",
            "weights_bytes": 10,
        }
    ]
    script.write_final_lock(lock, entries, limit=None)

    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["split"] == "test"
    assert payload["limit"] is None
    assert payload["created_utc"]
    assert payload["models"] == entries
    assert "test split" in payload["note"]

    with pytest.raises(script.TestSplitLockedError, match="already scored"):
        script.check_final_lock(lock, reuse=False)
    script.check_final_lock(lock, reuse=True)


def test_final_lock_reports_the_limit(script, tmp_path: Path) -> None:
    """A limited smoke run must be visible in the lock, not look like the hold-out."""
    lock = script.write_final_lock(tmp_path / "lock.json", [], limit=500)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["limit"] == 500


def test_validation_mode_rejects_final_only_flags(script) -> None:
    """`--limit`/`--device` without `--final` would be a silent no-op."""
    assert script.main(["--limit", "5"]) == 1
    assert script.main(["--device", "cpu"]) == 1


def test_test_rows_read_the_final_reports(script, tmp_path: Path) -> None:
    """The test table is built from the `_final` stems, not the validation ones."""
    directory = tmp_path / "reports"
    directory.mkdir()
    (directory / "sentiment_transformer_final.json").write_text(
        json.dumps(_neural_payload("test", 0.75)), encoding="utf-8"
    )
    rows, notes = script.collect_test_rows(directory)
    assert [row.label for row in rows] == ["DistilBERT", "Majority class (positive)"]
    assert all(row.split == "test" for row in rows)
    assert any("sentiment_baseline_final.json" in note for note in notes)

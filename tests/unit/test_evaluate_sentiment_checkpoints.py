"""Tests for the model comparison script (``scripts/evaluate_sentiment_checkpoints.py``).

The comparison table must be a faithful reading of the JSON reports, so these
tests feed hand-written fixture reports with distinctive numbers and check that
those exact numbers (and no invented ones) reach the Markdown and CSV. The
test-split lock is checked separately; no model is ever run here.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
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


TRAINING_DEVICE = "cuda"
EVALUATION_DEVICE = "cpu"
ELAPSED_SECONDS = 671.547288963


def _full_metrics(macro_f1: float = 0.7314) -> dict[str, object]:
    """Build a complete neural metrics block (the fields the renderer reads)."""
    return {
        "n": 200,
        "labels": [
            {"label": 0, "name": "negative"},
            {"label": 1, "name": "neutral"},
            {"label": 2, "name": "positive"},
        ],
        "accuracy": 0.9,
        "macro_f1": macro_f1,
        "per_class": [
            {
                "label": 0,
                "name": "negative",
                "precision": 0.8344,
                "recall": 0.8202,
                "f1": 0.8272,
                "support": 75,
            },
            {
                "label": 1,
                "name": "neutral",
                "precision": 0.3690,
                "recall": 0.4213,
                "f1": NEUTRAL_F1,
                "support": 25,
            },
            {
                "label": 2,
                "name": "positive",
                "precision": 0.9670,
                "recall": 0.9611,
                "f1": 0.9640,
                "support": 100,
            },
        ],
        "confusion": [[60, 10, 5], [5, 12, 8], [10, 10, 80]],
        "binary": {
            "n": 175,
            "prevalence": 0.1300,
            "accuracy": 0.9681,
            "precision": 0.9250,
            "recall": 0.8202,
            "f1": 0.8694,
            "roc_auc": 0.9893,
        },
        "majority_baseline": {
            "label": 2,
            "name": "positive",
            "share": 0.8250,
            "accuracy": MAJORITY_ACCURACY,
            "macro_f1": MAJORITY_MACRO_F1,
        },
    }


def _full_training(training_device: str | None) -> dict[str, object]:
    """Build a training block; omit the device to mimic a legacy report."""
    block: dict[str, object] = {
        "seed": 42,
        "batch_size": 32,
        "learning_rate": 3e-05,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "max_grad_norm": 1.0,
        "early_stopping_patience": 1,
        "epochs_requested": 3,
        "epochs_run": 2,
        "best_epoch": 2,
        "selection_metric": "macro_f1",
        "val_subset_requested": None,
        "selection_rows": 62768,
        "full_report_split": True,
        "train_rows": 72052,
        "elapsed_seconds": ELAPSED_SECONDS,
        "history": [
            {
                "epoch": 1,
                "train_loss": 0.4500,
                "val_macro_f1": 0.7117,
                "val_accuracy": 0.9158,
                "seconds": 142.5,
                "rows_per_second": 505.7,
                "improved": True,
            },
            {
                "epoch": 2,
                "train_loss": 0.3205,
                "val_macro_f1": 0.7314,
                "val_accuracy": 0.9211,
                "seconds": 143.6,
                "rows_per_second": 501.6,
                "improved": True,
            },
        ],
    }
    if training_device is not None:
        block["device"] = training_device
    return block


def _full_neural_payload(
    split: str, *, model_device: str, training_device: str | None
) -> dict[str, object]:
    """Build a report with every field the re-render path reads."""
    return {
        "model": {
            "architecture": "transformer",
            "base_model": "distilbert-base-uncased",
            "class_weighting": "sqrt_inverse",
            "class_weights": [1.6221774783833953, 2.614336667922146, 0.6358123532355503],
            "parameters": 66955779,
            "max_length_tokens": 128,
            "device": model_device,
        },
        "training": _full_training(training_device),
        "split": split,
        "evaluated_rows": 200,
        "metrics": _full_metrics(),
        "calibration": {
            "hard_predictions": {
                "mae": 0.02,
                "mean_signed_difference": -0.0039,
                "n_products": 12,
                "n_reviews": 200,
                "note": None,
                "pearson_r": 0.9779,
            },
            "mean_negative_probability": {
                "mae": 0.016,
                "mean_signed_difference": -0.0069,
                "n_products": 12,
                "n_reviews": 200,
                "note": None,
                "pearson_r": 0.9808,
            },
        },
        "labels": "weak supervision from star ratings",
    }


def _write_final_fixture(
    script, directory: Path, *, new_format: bool = False
) -> Path:
    """Write a val/final report pair and the matching (legacy) Markdown.

    Args:
        script: Loaded evaluation module.
        directory: Temporary report directory.
        new_format: Include the new ``training.device`` and ``evaluation``
            blocks and write the new Markdown.

    Returns:
        The final Markdown path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    val_payload = _full_neural_payload(
        "val", model_device=TRAINING_DEVICE, training_device=None
    )
    final_payload = _full_neural_payload(
        "test",
        model_device=EVALUATION_DEVICE,
        training_device=TRAINING_DEVICE if new_format else None,
    )
    if new_format:
        final_payload["evaluation"] = {
            "split": "test",
            "device": EVALUATION_DEVICE,
            "rows": 200,
        }
    (directory / "sentiment_transformer_val.json").write_text(
        json.dumps(val_payload), encoding="utf-8"
    )
    (directory / "sentiment_transformer_final.json").write_text(
        json.dumps(final_payload), encoding="utf-8"
    )
    training = script._validation_training_facts(
        val_payload, architecture="transformer"
    )
    evaluation = script._evaluation_facts(final_payload)
    lines = script.render_report_markdown(training=training, evaluation=evaluation)
    if not new_format:
        text = "\n".join(lines)
        text = text.replace(f"- Training device: {TRAINING_DEVICE}\n", "")
        text = text.replace(
            f"- Training wall clock: {ELAPSED_SECONDS:,.1f}s\n",
            f"- Training wall clock: {ELAPSED_SECONDS:,.1f}s on {EVALUATION_DEVICE}\n",
        )
        text = text.replace(f"- Evaluated on: {EVALUATION_DEVICE}\n", "")
        text = text.replace("- Evaluation split: test\n", "")
        text = text.replace("- Evaluated rows:", "- Reported rows:")
        lines = text.splitlines()
    markdown = directory / "sentiment_transformer_final.md"
    markdown.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return markdown


def _forbid(name: str):
    """Return a stand-in that fails the test if the re-render calls it."""

    def fail(*args: object, **kwargs: object) -> None:
        pytest.fail(f"--rerender must not call {name}")

    return fail


def test_rerender_labels_training_and_evaluation_devices(
    script, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The training device comes from the training report, not this machine."""
    directory = tmp_path / "reports"
    markdown = _write_final_fixture(script, directory)
    lock = script.final_lock_path(directory)
    lock.write_text('{"models": []}\n', encoding="utf-8")
    before_lock = lock.read_text(encoding="utf-8")
    for name in ("final_lock_path", "check_final_lock", "write_final_lock"):
        monkeypatch.setattr(script, name, _forbid(name), raising=False)

    paths = script.rerender_final_reports(directory)

    assert paths == [markdown]
    text = markdown.read_text(encoding="utf-8")
    assert "- Training device: cuda" in text
    assert "- Evaluated on: cpu" in text
    assert "- Evaluation split: test" in text
    assert "- Evaluated rows: 200 (the full test split)" in text
    wall_clock = next(
        line for line in text.splitlines() if line.startswith("- Training wall clock:")
    )
    assert wall_clock == "- Training wall clock: 671.5s"
    assert "| Accuracy | 0.9000 |" in text
    assert "| Macro-F1 | 0.7314 |" in text
    assert "| Mean absolute error | 0.0200 |" in text
    assert "0.0160" in text
    assert "ROC AUC=0.9893" in text
    assert lock.read_text(encoding="utf-8") == before_lock


def test_rerender_is_idempotent(script, tmp_path: Path) -> None:
    """Running it twice changes nothing after the first pass."""
    markdown = _write_final_fixture(script, tmp_path / "reports")
    script.rerender_final_reports(tmp_path / "reports")
    first = markdown.read_text(encoding="utf-8")
    script.rerender_final_reports(tmp_path / "reports")
    assert markdown.read_text(encoding="utf-8") == first


def test_rerender_accepts_the_new_report_blocks(script, tmp_path: Path) -> None:
    """A report already carrying evaluation facts is re-rendered unchanged."""
    markdown = _write_final_fixture(script, tmp_path / "reports", new_format=True)
    before = markdown.read_text(encoding="utf-8")
    script.rerender_final_reports(tmp_path / "reports")
    assert markdown.read_text(encoding="utf-8") == before


def test_rerender_refuses_to_change_a_metric(script, tmp_path: Path) -> None:
    """A JSON/Markdown metric disagreement fails instead of rewriting the number."""
    directory = tmp_path / "reports"
    markdown = _write_final_fixture(script, directory)
    before = markdown.read_text(encoding="utf-8")
    final_json = directory / "sentiment_transformer_final.json"
    payload = json.loads(final_json.read_text(encoding="utf-8"))
    payload["metrics"]["accuracy"] = 0.1
    final_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(script.DataError, match="would change"):
        script.rerender_final_reports(directory)
    assert markdown.read_text(encoding="utf-8") == before


def test_rerender_refuses_a_stale_wall_clock(script, tmp_path: Path) -> None:
    """The old wall-clock number must still equal the training report's."""
    directory = tmp_path / "reports"
    markdown = _write_final_fixture(script, directory)
    text = markdown.read_text(encoding="utf-8")
    markdown.write_text(text.replace("671.5s on cpu", "999.9s on cpu"), encoding="utf-8")

    with pytest.raises(script.DataError, match="wall clock"):
        script.rerender_final_reports(directory)


def test_rerender_needs_the_validation_report(script, tmp_path: Path) -> None:
    """Without the training report the training device cannot be recovered."""
    directory = tmp_path / "reports"
    _write_final_fixture(script, directory)
    (directory / "sentiment_transformer_val.json").unlink()

    with pytest.raises(script.DataError, match="sentiment_transformer_val.json"):
        script.rerender_final_reports(directory)


def test_rerender_mode_rejects_final_only_flags(script) -> None:
    """`--device`/`--final` would be silent no-ops in re-render mode."""
    assert script.main(["--rerender", "--device", "cpu"]) == 1
    assert script.main(["--rerender", "--final"]) == 1


def test_rerender_import_path_loads_no_torch_or_model() -> None:
    """Importing the script for `--rerender` must not import torch/transformers."""
    code = "\n".join(
        [
            "import importlib.util, sys",
            f"spec = importlib.util.spec_from_file_location('mk_eval', r'{_SCRIPT}')",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules[spec.name] = module",
            "spec.loader.exec_module(module)",
            "forbidden = ('torch', 'transformers', "
            "'munasib.review_intel.sentiment.models')",
            "loaded = [name for name in forbidden if name in sys.modules]",
            "assert not loaded, loaded",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_SCRIPT.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

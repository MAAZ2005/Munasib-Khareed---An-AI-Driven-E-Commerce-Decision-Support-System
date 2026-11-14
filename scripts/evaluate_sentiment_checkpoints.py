"""Compare the sentiment models from their reports, and optionally score the test split.

Three modes, one contract: **no metric is recomputed here**. The validation table
reads every value from the JSON reports written by
``scripts/train_sentiment_baseline.py`` and ``scripts/train_sentiment.py``
(``sentiment_baseline_val.json``, ``sentiment_rnn_val.json``,
``sentiment_transformer_val.json``), so the table cannot drift away from the
reports it summarises.

``--final`` is the one-shot hold-out run. It scores the ``test`` split with every
model that is available locally - the TF-IDF artifact named in the config, the
DistilBERT checkpoint, and the BiGRU checkpoint only if it exists - and writes
``sentiment_<model>_final`` reports plus a separate test comparison table. Test
results may be looked at once; reusing them for tuning turns the hold-out into a
second validation split. The script therefore writes
``reports/evaluation/sentiment_test_split.lock.json`` *before* scoring, with the
date, the row limit and the SHA-256 of every checkpoint involved, and refuses a
second ``--final`` run unless
``--i-know-this-reuses-the-test-split`` is passed explicitly.

``--rerender`` rewrites the Markdown of the existing neural ``_final`` reports
from their JSON plus the training facts in the matching ``_val`` report. It
loads no model and no review rows, never reads or writes the lock file, and
refuses to touch a metric: the metric body must render identically, or the run
fails.

Usage::

    python scripts/evaluate_sentiment_checkpoints.py            # validation table
    python scripts/evaluate_sentiment_checkpoints.py --final    # one-shot test run
    python scripts/evaluate_sentiment_checkpoints.py --final --device cpu --limit 500
    python scripts/evaluate_sentiment_checkpoints.py --final --i-know-this-reuses-the-test-split
    python scripts/evaluate_sentiment_checkpoints.py --rerender  # Markdown labels only

``--final`` scores the test split with the checkpoint as it is on disk; it never
trains, tunes or selects anything. ``--limit`` exists for CPU smoke runs and is
recorded in the report ("smoke slice") and in the lock file, so a limited run can
never be mistaken for the full hold-out.

Exit codes: 0 - the table (and, with ``--final``, the reports) were written;
1 - the run could not start or failed; 2 - ``--final`` was refused because the
lock file shows the test split was already used.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from munasib.core.config import SentimentConfig, Settings, load_settings
from munasib.core.constants import (
    SENTIMENT_LABEL_NAMES,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABELS,
    SPLIT_TEST,
    SPLIT_VAL,
)
from munasib.core.exceptions import ConfigError, DataError
from munasib.review_intel.sentiment.devices import DEVICE_CHOICES
from munasib.review_intel.sentiment.metrics import (
    ClassificationMetrics,
    ProductCalibration,
    render_calibration_table,
    render_metrics_table,
    write_calibration_csv,
    write_evaluation_report,
)
from munasib.review_intel.sentiment.reporting import (
    EpochHistoryRow,
    EvaluationFacts,
    TrainingFacts,
    render_report_markdown,
    report_stem,
)

if TYPE_CHECKING:
    import torch

    from munasib.review_intel.sentiment.train import (
        EvaluatedSplit,
        TrainConfig,
        TrainingOutcome,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_sentiment_checkpoints")

#: Script name recorded in the generated reports and the lock file.
SCRIPT = "scripts/evaluate_sentiment_checkpoints.py"

#: File stem of the validation baseline report (both variants live inside it).
BASELINE_VAL_STEM = "sentiment_baseline_val"

#: File stem of the final baseline report written by ``--final``.
BASELINE_FINAL_STEM = "sentiment_baseline_final"

#: Neural architectures and the report suffix used for each.
NEURAL_ARCHITECTURES: tuple[str, ...] = ("transformer", "rnn")

#: Display names used in the comparison table.
MODEL_DISPLAY: dict[str, str] = {
    "transformer": "DistilBERT",
    "rnn": "BiGRU",
    "tfidf": "TF-IDF + logistic regression",
}

#: Baseline variants recorded by the p12a report, in display order.
BASELINE_VARIANTS: tuple[str, ...] = ("balanced", "none")

#: Comparison table stems.
COMPARISON_VAL_STEM = "sentiment_model_comparison"
COMPARISON_FINAL_STEM = "sentiment_model_comparison_test"

#: Lock file proving ``--final`` was already used.
LOCK_FILENAME = "sentiment_test_split.lock.json"

#: Legacy header line: the old renderer put the *evaluating* machine's device on
#: the training wall clock. ``--rerender`` recognises it so it can relabel it.
_LEGACY_WALL_CLOCK = re.compile(r"^- Training wall clock: ([\d,]+(?:\.\d+)?)s on (\S+)$")

#: Legacy header line naming the evaluated rows.
_LEGACY_REPORTED_ROWS = re.compile(r"^- Reported rows: ([\d,]+)")

#: Review table the final evaluation reads (same file the trainers use).
REVIEWS_FILENAME = "reviews.parquet"

#: Column order of the comparison CSV.
CSV_COLUMNS: tuple[str, ...] = (
    "model",
    "variant",
    "split",
    "rows",
    "accuracy",
    "macro_f1",
    "neutral_f1",
    "binary_f1",
    "binary_roc_auc",
    "calibration_mae_hard",
    "calibration_mae_soft",
    "majority_accuracy",
    "majority_macro_f1",
    "source",
)

WEAK_LABEL_NOTE = (
    "Labels are **weak supervision from star ratings** (1-2 negative, 3 neutral, "
    "4-5 positive), so every score measures agreement with that proxy, not "
    "human-judged sentiment."
)


@dataclass(frozen=True)
class ComparisonRow:
    """One model's numbers, exactly as recorded in its report.

    Attributes:
        model: Display name (``"DistilBERT"``, ``"BiGRU"``, ...).
        variant: Baseline class weighting (``"balanced"``/``"none"``) or None.
        split: Split the numbers describe.
        rows: Rows evaluated.
        accuracy: Three-class accuracy.
        macro_f1: Unweighted mean of the three per-label F1 scores.
        neutral_f1: F1 of the neutral class (the weakest class).
        binary_f1: F1 of the binary negative-vs-rest view.
        binary_roc_auc: ROC AUC of that binary view.
        calibration_mae_hard: Product-level MAE of the hard-prediction negative share.
        calibration_mae_soft: Product-level MAE using the mean negative probability
            (``None`` when the source report did not compute it - and always
            ``None`` for the majority-class reference row).
        majority_accuracy: Majority-class baseline accuracy for the same rows.
        majority_macro_f1: Majority-class baseline macro-F1 for the same rows.
        source: Report file the row was read from.
    """

    model: str
    variant: str | None
    split: str
    rows: int
    accuracy: float
    macro_f1: float
    neutral_f1: float | None
    binary_f1: float | None
    binary_roc_auc: float | None
    calibration_mae_hard: float | None
    calibration_mae_soft: float | None
    majority_accuracy: float
    majority_macro_f1: float
    source: str

    @property
    def label(self) -> str:
        """Table label combining the model and baseline variant."""
        if self.variant is None:
            return self.model
        return f"{self.model} ({self.variant})"


@dataclass(frozen=True)
class FinalRun:
    """Artifacts written by one ``--final`` invocation.

    Attributes:
        lock: Lock file proving the test split was used.
        reports: Report files written by the run, relative to the report dir.
        rows: Test-split comparison rows.
        notes: Missing/limited-report notes shown with the table.
        markdown: Written comparison Markdown.
        csv: Written comparison CSV.
    """

    lock: Path
    reports: tuple[str, ...]
    rows: tuple[ComparisonRow, ...]
    notes: tuple[str, ...]
    markdown: Path
    csv: Path


class TestSplitLockedError(RuntimeError):
    """``--final`` was asked for after the test split had already been used."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the sentiment model comparison table from the JSON reports, "
            "score the test split once with --final, or re-render the final "
            "Markdown from the existing JSON with --rerender."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help=(
            "Score the test split once with every available model and write "
            "sentiment_<model>_final reports plus a test comparison table."
        ),
    )
    parser.add_argument(
        "--rerender",
        action="store_true",
        help=(
            "Re-render the Markdown of the existing neural _final reports from "
            "their JSON plus the training facts in the matching _val report. "
            "Loads no model or test data and never touches the lock file."
        ),
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Compute device for --final (default: auto, prefers CUDA).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Score only the first N test rows (CPU smoke runs); the report and the "
            "lock file record the limit."
        ),
    )
    parser.add_argument(
        "--i-know-this-reuses-the-test-split",
        dest="reuse_test_split",
        action="store_true",
        help=(
            "Run --final even though the lock file says the test split was already "
            "used. Only for a deliberate re-run after a failure."
        ),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one report JSON.

    Args:
        path: Report file.

    Returns:
        Parsed payload.

    Raises:
        DataError: If the file is missing, not JSON, or not an object.
    """
    if not path.is_file():
        raise DataError(f"report not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"report is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataError(f"report must be a JSON object: {path}")
    return payload


def _read_optional(path: Path, notes: list[str]) -> dict[str, Any] | None:
    """Read a report if it exists, recording a note when it does not.

    Args:
        path: Report file.
        notes: List the missing-file note is appended to.

    Returns:
        Parsed payload, or None when the file is missing.
    """
    if not path.is_file():
        notes.append(f"missing report: `{path.name}` (that model is not in this table)")
        return None
    return _read_json(path)


def _validation_training_facts(
    payload: dict[str, Any], *, architecture: str
) -> TrainingFacts:
    """Read the training facts a checkpoint's validation report recorded.

    Args:
        payload: ``sentiment_<architecture>_val.json``.
        architecture: ``"transformer"`` or ``"rnn"``.

    Returns:
        Training device, wall clock, epochs, selection metric and history.

    Raises:
        DataError: If the report records no training device.
    """
    model = payload["model"]
    training = payload["training"]
    device = training.get("device") or model.get("device")
    if not isinstance(device, str) or not device:
        raise DataError(
            f"the {architecture} validation report records no training device; "
            "the training facts cannot be reconstructed from the evaluating machine"
        )
    history = tuple(
        EpochHistoryRow(
            epoch=int(record["epoch"]),
            train_loss=float(record["train_loss"]),
            val_macro_f1=float(record["val_macro_f1"]),
            val_accuracy=float(record["val_accuracy"]),
            seconds=float(record["seconds"]),
            rows_per_second=float(record["rows_per_second"]),
            improved=bool(record["improved"]),
        )
        for record in training["history"]
    )
    val_subset = training["val_subset_requested"]
    return TrainingFacts(
        architecture=architecture,
        class_weighting=str(model["class_weighting"]),
        class_weights=tuple(float(weight) for weight in model["class_weights"]),
        parameters=int(model["parameters"]),
        train_rows=int(training["train_rows"]),
        selection_rows=int(training["selection_rows"]),
        val_subset=None if val_subset is None else int(val_subset),
        best_epoch=int(training["best_epoch"]),
        epochs_run=int(training["epochs_run"]),
        selection_metric=str(training["selection_metric"]),
        elapsed_seconds=float(training["elapsed_seconds"]),
        device=device,
        history=history,
    )


def _evaluation_facts(payload: dict[str, Any]) -> EvaluationFacts:
    """Read the evaluation facts from a final report, without recomputing them.

    Args:
        payload: ``sentiment_<architecture>_final.json``.

    Returns:
        Split, evaluating device, rows and the recorded metrics/calibration.

    Raises:
        DataError: If the report is not a test-split report, names no evaluating
            device, or its row counts disagree.
    """
    split = str(payload["split"])
    if split != SPLIT_TEST:
        raise DataError(f"not a one-shot test report: split={split!r}")
    evaluation = payload.get("evaluation") or {}
    device = evaluation.get("device") or payload["model"].get("device")
    if not isinstance(device, str) or not device:
        raise DataError("the final report records no evaluating device")
    rows = int(payload["evaluated_rows"])
    recorded_rows = evaluation.get("rows")
    if recorded_rows is not None and int(recorded_rows) != rows:
        raise DataError(
            "the final report disagrees with itself about the evaluated rows: "
            f"evaluation.rows={recorded_rows} evaluated_rows={rows}"
        )
    training = payload["training"]
    return EvaluationFacts(
        split=split,
        device=device,
        rows=rows,
        full_split=bool(training["full_report_split"]),
        metrics=ClassificationMetrics.from_dict(payload["metrics"]),
        calibration_hard=ProductCalibration.from_dict(
            payload["calibration"]["hard_predictions"]
        ),
        calibration_soft=ProductCalibration.from_dict(
            payload["calibration"]["mean_negative_probability"]
        ),
    )


def _check_training_facts_match(
    training: TrainingFacts, payload: dict[str, Any]
) -> None:
    """Verify a final report's training block still matches the validation report.

    The final report is written from the validation report, so the two must
    agree. A mismatch means one of them was edited; re-rendering would then pick
    a side and hide the discrepancy.

    Args:
        training: Facts read from the validation report.
        payload: Final report of the same checkpoint.

    Raises:
        DataError: If any training fact differs.
    """
    model = payload["model"]
    block = payload["training"]
    mismatches: list[str] = []
    checks = {
        "parameters": (int(model["parameters"]), training.parameters),
        "train_rows": (int(block["train_rows"]), training.train_rows),
        "selection_rows": (int(block["selection_rows"]), training.selection_rows),
        "best_epoch": (int(block["best_epoch"]), training.best_epoch),
        "epochs_run": (int(block["epochs_run"]), training.epochs_run),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            mismatches.append(f"{name} final={actual} val={expected}")
    if float(block["elapsed_seconds"]) != training.elapsed_seconds:
        mismatches.append(
            f"elapsed_seconds final={block['elapsed_seconds']} val={training.elapsed_seconds}"
        )
    recorded_device = block.get("device")
    if recorded_device is not None and recorded_device != training.device:
        mismatches.append(
            f"training device final={recorded_device} val={training.device}"
        )
    expected_history = [
        {
            "epoch": record.epoch,
            "train_loss": record.train_loss,
            "val_macro_f1": record.val_macro_f1,
            "val_accuracy": record.val_accuracy,
            "seconds": record.seconds,
            "rows_per_second": record.rows_per_second,
            "improved": record.improved,
        }
        for record in training.history
    ]
    if block["history"] != expected_history:
        mismatches.append("epoch history differs")
    if mismatches:
        raise DataError(
            "refusing to re-render: the final report's training facts disagree "
            "with the validation report (" + "; ".join(mismatches) + ")"
        )


def _assert_only_labels_changed(
    old_lines: Sequence[str],
    new_lines: Sequence[str],
    *,
    path: Path,
    training: TrainingFacts,
    evaluation: EvaluationFacts,
) -> None:
    """Refuse a re-render that would change any metric, epoch or number.

    Args:
        old_lines: Markdown currently on disk.
        new_lines: Markdown rendered from the JSON reports.
        path: Markdown path, used in error messages.
        training: Training facts used for the new render.
        evaluation: Evaluation facts used for the new render.

    Raises:
        DataError: If the body differs, or the old header does not hold exactly
            the recorded wall clock, evaluating device and row count.
    """
    if not old_lines or old_lines[0] != new_lines[0]:
        raise DataError(f"refusing to re-render {path.name}: the report title would change")
    marker = "## Epoch history"
    try:
        old_body = old_lines.index(marker)
        new_body = new_lines.index(marker)
    except ValueError as exc:
        raise DataError(
            f"refusing to re-render {path.name}: no {marker!r} section to compare"
        ) from exc
    if old_lines[old_body:] != new_lines[new_body:]:
        common = min(len(old_lines) - old_body, len(new_lines) - new_body)
        for offset in range(common):
            old_line = old_lines[old_body + offset]
            new_line = new_lines[new_body + offset]
            if old_line != new_line:
                raise DataError(
                    f"refusing to re-render {path.name}: a metric, epoch or table "
                    f"would change ({old_line!r} -> {new_line!r})"
                )
        raise DataError(
            f"refusing to re-render {path.name}: the metric body has a different length"
        )

    header = old_lines[:old_body]
    wall_line = next(
        (line for line in header if line.startswith("- Training wall clock:")), None
    )
    if wall_line is None:
        raise DataError(f"refusing to re-render {path.name}: no training wall-clock line")
    match = _LEGACY_WALL_CLOCK.match(wall_line)
    if match is None:
        raise DataError(
            f"refusing to re-render {path.name}: unrecognised header line {wall_line!r}; "
            "only 'Training wall clock: <seconds>s on <device>' can be relabelled"
        )
    old_seconds = float(match.group(1).replace(",", ""))
    if f"{old_seconds:,.1f}" != f"{training.elapsed_seconds:,.1f}":
        raise DataError(
            f"refusing to re-render {path.name}: the wall clock {old_seconds:.1f}s "
            f"does not match the training report's {training.elapsed_seconds:.1f}s"
        )
    if match.group(2) != evaluation.device:
        raise DataError(
            f"refusing to re-render {path.name}: expected the old label to name the "
            f"evaluating device {evaluation.device!r}, found {match.group(2)!r}"
        )

    reported_line = next(
        (line for line in header if line.startswith("- Reported rows:")), None
    )
    reported = None if reported_line is None else _LEGACY_REPORTED_ROWS.match(reported_line)
    if reported is None:
        raise DataError(f"refusing to re-render {path.name}: no reported-rows line")
    if int(reported.group(1).replace(",", "")) != evaluation.rows:
        raise DataError(
            f"refusing to re-render {path.name}: the reported row count "
            f"{reported.group(1)} does not match the JSON's {evaluation.rows:,}"
        )


def rerender_final_reports(report_dir: Path) -> list[Path]:
    """Re-render the neural final Markdown from JSON, changing labels only.

    The training facts come from each checkpoint's own validation report, never
    from the machine running this command. No model, review row or lock file is
    read or written. A report whose metric body would change is left untouched
    and the run fails.

    Args:
        report_dir: Directory holding the evaluation reports.

    Returns:
        The Markdown files that are now up to date (written or already correct).

    Raises:
        DataError: If no final neural report exists, its validation report or
            Markdown is missing, or a re-render would change a metric.
    """
    up_to_date: list[Path] = []
    for architecture in NEURAL_ARCHITECTURES:
        stem = report_stem(architecture, SPLIT_TEST)
        final_json = report_dir / f"{stem}.json"
        final_markdown = report_dir / f"{stem}.md"
        if not final_json.is_file():
            continue
        val_json = report_dir / f"sentiment_{architecture}_val.json"
        if not val_json.is_file():
            raise DataError(
                f"cannot re-render {final_json.name}: {val_json.name} is missing, "
                "so the training device and wall clock cannot be read from the run "
                "that produced the checkpoint"
            )
        if not final_markdown.is_file():
            raise DataError(
                f"cannot re-render {final_json.name}: {final_markdown.name} is missing"
            )
        training = _validation_training_facts(_read_json(val_json), architecture=architecture)
        payload = _read_json(final_json)
        evaluation = _evaluation_facts(payload)
        _check_training_facts_match(training, payload)
        new_lines = render_report_markdown(training=training, evaluation=evaluation)
        while new_lines and not new_lines[-1]:
            new_lines.pop()
        old_lines = final_markdown.read_text(encoding="utf-8").rstrip("\n").splitlines()
        if old_lines == new_lines:
            logger.info("rerender_unchanged path=%s", final_markdown)
            up_to_date.append(final_markdown)
            continue
        _assert_only_labels_changed(
            old_lines,
            new_lines,
            path=final_markdown,
            training=training,
            evaluation=evaluation,
        )
        final_markdown.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
        logger.info(
            "rerender_written path=%s training_device=%s evaluation_device=%s",
            final_markdown,
            training.device,
            evaluation.device,
        )
        up_to_date.append(final_markdown)
    if not up_to_date:
        raise DataError(
            f"no neural _final reports found in {report_dir}; run --final first"
        )
    return up_to_date


def _neutral_f1(metrics: dict[str, Any]) -> float:
    """Read the neutral-class F1 out of a report's per-class metrics.

    Args:
        metrics: ``metrics`` block of a neural or baseline report.

    Returns:
        Neutral F1.

    Raises:
        DataError: If the report has no neutral entry.
    """
    name = SENTIMENT_LABEL_NAMES[SENTIMENT_LABEL_NEUTRAL]
    for entry in metrics["per_class"]:
        if entry["name"] == name:
            return float(entry["f1"])
    raise DataError(f"report metrics have no {name!r} entry: {sorted(metrics['per_class'])}")


def _row_from_neural(
    payload: dict[str, Any], architecture: str, source: str
) -> ComparisonRow:
    """Turn one neural validation/final report into a comparison row.

    Args:
        payload: Parsed report.
        architecture: ``"transformer"`` or ``"rnn"``.
        source: Report filename.

    Returns:
        The row read from the report.

    Raises:
        KeyError: If the report lacks a field this table needs.
        DataError: If the neutral F1 entry is missing.
    """
    metrics = payload["metrics"]
    calibration = payload["calibration"]
    return ComparisonRow(
        model=MODEL_DISPLAY[architecture],
        variant=None,
        split=str(payload["split"]),
        rows=int(payload["evaluated_rows"]),
        accuracy=float(metrics["accuracy"]),
        macro_f1=float(metrics["macro_f1"]),
        neutral_f1=_neutral_f1(metrics),
        binary_f1=float(metrics["binary"]["f1"]),
        binary_roc_auc=float(metrics["binary"]["roc_auc"]),
        calibration_mae_hard=float(calibration["hard_predictions"]["mae"]),
        calibration_mae_soft=float(calibration["mean_negative_probability"]["mae"]),
        majority_accuracy=float(metrics["majority_baseline"]["accuracy"]),
        majority_macro_f1=float(metrics["majority_baseline"]["macro_f1"]),
        source=source,
    )


def _row_from_baseline(
    payload: dict[str, Any], split: str, variant: str, source: str
) -> ComparisonRow:
    """Turn one baseline variant into a comparison row.

    Args:
        payload: Parsed baseline report.
        split: ``"val"`` or ``"test"`` evaluation block.
        variant: ``"balanced"`` or ``"none"``.
        source: Report filename.

    Returns:
        The row read from the report.

    Raises:
        KeyError: If the report lacks the split/variant this table needs.
        DataError: If the neutral F1 entry is missing.
    """
    evaluation = payload["evaluations"][split]
    entry = evaluation["variants"][variant]
    metrics = entry["metrics"]
    return ComparisonRow(
        model=MODEL_DISPLAY["tfidf"],
        variant=variant,
        split=str(evaluation["split"]),
        rows=int(evaluation["rows"]),
        accuracy=float(metrics["accuracy"]),
        macro_f1=float(metrics["macro_f1"]),
        neutral_f1=_neutral_f1(metrics),
        binary_f1=float(metrics["binary"]["f1"]),
        binary_roc_auc=float(metrics["binary"]["roc_auc"]),
        calibration_mae_hard=float(entry["calibration"]["mae"]),
        calibration_mae_soft=None,
        majority_accuracy=float(metrics["majority_baseline"]["accuracy"]),
        majority_macro_f1=float(metrics["majority_baseline"]["macro_f1"]),
        source=source,
    )


def _majority_row(payload: dict[str, Any], source: str) -> ComparisonRow:
    """Build the majority-class reference row from a report's own baseline.

    Args:
        payload: Parsed neural report (its ``majority_baseline`` block).
        source: Report filename.

    Returns:
        The reference row.
    """
    metrics = payload["metrics"]
    majority = metrics["majority_baseline"]
    return ComparisonRow(
        model=f"Majority class ({majority['name']})",
        variant=None,
        split=str(payload["split"]),
        rows=int(payload["evaluated_rows"]),
        accuracy=float(majority["accuracy"]),
        macro_f1=float(majority["macro_f1"]),
        neutral_f1=None,
        binary_f1=None,
        binary_roc_auc=None,
        calibration_mae_hard=None,
        calibration_mae_soft=None,
        majority_accuracy=float(majority["accuracy"]),
        majority_macro_f1=float(majority["macro_f1"]),
        source=source,
    )


def collect_validation_rows(report_dir: Path) -> tuple[list[ComparisonRow], list[str]]:
    """Build the validation comparison rows from the committed reports.

    Args:
        report_dir: Directory holding the JSON reports.

    Returns:
        ``(rows, notes)`` with one row per model/variant plus a majority-class
        reference row, and one note per missing report.
    """
    notes: list[str] = []
    rows: list[ComparisonRow] = []

    baseline_payload = _read_optional(report_dir / f"{BASELINE_VAL_STEM}.json", notes)
    if baseline_payload is not None:
        available = baseline_payload["evaluations"][SPLIT_VAL]["variants"]
        for variant in BASELINE_VARIANTS:
            if variant in available:
                rows.append(
                    _row_from_baseline(
                        baseline_payload, SPLIT_VAL, variant, f"{BASELINE_VAL_STEM}.json"
                    )
                )
            else:
                notes.append(f"`{BASELINE_VAL_STEM}.json` records no `{variant}` variant")

    neural_payloads: list[tuple[str, dict[str, Any]]] = []
    for architecture in NEURAL_ARCHITECTURES:
        filename = f"sentiment_{architecture}_val.json"
        payload = _read_optional(report_dir / filename, notes)
        if payload is None:
            continue
        neural_payloads.append((architecture, payload))
        rows.append(_row_from_neural(payload, architecture, filename))

    if neural_payloads:
        architecture, payload = neural_payloads[0]
        rows.append(_majority_row(payload, f"sentiment_{architecture}_val.json"))
    return rows, notes


def collect_test_rows(report_dir: Path) -> tuple[list[ComparisonRow], list[str]]:
    """Build the test-split comparison rows from the ``--final`` reports.

    Args:
        report_dir: Directory holding the JSON reports.

    Returns:
        ``(rows, notes)``. A note flags reports that covered a different number
        of rows (for example a ``--limit`` smoke run), because those numbers are
        not directly comparable.
    """
    notes: list[str] = []
    rows: list[ComparisonRow] = []

    baseline_payload = _read_optional(report_dir / f"{BASELINE_FINAL_STEM}.json", notes)
    if baseline_payload is not None:
        variants = baseline_payload["evaluations"][SPLIT_TEST]["variants"]
        for variant in BASELINE_VARIANTS:
            if variant in variants:
                rows.append(
                    _row_from_baseline(
                        baseline_payload, SPLIT_TEST, variant, f"{BASELINE_FINAL_STEM}.json"
                    )
                )

    neural_payloads: list[tuple[str, dict[str, Any]]] = []
    for architecture in NEURAL_ARCHITECTURES:
        filename = f"{report_stem(architecture, SPLIT_TEST)}.json"
        payload = _read_optional(report_dir / filename, notes)
        if payload is None:
            continue
        neural_payloads.append((architecture, payload))
        rows.append(_row_from_neural(payload, architecture, filename))

    if neural_payloads:
        architecture, payload = neural_payloads[0]
        rows.append(
            _majority_row(
                payload, f"{report_stem(architecture, SPLIT_TEST)}.json"
            )
        )
    if len({row.rows for row in rows}) > 1:
        notes.append(
            "rows differ between reports: the runs did not cover the same test rows "
            "(check `--limit` in each report)"
        )
    return rows, notes


def _fmt(value: float | None, spec: str = ".4f") -> str:
    """Format an optional metric, rendering ``None`` as ``n/a``.

    Args:
        value: Metric value.
        spec: Format specification.

    Returns:
        Formatted text.
    """
    return "n/a" if value is None else format(value, spec)


def render_comparison(
    rows: Sequence[ComparisonRow], notes: Sequence[str], *, split: str
) -> list[str]:
    """Render the comparison table as Markdown.

    Args:
        rows: Rows collected from the reports.
        notes: Missing/limited-report notes.
        split: Split the table describes.

    Returns:
        Markdown lines.
    """
    heading = (
        f"{split} split - one-shot hold-out"
        if split == SPLIT_TEST
        else f"{split} split - model selection"
    )
    lines = [
        f"# Sentiment model comparison - {heading}",
        "",
        WEAK_LABEL_NOTE,
        "",
        "Every number below is read from the JSON report named in the **Source** "
        "column; this table computes no metric of its own.",
        "",
        "| Model | Rows | Accuracy | Macro-F1 | Neutral F1 | Binary F1 | "
        "Binary ROC AUC | Calib MAE (hard) | Calib MAE (soft) | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.label} | {row.rows:,} | {row.accuracy:.4f} | {row.macro_f1:.4f} | "
            f"{_fmt(row.neutral_f1)} | {_fmt(row.binary_f1)} | "
            f"{_fmt(row.binary_roc_auc)} | {_fmt(row.calibration_mae_hard)} | "
            f"{_fmt(row.calibration_mae_soft)} | `{row.source}` |"
        )
    lines.append("")
    lines.append(
        "Calibration MAE compares a product's predicted negative share with its "
        "weak-label negative share; the soft variant averages the negative "
        "probability instead of the argmax."
    )
    if split == SPLIT_TEST:
        lines.extend(
            [
                "",
                "The test split is a hold-out: it was scored once, after every "
                "setting and epoch was frozen on validation. Re-using these numbers "
                "to change the models would turn the hold-out into a second "
                "validation split.",
            ]
        )
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    lines.append("")
    return lines


def write_comparison(
    report_dir: Path,
    rows: Sequence[ComparisonRow],
    notes: Sequence[str],
    *,
    stem: str,
    split: str,
) -> tuple[Path, Path]:
    """Write the comparison Markdown and CSV.

    Args:
        report_dir: Destination directory.
        rows: Rows collected from the reports.
        notes: Missing/limited-report notes.
        stem: File stem.
        split: Split the table describes.

    Returns:
        ``(markdown_path, csv_path)``.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / f"{stem}.md"
    markdown_path.write_text(
        "\n".join(render_comparison(rows, notes, split=split)), encoding="utf-8"
    )
    csv_path = report_dir / f"{stem}.csv"
    frame = pd.DataFrame(
        [
            {
                "model": row.model,
                "variant": row.variant or "",
                "split": row.split,
                "rows": row.rows,
                "accuracy": row.accuracy,
                "macro_f1": row.macro_f1,
                "neutral_f1": row.neutral_f1,
                "binary_f1": row.binary_f1,
                "binary_roc_auc": row.binary_roc_auc,
                "calibration_mae_hard": row.calibration_mae_hard,
                "calibration_mae_soft": row.calibration_mae_soft,
                "majority_accuracy": row.majority_accuracy,
                "majority_macro_f1": row.majority_macro_f1,
                "source": row.source,
            }
            for row in rows
        ],
        columns=list(CSV_COLUMNS),
    )
    frame.to_csv(csv_path, index=False, float_format="%.6f")
    return markdown_path, csv_path


def final_lock_path(report_dir: Path) -> Path:
    """Path of the test-split lock file.

    Args:
        report_dir: Evaluation report directory.

    Returns:
        ``<report_dir>/sentiment_test_split.lock.json``.
    """
    return report_dir / LOCK_FILENAME


def check_final_lock(path: Path, *, reuse: bool) -> None:
    """Refuse a second ``--final`` run unless reuse was requested explicitly.

    Args:
        path: Lock file path.
        reuse: Whether ``--i-know-this-reuses-the-test-split`` was passed.

    Raises:
        TestSplitLockedError: If the lock exists and ``reuse`` is False.
    """
    if not path.is_file():
        return
    if reuse:
        logger.warning(
            "test_split_lock_overridden lock=%s - the test split is being scored again",
            path,
        )
        return
    raise TestSplitLockedError(
        f"{path} exists: the test split was already scored. Look at the existing "
        "sentiment_*_final reports instead, or pass "
        "--i-know-this-reuses-the-test-split after a failed run."
    )


def write_final_lock(
    path: Path,
    entries: Sequence[dict[str, Any]],
    *,
    limit: int | None,
    split: str = SPLIT_TEST,
) -> Path:
    """Record that the test split was used, before any scoring starts.

    Args:
        path: Lock file path.
        entries: One ``model``/``checkpoint``/hash mapping per model.
        limit: Row limit of the run, or ``None`` for the full split.
        split: Split being scored.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": SCRIPT,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": split,
        "limit": limit,
        "models": list(entries),
        "note": (
            "Written before scoring the test split. A second --final run is "
            "refused unless --i-know-this-reuses-the-test-split is passed."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _train_config_from_report(
    payload: dict[str, Any],
    *,
    architecture: str,
    checkpoint_dir: Path,
    device: torch.device,
    full_report_split: bool,
) -> TrainConfig:
    """Rebuild the training settings recorded in a validation report.

    Only the historical training metadata is read from the report; no value here
    is recomputed or guessed.

    Args:
        payload: Validation report of the same checkpoint.
        architecture: Architecture name.
        checkpoint_dir: ``best/`` directory that was scored.
        device: Device used for the test evaluation.
        full_report_split: Whether the test evaluation covers the whole split.

    Returns:
        The settings the checkpoint was trained with.

    Raises:
        DataError: If the report records no training device.
    """
    from munasib.review_intel.sentiment.train import TrainConfig

    model = payload["model"]
    training = payload["training"]
    val_subset = training["val_subset_requested"]
    training_device = training.get("device") or model.get("device")
    if not isinstance(training_device, str) or not training_device:
        raise DataError(
            "the validation report records no training device, so the final "
            "report cannot label the training wall clock without guessing"
        )
    return TrainConfig(
        architecture=architecture,
        output_dir=checkpoint_dir.parent,
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        epochs=int(training["epochs_requested"]),
        warmup_ratio=float(training["warmup_ratio"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        class_weighting=str(model["class_weighting"]),
        seed=int(training["seed"]),
        max_length=int(model["max_length_tokens"]),
        device=device,
        base_model=model.get("base_model"),
        resume=False,
        val_subset=None if val_subset is None else int(val_subset),
        full_report_split=full_report_split,
        training_device=training_device,
    )


def _outcome_from_report(
    payload: dict[str, Any], *, validation: EvaluatedSplit
) -> TrainingOutcome:
    """Rebuild the training outcome recorded in a validation report.

    Args:
        payload: Validation report.
        validation: Fresh test-split evaluation of the same checkpoint.

    Returns:
        Outcome whose history/epochs/weights are the recorded ones and whose
        ``validation`` is the test evaluation being reported.
    """
    from munasib.review_intel.sentiment.train import EpochRecord, TrainingOutcome

    model = payload["model"]
    training = payload["training"]
    history = tuple(EpochRecord(**record) for record in training["history"])
    return TrainingOutcome(
        best_epoch=int(training["best_epoch"]),
        epochs_run=int(training["epochs_run"]),
        history=history,
        validation=validation,
        elapsed_seconds=float(training["elapsed_seconds"]),
        class_weights=tuple(float(weight) for weight in model["class_weights"]),
        parameters=int(model["parameters"]),
    )


def _read_reviews(settings: Settings) -> pd.DataFrame:
    """Load the processed review table the final evaluation reads.

    Args:
        settings: Project settings (``paths.processed_data_dir``).

    Returns:
        The review frame.

    Raises:
        DataError: If ``reviews.parquet`` is missing or empty.
    """
    path = Path(settings.paths.processed_data_dir) / REVIEWS_FILENAME
    if not path.is_file():
        raise DataError(f"review table not found: {path}; run scripts/build_dataset.py first")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise DataError(f"review table is empty: {path}")
    logger.info("loaded reviews rows=%d path=%s", len(frame), path)
    return frame


def run_final_baseline(
    config: SentimentConfig,
    report_dir: Path,
    artifact: Path,
    reviews: pd.DataFrame,
    *,
    limit: int | None,
) -> list[str]:
    """Score the test split once with the saved TF-IDF artifact.

    Args:
        config: Sentiment settings (columns, artifact path, reports).
        report_dir: Destination report directory.
        artifact: Saved baseline joblib.
        reviews: Full review frame.
        limit: Row limit for a smoke run, or ``None``.

    Returns:
        Names of the files written.

    Raises:
        DataError: If the artifact or split is unusable.
    """
    from munasib.review_intel.sentiment.baseline import evaluate_baseline, load_baseline
    from munasib.review_intel.sentiment.metrics import compute_product_calibration
    from munasib.review_intel.sentiment.sampling import PRODUCT_COLUMN, load_eval_split

    baseline = load_baseline(artifact)
    frame = load_eval_split(reviews, SPLIT_TEST, config)
    if limit is not None:
        frame = frame.head(limit).reset_index(drop=True)
    started = time.perf_counter()
    metrics, calibration, probabilities = evaluate_baseline(baseline, frame, config)
    elapsed = time.perf_counter() - started

    truth = frame[config.label_column].astype("int64").to_numpy()
    predictions = np.asarray(
        [SENTIMENT_LABELS[index] for index in probabilities.argmax(axis=1)], dtype="int64"
    )
    soft = compute_product_calibration(
        frame[PRODUCT_COLUMN], truth, predictions, prob_negative=probabilities[:, 0]
    )
    variant = baseline.class_weight_name
    payload = {
        "script": SCRIPT,
        "artifact": {
            "path": str(artifact),
            "class_weight": variant,
            "metadata": baseline.to_metadata(),
        },
        "config": config.model_dump(mode="json"),
        "limit": limit,
        "evaluations": {
            SPLIT_TEST: {
                "split": SPLIT_TEST,
                "rows": len(frame),
                "seconds": round(elapsed, 3),
                "variants": {
                    variant: {
                        "class_weight": variant,
                        "model": baseline.to_metadata(),
                        "metrics": metrics.to_dict(),
                        "calibration": calibration.to_dict(),
                        "soft_calibration": soft.to_dict(),
                        "train_seconds": 0.0,
                        "eval_seconds": round(elapsed, 3),
                    }
                },
            }
        },
        "elapsed_seconds": round(elapsed, 3),
        "labels": "weak supervision from star ratings (1-2 negative, 3 neutral, 4-5 positive)",
    }
    report_note = "the full test split" if limit is None else "smoke slice, not the full split"
    markdown = [
        f"# Sentiment baseline report - test split ({variant})",
        "",
        WEAK_LABEL_NOTE,
        "",
        f"- Reported rows: {len(frame):,} ({report_note})",
        f"- Class weighting: {variant}",
        f"- Vocabulary size: {baseline.vocabulary_size:,}",
        f"- Training rows: {baseline.train_rows:,}",
        "",
        "This is the one-shot hold-out: the artifact was frozen before these rows "
        "were scored.",
    ]
    markdown.extend(render_metrics_table("Three-class metrics", metrics))
    markdown.extend(render_calibration_table("Product calibration - hard predictions", calibration))
    markdown.extend(
        render_calibration_table("Product calibration - negative probability", soft)
    )
    json_path, markdown_path = write_evaluation_report(
        report_dir, BASELINE_FINAL_STEM, payload, markdown
    )
    hard_csv = write_calibration_csv(
        report_dir / f"{BASELINE_FINAL_STEM}_hard_calibration.csv", calibration
    )
    soft_csv = write_calibration_csv(
        report_dir / f"{BASELINE_FINAL_STEM}_soft_calibration.csv", soft
    )
    return [path.name for path in (json_path, markdown_path, hard_csv, soft_csv)]


def run_final_neural(
    config: SentimentConfig,
    report_dir: Path,
    architecture: str,
    checkpoint_dir: Path,
    reviews: pd.DataFrame,
    *,
    device: torch.device,
    limit: int | None,
) -> list[str]:
    """Score the test split once with one neural checkpoint.

    Args:
        config: Sentiment settings.
        report_dir: Destination report directory.
        architecture: ``"transformer"`` or ``"rnn"``.
        checkpoint_dir: ``best/`` directory.
        reviews: Full review frame.
        device: Resolved torch device.
        limit: Row limit for a smoke run, or ``None``.

    Returns:
        Names of the files written.

    Raises:
        DataError: If the validation report needed for training metadata is
            missing, or the checkpoint cannot be loaded.
    """
    from munasib.review_intel.sentiment.data import TextEncoder
    from munasib.review_intel.sentiment.models import load_model
    from munasib.review_intel.sentiment.sampling import load_eval_split
    from munasib.review_intel.sentiment.train import (
        evaluate_predictions,
        frame_to_dataset,
        predict,
        write_reports,
    )

    val_path = report_dir / f"sentiment_{architecture}_val.json"
    if not val_path.is_file():
        raise DataError(
            f"cannot write the final {architecture} report: {val_path.name} is "
            "missing, so the training metadata it records cannot be reconstructed"
        )
    val_payload = _read_json(val_path)

    saved = load_model(checkpoint_dir)
    encoder = TextEncoder(
        max_length=saved.meta.max_length,
        tokenizer=saved.tokenizer,
        vocabulary=saved.vocabulary,
    )
    frame = load_eval_split(reviews, SPLIT_TEST, config)
    if limit is not None:
        frame = frame.head(limit).reset_index(drop=True)
    dataset = frame_to_dataset(frame, config, encoder)

    started = time.perf_counter()
    predictions = predict(
        saved.model,
        dataset,
        batch_size=int(val_payload["training"]["batch_size"]),
        device=device,
    )
    evaluation = evaluate_predictions(predictions, split=SPLIT_TEST)
    elapsed = time.perf_counter() - started

    settings = _train_config_from_report(
        val_payload,
        architecture=architecture,
        checkpoint_dir=checkpoint_dir,
        device=device,
        full_report_split=limit is None,
    )
    outcome = _outcome_from_report(val_payload, validation=evaluation)
    paths = write_reports(
        outcome=outcome,
        settings=settings,
        evaluation=evaluation,
        report_dir=report_dir,
        train_rows=int(val_payload["training"]["train_rows"]),
        selection_rows=int(val_payload["training"]["selection_rows"]),
    )
    logger.info(
        "final_report architecture=%s rows=%d seconds=%.1f", architecture, len(frame), elapsed
    )
    return [path.name for path in paths]


def run_final(settings: Settings, args: argparse.Namespace) -> FinalRun:
    """Score the test split once per available model and write the reports.

    Args:
        settings: Loaded project settings.
        args: Parsed arguments.

    Returns:
        Paths and comparison rows written by the run.

    Raises:
        TestSplitLockedError: If the lock file exists and reuse was not forced.
        DataError: If no model is available or a report/checkpoint is unusable.
        ValueError: On an invalid limit.
    """
    from munasib.review_intel.sentiment.predictor import (
        describe_checkpoint,
        file_sha256,
    )
    from munasib.review_intel.sentiment.train import (
        BEST_CHECKPOINT_DIR,
        resolve_device,
    )

    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be >= 1, got {args.limit}")
    config = settings.review_intel.sentiment
    report_dir = Path(config.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    lock_path = final_lock_path(report_dir)
    check_final_lock(lock_path, reuse=args.reuse_test_split)

    device = resolve_device(args.device)
    artifact = Path(config.model_path)
    checkpoints = {
        "transformer": Path(config.transformer.output_dir) / BEST_CHECKPOINT_DIR,
        "rnn": Path(config.rnn.output_dir) / BEST_CHECKPOINT_DIR,
    }

    entries: list[dict[str, Any]] = []
    if artifact.is_file():
        entries.append(
            {
                "model": "tfidf",
                "checkpoint": str(artifact),
                "weights_file": artifact.name,
                "weights_sha256": file_sha256(artifact),
                "weights_bytes": artifact.stat().st_size,
                "model_meta_sha256": None,
            }
        )
    else:
        logger.warning("model_skipped model=tfidf detail=artifact not found: %s", artifact)
    for architecture, checkpoint_dir in checkpoints.items():
        if not checkpoint_dir.is_dir():
            logger.warning(
                "model_skipped model=%s detail=checkpoint not found: %s",
                architecture,
                checkpoint_dir,
            )
            continue
        identity = describe_checkpoint(checkpoint_dir).to_dict()
        entries.append(
            {
                "model": architecture,
                "checkpoint": identity["path"],
                "weights_file": identity["weights_file"],
                "weights_sha256": identity["weights_sha256"],
                "weights_bytes": identity["weights_bytes"],
                "model_meta_sha256": identity["model_meta_sha256"],
            }
        )
    if not entries:
        raise DataError(
            "nothing to evaluate: no TF-IDF artifact and no neural checkpoint found; "
            "train a model or download its checkpoint first"
        )

    write_final_lock(lock_path, entries, limit=args.limit)
    logger.info("test_split_lock_written path=%s models=%d", lock_path, len(entries))

    reviews = _read_reviews(settings)
    written: list[str] = []
    if artifact.is_file():
        written.extend(
            run_final_baseline(config, report_dir, artifact, reviews, limit=args.limit)
        )
    for architecture in NEURAL_ARCHITECTURES:
        checkpoint_dir = checkpoints[architecture]
        if not checkpoint_dir.is_dir():
            continue
        written.extend(
            run_final_neural(
                config,
                report_dir,
                architecture,
                checkpoint_dir,
                reviews,
                device=device,
                limit=args.limit,
            )
        )

    rows, notes = collect_test_rows(report_dir)
    notes.append(
        f"Lock file: `{lock_path.name}` (written before scoring; it records the "
        "date, the limit and every checkpoint hash)."
    )
    markdown, csv_path = write_comparison(
        report_dir,
        rows,
        notes,
        stem=COMPARISON_FINAL_STEM,
        split=SPLIT_TEST,
    )
    written.extend([markdown.name, csv_path.name])
    return FinalRun(
        lock=lock_path,
        reports=tuple(written),
        rows=tuple(rows),
        notes=tuple(notes),
        markdown=markdown,
        csv=csv_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Build the comparison table, re-render final labels, or run the hold-out.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 written, 1 failed, 2 test split already used.
    """
    args = parse_args(argv)
    if args.final and args.rerender:
        logger.error("--final and --rerender are mutually exclusive")
        return 1
    if args.rerender and (
        args.limit is not None or args.device != "auto" or args.reuse_test_split
    ):
        logger.error(
            "--device/--limit/--i-know-this-reuses-the-test-split only apply with "
            "--final; --rerender only reads the JSON reports"
        )
        return 1
    if not args.final and not args.rerender and (
        args.limit is not None or args.device != "auto" or args.reuse_test_split
    ):
        logger.error(
            "--device/--limit/--i-know-this-reuses-the-test-split only apply with "
            "--final; validation mode only reads the JSON reports"
        )
        return 1

    try:
        settings: Settings = load_settings(args.config)
    except ConfigError as exc:
        logger.error("config_error path=%s detail=%s", args.config, exc)
        return 1

    try:
        if args.rerender:
            report_dir = Path(settings.review_intel.sentiment.report_dir)
            paths = rerender_final_reports(report_dir)
            print()
            print("re-rendered reports:")
            for path in paths:
                print(f"  {path}")
            return 0
        if args.final:
            final = run_final(settings, args)
            print()
            print(f"test split reports  : {', '.join(final.reports)}")
            print(f"comparison markdown : {final.markdown}")
            print(f"comparison csv      : {final.csv}")
            print(f"lock file           : {final.lock}")
            print()
            print("\n".join(render_comparison(final.rows, final.notes, split=SPLIT_TEST)))
            return 0

        report_dir = Path(settings.review_intel.sentiment.report_dir)
        rows, notes = collect_validation_rows(report_dir)
        markdown, csv_path = write_comparison(
            report_dir,
            rows,
            notes,
            stem=COMPARISON_VAL_STEM,
            split=SPLIT_VAL,
        )
        logger.info("comparison_written markdown=%s csv=%s", markdown, csv_path)
        print()
        print("\n".join(render_comparison(rows, notes, split=SPLIT_VAL)))
        return 0
    except TestSplitLockedError as exc:
        logger.error("test_split_locked detail=%s", exc)
        return 2
    except (ConfigError, DataError, ValueError, KeyError) as exc:
        logger.error("comparison_failed detail=%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

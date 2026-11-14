"""Markdown rendering for the sentiment evaluation reports.

This module is deliberately **torch-free**: it turns already-computed values into
the Markdown report, so a report can be re-rendered from its JSON without loading
a model, the test data or a GPU. Training facts (the device the checkpoint was
trained on, the training wall clock, the epochs and the selection metric) are
kept apart from evaluation facts (the device and split that produced the metrics,
and the number of rows evaluated). A checkpoint trained on CUDA and evaluated on
CPU therefore says exactly that, instead of labelling the training time with the
evaluating machine's device.
"""

from __future__ import annotations

from dataclasses import dataclass

from munasib.core.constants import SPLIT_TEST
from munasib.review_intel.sentiment.metrics import (
    ClassificationMetrics,
    ProductCalibration,
    render_calibration_table,
    render_confusion_and_scores,
    render_metrics_table,
)

#: Header shown in every neural report.
WEAK_LABEL_NOTE = (
    "Labels are **weak supervision from star ratings** (1-2 negative, 3 neutral, "
    "4-5 positive); every number below measures agreement with that proxy."
)


def report_stem(architecture: str, split: str) -> str:
    """Name the evaluation report for one architecture and split.

    Args:
        architecture: ``"transformer"`` or ``"rnn"``.
        split: ``"val"`` or ``"test"``.

    Returns:
        A file stem such as ``sentiment_transformer_val`` (``_final`` for the
        test split, matching the ``--final`` flag).
    """
    suffix = "final" if split == SPLIT_TEST else split
    return f"sentiment_{architecture}_{suffix}"


@dataclass(frozen=True)
class EpochHistoryRow:
    """One epoch's training and selection numbers, as recorded in the JSON.

    Attributes:
        epoch: One-based epoch number.
        train_loss: Mean training loss, weighted by rows.
        val_macro_f1: Macro-F1 on the selection data.
        val_accuracy: Accuracy on the selection data.
        seconds: Wall-clock seconds for the epoch.
        rows_per_second: Training throughput.
        improved: Whether the epoch set a new best selection score.
    """

    epoch: int
    train_loss: float
    val_macro_f1: float
    val_accuracy: float
    seconds: float
    rows_per_second: float
    improved: bool


@dataclass(frozen=True)
class TrainingFacts:
    """What the training run recorded, independent of where it is evaluated.

    Attributes:
        architecture: ``"transformer"`` or ``"rnn"``.
        class_weighting: ``none``, ``sqrt_inverse`` or ``balanced``.
        class_weights: Loss weight per weak label, in label order.
        parameters: Trainable parameter count.
        train_rows: Rows in the training sample.
        selection_rows: Rows used for per-epoch selection.
        val_subset: Requested selection-subset size, or ``None`` for the full
            validation split.
        best_epoch: Epoch whose checkpoint was kept.
        epochs_run: Epochs actually completed.
        selection_metric: Metric used for per-epoch selection (``"macro_f1"``).
        elapsed_seconds: Training wall clock, measured by the training run.
        device: Device the training run used (for example ``"cuda"``).
        history: Per-epoch records in order.
    """

    architecture: str
    class_weighting: str
    class_weights: tuple[float, ...]
    parameters: int
    train_rows: int
    selection_rows: int
    val_subset: int | None
    best_epoch: int
    epochs_run: int
    selection_metric: str
    elapsed_seconds: float
    device: str
    history: tuple[EpochHistoryRow, ...]


@dataclass(frozen=True)
class EvaluationFacts:
    """What the evaluation run recorded: split, device, rows and metrics.

    Attributes:
        split: ``"val"`` or ``"test"``.
        device: Device the evaluation ran on (not the training device).
        rows: Rows evaluated.
        full_split: Whether every row of the split was evaluated.
        metrics: Three-class and binary metrics read from the report.
        calibration_hard: Product calibration from hard predictions.
        calibration_soft: Product calibration from mean negative probability.
    """

    split: str
    device: str
    rows: int
    full_split: bool
    metrics: ClassificationMetrics
    calibration_hard: ProductCalibration
    calibration_soft: ProductCalibration


def _display_metric(metric: str) -> str:
    """Render the stored selection metric for humans.

    Args:
        metric: Recorded metric name, for example ``"macro_f1"``.

    Returns:
        ``"macro-F1"`` for the one known metric, otherwise the stored name.
    """
    return "macro-F1" if metric == "macro_f1" else metric


def render_report_markdown(
    *, training: TrainingFacts, evaluation: EvaluationFacts
) -> list[str]:
    """Render the Markdown report for one split.

    Args:
        training: Facts recorded by the training run.
        evaluation: Facts recorded by the evaluation run.

    Returns:
        Markdown lines. Training and evaluation devices are labelled separately,
        so a CUDA-trained checkpoint evaluated on CPU cannot claim its training
        wall clock was measured on CPU.
    """
    subset_note = (
        "full validation split"
        if training.val_subset is None
        else f"stratified subset of {training.val_subset:,} rows"
    )
    report_note = (
        f"the full {evaluation.split} split"
        if evaluation.full_split
        else "smoke slice, not the full split"
    )
    lines = [
        f"# Sentiment model report - {training.architecture} ({evaluation.split})",
        "",
        WEAK_LABEL_NOTE,
        "",
        f"- Trainable parameters: {training.parameters:,}",
        f"- Class weighting: {training.class_weighting} "
        f"(weights {', '.join(f'{weight:.4f}' for weight in training.class_weights)})",
        f"- Training rows: {training.train_rows:,}",
        f"- Per-epoch selection: {subset_note} ({training.selection_rows:,} rows), "
        f"metric {_display_metric(training.selection_metric)}, "
        f"best epoch {training.best_epoch} of {training.epochs_run}",
        f"- Training device: {training.device}",
        f"- Training wall clock: {training.elapsed_seconds:,.1f}s",
        f"- Evaluated on: {evaluation.device}",
        f"- Evaluation split: {evaluation.split}",
        f"- Evaluated rows: {evaluation.rows:,} ({report_note})",
        "",
        "## Epoch history",
        "",
        "| Epoch | Train loss | Selection macro-F1 | Accuracy | Seconds | Rows/s | Best |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for record in training.history:
        lines.append(
            f"| {record.epoch} | {record.train_loss:.4f} | {record.val_macro_f1:.4f} | "
            f"{record.val_accuracy:.4f} | {record.seconds:.1f} | "
            f"{record.rows_per_second:,.0f} | {'yes' if record.improved else ''} |"
        )
    lines.append("")
    lines.extend(render_metrics_table("Three-class metrics", evaluation.metrics))
    lines.extend(render_confusion_and_scores(evaluation.metrics))
    lines.extend(
        render_calibration_table(
            "Product calibration - hard predictions", evaluation.calibration_hard
        )
    )
    lines.extend(
        render_calibration_table(
            "Product calibration - mean negative probability", evaluation.calibration_soft
        )
    )
    return lines


__all__ = [
    "WEAK_LABEL_NOTE",
    "EpochHistoryRow",
    "EvaluationFacts",
    "TrainingFacts",
    "render_report_markdown",
    "report_stem",
]

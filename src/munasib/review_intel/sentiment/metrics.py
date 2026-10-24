"""Honest evaluation metrics for the weak-supervision sentiment models.

Accuracy is the wrong headline number for this dataset: positive reviews are
~83% of every split, so a constant "positive" prediction already scores ~0.83.
Every report produced here therefore carries the **majority-class baseline**
(accuracy *and* macro-F1, computed from the data being evaluated) next to the
model, and a **product-level calibration check** because the downstream risk
stage consumes per-product negative shares rather than per-review accuracy.

Definitions (all computed here, never assumed):

* ``accuracy`` - share of rows where ``y_pred == y_true``.
* ``precision/recall/F1`` per label - from the confusion matrix, with a zero
  denominator reported as ``0.0`` (never ``nan`` and never a fabricated 1.0).
* ``macro_f1`` - unweighted mean of the three per-label F1 scores, so a model
  that only learns the majority class cannot hide behind it.
* ``majority baseline`` - predict the most frequent weak label for every row,
  then score it with the same functions (its accuracy is exactly the largest
  class share).
* ``binary metrics`` - negative vs positive with **neutral rows excluded**, the
  event being "negative" because that is what the risk stage consumes;
  ``ROC AUC`` is reported only when probabilities are supplied.

Everything here is pure: arrays and frames in, dataclasses and files out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from munasib.core.constants import (
    SENTIMENT_LABEL_NAMES,
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABELS,
)

#: Column of the per-product calibration CSV holding the product id.
CALIBRATION_ASIN_COLUMN = "asin"
#: Columns of the per-product calibration CSV, in written order.
CALIBRATION_COLUMNS: tuple[str, ...] = (
    CALIBRATION_ASIN_COLUMN,
    "n_reviews",
    "predicted_negative_share",
    "weak_negative_share",
    "difference",
)


def _as_label_array(values: Any, name: str) -> np.ndarray:
    """Validate and convert a label sequence to an int64 numpy array.

    Args:
        values: Label codes (list, Series or array).
        name: Argument name used in the error message.

    Returns:
        ``int64`` array.

    Raises:
        ValueError: If the input holds a label outside
            :data:`munasib.core.constants.SENTIMENT_LABELS`.
    """
    array = np.asarray(values)
    if array.size == 0:
        raise ValueError(f"{name} is empty; metrics need at least one row")
    if array.dtype.kind == "f":
        if not np.all(np.equal(np.mod(array, 1), 0)):
            raise ValueError(f"{name} must contain integer label codes")
    array = array.astype("int64")
    unknown = sorted(set(array.tolist()) - set(SENTIMENT_LABELS))
    if unknown:
        raise ValueError(
            f"{name} contains unknown label code(s) {unknown}; "
            f"expected a subset of {list(SENTIMENT_LABELS)}"
        )
    return array


@dataclass(frozen=True)
class ClassMetrics:
    """Precision, recall, F1 and support for one weak label.

    Attributes:
        label: Weak label code.
        name: Human-readable label name.
        precision: ``tp / (tp + fp)``, ``0.0`` when the model never predicts it.
        recall: ``tp / (tp + fn)``, ``0.0`` when the label never occurs.
        f1: Harmonic mean of precision and recall, ``0.0`` when both are 0.
        support: Number of true rows with this label.
    """

    label: int
    name: str
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, object]:
        """Serialise the per-class figures for the JSON report.

        Returns:
            Plain-Python mapping.
        """
        return {
            "label": self.label,
            "name": self.name,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass(frozen=True)
class MajorityBaseline:
    """What "always predict the most frequent weak label" scores on this data.

    Attributes:
        label: The majority weak label code.
        name: Human-readable label name.
        share: Frequency of that label in ``y_true`` (the baseline accuracy).
        accuracy: Baseline accuracy, equal to ``share`` by construction.
        macro_f1: Baseline macro-F1, reported so the model is compared on the
            metric that a majority-only prediction cannot game.
    """

    label: int
    name: str
    share: float
    accuracy: float
    macro_f1: float

    def to_dict(self) -> dict[str, object]:
        """Serialise the baseline for the JSON report."""
        return {
            "label": self.label,
            "name": self.name,
            "share": self.share,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
        }


@dataclass(frozen=True)
class BinaryMetrics:
    """Negative-vs-positive metrics with neutral reviews excluded.

    The event being scored is **negative**, matching the per-product negative
    share the risk stage consumes.

    Attributes:
        n: Rows evaluated (neutral excluded).
        prevalence: Share of those rows that are truly negative.
        accuracy: Share predicted correctly.
        precision: Of the rows predicted negative, the share truly negative.
        recall: Of the truly negative rows, the share predicted negative.
        f1: Harmonic mean of precision and recall.
        roc_auc: Area under the ROC curve for ``P(negative)``, or ``None`` when
            the model produced no probabilities.
    """

    n: int
    prevalence: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None

    def to_dict(self) -> dict[str, object]:
        """Serialise the binary figures for the JSON report."""
        return {
            "n": self.n,
            "prevalence": self.prevalence,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


@dataclass(frozen=True)
class ClassificationMetrics:
    """The complete three-class evaluation of one model on one split.

    Attributes:
        n: Rows evaluated.
        labels: Label order used by every table (always ``SENTIMENT_LABELS``).
        accuracy: Share of rows predicted correctly.
        macro_f1: Unweighted mean of the per-label F1 scores.
        per_class: One :class:`ClassMetrics` per label, in ``labels`` order.
        confusion: ``confusion[true_index][predicted_index]`` counts.
        binary: Negative-vs-positive metrics (neutral excluded).
        majority: Baseline computed from the same ``y_true``.
    """

    n: int
    labels: tuple[int, ...]
    accuracy: float
    macro_f1: float
    per_class: tuple[ClassMetrics, ...]
    confusion: tuple[tuple[int, ...], ...]
    binary: BinaryMetrics
    majority: MajorityBaseline

    def to_dict(self) -> dict[str, object]:
        """Serialise the evaluation for the JSON report.

        Returns:
            Plain-Python mapping; every value is computed, none is a placeholder.
        """
        return {
            "n": self.n,
            "labels": [
                {"label": label, "name": SENTIMENT_LABEL_NAMES[label]} for label in self.labels
            ],
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": [item.to_dict() for item in self.per_class],
            "confusion": [list(row) for row in self.confusion],
            "binary": self.binary.to_dict(),
            "majority_baseline": self.majority.to_dict(),
        }

    def lines(self) -> list[str]:
        """Render the evaluation as aligned ``label : value`` lines.

        Returns:
            A stable sequence of lines: headline numbers, per-class table,
            confusion matrix, binary block and the majority baseline.
        """
        lines = [
            f"{'rows evaluated':<28}: {self.n:,}",
            f"{'accuracy':<28}: {self.accuracy:.4f}",
            f"{'macro-F1':<28}: {self.macro_f1:.4f}",
            f"{'majority accuracy':<28}: {self.majority.accuracy:.4f} "
            f"({SENTIMENT_LABEL_NAMES[self.majority.label]})",
            f"{'majority macro-F1':<28}: {self.majority.macro_f1:.4f}",
        ]
        for item in self.per_class:
            lines.append(
                f"  {item.name:<10} precision={item.precision:.4f} "
                f"recall={item.recall:.4f} f1={item.f1:.4f} support={item.support:,}"
            )
        header = "  " + "".join(f"{SENTIMENT_LABEL_NAMES[label]:>12}" for label in self.labels)
        lines.append(f"{'confusion (true x pred)':<28}:")
        lines.append(header)
        for label, row in zip(self.labels, self.confusion, strict=True):
            cells = "".join(f"{count:>12,}" for count in row)
            lines.append(f"  {SENTIMENT_LABEL_NAMES[label]:<10}{cells}")
        binary = self.binary
        lines.append(
            f"{'binary (neg vs pos) n':<28}: {binary.n:,} "
            f"prevalence={binary.prevalence:.4f}"
        )
        lines.append(
            f"{'binary precision/recall':<28}: {binary.precision:.4f} / {binary.recall:.4f} "
            f"f1={binary.f1:.4f}"
        )
        auc = "n/a" if binary.roc_auc is None else f"{binary.roc_auc:.4f}"
        lines.append(f"{'binary ROC AUC':<28}: {auc}")
        return lines


@dataclass(frozen=True)
class ProductCalibration:
    """How well review-level predictions reproduce per-product negative shares.

    Attributes:
        n_products: Products with at least one evaluated review.
        n_reviews: Total evaluated reviews behind the table.
        mae: Mean absolute difference between the predicted and weak negative
            shares, or ``None`` when there is nothing to compare.
        pearson_r: Pearson correlation of the two share columns, or ``None``
            when there are fewer than two products or either column is constant.
        mean_signed_difference: Mean of ``predicted - weak``; positive means the
            model over-predicts negativity.
        table: Per-product rows, in :data:`CALIBRATION_COLUMNS` order.
        note: Why a statistic is ``None``, when it is.
    """

    n_products: int
    n_reviews: int
    mae: float | None
    pearson_r: float | None
    mean_signed_difference: float | None
    table: pd.DataFrame
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialise the calibration summary for the JSON report.

        Returns:
            Plain-Python mapping without the per-product table (that is written
            as its own CSV).
        """
        return {
            "n_products": self.n_products,
            "n_reviews": self.n_reviews,
            "mae": self.mae,
            "pearson_r": self.pearson_r,
            "mean_signed_difference": self.mean_signed_difference,
            "note": self.note,
        }

    def lines(self) -> list[str]:
        """Render the calibration check as aligned ``label : value`` lines.

        Returns:
            A stable sequence of lines with ``n/a`` where a statistic could not
            be computed, plus the note explaining why.
        """
        mae = "n/a" if self.mae is None else f"{self.mae:.4f}"
        r = "n/a" if self.pearson_r is None else f"{self.pearson_r:.4f}"
        diff = "n/a" if self.mean_signed_difference is None else f"{self.mean_signed_difference:+.4f}"
        lines = [
            f"{'products calibrated':<28}: {self.n_products:,}",
            f"{'reviews behind them':<28}: {self.n_reviews:,}",
            f"{'mean absolute error':<28}: {mae}",
            f"{'pearson r':<28}: {r}",
            f"{'mean signed difference':<28}: {diff}",
        ]
        if self.note:
            lines.append(f"note: {self.note}")
        return lines


def _per_class_metrics(
    confusion: np.ndarray, labels: tuple[int, ...]
) -> tuple[ClassMetrics, ...]:
    """Derive per-label precision/recall/F1 from a confusion matrix.

    Args:
        confusion: Square count matrix, ``[true][predicted]``.
        labels: Label code for each matrix index.

    Returns:
        One :class:`ClassMetrics` per label, in ``labels`` order. A zero
        denominator yields ``0.0`` rather than ``nan``.
    """
    items: list[ClassMetrics] = []
    for index, label in enumerate(labels):
        true_positive = int(confusion[index, index])
        predicted_positive = int(confusion[:, index].sum())
        actual_positive = int(confusion[index, :].sum())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator else 0.0
        items.append(
            ClassMetrics(
                label=label,
                name=SENTIMENT_LABEL_NAMES[label],
                precision=precision,
                recall=recall,
                f1=f1,
                support=actual_positive,
            )
        )
    return tuple(items)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    """Three-class confusion matrix over an explicit label order.

    Args:
        y_true: True label codes.
        y_pred: Predicted label codes.
        labels: Label order for rows/columns.

    Returns:
        Integer matrix, ``[true][predicted]``.
    """
    return np.asarray(
        confusion_matrix(y_true, y_pred, labels=list(labels)),
        dtype="int64",
    )


def majority_baseline(y_true: Any) -> MajorityBaseline:
    """Score "always predict the most frequent weak label" on this data.

    Args:
        y_true: True label codes.

    Returns:
        The baseline label, its share, and the accuracy and macro-F1 of the
        constant prediction. Ties are broken by the smallest label code, so the
        result is deterministic.

    Raises:
        ValueError: If ``y_true`` is empty or holds an unknown label code.
    """
    array = _as_label_array(y_true, "y_true")
    counts = {label: int((array == label).sum()) for label in SENTIMENT_LABELS}
    majority_label = min(
        SENTIMENT_LABELS, key=lambda label: (-counts[label], label)
    )
    prediction = np.full(array.shape, majority_label, dtype="int64")
    confusion = _confusion(array, prediction, SENTIMENT_LABELS)
    per_class = _per_class_metrics(confusion, SENTIMENT_LABELS)
    accuracy = float(np.trace(confusion) / confusion.sum())
    return MajorityBaseline(
        label=majority_label,
        name=SENTIMENT_LABEL_NAMES[majority_label],
        share=counts[majority_label] / len(array),
        accuracy=accuracy,
        macro_f1=float(np.mean([item.f1 for item in per_class])),
    )


def binary_metrics(
    y_true: Any, y_pred: Any, prob_negative: Any | None = None
) -> BinaryMetrics:
    """Negative-vs-positive metrics with neutral rows excluded.

    Args:
        y_true: True label codes.
        y_pred: Predicted label codes.
        prob_negative: ``P(negative)`` per row, or ``None`` to skip ROC AUC.

    Returns:
        Binary metrics for the "negative" event.

    Raises:
        ValueError: If no row survives the neutral exclusion, if the inputs have
            different lengths, or if ``prob_negative`` has the wrong length.
    """
    true = _as_label_array(y_true, "y_true")
    predicted = _as_label_array(y_pred, "y_pred")
    if true.shape != predicted.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length, got {true.shape} and {predicted.shape}"
        )
    # Neutral rows are excluded on the TRUE side; a neutral *prediction* on a
    # non-neutral row is simply a (wrong) non-negative prediction.
    keep = true != SENTIMENT_LABEL_NEUTRAL
    true = true[keep]
    predicted = predicted[keep]
    if true.size == 0:
        raise ValueError("no non-neutral rows to evaluate")

    event = (true == SENTIMENT_LABEL_NEGATIVE).astype("int64")
    flagged = (predicted == SENTIMENT_LABEL_NEGATIVE).astype("int64")
    true_positive = int(((event == 1) & (flagged == 1)).sum())
    false_positive = int(((event == 0) & (flagged == 1)).sum())
    false_negative = int(((event == 1) & (flagged == 0)).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    denominator = precision + recall
    f1 = 2 * precision * recall / denominator if denominator else 0.0

    auc: float | None = None
    if prob_negative is not None:
        scores = np.asarray(prob_negative, dtype="float64")
        if scores.shape[0] != keep.shape[0]:
            raise ValueError(
                f"prob_negative has {scores.shape[0]} rows but the labels have {keep.shape[0]}"
            )
        scores = scores[keep]
        # ROC AUC is undefined when only one class is present.
        if np.unique(event).size == 2:
            auc = float(roc_auc_score(event, scores))

    return BinaryMetrics(
        n=int(true.size),
        prevalence=float(event.mean()),
        accuracy=float((event == flagged).mean()),
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=auc,
    )


def classification_metrics(
    y_true: Any, y_pred: Any, proba: Any | None = None
) -> ClassificationMetrics:
    """Evaluate three-class predictions, always alongside the majority baseline.

    Args:
        y_true: True weak label codes.
        y_pred: Predicted label codes.
        proba: Optional ``(n_rows, 3)`` probability matrix whose columns follow
            :data:`munasib.core.constants.SENTIMENT_LABELS`; column 0 is
            ``P(negative)`` and feeds the binary ROC AUC.

    Returns:
        :class:`ClassificationMetrics` with per-class scores, the confusion
        matrix, binary metrics and the majority baseline.

    Raises:
        ValueError: If the inputs are empty, mismatched in length, hold an
            unknown label, or ``proba`` has the wrong shape.
    """
    true = _as_label_array(y_true, "y_true")
    predicted = _as_label_array(y_pred, "y_pred")
    if true.shape != predicted.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length, got {true.shape} and {predicted.shape}"
        )
    confusion = _confusion(true, predicted, SENTIMENT_LABELS)
    per_class = _per_class_metrics(confusion, SENTIMENT_LABELS)
    accuracy = float(np.trace(confusion) / confusion.sum())

    prob_negative: np.ndarray | None = None
    if proba is not None:
        matrix = np.asarray(proba, dtype="float64")
        if matrix.ndim != 2 or matrix.shape != (true.size, len(SENTIMENT_LABELS)):
            raise ValueError(
                f"proba must be ({true.size}, {len(SENTIMENT_LABELS)}), got {matrix.shape}"
            )
        prob_negative = matrix[:, SENTIMENT_LABELS.index(SENTIMENT_LABEL_NEGATIVE)]

    return ClassificationMetrics(
        n=int(true.size),
        labels=SENTIMENT_LABELS,
        accuracy=accuracy,
        macro_f1=float(np.mean([item.f1 for item in per_class])),
        per_class=per_class,
        confusion=tuple(tuple(int(count) for count in row) for row in confusion.tolist()),
        binary=binary_metrics(true, predicted, prob_negative),
        majority=majority_baseline(true),
    )


def compute_product_calibration(
    asins: Any, y_true: Any, y_pred: Any
) -> ProductCalibration:
    """Compare predicted and weak-label negative shares product by product.

    For every product the share of its reviews predicted negative is compared
    with the share whose weak label is negative. This is the quantity the risk
    stage consumes: review-level F1 can look good while a product's negative
    share is systematically over- or under-stated, and this check exposes that.

    Args:
        asins: Product id per evaluated review.
        y_true: True weak label codes.
        y_pred: Predicted label codes.

    Returns:
        Per-product table plus mean absolute error, Pearson correlation and the
        signed mean difference (``predicted - weak``).

    Raises:
        ValueError: If the inputs are empty or have different lengths.
    """
    ids = pd.Series(np.asarray(asins), name=CALIBRATION_ASIN_COLUMN)
    true = pd.Series(_as_label_array(y_true, "y_true"), name="y_true")
    predicted = pd.Series(_as_label_array(y_pred, "y_pred"), name="y_pred")
    if not (len(ids) == len(true) == len(predicted)):
        raise ValueError(
            "asins, y_true and y_pred must have the same length, got "
            f"{len(ids)}, {len(true)}, {len(predicted)}"
        )
    frame = pd.DataFrame(
        {
            CALIBRATION_ASIN_COLUMN: ids,
            "true_negative": (true == SENTIMENT_LABEL_NEGATIVE).astype("int64"),
            "predicted_negative": (predicted == SENTIMENT_LABEL_NEGATIVE).astype("int64"),
        }
    )
    grouped = frame.groupby(CALIBRATION_ASIN_COLUMN, sort=True)
    table = grouped.agg(
        n_reviews=("true_negative", "size"),
        weak_negative_share=("true_negative", "mean"),
        predicted_negative_share=("predicted_negative", "mean"),
    ).reset_index()
    table["difference"] = table["predicted_negative_share"] - table["weak_negative_share"]
    table = table.loc[:, list(CALIBRATION_COLUMNS)].reset_index(drop=True)

    n_products = len(table)
    n_reviews = int(table["n_reviews"].sum())
    if n_products == 0:
        return ProductCalibration(
            n_products=0,
            n_reviews=0,
            mae=None,
            pearson_r=None,
            mean_signed_difference=None,
            table=table,
            note="no products to calibrate",
        )

    difference = table["difference"].to_numpy(dtype="float64")
    predicted_share = table["predicted_negative_share"].to_numpy(dtype="float64")
    weak_share = table["weak_negative_share"].to_numpy(dtype="float64")
    mae = float(np.mean(np.abs(difference)))
    mean_signed = float(np.mean(difference))

    corr: float | None
    note: str | None = None
    if n_products < 2:
        corr, note = None, "a correlation needs at least two products"
    elif np.std(predicted_share) == 0.0 or np.std(weak_share) == 0.0:
        corr, note = None, "one of the share columns is constant, so r is undefined"
    else:
        corr = float(np.corrcoef(predicted_share, weak_share)[0, 1])

    return ProductCalibration(
        n_products=n_products,
        n_reviews=n_reviews,
        mae=mae,
        pearson_r=corr,
        mean_signed_difference=mean_signed,
        table=table,
        note=note,
    )


def write_calibration_csv(path: Path, calibration: ProductCalibration) -> Path:
    """Write the per-product calibration table as CSV.

    Args:
        path: Destination file (parent directories are created).
        calibration: Result of :func:`compute_product_calibration`.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    calibration.table.to_csv(path, index=False, float_format="%.6f")
    return path


def render_metrics_table(title: str, metrics: ClassificationMetrics) -> list[str]:
    """Render one model evaluation as Markdown.

    Args:
        title: Section heading text (rendered as a level-3 heading).
        metrics: Computed evaluation for this model and split.

    Returns:
        Markdown lines, all values derived from ``metrics``.
    """
    lines = [
        f"### {title}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows evaluated | {metrics.n:,} |",
        f"| Accuracy | {metrics.accuracy:.4f} |",
        f"| Macro-F1 | {metrics.macro_f1:.4f} |",
        f"| Majority-class accuracy | {metrics.majority.accuracy:.4f} |",
        f"| Majority-class macro-F1 | {metrics.majority.macro_f1:.4f} |",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in metrics.per_class:
        lines.append(
            f"| {item.name} | {item.precision:.4f} | {item.recall:.4f} | "
            f"{item.f1:.4f} | {item.support:,} |"
        )
    header = "| True \\ Predicted |" + "".join(
        f" {SENTIMENT_LABEL_NAMES[label]} |" for label in metrics.labels
    )
    separator = "|---|" + "---:|" * len(metrics.labels)
    lines.extend(["", header, separator])
    for label, row in zip(metrics.labels, metrics.confusion, strict=True):
        cells = "".join(f" {count:,} |" for count in row)
        lines.append(f"| {SENTIMENT_LABEL_NAMES[label]} |{cells}")
    binary = metrics.binary
    auc = "n/a" if binary.roc_auc is None else f"{binary.roc_auc:.4f}"
    lines.extend(
        [
            "",
            f"Binary (negative vs positive, neutral excluded): n={binary.n:,}, "
            f"prevalence={binary.prevalence:.4f}, accuracy={binary.accuracy:.4f}, "
            f"precision={binary.precision:.4f}, recall={binary.recall:.4f}, "
            f"F1={binary.f1:.4f}, ROC AUC={auc}.",
            "",
        ]
    )
    return lines


def render_calibration_table(title: str, calibration: ProductCalibration) -> list[str]:
    """Render a product-level calibration summary as Markdown.

    Args:
        title: Section heading text (rendered as a level-3 heading).
        calibration: Computed calibration result.

    Returns:
        Markdown lines with the summary statistics (the per-product rows live in
        the CSV, not here).
    """
    def _fmt(value: float | None, spec: str) -> str:
        return "n/a" if value is None else format(value, spec)

    lines = [
        f"### {title}",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Products | {calibration.n_products:,} |",
        f"| Reviews | {calibration.n_reviews:,} |",
        f"| Mean absolute error | {_fmt(calibration.mae, '.4f')} |",
        f"| Pearson r | {_fmt(calibration.pearson_r, '.4f')} |",
        f"| Mean signed difference (predicted - weak) | "
        f"{_fmt(calibration.mean_signed_difference, '+.4f')} |",
    ]
    if calibration.note:
        lines.extend(["", f"{calibration.note}."])
    lines.append("")
    return lines


def render_confusion_and_scores(metrics: ClassificationMetrics) -> list[str]:
    """Render a compact fenced block of the headline numbers.

    Args:
        metrics: Computed evaluation for one model and split.

    Returns:
        Markdown lines wrapping :meth:`ClassificationMetrics.lines` in a code
        fence, so the printed and written forms agree exactly.
    """
    return ["```", *metrics.lines(), "```", ""]


def write_evaluation_report(
    report_dir: Path,
    stem: str,
    payload: dict[str, Any],
    markdown_lines: list[str],
) -> tuple[Path, Path]:
    """Write the JSON payload and the Markdown report for one run.

    Args:
        report_dir: Destination directory (created if missing).
        stem: File stem, e.g. ``sentiment_baseline_val``.
        payload: Computed values only (metrics, counts, config inputs).
        markdown_lines: Pre-rendered Markdown body.

    Returns:
        ``(json_path, markdown_path)`` in that order.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path = report_dir / f"{stem}.md"
    markdown_path.write_text("\n".join(markdown_lines).rstrip() + "\n", encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "CALIBRATION_ASIN_COLUMN",
    "CALIBRATION_COLUMNS",
    "BinaryMetrics",
    "ClassMetrics",
    "ClassificationMetrics",
    "MajorityBaseline",
    "ProductCalibration",
    "binary_metrics",
    "classification_metrics",
    "compute_product_calibration",
    "majority_baseline",
    "render_calibration_table",
    "render_confusion_and_scores",
    "render_metrics_table",
    "write_calibration_csv",
    "write_evaluation_report",
]

"""Tests for the sentiment evaluation metrics.

The three-class figures are checked against a confusion matrix worked out by
hand, so the definitions (not just scikit-learn's output) are pinned: a zero
denominator is ``0.0``, macro-F1 is the unweighted mean of the per-class F1
scores, and the majority baseline is what a constant prediction scores.

The product-level calibration is checked on toy frames where the predicted and
weak negative shares are known exactly, including the cases where Pearson ``r``
is undefined (fewer than two products, or a constant column).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from munasib.core.constants import (
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABEL_POSITIVE,
)
from munasib.review_intel.sentiment.metrics import (
    CALIBRATION_COLUMNS,
    binary_metrics,
    classification_metrics,
    compute_product_calibration,
    majority_baseline,
    render_calibration_table,
    render_metrics_table,
    write_calibration_csv,
    write_evaluation_report,
)

#: Hand-built case: 3 negatives, 2 neutrals, 4 positives (9 rows).
Y_TRUE = [0, 0, 0, 1, 1, 2, 2, 2, 2]
#: Predictions giving the confusion matrix ``[[2,1,0],[1,1,0],[1,1,2]]``.
Y_PRED = [0, 0, 1, 1, 0, 2, 2, 0, 1]


class TestThreeClassMetrics:
    """Accuracy, per-class scores and macro-F1 on the hand-worked case."""

    def test_confusion_matrix_matches_the_hand_count(self) -> None:
        """The matrix is ``[true][predicted]`` and sums to the row count."""
        metrics = classification_metrics(Y_TRUE, Y_PRED)
        assert metrics.confusion == ((2, 1, 0), (1, 1, 0), (1, 1, 2))
        assert sum(sum(row) for row in metrics.confusion) == len(Y_TRUE)

    def test_accuracy_matches_the_hand_count(self) -> None:
        """5 of 9 rows are correct."""
        assert classification_metrics(Y_TRUE, Y_PRED).accuracy == pytest.approx(5 / 9)

    def test_per_class_scores_match_the_hand_count(self) -> None:
        """Precision/recall/F1 are the textbook formulas per class."""
        metrics = classification_metrics(Y_TRUE, Y_PRED)
        by_label = {item.label: item for item in metrics.per_class}
        negative = by_label[SENTIMENT_LABEL_NEGATIVE]
        neutral = by_label[SENTIMENT_LABEL_NEUTRAL]
        positive = by_label[SENTIMENT_LABEL_POSITIVE]

        assert (negative.precision, negative.recall) == (pytest.approx(0.5), pytest.approx(2 / 3))
        assert negative.f1 == pytest.approx(4 / 7)
        assert negative.support == 3

        assert (neutral.precision, neutral.recall) == (
            pytest.approx(1 / 3),
            pytest.approx(0.5),
        )
        assert neutral.f1 == pytest.approx(0.4)
        assert neutral.support == 2

        assert (positive.precision, positive.recall) == (pytest.approx(1.0), pytest.approx(0.5))
        assert positive.f1 == pytest.approx(2 / 3)
        assert positive.support == 4

    def test_macro_f1_is_the_unweighted_mean(self) -> None:
        """Macro-F1 averages the three per-class F1 scores with equal weight."""
        metrics = classification_metrics(Y_TRUE, Y_PRED)
        expected = (4 / 7 + 0.4 + 2 / 3) / 3
        assert metrics.macro_f1 == pytest.approx(expected)
        assert metrics.macro_f1 == pytest.approx(
            float(np.mean([item.f1 for item in metrics.per_class]))
        )

    def test_class_names_are_carried(self) -> None:
        """Each per-class row names its weak label for the report."""
        metrics = classification_metrics(Y_TRUE, Y_PRED)
        assert [item.name for item in metrics.per_class] == ["negative", "neutral", "positive"]


class TestMajorityBaseline:
    """The constant-prediction baseline is computed from the data."""

    def test_accuracy_equals_the_largest_class_share(self) -> None:
        """Predicting the majority label scores exactly its frequency."""
        baseline = majority_baseline(Y_TRUE)
        assert baseline.label == SENTIMENT_LABEL_POSITIVE
        assert baseline.share == pytest.approx(4 / 9)
        assert baseline.accuracy == pytest.approx(4 / 9)

    def test_macro_f1_is_computed_not_assumed(self) -> None:
        """Only the majority class can score; the other two F1 are zero."""
        baseline = majority_baseline(Y_TRUE)
        assert baseline.macro_f1 == pytest.approx((8 / 13) / 3)

    def test_ties_break_on_the_smallest_label(self) -> None:
        """A tie must not depend on row order (or on dict insertion order)."""
        baseline = majority_baseline([0, 2])
        assert baseline.label == SENTIMENT_LABEL_NEGATIVE
        assert baseline.share == pytest.approx(0.5)

    def test_is_attached_to_every_evaluation(self) -> None:
        """The model's metrics always carry the baseline next to them."""
        metrics = classification_metrics(Y_TRUE, Y_PRED)
        assert metrics.majority.accuracy == pytest.approx(4 / 9)


class TestBinaryMetrics:
    """Negative-vs-positive metrics with neutral reviews excluded."""

    def test_neutral_rows_are_excluded(self) -> None:
        """7 of the 9 rows are non-neutral; the precision follows the hand count."""
        binary = binary_metrics(Y_TRUE, Y_PRED)
        assert binary.n == 7
        assert binary.prevalence == pytest.approx(3 / 7)
        assert binary.precision == pytest.approx(2 / 3)
        assert binary.recall == pytest.approx(2 / 3)
        assert binary.f1 == pytest.approx(2 / 3)
        assert binary.accuracy == pytest.approx(5 / 7)

    def test_roc_auc_uses_the_negative_probability(self) -> None:
        """One negative ranks below a positive, so AUC is 11 of 12 pairs."""
        scores = [0.9, 0.4, 0.7, 0.5, 0.35, 0.1, 0.2, 0.5, 0.25]
        binary = binary_metrics(Y_TRUE, Y_PRED, scores)
        assert binary.roc_auc == pytest.approx(11 / 12)

    def test_roc_auc_is_absent_without_probabilities(self) -> None:
        """No probabilities means no AUC rather than a fabricated number."""
        assert binary_metrics(Y_TRUE, Y_PRED).roc_auc is None

    def test_single_class_has_no_auc(self) -> None:
        """ROC AUC is undefined when only positives are left after filtering."""
        binary = binary_metrics([2, 2], [2, 2], [0.1, 0.2])
        assert binary.roc_auc is None

    def test_all_neutral_is_rejected(self) -> None:
        """Nothing to compare means a clear error, not a zeroed report."""
        with pytest.raises(ValueError, match="no non-neutral rows"):
            binary_metrics([1, 1], [1, 0])


class TestValidation:
    """Bad inputs fail loudly instead of producing plausible-looking numbers."""

    def test_unknown_label_is_rejected(self) -> None:
        """A label outside 0/1/2 cannot be scored."""
        with pytest.raises(ValueError, match="unknown label"):
            classification_metrics([0, 3], [0, 0])

    def test_empty_input_is_rejected(self) -> None:
        """An empty evaluation has no metrics."""
        with pytest.raises(ValueError, match="empty"):
            classification_metrics([], [])

    def test_length_mismatch_is_rejected(self) -> None:
        """Predictions must align with the truth row by row."""
        with pytest.raises(ValueError, match="same length"):
            classification_metrics([0, 1], [0])

    def test_wrong_probability_shape_is_rejected(self) -> None:
        """A probability matrix must have one column per label."""
        with pytest.raises(ValueError, match="proba must be"):
            classification_metrics(Y_TRUE, Y_PRED, np.zeros((len(Y_TRUE), 2)))


class TestProductCalibration:
    """Per-product negative shares are compared with the weak-label shares."""

    def test_perfect_agreement(self) -> None:
        """Shares that match product by product give MAE 0 and r 1."""
        asins = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
        truth = [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2]
        predicted = list(truth)
        calibration = compute_product_calibration(asins, truth, predicted)
        assert calibration.n_products == 3
        assert calibration.n_reviews == 12
        assert calibration.mae == pytest.approx(0.0)
        assert calibration.pearson_r == pytest.approx(1.0)
        assert calibration.mean_signed_difference == pytest.approx(0.0)

    def test_over_prediction_has_a_positive_signed_difference(self) -> None:
        """Model calls more reviews negative than the weak label does."""
        asins = ["A"] * 4
        truth = [2, 2, 2, 2]
        predicted = [0, 0, 2, 2]
        calibration = compute_product_calibration(asins, truth, predicted)
        assert calibration.mae == pytest.approx(0.5)
        assert calibration.mean_signed_difference == pytest.approx(0.5)
        assert calibration.note == "a correlation needs at least two products"

    def test_constant_share_has_no_correlation(self) -> None:
        """A constant column makes ``r`` undefined; the reason is reported."""
        asins = ["A", "A", "B", "B"]
        truth = [0, 0, 2, 2]
        predicted = [0, 2, 0, 2]
        calibration = compute_product_calibration(asins, truth, predicted)
        assert calibration.mae == pytest.approx(0.5)
        assert calibration.pearson_r is None
        assert calibration.note is not None

    def test_table_is_ordered_and_complete(self) -> None:
        """One row per product, ascending, with the documented columns."""
        asins = ["B", "A", "B", "A"]
        truth = [0, 0, 2, 2]
        predicted = [0, 0, 2, 2]
        calibration = compute_product_calibration(asins, truth, predicted)
        assert list(calibration.table.columns) == list(CALIBRATION_COLUMNS)
        table = calibration.table.set_index("asin")
        assert list(table.index) == ["A", "B"]
        assert table.loc["A", "predicted_negative_share"] == pytest.approx(0.5)
        assert table.loc["B", "n_reviews"] == 2

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        """The per-product CSV holds exactly the table, readable by pandas."""
        asins = ["A", "A", "B", "B"]
        truth = [0, 0, 2, 2]
        predicted = [0, 0, 2, 0]
        calibration = compute_product_calibration(asins, truth, predicted)
        path = write_calibration_csv(tmp_path / "nested" / "calibration.csv", calibration)
        assert path.exists()
        loaded = pd.read_csv(path)
        assert list(loaded.columns) == list(CALIBRATION_COLUMNS)
        assert len(loaded) == calibration.n_products


class TestRendering:
    """The report writer emits computed values only."""

    def test_metrics_table_contains_the_numbers(self) -> None:
        """Every headline figure appears in the Markdown table."""
        metrics = classification_metrics(Y_TRUE, Y_PRED)
        rendered = "\n".join(render_metrics_table("Baseline", metrics))
        assert "### Baseline" in rendered
        assert f"{metrics.accuracy:.4f}" in rendered
        assert f"{metrics.macro_f1:.4f}" in rendered
        assert f"{metrics.majority.accuracy:.4f}" in rendered
        assert f"{metrics.binary.precision:.4f}" in rendered

    def test_calibration_table_renders_na(self) -> None:
        """A statistic that could not be computed is shown as ``n/a``."""
        calibration = compute_product_calibration(["A"], [0], [0])
        rendered = "\n".join(render_calibration_table("Calibration", calibration))
        assert "n/a" in rendered
        assert calibration.note is not None
        assert calibration.note in rendered

    def test_report_writer_writes_json_and_markdown(self, tmp_path: Path) -> None:
        """Both files are written, the JSON parses and the Markdown ends cleanly."""
        payload = {"value": 0.5, "rows": 9}
        json_path, markdown_path = write_evaluation_report(
            tmp_path / "evaluation", "stem", payload, ["# Title", "", "body"]
        )
        assert json.loads(json_path.read_text(encoding="utf-8")) == payload
        text = markdown_path.read_text(encoding="utf-8")
        assert text.startswith("# Title")
        assert text.endswith("\n")
        assert not text.endswith("\n\n")

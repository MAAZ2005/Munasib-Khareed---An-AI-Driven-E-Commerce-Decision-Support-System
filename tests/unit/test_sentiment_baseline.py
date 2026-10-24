"""Tests for the TF-IDF + logistic-regression baseline.

Everything runs on a handful of hand-written reviews, so the suite is hermetic
(no parquet, no network, no GPU) and fast. The properties under test are the ones
that would silently invalidate a reported score: the vocabulary must be fitted on
the training sample only, the train-only guard must hold, and the saved artifact
must reload into a model that predicts the same thing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from munasib.core.config import LogRegConfig, SentimentConfig, TfidfConfig
from munasib.core.constants import (
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABEL_POSITIVE,
    SENTIMENT_LABELS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from munasib.core.exceptions import DataError
from munasib.review_intel.sentiment.baseline import (
    CLASS_WEIGHT_CHOICES,
    SentimentBaseline,
    class_weight_name,
    evaluate_baseline,
    load_baseline,
    resolve_class_weight,
    save_baseline,
    train_baseline,
)
from munasib.review_intel.sentiment.baseline import (
    build_classifier as make_classifier,
)
from munasib.review_intel.sentiment.baseline import (
    build_vectorizer as make_vectorizer,
)

#: Six clearly separable reviews, two per class, using distinct vocabulary.
CORPUS: tuple[tuple[str, int], ...] = (
    ("broken terrible defective", SENTIMENT_LABEL_NEGATIVE),
    ("terrible defective broken", SENTIMENT_LABEL_NEGATIVE),
    ("mediocre average unremarkable", SENTIMENT_LABEL_NEUTRAL),
    ("average unremarkable mediocre", SENTIMENT_LABEL_NEUTRAL),
    ("excellent wonderful superb", SENTIMENT_LABEL_POSITIVE),
    ("superb excellent wonderful", SENTIMENT_LABEL_POSITIVE),
)


def _config(**overrides: object) -> SentimentConfig:
    """Build a config sized for a tiny corpus.

    Args:
        **overrides: Fields to override on :class:`SentimentConfig`.

    Returns:
        Settings with ``min_df=1`` (the default ``min_df=3`` would empty the
        vocabulary of a six-document corpus) and a small ``max_features``.
    """
    defaults: dict[str, object] = {
        "tfidf": TfidfConfig(ngram_range=(1, 2), min_df=1, max_features=1_000),
        "logreg": LogRegConfig(C=1.0, max_iter=200),
        "max_reviews_per_product": 50,
        "max_train_rows": 100_000,
        "min_chars": 3,
        "seed": 42,
    }
    defaults.update(overrides)
    return SentimentConfig(**defaults)


def _frame(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    """Assemble ``(review_id, asin, text, label)`` tuples into a review frame.

    Args:
        rows: Review id, product id, text and weak label per row.

    Returns:
        Frame with the columns the baseline reads, all in the ``train`` split.
    """
    return pd.DataFrame(
        {
            "review_id": [row[0] for row in rows],
            "asin": [row[1] for row in rows],
            "split": [SPLIT_TRAIN] * len(rows),
            "review_text_clean": [row[2] for row in rows],
            "sentiment_label_weak": pd.Series(
                [row[3] for row in rows], dtype="int8"
            ),
        }
    )


def _training_frame() -> pd.DataFrame:
    """Build the tiny training frame from :data:`CORPUS`.

    Returns:
        One product holding all six reviews.
    """
    return _frame(
        [(f"r{index}", "A", text, label) for index, (text, label) in enumerate(CORPUS)]
    )


class TestTinyCorpus:
    """The baseline fits, predicts and scores a tiny corpus."""

    def test_trains_and_predicts_every_class(self) -> None:
        """A separable corpus is classified correctly and covers all labels."""
        baseline = train_baseline(_training_frame(), _config(), None)
        predictions = baseline.predict([text for text, _ in CORPUS])
        assert predictions.tolist() == [label for _, label in CORPUS]
        assert set(predictions.tolist()) == set(SENTIMENT_LABELS)

    def test_probabilities_are_aligned_with_the_label_order(self) -> None:
        """``predict_proba`` returns three columns that sum to 1, in label order."""
        baseline = train_baseline(_training_frame(), _config(), None)
        probabilities = baseline.predict_proba([text for text, _ in CORPUS])
        assert probabilities.shape == (len(CORPUS), len(SENTIMENT_LABELS))
        assert baseline.labels == SENTIMENT_LABELS
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        # The predicted class is the argmax of the aligned columns.
        argmax_labels = [SENTIMENT_LABELS[index] for index in probabilities.argmax(axis=1)]
        assert argmax_labels == baseline.predict([text for text, _ in CORPUS]).tolist()

    def test_metadata_is_recorded(self) -> None:
        """The artifact metadata describes the fit, without wall-clock values."""
        baseline = train_baseline(_training_frame(), _config(), "balanced")
        metadata = baseline.to_metadata()
        assert metadata["train_rows"] == len(CORPUS)
        assert metadata["train_products"] == 1
        assert metadata["vocabulary_size"] == baseline.vocabulary_size
        assert metadata["class_weight"] == "balanced"
        assert metadata["labels"] == list(SENTIMENT_LABELS)
        assert "trained_at" not in metadata


class TestVocabularyLeakage:
    """The vocabulary comes from the training sample, never from evaluation."""

    def test_unseen_terms_are_not_in_the_vocabulary(self) -> None:
        """A word that only appears in the evaluation text is unknown to the model."""
        baseline = train_baseline(_training_frame(), _config(), None)
        vocabulary = baseline.vectorizer.vocabulary_
        assert "wonderful" in vocabulary
        assert "unseenholdoutterm" not in vocabulary

    def test_unseen_only_text_scores_without_error(self) -> None:
        """Scoring text made only of unseen terms yields a valid, non-crashing row."""
        baseline = train_baseline(_training_frame(), _config(), None)
        probabilities = baseline.predict_proba(["unseenholdoutterm zzzqqq"])
        assert probabilities.shape == (1, len(SENTIMENT_LABELS))
        assert np.isclose(probabilities.sum(), 1.0)

    def test_fitting_does_not_change_when_evaluation_rows_are_added(self) -> None:
        """Adding evaluation rows to the *evaluation* frame does not change the fit."""
        train = _training_frame()
        baseline = train_baseline(train, _config(), None)

        evaluation = pd.DataFrame(
            {
                "review_id": ["e1"],
                "asin": ["B"],
                "split": [SPLIT_VAL],
                "review_text_clean": ["superbunseen"],
                "sentiment_label_weak": pd.Series([SENTIMENT_LABEL_POSITIVE], dtype="int8"),
            }
        )
        unseen_before = "superbunseen" in baseline.vectorizer.vocabulary_
        assert unseen_before is False
        # The evaluation frame is only transformed, so the vocabulary is untouched.
        baseline.predict(evaluation["review_text_clean"].tolist())
        assert "superbunseen" not in baseline.vectorizer.vocabulary_


class TestArtifactRoundTrip:
    """The joblib artifact reloads into an identical model."""

    def test_round_trip_reproduces_predictions(self, tmp_path: Path) -> None:
        """Reloading a saved model gives the same labels and probabilities."""
        baseline = train_baseline(_training_frame(), _config(), None)
        path = save_baseline(baseline, tmp_path / "nested" / "model.joblib")
        assert path.exists()

        reloaded = load_baseline(path)
        assert isinstance(reloaded, SentimentBaseline)
        texts = [text for text, _ in CORPUS]
        assert reloaded.predict(texts).tolist() == baseline.predict(texts).tolist()
        assert np.allclose(reloaded.predict_proba(texts), baseline.predict_proba(texts))
        assert reloaded.to_metadata() == baseline.to_metadata()

    def test_loading_a_missing_file_is_reported(self, tmp_path: Path) -> None:
        """A missing artifact is a clear error, not a stack trace from joblib."""
        with pytest.raises(DataError, match="artifact not found"):
            load_baseline(tmp_path / "absent.joblib")

    def test_loading_something_else_is_reported(self, tmp_path: Path) -> None:
        """A joblib file holding the wrong object is rejected by type."""
        import joblib  # type: ignore[import-untyped]

        path = tmp_path / "not_a_model.joblib"
        joblib.dump({"model": "nope"}, path)
        with pytest.raises(DataError, match="does not contain a SentimentBaseline"):
            load_baseline(path)


class TestClassWeight:
    """Both class_weight variants are supported and labelled honestly."""

    def test_choices_are_balanced_and_none(self) -> None:
        """The comparison the script reports is exactly these two variants."""
        assert CLASS_WEIGHT_CHOICES == ("balanced", "none")

    def test_resolve_maps_none_to_sklearn_none(self) -> None:
        """``"none"`` means scikit-learn's ``class_weight=None``."""
        assert resolve_class_weight("none") is None
        assert resolve_class_weight("balanced") == "balanced"

    def test_unknown_choice_is_rejected(self) -> None:
        """A typo cannot silently fall back to an unweighted model."""
        with pytest.raises(DataError, match="class_weight must be one of"):
            resolve_class_weight("balanced-ish")

    def test_naming_is_symmetric(self) -> None:
        """``None`` is reported as ``"none"``, never as ``"None"``."""
        assert class_weight_name(None) == "none"
        assert class_weight_name("balanced") == "balanced"

    @pytest.mark.parametrize("choice", CLASS_WEIGHT_CHOICES)
    def test_both_variants_fit(self, choice: str) -> None:
        """Either variant produces a usable model with the expected weight."""
        baseline = train_baseline(_training_frame(), _config(), resolve_class_weight(choice))
        assert baseline.class_weight_name == choice
        probabilities = baseline.predict_proba([text for text, _ in CORPUS])
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestGuards:
    """Invalid inputs fail loudly instead of fitting something meaningless."""

    def test_non_train_rows_are_rejected(self) -> None:
        """A val row inside the training frame trips the leakage guard."""
        frame = pd.concat(
            [
                _training_frame(),
                pd.DataFrame(
                    {
                        "review_id": ["v1"],
                        "asin": ["B"],
                        "split": [SPLIT_VAL],
                        "review_text_clean": ["terrible broken"],
                        "sentiment_label_weak": pd.Series([SENTIMENT_LABEL_NEGATIVE], dtype="int8"),
                    }
                ),
            ],
            ignore_index=True,
        )
        with pytest.raises(DataError, match="leakage guard"):
            train_baseline(frame, _config(), None)

    def test_empty_frame_is_rejected(self) -> None:
        """Nothing to fit is an error, not an empty model."""
        empty = _training_frame().iloc[0:0]
        with pytest.raises(DataError, match="empty"):
            train_baseline(empty, _config(), None)

    def test_missing_label_is_rejected(self) -> None:
        """A sample without the neutral class cannot predict all three labels."""
        frame = _training_frame()
        frame = frame.loc[frame["sentiment_label_weak"] != SENTIMENT_LABEL_NEUTRAL]
        with pytest.raises(DataError, match="missing label"):
            train_baseline(frame, _config(), None)

    def test_short_text_only_frame_is_rejected(self) -> None:
        """Texts below ``min_chars`` leave nothing to fit."""
        frame = _training_frame().assign(review_text_clean="ab")
        with pytest.raises(DataError, match="no row with enough text"):
            train_baseline(frame, _config(), None)


class TestEvaluateBaseline:
    """Scoring an evaluation split produces metrics and a calibration table."""

    def test_metrics_and_calibration_are_returned(self) -> None:
        """Evaluation covers every product in the frame and returns probabilities."""
        baseline = train_baseline(_training_frame(), _config(), None)
        evaluation = pd.DataFrame(
            {
                "review_id": ["e1", "e2", "e3", "e4"],
                "asin": ["A", "A", "B", "B"],
                "split": [SPLIT_VAL] * 4,
                "review_text_clean": [
                    "terrible broken",
                    "excellent wonderful",
                    "mediocre average",
                    "defective terrible",
                ],
                "sentiment_label_weak": pd.Series(
                    [
                        SENTIMENT_LABEL_NEGATIVE,
                        SENTIMENT_LABEL_POSITIVE,
                        SENTIMENT_LABEL_NEUTRAL,
                        SENTIMENT_LABEL_NEGATIVE,
                    ],
                    dtype="int8",
                ),
            }
        )
        metrics, calibration, probabilities = evaluate_baseline(
            baseline, evaluation, _config()
        )
        assert metrics.n == 4
        assert probabilities.shape == (4, len(SENTIMENT_LABELS))
        assert calibration.n_products == 2
        assert calibration.n_reviews == 4
        assert calibration.mae is not None

    def test_empty_evaluation_frame_is_rejected(self) -> None:
        """Scoring nothing is an error."""
        baseline = train_baseline(_training_frame(), _config(), None)
        with pytest.raises(DataError, match="empty"):
            evaluate_baseline(baseline, _training_frame().iloc[0:0], _config())

    def test_missing_product_column_is_rejected(self) -> None:
        """Calibration needs the product id, so a frame without it is refused."""
        baseline = train_baseline(_training_frame(), _config(), None)
        frame = _training_frame().drop(columns=["asin"])
        with pytest.raises(DataError, match="asin"):
            evaluate_baseline(baseline, frame, _config())


class TestBuilders:
    """The scikit-learn objects come straight from the configured values."""

    def test_vectorizer_reflects_the_config(self) -> None:
        """TF-IDF settings are not hardcoded in the builder."""
        vectorizer = make_vectorizer(_config())
        assert vectorizer.ngram_range == (1, 2)
        assert vectorizer.min_df == 1
        assert vectorizer.max_features == 1_000

    def test_classifier_reflects_the_config(self) -> None:
        """Regularisation, iteration cap, seed and weight come from the config."""
        classifier = make_classifier(_config(), "balanced")
        assert classifier.C == 1.0
        assert classifier.max_iter == 200
        assert classifier.class_weight == "balanced"
        assert classifier.random_state == 42

    def test_test_split_name_is_available_for_callers(self) -> None:
        """The module exposes the split constant callers should use, not a literal."""
        assert SPLIT_TEST != SPLIT_TRAIN
        assert SPLIT_VAL != SPLIT_TEST

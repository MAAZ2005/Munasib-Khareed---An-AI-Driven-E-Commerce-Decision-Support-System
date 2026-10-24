"""TF-IDF + logistic-regression baseline for weak-supervision sentiment.

This is the reference model the transformer stage has to beat, and it is
deliberately boring: word unigrams + bigrams, sublinear TF, a capped vocabulary,
multinomial logistic regression. Two properties matter more than the score:

* **The vocabulary is fitted on the training sample only.** The vectoriser is
  ``fit`` inside :func:`train_baseline` on the sample the caller passes, and the
  evaluation frame is only ever ``transform``-ed. A term that appears solely in
  ``val``/``test`` therefore cannot influence the model - an easy leak to
  introduce when a vectoriser is fitted on a whole corpus for convenience.
* **The artifact is self-contained and reloadable.** The vectoriser and the
  classifier travel together through ``joblib``, alongside the column names and
  label order they were trained with, so :func:`load_baseline` can predict
  without the caller having to remember any of it.

``class_weight`` is compared instead of assumed: ``"balanced"`` re-weights the
loss by inverse class frequency (negatives ~12%, neutrals ~5%), ``"none"`` uses
the raw frequencies. Both are reported because the better choice depends on
whether the downstream consumer wants calibrated shares or raw accuracy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from munasib.core.config import SentimentConfig, TfidfConfig
from munasib.core.constants import SENTIMENT_LABELS, SPLIT_TRAIN
from munasib.core.exceptions import DataError
from munasib.review_intel.sentiment.metrics import (
    ClassificationMetrics,
    ProductCalibration,
    classification_metrics,
    compute_product_calibration,
)
from munasib.review_intel.sentiment.sampling import (
    PRODUCT_COLUMN,
    assert_split_only,
    text_mask,
)

#: Accepted ``class_weight`` names. ``"none"`` means "use the raw class
#: frequencies", i.e. scikit-learn's ``class_weight=None``.
CLASS_WEIGHT_CHOICES: tuple[str, ...] = ("balanced", "none")

#: Name used when the fitted model uses no class weighting.
NO_CLASS_WEIGHT = "none"

#: Version tag stored in the artifact so a future incompatible change is
#: detectable instead of silently mis-loading an old bundle.
ARTIFACT_FORMAT_VERSION = 1


def resolve_class_weight(name: str) -> str | None:
    """Translate a CLI/config class-weight name into scikit-learn's argument.

    Args:
        name: ``"balanced"`` or ``"none"``.

    Returns:
        ``"balanced"`` or ``None`` (scikit-learn's spelling for "no weighting").

    Raises:
        DataError: If ``name`` is not one of :data:`CLASS_WEIGHT_CHOICES`.
    """
    if name not in CLASS_WEIGHT_CHOICES:
        raise DataError(
            f"class_weight must be one of {list(CLASS_WEIGHT_CHOICES)}, got {name!r}"
        )
    return None if name == NO_CLASS_WEIGHT else name


def class_weight_name(weight: str | None) -> str:
    """Translate scikit-learn's ``class_weight`` back into a report name.

    Args:
        weight: ``"balanced"``/``None`` (or any string scikit-learn accepts).

    Returns:
        The reporting name (``"none"`` for ``None``).
    """
    return NO_CLASS_WEIGHT if weight is None else str(weight)


def build_vectorizer(config: SentimentConfig) -> TfidfVectorizer:
    """Create an unfitted TF-IDF vectoriser from the configured settings.

    Args:
        config: Sentiment settings whose ``tfidf`` block drives the vectoriser.

    Returns:
        An unfitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
        ``lowercase`` and ``strip_accents`` stay at their defaults; the corpus is
        already cleaned by the data layer.
    """
    tfidf: TfidfConfig = config.tfidf
    return TfidfVectorizer(
        ngram_range=tfidf.ngram_range,
        min_df=tfidf.min_df,
        max_features=tfidf.max_features,
        sublinear_tf=tfidf.sublinear_tf,
    )


def build_classifier(config: SentimentConfig, class_weight: str | None) -> LogisticRegression:
    """Create an unfitted logistic-regression classifier.

    Args:
        config: Sentiment settings whose ``logreg`` block drives the classifier.
        class_weight: scikit-learn ``class_weight`` value (``"balanced"`` or
            ``None``).

    Returns:
        An unfitted :class:`~sklearn.linear_model.LogisticRegression` with a
        deterministic solver and the configured seed.
    """
    return LogisticRegression(
        C=config.logreg.C,
        max_iter=config.logreg.max_iter,
        class_weight=class_weight,
        solver="lbfgs",
        random_state=config.seed,
    )


def _text_array(frame: pd.DataFrame, config: SentimentConfig) -> np.ndarray:
    """Extract the configured text column as a string array.

    Args:
        frame: Review rows that already passed the text predicate.
        config: Sentiment settings (``text_column``).

    Returns:
        Object array of strings, as scikit-learn expects.

    Raises:
        DataError: If the frame is empty.
    """
    if frame.empty:
        raise DataError("cannot fit or score a sentiment model on an empty frame")
    return frame[config.text_column].astype(str).to_numpy()


def _align_probabilities(
    classifier: LogisticRegression, texts: np.ndarray
) -> np.ndarray:
    """Predict probabilities ordered by :data:`SENTIMENT_LABELS`.

    scikit-learn orders ``predict_proba`` columns by ``classifier.classes_``,
    which is only guaranteed to match the project's label order when every label
    was present in training. This re-aligns the columns explicitly and fills an
    absent class with zeros, so callers can index column ``0`` as "negative"
    without checking.

    Args:
        classifier: Fitted classifier.
        texts: Text array to score.

    Returns:
        ``(n_rows, len(SENTIMENT_LABELS))`` probability matrix.
    """
    raw = classifier.predict_proba(texts)
    aligned = np.zeros((raw.shape[0], len(SENTIMENT_LABELS)), dtype="float64")
    for column, label in enumerate(classifier.classes_):
        position = SENTIMENT_LABELS.index(int(label))
        aligned[:, position] = raw[:, column]
    return aligned


@dataclass
class SentimentBaseline:
    """A fitted TF-IDF + logistic-regression sentiment model.

    Attributes:
        vectorizer: Fitted vectoriser (vocabulary learned on the training sample).
        classifier: Fitted logistic-regression classifier.
        text_column: Review column the model reads.
        label_column: Review column holding the weak label it predicts.
        labels: Label order used by :meth:`predict_proba` (always
            :data:`munasib.core.constants.SENTIMENT_LABELS`).
        class_weight: scikit-learn ``class_weight`` value used to fit it.
        seed: Seed recorded for reproducibility.
        train_rows: Rows the model was fitted on.
        train_products: Distinct products in the training sample.
        vocabulary_size: Number of TF-IDF features retained.
        converged: Whether L-BFGS converged within ``max_iter``.
        n_iter: Iterations actually run, when available.
    """

    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    text_column: str
    label_column: str
    labels: tuple[int, ...]
    class_weight: str | None
    seed: int
    train_rows: int
    train_products: int
    vocabulary_size: int
    converged: bool
    n_iter: int | None

    @property
    def class_weight_name(self) -> str:
        """Reporting name of the class weighting (``"none"`` when unweighted)."""
        return class_weight_name(self.class_weight)

    def predict(self, texts: Any) -> np.ndarray:
        """Predict weak label codes for raw review texts.

        Args:
            texts: Sequence of review strings.

        Returns:
            ``int64`` array of label codes.
        """
        matrix = self.vectorizer.transform(texts)
        return np.asarray(self.classifier.predict(matrix), dtype="int64")

    def predict_proba(self, texts: Any) -> np.ndarray:
        """Predict label probabilities for raw review texts.

        Args:
            texts: Sequence of review strings.

        Returns:
            ``(n_rows, 3)`` matrix whose columns follow :attr:`labels`.
        """
        matrix = self.vectorizer.transform(texts)
        return _align_probabilities(self.classifier, matrix)

    def to_metadata(self) -> dict[str, object]:
        """Serialise the artifact metadata for the evaluation report.

        Returns:
            Plain-Python mapping of everything needed to interpret the artifact
            (no timestamps, so two identical runs produce identical metadata).
        """
        return {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "text_column": self.text_column,
            "label_column": self.label_column,
            "labels": list(self.labels),
            "class_weight": self.class_weight_name,
            "seed": self.seed,
            "train_rows": self.train_rows,
            "train_products": self.train_products,
            "vocabulary_size": self.vocabulary_size,
            "converged": self.converged,
            "n_iter": self.n_iter,
        }


def train_baseline(
    sample: pd.DataFrame, config: SentimentConfig, class_weight: str | None
) -> SentimentBaseline:
    """Fit the baseline on a training sample.

    The vectoriser is fitted here, on this frame only, and never sees an
    evaluation row.

    Args:
        sample: Training rows produced by
            :func:`munasib.review_intel.sentiment.sampling.build_train_sample`.
        config: Sentiment settings (TF-IDF, logistic regression, columns, seed).
        class_weight: scikit-learn ``class_weight`` value (``"balanced"`` or
            ``None``).

    Returns:
        The fitted :class:`SentimentBaseline`.

    Raises:
        DataError: If the sample is empty, lacks a label, is not train-only
            (leakage guard) or contains unusable text.
    """
    assert_split_only(sample, SPLIT_TRAIN)
    if sample.empty:
        raise DataError("training sample is empty; nothing to fit")
    usable = sample.loc[text_mask(sample, config)]
    if usable.empty:
        raise DataError("training sample has no row with enough text to fit")
    labels = usable[config.label_column]
    missing = [label for label in SENTIMENT_LABELS if int((labels == label).sum()) == 0]
    if missing:
        raise DataError(
            f"training sample is missing label(s) {missing}; the baseline needs "
            "every weak label to predict all three classes"
        )

    texts = _text_array(usable, config)
    targets = labels.astype("int64").to_numpy()
    vectorizer = build_vectorizer(config)
    features = vectorizer.fit_transform(texts)
    classifier = build_classifier(config, class_weight)

    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(features, targets)
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)

    n_iter: int | None = None
    iterations = getattr(classifier, "n_iter_", None)
    if iterations is not None:
        n_iter = int(np.max(iterations))

    return SentimentBaseline(
        vectorizer=vectorizer,
        classifier=classifier,
        text_column=config.text_column,
        label_column=config.label_column,
        labels=SENTIMENT_LABELS,
        class_weight=class_weight,
        seed=config.seed,
        train_rows=len(usable),
        train_products=int(usable[PRODUCT_COLUMN].nunique()),
        vocabulary_size=int(len(vectorizer.vocabulary_)),
        converged=converged,
        n_iter=n_iter,
    )


def evaluate_baseline(
    baseline: SentimentBaseline, frame: pd.DataFrame, config: SentimentConfig
) -> tuple[ClassificationMetrics, ProductCalibration, np.ndarray]:
    """Score the baseline on an evaluation split and check product calibration.

    Args:
        baseline: Fitted model.
        frame: Evaluation rows (``val`` or ``test``) produced by
            :func:`munasib.review_intel.sentiment.sampling.load_eval_split`.
        config: Sentiment settings (text and label columns).

    Returns:
        ``(metrics, calibration, probabilities)`` where ``probabilities`` is the
        ``(n_rows, 3)`` matrix aligned with ``SENTIMENT_LABELS``.

    Raises:
        DataError: If the frame is empty or lacks the required columns.
    """
    if frame.empty:
        raise DataError("evaluation frame is empty; nothing to score")
    for column in (config.text_column, config.label_column, PRODUCT_COLUMN):
        if column not in frame.columns:
            raise DataError(f"evaluation frame is missing column {column!r}")
    texts = frame[config.text_column].astype(str).to_numpy()
    probabilities = baseline.predict_proba(texts)
    predictions = np.asarray(
        [SENTIMENT_LABELS[index] for index in probabilities.argmax(axis=1)], dtype="int64"
    )
    truth = frame[config.label_column].astype("int64").to_numpy()
    metrics = classification_metrics(truth, predictions, probabilities)
    calibration = compute_product_calibration(frame[PRODUCT_COLUMN], truth, predictions)
    return metrics, calibration, probabilities


def save_baseline(baseline: SentimentBaseline, path: Path) -> Path:
    """Write the fitted model to disk with ``joblib``.

    Args:
        baseline: Fitted model.
        path: Destination file (parent directories are created).

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, path)
    return path


def load_baseline(path: Path) -> SentimentBaseline:
    """Load a model written by :func:`save_baseline`.

    Args:
        path: Artifact path.

    Returns:
        The deserialised :class:`SentimentBaseline`.

    Raises:
        DataError: If the file is missing or does not hold a baseline model.
    """
    if not path.exists():
        raise DataError(f"model artifact not found: {path}")
    loaded = joblib.load(path)
    if not isinstance(loaded, SentimentBaseline):
        raise DataError(
            f"{path} does not contain a SentimentBaseline (got {type(loaded).__name__})"
        )
    return loaded


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "CLASS_WEIGHT_CHOICES",
    "NO_CLASS_WEIGHT",
    "SentimentBaseline",
    "build_classifier",
    "build_vectorizer",
    "class_weight_name",
    "evaluate_baseline",
    "load_baseline",
    "resolve_class_weight",
    "save_baseline",
    "train_baseline",
]

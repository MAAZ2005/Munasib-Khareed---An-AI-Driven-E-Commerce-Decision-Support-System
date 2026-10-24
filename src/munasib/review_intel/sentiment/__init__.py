"""Review sentiment analysis built on weak supervision from star ratings.

The labels here are **not** human-judged sentiment: ``sentiment_label_weak`` is
derived from the reviewer's star rating (1-2 negative, 3 neutral, 4-5 positive),
so every metric in this package measures agreement with that proxy and nothing
more. Three modules share the work:

* :mod:`~munasib.review_intel.sentiment.sampling` - leakage-safe training sample
  and evaluation splits.
* :mod:`~munasib.review_intel.sentiment.metrics` - metrics that always carry the
  majority-class baseline, plus the product-level calibration check.
* :mod:`~munasib.review_intel.sentiment.baseline` - the TF-IDF + logistic
  regression reference model and its reloadable artifact.
"""

from munasib.review_intel.sentiment.baseline import (
    CLASS_WEIGHT_CHOICES,
    SentimentBaseline,
    build_classifier,
    build_vectorizer,
    class_weight_name,
    evaluate_baseline,
    load_baseline,
    resolve_class_weight,
    save_baseline,
    train_baseline,
)
from munasib.review_intel.sentiment.metrics import (
    BinaryMetrics,
    ClassificationMetrics,
    ClassMetrics,
    MajorityBaseline,
    ProductCalibration,
    binary_metrics,
    classification_metrics,
    compute_product_calibration,
    majority_baseline,
    render_calibration_table,
    render_metrics_table,
    write_calibration_csv,
    write_evaluation_report,
)
from munasib.review_intel.sentiment.sampling import (
    EVAL_SPLITS,
    PRODUCT_COLUMN,
    SPLIT_COLUMN,
    CountStep,
    SampleReport,
    TrainingSample,
    assert_split_only,
    build_train_sample,
    load_eval_split,
    text_mask,
)

__all__ = [
    "CLASS_WEIGHT_CHOICES",
    "EVAL_SPLITS",
    "PRODUCT_COLUMN",
    "SPLIT_COLUMN",
    "BinaryMetrics",
    "ClassMetrics",
    "ClassificationMetrics",
    "CountStep",
    "MajorityBaseline",
    "ProductCalibration",
    "SampleReport",
    "SentimentBaseline",
    "TrainingSample",
    "assert_split_only",
    "binary_metrics",
    "build_classifier",
    "build_train_sample",
    "build_vectorizer",
    "class_weight_name",
    "classification_metrics",
    "compute_product_calibration",
    "evaluate_baseline",
    "load_baseline",
    "load_eval_split",
    "majority_baseline",
    "render_calibration_table",
    "render_metrics_table",
    "resolve_class_weight",
    "save_baseline",
    "text_mask",
    "train_baseline",
    "write_calibration_csv",
    "write_evaluation_report",
]

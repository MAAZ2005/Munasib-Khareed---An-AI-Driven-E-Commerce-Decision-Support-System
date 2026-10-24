"""Tests for the ``review_intel.sentiment`` configuration block.

Covers the documented defaults, the shipped YAML agreeing with the code, and the
rejection of every value that would otherwise produce a silently broken sample
(a zero row budget, a zero per-product cap, a typo'd text column that would make
every review look unusable).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from munasib.core.config import (
    SENTIMENT_LABEL_COLUMN_VOCABULARY,
    SENTIMENT_TEXT_COLUMN_VOCABULARY,
    LogRegConfig,
    SentimentConfig,
    Settings,
    TfidfConfig,
    load_settings,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def test_documented_defaults() -> None:
    """The defaults match those documented in docs/sentiment.md."""
    config = SentimentConfig()
    assert config.text_column == "review_text_clean"
    assert config.label_column == "sentiment_label_weak"
    assert config.max_reviews_per_product == 50
    assert config.max_train_rows == 100_000
    assert config.min_chars == 3
    assert config.seed == 42
    assert config.max_length_tokens == 128
    assert config.tfidf == TfidfConfig(
        ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True
    )
    assert config.logreg == LogRegConfig(C=1.0, max_iter=1000)
    assert config.model_path == Path("models/sentiment_tfidf_logreg.joblib")
    assert config.report_dir == Path("reports/evaluation")


def test_vocabularies_match_the_review_schema() -> None:
    """The accepted columns are the ones the review schema actually has."""
    assert SENTIMENT_TEXT_COLUMN_VOCABULARY == ("review_text_clean", "review_text_raw")
    assert SENTIMENT_LABEL_COLUMN_VOCABULARY == ("sentiment_label_weak",)


def test_shipped_config_declares_every_setting() -> None:
    """The repo config file names each key, so nothing falls back silently."""
    text = REPO_CONFIG.read_text(encoding="utf-8")
    for key in (
        "text_column:",
        "label_column:",
        "max_reviews_per_product:",
        "max_train_rows:",
        "min_chars:",
        "seed:",
        "max_length_tokens:",
        "tfidf:",
        "logreg:",
        "model_path:",
        "report_dir:",
    ):
        assert key in text


def test_shipped_config_loads_with_the_repo_defaults() -> None:
    """The shipped YAML matches the code defaults exactly (no drift)."""
    loaded = load_settings(REPO_CONFIG).review_intel.sentiment
    assert loaded == SentimentConfig()


def test_settings_exposes_the_block() -> None:
    """``Settings`` surfaces the block under ``review_intel.sentiment``."""
    assert Settings().review_intel.sentiment.max_train_rows == 100_000


def test_unknown_text_column_is_rejected() -> None:
    """A typo'd text column would make every review look unusable."""
    with pytest.raises(ValidationError, match="text_column"):
        SentimentConfig(text_column="text")


def test_unknown_label_column_is_rejected() -> None:
    """A typo'd label column would fail deep inside the sample builder."""
    with pytest.raises(ValidationError, match="label_column"):
        SentimentConfig(label_column="sentiment_label")


@pytest.mark.parametrize(
    "field",
    [
        "max_reviews_per_product",
        "max_train_rows",
        "min_chars",
        "max_length_tokens",
        "batch_size",
        "max_sequence_length",
    ],
)
def test_zero_counts_are_rejected(field: str) -> None:
    """A zero cap/budget/length cannot produce a usable sample - rejected."""
    with pytest.raises(ValidationError, match=field):
        SentimentConfig(**{field: 0})


def test_negative_min_chars_is_rejected() -> None:
    """Negative lengths are as invalid as zero ones."""
    with pytest.raises(ValidationError, match="min_chars"):
        SentimentConfig(min_chars=-1)


def test_descending_ngram_range_is_rejected() -> None:
    """``ngram_range=(2, 1)`` makes scikit-learn raise much later and less clearly."""
    with pytest.raises(ValidationError, match="ngram_range"):
        TfidfConfig(ngram_range=(2, 1))


def test_zero_ngram_size_is_rejected() -> None:
    """Unigram index 0 does not exist; the range must start at 1."""
    with pytest.raises(ValidationError, match="ngram_range"):
        TfidfConfig(ngram_range=(0, 2))


@pytest.mark.parametrize("field", ["min_df", "max_features"])
def test_zero_tfidf_counts_are_rejected(field: str) -> None:
    """A zero ``min_df``/``max_features`` cannot yield a vocabulary."""
    with pytest.raises(ValidationError, match=field):
        TfidfConfig(**{field: 0})


def test_non_positive_regularisation_is_rejected() -> None:
    """``C <= 0`` is invalid for logistic regression."""
    with pytest.raises(ValidationError, match="logreg.C"):
        LogRegConfig(C=0.0)


def test_zero_max_iter_is_rejected() -> None:
    """``max_iter=0`` would never fit anything."""
    with pytest.raises(ValidationError, match="max_iter"):
        LogRegConfig(max_iter=0)


def test_single_ngram_and_larger_cap_are_accepted() -> None:
    """Non-default but valid settings must be accepted, not just the defaults."""
    config = SentimentConfig(
        max_reviews_per_product=5,
        max_train_rows=1_000,
        min_chars=10,
        tfidf=TfidfConfig(ngram_range=(1, 1), min_df=1, max_features=1_000, sublinear_tf=False),
        logreg=LogRegConfig(C=0.5, max_iter=50),
    )
    assert config.tfidf.ngram_range == (1, 1)
    assert config.logreg.C == 0.5

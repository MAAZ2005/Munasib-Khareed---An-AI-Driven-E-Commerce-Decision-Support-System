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
    CLASS_WEIGHTING_CHOICES,
    RNN_CELL_CHOICES,
    SENTIMENT_LABEL_COLUMN_VOCABULARY,
    SENTIMENT_TEXT_COLUMN_VOCABULARY,
    LogRegConfig,
    RnnConfig,
    SentimentConfig,
    Settings,
    TfidfConfig,
    TransformerConfig,
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
        "class_weighting:",
        "tfidf:",
        "logreg:",
        "transformer:",
        "rnn:",
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


def test_neural_defaults() -> None:
    """The transformer and RNN defaults match docs/sentiment.md."""
    config = SentimentConfig()
    assert config.class_weighting == "sqrt_inverse"
    assert config.transformer == TransformerConfig(
        model_name="distilbert-base-uncased",
        learning_rate=3e-5,
        batch_size=32,
        epochs=3,
        warmup_ratio=0.06,
        weight_decay=0.01,
        max_grad_norm=1.0,
        early_stopping_patience=1,
        output_dir=Path("models/sentiment_distilbert"),
    )
    assert config.rnn == RnnConfig(
        embedding_dim=128,
        hidden_size=128,
        num_layers=1,
        bidirectional=True,
        dropout=0.3,
        vocab_size=50_000,
        min_freq=2,
        cell="gru",
        learning_rate=1e-3,
        output_dir=Path("models/sentiment_rnn"),
    )


def test_class_weighting_choices_are_derived_from_the_literal() -> None:
    """The CLI choices cannot drift from the validated type."""
    assert CLASS_WEIGHTING_CHOICES == ("none", "sqrt_inverse", "balanced")
    assert RNN_CELL_CHOICES == ("gru", "lstm")


def test_unknown_class_weighting_is_rejected() -> None:
    """A typo in the weighting scheme would silently change the loss."""
    with pytest.raises(ValidationError, match="class_weighting"):
        SentimentConfig(class_weighting="inverse")  # type: ignore[arg-type]


def test_unknown_rnn_cell_is_rejected() -> None:
    """Only the two implemented recurrences are accepted."""
    with pytest.raises(ValidationError, match="cell"):
        RnnConfig(cell="rnn")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "   "),
        ("learning_rate", 0.0),
        ("learning_rate", 1.5),
        ("batch_size", 0),
        ("epochs", 0),
        ("warmup_ratio", 1.0),
        ("warmup_ratio", -0.1),
        ("weight_decay", -0.01),
        ("max_grad_norm", 0.0),
        ("early_stopping_patience", -1),
    ],
)
def test_invalid_transformer_settings_are_rejected(field: str, value: object) -> None:
    """Every invalid fine-tuning value fails at load time, not mid-run."""
    with pytest.raises(ValidationError, match=field):
        TransformerConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_dim", 0),
        ("hidden_size", 0),
        ("num_layers", 0),
        ("vocab_size", 1),
        ("min_freq", 0),
        ("dropout", 1.0),
        ("dropout", -0.1),
    ],
)
def test_invalid_rnn_settings_are_rejected(field: str, value: object) -> None:
    """Every invalid recurrent value fails at load time, not mid-run."""
    with pytest.raises(ValidationError, match=field):
        RnnConfig(**{field: value})

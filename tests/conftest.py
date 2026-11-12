"""Shared fixtures for the test suite.

Only the fixtures that must outlive a module live here. Everything is
network-free and CPU-only: the sentiment checkpoint is a randomly initialised
miniature DistilBERT with a scratch tokenizer, exactly like the smoke run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TINY_TEXTS = [
    "good vacuum works great",
    "bad broke after a day",
    "great price and quiet",
    "the filter broke",
]


@pytest.fixture(scope="session")
def tiny_sentiment_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a tiny transformer checkpoint that needs no download.

    Args:
        tmp_path_factory: Pytest session-scoped temporary directory factory.

    Returns:
        The ``best/`` directory written by ``save_model``.
    """
    from munasib.review_intel.sentiment.data import build_scratch_tokenizer
    from munasib.review_intel.sentiment.models import (
        build_tiny_transformer,
        save_model,
    )

    tokenizer = build_scratch_tokenizer(TINY_TEXTS, max_length=16)
    model = build_tiny_transformer(vocab_size=tokenizer.vocab_size, max_length=16)
    model.eval()
    directory = tmp_path_factory.mktemp("tiny-sentiment") / "best"
    save_model(
        model,
        directory,
        architecture="transformer",
        max_length=16,
        class_weighting="sqrt_inverse",
        tokenizer=tokenizer,
        base_model="tiny-scratch",
    )
    return directory

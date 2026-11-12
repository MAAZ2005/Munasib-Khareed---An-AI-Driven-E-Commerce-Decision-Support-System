"""Tests for the reusable sentiment predictor (``predictor.py``).

Every test runs on the CPU against the session's randomly initialised miniature
transformer, so nothing downloads weights and nothing needs a GPU.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from munasib.core.config import get_settings
from munasib.core.exceptions import DataError
from munasib.review_intel.sentiment.models import NUM_LABELS
from munasib.review_intel.sentiment.predictor import (
    SIDECAR_SUFFIX,
    SentimentPredictor,
    describe_checkpoint,
    file_sha256,
    has_text,
    sidecar_path,
)

TEXTS = [
    "good vacuum works great",
    "bad broke after a day",
    "great price and quiet",
    "the filter broke",
]


@pytest.fixture(scope="module")
def predictor(tiny_sentiment_checkpoint: Path) -> SentimentPredictor:
    """Predictor loaded from the tiny checkpoint.

    Args:
        tiny_sentiment_checkpoint: Session fixture writing the checkpoint.

    Returns:
        The loaded predictor.
    """
    return SentimentPredictor.from_checkpoint(
        tiny_sentiment_checkpoint, device="cpu", batch_size=2
    )


def test_has_text_treats_missing_and_blank_as_no_text() -> None:
    """No probability may be invented for a review that has no text."""
    assert has_text("good")
    assert not has_text(None)
    assert not has_text(float("nan"))
    assert not has_text("   ")
    assert not has_text("\t\n")


def test_sidecar_path_keeps_the_stem(tmp_path: Path) -> None:
    """The sidecar sits next to the parquet and shares its stem."""
    output = tmp_path / "preds.parquet"
    assert sidecar_path(output) == tmp_path / f"preds{SIDECAR_SUFFIX}"
    assert sidecar_path(output).name == "preds.meta.json"


def test_predict_proba_returns_input_order_and_nan_rows(predictor: SentimentPredictor) -> None:
    """One row per input, aligned by position; empty text gets a NaN row."""
    texts = [TEXTS[0], None, "", TEXTS[1]]
    result = predictor.predict_proba(texts)

    assert result.probabilities.shape == (4, NUM_LABELS)
    assert result.rows == 4
    assert result.predicted_rows == 2
    assert result.has_prediction.tolist() == [True, False, False, True]
    assert np.isnan(result.probabilities[1]).all()
    assert np.isnan(result.probabilities[2]).all()
    assert result.labels.tolist()[1] == -1
    assert result.labels.tolist()[2] == -1
    assert result.labels.tolist()[0] == int(result.probabilities[0].argmax())
    assert math.isclose(float(result.probabilities[0].sum()), 1.0, abs_tol=1e-6)


def test_predict_labels_returns_none_for_missing_text(predictor: SentimentPredictor) -> None:
    """The label list drops to ``None`` exactly where there was no text."""
    labels = predictor.predict_labels([TEXTS[0], None, TEXTS[2]])
    assert labels[1] is None
    assert labels[0] in {0, 1, 2}
    assert labels[2] in {0, 1, 2}


def test_predict_proba_is_deterministic(predictor: SentimentPredictor) -> None:
    """Two calls on the same texts return bit-identical scores."""
    first = predictor.predict_proba(TEXTS)
    second = predictor.predict_proba(TEXTS)
    assert np.array_equal(first.probabilities, second.probabilities)
    assert np.array_equal(first.labels, second.labels)


def test_bucketing_does_not_change_probabilities(
    tiny_sentiment_checkpoint: Path, predictor: SentimentPredictor
) -> None:
    """Padding is inert, so batch size must not move the probabilities."""
    one_at_a_time = SentimentPredictor.from_checkpoint(
        tiny_sentiment_checkpoint, device="cpu", batch_size=1
    )
    batched = one_at_a_time.predict_proba(TEXTS).probabilities
    pooled = predictor.predict_proba(TEXTS).probabilities
    assert np.allclose(batched, pooled, atol=1e-6)


def test_predict_proba_handles_an_empty_input(predictor: SentimentPredictor) -> None:
    """An empty input yields an empty result instead of touching the model."""
    result = predictor.predict_proba([])
    assert result.probabilities.shape == (0, NUM_LABELS)
    assert result.labels.size == 0
    assert result.has_prediction.size == 0


def test_from_checkpoint_uses_the_metadata_token_budget(
    tiny_sentiment_checkpoint: Path,
) -> None:
    """The encoder is built with the checkpoint's own max length, not a default."""
    loaded = SentimentPredictor.from_checkpoint(
        tiny_sentiment_checkpoint, device="cpu", batch_size=4
    )
    assert loaded.meta.max_length == 16
    assert loaded.encoder.max_length == 16
    assert loaded.meta.architecture == "transformer"
    assert loaded.batch_size == 4
    assert loaded.device.type == "cpu"


def test_from_checkpoint_reads_the_configured_batch_size(
    tiny_sentiment_checkpoint: Path,
) -> None:
    """``batch_size=None`` comes from ``review_intel.sentiment.prediction``."""
    loaded = SentimentPredictor.from_checkpoint(tiny_sentiment_checkpoint, device="cpu")
    configured = get_settings().review_intel.sentiment.prediction.batch_size
    assert loaded.batch_size == configured


def test_from_checkpoint_rejects_an_unknown_device(tiny_sentiment_checkpoint: Path) -> None:
    """A typo'd device must fail instead of silently falling back."""
    with pytest.raises(DataError, match="device"):
        SentimentPredictor.from_checkpoint(
            tiny_sentiment_checkpoint, device="quantum", batch_size=2
        )


def test_describe_checkpoint_hashes_weights_and_metadata(
    tiny_sentiment_checkpoint: Path,
) -> None:
    """The identity records what the predictions were produced with."""
    identity = describe_checkpoint(tiny_sentiment_checkpoint)
    assert identity.architecture == "transformer"
    assert identity.max_length == 16
    assert identity.weights_file == "model.safetensors"
    assert identity.weights_sha256 == file_sha256(
        tiny_sentiment_checkpoint / "model.safetensors"
    )
    assert identity.model_meta_sha256 == file_sha256(
        tiny_sentiment_checkpoint / "model_meta.json"
    )
    payload = identity.to_dict()
    assert payload["weights_sha256"] == identity.weights_sha256
    assert payload["path"] == str(tiny_sentiment_checkpoint)


def test_describe_checkpoint_rejects_a_missing_weights_file(
    tmp_path: Path, tiny_sentiment_checkpoint: Path
) -> None:
    """A directory with metadata but no weights cannot be scored."""
    copied = tmp_path / "best"
    shutil.copytree(tiny_sentiment_checkpoint, copied)
    (copied / "model.safetensors").unlink()
    with pytest.raises(DataError, match="neither"):
        describe_checkpoint(copied)


def test_file_sha256_rejects_a_missing_file(tmp_path: Path) -> None:
    """Hashing a file that is not there is an error, not an empty digest."""
    with pytest.raises(DataError, match="missing file"):
        file_sha256(tmp_path / "nope.bin")


def test_predict_proba_accepts_any_sequence(predictor: SentimentPredictor) -> None:
    """A tuple or generator is scored like a list, in the same order."""
    as_tuple = predictor.predict_proba(tuple(TEXTS)).probabilities
    as_list = predictor.predict_proba(TEXTS).probabilities
    assert np.array_equal(as_tuple, as_list)

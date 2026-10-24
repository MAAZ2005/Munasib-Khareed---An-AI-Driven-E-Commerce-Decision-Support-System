"""Tests for the PyTorch sentiment models (``models.py``).

Everything here runs on the CPU against the randomly initialised miniature
transformer: no pretrained weights are downloaded, no GPU is touched, and the
only tokenizer is the in-memory scratch one built from a handful of strings.

Environment note: pytest enables ``faulthandler``, and this torch build
(2.14.1+cpu on Windows/Python 3.13) reports a *first-chance* access violation
from some CPU kernels (``layer_norm``, ``max``) before handling it internally.
It reproduces with no project code at all::

    python -X faulthandler -c "import torch, torch.nn as nn; nn.LayerNorm(8)(torch.randn(2, 8))"

The dumps in the pytest output are therefore noise from the interpreter's
handler, not a failure: the tests pass and the process exits 0.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from munasib.core.config import RnnConfig
from munasib.core.constants import SENTIMENT_LABELS
from munasib.core.exceptions import DataError
from munasib.review_intel.sentiment.data import build_scratch_tokenizer, collate_batch
from munasib.review_intel.sentiment.models import (
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    PAD_ID,
    UNK_ID,
    ModelMeta,
    SentimentRNN,
    Vocabulary,
    build_rnn,
    build_tiny_transformer,
    forward_logits,
    load_model,
    save_model,
    tokenize,
)

TRAIN_TEXTS = [
    "good vacuum works great",
    "bad broke after a day",
    "great price and quiet",
    "the filter broke",
]


def _rnn_config() -> RnnConfig:
    """Small recurrent config so the tests stay fast.

    Returns:
        A validated config with tiny dimensions.
    """
    return RnnConfig(
        embedding_dim=8,
        hidden_size=8,
        num_layers=1,
        bidirectional=True,
        dropout=0.0,
        vocab_size=200,
        min_freq=1,
        cell="gru",
    )


def _vocabulary() -> Vocabulary:
    """Vocabulary fitted on the training strings only.

    Returns:
        The fitted vocabulary.
    """
    return Vocabulary.from_texts(TRAIN_TEXTS, min_freq=1, max_size=200)


def test_tokenize_lowercases_and_splits_on_whitespace() -> None:
    """Encoding is a pure function of the text, so vocabulary fits are stable."""
    assert tokenize("Good   VACUUM\tworks") == ["good", "vacuum", "works"]
    assert tokenize("") == []


def test_vocabulary_reserves_pad_and_unknown_ids() -> None:
    """``[PAD]`` is 0 and ``[UNK]`` is 1, whatever the training texts contain."""
    vocabulary = _vocabulary()
    assert vocabulary.tokens["[PAD]"] == PAD_ID
    assert vocabulary.tokens["[UNK]"] == UNK_ID
    assert len(vocabulary) == len(vocabulary.tokens)
    assert vocabulary.encode("good vacuum")[0] == vocabulary.tokens["good"]


def test_vocabulary_maps_unseen_tokens_to_unknown() -> None:
    """A token that only appears in evaluation text must not get an id."""
    vocabulary = _vocabulary()
    assert "refunded" not in vocabulary.tokens
    assert vocabulary.encode("refunded") == [UNK_ID]


def test_vocabulary_never_encodes_an_empty_sequence() -> None:
    """Blank text still yields one token, so no batch row has zero length."""
    assert _vocabulary().encode("   ") == [UNK_ID]


def test_vocabulary_respects_min_freq_and_the_size_cap() -> None:
    """Rare tokens are dropped and the cap includes the two special tokens."""
    vocabulary = Vocabulary.from_texts(["a a b c"], min_freq=2, max_size=3)
    assert "b" not in vocabulary.tokens
    assert "c" not in vocabulary.tokens
    assert len(vocabulary) == 3  # [PAD], [UNK], "a"


def test_vocabulary_rejects_invalid_fit_settings() -> None:
    """A zero frequency floor or a cap below two cannot produce a usable table."""
    with pytest.raises(ValueError, match="min_freq"):
        Vocabulary.from_texts(["a"], min_freq=0, max_size=10)
    with pytest.raises(ValueError, match="max_size"):
        Vocabulary.from_texts(["a"], min_freq=1, max_size=1)


def test_vocabulary_round_trip() -> None:
    """The JSON form restores the exact id assignment."""
    vocabulary = _vocabulary()
    restored = Vocabulary.from_dict(vocabulary.to_dict())
    assert restored == vocabulary
    assert restored.encode("great price") == vocabulary.encode("great price")


def test_vocabulary_rejects_a_broken_payload() -> None:
    """A malformed vocab file fails with a domain error, not a KeyError."""
    with pytest.raises(DataError, match="invalid vocabulary"):
        Vocabulary.from_dict({"min_freq": 1})


def test_tiny_transformer_forward_shape_and_label_schema() -> None:
    """The tiny model returns one logit row per review and the project labels."""
    model = build_tiny_transformer(vocab_size=32, max_length=16)
    model.eval()
    input_ids = torch.randint(2, 32, (5, 7))
    attention_mask = torch.ones_like(input_ids)
    logits = forward_logits(model, input_ids, attention_mask)
    assert logits.shape == (5, NUM_LABELS)
    assert model.config.num_labels == len(SENTIMENT_LABELS)
    assert model.config.id2label == ID2LABEL
    assert model.config.label2id == LABEL2ID
    assert model.config.max_position_embeddings == 16


def test_rnn_forward_shape() -> None:
    """The recurrent model also returns ``(batch, 3)`` logits."""
    vocabulary = _vocabulary()
    model = build_rnn(_rnn_config(), vocab_size=len(vocabulary))
    batch = collate_batch(
        [
            {
                "index": 0,
                "input_ids": vocabulary.encode("good vacuum"),
                "label": 2,
                "asin": "a",
            },
            {
                "index": 1,
                "input_ids": vocabulary.encode("bad broke after a day"),
                "label": 0,
                "asin": "b",
            },
        ],
        pad_id=PAD_ID,
    )
    logits = forward_logits(model, batch.input_ids, batch.attention_mask)
    assert logits.shape == (2, NUM_LABELS)


def test_rnn_padding_cannot_change_a_prediction() -> None:
    """Masked mean/max pooling ignores padding, so trailing pads are inert."""
    vocabulary = _vocabulary()
    model = build_rnn(_rnn_config(), vocab_size=len(vocabulary))
    model.eval()
    short = torch.tensor([vocabulary.encode("good vacuum works")])
    padded = torch.cat([short, torch.full((1, 3), PAD_ID)], dim=1)
    with torch.no_grad():
        bare = model(short, torch.ones_like(short))
        with_pads = model(padded, torch.tensor([[1, 1, 1, 0, 0, 0]]))
    assert torch.allclose(bare, with_pads, atol=1e-6)


def test_rnn_rejects_an_unknown_cell() -> None:
    """Only the two implemented recurrences are allowed, at construction too."""
    with pytest.raises(ValueError, match="cell"):
        SentimentRNN(
            vocab_size=10,
            embedding_dim=4,
            hidden_size=4,
            num_layers=1,
            bidirectional=False,
            dropout=0.0,
            cell="transformer",
        )


def test_transformer_checkpoint_round_trip_is_offline(tmp_path: Path) -> None:
    """``save_model`` then ``load_model`` reproduces the logits with no network."""
    tokenizer = build_scratch_tokenizer(TRAIN_TEXTS, max_length=16)
    model = build_tiny_transformer(vocab_size=tokenizer.vocab_size, max_length=16)
    model.eval()
    directory = tmp_path / "best"
    meta = save_model(
        model,
        directory,
        architecture="transformer",
        max_length=16,
        class_weighting="sqrt_inverse",
        tokenizer=tokenizer,
        base_model="tiny-scratch",
    )
    assert meta.architecture == "transformer"
    assert (directory / "model_meta.json").is_file()
    assert (directory / "config.json").is_file()
    assert (directory / "tokenizer.json").is_file()

    input_ids = torch.tensor([tokenizer.encode("good vacuum"), tokenizer.encode("bad broke")])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        expected = forward_logits(model, input_ids, attention_mask)

    saved = load_model(directory)
    assert saved.meta == meta
    assert saved.tokenizer is not None
    assert saved.vocabulary is None
    with torch.no_grad():
        actual = forward_logits(saved.model, input_ids, attention_mask)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_rnn_checkpoint_round_trip(tmp_path: Path) -> None:
    """The recurrent artifact restores weights *and* the vocabulary that made them."""
    vocabulary = _vocabulary()
    model = build_rnn(_rnn_config(), vocab_size=len(vocabulary))
    model.eval()
    directory = tmp_path / "best"
    save_model(
        model,
        directory,
        architecture="rnn",
        max_length=12,
        class_weighting="balanced",
        vocabulary=vocabulary,
        rnn_config=_rnn_config(),
    )
    assert (directory / "vocab.json").is_file()
    assert (directory / "model.pt").is_file()

    input_ids = torch.tensor([vocabulary.encode("great price")])
    with torch.no_grad():
        expected = model(input_ids, torch.ones_like(input_ids))

    saved = load_model(directory)
    assert saved.vocabulary == vocabulary
    assert saved.tokenizer is None
    with torch.no_grad():
        actual = forward_logits(saved.model, input_ids, torch.ones_like(input_ids))
    assert torch.allclose(expected, actual, atol=1e-6)


def test_save_model_requires_the_encoder(tmp_path: Path) -> None:
    """A transformer without its tokenizer could never encode text again."""
    model = build_tiny_transformer(vocab_size=16, max_length=8)
    with pytest.raises(DataError, match="tokenizer"):
        save_model(
            model,
            tmp_path / "x",
            architecture="transformer",
            max_length=8,
            class_weighting="none",
        )
    with pytest.raises(DataError, match="vocabulary"):
        save_model(
            model,
            tmp_path / "y",
            architecture="rnn",
            max_length=8,
            class_weighting="none",
        )


def test_save_model_rejects_an_unknown_architecture(tmp_path: Path) -> None:
    """A typo'd architecture must not produce a half-written directory."""
    with pytest.raises(DataError, match="architecture"):
        save_model(
            build_tiny_transformer(vocab_size=16, max_length=8),
            tmp_path / "z",
            architecture="cnn",
            max_length=8,
            class_weighting="none",
        )


def test_model_meta_rejects_an_unknown_architecture() -> None:
    """Metadata read from disk is validated too."""
    with pytest.raises(DataError, match="architecture"):
        ModelMeta(architecture="cnn", max_length=8, class_weighting="none")


def test_load_model_reports_a_missing_directory(tmp_path: Path) -> None:
    """A missing checkpoint fails with a path in the message."""
    with pytest.raises(DataError, match="model_meta.json"):
        load_model(tmp_path / "nowhere")

"""Tests for tokenisation, dynamic padding and length-bucketed batching.

The tokenizer is built in memory from a handful of strings, so nothing here
touches the network; the models themselves are not involved.

Environment note: pytest enables ``faulthandler``, and this torch build reports a
benign first-chance access violation from some CPU kernels before handling it
internally. It reproduces without any project code
(``python -X faulthandler -c "import torch, torch.nn as nn; nn.LayerNorm(8)(torch.randn(2, 8))"``),
so a dump in the output is not a test failure.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from munasib.core.exceptions import DataError
from munasib.review_intel.sentiment.data import (
    SPECIAL_TOKENS,
    LengthBucketedBatchSampler,
    SentimentDataset,
    TextEncoder,
    build_scratch_tokenizer,
    collate_batch,
)
from munasib.review_intel.sentiment.models import PAD_ID, UNK_ID, Vocabulary

TEXTS = ["good vacuum works great", "bad broke after a day", "great price and quiet"]


def _vocabulary() -> Vocabulary:
    """Vocabulary fitted on the test strings.

    Returns:
        The fitted vocabulary.
    """
    return Vocabulary.from_texts(TEXTS, min_freq=1, max_size=100)


def _tokenizer() -> PreTrainedTokenizerFast:
    """Scratch tokenizer fitted on the test strings.

    Returns:
        A fast word-level tokenizer.
    """
    return build_scratch_tokenizer(TEXTS, max_length=12)


def _dataset() -> SentimentDataset:
    """Three-review dataset using the recurrent vocabulary.

    Returns:
        The dataset.
    """
    return SentimentDataset(
        texts=TEXTS,
        labels=[2, 0, 2],
        asins=["p1", "p2", "p1"],
        encoder=TextEncoder(max_length=12, vocabulary=_vocabulary()),
    )


def test_collate_pads_to_the_batch_maximum_and_tracks_indices() -> None:
    """Padding is per batch, not per split, and each row keeps its dataset index."""
    items = [
        {"index": 4, "input_ids": [5, 6], "label": 2, "asin": "a"},
        {"index": 1, "input_ids": [7, 8, 9, 10], "label": 0, "asin": "b"},
    ]
    batch = collate_batch(items, pad_id=PAD_ID)
    assert tuple(batch.input_ids.shape) == (2, 4)
    assert batch.attention_mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert batch.input_ids[0].tolist() == [5, 6, PAD_ID, PAD_ID]
    assert batch.indices.tolist() == [4, 1]
    assert batch.labels.tolist() == [2, 0]
    assert batch.asins == ("a", "b")


def test_collate_uses_the_configured_pad_id() -> None:
    """The transformer and the recurrent model may pad with different ids."""
    batch = collate_batch(
        [{"index": 0, "input_ids": [3], "label": 1, "asin": "a"}], pad_id=99
    )
    assert batch.input_ids.tolist() == [[3]]


def test_collate_rejects_an_empty_batch() -> None:
    """An empty batch would otherwise produce an unshaped tensor."""
    with pytest.raises(DataError, match="empty batch"):
        collate_batch([])


def test_dataset_rejects_misaligned_columns() -> None:
    """Texts, labels and products must describe the same rows."""
    with pytest.raises(DataError, match="same length"):
        SentimentDataset(
            texts=TEXTS,
            labels=[2, 0],
            asins=["p1", "p2", "p1"],
            encoder=TextEncoder(max_length=12, vocabulary=_vocabulary()),
        )


def test_dataset_encodes_per_item_and_caches_lengths() -> None:
    """``lengths()`` encodes once; every item exposes index, ids, label and product."""
    dataset = _dataset()
    assert len(dataset) == 3
    item = dataset[1]
    assert item["index"] == 1
    assert item["label"] == 0
    assert item["asin"] == "p2"
    assert item["input_ids"] == _vocabulary().encode(TEXTS[1])
    first = dataset.lengths()
    assert first == [len(ids) for ids in (dataset[0]["input_ids"], dataset[1]["input_ids"], dataset[2]["input_ids"])]
    assert dataset.lengths() is first


def test_text_encoder_requires_exactly_one_encoder() -> None:
    """Mixing a tokenizer and a vocabulary (or neither) is a configuration bug."""
    with pytest.raises(DataError, match="exactly one"):
        TextEncoder(max_length=8)
    with pytest.raises(DataError, match="exactly one"):
        TextEncoder(max_length=8, tokenizer=_tokenizer(), vocabulary=_vocabulary())


def test_text_encoder_rejects_a_non_positive_length() -> None:
    """A zero token budget cannot encode anything."""
    with pytest.raises(DataError, match="max_length"):
        TextEncoder(max_length=0, vocabulary=_vocabulary())


def test_text_encoder_truncates_both_architectures() -> None:
    """The configured token budget is a hard cap for ids, not just for attention."""
    vocabulary = _vocabulary()
    recurrent = TextEncoder(max_length=3, vocabulary=vocabulary)
    assert len(recurrent.encode("good vacuum works great")) == 3

    tokenizer = _tokenizer()
    transformer = TextEncoder(max_length=4, tokenizer=tokenizer)
    ids = transformer.encode("good vacuum works great")
    assert len(ids) == 4
    assert ids[0] == tokenizer.cls_token_id
    assert ids[-1] == tokenizer.sep_token_id


def test_scratch_tokenizer_wraps_reviews_like_distilbert() -> None:
    """``[CLS] ... [SEP]`` and a lower-cased word vocabulary, all offline."""
    tokenizer = _tokenizer()
    ids = tokenizer.encode("Good VACUUM")
    assert ids[0] == tokenizer.cls_token_id
    assert ids[-1] == tokenizer.sep_token_id
    assert tokenizer.vocab_size >= len(SPECIAL_TOKENS)
    assert tokenizer.pad_token_id == PAD_ID
    assert tokenizer.unk_token_id == UNK_ID
    decoded_ids = tokenizer.encode("unseenword")
    assert decoded_ids[1] == UNK_ID


def test_scratch_tokenizer_round_trips_without_the_network(tmp_path: Path) -> None:
    """The saved tokenizer reloads offline, so a checkpoint directory is enough."""
    tokenizer = _tokenizer()
    tokenizer.save_pretrained(tmp_path)
    restored = PreTrainedTokenizerFast.from_pretrained(tmp_path, local_files_only=True)
    text = "great price and quiet"
    assert restored.encode(text) == tokenizer.encode(text)


def test_scratch_tokenizer_rejects_an_impossible_vocabulary() -> None:
    """A vocabulary smaller than the special tokens cannot be built."""
    with pytest.raises(ValueError, match="special tokens"):
        build_scratch_tokenizer(TEXTS, max_length=8, vocab_size=3)


def test_sampler_covers_every_row_exactly_once() -> None:
    """Length pooling must not drop or duplicate a review."""
    lengths = [((index * 7) % 23) + 1 for index in range(200)]
    sampler = LengthBucketedBatchSampler(lengths, batch_size=8, seed=1)
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == list(range(200))
    assert len(batches) == len(sampler) == pytest.approx(25)


def test_sampler_is_deterministic_for_a_seed_and_epoch() -> None:
    """Same seed and epoch give identical batches; a different epoch does not."""
    lengths = [((index * 5) % 31) + 1 for index in range(400)]
    first = LengthBucketedBatchSampler(lengths, batch_size=8, seed=3)
    second = LengthBucketedBatchSampler(lengths, batch_size=8, seed=3)
    assert list(first) == list(second)

    first.set_epoch(1)
    epoch_one = list(first)
    assert epoch_one != list(second)
    assert sorted(index for batch in epoch_one for index in batch) == list(range(400))


def test_sampler_groups_similar_lengths_when_not_shuffling() -> None:
    """Unshuffled batches follow length order, which is what the eval pass uses."""
    lengths = [40, 3, 12, 3, 25, 40, 1, 12]
    sampler = LengthBucketedBatchSampler(lengths, batch_size=3, seed=0, shuffle=False)
    flattened = [index for batch in sampler for index in batch]
    observed = [lengths[index] for index in flattened]
    assert observed == sorted(lengths)


def test_sampler_rejects_invalid_settings() -> None:
    """A zero batch size or a zero-length review cannot be batched."""
    with pytest.raises(ValueError, match="batch_size"):
        LengthBucketedBatchSampler([1, 2], batch_size=0, seed=0)
    with pytest.raises(ValueError, match="bucket_multiplier"):
        LengthBucketedBatchSampler([1, 2], batch_size=2, seed=0, bucket_multiplier=0)
    with pytest.raises(ValueError, match="at least one token"):
        LengthBucketedBatchSampler([1, 0], batch_size=2, seed=0)


def test_sampler_seed_changes_the_batch_order() -> None:
    """Two seeds must not collapse onto the same permutation."""
    lengths = [random.Random(index).randint(1, 40) for index in range(120)]
    one = list(LengthBucketedBatchSampler(lengths, batch_size=4, seed=1))
    two = list(LengthBucketedBatchSampler(lengths, batch_size=4, seed=2))
    assert one != two


def test_collate_output_is_long_tensors() -> None:
    """Cross-entropy needs ``long`` labels and indices; ids must be integers too."""
    batch = collate_batch(
        [{"index": 0, "input_ids": [4, 5], "label": 2, "asin": "a"}], pad_id=PAD_ID
    )
    assert batch.input_ids.dtype == torch.long
    assert batch.attention_mask.dtype == torch.long
    assert batch.labels.dtype == torch.long
    assert batch.indices.dtype == torch.long

"""Datasets, dynamic padding and length-bucketed batching for the trainers.

Three pieces keep the training loop architecture-agnostic:

* :class:`TextEncoder` turns one review into token ids through **either** a
  Hugging Face tokenizer (DistilBERT) or the training-fitted
  :class:`~munasib.review_intel.sentiment.models.Vocabulary` (recurrent model).
  Exactly one encoder is set, so a model can never be fed another model's ids.
* :class:`SentimentDataset` exposes one review per index and defers encoding to
  ``__getitem__``, which keeps the whole split out of memory as token ids.
* :func:`collate_batch` pads **to the longest review in the batch**, not to the
  configured maximum. With a median review of 16 words that is most of the
  speed-up; the configured ``max_length_tokens`` remains the hard cap.

:class:`LengthBucketedBatchSampler` groups reviews of similar length so that
dynamic padding actually pays off, and reshuffles deterministically per epoch:
review lengths are sorted, cut into pools of ``batch_size * bucket_multiplier``
rows, and each pool is shuffled internally before the pool order is shuffled.
Every review appears exactly once per epoch, and the same ``(seed, epoch)``
always yields the same batches - no global RNG is touched.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from munasib.core.exceptions import DataError
from munasib.review_intel.sentiment.models import (
    PAD_ID,
    PAD_TOKEN,
    UNK_TOKEN,
    Vocabulary,
    tokenize,
)

#: Special tokens the scratch tokenizer reserves before any review token. The
#: order is fixed so ids are stable between runs and tests.
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, "[CLS]", "[SEP]", "[MASK]")


def build_scratch_tokenizer(
    texts: Sequence[str],
    *,
    max_length: int,
    vocab_size: int = 30_000,
    min_freq: int = 1,
) -> PreTrainedTokenizerBase:
    """Build a word-level tokenizer from **training** texts, with no download.

    The tokenizer mirrors what ``distilbert-base-uncased`` does to a review -
    lower-cased whitespace tokens wrapped in ``[CLS]``/``[SEP]`` - so the smoke
    run exercises the same code path as the real fine-tune while staying
    completely offline.

    Args:
        texts: Training review texts (never evaluation rows).
        max_length: ``model_max_length`` recorded on the tokenizer.
        vocab_size: Maximum vocabulary size, including the special tokens.
        min_freq: Minimum occurrences for a review token to be indexed.

    Returns:
        A ``PreTrainedTokenizerFast`` usable with ``save_pretrained`` and with
        ``from_pretrained(..., local_files_only=True)``.

    Raises:
        ValueError: If ``vocab_size`` is smaller than :data:`SPECIAL_TOKENS` or
            ``min_freq`` is below one.
    """
    if vocab_size < len(SPECIAL_TOKENS):
        raise ValueError(
            f"vocab_size must be >= {len(SPECIAL_TOKENS)} to hold the special tokens, "
            f"got {vocab_size}"
        )
    if min_freq < 1:
        raise ValueError(f"min_freq must be >= 1, got {min_freq}")

    counts: dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if len(vocabulary) >= vocab_size or count < min_freq:
            break
        vocabulary[token] = len(vocabulary)

    backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token=UNK_TOKEN))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=max_length,
    )


def load_pretrained_tokenizer(
    model_name: str, *, local_files_only: bool = False
) -> PreTrainedTokenizerBase:
    """Load the tokenizer belonging to a pretrained transformer.

    Args:
        model_name: Hugging Face model id or a directory saved locally.
        local_files_only: When ``True`` the Hub is never contacted.

    Returns:
        The tokenizer configured with the model's own defaults.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    return cast(PreTrainedTokenizerBase, tokenizer)


@dataclass(frozen=True)
class TextEncoder:
    """Encode reviews with exactly one of a tokenizer or a vocabulary.

    Attributes:
        max_length: Hard cap on the token count (``max_length_tokens``).
        tokenizer: Transformer tokenizer, when training a transformer.
        vocabulary: Training-fitted vocabulary, when training the RNN.
    """

    max_length: int
    tokenizer: PreTrainedTokenizerBase | None = None
    vocabulary: Vocabulary | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous or empty encoders.

        Raises:
            DataError: If neither or both encoders are supplied.
        """
        if (self.tokenizer is None) == (self.vocabulary is None):
            raise DataError("TextEncoder needs exactly one of tokenizer or vocabulary")
        if self.max_length < 1:
            raise DataError(f"max_length must be >= 1, got {self.max_length}")

    @property
    def pad_id(self) -> int:
        """Id used for padding by the active encoder."""
        if self.vocabulary is not None:
            return PAD_ID
        assert self.tokenizer is not None  # guaranteed by __post_init__
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise DataError(f"tokenizer {self.tokenizer.name_or_path!r} has no pad token")
        return int(pad_id)

    def encode(self, text: str) -> list[int]:
        """Encode one review into (at most ``max_length``) token ids.

        Args:
            text: Review text.

        Returns:
            Token ids; the transformer path includes ``[CLS]``/``[SEP]``.
        """
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            return cast(list[int], list(encoded["input_ids"]))
        assert self.vocabulary is not None  # guaranteed by __post_init__
        return self.vocabulary.encode(text)[: self.max_length]


class SentimentDataset(Dataset[dict[str, Any]]):
    """One review per index: token ids, weak label and product id.

    Encoding happens in ``__getitem__`` so the collate function sees real token
    lengths and can pad to the batch maximum.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        asins: Sequence[str],
        encoder: TextEncoder,
    ) -> None:
        """Store the inputs after checking they line up.

        Args:
            texts: Review texts.
            labels: Weak label code per review.
            asins: Product id per review, carried through for calibration.
            encoder: Tokenizer or vocabulary wrapper.

        Raises:
            DataError: If the three sequences do not have equal length.
        """
        if not (len(texts) == len(labels) == len(asins)):
            raise DataError(
                "texts, labels and asins must have the same length, got "
                f"{len(texts)}, {len(labels)}, {len(asins)}"
            )
        self.texts = list(texts)
        self.labels = list(labels)
        self.asins = list(asins)
        self.encoder = encoder
        self._lengths: list[int] | None = None

    def __len__(self) -> int:
        """Number of reviews."""
        return len(self.texts)

    def lengths(self) -> list[int]:
        """Token count per review, encoded once and cached.

        The batch sampler needs lengths before any batch exists; computing them
        here keeps that cost out of the epoch loop.

        Returns:
            Token count per row, in dataset order.
        """
        if self._lengths is None:
            self._lengths = [len(self.encoder.encode(text)) for text in self.texts]
        return self._lengths

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Encode one review.

        Args:
            index: Position in the split.

        Returns:
            Mapping with ``index`` (position in the split), ``input_ids`` (list
            of ints), ``label`` and ``asin``.
        """
        return {
            "index": index,
            "input_ids": self.encoder.encode(self.texts[index]),
            "label": int(self.labels[index]),
            "asin": self.asins[index],
        }


@dataclass(frozen=True)
class Batch:
    """One padded batch ready for either model.

    Attributes:
        indices: ``(batch,)`` positions in the dataset, so predictions can be
            written back in dataset order even though batches are length-sorted.
        input_ids: ``(batch, longest review in batch)`` ids.
        attention_mask: ``(batch, sequence)`` 1 for real tokens, 0 for padding.
        labels: ``(batch,)`` weak label codes.
        asins: Product id per row, in batch order.
    """

    indices: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    asins: tuple[str, ...]


def collate_batch(
    batch: Sequence[Mapping[str, Any]], *, pad_id: int = PAD_ID
) -> Batch:
    """Pad a batch to its own longest review and stack it into tensors.

    Args:
        batch: Items produced by :class:`SentimentDataset`.
        pad_id: Id written into padding positions.

    Returns:
        A :class:`Batch` with ``long`` tensors.

    Raises:
        DataError: If the batch is empty.
    """
    if not batch:
        raise DataError("cannot collate an empty batch")
    lengths = [len(item["input_ids"]) for item in batch]
    width = max(lengths)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, item in enumerate(batch):
        ids = list(item["input_ids"])
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
    indices = torch.tensor([int(item["index"]) for item in batch], dtype=torch.long)
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    asins = tuple(str(item["asin"]) for item in batch)
    return Batch(
        indices=indices,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        asins=asins,
    )


class LengthBucketedBatchSampler(Sampler[list[int]]):
    """Yield batches of similarly sized reviews, deterministically per epoch.

    Reviews are sorted by ``(length, index)`` and cut into pools of
    ``batch_size * bucket_multiplier`` rows. For a shuffled epoch each pool is
    shuffled internally and the pool order is shuffled, both from a private
    ``random.Random`` seeded with ``seed`` and the epoch number. Sorting alone
    would re-serve identical batches every epoch, and hashing the epoch into a
    global RNG would make runs unreproducible, hence the pool approach.

    Attributes:
        lengths: Token count per row, in dataset index order.
        batch_size: Rows per batch.
        seed: Base seed for the deterministic shuffle.
        shuffle: Whether to reshuffle pools per epoch (``False`` keeps length
            order, which the tests use to pin coverage exactly).
        bucket_multiplier: Pool size as a multiple of ``batch_size``.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        seed: int,
        shuffle: bool = True,
        bucket_multiplier: int = 50,
    ) -> None:
        """Record the row lengths and batching settings.

        Args:
            lengths: Token count per row, in dataset index order.
            batch_size: Rows per batch.
            seed: Base seed for the deterministic shuffle.
            shuffle: Whether to reshuffle pools each epoch.
            bucket_multiplier: Pool size as a multiple of ``batch_size``.

        Raises:
            ValueError: If ``batch_size`` or ``bucket_multiplier`` is below one,
                or a length is not positive.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if bucket_multiplier < 1:
            raise ValueError(f"bucket_multiplier must be >= 1, got {bucket_multiplier}")
        if any(length < 1 for length in lengths):
            raise ValueError("every review must have at least one token")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.bucket_multiplier = bucket_multiplier
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select which deterministic permutation the next iteration uses.

        Args:
            epoch: Zero-based epoch number.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Number of batches produced per epoch."""
        return math.ceil(len(self.lengths) / self.batch_size)

    def _batches(self) -> list[list[int]]:
        """Build this epoch's batch list.

        Returns:
            Batches of dataset indices, covering every index exactly once.
        """
        order = sorted(range(len(self.lengths)), key=lambda index: (self.lengths[index], index))
        pool_size = max(self.batch_size, self.batch_size * self.bucket_multiplier)
        pools = [order[start : start + pool_size] for start in range(0, len(order), pool_size)]
        if self.shuffle:
            rng = random.Random(self.seed * 1_000_003 + self.epoch)
            for pool in pools:
                rng.shuffle(pool)
            rng.shuffle(pools)
        batches: list[list[int]] = []
        for pool in pools:
            for start in range(0, len(pool), self.batch_size):
                batches.append(pool[start : start + self.batch_size])
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over this epoch's batches.

        Yields:
            Lists of dataset indices.
        """
        yield from self._batches()


__all__ = [
    "SPECIAL_TOKENS",
    "Batch",
    "LengthBucketedBatchSampler",
    "SentimentDataset",
    "TextEncoder",
    "build_scratch_tokenizer",
    "collate_batch",
    "load_pretrained_tokenizer",
]

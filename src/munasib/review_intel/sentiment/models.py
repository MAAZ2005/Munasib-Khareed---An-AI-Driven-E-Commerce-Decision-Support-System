"""PyTorch sentiment models: fine-tuned DistilBERT and a recurrent baseline.

Two architectures share one calling contract, ``model(input_ids=...,
attention_mask=...)`` returning per-review logits over the three weak labels:

* **DistilBERT** (:func:`load_transformer`) fine-tuned from
  ``distilbert-base-uncased``. The classification head is sized from
  :data:`munasib.core.constants.SENTIMENT_LABELS`, and ``id2label``/``label2id``
  are written into the model config so a reloaded checkpoint cannot be silently
  re-indexed (label order bugs are invisible until the confusion matrix looks
  strange).
* **BiGRU/BiLSTM** (:func:`build_rnn`) trained from scratch, with masked mean and
  max pooling: padding positions are excluded from both statistics, so appending
  pad tokens to a review cannot change its prediction.

:func:`build_tiny_transformer` builds the same DistilBERT head from a randomly
initialised miniature config. It never downloads anything, which is what makes
the CPU smoke run and the unit tests network-free.

Everything here is persistence-aware: a saved directory contains the weights,
the tokenizer or vocabulary needed to encode text, and a ``model_meta.json``
describing the architecture, so :func:`load_model` can rebuild the right object
with ``local_files_only=True`` on a machine with no network.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from munasib.core.config import RnnConfig
from munasib.core.constants import SENTIMENT_LABEL_NAMES, SENTIMENT_LABELS
from munasib.core.exceptions import DataError

#: Label order every model, dataset and metric uses (0 negative, 1 neutral,
#: 2 positive), derived from the project constants rather than restated.
NUM_LABELS = len(SENTIMENT_LABELS)
ID2LABEL: dict[int, str] = {label: SENTIMENT_LABEL_NAMES[label] for label in SENTIMENT_LABELS}
LABEL2ID: dict[str, int] = {name: label for label, name in ID2LABEL.items()}

#: Keyword arguments that pin the label schema on every transformer construction.
LABEL_KWARGS: dict[str, Any] = {
    "num_labels": NUM_LABELS,
    "id2label": ID2LABEL,
    "label2id": LABEL2ID,
}

#: Padding and unknown token strings shared by the recurrent vocabulary and the
#: scratch transformer tokenizer.
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_ID = 0
UNK_ID = 1

#: Architecture names accepted by :func:`save_model` / :func:`load_model`.
MODEL_ARCHITECTURES: tuple[str, ...] = ("transformer", "rnn")

#: File names inside a saved model directory.
META_FILENAME = "model_meta.json"
VOCAB_FILENAME = "vocab.json"
RNN_WEIGHTS_FILENAME = "model.pt"

#: Version tag so an incompatible on-disk layout is detected, not mis-read.
ARTIFACT_FORMAT_VERSION = 1


def tokenize(text: str) -> list[str]:
    """Split review text into the tokens the recurrent vocabulary indexes.

    Lower-casing and whitespace splitting are deliberately simple: a
    cross-entropy baseline should not hide behind a clever tokenizer, and a
    vocabulary fitted on training rows is only meaningful if encoding is a pure
    function of the text.

    Args:
        text: Review text.

    Returns:
        Lower-cased whitespace tokens (empty for blank text).
    """
    return text.lower().split()


@dataclass(frozen=True)
class Vocabulary:
    """Token-to-id table fitted on the **training** sample only.

    ``[PAD]`` is always id 0 and ``[UNK]`` always id 1, so padding and
    out-of-vocabulary tokens keep stable ids across fits.

    Attributes:
        tokens: Mapping from token string to id.
        min_freq: Minimum training frequency a token needed to be indexed.
        max_size: Vocabulary cap (including the two special tokens).
    """

    tokens: dict[str, int]
    min_freq: int
    max_size: int

    @classmethod
    def from_texts(cls, texts: Iterable[str], *, min_freq: int, max_size: int) -> Vocabulary:
        """Build a vocabulary from training texts.

        Tokens are kept when seen at least ``min_freq`` times, ordered by
        descending frequency and then alphabetically (so ties never depend on
        dictionary insertion order), and capped at ``max_size``.

        Args:
            texts: Training review texts, never evaluation rows.
            min_freq: Minimum number of occurrences a token needs.
            max_size: Maximum vocabulary size, including ``[PAD]`` and ``[UNK]``.

        Returns:
            The fitted vocabulary.

        Raises:
            ValueError: If ``min_freq`` is below one or ``max_size`` below two.
        """
        if min_freq < 1:
            raise ValueError(f"min_freq must be >= 1, got {min_freq}")
        if max_size < 2:
            raise ValueError(f"max_size must be >= 2 (padding + unknown), got {max_size}")

        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))

        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        tokens = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        for token, count in ordered:
            if len(tokens) >= max_size or count < min_freq:
                break
            tokens[token] = len(tokens)
        return cls(tokens=tokens, min_freq=min_freq, max_size=max_size)

    def __len__(self) -> int:
        """Number of indexed tokens, including ``[PAD]`` and ``[UNK]``."""
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        """Encode one review into token ids.

        Args:
            text: Review text.

        Returns:
            Ids in text order; out-of-vocabulary tokens map to ``[UNK]``. Blank
            text yields a single ``[UNK]`` so a sequence is never empty.
        """
        ids = [self.tokens.get(token, UNK_ID) for token in tokenize(text)]
        return ids or [UNK_ID]

    def to_dict(self) -> dict[str, object]:
        """Serialise the vocabulary for ``vocab.json``.

        Returns:
            Mapping with the token table and the fit settings that produced it.
        """
        return {"tokens": self.tokens, "min_freq": self.min_freq, "max_size": self.max_size}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Vocabulary:
        """Rebuild a vocabulary from :meth:`to_dict` output.

        Args:
            payload: Mapping written by :meth:`to_dict`.

        Returns:
            The restored vocabulary.

        Raises:
            DataError: If the payload is missing required keys.
        """
        try:
            tokens = {str(token): int(index) for token, index in payload["tokens"].items()}
            return cls(
                tokens=tokens,
                min_freq=int(payload["min_freq"]),
                max_size=int(payload["max_size"]),
            )
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise DataError(f"invalid vocabulary payload: {exc}") from exc


#: Fill value for masked-out positions before the max-pool. GRU/LSTM hidden
#: states are bounded by ``tanh`` (``[-1, 1]``), so any value below -1 is inert;
#: a bounded sentinel is used instead of ``-inf`` because extreme fills trip a
#: first-chance access violation inside the CPU max kernel under Windows.
MASKED_POOL_FILL = -1.0e4


class SentimentRNN(nn.Module):
    """Embedding -> recurrent layer -> masked mean+max pooling -> classifier.

    Both pooling statistics are computed under the attention mask, so padded
    positions contribute neither to the mean nor to the max; the concatenation
    keeps an average view of the review and its strongest single signal.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        cell: str,
        num_labels: int = NUM_LABELS,
        padding_idx: int = PAD_ID,
    ) -> None:
        """Build the recurrent classifier.

        Args:
            vocab_size: Embedding rows (vocabulary size).
            embedding_dim: Embedding width.
            hidden_size: Hidden units per direction.
            num_layers: Stacked recurrent layers.
            bidirectional: Whether to run the recurrence in both directions.
            dropout: Dropout applied after the embedding and before the head.
            cell: ``"gru"`` or ``"lstm"``.
            num_labels: Output classes (three for the weak sentiment labels).
            padding_idx: Embedding row kept at zero for padding.

        Raises:
            ValueError: If ``cell`` is not ``"gru"`` or ``"lstm"``.
        """
        super().__init__()
        if cell not in {"gru", "lstm"}:
            raise ValueError(f"cell must be 'gru' or 'lstm', got {cell!r}")
        self.vocab_size = vocab_size
        self.cell = cell
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        recurrent_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.recurrent = recurrent_cls(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        directions = 2 if bidirectional else 1
        pooled_width = 2 * hidden_size * directions
        self.pooled_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(pooled_width, num_labels)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Score each review.

        The sequence is packed before the recurrence, so a bidirectional model
        never runs its backward direction over padding; that is what makes
        trailing pad tokens provably inert rather than approximately ignored.

        Args:
            input_ids: ``(batch, sequence)`` token ids, zero-padded.
            attention_mask: ``(batch, sequence)`` 1 for real tokens and 0 for
                padding. ``None`` treats every position as real.

        Returns:
            ``(batch, num_labels)`` logits.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embedded = self.embedding_dropout(self.embedding(input_ids))
        mask = attention_mask.to(dtype=embedded.dtype).unsqueeze(-1)
        lengths = attention_mask.sum(dim=1).to(torch.long).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.recurrent(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=embedded.size(1)
        )
        lengths_float = mask.sum(dim=1).clamp(min=1.0)
        pooled_mean = (output * mask).sum(dim=1) / lengths_float
        masked = output.masked_fill(mask == 0, MASKED_POOL_FILL)
        pooled_max = masked.max(dim=1).values
        # A fully padded row would otherwise pool the fill value; clamp to zero.
        pooled_max = torch.where(mask.sum(dim=1) > 0, pooled_max, torch.zeros_like(pooled_max))
        pooled = torch.cat([pooled_mean, pooled_max], dim=1)
        return cast(Tensor, self.classifier(self.pooled_dropout(pooled)))


def build_rnn(config: RnnConfig, *, vocab_size: int) -> SentimentRNN:
    """Build the recurrent classifier from validated settings.

    Args:
        config: ``review_intel.sentiment.rnn`` settings.
        vocab_size: Size of the fitted training vocabulary.

    Returns:
        An untrained :class:`SentimentRNN`.
    """
    return SentimentRNN(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        bidirectional=config.bidirectional,
        dropout=config.dropout,
        cell=config.cell,
    )


def build_tiny_transformer(
    *,
    vocab_size: int,
    max_length: int,
    dim: int = 64,
    hidden_dim: int = 128,
    n_layers: int = 2,
    n_heads: int = 2,
    dropout: float = 0.1,
) -> DistilBertForSequenceClassification:
    """Build a randomly initialised miniature DistilBERT, with no download.

    Used by the smoke run and the tests so that the full training path can be
    exercised on CPU without touching the network.

    Args:
        vocab_size: Embedding rows (size of the scratch tokenizer vocabulary).
        max_length: Maximum sequence length the position table must cover.
        dim: Hidden size of the miniature model.
        hidden_dim: Feed-forward size.
        n_layers: Transformer layers.
        n_heads: Attention heads.
        dropout: Dropout probability.

    Returns:
        An untrained DistilBERT classifier with the project's label schema.
    """
    config = DistilBertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_length,
        dim=dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        attention_dropout=dropout,
        **LABEL_KWARGS,
    )
    model = AutoModelForSequenceClassification.from_config(config)
    return cast(DistilBertForSequenceClassification, model)


def load_transformer(
    model_name: str, *, local_files_only: bool = False
) -> DistilBertForSequenceClassification:
    """Load a DistilBERT classifier with the project's three-label head.

    Args:
        model_name: Hugging Face model id or a local directory saved with
            ``save_pretrained``.
        local_files_only: When ``True`` the Hub is never contacted; a directory
            saved by :func:`save_model` loads fully offline.

    Returns:
        The classifier, ready for fine-tuning or inference.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, local_files_only=local_files_only, **LABEL_KWARGS
    )
    return cast(DistilBertForSequenceClassification, model)


def forward_logits(model: nn.Module, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """Return logits from either architecture through one calling contract.

    ``DistilBertForSequenceClassification`` returns a ``ModelOutput`` while
    :class:`SentimentRNN` returns a tensor; this unwraps the former so the
    training loop needs no architecture branches.

    Args:
        model: Either architecture.
        input_ids: Padded token ids.
        attention_mask: Padding mask.

    Returns:
        ``(batch, num_labels)`` logits.
    """
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits if hasattr(output, "logits") else output
    return cast(Tensor, logits)


@dataclass(frozen=True)
class ModelMeta:
    """Description of a saved model directory.

    Attributes:
        architecture: ``"transformer"`` or ``"rnn"``.
        max_length: Token budget the model was trained with.
        class_weighting: Loss weighting used during training.
        base_model: Hub id the transformer was fine-tuned from, when known.
        rnn: Recurrent hyper-parameters needed to rebuild the module.
        format_version: On-disk layout version.
        num_labels: Number of output classes.
        id2label: Label order stored with the artifact.
    """

    architecture: str
    max_length: int
    class_weighting: str
    base_model: str | None = None
    rnn: dict[str, Any] | None = None
    format_version: int = ARTIFACT_FORMAT_VERSION
    num_labels: int = NUM_LABELS
    id2label: dict[str, str] = field(
        default_factory=lambda: {str(label): name for label, name in ID2LABEL.items()}
    )

    def __post_init__(self) -> None:
        """Validate the architecture name.

        Raises:
            DataError: If the architecture is unknown.
        """
        if self.architecture not in MODEL_ARCHITECTURES:
            raise DataError(
                f"architecture must be one of {list(MODEL_ARCHITECTURES)}, got "
                f"{self.architecture!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the metadata for ``model_meta.json``.

        Returns:
            Plain-Python mapping.
        """
        return {
            "architecture": self.architecture,
            "max_length": self.max_length,
            "class_weighting": self.class_weighting,
            "base_model": self.base_model,
            "rnn": self.rnn,
            "format_version": self.format_version,
            "num_labels": self.num_labels,
            "id2label": self.id2label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelMeta:
        """Rebuild metadata from :meth:`to_dict` output.

        Args:
            payload: Mapping read from ``model_meta.json``.

        Returns:
            The restored metadata.

        Raises:
            DataError: If required keys are missing or mistyped.
        """
        try:
            return cls(
                architecture=str(payload["architecture"]),
                max_length=int(payload["max_length"]),
                class_weighting=str(payload["class_weighting"]),
                base_model=(
                    None if payload.get("base_model") is None else str(payload["base_model"])
                ),
                rnn=None if payload.get("rnn") is None else dict(payload["rnn"]),
                format_version=int(payload.get("format_version", ARTIFACT_FORMAT_VERSION)),
                num_labels=int(payload.get("num_labels", NUM_LABELS)),
                id2label={
                    str(key): str(value)
                    for key, value in dict(payload.get("id2label") or {}).items()
                }
                or {str(k): v for k, v in ID2LABEL.items()},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataError(f"invalid model metadata: {exc}") from exc


@dataclass(frozen=True)
class SavedSentimentModel:
    """A reloaded model plus whatever it needs to encode text.

    Attributes:
        meta: Metadata read from the directory.
        model: The restored module, in evaluation mode on the CPU.
        tokenizer: Transformer tokenizer, or ``None`` for the recurrent model.
        vocabulary: Recurrent vocabulary, or ``None`` for the transformer.
    """

    meta: ModelMeta
    model: nn.Module
    tokenizer: PreTrainedTokenizerBase | None = None
    vocabulary: Vocabulary | None = None


def _write_meta(output_dir: Path, meta: ModelMeta) -> Path:
    """Write ``model_meta.json``.

    Args:
        output_dir: Model directory (assumed to exist).
        meta: Metadata to serialise.

    Returns:
        The written path.
    """
    path = output_dir / META_FILENAME
    path.write_text(
        json.dumps(meta.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def save_model(
    model: nn.Module,
    output_dir: Path,
    *,
    architecture: str,
    max_length: int,
    class_weighting: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
    vocabulary: Vocabulary | None = None,
    rnn_config: RnnConfig | None = None,
    base_model: str | None = None,
) -> ModelMeta:
    """Save weights, encoder and metadata into one reloadable directory.

    Args:
        model: Trained module.
        output_dir: Destination directory (created if missing).
        architecture: ``"transformer"`` or ``"rnn"``.
        max_length: Token budget used during training.
        class_weighting: Loss weighting used during training.
        tokenizer: Required for the transformer; saved next to the weights.
        vocabulary: Required for the recurrent model.
        rnn_config: Required for the recurrent model, so it can be rebuilt.
        base_model: Hub id the transformer started from.

    Returns:
        The metadata that was written.

    Raises:
        DataError: If the architecture is unknown or a required companion is
            missing.
    """
    if architecture not in MODEL_ARCHITECTURES:
        raise DataError(
            f"architecture must be one of {list(MODEL_ARCHITECTURES)}, got {architecture!r}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = ModelMeta(
        architecture=architecture,
        max_length=max_length,
        class_weighting=class_weighting,
        base_model=base_model,
        rnn=None if rnn_config is None else rnn_config.model_dump(mode="json"),
    )
    if architecture == "transformer":
        if tokenizer is None:
            raise DataError("saving a transformer requires its tokenizer")
        cast(PreTrainedModel, model).save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        if vocabulary is None or rnn_config is None:
            raise DataError("saving the recurrent model requires its vocabulary and RnnConfig")
        state = cast(SentimentRNN, model).state_dict()
        torch.save(state, output_dir / RNN_WEIGHTS_FILENAME)
        (output_dir / VOCAB_FILENAME).write_text(
            json.dumps(vocabulary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _write_meta(output_dir, meta)
    return meta


def load_model(output_dir: Path) -> SavedSentimentModel:
    """Reload a directory written by :func:`save_model`, without any network.

    Args:
        output_dir: Directory containing ``model_meta.json`` and the weights.

    Returns:
        The restored model in evaluation mode, plus its tokenizer or vocabulary.

    Raises:
        DataError: If the directory or its metadata is missing, unreadable, or
            describes an unsupported architecture.
    """
    meta_path = output_dir / META_FILENAME
    if not meta_path.is_file():
        raise DataError(f"no {META_FILENAME} in {output_dir}")
    meta = ModelMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
    model: nn.Module

    if meta.architecture == "transformer":
        model = load_transformer(str(output_dir), local_files_only=True)
        tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(str(output_dir), local_files_only=True),
        )
        model.eval()
        return SavedSentimentModel(meta=meta, model=model, tokenizer=tokenizer)

    vocab_path = output_dir / VOCAB_FILENAME
    if not vocab_path.is_file():
        raise DataError(f"no {VOCAB_FILENAME} in {output_dir}")
    vocabulary = Vocabulary.from_dict(json.loads(vocab_path.read_text(encoding="utf-8")))
    if meta.rnn is None:
        raise DataError(f"{META_FILENAME} has no rnn settings for a recurrent model")
    rnn_config = RnnConfig.model_validate(
        {key: value for key, value in meta.rnn.items() if key in RnnConfig.model_fields}
    )
    model = build_rnn(rnn_config, vocab_size=len(vocabulary))
    state = torch.load(output_dir / RNN_WEIGHTS_FILENAME, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return SavedSentimentModel(meta=meta, model=model, vocabulary=vocabulary)


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "ID2LABEL",
    "LABEL2ID",
    "LABEL_KWARGS",
    "META_FILENAME",
    "MODEL_ARCHITECTURES",
    "NUM_LABELS",
    "PAD_ID",
    "PAD_TOKEN",
    "RNN_WEIGHTS_FILENAME",
    "UNK_ID",
    "UNK_TOKEN",
    "VOCAB_FILENAME",
    "ModelMeta",
    "SavedSentimentModel",
    "SentimentRNN",
    "Vocabulary",
    "build_rnn",
    "build_tiny_transformer",
    "forward_logits",
    "load_model",
    "load_transformer",
    "save_model",
    "tokenize",
]

"""Reusable inference wrapper around a saved sentiment checkpoint.

:class:`SentimentPredictor` is the only place that turns review text into
probabilities after training, so the training script, the batch scorer and the
tests share one code path:

* The checkpoint is loaded through :func:`~munasib.review_intel.sentiment.models.load_model`,
  which passes ``local_files_only=True``. A checkpoint therefore loads on a
  machine with no network, and a typo'd path fails instead of silently pulling a
  Hub model.
* Encoding uses the checkpoint's **own** tokenizer or vocabulary, and
  ``model_meta.json`` supplies the token budget. A DistilBERT checkpoint can
  never be fed the recurrent vocabulary, and a model trained at 128 tokens is
  not scored at 512 by accident.
* The batch order is length-bucketed (the same
  :class:`~munasib.review_intel.sentiment.data.LengthBucketedBatchSampler` the
  validators use) but the **output order is the input order**: every batch
  carries the dataset index it came from, and rows are written back by index.
* Rows with no text get a row of ``nan`` and ``has_prediction=False``. They are
  never assigned a probability, because an invented score would enter the
  product-level aggregates as if it were evidence.

The hash helpers in this module are what make a prediction file traceable: the
sidecar written by ``scripts/predict_reviews.py`` records the SHA-256 of the
weights and of ``model_meta.json``, and ``scripts/ingest_predictions.py`` refuses
to load predictions whose checkpoint no longer matches.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from munasib.core.config import get_settings
from munasib.core.exceptions import DataError
from munasib.review_intel.sentiment.data import (
    LengthBucketedBatchSampler,
    SentimentDataset,
    TextEncoder,
    collate_batch,
)
from munasib.review_intel.sentiment.models import (
    META_FILENAME,
    NUM_LABELS,
    RNN_WEIGHTS_FILENAME,
    ModelMeta,
    SavedSentimentModel,
    forward_logits,
    load_model,
)
from munasib.review_intel.sentiment.train import DEVICE_CHOICES, resolve_device

#: Weights file ``save_pretrained`` writes for a transformer checkpoint.
TRANSFORMER_WEIGHTS_FILENAME = "model.safetensors"

#: Read size for the streaming SHA-256 helpers (1 MiB).
HASH_CHUNK_BYTES = 1 << 20

#: Suffix of the sidecar JSON that travels next to a prediction parquet.
SIDECAR_SUFFIX = ".meta.json"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of one file.

    Args:
        path: File to hash.

    Returns:
        Lower-case hex digest.

    Raises:
        DataError: If the file does not exist.
    """
    if not path.is_file():
        raise DataError(f"cannot hash missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(output_path: Path) -> Path:
    """Path of the sidecar JSON belonging to a prediction parquet.

    Args:
        output_path: The prediction parquet path.

    Returns:
        The same directory and stem with :data:`SIDECAR_SUFFIX` appended, e.g.
        ``review_predictions.parquet`` -> ``review_predictions.meta.json``.
    """
    return output_path.with_name(output_path.stem + SIDECAR_SUFFIX)


@dataclass(frozen=True)
class CheckpointIdentity:
    """What a prediction file was produced with, recorded in its sidecar.

    Attributes:
        path: Directory the checkpoint was loaded from.
        architecture: ``"transformer"`` or ``"rnn"``.
        max_length: Token budget read from ``model_meta.json``.
        class_weighting: Loss weighting the model was trained with.
        model_meta: Parsed ``model_meta.json`` (recorded verbatim for auditing).
        model_meta_sha256: SHA-256 of ``model_meta.json``.
        weights_file: Name of the weights file that was hashed.
        weights_sha256: SHA-256 of the weights file.
        weights_bytes: Size of the weights file in bytes.
    """

    path: str
    architecture: str
    max_length: int
    class_weighting: str
    model_meta: dict[str, Any]
    model_meta_sha256: str
    weights_file: str
    weights_sha256: str
    weights_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise the identity for the sidecar JSON.

        Returns:
            Plain-Python mapping.
        """
        return {
            "path": self.path,
            "architecture": self.architecture,
            "max_length": self.max_length,
            "class_weighting": self.class_weighting,
            "model_meta": self.model_meta,
            "model_meta_sha256": self.model_meta_sha256,
            "weights_file": self.weights_file,
            "weights_sha256": self.weights_sha256,
            "weights_bytes": self.weights_bytes,
        }


def describe_checkpoint(checkpoint: Path) -> CheckpointIdentity:
    """Hash and describe a checkpoint directory.

    Args:
        checkpoint: Directory written by :func:`~munasib.review_intel.sentiment.models.save_model`.

    Returns:
        The architecture, token budget, metadata and file hashes.

    Raises:
        DataError: If the directory, ``model_meta.json`` or the weights file is
            missing, or the metadata is unreadable.
    """
    meta_path = checkpoint / META_FILENAME
    if not meta_path.is_file():
        raise DataError(f"no {META_FILENAME} in checkpoint directory: {checkpoint}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = ModelMeta.from_dict(payload)

    for filename in (TRANSFORMER_WEIGHTS_FILENAME, RNN_WEIGHTS_FILENAME):
        weights_path = checkpoint / filename
        if weights_path.is_file():
            break
    else:
        raise DataError(
            f"checkpoint {checkpoint} holds neither {TRANSFORMER_WEIGHTS_FILENAME} "
            f"nor {RNN_WEIGHTS_FILENAME}"
        )

    return CheckpointIdentity(
        path=str(checkpoint),
        architecture=meta.architecture,
        max_length=meta.max_length,
        class_weighting=meta.class_weighting,
        model_meta=dict(payload),
        model_meta_sha256=file_sha256(meta_path),
        weights_file=filename,
        weights_sha256=file_sha256(weights_path),
        weights_bytes=weights_path.stat().st_size,
    )


@dataclass(frozen=True)
class PredictionResult:
    """Per-row predictions, aligned with the input order.

    Attributes:
        probabilities: ``(n, 3)`` float32 matrix in
            :data:`~munasib.core.constants.SENTIMENT_LABELS` order. Rows whose
            ``has_prediction`` entry is ``False`` are all ``nan``.
        labels: ``(n,)`` int64 weak-label codes (argmax); ``-1`` where there was
            no text, so it cannot be mistaken for the negative class.
        has_prediction: ``(n,)`` bool mask - ``False`` exactly for empty or
            missing text.
    """

    probabilities: np.ndarray
    labels: np.ndarray
    has_prediction: np.ndarray

    @property
    def rows(self) -> int:
        """Number of scored rows (including rows without text)."""
        return int(self.probabilities.shape[0])

    @property
    def predicted_rows(self) -> int:
        """Number of rows that produced a probability row."""
        return int(self.has_prediction.sum())


def has_text(value: object) -> bool:
    """Decide whether a cell holds usable review text.

    Missing values must be passed as ``None`` (the batch scorer normalises
    pandas nulls before calling). A float ``nan`` is also treated as missing;
    whitespace-only strings count as empty, because nothing can be scored from
    them and an empty review is not evidence of any sentiment.

    Args:
        value: One review text cell.

    Returns:
        ``True`` when the cell can be encoded.
    """
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip() != ""


class SentimentPredictor:
    """Load one checkpoint and score review texts with it.

    Attributes:
        checkpoint: Directory the checkpoint was loaded from.
        saved: The reloaded model plus its tokenizer/vocabulary.
        encoder: Encoder built from the checkpoint's own artifacts.
        device: Torch device the model is moved to.
        batch_size: Rows per forward pass.
    """

    def __init__(
        self,
        checkpoint: Path,
        saved: SavedSentimentModel,
        encoder: TextEncoder,
        *,
        device: torch.device,
        batch_size: int,
    ) -> None:
        """Wrap a loaded checkpoint (use :meth:`from_checkpoint`).

        Args:
            checkpoint: Directory the checkpoint was loaded from.
            saved: Result of :func:`~munasib.review_intel.sentiment.models.load_model`.
            encoder: Encoder using the same tokenizer or vocabulary.
            device: Device to score on.
            batch_size: Rows per forward pass.

        Raises:
            ValueError: If ``batch_size`` is not positive.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.checkpoint = checkpoint
        self.saved = saved
        self.encoder = encoder
        self.device = device
        self.batch_size = batch_size
        self._model: nn.Module = saved.model
        self._model.eval()
        self._model.to(device)

    @property
    def meta(self) -> ModelMeta:
        """Metadata of the loaded checkpoint."""
        return self.saved.meta

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        device: str = "auto",
        batch_size: int | None = None,
    ) -> SentimentPredictor:
        """Load a saved checkpoint for inference, fully offline.

        Args:
            checkpoint: Directory written by ``save_model`` (``best/``).
            device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
            batch_size: Rows per forward pass. ``None`` reads
                ``review_intel.sentiment.prediction.batch_size`` from the
                project configuration.

        Returns:
            A predictor in evaluation mode on the resolved device.

        Raises:
            DataError: If the checkpoint is missing or unreadable, or ``device`` is
                unknown or unavailable.
        """
        resolved = resolve_device(device)
        saved = load_model(Path(checkpoint))
        encoder = TextEncoder(
            max_length=saved.meta.max_length,
            tokenizer=saved.tokenizer,
            vocabulary=saved.vocabulary,
        )
        if batch_size is None:
            batch_size = get_settings().review_intel.sentiment.prediction.batch_size
        return cls(
            Path(checkpoint),
            saved,
            encoder,
            device=resolved,
            batch_size=batch_size,
        )

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[object]) -> PredictionResult:
        """Score a sequence of review texts.

        Length bucketing only changes which rows travel together; padded
        positions are excluded by both architectures, so the probabilities do
        not depend on the batch size. Rows are written back by dataset index, so
        the output order always equals the input order.

        Args:
            texts: Review texts; ``None`` (or blank text) marks a row without
                usable text.

        Returns:
            A :class:`PredictionResult` with ``(n, 3)`` probabilities, argmax
            labels and the ``has_prediction`` mask.
        """
        items = list(texts)
        mask = np.fromiter((has_text(item) for item in items), dtype=bool, count=len(items))
        probabilities = np.full((len(items), NUM_LABELS), np.nan, dtype=np.float32)
        labels = np.full(len(items), -1, dtype=np.int64)
        positions = np.flatnonzero(mask)
        if positions.size == 0:
            return PredictionResult(probabilities, labels, mask)

        valid_texts = [str(items[index]) for index in positions.tolist()]
        dataset = SentimentDataset(
            texts=valid_texts,
            labels=[0] * len(valid_texts),
            asins=[""] * len(valid_texts),
            encoder=self.encoder,
        )
        sampler = LengthBucketedBatchSampler(
            dataset.lengths(), batch_size=self.batch_size, seed=0, shuffle=False
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=partial(collate_batch, pad_id=self.encoder.pad_id),
            num_workers=0,
        )

        self._model.eval()
        for batch in loader:
            logits = forward_logits(
                self._model,
                batch.input_ids.to(self.device),
                batch.attention_mask.to(self.device),
            )
            scores = torch.softmax(logits, dim=-1).cpu().to(torch.float32).numpy()
            rows = positions[batch.indices.numpy()]
            probabilities[rows] = scores

        labels[positions] = probabilities[positions].argmax(axis=1).astype(np.int64)
        return PredictionResult(probabilities, labels, mask)

    def predict_labels(self, texts: Sequence[object]) -> list[int | None]:
        """Score texts and return one weak-label code per input row.

        Args:
            texts: Review texts; ``None`` (or blank text) yields ``None``.

        Returns:
            Label codes in input order, with ``None`` exactly where the row had
            no usable text.
        """
        result = self.predict_proba(texts)
        return [
            int(result.labels[index]) if result.has_prediction[index] else None
            for index in range(result.rows)
        ]


__all__ = [
    "DEVICE_CHOICES",
    "HASH_CHUNK_BYTES",
    "SIDECAR_SUFFIX",
    "TRANSFORMER_WEIGHTS_FILENAME",
    "CheckpointIdentity",
    "PredictionResult",
    "SentimentPredictor",
    "describe_checkpoint",
    "file_sha256",
    "has_text",
    "sidecar_path",
]

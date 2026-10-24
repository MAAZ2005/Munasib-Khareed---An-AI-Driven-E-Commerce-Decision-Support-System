"""Training loop for the neural sentiment models.

The loop is shared by both architectures: it only ever calls
:func:`~munasib.review_intel.sentiment.models.forward_logits`, so DistilBERT and
the BiGRU/BiLSTM differ in construction, not in how they are trained.

What the loop guarantees, and why:

* **Deterministic seeding** of ``random``, ``numpy``, ``torch`` and the CUDA
  generator, plus a DataLoader generator and worker init hook. Two runs with the
  same seed and settings therefore produce the same batches and the same
  initial weights.
* **Selection on macro-F1, never accuracy.** Positives are ~83% of the data, so
  accuracy prefers the model that ignores neutrals; macro-F1 cannot.
* **A checkpoint every epoch** (``best/`` and ``last/``), so a disconnected
  Colab session can be resumed with ``--resume`` instead of starting over.
  ``--resume`` restores weights, optimizer, scheduler and the completed epoch
  count; it does not replay the RNG stream, so a resumed run is not bit-identical
  to an uninterrupted one.
* **The final report always uses the full evaluation split**, even when
  ``--val-subset`` was used for per-epoch selection. The report states which was
  used, because a subset-based number would flatter or penalise the model
  depending on the draw.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from munasib.core.config import CLASS_WEIGHTING_CHOICES, RnnConfig, SentimentConfig
from munasib.core.constants import SENTIMENT_LABELS, SPLIT_TEST
from munasib.core.exceptions import DataError
from munasib.review_intel.sentiment.data import (
    Batch,
    LengthBucketedBatchSampler,
    SentimentDataset,
    TextEncoder,
    collate_batch,
)
from munasib.review_intel.sentiment.metrics import (
    ClassificationMetrics,
    ProductCalibration,
    classification_metrics,
    compute_product_calibration,
    render_calibration_table,
    render_confusion_and_scores,
    render_metrics_table,
    write_calibration_csv,
    write_evaluation_report,
)
from munasib.review_intel.sentiment.models import (
    NUM_LABELS,
    SavedSentimentModel,
    Vocabulary,
    forward_logits,
    load_model,
    save_model,
)
from munasib.review_intel.sentiment.sampling import PRODUCT_COLUMN

#: Devices ``--device`` accepts. ``auto`` prefers CUDA when it is available.
DEVICE_CHOICES: tuple[str, ...] = ("auto", "cpu", "cuda")

#: DataLoader process count. Kept at zero: the encoder holds a Rust tokenizer
#: whose copy-per-epoch cost dwarfs the cost of building a batch, and a single
#: process keeps runs reproducible. The seeding hooks below still make workers
#: deterministic if this is ever raised.
NUM_WORKERS = 0

#: Improvement threshold for "a better macro-F1", so float noise cannot trigger
#: a new best checkpoint.
IMPROVEMENT_EPSILON = 1e-9

#: File holding the per-epoch history inside the model directory.
TRAINING_LOG_FILENAME = "training_log.json"

#: File holding the resumable optimiser/scheduler state inside ``last/``.
TRAINER_STATE_FILENAME = "trainer_state.pt"

#: Sub-directories of ``output_dir``.
BEST_CHECKPOINT_DIR = "best"
LAST_CHECKPOINT_DIR = "last"


def seed_everything(seed: int) -> None:
    """Seed every random source the training loop touches.

    Args:
        seed: Non-negative seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """Seed one DataLoader worker from the loader's base seed.

    Args:
        worker_id: Worker index supplied by PyTorch.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed + worker_id)


def resolve_device(preference: str) -> torch.device:
    """Translate ``--device`` into a concrete torch device.

    Args:
        preference: One of :data:`DEVICE_CHOICES`.

    Returns:
        The resolved device.

    Raises:
        DataError: If the preference is unknown or CUDA was requested but is not
            available (silently falling back to CPU would invalidate every time
            estimate).
    """
    if preference not in DEVICE_CHOICES:
        raise DataError(f"device must be one of {list(DEVICE_CHOICES)}, got {preference!r}")
    if preference == "cuda" and not torch.cuda.is_available():
        raise DataError("device 'cuda' was requested but CUDA is not available")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def class_weights(labels: Sequence[int], weighting: str) -> np.ndarray:
    """Compute loss weights for the three weak classes.

    ``none`` uses ones (raw frequencies), ``balanced`` uses
    ``n / (k * count_c)`` (inverse frequency), and ``sqrt_inverse`` takes the
    square root of that, so minority classes are up-weighted without the full
    inverse-frequency push that systematically over-states negativity
    downstream.

    Args:
        labels: Training label codes.
        weighting: ``none``, ``sqrt_inverse`` or ``balanced``.

    Returns:
        ``(3,)`` float64 weights aligned with ``SENTIMENT_LABELS``.

    Raises:
        DataError: If the weighting is unknown or a class is absent from the
            training sample.
    """
    if weighting not in CLASS_WEIGHTING_CHOICES:
        raise DataError(
            f"class_weighting must be one of {list(CLASS_WEIGHTING_CHOICES)}, got {weighting!r}"
        )
    array = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(array, minlength=NUM_LABELS).astype("float64")
    if (counts == 0).any():
        missing = [label for label in SENTIMENT_LABELS if counts[label] == 0]
        raise DataError(f"cannot weight classes absent from the training sample: {missing}")
    if weighting == "none":
        return np.ones(NUM_LABELS, dtype="float64")
    balanced = array.size / (NUM_LABELS * counts)
    if weighting == "balanced":
        return balanced
    return np.sqrt(balanced)


def stratified_subset(labels: Sequence[int], size: int | None, *, seed: int) -> np.ndarray:
    """Choose a deterministic class-stratified subset of evaluation rows.

    Every non-empty class gets one row first, then the remainder is filled
    proportionally to class size (largest shortfall first, ties by label code).
    Used only for per-epoch model selection; the reported metrics always come
    from the full split.

    Args:
        labels: Label code per evaluation row.
        size: Requested subset size, or ``None`` for every row.
        seed: Seed of the per-class shuffle.

    Returns:
        Sorted row positions of the subset.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    total = len(labels)
    if size is None or size >= total:
        return np.arange(total, dtype=np.int64)
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")

    grouped: dict[int, list[int]] = defaultdict(list)
    for position, label in enumerate(labels):
        grouped[int(label)].append(position)
    classes = sorted(grouped)
    rng = random.Random(seed)
    for label in classes:
        rng.shuffle(grouped[label])

    counts = {label: len(grouped[label]) for label in classes}
    quota = dict.fromkeys(classes, 0)
    remaining = size
    for label in classes:
        if remaining > 0:
            quota[label] = 1
            remaining -= 1
    shares = {label: counts[label] / total for label in classes}
    while remaining > 0:
        candidates = [label for label in classes if quota[label] < counts[label]]
        if not candidates:
            break
        chosen = max(candidates, key=lambda label: (shares[label] * size - quota[label], -label))
        quota[chosen] += 1
        remaining -= 1
    positions = [position for label in classes for position in grouped[label][: quota[label]]]
    return np.array(sorted(positions), dtype=np.int64)


@dataclass(frozen=True)
class TrainConfig:
    """Everything one training run needs beyond the model and the data.

    Attributes:
        architecture: ``"transformer"`` or ``"rnn"``.
        output_dir: Directory for ``best/``, ``last/`` and the training log.
        batch_size: Rows per optimisation step.
        learning_rate: Peak AdamW learning rate.
        epochs: Maximum epochs (early stopping may end the run sooner).
        warmup_ratio: Share of total steps spent linearly warming up.
        weight_decay: AdamW decoupled weight decay.
        max_grad_norm: Gradient clipping threshold.
        early_stopping_patience: Epochs without improvement before stopping.
        class_weighting: ``none``, ``sqrt_inverse`` or ``balanced``.
        seed: Seed for every random source.
        max_length: Token budget the data was truncated to.
        device: Resolved torch device.
        base_model: Hub id the transformer started from, when known.
        resume: Continue from ``output_dir/last`` instead of starting fresh.
        val_subset: Subset size used for selection, or ``None`` for the full
            validation split. Recorded in the report so a subset number is not
            mistaken for a full-split result.
        full_report_split: Whether the reported numbers cover the whole split.
            ``False`` only for the smoke run, and stated in its report.
    """

    architecture: str
    output_dir: Path
    batch_size: int
    learning_rate: float
    epochs: int
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    early_stopping_patience: int
    class_weighting: str
    seed: int
    max_length: int
    device: torch.device
    base_model: str | None = None
    resume: bool = False
    val_subset: int | None = None
    full_report_split: bool = True


@dataclass(frozen=True)
class EpochRecord:
    """One epoch's training and selection numbers.

    Attributes:
        epoch: One-based epoch number.
        train_loss: Mean training loss, weighted by rows.
        val_macro_f1: Macro-F1 on the selection data.
        val_accuracy: Accuracy on the selection data.
        seconds: Wall-clock seconds for the epoch.
        rows_per_second: Training throughput.
        improved: Whether the epoch set a new best selection score.
    """

    epoch: int
    train_loss: float
    val_macro_f1: float
    val_accuracy: float
    seconds: float
    rows_per_second: float
    improved: bool

    def to_dict(self) -> dict[str, object]:
        """Serialise the record for the training log and report.

        Returns:
            Plain-Python mapping.
        """
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_macro_f1": self.val_macro_f1,
            "val_accuracy": self.val_accuracy,
            "seconds": self.seconds,
            "rows_per_second": self.rows_per_second,
            "improved": self.improved,
        }

    def line(self, total_epochs: int) -> str:
        """Render the per-epoch log line.

        Args:
            total_epochs: Number of epochs the run was configured for.

        Returns:
            One line with loss, macro-F1, elapsed time and throughput.
        """
        marker = " *best*" if self.improved else ""
        return (
            f"epoch {self.epoch}/{total_epochs} loss={self.train_loss:.4f} "
            f"macro-F1={self.val_macro_f1:.4f} acc={self.val_accuracy:.4f} "
            f"{self.seconds:.1f}s {self.rows_per_second:,.0f} rows/s{marker}"
        )


@dataclass(frozen=True)
class Predictions:
    """Aligned predictions for one evaluation dataset.

    Attributes:
        asins: Product id per row.
        y_true: Weak label codes.
        y_pred: Argmax label codes.
        proba: ``(n, 3)`` softmax probabilities.
    """

    asins: tuple[str, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    proba: np.ndarray


@dataclass(frozen=True)
class EvaluatedSplit:
    """Metrics, calibration and predictions for one evaluated split.

    Attributes:
        split: ``"val"`` or ``"test"``.
        predictions: Per-row predictions behind every number here.
        metrics: Three-class and binary metrics.
        calibration_hard: Product calibration from hard predictions.
        calibration_soft: Product calibration from mean negative probability.
    """

    split: str
    predictions: Predictions
    metrics: ClassificationMetrics
    calibration_hard: ProductCalibration
    calibration_soft: ProductCalibration

    @property
    def rows(self) -> int:
        """Number of evaluated rows."""
        return len(self.predictions.y_true)


@dataclass(frozen=True)
class TrainingOutcome:
    """Result of one training run.

    Attributes:
        best_epoch: Epoch whose checkpoint was kept (one-based).
        epochs_run: Epochs actually completed.
        history: Per-epoch records in order.
        validation: Evaluation of the **best** checkpoint on the full validation
            split.
        elapsed_seconds: Total training wall-clock time.
        class_weights: Loss weights that were used, in label order.
        parameters: Trainable parameter count.
    """

    best_epoch: int
    epochs_run: int
    history: tuple[EpochRecord, ...]
    validation: EvaluatedSplit
    elapsed_seconds: float
    class_weights: tuple[float, ...] = field(default_factory=tuple)
    parameters: int = 0


def frame_to_dataset(
    frame: pd.DataFrame, config: SentimentConfig, encoder: TextEncoder
) -> SentimentDataset:
    """Wrap a review frame as a dataset the trainers can iterate.

    Args:
        frame: Rows with the configured text and label columns.
        config: Sentiment settings naming those columns.
        encoder: Tokenizer or vocabulary wrapper.

    Returns:
        A :class:`SentimentDataset` in frame order.

    Raises:
        DataError: If a required column is missing.
    """
    missing = [
        column
        for column in (config.text_column, config.label_column, PRODUCT_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise DataError(f"review frame is missing column(s) {missing}")
    return SentimentDataset(
        texts=[str(text) for text in frame[config.text_column].tolist()],
        labels=[int(label) for label in frame[config.label_column].tolist()],
        asins=[str(asin) for asin in frame[PRODUCT_COLUMN].tolist()],
        encoder=encoder,
    )


def _make_loader(
    dataset: SentimentDataset,
    sampler: LengthBucketedBatchSampler,
    *,
    seed: int,
) -> Iterable[Batch]:
    """Build a DataLoader with dynamic padding and seeded workers.

    Args:
        dataset: Rows to iterate.
        sampler: Batch sampler controlling order and batch size.
        seed: Seed for the loader's own generator.

    Returns:
        An iterable yielding :class:`Batch` objects.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(collate_batch, pad_id=dataset.encoder.pad_id),
        num_workers=NUM_WORKERS,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    dataset: SentimentDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> Predictions:
    """Score every row of an evaluation dataset.

    Args:
        model: Trained module, switched to evaluation mode here.
        dataset: Rows to score.
        batch_size: Rows per forward pass.
        device: Device to run on.

    Returns:
        Predictions in dataset order.

    Raises:
        DataError: If the dataset is empty.
    """
    if len(dataset) == 0:
        raise DataError("cannot evaluate an empty dataset")
    sampler = LengthBucketedBatchSampler(
        dataset.lengths(), batch_size=batch_size, seed=0, shuffle=False
    )
    loader = _make_loader(dataset, sampler, seed=0)
    model.eval()
    model.to(device)

    proba = np.zeros((len(dataset), NUM_LABELS), dtype=np.float32)
    y_true = np.zeros(len(dataset), dtype=np.int64)
    asins: list[str | None] = [None] * len(dataset)
    for batch in loader:
        logits = forward_logits(
            model, batch.input_ids.to(device), batch.attention_mask.to(device)
        )
        probabilities = torch.softmax(logits, dim=-1).cpu().to(torch.float32).numpy()
        indices = batch.indices.numpy()
        proba[indices] = probabilities
        y_true[indices] = batch.labels.numpy()
        for position, asin in zip(indices.tolist(), batch.asins, strict=True):
            asins[position] = asin
    if any(asin is None for asin in asins):
        raise DataError("prediction rows were not all visited; the sampler is incomplete")
    ordered_asins = tuple(str(asin) for asin in asins)
    return Predictions(
        asins=ordered_asins, y_true=y_true, y_pred=proba.argmax(axis=1).astype(np.int64), proba=proba
    )


def evaluate_predictions(predictions: Predictions, *, split: str) -> EvaluatedSplit:
    """Compute every reported number for one split's predictions.

    Args:
        predictions: Rows to score.
        split: ``"val"`` or ``"test"``, recorded in the report.

    Returns:
        Metrics plus hard and soft product calibration.
    """
    metrics = classification_metrics(
        predictions.y_true, predictions.y_pred, predictions.proba
    )
    hard = compute_product_calibration(
        predictions.asins, predictions.y_true, predictions.y_pred
    )
    soft = compute_product_calibration(
        predictions.asins,
        predictions.y_true,
        predictions.y_pred,
        prob_negative=predictions.proba[:, 0],
    )
    return EvaluatedSplit(
        split=split,
        predictions=predictions,
        metrics=metrics,
        calibration_hard=hard,
        calibration_soft=soft,
    )


def _save_checkpoint(
    *,
    model: nn.Module,
    directory: Path,
    settings: TrainConfig,
    tokenizer: PreTrainedTokenizerBase | None,
    vocabulary: Vocabulary | None,
    rnn_config: RnnConfig | None,
) -> None:
    """Save one reloadable checkpoint directory.

    Args:
        model: Module to save.
        directory: Destination (``best`` or ``last``).
        settings: Run settings recorded in the metadata.
        tokenizer: Required for the transformer.
        vocabulary: Required for the recurrent model.
        rnn_config: Required for the recurrent model.
    """
    save_model(
        model,
        directory,
        architecture=settings.architecture,
        max_length=settings.max_length,
        class_weighting=settings.class_weighting,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        rnn_config=rnn_config,
        base_model=settings.base_model,
    )


def _write_training_log(output_dir: Path, history: Sequence[EpochRecord]) -> Path:
    """Write the per-epoch history next to the checkpoints.

    Args:
        output_dir: Run directory.
        history: Records so far.

    Returns:
        The written path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / TRAINING_LOG_FILENAME
    path.write_text(
        json.dumps([record.to_dict() for record in history], indent=2) + "\n", encoding="utf-8"
    )
    return path


def _read_training_log(output_dir: Path) -> list[EpochRecord]:
    """Read a previous run's epoch history, if any.

    Args:
        output_dir: Run directory.

    Returns:
        Records in epoch order (empty when no log exists).

    Raises:
        DataError: If the log exists but is unreadable.
    """
    path = output_dir / TRAINING_LOG_FILENAME
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [EpochRecord(**record) for record in payload]
    except (json.JSONDecodeError, TypeError) as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc


def train_model(
    *,
    model: nn.Module,
    train_dataset: SentimentDataset,
    selection_dataset: SentimentDataset,
    report_dataset: SentimentDataset,
    settings: TrainConfig,
    tokenizer: PreTrainedTokenizerBase | None = None,
    vocabulary: Vocabulary | None = None,
    rnn_config: RnnConfig | None = None,
) -> TrainingOutcome:
    """Fine-tune one model and evaluate the best checkpoint.

    Per-epoch validation runs on ``selection_dataset`` (the full validation
    split or a deterministic stratified subset of it); the returned metrics and
    calibrations always come from ``report_dataset``, which the caller fills
    with the full validation split.

    Args:
        model: Untrained (or resumed) module.
        train_dataset: Leakage-checked training sample.
        selection_dataset: Data used for per-epoch macro-F1 and early stopping.
        report_dataset: Full validation split used for the final report.
        settings: Optimiser, schedule, checkpointing and reporting settings.
        tokenizer: Required for the transformer.
        vocabulary: Required for the recurrent model.
        rnn_config: Required for the recurrent model.

    Returns:
        The best checkpoint's evaluation plus the training history.

    Raises:
        DataError: If a dataset is empty, the architecture needs a companion
            object that was not supplied, or ``--resume`` finds no checkpoint.
    """
    if len(train_dataset) == 0:
        raise DataError("the training sample is empty")
    if len(selection_dataset) == 0 or len(report_dataset) == 0:
        raise DataError("evaluation datasets must not be empty")
    if settings.architecture == "transformer" and tokenizer is None:
        raise DataError("transformer training requires a tokenizer")
    if settings.architecture == "rnn" and (vocabulary is None or rnn_config is None):
        raise DataError("rnn training requires a vocabulary and its RnnConfig")

    device = settings.device
    seed_everything(settings.seed)
    model.to(device)

    weights = class_weights(train_dataset.labels, settings.class_weighting)
    print(
        f"class weights ({settings.class_weighting}): "
        + ", ".join(
            f"{name}={weight:.4f}"
            for name, weight in zip(SENTIMENT_LABELS, weights, strict=True)
        ),
        flush=True,
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    train_sampler = LengthBucketedBatchSampler(
        train_dataset.lengths(),
        batch_size=settings.batch_size,
        seed=settings.seed,
    )
    steps_per_epoch = len(train_sampler)
    total_steps = max(steps_per_epoch * settings.epochs, 1)
    warmup_steps = min(int(math.floor(total_steps * settings.warmup_ratio)), total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device_type, enabled=amp_enabled)

    history: list[EpochRecord] = []
    start_epoch = 0
    best_macro_f1 = -1.0
    best_epoch = 0

    if settings.resume:
        last_dir = settings.output_dir / LAST_CHECKPOINT_DIR
        state_path = last_dir / TRAINER_STATE_FILENAME
        if not state_path.is_file():
            raise DataError(f"--resume needs {state_path}, which does not exist")
        restored: SavedSentimentModel = load_model(last_dir)
        model.load_state_dict(restored.model.state_dict())
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        best_epoch = int(state["best_epoch"])
        best_macro_f1 = float(state["best_macro_f1"])
        history = _read_training_log(settings.output_dir)
        print(
            f"resumed from {last_dir} after epoch {start_epoch} "
            f"(best selection macro-F1 {best_macro_f1:.4f})",
            flush=True,
        )

    print(
        f"training {settings.architecture} on {device.type}: {len(train_dataset):,} rows, "
        f"{steps_per_epoch:,} steps/epoch, {settings.epochs} epoch(s), "
        f"{settings.batch_size} per batch",
        flush=True,
    )

    begun = time.perf_counter()
    epochs_without_improvement = 0
    for epoch in range(start_epoch, settings.epochs):
        train_sampler.set_epoch(epoch)
        loader = _make_loader(train_dataset, train_sampler, seed=settings.seed + epoch)
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        seen = 0
        for batch in loader:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                logits = forward_logits(model, input_ids, attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.item()) * len(labels)
            seen += len(labels)
        elapsed = time.perf_counter() - started

        selection = predict(
            model, selection_dataset, batch_size=settings.batch_size, device=device
        )
        selection_metrics = classification_metrics(
            selection.y_true, selection.y_pred, selection.proba
        )
        improved = selection_metrics.macro_f1 > best_macro_f1 + IMPROVEMENT_EPSILON
        if improved:
            best_macro_f1 = selection_metrics.macro_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record = EpochRecord(
            epoch=epoch + 1,
            train_loss=total_loss / max(seen, 1),
            val_macro_f1=selection_metrics.macro_f1,
            val_accuracy=selection_metrics.accuracy,
            seconds=elapsed,
            rows_per_second=seen / elapsed if elapsed > 0 else 0.0,
            improved=improved,
        )
        history.append(record)
        print(record.line(settings.epochs), flush=True)

        last_dir = settings.output_dir / LAST_CHECKPOINT_DIR
        _save_checkpoint(
            model=model,
            directory=last_dir,
            settings=settings,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            rnn_config=rnn_config,
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "best_epoch": best_epoch,
                "best_macro_f1": best_macro_f1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            last_dir / TRAINER_STATE_FILENAME,
        )
        if improved:
            _save_checkpoint(
                model=model,
                directory=settings.output_dir / BEST_CHECKPOINT_DIR,
                settings=settings,
                tokenizer=tokenizer,
                vocabulary=vocabulary,
                rnn_config=rnn_config,
            )
        _write_training_log(settings.output_dir, history)

        if settings.early_stopping_patience and (
            epochs_without_improvement >= settings.early_stopping_patience
        ):
            print(
                "early stopping: no macro-F1 improvement for "
                f"{epochs_without_improvement} epoch(s)",
                flush=True,
            )
            break

    total_elapsed = time.perf_counter() - begun
    best_dir = settings.output_dir / BEST_CHECKPOINT_DIR
    if not (best_dir / "model_meta.json").is_file():
        raise DataError(f"training finished without writing a checkpoint to {best_dir}")

    best = load_model(best_dir)
    final = predict(best.model, report_dataset, batch_size=settings.batch_size, device=device)
    validation = evaluate_predictions(final, split="val")

    report_scope = (
        f"the full {validation.split} split"
        if settings.full_report_split
        else "a smoke slice, not the full split"
    )
    print(
        f"\nbest epoch: {best_epoch} "
        f"(selection macro-F1 {best_macro_f1:.4f}, reported on {report_scope})"
    )
    for line in validation.metrics.lines():
        print(line)
    print("calibration (hard predictions):")
    for line in validation.calibration_hard.lines():
        print(line)
    print("calibration (mean negative probability):")
    for line in validation.calibration_soft.lines():
        print(line)

    return TrainingOutcome(
        best_epoch=best_epoch,
        epochs_run=len(history),
        history=tuple(history),
        validation=validation,
        elapsed_seconds=total_elapsed,
        class_weights=tuple(float(weight) for weight in weights),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
    )


def evaluate_saved_model(
    model_dir: Path,
    dataset: SentimentDataset,
    *,
    split: str,
    batch_size: int,
    device: torch.device,
) -> EvaluatedSplit:
    """Score a saved checkpoint on one dataset and compute every metric.

    Used for the optional test-split report so the reported numbers come from
    the artifact that was written to disk, not from an in-memory copy.

    Args:
        model_dir: Directory written by :func:`save_model`.
        dataset: Rows to score.
        split: Split name recorded in the result.
        batch_size: Rows per forward pass.
        device: Device to run on.

    Returns:
        Metrics plus hard and soft product calibration for the split.
    """
    saved = load_model(model_dir)
    predictions = predict(saved.model, dataset, batch_size=batch_size, device=device)
    return evaluate_predictions(predictions, split=split)


def report_stem(architecture: str, split: str) -> str:
    """Name the evaluation report for one architecture and split.

    Args:
        architecture: ``"transformer"`` or ``"rnn"``.
        split: ``"val"`` or ``"test"``.

    Returns:
        A file stem such as ``sentiment_transformer_val`` (``_final`` for the
        test split, matching the ``--final`` flag).
    """
    suffix = "final" if split == SPLIT_TEST else split
    return f"sentiment_{architecture}_{suffix}"


def build_report_payload(
    *,
    outcome: TrainingOutcome,
    settings: TrainConfig,
    evaluation: EvaluatedSplit,
    train_rows: int,
    selection_rows: int,
) -> dict[str, Any]:
    """Assemble the computed JSON payload for one split's report.

    Args:
        outcome: Training result.
        settings: Run settings.
        evaluation: Evaluation of one split.
        train_rows: Rows in the training sample.
        selection_rows: Rows used for per-epoch selection.

    Returns:
        Plain-Python, JSON-serialisable payload with every computed number.
    """
    return {
        "model": {
            "architecture": settings.architecture,
            "base_model": settings.base_model,
            "class_weighting": settings.class_weighting,
            "class_weights": list(outcome.class_weights),
            "parameters": outcome.parameters,
            "max_length_tokens": settings.max_length,
            "device": settings.device.type,
        },
        "training": {
            "seed": settings.seed,
            "batch_size": settings.batch_size,
            "learning_rate": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "warmup_ratio": settings.warmup_ratio,
            "max_grad_norm": settings.max_grad_norm,
            "early_stopping_patience": settings.early_stopping_patience,
            "epochs_requested": settings.epochs,
            "epochs_run": outcome.epochs_run,
            "best_epoch": outcome.best_epoch,
            "selection_metric": "macro_f1",
            "val_subset_requested": settings.val_subset,
            "selection_rows": selection_rows,
            "full_report_split": settings.full_report_split,
            "train_rows": train_rows,
            "elapsed_seconds": outcome.elapsed_seconds,
            "history": [record.to_dict() for record in outcome.history],
        },
        "split": evaluation.split,
        "evaluated_rows": evaluation.rows,
        "metrics": evaluation.metrics.to_dict(),
        "calibration": {
            "hard_predictions": evaluation.calibration_hard.to_dict(),
            "mean_negative_probability": evaluation.calibration_soft.to_dict(),
        },
        "labels": "weak supervision from star ratings (1-2 negative, 3 neutral, 4-5 positive)",
    }


def build_report_markdown(
    *,
    outcome: TrainingOutcome,
    settings: TrainConfig,
    evaluation: EvaluatedSplit,
    train_rows: int,
    selection_rows: int,
) -> list[str]:
    """Render the Markdown report for one split.

    Args:
        outcome: Training result.
        settings: Run settings.
        evaluation: Evaluation of one split.
        train_rows: Rows in the training sample.
        selection_rows: Rows used for per-epoch selection.

    Returns:
        Markdown lines.
    """
    subset_note = (
        "full validation split"
        if settings.val_subset is None
        else f"stratified subset of {settings.val_subset:,} rows"
    )
    report_note = (
        f"the full {evaluation.split} split"
        if settings.full_report_split
        else "smoke slice, not the full split"
    )
    lines = [
        f"# Sentiment model report - {settings.architecture} ({evaluation.split})",
        "",
        "Labels are **weak supervision from star ratings** (1-2 negative, 3 neutral, "
        "4-5 positive); every number below measures agreement with that proxy.",
        "",
        f"- Trainable parameters: {outcome.parameters:,}",
        f"- Class weighting: {settings.class_weighting} "
        f"(weights {', '.join(f'{weight:.4f}' for weight in outcome.class_weights)})",
        f"- Training rows: {train_rows:,}",
        f"- Per-epoch selection: {subset_note} ({selection_rows:,} rows), "
        f"metric macro-F1, best epoch {outcome.best_epoch} of {outcome.epochs_run}",
        f"- Reported rows: {evaluation.rows:,} ({report_note})",
        f"- Training wall clock: {outcome.elapsed_seconds:,.1f}s on {settings.device.type}",
        "",
        "## Epoch history",
        "",
        "| Epoch | Train loss | Selection macro-F1 | Accuracy | Seconds | Rows/s | Best |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for record in outcome.history:
        lines.append(
            f"| {record.epoch} | {record.train_loss:.4f} | {record.val_macro_f1:.4f} | "
            f"{record.val_accuracy:.4f} | {record.seconds:.1f} | "
            f"{record.rows_per_second:,.0f} | {'yes' if record.improved else ''} |"
        )
    lines.append("")
    lines.extend(render_metrics_table("Three-class metrics", evaluation.metrics))
    lines.extend(render_confusion_and_scores(evaluation.metrics))
    lines.extend(
        render_calibration_table(
            "Product calibration - hard predictions", evaluation.calibration_hard
        )
    )
    lines.extend(
        render_calibration_table(
            "Product calibration - mean negative probability", evaluation.calibration_soft
        )
    )
    return lines


def write_reports(
    *,
    outcome: TrainingOutcome,
    settings: TrainConfig,
    evaluation: EvaluatedSplit,
    report_dir: Path,
    train_rows: int,
    selection_rows: int,
) -> tuple[Path, Path, Path, Path]:
    """Write the JSON/Markdown report and both per-product calibration CSVs.

    Args:
        outcome: Training result.
        settings: Run settings.
        evaluation: Evaluation of one split.
        report_dir: Destination directory.
        train_rows: Rows in the training sample.
        selection_rows: Rows used for per-epoch selection.

    Returns:
        ``(json_path, markdown_path, hard_csv_path, soft_csv_path)``.
    """
    stem = report_stem(settings.architecture, evaluation.split)
    payload = build_report_payload(
        outcome=outcome,
        settings=settings,
        evaluation=evaluation,
        train_rows=train_rows,
        selection_rows=selection_rows,
    )
    markdown = build_report_markdown(
        outcome=outcome,
        settings=settings,
        evaluation=evaluation,
        train_rows=train_rows,
        selection_rows=selection_rows,
    )
    json_path, markdown_path = write_evaluation_report(report_dir, stem, payload, markdown)
    report_dir.mkdir(parents=True, exist_ok=True)
    hard_csv = write_calibration_csv(
        report_dir / f"{stem}_hard_calibration.csv", evaluation.calibration_hard
    )
    soft_csv = write_calibration_csv(
        report_dir / f"{stem}_soft_calibration.csv", evaluation.calibration_soft
    )
    return json_path, markdown_path, hard_csv, soft_csv


__all__ = [
    "BEST_CHECKPOINT_DIR",
    "DEVICE_CHOICES",
    "LAST_CHECKPOINT_DIR",
    "TRAINER_STATE_FILENAME",
    "TRAINING_LOG_FILENAME",
    "EpochRecord",
    "EvaluatedSplit",
    "Predictions",
    "TrainConfig",
    "TrainingOutcome",
    "build_report_markdown",
    "build_report_payload",
    "class_weights",
    "evaluate_predictions",
    "evaluate_saved_model",
    "frame_to_dataset",
    "predict",
    "report_stem",
    "resolve_device",
    "seed_everything",
    "stratified_subset",
    "train_model",
    "write_reports",
]

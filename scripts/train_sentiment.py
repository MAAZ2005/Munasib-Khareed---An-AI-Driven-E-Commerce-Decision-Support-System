"""Train a PyTorch sentiment model (DistilBERT or BiGRU/BiLSTM).

The script builds the leakage-safe training sample from the ``train`` split,
fine-tunes one architecture, selects the best epoch on validation macro-F1 and
writes a computed evaluation report to ``reports/evaluation/``. The ``test``
split is only touched when ``--final`` is passed, and never together with
``--smoke``.

Because the labels are weak supervision from star ratings, every report pairs
the model with the majority-class baseline: ~83% of reviews are positive, so
accuracy alone cannot distinguish a model from a constant prediction.

Usage::

    python scripts/train_sentiment.py --model transformer --smoke
    python scripts/train_sentiment.py --model rnn --smoke
    python scripts/train_sentiment.py --model transformer --max-train-rows 20000 --epochs 1
    python scripts/train_sentiment.py --model transformer --resume
    python scripts/train_sentiment.py --model transformer --final

``--smoke`` never downloads weights, always runs on the CPU for one epoch on a
small stratified slice, and writes every artifact to a fresh temporary directory,
so it cannot overwrite a real checkpoint or report.

Exit codes: 0 - the run finished and the reports were written; 1 - the run could
not start (bad config, missing review table) or failed while training; 2 - the
arguments contradict each other (``--smoke`` with ``--final`` or ``--resume``).
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch import nn
from transformers import PreTrainedTokenizerBase

from munasib.core.config import CLASS_WEIGHTING_CHOICES, SentimentConfig, Settings, load_settings
from munasib.core.constants import SENTIMENT_LABELS, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from munasib.core.exceptions import ConfigError, DataError
from munasib.review_intel.sentiment.data import (
    TextEncoder,
    build_scratch_tokenizer,
    load_pretrained_tokenizer,
)
from munasib.review_intel.sentiment.models import (
    Vocabulary,
    build_rnn,
    build_tiny_transformer,
    load_transformer,
)
from munasib.review_intel.sentiment.sampling import (
    SPLIT_COLUMN,
    assert_split_only,
    build_train_sample,
    load_eval_split,
    text_mask,
)
from munasib.review_intel.sentiment.train import (
    BEST_CHECKPOINT_DIR,
    DEVICE_CHOICES,
    TrainConfig,
    TrainingOutcome,
    evaluate_saved_model,
    frame_to_dataset,
    report_stem,
    resolve_device,
    stratified_subset,
    train_model,
    write_reports,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("train_sentiment")

#: Name of the processed review table the script reads.
REVIEWS_FILENAME = "reviews.parquet"

#: Architectures ``--model`` accepts.
MODEL_CHOICES: tuple[str, ...] = ("transformer", "rnn")

#: Rows in the smoke training slice. Small enough to finish in seconds, large
#: enough that every class appears.
SMOKE_ROWS = 300

#: Rows in the smoke validation slice (used for selection *and* reporting, which
#: the report states explicitly).
SMOKE_VAL_ROWS = 300

#: Seed of the smoke slice, independent of the configured sample seed so a smoke
#: run always draws the same rows.
SMOKE_SEED = 0

#: Stated in every report and printed at the end of a run.
WEAK_SUPERVISION_STATEMENT = (
    "Labels are **weak supervision** derived from star ratings "
    "(1-2 = negative, 3 = neutral, 4-5 = positive), not human-judged sentiment. "
    "Every metric measures agreement with that proxy, so it is an upper bound on "
    "how well the model would track true sentiment."
)

#: Printed once, before the numbers.
ACCURACY_CAVEAT = (
    "Positive reviews are roughly 83% of every split, so a constant 'positive' "
    "prediction already scores about 0.83 accuracy. Each report prints the "
    "majority-class baseline next to the model, and model selection uses "
    "macro-F1 rather than accuracy."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a PyTorch sentiment model on the weak-supervision sample and "
            "evaluate it on the validation split."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="transformer",
        help="Architecture to train (default: transformer).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Offline CPU check: tiny randomly-initialised transformer, 300 rows, "
            "1 epoch, artifacts in a temporary directory. Cannot be combined with "
            "--final or --resume."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the configured epoch count.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help=(
            "Override review_intel.sentiment.max_train_rows for this run "
            "(positive reviews are subsampled to fit the budget)."
        ),
    )
    parser.add_argument(
        "--per-product-cap",
        type=int,
        default=None,
        help=(
            "Override review_intel.sentiment.max_reviews_per_product "
            "(reviews one product may contribute to the sample)."
        ),
    )
    parser.add_argument(
        "--class-weighting",
        choices=CLASS_WEIGHTING_CHOICES,
        default=None,
        help="Override review_intel.sentiment.class_weighting (default: the configured value).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Where to write best/, last/ and the training log "
            "(default: the configured transformer or rnn output_dir)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from <output-dir>/last instead of starting a new run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override review_intel.sentiment.seed for the sample and training.",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Compute device (default: auto, which prefers CUDA when available).",
    )
    parser.add_argument(
        "--val-subset",
        type=int,
        default=None,
        help=(
            "Use a deterministic class-stratified subset of this many validation "
            "rows for per-epoch selection and early stopping. The final report "
            "always uses the full validation split."
        ),
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help=(
            "Also evaluate the test split with the best checkpoint. Use once, "
            "after the settings are frozen."
        ),
    )
    return parser.parse_args(argv)


def read_reviews(settings: Settings) -> pd.DataFrame:
    """Load the processed review table.

    Args:
        settings: Loaded project settings (``paths.processed_data_dir``).

    Returns:
        The reviews frame.

    Raises:
        DataError: If ``reviews.parquet`` is missing or empty.
    """
    path = Path(settings.paths.processed_data_dir) / REVIEWS_FILENAME
    if not path.exists():
        raise DataError(f"review table not found: {path}; run scripts/build_dataset.py first")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise DataError(f"review table is empty: {path}")
    logger.info("loaded reviews rows=%d path=%s", len(frame), path)
    return frame


def apply_overrides(config: SentimentConfig, args: argparse.Namespace) -> SentimentConfig:
    """Apply CLI overrides on top of the configured sentiment settings.

    Args:
        config: Sentiment settings loaded from YAML.
        args: Parsed arguments carrying the override values.

    Returns:
        A validated copy of ``config`` with the overrides applied.

    Raises:
        ValueError: If an override is invalid (re-raised by validation).
    """
    updates: dict[str, Any] = {}
    if args.max_train_rows is not None:
        updates["max_train_rows"] = args.max_train_rows
    if args.per_product_cap is not None:
        updates["max_reviews_per_product"] = args.per_product_cap
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.class_weighting is not None:
        updates["class_weighting"] = args.class_weighting
    if not updates:
        return config
    # Re-validate rather than ``model_copy(update=...)``: a copy skips
    # validators, and the point of an override is that a bad value fails loudly.
    return SentimentConfig.model_validate({**config.model_dump(), **updates})


def build_smoke_sample(frame: pd.DataFrame, config: SentimentConfig) -> pd.DataFrame:
    """Draw a deterministic class-stratified slice of usable train rows.

    This is **smoke-run scaffolding**, not the production sampler:
    :func:`build_train_sample` never drops negatives or neutrals, so at a
    300-row budget its minority floor would exceed the budget and leave no
    room for positives. Taking a balanced slice keeps the smoke run small while
    still exercising all three classes. The production path is untouched.

    Args:
        frame: Reviews frame as loaded.
        config: Sentiment settings (text and label columns, ``min_chars``).

    Returns:
        At most :data:`SMOKE_ROWS` train rows, balanced across the classes where
        the split allows it.

    Raises:
        DataError: If the resulting slice is empty or not train-only.
    """
    usable = frame.loc[frame[SPLIT_COLUMN] == SPLIT_TRAIN]
    usable = usable.loc[text_mask(usable, config)].reset_index(drop=True)
    assert_split_only(usable, SPLIT_TRAIN)
    if usable.empty:
        raise DataError("the train split has no usable reviews for the smoke run")

    labels = usable[config.label_column].to_numpy()
    rng = random.Random(SMOKE_SEED)
    per_class = max(SMOKE_ROWS // len(SENTIMENT_LABELS), 1)
    chosen: list[int] = []
    for label in SENTIMENT_LABELS:
        positions = np.flatnonzero(labels == label).tolist()
        rng.shuffle(positions)
        chosen.extend(positions[:per_class])

    leftover = SMOKE_ROWS - len(chosen)
    if leftover > 0:
        taken = set(chosen)
        remaining = [position for position in range(len(usable)) if position not in taken]
        rng.shuffle(remaining)
        chosen.extend(remaining[:leftover])
    rows = usable.iloc[sorted(chosen)].reset_index(drop=True)
    assert_split_only(rows, SPLIT_TRAIN)
    return rows


def _train_config(
    config: SentimentConfig,
    *,
    architecture: str,
    output_dir: Path,
    device: str,
    epochs: int | None,
    resume: bool,
    val_subset: int | None,
    full_report_split: bool,
) -> TrainConfig:
    """Assemble the training settings for one architecture.

    Both architectures share the schedule in the ``transformer`` block except the
    peak learning rate: a scratch recurrent model trained at a fine-tuning rate
    would barely move, so ``rnn.learning_rate`` supplies its own.

    Args:
        config: Sentiment settings.
        architecture: ``"transformer"`` or ``"rnn"``.
        output_dir: Where checkpoints and the training log go.
        device: ``--device`` value.
        epochs: Optional epoch override.
        resume: Whether to continue from ``last/``.
        val_subset: Per-epoch selection subset size, or ``None``.
        full_report_split: Whether the final report covers the whole split.

    Returns:
        The resolved settings.
    """
    schedule = config.transformer
    learning_rate = schedule.learning_rate if architecture == "transformer" else config.rnn.learning_rate
    return TrainConfig(
        architecture=architecture,
        output_dir=output_dir,
        batch_size=schedule.batch_size,
        learning_rate=learning_rate,
        epochs=epochs if epochs is not None else schedule.epochs,
        warmup_ratio=schedule.warmup_ratio,
        weight_decay=schedule.weight_decay,
        max_grad_norm=schedule.max_grad_norm,
        early_stopping_patience=schedule.early_stopping_patience,
        class_weighting=config.class_weighting,
        seed=config.seed,
        max_length=config.max_length_tokens,
        device=resolve_device(device),
        base_model=schedule.model_name if architecture == "transformer" else None,
        resume=resume,
        val_subset=val_subset,
        full_report_split=full_report_split,
    )


def run(settings: Settings, args: argparse.Namespace) -> TrainingOutcome:
    """Execute one training run and write its reports.

    Args:
        settings: Loaded project settings.
        args: Parsed arguments.

    Returns:
        The training outcome (validation metrics are already written).

    Raises:
        DataError: On a missing table, an unusable sample, or a wasteful flag
            combination.
        ValueError: On an invalid configuration override.
    """
    config = apply_overrides(settings.review_intel.sentiment, args)
    smoke = bool(args.smoke)
    architecture = args.model

    if smoke:
        epochs = 1
        output_dir = Path(tempfile.mkdtemp(prefix="munasib-sentiment-smoke-"))
        report_dir = output_dir / "reports"
        device_name = "cpu"
        val_subset = SMOKE_VAL_ROWS
        full_report_split = False
        logger.info("smoke run: artifacts go to %s", output_dir)
    else:
        epochs = args.epochs
        default_dir = (
            config.transformer.output_dir if architecture == "transformer" else config.rnn.output_dir
        )
        output_dir = Path(args.output_dir) if args.output_dir else Path(default_dir)
        report_dir = Path(config.report_dir)
        device_name = args.device
        val_subset = args.val_subset
        full_report_split = True

    device = resolve_device(device_name)
    frame = read_reviews(settings)
    started = time.perf_counter()

    if smoke:
        train_frame = build_smoke_sample(frame, config)
        logger.info("smoke training slice rows=%d", len(train_frame))
    else:
        sample = build_train_sample(frame, config)
        train_frame = sample.frame
        logger.info(
            "training sample rows=%d products=%d classes=%s",
            len(train_frame),
            sample.report.products,
            sample.report.class_counts,
        )

    train_texts = [str(text) for text in train_frame[config.text_column].tolist()]

    tokenizer: PreTrainedTokenizerBase | None
    vocabulary: Vocabulary | None
    model: nn.Module
    if architecture == "transformer":
        if smoke:
            scratch = build_scratch_tokenizer(
                train_texts, max_length=config.max_length_tokens
            )
            tokenizer = scratch
            model = build_tiny_transformer(
                vocab_size=scratch.vocab_size,
                max_length=config.max_length_tokens,
            )
        else:
            tokenizer = load_pretrained_tokenizer(config.transformer.model_name)
            model = load_transformer(config.transformer.model_name)
        vocabulary = None
        logger.info(
            "transformer ready model=%s vocab=%d",
            "tiny-scratch" if smoke else config.transformer.model_name,
            tokenizer.vocab_size,
        )
    else:
        vocabulary = Vocabulary.from_texts(
            train_texts, min_freq=config.rnn.min_freq, max_size=config.rnn.vocab_size
        )
        model = build_rnn(config.rnn, vocab_size=len(vocabulary))
        tokenizer = None
        logger.info("rnn ready cell=%s vocab=%d", config.rnn.cell, len(vocabulary))

    encoder = TextEncoder(
        max_length=config.max_length_tokens, tokenizer=tokenizer, vocabulary=vocabulary
    )
    train_dataset = frame_to_dataset(train_frame, config, encoder)

    val_frame = load_eval_split(frame, SPLIT_VAL, config)
    if val_subset is not None:
        positions = stratified_subset(
            val_frame[config.label_column].tolist(), val_subset, seed=config.seed
        )
        selection_frame = val_frame.iloc[positions].reset_index(drop=True)
    else:
        selection_frame = val_frame
    selection_dataset = frame_to_dataset(selection_frame, config, encoder)
    report_dataset = frame_to_dataset(selection_frame if smoke else val_frame, config, encoder)

    train_settings = _train_config(
        config,
        architecture=architecture,
        output_dir=output_dir,
        device=device_name,
        epochs=epochs,
        resume=bool(args.resume),
        val_subset=val_subset,
        full_report_split=full_report_split,
    )
    logger.info(
        "device=%s epochs=%d batch=%d lr=%g weighting=%s output=%s",
        device.type,
        train_settings.epochs,
        train_settings.batch_size,
        train_settings.learning_rate,
        train_settings.class_weighting,
        output_dir,
    )

    rnn_config = config.rnn if architecture == "rnn" else None
    outcome = train_model(
        model=model,
        train_dataset=train_dataset,
        selection_dataset=selection_dataset,
        report_dataset=report_dataset,
        settings=train_settings,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        rnn_config=rnn_config,
    )

    paths = write_reports(
        outcome=outcome,
        settings=train_settings,
        evaluation=outcome.validation,
        report_dir=report_dir,
        train_rows=len(train_frame),
        selection_rows=len(selection_frame),
    )
    logger.info("wrote report json=%s markdown=%s", paths[0], paths[1])
    logger.info("wrote calibration csv hard=%s soft=%s", paths[2], paths[3])

    if args.final:
        test_frame = load_eval_split(frame, SPLIT_TEST, config)
        test_dataset = frame_to_dataset(test_frame, config, encoder)
        evaluation = evaluate_saved_model(
            output_dir / BEST_CHECKPOINT_DIR,
            test_dataset,
            split=SPLIT_TEST,
            batch_size=train_settings.batch_size,
            device=device,
        )
        test_paths = write_reports(
            outcome=outcome,
            settings=train_settings,
            evaluation=evaluation,
            report_dir=report_dir,
            train_rows=len(train_frame),
            selection_rows=len(selection_frame),
        )
        logger.info(
            "wrote final (test) report json=%s markdown=%s", test_paths[0], test_paths[1]
        )

    elapsed = time.perf_counter() - started
    print()
    print(ACCURACY_CAVEAT)
    print()
    print(WEAK_SUPERVISION_STATEMENT)
    print()
    print(f"checkpoints: {output_dir}")
    print(f"reports: {report_dir} ({report_stem(architecture, outcome.validation.split)})")
    print(f"total wall clock: {elapsed:,.1f}s")
    return outcome


def main(argv: Sequence[str] | None = None) -> int:
    """Train one model and print the computed numbers.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 finished, 1 failed, 2 contradictory arguments.
    """
    args = parse_args(argv)
    if args.smoke and args.final:
        logger.error(
            "--final cannot be combined with --smoke: the smoke run is an offline "
            "CPU check and never touches the test split"
        )
        return 2
    if args.smoke and args.resume:
        logger.error("--resume cannot be combined with --smoke: smoke runs use a fresh temp dir")
        return 2

    try:
        settings: Settings = load_settings(args.config)
    except ConfigError as exc:
        logger.error("config_error path=%s detail=%s", args.config, exc)
        return 1

    try:
        run(settings, args)
    except (ConfigError, DataError, ValueError) as exc:
        logger.error("run_failed detail=%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

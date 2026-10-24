"""Train and evaluate the TF-IDF + logistic-regression sentiment baseline.

The command trains the reference model on a leakage-safe sample of the ``train``
split, compares the two ``class_weight`` variants on the **full** ``val`` split,
and writes the numbers to ``reports/evaluation/``. The ``test`` split is only
touched when ``--final`` is passed, so it stays a one-shot hold-out instead of
being tuned against.

Because the labels are weak supervision from star ratings, every table this
script prints carries the majority-class baseline next to the model: with ~83%
positive reviews, accuracy alone cannot tell a real model from a constant
prediction.

Usage::

    python scripts/train_sentiment_baseline.py                      # val only
    python scripts/train_sentiment_baseline.py --max-train-rows 20000
    python scripts/train_sentiment_baseline.py --class-weight none
    python scripts/train_sentiment_baseline.py --final              # scores test

Exit codes: 0 - the run finished and the reports were written; 1 - the run could
not start (bad config, missing or invalid review table) or failed while training.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from munasib.core.config import SentimentConfig, Settings, load_settings
from munasib.core.constants import SENTIMENT_LABEL_NAMES, SPLIT_TEST, SPLIT_VAL
from munasib.core.exceptions import ConfigError, DataError
from munasib.review_intel.sentiment.baseline import (
    CLASS_WEIGHT_CHOICES,
    SentimentBaseline,
    evaluate_baseline,
    resolve_class_weight,
    save_baseline,
    train_baseline,
)
from munasib.review_intel.sentiment.metrics import (
    ClassificationMetrics,
    ProductCalibration,
    render_calibration_table,
    render_metrics_table,
    write_calibration_csv,
    write_evaluation_report,
)
from munasib.review_intel.sentiment.sampling import (
    SampleReport,
    build_train_sample,
    load_eval_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("train_sentiment_baseline")

#: Name of the processed review table the script reads.
REVIEWS_FILENAME = "reviews.parquet"

#: Report file stem used when only ``val`` is scored.
STEM_VAL = "sentiment_baseline_val"

#: Report file stem used when ``--final`` also scores ``test``.
STEM_FINAL = "sentiment_baseline_final"

#: Prefix for the per-product calibration CSVs (one file per split and variant,
#: so it is shared by the validation-only and final runs instead of duplicating
#: the report stem).
CALIBRATION_PREFIX = "sentiment_baseline"

#: Stated verbatim in every generated report; the labels are a proxy, and no
#: number in this file should be read as human-judged sentiment accuracy.
WEAK_SUPERVISION_STATEMENT = (
    "Labels are **weak supervision** derived from star ratings "
    "(1-2 = negative, 3 = neutral, 4-5 = positive), not human-judged sentiment. "
    "Every metric below measures agreement with that proxy, so it is an upper "
    "bound on how well the model would track true sentiment."
)

#: Rationale printed once, before the numbers, so results are never read as
#: accuracy contests on a balanced problem.
ACCURACY_CAVEAT = (
    "Positive reviews are roughly 83% of every split, so a constant "
    "'positive' prediction already scores about 0.83 accuracy. The "
    "majority-class baseline is printed next to every model, and macro-F1 is the "
    "headline comparison because it weights the 12% negative and 5% neutral "
    "classes equally with the majority class."
)


@dataclass
class VariantResult:
    """One ``class_weight`` variant trained and scored on one split.

    Attributes:
        name: Reporting name of the variant (``"balanced"`` or ``"none"``).
        baseline: The fitted model (shared across splits).
        metrics: Three-class metrics on the evaluated split.
        calibration: Product-level calibration on the evaluated split.
        train_seconds: Wall-clock seconds spent fitting this variant.
        eval_seconds: Wall-clock seconds spent scoring this split.
    """

    name: str
    baseline: SentimentBaseline
    metrics: ClassificationMetrics
    calibration: ProductCalibration
    train_seconds: float
    eval_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise the variant's computed figures for the JSON report.

        Returns:
            Plain-Python mapping with the model metadata, metrics and
            calibration summary.
        """
        return {
            "class_weight": self.name,
            "model": self.baseline.to_metadata(),
            "metrics": self.metrics.to_dict(),
            "calibration": self.calibration.to_dict(),
            "train_seconds": round(self.train_seconds, 3),
            "eval_seconds": round(self.eval_seconds, 3),
        }


@dataclass
class SplitResult:
    """Every variant scored on one evaluation split.

    Attributes:
        split: Split name (``val`` or ``test``).
        rows: Rows scored.
        variants: One :class:`VariantResult` per ``class_weight`` variant, in
            :data:`CLASS_WEIGHT_CHOICES` order.
        seconds: Wall-clock seconds spent scoring the split.
    """

    split: str
    rows: int
    variants: list[VariantResult]
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise the split's evaluations for the JSON report.

        Returns:
            Plain-Python mapping keyed by variant name.
        """
        return {
            "split": self.split,
            "rows": self.rows,
            "seconds": round(self.seconds, 3),
            "variants": {variant.name: variant.to_dict() for variant in self.variants},
        }


@dataclass
class RunOutcome:
    """Everything one run produced, ready to print.

    Attributes:
        sample: Training-sample report (counts before/after each step).
        splits: Evaluated splits in report order (``val``, then ``test`` when
            ``--final`` was passed).
        artifact_path: Where the selected variant was written.
        selected_variant: The ``class_weight`` variant saved as the artifact.
        report_paths: ``(json_path, markdown_path)`` of the written report.
        calibration_paths: Per-variant calibration CSV paths, keyed by split name
            and then by variant name.
        payload: The JSON payload actually written.
        elapsed_seconds: Total wall-clock seconds.
    """

    sample: SampleReport
    splits: list[SplitResult]
    artifact_path: Path
    selected_variant: str
    report_paths: tuple[Path, Path]
    calibration_paths: dict[str, dict[str, Path]]
    payload: dict[str, Any]
    elapsed_seconds: float

    def lines(self) -> list[str]:
        """Render the whole run as printable lines.

        Returns:
            A stable sequence: sample report, per-split metric tables and
            calibration blocks, artifact location and elapsed time.
        """
        lines: list[str] = [
            "Training sample",
            "-" * 72,
            *self.sample.lines(),
            "",
        ]
        for split in self.splits:
            note = (
                " (hold-out: scored once, never used for model selection)"
                if split.split == SPLIT_TEST
                else ""
            )
            lines.extend(
                [
                    f"Split: {split.split} - {split.rows:,} reviews{note}",
                    "-" * 72,
                ]
            )
            for variant in split.variants:
                lines.append(f"[class_weight={variant.name}]")
                lines.extend(variant.metrics.lines())
                lines.append("  product-level calibration")
                lines.extend(f"  {line}" for line in variant.calibration.lines())
                lines.append("")
        lines.extend(
            [
                f"{'artifact saved':<28}: {self.artifact_path} "
                f"(class_weight={self.selected_variant})",
                f"{'json report':<28}: {self.report_paths[0]}",
                f"{'markdown report':<28}: {self.report_paths[1]}",
                *(
                    f"{f'calibration csv ({split})':<28}: {path}"
                    for split, paths in self.calibration_paths.items()
                    for path in paths.values()
                ),
                f"{'elapsed':<28}: {self.elapsed_seconds:.1f}s",
            ]
        )
        return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train the TF-IDF + logistic-regression sentiment baseline and evaluate "
            "it on the validation split."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration (default: configs/config.yaml).",
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
        "--seed",
        type=int,
        default=None,
        help="Override review_intel.sentiment.seed for the deterministic sample.",
    )
    parser.add_argument(
        "--class-weight",
        choices=CLASS_WEIGHT_CHOICES,
        default="balanced",
        help=(
            "Which class-weight variant to save as the artifact (both are always "
            "trained and reported). Default: balanced."
        ),
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help=(
            "Also evaluate the test split. Use once, after settings are frozen; "
            "the test split is never used for model selection."
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
        raise DataError(
            f"review table not found: {path}; run scripts/build_dataset.py first"
        )
    frame = pd.read_parquet(path)
    if frame.empty:
        raise DataError(f"review table is empty: {path}")
    logger.info("loaded reviews rows=%d path=%s", len(frame), path)
    return frame


def apply_overrides(
    config: SentimentConfig,
    max_train_rows: int | None,
    per_product_cap: int | None,
    seed: int | None,
) -> SentimentConfig:
    """Apply CLI overrides on top of the configured sentiment settings.

    Args:
        config: Sentiment settings loaded from YAML.
        max_train_rows: New positive-subsample budget, or None to keep the config.
        per_product_cap: New per-product cap, or None to keep the config.
        seed: New sample seed, or None to keep the config.

    Returns:
        A validated copy of ``config`` with the overrides applied.

    Raises:
        ValueError: If an override is not positive (re-raised by validation).
    """
    updates: dict[str, Any] = {}
    if max_train_rows is not None:
        updates["max_train_rows"] = max_train_rows
    if per_product_cap is not None:
        updates["max_reviews_per_product"] = per_product_cap
    if seed is not None:
        updates["seed"] = seed
    if not updates:
        return config
    # Re-validate the merged mapping instead of ``model_copy(update=...)``: a
    # copy bypasses validators, and the point of an override is that a bad value
    # (e.g. ``--max-train-rows 0``) fails loudly.
    return SentimentConfig.model_validate({**config.model_dump(), **updates})


def _sample_markdown(report: SampleReport) -> list[str]:
    """Render the training-sample report as Markdown.

    Args:
        report: Report produced by :func:`build_train_sample`.

    Returns:
        Markdown lines with the step table and the class shares.
    """
    lines = [
        f"## Training sample (split `{report.split}`, {report.rows:,} rows, "
        f"{report.products:,} products)",
        "",
        "| Step | Rows | Detail |",
        "|---|---:|---|",
    ]
    for step in report.steps:
        lines.append(f"| {step.step} | {step.rows:,} | {step.detail} |")
    lines.extend(
        [
            "",
            "| Class | Rows | Share |",
            "|---|---:|---:|",
        ]
    )
    for label, count in report.class_counts.items():
        share = report.class_shares[label]
        lines.append(f"| {SENTIMENT_LABEL_NAMES[label]} | {count:,} | {share:.2%} |")
    if report.notes:
        lines.append("")
        lines.extend(f"- {note}" for note in report.notes)
    lines.append("")
    return lines


def _split_markdown(
    split: SplitResult, calibration_paths: dict[str, dict[str, Path]]
) -> list[str]:
    """Render one evaluated split as Markdown.

    Args:
        split: Computed evaluations for the split.
        calibration_paths: Per-split, per-variant calibration CSV paths.

    Returns:
        Markdown lines for every variant, each followed by its calibration
        summary.
    """
    heading = f"## {split.split.capitalize()} split - {split.rows:,} reviews"
    lines = [heading, ""]
    if split.split == SPLIT_TEST:
        lines.extend(
            [
                "This hold-out is scored once, with the frozen configuration; it is "
                "never used to choose a model or a threshold.",
                "",
            ]
        )
    for variant in split.variants:
        lines.extend(
            render_metrics_table(
                f"TF-IDF + logistic regression (class_weight={variant.name})", variant.metrics
            )
        )
        lines.extend(
            render_calibration_table(
                f"Product-level calibration (class_weight={variant.name})", variant.calibration
            )
        )
        csv_path = calibration_paths.get(split.split, {}).get(variant.name)
        if csv_path is not None:
            lines.extend([f"Per-product rows: `{csv_path}`.", ""])
    return lines


def _evaluate_split(
    split: str,
    frame: pd.DataFrame,
    baselines: dict[str, SentimentBaseline],
    config: SentimentConfig,
) -> SplitResult:
    """Score every fitted variant on one split.

    Args:
        split: Split name.
        frame: Evaluation rows.
        baselines: Fitted models keyed by variant name.
        config: Sentiment settings.

    Returns:
        The split result with one :class:`VariantResult` per baseline.
    """
    started = time.perf_counter()
    variants: list[VariantResult] = []
    for name, baseline in baselines.items():
        variant_started = time.perf_counter()
        metrics, calibration, _ = evaluate_baseline(baseline, frame, config)
        variants.append(
            VariantResult(
                name=name,
                baseline=baseline,
                metrics=metrics,
                calibration=calibration,
                train_seconds=0.0,
                eval_seconds=time.perf_counter() - variant_started,
            )
        )
        logger.info(
            "evaluated split=%s class_weight=%s rows=%d macro_f1=%.4f",
            split,
            name,
            metrics.n,
            metrics.macro_f1,
        )
    return SplitResult(
        split=split,
        rows=len(frame),
        variants=variants,
        seconds=time.perf_counter() - started,
    )


def run(settings: Settings, args: argparse.Namespace) -> RunOutcome:
    """Train both variants, evaluate them and write the reports.

    Args:
        settings: Loaded project settings.
        args: Parsed command-line arguments.

    Returns:
        The run outcome (sample report, split results, written paths, payload).

    Raises:
        DataError: If the review table is missing/empty or a split cannot be built.
        ValueError: If an override is invalid.
    """
    started = time.perf_counter()
    config = apply_overrides(
        settings.review_intel.sentiment,
        args.max_train_rows,
        args.per_product_cap,
        args.seed,
    )
    logger.info(
        "config max_train_rows=%d per_product_cap=%d seed=%d text_column=%s",
        config.max_train_rows,
        config.max_reviews_per_product,
        config.seed,
        config.text_column,
    )

    reviews = read_reviews(settings)

    sample_started = time.perf_counter()
    sample = build_train_sample(reviews, config)
    logger.info(
        "sample rows=%d products=%d seconds=%.1f",
        sample.report.rows,
        sample.report.products,
        time.perf_counter() - sample_started,
    )
    if sample.report.rows == 0:
        raise DataError("the training sample is empty; check the split and text columns")

    baselines: dict[str, SentimentBaseline] = {}
    train_seconds: dict[str, float] = {}
    for name in CLASS_WEIGHT_CHOICES:
        train_started = time.perf_counter()
        baselines[name] = train_baseline(
            sample.frame, config, resolve_class_weight(name)
        )
        train_seconds[name] = time.perf_counter() - train_started
        logger.info(
            "trained class_weight=%s rows=%d features=%d converged=%s seconds=%.1f",
            name,
            baselines[name].train_rows,
            baselines[name].vocabulary_size,
            baselines[name].converged,
            train_seconds[name],
        )

    splits: list[SplitResult] = []
    val = load_eval_split(reviews, SPLIT_VAL, config)
    val_result = _evaluate_split(SPLIT_VAL, val, baselines, config)
    for variant in val_result.variants:
        variant.train_seconds = train_seconds[variant.name]
    splits.append(val_result)

    if args.final:
        test = load_eval_split(reviews, SPLIT_TEST, config)
        test_result = _evaluate_split(SPLIT_TEST, test, baselines, config)
        for variant in test_result.variants:
            variant.train_seconds = train_seconds[variant.name]
        splits.append(test_result)
    else:
        logger.info("test split not evaluated (pass --final to score it once)")

    selected = baselines[args.class_weight]
    artifact_path = save_baseline(selected, Path(config.model_path))
    logger.info("saved artifact path=%s class_weight=%s", artifact_path, args.class_weight)

    report_dir = Path(config.report_dir)
    stem = STEM_FINAL if args.final else STEM_VAL
    calibration_paths: dict[str, dict[str, Path]] = {}
    for split in splits:
        calibration_paths[split.split] = {
            variant.name: write_calibration_csv(
                report_dir
                / f"{CALIBRATION_PREFIX}_{split.split}_{variant.name}_calibration.csv",
                variant.calibration,
            )
            for variant in split.variants
        }

    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {
        "script": "scripts/train_sentiment_baseline.py",
        "config": {
            "text_column": config.text_column,
            "label_column": config.label_column,
            "max_reviews_per_product": config.max_reviews_per_product,
            "max_train_rows": config.max_train_rows,
            "min_chars": config.min_chars,
            "seed": config.seed,
            "max_length_tokens": config.max_length_tokens,
            "tfidf": config.tfidf.model_dump(),
            "logreg": config.logreg.model_dump(),
        },
        "sample": sample.report.to_dict(),
        "evaluations": {split.split: split.to_dict() for split in splits},
        "artifact": {
            "path": str(artifact_path),
            "class_weight": args.class_weight,
            "metadata": selected.to_metadata(),
        },
        "elapsed_seconds": round(elapsed, 3),
    }

    markdown: list[str] = [
        f"# Sentiment baseline - {'validation + test' if args.final else 'validation'} split",
        "",
        "## Weak supervision",
        "",
        WEAK_SUPERVISION_STATEMENT,
        "",
        ACCURACY_CAVEAT,
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Text column | `{config.text_column}` |",
        f"| Label column | `{config.label_column}` |",
        f"| Max reviews per product | {config.max_reviews_per_product:,} |",
        f"| Max training rows | {config.max_train_rows:,} |",
        f"| Min characters | {config.min_chars} |",
        f"| Seed | {config.seed} |",
        f"| TF-IDF n-gram range | {list(config.tfidf.ngram_range)} |",
        f"| TF-IDF min_df | {config.tfidf.min_df} |",
        f"| TF-IDF max_features | {config.tfidf.max_features:,} |",
        f"| TF-IDF sublinear_tf | {config.tfidf.sublinear_tf} |",
        f"| Logistic regression C | {config.logreg.C} |",
        f"| Logistic regression max_iter | {config.logreg.max_iter} |",
        "",
        *_sample_markdown(sample.report),
    ]
    for split in splits:
        markdown.extend(_split_markdown(split, calibration_paths))
    markdown.extend(
        [
            "## Artifact",
            "",
            f"- Path: `{artifact_path}`",
            f"- Variant: `class_weight={args.class_weight}`",
            f"- Training rows: {selected.train_rows:,}",
            f"- Distinct training products: {selected.train_products:,}",
            f"- TF-IDF vocabulary: {selected.vocabulary_size:,} features",
            f"- Converged: {selected.converged} (iterations: {selected.n_iter})",
            "",
            f"Total wall-clock time: {elapsed:.1f}s.",
        ]
    )

    report_paths = write_evaluation_report(report_dir, stem, payload, markdown)
    logger.info("wrote report json=%s markdown=%s", *report_paths)

    return RunOutcome(
        sample=sample.report,
        splits=splits,
        artifact_path=artifact_path,
        selected_variant=args.class_weight,
        report_paths=report_paths,
        calibration_paths=calibration_paths,
        payload=payload,
        elapsed_seconds=elapsed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Train/evaluate the baseline and print the computed numbers.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 = finished, 1 = could not start or failed).
    """
    args = parse_args(argv)
    try:
        settings: Settings = load_settings(args.config)
    except ConfigError as exc:
        logger.error("config_error path=%s detail=%s", args.config, exc)
        return 1

    try:
        outcome = run(settings, args)
    except (ConfigError, DataError, ValueError) as exc:
        logger.error("run_failed detail=%s", exc)
        return 1

    print()
    print(WEAK_SUPERVISION_STATEMENT)
    print()
    for line in outcome.lines():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

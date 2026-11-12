"""Score every review with the frozen sentiment checkpoint.

The script reads ``review_id`` and the configured clean-text column from the
processed review table, scores the rows with
:class:`~munasib.review_intel.sentiment.predictor.SentimentPredictor`, and writes
one row per review:

.. code-block:: text

    review_id | prob_negative | prob_neutral | prob_positive | predicted_label | has_prediction

``predicted_label`` is ``NULL`` and the three probabilities are ``NaN`` exactly
when the review has no usable text; ``has_prediction`` records why. Nothing is
invented for an empty review.

Scoring runs in fixed-size chunks and each completed chunk is written to its own
part file under ``<output-stem>.parts/``. A Colab disconnect therefore costs at
most the chunk that was in flight: re-running with ``--resume`` keeps the parts
whose input fingerprint and checkpoint hash still match. The final parquet is
assembled from the parts only when every row has been scored.

A sidecar JSON next to the output (``review_predictions.meta.json``) records the
checkpoint path, the SHA-256 of its weights and ``model_meta.json``, the input
fingerprint and the output hash. ``scripts/ingest_predictions.py`` reads it and
refuses predictions that do not belong to the checkpoint now on disk.

Usage::

    python scripts/predict_reviews.py                      # full table, configured paths
    python scripts/predict_reviews.py --limit 1000 --device cpu
    python scripts/predict_reviews.py --device cuda --output /content/run/preds.parquet
    python scripts/predict_reviews.py --resume             # continue a chunked run

Exit codes: 0 - every row was scored and the parquet plus sidecar were written;
1 - the run could not start or failed; 2 - ``--resume`` was used with a different
input, checkpoint or chunk size than the run it is continuing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from munasib.core.config import Settings, load_settings
from munasib.core.exceptions import ConfigError, DataError
from munasib.review_intel.sentiment.predictor import (
    DEVICE_CHOICES,
    SentimentPredictor,
    describe_checkpoint,
    file_sha256,
    sidecar_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("predict_reviews")

#: Script name recorded in the sidecar.
SCRIPT = "scripts/predict_reviews.py"

#: Columns written to the prediction parquet, in order. ``scripts/ingest_predictions.py``
#: validates exactly this list, so a schema change is a two-script change.
PREDICTION_COLUMNS: tuple[str, ...] = (
    "review_id",
    "prob_negative",
    "prob_neutral",
    "prob_positive",
    "predicted_label",
    "has_prediction",
)

#: Suffix of the directory holding completed chunk files.
PARTS_DIR_SUFFIX = ".parts"

#: Marker file inside the parts directory describing the run being continued.
RUN_MARKER_FILENAME = "run.json"

#: Review column holding the product id the batch scorer never writes.
REVIEW_ID_COLUMN = "review_id"


class ResumeError(DataError):
    """``--resume`` was asked to continue a run whose inputs have changed."""


@dataclass(frozen=True)
class PredictionRun:
    """Result of one batch-scoring invocation.

    Attributes:
        input_path: Review table that was scored.
        output_path: Written prediction parquet.
        sidecar: Written sidecar JSON.
        rows: Rows in the input table (after ``--limit``).
        rows_with_prediction: Rows that produced probabilities.
        scored_rows: Rows scored by *this* invocation (0 when every part was
            reused by ``--resume``).
        elapsed_seconds: Wall-clock time for this invocation.
        rows_per_second: Throughput of this invocation, or ``None`` when nothing
            was scored.
        device: Device the model ran on.
    """

    input_path: Path
    output_path: Path
    sidecar: Path
    rows: int
    rows_with_prediction: int
    scored_rows: int
    elapsed_seconds: float
    rows_per_second: float | None
    device: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Score every review with a saved sentiment checkpoint and write a "
            "per-review prediction parquet plus a checkpoint-identity sidecar."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint directory (default: review_intel.sentiment.prediction.checkpoint)."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Review parquet to score (default: "
            "review_intel.sentiment.prediction.input_path)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Prediction parquet to write (default: "
            "review_intel.sentiment.prediction.output_path)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Compute device (default: auto, which prefers CUDA when available).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Rows per forward pass (default: review_intel.sentiment.prediction.batch_size).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N rows (smoke runs); the sidecar records the limit.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=None,
        help="Rows per resumable chunk (default: review_intel.sentiment.prediction.chunk_rows).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a chunked run: keep parts whose input, checkpoint and chunk "
            "size match; fail if they do not."
        ),
    )
    return parser.parse_args(argv)


def read_reviews(path: Path, text_column: str) -> pd.DataFrame:
    """Load just the two columns the batch scorer needs.

    Args:
        path: Processed review parquet.
        text_column: Column holding the cleaned review text.

    Returns:
        Frame with ``review_id`` and ``text_column``, in file order.

    Raises:
        DataError: If the file is missing, empty, lacks a column, or holds
            duplicate/blank ids.
    """
    if not path.is_file():
        raise DataError(f"review table not found: {path}")
    try:
        frame = pd.read_parquet(path, columns=[REVIEW_ID_COLUMN, text_column])
    except (KeyError, ValueError) as exc:
        raise DataError(
            f"{path} is missing {REVIEW_ID_COLUMN!r} or {text_column!r}: {exc}"
        ) from exc
    if frame.empty:
        raise DataError(f"review table is empty: {path}")
    if frame[REVIEW_ID_COLUMN].isna().any():
        raise DataError(f"{path} has review rows without a review_id")
    if frame[REVIEW_ID_COLUMN].duplicated().any():
        duplicates = int(frame[REVIEW_ID_COLUMN].duplicated().sum())
        raise DataError(f"{path} has {duplicates:,} duplicate review_id value(s)")
    logger.info("loaded reviews rows=%d path=%s", len(frame), path)
    return frame.reset_index(drop=True)


def normalise_texts(frame: pd.DataFrame, text_column: str) -> list[str | None]:
    """Convert the text column to ``str``/``None`` without inventing content.

    Args:
        frame: Rows to convert.
        text_column: Column holding the review text.

    Returns:
        One entry per row: the text, or ``None`` for a missing value.
    """
    return [
        None if pd.isna(value) else str(value) for value in frame[text_column].tolist()
    ]


def run_fingerprint(
    *,
    input_path: Path,
    checkpoint: dict[str, Any],
    chunk_rows: int,
    limit: int | None,
) -> dict[str, Any]:
    """Describe the inputs that make completed parts reusable.

    Args:
        input_path: Review parquet that was scored.
        checkpoint: Checkpoint identity (from :func:`describe_checkpoint`).
        chunk_rows: Rows per chunk.
        limit: Row limit applied to the input, or ``None``.

    Returns:
        A JSON-serialisable fingerprint compared by ``--resume``.
    """
    return {
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "checkpoint_model_meta_sha256": checkpoint["model_meta_sha256"],
        "checkpoint_weights_sha256": checkpoint["weights_sha256"],
        "chunk_rows": chunk_rows,
        "limit": limit,
    }


def _part_path(parts_dir: Path, start: int, stop: int) -> Path:
    """Name one chunk's part file by its half-open row range.

    Args:
        parts_dir: Directory holding the parts.
        start: First row index in the chunk.
        stop: One past the last row index.

    Returns:
        The part path.
    """
    return parts_dir / f"part-{start:09d}-{stop:09d}.parquet"


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    """Write a parquet atomically (temporary sibling, then replace).

    Args:
        frame: Rows to write.
        path: Destination path.
    """
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _chunk_frame(
    predictor: SentimentPredictor, frame: pd.DataFrame, text_column: str
) -> pd.DataFrame:
    """Score one chunk and build its output frame.

    Args:
        predictor: Loaded checkpoint.
        frame: Rows to score.
        text_column: Column holding the review text.

    Returns:
        A frame in :data:`PREDICTION_COLUMNS` order, aligned with ``frame``.
    """
    result = predictor.predict_proba(normalise_texts(frame, text_column))
    labels = pd.array(
        [int(label) if label >= 0 else None for label in result.labels.tolist()],
        dtype="Int64",
    )
    return pd.DataFrame(
        {
            "review_id": frame[REVIEW_ID_COLUMN].astype("string").tolist(),
            "prob_negative": result.probabilities[:, 0],
            "prob_neutral": result.probabilities[:, 1],
            "prob_positive": result.probabilities[:, 2],
            "predicted_label": labels,
            "has_prediction": result.has_prediction,
        },
        columns=list(PREDICTION_COLUMNS),
    )


def score_chunks(
    frame: pd.DataFrame,
    predictor: SentimentPredictor,
    *,
    text_column: str,
    parts_dir: Path,
    chunk_rows: int,
    resume: bool,
) -> tuple[int, int]:
    """Score the frame chunk by chunk, reusing completed parts on resume.

    Args:
        frame: Rows to score, in input order.
        predictor: Loaded checkpoint.
        text_column: Column holding the review text.
        parts_dir: Directory for the part files (created if missing).
        chunk_rows: Rows per chunk.
        resume: Keep existing parts instead of clearing them.

    Returns:
        ``(scored_rows, reused_parts)``.

    Raises:
        DataError: If an existing part does not cover its row range.
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    scored = 0
    reused = 0
    for start in range(0, len(frame), chunk_rows):
        stop = min(start + chunk_rows, len(frame))
        part = _part_path(parts_dir, start, stop)
        if resume and part.is_file():
            rows = len(pd.read_parquet(part, columns=[REVIEW_ID_COLUMN]))
            if rows != stop - start:
                raise DataError(
                    f"resumed part {part} holds {rows:,} rows, expected {stop - start:,}; "
                    "delete the parts directory and score again"
                )
            logger.info("part_reused rows=%d path=%s", rows, part)
            reused += 1
            continue
        chunk = frame.iloc[start:stop].reset_index(drop=True)
        _write_frame(_chunk_frame(predictor, chunk, text_column), part)
        scored += len(chunk)
        logger.info(
            "part_scored start=%d stop=%d rows=%d path=%s", start, stop, len(chunk), part
        )
    return scored, reused


def assemble(parts_dir: Path, frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Concatenate the parts and verify they still describe the input rows.

    Args:
        parts_dir: Directory holding the part files.
        frame: Input rows (``review_id`` order is the expected output order).
        output_path: Final parquet path.

    Returns:
        The assembled prediction frame.

    Raises:
        DataError: If a part is missing or the ids do not match the input order.
    """
    parts = sorted(parts_dir.glob("part-*.parquet"))
    if not parts:
        raise DataError(f"no scored parts in {parts_dir}; nothing to assemble")
    assembled = pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)
    if len(assembled) != len(frame):
        raise DataError(
            f"assembled {len(assembled):,} prediction rows for {len(frame):,} reviews"
        )
    if assembled[REVIEW_ID_COLUMN].tolist() != frame[REVIEW_ID_COLUMN].astype("string").tolist():
        raise DataError("assembled prediction ids do not match the input order")
    _write_frame(assembled.loc[:, list(PREDICTION_COLUMNS)], output_path)
    return assembled


def build_sidecar(
    *,
    run: PredictionRun,
    checkpoint: dict[str, Any],
    frame: pd.DataFrame,
    output_sha256: str,
    batch_size: int,
    chunk_rows: int,
    limit: int | None,
    chunks: list[dict[str, int]],
) -> dict[str, Any]:
    """Assemble the sidecar payload for one finished run.

    Args:
        run: Printed result of the run.
        checkpoint: Checkpoint identity.
        frame: Input rows that were scored.
        output_sha256: SHA-256 of the written parquet.
        batch_size: Rows per forward pass.
        chunk_rows: Rows per chunk.
        limit: Row limit, or ``None``.
        chunks: ``start``/``stop``/``rows`` for every part.

    Returns:
        JSON-serialisable payload.
    """
    return {
        "script": SCRIPT,
        "predictions": {
            "path": str(run.output_path),
            "rows": int(len(frame)),
            "rows_with_prediction": run.rows_with_prediction,
            "rows_without_text": int(len(frame) - run.rows_with_prediction),
            "columns": list(PREDICTION_COLUMNS),
            "sha256": output_sha256,
        },
        "checkpoint": checkpoint,
        "input": {
            "path": str(run.input_path),
            "rows": int(len(frame)),
            "sha256": file_sha256(run.input_path),
            "limit": limit,
        },
        "batching": {
            "batch_size": batch_size,
            "chunk_rows": chunk_rows,
            "device": run.device,
        },
        "chunks": chunks,
        "elapsed_seconds": round(run.elapsed_seconds, 3),
        "rows_per_second": run.rows_per_second,
        "notes": [
            "Rows without usable text have NaN probabilities and a NULL "
            "predicted_label; no probability is invented for them.",
            "Labels are weak supervision from star ratings; the probabilities are "
            "the model's agreement with that proxy, not human-judged sentiment.",
        ],
    }


def run(settings: Settings, args: argparse.Namespace) -> PredictionRun:
    """Score the review table and write the parquet plus its sidecar.

    Args:
        settings: Loaded project settings.
        args: Parsed arguments.

    Returns:
        The run summary.

    Raises:
        DataError: On a missing table/checkpoint, an unusable part, or a resume
            fingerprint mismatch.
        ValueError: On a non-positive row count.
    """
    sentiment = settings.review_intel.sentiment
    prediction = sentiment.prediction
    checkpoint_path = Path(args.checkpoint or prediction.checkpoint)
    input_path = Path(args.input or prediction.input_path)
    output_path = Path(args.output or prediction.output_path)
    batch_size = args.batch_size if args.batch_size is not None else prediction.batch_size
    chunk_rows = args.chunk_rows if args.chunk_rows is not None else prediction.chunk_rows
    if batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {batch_size}")
    if chunk_rows < 1:
        raise ValueError(f"--chunk-rows must be >= 1, got {chunk_rows}")
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be >= 1, got {args.limit}")

    started = time.perf_counter()
    frame = read_reviews(input_path, sentiment.text_column)
    if args.limit is not None:
        frame = frame.head(args.limit).reset_index(drop=True)
        logger.info("limit applied rows=%d", len(frame))

    identity = describe_checkpoint(checkpoint_path).to_dict()
    predictor = SentimentPredictor.from_checkpoint(
        checkpoint_path, device=args.device, batch_size=batch_size
    )

    parts_dir = output_path.parent / (output_path.stem + PARTS_DIR_SUFFIX)
    fingerprint = run_fingerprint(
        input_path=input_path, checkpoint=identity, chunk_rows=chunk_rows, limit=args.limit
    )
    marker = parts_dir / RUN_MARKER_FILENAME
    if args.resume:
        if not marker.is_file():
            raise ResumeError(
                f"--resume found no {marker}; start without --resume to score from scratch"
            )
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise ResumeError(
                f"--resume inputs differ from the run in {parts_dir} "
                "(input file, checkpoint or chunk size changed); delete that directory "
                "or re-run without --resume"
            )
        logger.info("resume_matched parts=%s", parts_dir)
    else:
        for stale in parts_dir.glob("part-*.parquet"):
            stale.unlink()
        if marker.exists():
            marker.unlink()

    # The marker is written *before* scoring: a disconnect during chunk k leaves
    # the completed parts and a marker that still describes them, which is
    # exactly what ``--resume`` needs to validate.
    parts_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"fingerprint": fingerprint}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    scored, reused = score_chunks(
        frame,
        predictor,
        text_column=sentiment.text_column,
        parts_dir=parts_dir,
        chunk_rows=chunk_rows,
        resume=args.resume,
    )
    assembled = assemble(parts_dir, frame, output_path)
    rows_with_prediction = int(assembled["has_prediction"].sum())

    elapsed = time.perf_counter() - started
    rows_per_second = scored / elapsed if scored else None
    chunks: list[dict[str, int]] = []
    for start in range(0, len(frame), chunk_rows):
        stop = min(start + chunk_rows, len(frame))
        chunks.append({"start": start, "stop": stop, "rows": stop - start})
    result = PredictionRun(
        input_path=input_path,
        output_path=output_path,
        sidecar=sidecar_path(output_path),
        rows=len(frame),
        rows_with_prediction=rows_with_prediction,
        scored_rows=scored,
        elapsed_seconds=elapsed,
        rows_per_second=rows_per_second,
        device=predictor.device.type,
    )
    payload = build_sidecar(
        run=result,
        checkpoint=identity,
        frame=frame,
        output_sha256=file_sha256(output_path),
        batch_size=batch_size,
        chunk_rows=chunk_rows,
        limit=args.limit,
        chunks=chunks,
    )
    result.sidecar.parent.mkdir(parents=True, exist_ok=True)
    result.sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print()
    print(f"rows scored          : {scored:,} ({reused:,} chunk(s) reused)")
    if rows_per_second is None:
        print("rows per second       : n/a (nothing scored in this invocation)")
    else:
        print(f"rows per second      : {rows_per_second:,.1f}")
    print(f"elapsed              : {elapsed:,.1f}s")
    print(f"rows with prediction : {rows_with_prediction:,} of {len(frame):,}")
    print(f"predictions          : {output_path}")
    print(f"sidecar              : {result.sidecar}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Score the review table and print throughput and output paths.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 written, 1 failed, 2 resume mismatch or bad flag.
    """
    args = parse_args(argv)
    try:
        settings: Settings = load_settings(args.config)
    except ConfigError as exc:
        logger.error("config_error path=%s detail=%s", args.config, exc)
        return 1

    try:
        run(settings, args)
    except ResumeError as exc:
        logger.error("resume_failed detail=%s", exc)
        return 2
    except (ConfigError, DataError, ValueError) as exc:
        logger.error("prediction_failed detail=%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

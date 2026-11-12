"""Load the per-review sentiment predictions into the SQLite serving cache.

``scripts/predict_reviews.py`` writes a parquet with one row per review plus a
sidecar JSON recording the checkpoint it was produced with. This script checks
that contract before writing anything, then upserts the prediction columns into
``review_features``:

* the parquet hash still matches the sidecar (the file was not replaced),
* the checkpoint on disk still matches the recorded hashes of ``model_meta.json``
  and the weights,
* every ``review_id`` exists in ``reviews`` (foreign keys are ON),
* every review with usable text has a prediction, and no review without text has
  one,
* each probability lies in ``[0, 1]``, the three sum to 1 within
  ``PROBABILITY_TOLERANCE``, and ``predicted_label`` is the argmax.

The upsert is idempotent: re-running keeps the same row count and overwrites the
same rows. If a check fails the transaction is rolled back, so a bad file never
leaves a half-written table.

Usage::

    python scripts/ingest_predictions.py
    python scripts/ingest_predictions.py --checkpoint models/sentiment_distilbert/best
    python scripts/ingest_predictions.py --input /tmp/preds.parquet

Exit codes: 0 - verified and ingested; 1 - a check failed or a file is missing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from munasib.core.config import Settings, load_settings
from munasib.core.constants import SENTIMENT_LABELS
from munasib.core.exceptions import ConfigError, DataError
from munasib.db.repositories import ReviewFeatureRepository
from munasib.db.session import (
    create_engine_from_settings,
    create_session_factory,
    init_db,
    session_scope,
)
from munasib.review_intel.sentiment.predictor import (
    CheckpointIdentity,
    describe_checkpoint,
    file_sha256,
    sidecar_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_predictions")

#: Columns the prediction parquet must carry, in this order.
PREDICTION_COLUMNS: tuple[str, ...] = (
    "review_id",
    "prob_negative",
    "prob_neutral",
    "prob_positive",
    "predicted_label",
    "has_prediction",
)

#: Probability columns, in :data:`munasib.core.constants.SENTIMENT_LABELS` order.
PROBABILITY_COLUMNS: tuple[str, ...] = (
    "prob_negative",
    "prob_neutral",
    "prob_positive",
)

#: Allowed absolute deviation of a probability row's sum from 1.0.
PROBABILITY_TOLERANCE = 1e-5

#: Reviews checked per foreign-key query (SQLite bound-variable limit).
FK_BATCH_SIZE = 500

#: Rows per upsert statement.
UPSERT_BATCH_SIZE = 1_000

#: Sidecar keys that must be present before anything is trusted.
REQUIRED_SIDECAR_KEYS = ("predictions", "checkpoint", "input")


@dataclass(frozen=True)
class IngestReport:
    """Counts printed after one verified ingest.

    Attributes:
        predictions: Prediction parquet that was loaded.
        sidecar: Sidecar JSON that was verified.
        rows: Rows in the prediction file.
        rows_with_prediction: Rows that carry probabilities.
        rows_without_text: Rows with ``has_prediction=False``.
        features_before: ``review_features`` row count before the upsert.
        features_after: Row count after the upsert.
        written: Rows written by the upsert.
        checkpoint: Checkpoint directory whose hashes were verified.
    """

    predictions: Path
    sidecar: Path
    rows: int
    rows_with_prediction: int
    rows_without_text: int
    features_before: int
    features_after: int
    written: int
    checkpoint: str

    def lines(self) -> list[str]:
        """Render the report as aligned ``label : value`` lines.

        Returns:
            Stable sequence of lines with the counts the checks produced.
        """
        return [
            f"{'predictions':<28}: {self.predictions}",
            f"{'sidecar':<28}: {self.sidecar}",
            f"{'checkpoint (hashes verified)':<28}: {self.checkpoint}",
            f"{'rows in file':<28}: {self.rows:,}",
            f"{'rows with prediction':<28}: {self.rows_with_prediction:,}",
            f"{'rows without text':<28}: {self.rows_without_text:,}",
            f"{'review_features before':<28}: {self.features_before:,}",
            f"{'review_features after':<28}: {self.features_after:,}",
            f"{'rows written (upsert)':<28}: {self.written:,}",
        ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Verify and ingest the per-review sentiment predictions into the "
            "review_features table."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Prediction parquet (default: review_intel.sentiment.prediction.output_path)."
        ),
    )
    parser.add_argument(
        "--sidecar",
        default=None,
        help="Sidecar JSON (default: <input stem>.meta.json next to the parquet).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint directory to verify against (default: the path recorded in "
            "the sidecar). Use this after copying the predictions to another machine."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=UPSERT_BATCH_SIZE,
        help=f"Rows per upsert statement (default: {UPSERT_BATCH_SIZE}).",
    )
    return parser.parse_args(argv)


def load_sidecar(path: Path) -> dict[str, Any]:
    """Read and structurally validate the sidecar JSON.

    Args:
        path: Sidecar file.

    Returns:
        Parsed payload.

    Raises:
        DataError: If the file is missing, not JSON, or lacks a required block.
    """
    if not path.is_file():
        raise DataError(
            f"sidecar not found: {path} - the prediction file must travel with its "
            "checkpoint identity"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"sidecar is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataError(f"sidecar must hold a JSON object: {path}")
    missing = [key for key in REQUIRED_SIDECAR_KEYS if key not in payload]
    if missing:
        raise DataError(f"sidecar is missing required block(s) {missing}: {path}")
    return payload


def verify_sidecar(
    sidecar: dict[str, Any], predictions_path: Path, checkpoint_dir: Path | None
) -> CheckpointIdentity:
    """Check that the parquet and the checkpoint still match the sidecar.

    Args:
        sidecar: Parsed sidecar payload.
        predictions_path: Prediction parquet.
        checkpoint_dir: Explicit checkpoint directory, or ``None`` to use the
            path recorded in the sidecar.

    Returns:
        Identity of the verified checkpoint.

    Raises:
        DataError: If the parquet hash differs, the checkpoint directory is
            missing, or its hashes differ from the recorded ones.
    """
    try:
        recorded = sidecar["predictions"]
        checkpoint = sidecar["checkpoint"]
        expected_sha = str(recorded["sha256"])
        recorded_meta = str(checkpoint["model_meta_sha256"])
        recorded_weights = str(checkpoint["weights_sha256"])
        recorded_path = str(checkpoint["path"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(f"sidecar is missing a required hash or path: {exc}") from exc

    actual_sha = file_sha256(predictions_path)
    if actual_sha != expected_sha:
        raise DataError(
            "prediction parquet hash does not match its sidecar "
            f"(sidecar {expected_sha[:12]}..., file {actual_sha[:12]}...); the file "
            "was modified or replaced after scoring"
        )

    directory = checkpoint_dir if checkpoint_dir is not None else Path(recorded_path)
    if not directory.is_dir():
        raise DataError(
            f"checkpoint directory not found: {directory}; pass --checkpoint with the "
            "checkpoint the predictions were produced with, so their identity can be verified"
        )
    identity = describe_checkpoint(directory)
    if (
        identity.model_meta_sha256 != recorded_meta
        or identity.weights_sha256 != recorded_weights
    ):
        raise DataError(
            f"checkpoint {directory} does not match the sidecar: "
            f"model_meta.json {identity.model_meta_sha256[:12]}... vs recorded "
            f"{recorded_meta[:12]}..., weights {identity.weights_sha256[:12]}... vs "
            f"recorded {recorded_weights[:12]}..."
        )
    return identity


def validate_frame(frame: pd.DataFrame) -> None:
    """Validate the prediction rows before any database write.

    Args:
        frame: Prediction parquet contents.

    Raises:
        DataError: If the columns, labels or probabilities violate the contract.
    """
    if list(frame.columns) != list(PREDICTION_COLUMNS):
        raise DataError(
            f"prediction file columns must be {list(PREDICTION_COLUMNS)}, "
            f"got {list(frame.columns)}"
        )
    if frame.empty:
        raise DataError("prediction file is empty")
    if frame["review_id"].isna().any():
        raise DataError("prediction file has rows without a review_id")
    if frame["review_id"].duplicated().any():
        duplicates = int(frame["review_id"].duplicated().sum())
        raise DataError(f"prediction file has {duplicates:,} duplicate review_id value(s)")
    if frame["has_prediction"].isna().any():
        raise DataError("has_prediction must never be null")

    has_prediction = frame["has_prediction"].astype(bool).to_numpy()
    probabilities = frame.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype="float64")
    covered = probabilities[has_prediction]
    uncovered = probabilities[~has_prediction]

    if np.isnan(covered).any():
        raise DataError("rows with has_prediction=True have a missing probability")
    if covered.min(initial=0.0) < 0.0 or covered.max(initial=0.0) > 1.0:
        raise DataError("probabilities must lie in [0, 1]")
    deviation = np.abs(covered.sum(axis=1) - 1.0)
    if deviation.size and float(deviation.max()) > PROBABILITY_TOLERANCE:
        raise DataError(
            f"probabilities must sum to 1 within {PROBABILITY_TOLERANCE:g}; "
            f"maximum deviation is {float(deviation.max()):.6f}"
        )
    if uncovered.size and not np.isnan(uncovered).all():
        raise DataError(
            "rows with has_prediction=False must have NaN probabilities, not a value"
        )

    labels = frame["predicted_label"]
    if labels[has_prediction].isna().any():
        raise DataError("rows with has_prediction=True have a null predicted_label")
    if labels[~has_prediction].notna().any():
        raise DataError("rows with has_prediction=False must have a null predicted_label")
    valid_labels = labels[has_prediction].astype("int64").to_numpy()
    if not set(valid_labels.tolist()) <= set(SENTIMENT_LABELS):
        raise DataError(
            f"predicted_label must be one of {list(SENTIMENT_LABELS)}, "
            f"got {sorted(set(valid_labels.tolist()) - set(SENTIMENT_LABELS))}"
        )
    argmax = covered.argmax(axis=1)
    if not np.array_equal(argmax, valid_labels):
        mismatches = int((argmax != valid_labels).sum())
        raise DataError(
            f"{mismatches:,} predicted_label value(s) are not the argmax of the probabilities"
        )


def _rows_for_upsert(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert the validated frame into upsert payloads.

    Args:
        frame: Validated prediction rows.

    Returns:
        Mappings with SQL ``NULL`` (``None``) for rows without text.
    """
    rows: list[dict[str, object]] = []
    for record in frame.itertuples(index=False):
        if bool(record.has_prediction):
            rows.append(
                {
                    "review_id": str(record.review_id),
                    "predicted_label": int(record.predicted_label),
                    "prob_negative": float(record.prob_negative),
                    "prob_neutral": float(record.prob_neutral),
                    "prob_positive": float(record.prob_positive),
                }
            )
        else:
            rows.append(
                {
                    "review_id": str(record.review_id),
                    "predicted_label": None,
                    "prob_negative": None,
                    "prob_neutral": None,
                    "prob_positive": None,
                }
            )
    return rows


def ingest_frame(
    factory: Any,
    frame: pd.DataFrame,
    *,
    checkpoint: str,
    batch_size: int = UPSERT_BATCH_SIZE,
    tolerance: float = PROBABILITY_TOLERANCE,
) -> tuple[int, int, int]:
    """Upsert one validated prediction frame inside a single transaction.

    Args:
        factory: Session factory bound to the serving cache.
        frame: Validated prediction rows.
        checkpoint: Verified checkpoint identity string recorded in the report.
        batch_size: Rows per upsert statement.
        tolerance: Allowed probability-sum deviation, re-checked from the table.

    Returns:
        ``(rows_before, rows_after, written)``.

    Raises:
        DataError: If a review id is unknown, coverage is incomplete, or a stored
            probability row is invalid (the transaction is rolled back).
    """
    with session_scope(factory) as session:
        repository = ReviewFeatureRepository(session)
        before = repository.count_features()
        logger.info(
            "ingest_start checkpoint=%s rows=%d features_before=%d",
            checkpoint,
            len(frame),
            before,
        )
        missing = repository.missing_review_ids(
            frame["review_id"].tolist(), batch_size=FK_BATCH_SIZE
        )
        if missing:
            preview = ", ".join(missing[:5])
            raise DataError(
                f"{len(missing):,} prediction review_id(s) do not exist in reviews "
                f"(first: {preview}); the prediction file belongs to another database"
            )

        written = repository.upsert_predictions(
            _rows_for_upsert(frame), batch_size=batch_size
        )
        after = repository.count_features()

        uncovered = repository.count_text_reviews_without_predictions()
        if uncovered:
            raise DataError(
                f"{uncovered:,} review(s) with usable text have no prediction; "
                "re-run scripts/predict_reviews.py over the whole table"
            )
        on_textless = repository.count_predictions_on_textless_reviews()
        if on_textless:
            raise DataError(
                f"{on_textless:,} prediction(s) were attached to reviews without text"
            )
        violations = repository.count_probability_violations(tolerance=tolerance)
        if violations:
            raise DataError(
                f"{violations:,} stored probability row(s) do not sum to 1 within "
                f"{tolerance:g}"
            )
    return before, after, written


def run(settings: Settings, args: argparse.Namespace) -> IngestReport:
    """Verify and ingest one prediction file.

    Args:
        settings: Loaded project settings.
        args: Parsed arguments.

    Returns:
        The ingest counts.

    Raises:
        DataError: If a verification or coverage check fails.
        ValueError: On an invalid batch size.
    """
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    prediction_config = settings.review_intel.sentiment.prediction
    predictions_path = Path(args.input or prediction_config.output_path)
    sidecar_file = Path(args.sidecar) if args.sidecar else sidecar_path(predictions_path)
    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else None

    if not predictions_path.is_file():
        raise DataError(f"prediction file not found: {predictions_path}")
    sidecar = load_sidecar(sidecar_file)
    identity = verify_sidecar(sidecar, predictions_path, checkpoint_dir)
    logger.info(
        "identity_verified checkpoint=%s meta=%s weights=%s",
        identity.path,
        identity.model_meta_sha256[:12],
        identity.weights_sha256[:12],
    )

    frame = pd.read_parquet(predictions_path)
    validate_frame(frame)

    engine = create_engine_from_settings(settings)
    try:
        init_db(engine)
        factory = create_session_factory(engine)
        before, after, written = ingest_frame(
            factory, frame, checkpoint=identity.path, batch_size=args.batch_size
        )
    finally:
        engine.dispose()

    return IngestReport(
        predictions=predictions_path,
        sidecar=sidecar_file,
        rows=len(frame),
        rows_with_prediction=int(frame["has_prediction"].astype(bool).sum()),
        rows_without_text=int((~frame["has_prediction"].astype(bool)).sum()),
        features_before=before,
        features_after=after,
        written=written,
        checkpoint=identity.path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the predictions, ingest them and print the counts.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 = ingested, 1 = a check failed).
    """
    args = parse_args(argv)
    try:
        settings: Settings = load_settings(args.config)
    except ConfigError as exc:
        logger.error("config_error path=%s detail=%s", args.config, exc)
        return 1

    try:
        report = run(settings, args)
    except (ConfigError, DataError, ValueError) as exc:
        logger.error("ingest_failed detail=%s", exc)
        return 1

    print()
    for line in report.lines():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

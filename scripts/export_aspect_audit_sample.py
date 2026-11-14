"""Export or evaluate the manual aspect-quality audit sample.

The audit is a long-format CSV: one row per ``(review, aspect)`` pair drawn
uniformly from all stored review-aspect mentions. The annotation columns
``is_correct`` and ``notes`` are written blank. No precision is printed by the
export path - a number only appears after a human fills ``is_correct`` and runs
``--evaluate``.

Usage::

    python scripts/export_aspect_audit_sample.py
    python scripts/export_aspect_audit_sample.py --sample-size 200 --seed 7
    python scripts/export_aspect_audit_sample.py --evaluate

Exit codes:
  0 - sample written, or evaluation completed with no blanks.
  1 - missing/incomplete data, blank or invalid annotations, or a bad config.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from munasib.core.config import Settings, load_settings
from munasib.core.exceptions import ConfigError, DataError
from munasib.db.models import Product, Review, ReviewFeature
from munasib.db.session import create_engine_from_settings, create_session_factory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
)
logger = logging.getLogger("munasib.aspect_audit")

#: Required columns of the audit CSV (annotation columns last).
AUDIT_COLUMNS = (
    "review_id",
    "asin",
    "product_title",
    "aspect",
    "predicted_label",
    "helpful_votes",
    "review_text",
    "is_correct",
    "notes",
)


@dataclass(frozen=True)
class AuditRow:
    """One review-aspect mention in the audit sample."""

    review_id: str
    asin: str
    product_title: str
    aspect: str
    predicted_label: int | None
    helpful_votes: int
    review_text: str

    def to_row(self) -> dict[str, object]:
        """Return the CSV payload with blank annotation columns.

        Returns:
            Mapping for :class:`csv.DictWriter`.
        """
        return {
            "review_id": self.review_id,
            "asin": self.asin,
            "product_title": self.product_title,
            "aspect": self.aspect,
            "predicted_label": "" if self.predicted_label is None else self.predicted_label,
            "helpful_votes": self.helpful_votes,
            "review_text": self.review_text,
            "is_correct": "",
            "notes": "",
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Export or evaluate the manual aspect-quality audit sample."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Mentions to sample (default: review_intel.aspects.audit_sample_size).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed (default: project.random_seed).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Audit CSV path (default: review_intel.aspects.audit_sample_path).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Read the annotated CSV and report precision (no export).",
    )
    return parser.parse_args(argv)


def _aspects(value: object) -> list[str]:
    """Normalise a stored aspect payload.

    Args:
        value: JSON-decoded cell.

    Returns:
        Aspect strings.

    Raises:
        DataError: If the payload is not a list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(aspect) for aspect in value]
    raise DataError(
        f"review_features.aspects_json must be a JSON list, got {type(value).__name__}"
    )


def sample_mentions(
    settings: Settings,
    *,
    sample_size: int,
    seed: int,
) -> list[AuditRow]:
    """Draw a deterministic uniform reservoir sample of review-aspect mentions.

    Args:
        settings: Loaded project settings.
        sample_size: Maximum mentions to return.
        seed: Sampling seed.

    Returns:
        Sampled rows sorted by ``(review_id, aspect)``.

    Raises:
        DataError: If there are no stored review-aspect mentions.
    """
    if sample_size < 1:
        raise ValueError(f"sample_size must be >= 1, got {sample_size}")
    engine = create_engine_from_settings(settings)
    factory = create_session_factory(engine)
    rng = random.Random(seed)
    reservoir: list[AuditRow] = []
    seen = 0
    try:
        session = factory()
        try:
            statement = (
                select(
                    Review.review_id,
                    Review.asin,
                    Product.title,
                    ReviewFeature.predicted_label,
                    Review.helpful_votes,
                    Review.review_text_raw,
                    Review.review_text_clean,
                    ReviewFeature.aspects_json,
                )
                .join(ReviewFeature, ReviewFeature.review_id == Review.review_id)
                .join(Product, Product.asin == Review.asin)
                .where(ReviewFeature.aspects_json.is_not(None))
                .order_by(Review.review_id)
            )
            for (
                review_id,
                asin,
                title,
                label,
                helpful_votes,
                raw_text,
                clean_text,
                aspects_json,
            ) in session.execute(statement.execution_options(yield_per=5000)):
                aspects = _aspects(aspects_json)
                if not aspects:
                    continue
                text = str(raw_text or clean_text or "")
                for aspect in aspects:
                    row = AuditRow(
                        review_id=str(review_id),
                        asin=str(asin),
                        product_title=str(title or ""),
                        aspect=aspect,
                        predicted_label=None if label is None else int(label),
                        helpful_votes=int(helpful_votes or 0),
                        review_text=text,
                    )
                    seen += 1
                    if len(reservoir) < sample_size:
                        reservoir.append(row)
                    else:
                        replacement = rng.randrange(seen)
                        if replacement < sample_size:
                            reservoir[replacement] = row
        finally:
            session.close()
    finally:
        engine.dispose()
    if not reservoir:
        raise DataError(
            "no stored review-aspect mentions; run "
            "'python scripts/compute_aspects.py --reviews' first"
        )
    reservoir.sort(key=lambda row: (row.review_id, row.aspect))
    return reservoir


def write_sample(rows: Sequence[AuditRow], path: str | Path) -> int:
    """Write the blank-annotation audit CSV.

    Args:
        rows: Sampled mentions.
        path: Destination CSV path.

    Returns:
        Number of rows written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AUDIT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())
    return len(rows)


def evaluate_sample(path: str | Path) -> tuple[int, int, float]:
    """Compute precision from human annotations.

    Args:
        path: Annotated audit CSV.

    Returns:
        ``(correct, labelled, precision)``.

    Raises:
        DataError: If the file is missing, lacks columns, has blank
            annotations, or uses values other than yes/no.
    """
    target = Path(path)
    if not target.exists():
        raise DataError(f"audit CSV not found: {target}")
    correct = 0
    labelled = 0
    blanks = 0
    with target.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in AUDIT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise DataError(f"audit CSV {target} lacks column(s) {missing}")
        for line_number, row in enumerate(reader, start=2):
            raw = (row.get("is_correct") or "").strip().lower()
            if raw == "":
                blanks += 1
                continue
            if raw in {"yes", "y", "true", "1", "correct"}:
                correct += 1
            elif raw in {"no", "n", "false", "0", "incorrect"}:
                pass
            else:
                raise DataError(
                    f"audit CSV {target} line {line_number} has is_correct={raw!r}; "
                    "use yes/no"
                )
            labelled += 1
    if blanks:
        raise DataError(
            f"audit CSV {target} has {blanks:,} blank is_correct value(s); "
            "precision is not reported until every sampled mention is annotated"
        )
    if labelled == 0:
        raise DataError(f"audit CSV {target} has no annotated rows")
    return correct, labelled, correct / labelled


def main(argv: Sequence[str] | None = None) -> int:
    """Export or evaluate the audit sample.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 success, 1 failure).
    """
    args = parse_args(argv)
    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        logger.error("config_error path=%s detail=%s", args.config, exc)
        return 1
    aspects_config = settings.review_intel.aspects
    output = Path(args.output or aspects_config.audit_sample_path)

    try:
        if args.evaluate:
            correct, labelled, precision = evaluate_sample(output)
            print(f"annotated mentions          : {labelled:,}")
            print(f"correct aspects             : {correct:,}")
            print(f"aspect precision            : {precision:.4f}")
            return 0
        sample_size = args.sample_size or aspects_config.audit_sample_size
        seed = settings.project.random_seed if args.seed is None else args.seed
        rows = sample_mentions(settings, sample_size=sample_size, seed=seed)
        written = write_sample(rows, output)
        print(f"audit sample                : {output}")
        print(f"sampled aspect mentions     : {written:,}")
        print(
            "annotation                  : fill is_correct (yes/no) and notes, then run "
            "--evaluate"
        )
        return 0
    except (DataError, ValueError) as exc:
        logger.error("audit_failed detail=%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

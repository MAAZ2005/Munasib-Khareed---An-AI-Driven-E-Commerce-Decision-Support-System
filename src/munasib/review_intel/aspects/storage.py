"""Database reads and parquet writes for the aspect stage.

The read side is deliberately narrow:

* :func:`iter_text_reviews` performs keyset pagination over usable review text,
  so a 425k-review extraction never loads the whole table into memory.
* :func:`iter_prediction_rows` streams every review that has a stored predicted
  label, sorted by product then review id. Reviews **without** a prediction are
  excluded from aspect polarity, but they are also not silently assigned a
  label.

The write side produces the intermediate ``review_aspects.parquet`` with one row
per review and an ``aspects`` list. Writes go to a temporary file and replace the
target only on success, so an interrupted run never leaves a half parquet file
that looks valid.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from munasib.core.exceptions import DataError
from munasib.db.models import Review, ReviewFeature
from munasib.review_intel.aspects.aggregation import AspectPredictionRow
from munasib.review_intel.aspects.cache import text_review_condition

#: Parquet schema of the per-review aspect artifact.
REVIEW_ASPECTS_SCHEMA = pa.schema(
    [
        pa.field("review_id", pa.string(), nullable=False),
        pa.field("aspects", pa.list_(pa.string()), nullable=False),
    ]
)


@dataclass(frozen=True)
class TextReviewRow:
    """One review with usable text (extraction input)."""

    review_id: str
    asin: str
    text: str


def count_text_reviews(session: Session) -> int:
    """Count reviews with usable clean text.

    Args:
        session: Open session.

    Returns:
        Number of extraction inputs.
    """
    total = session.scalar(
        select(func.count()).select_from(Review).where(*text_review_condition())
    )
    return int(total or 0)


def iter_text_reviews(
    session: Session,
    *,
    after_review_id: str | None = None,
    limit: int | None = None,
) -> list[TextReviewRow]:
    """Read one keyset page of reviews with usable text.

    Args:
        session: Open session.
        after_review_id: Return rows whose ``review_id`` is strictly greater
            than this value (None starts at the beginning).
        limit: Maximum rows (None = everything after the cursor).

    Returns:
        Review rows in ascending ``review_id`` order, truncated to ``limit``.

    Raises:
        ValueError: If ``limit`` is negative.
    """
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")
    statement = (
        select(Review.review_id, Review.asin, Review.review_text_clean)
        .where(*text_review_condition())
        .order_by(Review.review_id)
    )
    if after_review_id is not None:
        statement = statement.where(Review.review_id > after_review_id)
    if limit is not None:
        statement = statement.limit(limit)
    return [
        TextReviewRow(review_id=str(review_id), asin=str(asin), text=str(text))
        for review_id, asin, text in session.execute(statement)
    ]


def _as_aspect_list(value: object) -> list[str]:
    """Normalise a stored ``aspects_json`` value into a string list.

    Args:
        value: JSON-decoded cell.

    Returns:
        Stored aspects; an empty list when the review has no stored aspects.

    Raises:
        DataError: If the stored value is not a list (a corrupt row must not be
            silently treated as "no aspects").
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(aspect) for aspect in value]
    raise DataError(
        f"review_features.aspects_json must be a JSON list, got {type(value).__name__}"
    )


def iter_prediction_rows(session: Session) -> Iterator[AspectPredictionRow]:
    """Stream predicted reviews and their stored aspects for aggregation.

    Rows are sorted by ``(asin, review_id)`` so
    :func:`~munasib.review_intel.aspects.aggregation.iter_product_aspects` can
    group them with constant memory per product.

    Args:
        session: Open session.

    Yields:
        One :class:`AspectPredictionRow` per review with a predicted label.

    Raises:
        DataError: If a stored aspect payload is not a JSON list.
    """
    statement = (
        select(
            Review.asin,
            Review.review_id,
            ReviewFeature.predicted_label,
            ReviewFeature.aspects_json,
            func.coalesce(Review.review_text_raw, Review.review_text_clean, ""),
            Review.helpful_votes,
        )
        .join(ReviewFeature, ReviewFeature.review_id == Review.review_id)
        .where(ReviewFeature.predicted_label.is_not(None))
        .order_by(Review.asin, Review.review_id)
    )
    result = session.execute(statement.execution_options(yield_per=5000))
    for asin, review_id, label, aspects_json, text, helpful_votes in result:
        yield AspectPredictionRow(
            asin=str(asin),
            review_id=str(review_id),
            predicted_label=int(label),
            aspects=_as_aspect_list(aspects_json),
            text=str(text or ""),
            helpful_votes=int(helpful_votes or 0),
        )


class ReviewAspectWriter:
    """Streaming writer for ``review_aspects.parquet``.

    Use as a context manager. Rows are buffered into record batches; the
    temporary file replaces the target only when the context exits cleanly.
    """

    def __init__(self, path: str | Path, *, batch_size: int = 10_000) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.path = Path(path)
        self.batch_size = batch_size
        self._writer: pq.ParquetWriter | None = None
        self._buffer: list[dict[str, object]] = []
        self._temporary: Path | None = None

    def __enter__(self) -> ReviewAspectWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary = self.path.with_suffix(".parquet.tmp")
        self._writer = pq.ParquetWriter(self._temporary, REVIEW_ASPECTS_SCHEMA)
        return self

    def write(self, review_id: str, aspects: Sequence[str]) -> None:
        """Buffer one review's aspects.

        Args:
            review_id: Stable review identifier.
            aspects: Filtered aspect phrases (may be empty).
        """
        self._buffer.append({"review_id": review_id, "aspects": list(aspects)})
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        """Write the buffered rows as one record batch."""
        if self._writer is None:
            raise RuntimeError("ReviewAspectWriter used outside its context manager")
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=REVIEW_ASPECTS_SCHEMA)
        self._writer.write_table(table)
        self._buffer.clear()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> Literal[False]:
        """Close the writer, keeping the file only on a clean exit.

        Args:
            exc_type: Exception type, if any.
            exc: Exception instance, if any.
            traceback: Traceback, if any.

        Returns:
            False so exceptions propagate.
        """
        assert self._writer is not None and self._temporary is not None
        try:
            if exc_type is None:
                self._flush()
        finally:
            self._writer.close()
        if exc_type is None:
            self._temporary.replace(self.path)
        elif self._temporary.exists():
            self._temporary.unlink()
        return False


def write_review_aspects(
    rows: Iterable[tuple[str, Sequence[str]]],
    path: str | Path,
) -> int:
    """Write a complete per-review aspect parquet from an iterable.

    Args:
        rows: ``(review_id, aspects)`` pairs.
        path: Destination parquet.

    Returns:
        Number of rows written.
    """
    written = 0
    with ReviewAspectWriter(path) as writer:
        for review_id, aspects in rows:
            writer.write(review_id, aspects)
            written += 1
    return written


def read_review_aspects(path: str | Path) -> list[tuple[str, list[str]]]:
    """Read the per-review aspect artifact.

    Args:
        path: Parquet path.

    Returns:
        ``(review_id, aspects)`` pairs in stored order.

    Raises:
        DataError: If the artifact is missing or has the wrong schema.
    """
    target = Path(path)
    if not target.exists():
        raise DataError(f"per-review aspect artifact not found: {target}")
    table = pq.read_table(target)
    if table.schema.names != ["review_id", "aspects"]:
        raise DataError(
            f"per-review aspect artifact {target} has columns {table.schema.names}; "
            "expected ['review_id', 'aspects']"
        )
    return [
        (str(row["review_id"]), [str(aspect) for aspect in (row["aspects"] or [])])
        for row in table.to_pylist()
    ]


__all__ = [
    "REVIEW_ASPECTS_SCHEMA",
    "ReviewAspectWriter",
    "TextReviewRow",
    "count_text_reviews",
    "iter_prediction_rows",
    "iter_text_reviews",
    "read_review_aspects",
    "write_review_aspects",
]

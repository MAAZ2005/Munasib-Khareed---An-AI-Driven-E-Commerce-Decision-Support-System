"""Resumable candidate-cache for the expensive spaCy extraction pass.

The cache writes one parquet part per fixed-size review slice. Each part is
written to a temporary file and atomically renamed, so a part that exists is a
complete part. The manifest stores a fingerprint of the extraction settings and
of the review text source; a mismatch refuses to mix parts produced by different
settings (exit code 2 in the CLI), while ``--resume`` simply skips every complete
part.

Candidate parts deliberately store **raw** candidates (the generic-noun
stoplist is not applied here). The vocabulary and per-review stages filter with
the current stoplist, which makes a stoplist extension a fast vocabulary rebuild
instead of another full spaCy pass.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from munasib.core.config import AspectsConfig
from munasib.core.exceptions import DataError
from munasib.db.models import Review

#: Columns written to every candidate part.
PART_COLUMNS: tuple[str, ...] = ("review_id", "asin", "aspects")

MANIFEST_FILENAME = "manifest.json"


class AspectCacheError(DataError):
    """Base class for candidate-cache state errors."""


class AspectCacheMismatch(AspectCacheError):
    """The cache was produced by different extraction settings or source text."""


class AspectCacheIncomplete(AspectCacheError):
    """Parts are missing and the caller did not pass ``--resume``."""


def text_review_condition() -> tuple[Any, ...]:
    """Return the SQL conditions that define a usable review text.

    Returns:
        Conditions matching the same predicate the batch scorer uses: clean text
        is not NULL and is not blank.
    """
    return (
        Review.review_text_clean.is_not(None),
        func.trim(Review.review_text_clean) != "",
    )


def source_signature(session: Session) -> dict[str, object]:
    """Summarise the review text source without hashing every row.

    The count, total character length, and minimum/maximum stable review id
    change when reviews are added, removed, or substantially rewritten. This is
    cheap enough to run on every stage and catches mixing parts from another
    review table.

    Args:
        session: Open session over the serving cache.

    Returns:
        JSON-serialisable source signature.
    """
    row = session.execute(
        select(
            func.count(Review.review_id),
            func.coalesce(func.sum(func.length(Review.review_text_clean)), 0),
            func.min(Review.review_id),
            func.max(Review.review_id),
        ).where(*text_review_condition())
    ).one()
    return {
        "text_reviews": int(row[0] or 0),
        "text_chars": int(row[1] or 0),
        "first_review_id": row[2],
        "last_review_id": row[3],
    }


def extraction_fingerprint(
    config: AspectsConfig,
    source: dict[str, object],
) -> str:
    """Hash the settings that change extracted candidates.

    Generic nouns and batching settings are intentionally excluded: the cache
    is raw, and neither changes a parsed candidate. The source signature is
    included so parts cannot be mixed across different review tables.

    Args:
        config: Aspect configuration.
        source: Output of :func:`source_signature`.

    Returns:
        Hex SHA-1 fingerprint.
    """
    payload = {
        "spacy_model": config.spacy_model,
        "spacy_disable_components": list(config.spacy_disable_components),
        "max_text_chars": config.max_text_chars,
        "min_token_chars": config.min_token_chars,
        "max_aspect_tokens": config.max_aspect_tokens,
        "chunk_rows": config.chunk_rows,
        "source": source,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


@dataclass(frozen=True)
class CandidateManifest:
    """Stored cache identity and shape."""

    fingerprint: str
    payload: dict[str, object]
    total_reviews: int
    chunk_rows: int
    part_count: int

    def to_json(self) -> dict[str, object]:
        """Return the JSON-serialisable manifest.

        Returns:
            Manifest mapping.
        """
        return {
            "fingerprint": self.fingerprint,
            "fingerprint_payload": self.payload,
            "total_reviews": self.total_reviews,
            "chunk_rows": self.chunk_rows,
            "part_count": self.part_count,
        }

    @classmethod
    def from_json(cls, raw: dict[str, object]) -> CandidateManifest:
        """Parse a manifest mapping.

        Args:
            raw: Loaded JSON object.

        Returns:
            Parsed manifest.

        Raises:
            AspectCacheMismatch: If a required field is missing or malformed.
        """
        try:
            payload_raw = raw["fingerprint_payload"]
            if not isinstance(payload_raw, dict):
                raise TypeError("fingerprint_payload must be an object")
            return cls(
                fingerprint=str(raw["fingerprint"]),
                payload={str(key): value for key, value in payload_raw.items()},
                total_reviews=int(str(raw["total_reviews"])),
                chunk_rows=int(str(raw["chunk_rows"])),
                part_count=int(str(raw["part_count"])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AspectCacheMismatch(
                f"candidate manifest is malformed; remove {MANIFEST_FILENAME} and re-extract"
            ) from exc


@dataclass(frozen=True)
class CacheStatus:
    """What the cache currently contains for one run."""

    manifest: CandidateManifest
    missing_parts: list[int]
    completed_parts: list[int]

    @property
    def complete(self) -> bool:
        """Return whether every expected part exists.

        Returns:
            True when no part is missing.
        """
        return not self.missing_parts


class CandidateCache:
    """One manifest plus numbered parquet parts under a directory."""

    def __init__(self, parts_dir: Path, chunk_rows: int) -> None:
        if chunk_rows < 1:
            raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
        self.parts_dir = Path(parts_dir)
        self.chunk_rows = chunk_rows

    @property
    def manifest_path(self) -> Path:
        """Return the manifest path.

        Returns:
            ``<parts_dir>/manifest.json``.
        """
        return self.parts_dir / MANIFEST_FILENAME

    def part_path(self, index: int) -> Path:
        """Return the path of one part.

        Args:
            index: Zero-based part index.

        Returns:
            ``part-00000.parquet`` style path.
        """
        return self.parts_dir / f"part-{index:05d}.parquet"

    def expected_rows(self, index: int, total_reviews: int) -> int:
        """Return the exact row count of one part.

        Args:
            index: Zero-based part index.
            total_reviews: Total text reviews in the source.

        Returns:
            Row count for the part (0 past the end).
        """
        start = index * self.chunk_rows
        return max(0, min(self.chunk_rows, total_reviews - start))

    def part_count(self, total_reviews: int) -> int:
        """Return how many parts a full extraction needs.

        Args:
            total_reviews: Total text reviews in the source.

        Returns:
            ``ceil(total_reviews / chunk_rows)``.
        """
        if total_reviews < 0:
            raise ValueError(f"total_reviews must be non-negative, got {total_reviews}")
        return (total_reviews + self.chunk_rows - 1) // self.chunk_rows

    def load_manifest(self) -> CandidateManifest | None:
        """Read the manifest when it exists.

        Returns:
            Parsed manifest, or None for a fresh directory.

        Raises:
            AspectCacheMismatch: If the manifest is not valid JSON or malformed.
        """
        if not self.manifest_path.exists():
            return None
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AspectCacheMismatch(
                f"candidate manifest {self.manifest_path} is not valid JSON; remove it and re-extract"
            ) from exc
        if not isinstance(raw, dict):
            raise AspectCacheMismatch(
                f"candidate manifest {self.manifest_path} is not a JSON object; remove it and re-extract"
            )
        return CandidateManifest.from_json(raw)

    def _part_complete(self, index: int, expected_rows: int) -> bool:
        """Check a part's presence, schema and exact row count.

        Args:
            index: Part index.
            expected_rows: Rows the part must contain.

        Returns:
            True when the file is a complete parquet part with the expected
            columns and row count.
        """
        path = self.part_path(index)
        if not path.exists():
            return False
        try:
            parquet_file = pq.ParquetFile(path)
            metadata = parquet_file.metadata
            # ``metadata.schema.names`` reports the physical leaf name
            # (``element``) for a list column; the logical schema names are what
            # the cache contract is about.
            names = tuple(parquet_file.schema_arrow.names)
        except Exception:  # pragma: no cover - corrupt files are environment errors
            return False
        return names == PART_COLUMNS and metadata.num_rows == expected_rows

    def prepare(
        self,
        *,
        fingerprint: str,
        fingerprint_payload: dict[str, object],
        total_reviews: int,
        resume: bool,
    ) -> CacheStatus:
        """Create or validate the cache for one extraction run.

        Args:
            fingerprint: Current extraction fingerprint.
            fingerprint_payload: Human-readable payload behind the fingerprint.
            total_reviews: Total text reviews in the source.
            resume: Continue when parts are missing.

        Returns:
            Cache status with the missing/completed part indices.

        Raises:
            AspectCacheMismatch: If the existing manifest has another
                fingerprint/shape, or parts exist without a manifest.
            AspectCacheIncomplete: If parts are missing and ``resume`` is False
                for an existing cache.
        """
        part_count = self.part_count(total_reviews)
        manifest = self.load_manifest()
        fresh = manifest is None

        if manifest is None:
            existing = list(self.parts_dir.glob("part-*.parquet")) if self.parts_dir.exists() else []
            if existing:
                raise AspectCacheMismatch(
                    f"candidate parts exist in {self.parts_dir} but no manifest does; "
                    "remove that directory and re-extract"
                )
            manifest = CandidateManifest(
                fingerprint=fingerprint,
                payload=fingerprint_payload,
                total_reviews=total_reviews,
                chunk_rows=self.chunk_rows,
                part_count=part_count,
            )
            self.parts_dir.mkdir(parents=True, exist_ok=True)
            self.manifest_path.write_text(
                json.dumps(manifest.to_json(), indent=2, sort_keys=True),
                encoding="utf-8",
            )

        if (
            manifest.fingerprint != fingerprint
            or manifest.total_reviews != total_reviews
            or manifest.chunk_rows != self.chunk_rows
            or manifest.part_count != part_count
        ):
            raise AspectCacheMismatch(
                "candidate cache does not match the current extraction settings or review "
                f"source ({self.parts_dir}); remove that directory and re-extract, or "
                "restore the settings/source that produced it"
            )

        completed: list[int] = []
        missing: list[int] = []
        for index in range(part_count):
            expected = self.expected_rows(index, total_reviews)
            target = completed if self._part_complete(index, expected) else missing
            target.append(index)
        if missing and not resume and not fresh:
            raise AspectCacheIncomplete(
                f"{len(missing)} candidate part(s) are missing in {self.parts_dir}; "
                "pass --resume to continue the extraction"
            )
        return CacheStatus(
            manifest=manifest,
            missing_parts=missing,
            completed_parts=completed,
        )

    def write_part(self, index: int, records: Sequence[dict[str, object]]) -> Path:
        """Atomically write one complete part.

        Args:
            index: Part index.
            records: Records with exactly ``review_id``, ``asin`` and ``aspects``.

        Returns:
            Final part path.

        Raises:
            ValueError: If a record lacks one of :data:`PART_COLUMNS`.
        """
        for record in records:
            missing = [column for column in PART_COLUMNS if column not in record]
            if missing:
                raise ValueError(f"candidate record lacks column(s) {missing}")
        frame = pd.DataFrame(records, columns=list(PART_COLUMNS))
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        target = self.part_path(index)
        temporary = target.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
        return target

    def read_part(self, index: int) -> pd.DataFrame:
        """Read one candidate part.

        Args:
            index: Part index.

        Returns:
            Part frame.

        Raises:
            AspectCacheIncomplete: If the part does not exist.
        """
        path = self.part_path(index)
        if not path.exists():
            raise AspectCacheIncomplete(f"candidate part not found: {path}")
        return pd.read_parquet(path)

    def iter_parts(self) -> Iterator[pd.DataFrame]:
        """Yield every part in index order.

        Yields:
            Part frames.
        """
        for path in sorted(self.parts_dir.glob("part-*.parquet")):
            yield pd.read_parquet(path)

    def last_review_id(self, index: int) -> str:
        """Return the stable review id that ends one part.

        Args:
            index: Part index.

        Returns:
            Last ``review_id`` in that part.

        Raises:
            AspectCacheIncomplete: If the part is empty or missing.
        """
        frame = self.read_part(index)
        if frame.empty:
            raise AspectCacheIncomplete(f"candidate part is empty: {self.part_path(index)}")
        return str(frame["review_id"].iloc[-1])


def iter_part_records(frame: pd.DataFrame) -> Iterable[tuple[str, str, list[str]]]:
    """Yield ``(review_id, asin, aspects)`` from one candidate part.

    Args:
        frame: Candidate part frame.

    Yields:
        One tuple per review.
    """
    for row in frame.itertuples(index=False):
        aspects = row.aspects
        if aspects is None:
            values: list[str] = []
        elif isinstance(aspects, str):
            # A malformed one-column payload is kept as one literal aspect
            # rather than exploded into characters; the vocabulary stage will
            # reject it if it is not a real phrase.
            values = [aspects]
        elif isinstance(aspects, Iterable):
            values = [str(aspect) for aspect in aspects]
        else:
            values = [str(aspects)]
        yield str(row.review_id), str(row.asin), values


__all__ = [
    "AspectCacheError",
    "AspectCacheIncomplete",
    "AspectCacheMismatch",
    "CacheStatus",
    "CandidateCache",
    "CandidateManifest",
    "PART_COLUMNS",
    "extraction_fingerprint",
    "iter_part_records",
    "source_signature",
    "text_review_condition",
]

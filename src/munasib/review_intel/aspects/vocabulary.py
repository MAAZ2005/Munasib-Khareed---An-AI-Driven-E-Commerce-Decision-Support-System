"""Category-level aspect vocabulary built from cached candidate mentions.

An aspect enters the vocabulary when it appears in at least
``min_review_count`` **reviews** (a review that says "noise" twice still counts
once). The vocabulary is rebuilt from candidate parts, never by re-running spaCy
unless the parts themselves are missing.

The two readable outputs are:

* ``reports/aspect_vocabulary.csv`` - ranked vocabulary with review/product
  counts for inspection.
* the CLI preview - the top 100 aspects plus the 50 aspects nearest the cutoff,
  which is how junk nouns are spotted before the review and aggregation stages.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pandas as pd

from munasib.core.config import AspectsConfig
from munasib.core.exceptions import DataError
from munasib.review_intel.aspects.cache import iter_part_records
from munasib.review_intel.aspects.extraction import filter_aspect_phrase
from munasib.review_intel.aspects.types import (
    VocabularyBuildResult,
    VocabularyEntry,
)

#: Columns of the vocabulary parquet/csv (csv adds a 1-based rank).
VOCABULARY_COLUMNS: tuple[str, ...] = ("aspect", "review_count", "product_count")


def build_vocabulary(
    parts: Iterable[pd.DataFrame],
    config: AspectsConfig,
) -> VocabularyBuildResult:
    """Count cached candidates and keep those above the vocabulary cutoff.

    Args:
        parts: Candidate part frames.
        config: Aspect configuration (stoplist and ``min_review_count``).

    Returns:
        Vocabulary entries plus every raw count for the inspection preview.

    Raises:
        DataError: If no candidate mention was found at all, which means the
            parts are empty or the extraction produced nothing usable.
    """
    review_counts: Counter[str] = Counter()
    product_sets: defaultdict[str, set[str]] = defaultdict(set)

    for frame in parts:
        for _review_id, asin, aspects in iter_part_records(frame):
            # A review mention is counted once even if the chunker emitted the
            # same phrase twice; set() keeps this invariant explicit. Each raw
            # phrase is first re-canonicalised under the *current* stoplist, so
            # extending the stoplist rebuilds this vocabulary from the cache
            # without another spaCy pass (docs/aspects.md).
            rewritten = {
                filtered
                for aspect in aspects
                if (filtered := filter_aspect_phrase(aspect, config)) is not None
            }
            for aspect in rewritten:
                review_counts[aspect] += 1
                product_sets[aspect].add(asin)

    if not review_counts:
        raise DataError(
            "the candidate parts contain no usable aspect mentions; check the spaCy "
            "extraction output and the configured stoplist"
        )

    entries = [
        VocabularyEntry(
            aspect=aspect,
            review_count=count,
            product_count=len(product_sets[aspect]),
        )
        for aspect, count in review_counts.items()
        if count >= config.min_review_count
    ]
    entries.sort(key=lambda entry: (-entry.review_count, entry.aspect))
    return VocabularyBuildResult(
        entries=entries,
        all_review_counts=dict(review_counts),
        all_product_counts={aspect: len(asins) for aspect, asins in product_sets.items()},
        min_review_count=config.min_review_count,
    )


def write_vocabulary(entries: list[VocabularyEntry], path: str | Path) -> int:
    """Write the vocabulary parquet.

    Args:
        entries: Vocabulary entries in ranked order.
        path: Destination parquet path.

    Returns:
        Number of rows written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "aspect": entry.aspect,
                "review_count": entry.review_count,
                "product_count": entry.product_count,
            }
            for entry in entries
        ],
        columns=list(VOCABULARY_COLUMNS),
    )
    frame.to_parquet(target, index=False)
    return len(frame)


def write_vocabulary_csv(entries: list[VocabularyEntry], path: str | Path) -> int:
    """Write the ranked, human-readable vocabulary CSV.

    Args:
        entries: Vocabulary entries in ranked order.
        path: Destination CSV path.

    Returns:
        Number of rows written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "rank": rank,
                "aspect": entry.aspect,
                "review_count": entry.review_count,
                "product_count": entry.product_count,
            }
            for rank, entry in enumerate(entries, start=1)
        ],
        columns=["rank", *VOCABULARY_COLUMNS],
    )
    frame.to_csv(target, index=False, encoding="utf-8")
    return len(frame)


def load_vocabulary(path: str | Path) -> list[VocabularyEntry]:
    """Load and validate the vocabulary parquet.

    Args:
        path: Vocabulary parquet path.

    Returns:
        Entries in stored order.

    Raises:
        DataError: If the file is missing, malformed, or contains duplicates.
    """
    target = Path(path)
    if not target.exists():
        raise DataError(
            f"aspect vocabulary not found at {target}; run "
            "'python scripts/compute_aspects.py --vocabulary' first"
        )
    frame = pd.read_parquet(target)
    missing = [column for column in VOCABULARY_COLUMNS if column not in frame.columns]
    if missing:
        raise DataError(f"aspect vocabulary {target} lacks column(s) {missing}")
    entries: list[VocabularyEntry] = []
    seen: set[str] = set()
    for row in frame.loc[:, list(VOCABULARY_COLUMNS)].itertuples(index=False):
        aspect = str(row.aspect)
        review_count = int(cast(int, row.review_count))
        product_count = int(cast(int, row.product_count))
        if not aspect.strip():
            raise DataError(f"aspect vocabulary {target} contains a blank aspect")
        if aspect in seen:
            raise DataError(f"aspect vocabulary {target} repeats aspect {aspect!r}")
        if review_count < 1 or product_count < 1:
            raise DataError(
                f"aspect vocabulary {target} has a non-positive count for {aspect!r}"
            )
        seen.add(aspect)
        entries.append(
            VocabularyEntry(
                aspect=aspect,
                review_count=review_count,
                product_count=product_count,
            )
        )
    return entries


def vocabulary_preview(
    result: VocabularyBuildResult,
    *,
    top_n: int = 100,
    near_n: int = 50,
) -> list[str]:
    """Render the vocabulary inspection report.

    Args:
        result: Vocabulary build result.
        top_n: How many top aspects to print.
        near_n: How many aspects nearest the cutoff to print.

    Returns:
        One line per row, ready to print.
    """
    entries = result.entries
    lines = [
        (
            f"vocabulary: {len(entries):,} aspect(s) with >= "
            f"{result.min_review_count:,} reviews out of "
            f"{len(result.all_review_counts):,} raw candidate(s)"
        ),
        "",
        f"Top {min(top_n, len(entries))} aspects by review count:",
    ]
    for rank, entry in enumerate(entries[:top_n], start=1):
        lines.append(
            f"  {rank:>3}. {entry.aspect:<32} "
            f"reviews={entry.review_count:>7,} products={entry.product_count:>5,}"
        )
    nearest = result.nearest_cutoff(near_n)
    lines.extend(
        [
            "",
            f"{len(nearest)} aspect(s) nearest the {result.min_review_count:,}-review cutoff:",
        ]
    )
    for aspect, count in nearest:
        status = "IN" if count >= result.min_review_count else "out"
        lines.append(f"  {status:<3} {aspect:<32} reviews={count:>7,}")
    return lines


__all__ = [
    "VOCABULARY_COLUMNS",
    "build_vocabulary",
    "load_vocabulary",
    "vocabulary_preview",
    "write_vocabulary",
    "write_vocabulary_csv",
]

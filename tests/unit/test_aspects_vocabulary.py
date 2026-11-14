"""Tests for the category vocabulary and its inspection preview."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from munasib.core.config import AspectsConfig
from munasib.core.exceptions import DataError
from munasib.review_intel.aspects.vocabulary import (
    build_vocabulary,
    load_vocabulary,
    vocabulary_preview,
    write_vocabulary,
    write_vocabulary_csv,
)


def part(rows: list[tuple[str, str, list[str]]]) -> pd.DataFrame:
    """Build one candidate part frame.

    Args:
        rows: ``(review_id, asin, aspects)`` tuples.

    Returns:
        Frame with the candidate-cache columns.
    """
    return pd.DataFrame(rows, columns=["review_id", "asin", "aspects"])


def test_build_vocabulary_counts_reviews_not_occurrences() -> None:
    """A repeated phrase in one review contributes one review count."""
    result = build_vocabulary(
        [
            part(
                [
                    ("r1", "A", ["noise", "noise"]),
                    ("r2", "A", ["noise"]),
                    ("r3", "B", ["filter"]),
                ]
            )
        ],
        AspectsConfig(min_review_count=2),
    )
    assert [(e.aspect, e.review_count, e.product_count) for e in result.entries] == [
        ("noise", 2, 1)
    ]
    assert result.all_review_counts == {"noise": 2, "filter": 1}
    assert result.all_product_counts == {"noise": 1, "filter": 1}


def test_build_vocabulary_applies_the_stoplist_and_orders_by_count() -> None:
    """Generic nouns are removed; entries sort by count then aspect."""
    config = AspectsConfig(min_review_count=1, generic_nouns=["thing"])
    result = build_vocabulary(
        [
            part(
                [
                    ("r1", "A", ["thing", "filter", "motor"]),
                    ("r2", "B", ["filter"]),
                ]
            )
        ],
        config,
    )
    assert [(e.aspect, e.review_count) for e in result.entries] == [
        ("filter", 2),
        ("motor", 1),
    ]
    assert "thing" not in result.all_review_counts


def test_build_vocabulary_recanonicalises_raw_cached_phrases() -> None:
    """Raw cached phrases are rewritten under the current stoplist.

    Candidate parts are stoplist-free by design, so ``amazon review`` must
    disappear while ``customer service`` collapses to ``service`` (the head the
    live extractor would have produced).
    """
    config = AspectsConfig(min_review_count=1)
    result = build_vocabulary(
        [
            part(
                [
                    ("r1", "A", ["amazon review", "customer service", "ice maker"]),
                    ("r2", "B", ["star rating", "baby clothe", "ice maker"]),
                ]
            )
        ],
        config,
    )
    assert sorted(result.all_review_counts) == ["clothe", "ice maker", "service"]
    assert [(e.aspect, e.review_count) for e in result.entries] == [
        ("ice maker", 2),
        ("clothe", 1),
        ("service", 1),
    ]


def test_build_vocabulary_requires_at_least_one_candidate() -> None:
    """Empty parts are a data error, not an empty vocabulary artifact."""
    with pytest.raises(DataError, match="no usable aspect mentions"):
        build_vocabulary([part([])], AspectsConfig())


def test_vocabulary_roundtrip_through_parquet_and_csv(tmp_path: Path) -> None:
    """The parquet loader and the readable CSV agree on every entry."""
    config = AspectsConfig(min_review_count=1)
    result = build_vocabulary(
        [part([("r1", "A", ["filter"]), ("r2", "A", ["motor"])])],
        config,
    )
    parquet = tmp_path / "vocabulary.parquet"
    csv_path = tmp_path / "vocabulary.csv"
    assert write_vocabulary(result.entries, parquet) == 2
    assert write_vocabulary_csv(result.entries, csv_path) == 2
    assert load_vocabulary(parquet) == result.entries
    frame = pd.read_csv(csv_path)
    assert list(frame.columns) == ["rank", "aspect", "review_count", "product_count"]
    assert frame["rank"].tolist() == [1, 2]


def test_vocabulary_preview_prints_top_and_near_cutoff() -> None:
    """The inspection report names the top aspects and both sides of the cutoff."""
    config = AspectsConfig(min_review_count=3)
    result = build_vocabulary(
        [
            part(
                [
                    ("r1", "A", ["filter", "motor"]),
                    ("r2", "A", ["filter", "motor"]),
                    ("r3", "B", ["filter", "noise"]),
                    ("r4", "B", ["noise"]),
                ]
            )
        ],
        config,
    )
    lines = vocabulary_preview(result, top_n=2, near_n=10)
    assert any("vocabulary: 1 aspect(s)" in line for line in lines)
    assert any("filter" in line and "IN" not in line for line in lines)
    assert any("out" in line and "noise" in line for line in lines)


def test_load_vocabulary_rejects_duplicate_rows(tmp_path: Path) -> None:
    """A duplicated aspect is a broken artifact, not a silent merge."""
    path = tmp_path / "bad.parquet"
    pd.DataFrame(
        [
            {"aspect": "filter", "review_count": 3, "product_count": 2},
            {"aspect": "filter", "review_count": 4, "product_count": 2},
        ]
    ).to_parquet(path, index=False)
    with pytest.raises(DataError, match="repeats aspect"):
        load_vocabulary(path)


def test_load_vocabulary_reports_a_missing_file(tmp_path: Path) -> None:
    """A missing vocabulary names the stage that creates it."""
    with pytest.raises(DataError, match="--vocabulary"):
        load_vocabulary(tmp_path / "absent.parquet")

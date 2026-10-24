"""Tests for the leakage-safe training sample and evaluation-split loading.

Every case uses a small synthetic frame, so the assertions are about the rules
and not about the dataset: the per-product cap, the minority-preservation rule,
the positive budget, determinism (same input + seed -> same output, whatever the
row order), the text filter and the train-only leakage guard.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from munasib.core.config import SentimentConfig
from munasib.core.constants import (
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABEL_POSITIVE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from munasib.core.exceptions import DataError
from munasib.review_intel.sentiment.sampling import (
    EVAL_SPLITS,
    build_train_sample,
    load_eval_split,
    text_mask,
)

#: A text long enough to pass the default ``min_chars``.
TEXT = "the unit fits the socket and works as described"


def _review(review_id: str, asin: str, split: str, label: int, text: str = TEXT) -> dict:
    """Build one synthetic review row.

    Args:
        review_id: Unique review id.
        asin: Product id.
        split: Split name.
        label: Weak label code.
        text: Review text (``None`` for a text-less row).

    Returns:
        A dict with the columns the sample builder reads.
    """
    return {
        "review_id": review_id,
        "asin": asin,
        "split": split,
        "review_text_clean": text,
        "sentiment_label_weak": label,
    }


def _product(
    asin: str,
    split: str,
    n_negative: int,
    n_neutral: int,
    n_positive: int,
    text: str = TEXT,
) -> list[dict]:
    """Build a product's reviews with a fixed class mix.

    Args:
        asin: Product id.
        split: Split name.
        n_negative: Number of negative reviews.
        n_neutral: Number of neutral reviews.
        n_positive: Number of positive reviews.
        text: Review text for every row.

    Returns:
        One row dict per generated review.
    """
    rows: list[dict] = []
    for label, count in (
        (SENTIMENT_LABEL_NEGATIVE, n_negative),
        (SENTIMENT_LABEL_NEUTRAL, n_neutral),
        (SENTIMENT_LABEL_POSITIVE, n_positive),
    ):
        for index in range(count):
            rows.append(_review(f"{asin}-{label}-{index}", asin, split, label, text))
    return rows


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Assemble rows into a frame with the review columns typed sensibly.

    Args:
        rows: Row dicts from :func:`_review`/:func:`_product`.

    Returns:
        DataFrame with an object text column and an ``int8`` label column.
    """
    frame = pd.DataFrame(rows)
    frame["sentiment_label_weak"] = frame["sentiment_label_weak"].astype("int8")
    return frame


def _config(**overrides: object) -> SentimentConfig:
    """Build a sentiment config with explicit test values.

    Args:
        **overrides: Fields to override on :class:`SentimentConfig`.

    Returns:
        The configured settings object.
    """
    defaults: dict[str, object] = {
        "max_reviews_per_product": 50,
        "max_train_rows": 100_000,
        "min_chars": 3,
        "seed": 42,
    }
    defaults.update(overrides)
    return SentimentConfig(**defaults)


class TestPerProductCap:
    """The per-product cap limits one product's influence."""

    def test_cap_limits_rows_per_product(self) -> None:
        """Three products of 12 reviews with a cap of 4 give 12 rows, 4 each."""
        rows = (
            _product("A", SPLIT_TRAIN, 2, 1, 9)
            + _product("B", SPLIT_TRAIN, 2, 1, 9)
            + _product("C", SPLIT_TRAIN, 2, 1, 9)
        )
        sample = build_train_sample(_frame(rows), _config(max_reviews_per_product=4))
        counts = sample.frame["asin"].value_counts()
        assert len(sample.frame) == 12
        assert set(counts.tolist()) == {4}
        assert sample.report.products == 3

    def test_cap_does_not_invent_rows(self) -> None:
        """A product with fewer reviews than the cap contributes all of them."""
        rows = _product("A", SPLIT_TRAIN, 1, 0, 2)
        sample = build_train_sample(_frame(rows), _config(max_reviews_per_product=50))
        assert len(sample.frame) == 3

    def test_cap_step_is_reported(self) -> None:
        """The report carries the cap as its own checkpoint."""
        rows = _product("A", SPLIT_TRAIN, 1, 1, 18)
        sample = build_train_sample(_frame(rows), _config(max_reviews_per_product=5))
        steps = [step.step for step in sample.report.steps]
        assert "per-product cap" in steps
        capped = next(step for step in sample.report.steps if step.step == "per-product cap")
        assert capped.rows == 5


class TestMinorityPreservation:
    """The positive budget never drops a negative or a neutral review."""

    def test_all_minority_reviews_are_kept(self) -> None:
        """Every negative/neutral row survives the positive subsampling."""
        rows = _product("A", SPLIT_TRAIN, 5, 3, 40)
        sample = build_train_sample(_frame(rows), _config(max_train_rows=10))
        labels = sample.frame["sentiment_label_weak"].value_counts().to_dict()
        assert labels[SENTIMENT_LABEL_NEGATIVE] == 5
        assert labels[SENTIMENT_LABEL_NEUTRAL] == 3
        assert labels[SENTIMENT_LABEL_POSITIVE] == 2  # 10 - (5 + 3) remaining budget

    def test_positives_subsampled_to_the_budget(self) -> None:
        """The final row count hits ``max_train_rows`` when positives allow it."""
        rows = _product("A", SPLIT_TRAIN, 4, 2, 60) + _product("B", SPLIT_TRAIN, 1, 1, 60)
        config = _config(max_train_rows=20, max_reviews_per_product=100)
        sample = build_train_sample(_frame(rows), config)
        assert len(sample.frame) == 20
        assert sample.report.positives_kept == 20 - 8
        assert sample.report.positives_available == 120

    def test_minority_floor_can_exceed_the_budget(self) -> None:
        """A budget smaller than the minority count keeps the minorities anyway."""
        rows = _product("A", SPLIT_TRAIN, 6, 4, 30)
        sample = build_train_sample(_frame(rows), _config(max_train_rows=5))
        assert len(sample.frame) == 10
        assert sample.report.positives_kept == 0
        assert any("exceed max_train_rows" in note for note in sample.report.notes)

    def test_no_subsampling_when_under_budget(self) -> None:
        """A generous budget keeps every capped row and reports no truncation."""
        rows = _product("A", SPLIT_TRAIN, 2, 1, 5)
        sample = build_train_sample(_frame(rows), _config(max_train_rows=10_000))
        assert len(sample.frame) == 8
        assert sample.report.positives_available == sample.report.positives_kept
        assert sample.report.notes == ()


class TestDeterminism:
    """The same input and seed always produce the same sample."""

    def test_repeated_calls_are_identical(self) -> None:
        """Two calls on the same frame return identical review ids in order."""
        rows = _product("A", SPLIT_TRAIN, 4, 2, 40) + _product("B", SPLIT_TRAIN, 3, 1, 30)
        frame = _frame(rows)
        config = _config(max_train_rows=25, max_reviews_per_product=10)
        first = build_train_sample(frame, config)
        second = build_train_sample(frame, config)
        assert first.frame["review_id"].tolist() == second.frame["review_id"].tolist()
        assert first.report.to_dict() == second.report.to_dict()

    def test_row_order_does_not_change_the_sample(self) -> None:
        """Shuffling the input frame yields the same selected reviews."""
        rows = _product("A", SPLIT_TRAIN, 4, 2, 40) + _product("B", SPLIT_TRAIN, 3, 1, 30)
        frame = _frame(rows)
        config = _config(max_train_rows=25, max_reviews_per_product=10)
        first = build_train_sample(frame, config)
        shuffled = build_train_sample(frame.sample(frac=1.0, random_state=7), config)
        assert first.frame["review_id"].tolist() == shuffled.frame["review_id"].tolist()

    def test_seed_changes_the_selection(self) -> None:
        """A different seed selects different rows (same size)."""
        rows = _product("A", SPLIT_TRAIN, 4, 2, 40)
        frame = _frame(rows)
        config = _config(max_train_rows=20)
        first = build_train_sample(frame, config)
        other = build_train_sample(frame, _config(max_train_rows=20, seed=7))
        assert len(first.frame) == len(other.frame) == 20
        assert set(first.frame["review_id"]) != set(other.frame["review_id"])


class TestLeakageGuard:
    """A non-train row can never reach a training sample."""

    def test_build_sample_output_is_train_only(self) -> None:
        """Rows from val/test are filtered out before anything else happens."""
        rows = _product("A", SPLIT_TRAIN, 1, 1, 3) + _product("B", SPLIT_VAL, 2, 1, 5)
        sample = build_train_sample(_frame(rows), _config())
        assert set(sample.frame["split"]) == {SPLIT_TRAIN}
        assert set(sample.frame["asin"]) == {"A"}

    def test_guard_raises_on_a_mixed_frame(self) -> None:
        """A frame holding a val row fails the guard with a clear message."""
        from munasib.review_intel.sentiment.sampling import assert_split_only

        frame = _frame(
            _product("A", SPLIT_TRAIN, 1, 0, 1) + _product("B", SPLIT_VAL, 1, 0, 1)
        )
        with pytest.raises(DataError, match="leakage guard"):
            assert_split_only(frame, SPLIT_TRAIN)

    def test_guard_raises_on_a_missing_split_value(self) -> None:
        """A row with no split cannot be proven safe, so it is rejected."""
        from munasib.review_intel.sentiment.sampling import assert_split_only

        frame = _frame(_product("A", SPLIT_TRAIN, 1, 0, 1))
        frame.loc[0, "split"] = None
        with pytest.raises(DataError, match="no 'split' value"):
            assert_split_only(frame, SPLIT_TRAIN)

    def test_guard_accepts_a_clean_frame(self) -> None:
        """The guard is a check, not a filter: a clean frame passes untouched."""
        from munasib.review_intel.sentiment.sampling import assert_split_only

        frame = _frame(_product("A", SPLIT_TRAIN, 1, 0, 1))
        assert_split_only(frame, SPLIT_TRAIN)


class TestTextFilter:
    """Reviews without usable text never enter training or evaluation."""

    def test_min_chars_filters_short_and_missing_text(self) -> None:
        """``None``, ``""`` and ``"ab"`` are dropped; ``"abc"`` survives."""
        rows = [
            _review("r1", "A", SPLIT_TRAIN, SENTIMENT_LABEL_POSITIVE, None),
            _review("r2", "A", SPLIT_TRAIN, SENTIMENT_LABEL_POSITIVE, ""),
            _review("r3", "A", SPLIT_TRAIN, SENTIMENT_LABEL_POSITIVE, "ab"),
            _review("r4", "A", SPLIT_TRAIN, SENTIMENT_LABEL_POSITIVE, "abc"),
            _review("r5", "A", SPLIT_TRAIN, SENTIMENT_LABEL_NEGATIVE, "  "),
        ]
        sample = build_train_sample(_frame(rows), _config(min_chars=3))
        assert sample.frame["review_id"].tolist() == ["r4"]
        text_step = sample.report.steps[2]
        assert text_step.rows == 1

    def test_text_mask_is_false_for_missing_text(self) -> None:
        """The shared predicate flags text-less rows rather than coercing them."""
        frame = _frame(
            [
                _review("r1", "A", SPLIT_TRAIN, SENTIMENT_LABEL_POSITIVE, None),
                _review("r2", "A", SPLIT_TRAIN, SENTIMENT_LABEL_POSITIVE, "long enough"),
            ]
        )
        mask = text_mask(frame, _config())
        assert mask.tolist() == [False, True]

    def test_empty_training_split_reports_zero(self) -> None:
        """A frame with no usable train rows yields an empty sample and a note."""
        rows = _product("A", SPLIT_VAL, 1, 0, 2)
        sample = build_train_sample(_frame(rows), _config())
        assert sample.frame.empty
        assert sample.report.rows == 0
        assert sample.report.class_counts == {
            SENTIMENT_LABEL_NEGATIVE: 0,
            SENTIMENT_LABEL_NEUTRAL: 0,
            SENTIMENT_LABEL_POSITIVE: 0,
        }
        assert any("nothing to fit" in note for note in sample.report.notes)


class TestReport:
    """The report describes the sample without inventing anything."""

    def test_class_shares_sum_to_one(self) -> None:
        """Shares are fractions of the final sample and add up to 1."""
        rows = _product("A", SPLIT_TRAIN, 3, 2, 20) + _product("B", SPLIT_TRAIN, 1, 1, 10)
        sample = build_train_sample(_frame(rows), _config(max_train_rows=15))
        total = sum(sample.report.class_shares.values())
        assert total == pytest.approx(1.0)
        assert sum(sample.report.class_counts.values()) == len(sample.frame)

    def test_steps_are_monotone(self) -> None:
        """Each checkpoint can only shrink the row count, never grow it."""
        rows = _product("A", SPLIT_TRAIN, 3, 2, 20) + _product("B", SPLIT_VAL, 1, 1, 5)
        sample = build_train_sample(_frame(rows), _config(max_train_rows=10))
        counts = [step.rows for step in sample.report.steps]
        assert counts == sorted(counts, reverse=True)
        assert counts[0] == len(rows)
        assert counts[-1] == len(sample.frame)

    def test_lines_are_rendered(self) -> None:
        """The printable form names every class and the product count."""
        rows = _product("A", SPLIT_TRAIN, 1, 1, 4)
        sample = build_train_sample(_frame(rows), _config())
        rendered = "\n".join(sample.report.lines())
        assert "negative rows" in rendered
        assert "neutral rows" in rendered
        assert "positive rows" in rendered
        assert "products in sample" in rendered

    def test_missing_column_is_reported(self) -> None:
        """A frame without the label column fails with a column-level message."""
        frame = _frame(_product("A", SPLIT_TRAIN, 1, 0, 1)).drop(
            columns=["sentiment_label_weak"]
        )
        with pytest.raises(DataError, match="sentiment_label_weak"):
            build_train_sample(frame, _config())


class TestLoadEvalSplit:
    """Evaluation splits are returned in full, never sampled."""
    def test_returns_every_val_row_with_text(self) -> None:
        """No cap and no subsampling: 12 val rows (minus one text-less) come back."""
        rows = _product("A", SPLIT_VAL, 2, 1, 5) + _product("B", SPLIT_VAL, 1, 1, 2)
        rows.append(_review("empty", "C", SPLIT_VAL, SENTIMENT_LABEL_POSITIVE, None))
        frame = load_eval_split(_frame(rows), SPLIT_VAL, _config(max_reviews_per_product=1))
        assert len(frame) == 12
        assert set(frame["split"]) == {SPLIT_VAL}
        assert frame["review_text_clean"].notna().all()

    def test_returns_every_test_row(self) -> None:
        """``test`` is a valid evaluation split."""
        rows = _product("A", SPLIT_TEST, 1, 1, 3)
        frame = load_eval_split(_frame(rows), SPLIT_TEST, _config())
        assert len(frame) == 5

    def test_train_is_rejected(self) -> None:
        """Loading ``train`` here would bypass the sample builder, so it is refused."""
        rows = _product("A", SPLIT_TRAIN, 1, 0, 1)
        with pytest.raises(DataError, match="accepts only"):
            load_eval_split(_frame(rows), SPLIT_TRAIN, _config())

    def test_unknown_split_is_rejected(self) -> None:
        """An unknown split name fails instead of returning an empty frame."""
        rows = _product("A", SPLIT_VAL, 1, 0, 1)
        with pytest.raises(DataError, match="accepts only"):
            load_eval_split(_frame(rows), "holdout", _config())

    def test_eval_splits_are_val_and_test(self) -> None:
        """The accepted evaluation splits are exactly val and test."""
        assert EVAL_SPLITS == (SPLIT_VAL, SPLIT_TEST)

    def test_does_not_mutate_the_input(self) -> None:
        """The returned frame is a copy, so callers cannot edit the source."""
        rows = _product("A", SPLIT_VAL, 1, 0, 2)
        frame = _frame(rows)
        original = frame.copy(deep=True)
        loaded = load_eval_split(frame, SPLIT_VAL, _config())
        loaded.loc[loaded.index[0], "asin"] = "Z"
        pd.testing.assert_frame_equal(frame, original)


REAL_REVIEWS = Path(__file__).resolve().parents[2] / "data" / "processed" / "reviews.parquet"


@pytest.mark.skipif(
    not REAL_REVIEWS.exists(),
    reason="processed reviews.parquet is not available (data/ is git-ignored)",
)
def test_real_dataset_sample_holds_its_invariants() -> None:
    """The shipped config on the real parquet honours every sampling rule.

    This is the only test that reads the dataset; it skips when the processed
    table is absent, so the suite stays hermetic on a fresh clone.
    """
    from munasib.core.config import SentimentConfig

    reviews = pd.read_parquet(REAL_REVIEWS)
    config = SentimentConfig()
    sample = build_train_sample(reviews, config)

    assert len(sample.frame) == sample.report.rows
    assert set(sample.frame["split"]) == {SPLIT_TRAIN}
    assert sample.frame["review_text_clean"].str.len().min() >= config.min_chars
    assert sample.frame.groupby("asin").size().max() <= config.max_reviews_per_product
    assert len(sample.frame) <= config.max_train_rows or sample.report.notes
    assert sum(sample.report.class_shares.values()) == pytest.approx(1.0)
    # The evaluation split comes back whole: exactly the usable val rows, with no
    # per-product cap applied.
    val_rows = reviews.loc[reviews["split"] == SPLIT_VAL]
    expected = val_rows.loc[
        val_rows["review_text_clean"].notna()
        & (val_rows["review_text_clean"].str.len() >= config.min_chars)
    ]
    val = load_eval_split(reviews, SPLIT_VAL, config)
    assert len(val) == len(expected)
    assert set(val["split"]) == {SPLIT_VAL}


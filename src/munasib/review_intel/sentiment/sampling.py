"""Leakage-safe training-sample construction for the sentiment models.

Three rules shape this module, and every one of them exists because the review
table is **not** a balanced, independent sample:

1. **Train only, ever.** The split is assigned at the product level, so a review
   can only leak into training if a row from ``val``/``test`` is selected. Every
   sample builder here filters on :data:`munasib.core.constants.SPLIT_TRAIN`
   *and* re-checks the result, raising :class:`~munasib.core.exceptions.DataError`
   if a single non-train row survives (:func:`assert_split_only`).
2. **A per-product cap, applied class-blind.** One product can carry 6,510
   reviews (the real maximum), so an uncapped sample would learn products rather
   than sentiment. The cap keeps the survivors' natural class mix on purpose;
   the *subsampling* step is what protects the minority classes, so the sample
   does not drift away from the population it is supposed to represent.
3. **Deterministic selection without global state.** Rows are ordered by a SHA-1
   digest of ``seed:purpose:review_id``, i.e. a fixed pseudo-random permutation
   derived from the configured seed. The same frame and seed always yield the
   same sample, independently of row order, and no
   ``numpy.random``/``random`` global state is touched.

Everything here is pure: frames in, frames out, no file or database access.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from munasib.core.config import SentimentConfig
from munasib.core.constants import (
    SENTIMENT_LABEL_NAMES,
    SENTIMENT_LABEL_NEGATIVE,
    SENTIMENT_LABEL_NEUTRAL,
    SENTIMENT_LABELS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from munasib.core.exceptions import DataError

#: Review column holding the product-level split (mirrors ``REVIEW_SCHEMA``).
SPLIT_COLUMN = "split"

#: Review column holding the product id the per-product cap groups on.
PRODUCT_COLUMN = "asin"

#: Splits that may be used for evaluation (never the training split).
EVAL_SPLITS: tuple[str, ...] = (SPLIT_VAL, SPLIT_TEST)

#: Salt that keeps the per-product-cap permutation independent from the
#: positive-subsample permutation over the same review ids.
_CAP_PURPOSE = "cap"
_POSITIVE_PURPOSE = "positive"

#: Labels that are never subsampled (the minority classes).
MINORITY_LABELS: tuple[int, ...] = (SENTIMENT_LABEL_NEGATIVE, SENTIMENT_LABEL_NEUTRAL)


@dataclass(frozen=True)
class CountStep:
    """One row-count checkpoint in the sample-construction pipeline.

    Attributes:
        step: Short name of the filter/selection step.
        rows: Rows remaining after the step.
        detail: Human-readable description of what the step did.
    """

    step: str
    rows: int
    detail: str = ""


@dataclass(frozen=True)
class SampleReport:
    """Before/after counts and class shares for one constructed sample.

    Attributes:
        split: The split the sample was drawn from (always ``train``).
        label_column: Column the class counts were taken from.
        steps: Ordered row-count checkpoints, first = input, last = final sample.
        products: Distinct products in the final sample.
        class_counts: Final row count per label code.
        class_shares: Final row share per label code (same keys as counts).
        positives_available: Positive rows that survived the per-product cap.
        positives_kept: Positive rows left after subsampling; equal to
            ``positives_available`` when the budget did not bite.
    """

    split: str
    label_column: str
    steps: tuple[CountStep, ...]
    products: int
    class_counts: dict[int, int]
    class_shares: dict[int, float]
    positives_available: int
    positives_kept: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def rows(self) -> int:
        """Rows in the final sample (the last checkpoint)."""
        return self.steps[-1].rows if self.steps else 0

    def lines(self) -> list[str]:
        """Render the report as aligned ``label : value`` lines.

        Returns:
            One line per checkpoint and per class share, in a stable order so two
            runs diff cleanly.
        """
        lines = [f"{step.step:<38}: {step.rows:,}  ({step.detail})" for step in self.steps]
        lines.append(f"{'products in sample':<38}: {self.products:,}")
        for label in SENTIMENT_LABELS:
            count = self.class_counts.get(label, 0)
            share = self.class_shares.get(label, 0.0)
            lines.append(f"{SENTIMENT_LABEL_NAMES[label] + ' rows':<38}: {count:,} ({share:.1%})")
        if self.positives_available != self.positives_kept:
            lines.append(
                f"{'positive rows subsampled':<38}: "
                f"{self.positives_kept:,} of {self.positives_available:,}"
            )
        lines.extend(f"note: {note}" for note in self.notes)
        return lines

    def to_dict(self) -> dict[str, object]:
        """Serialise the report for the JSON evaluation report.

        Returns:
            Plain-Python mapping (no numpy/pandas scalars), with class names as
            keys so the JSON is readable without the constants module.
        """
        return {
            "split": self.split,
            "label_column": self.label_column,
            "steps": [
                {"step": step.step, "rows": step.rows, "detail": step.detail}
                for step in self.steps
            ],
            "rows": self.rows,
            "products": self.products,
            "class_counts": {
                SENTIMENT_LABEL_NAMES[label]: self.class_counts.get(label, 0)
                for label in SENTIMENT_LABELS
            },
            "class_shares": {
                SENTIMENT_LABEL_NAMES[label]: self.class_shares.get(label, 0.0)
                for label in SENTIMENT_LABELS
            },
            "positives_available": self.positives_available,
            "positives_kept": self.positives_kept,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class TrainingSample:
    """A constructed training sample plus the report describing how it was made.

    Attributes:
        frame: The selected review rows (original columns, original order of
            columns; a fresh frame, never a view of the input).
        report: Counts and class shares at every step.
    """

    frame: pd.DataFrame
    report: SampleReport

    def __len__(self) -> int:
        """Number of rows in the sample."""
        return len(self.frame)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Fail loudly when the review frame lacks a column the builder needs.

    Args:
        frame: Review rows.
        columns: Column names that must be present.

    Raises:
        DataError: If any column is missing, naming all of them at once.
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DataError(
            f"review frame is missing required column(s) {missing}; "
            f"available columns: {list(frame.columns)}"
        )


def _hash_keys(seed: int, purpose: str, review_ids: pd.Series) -> pd.Series:
    """Derive a deterministic pseudo-random sort key per review.

    The key is the first 64 bits of ``sha1(f"{seed}:{purpose}:{review_id}")``.
    SHA-1 is used here as a cheap uniform hash, not for security; because the
    seed and purpose are part of the digest, changing either reshuffles the
    sample while a fixed pair always reproduces it exactly.

    Args:
        seed: Configured sample seed.
        purpose: Salt that decorrelates independent selections.
        review_ids: Review identifiers.

    Returns:
        ``int64`` keys aligned with ``review_ids`` (ties are resolved by the
        caller's stable sort, so duplicate ids still behave deterministically).
    """
    prefix = f"{seed}:{purpose}:"

    def _digest(review_id: object) -> int:
        raw = hashlib.sha1(f"{prefix}{review_id}".encode()).digest()
        return int.from_bytes(raw[:8], "big")

    return review_ids.map(_digest).astype("int64")


def text_mask(frame: pd.DataFrame, config: SentimentConfig) -> pd.Series:
    """Boolean mask of rows whose configured text column is usable.

    "Usable" means the value is a non-null string with at least
    ``config.min_chars`` characters. Rows kept without text by the pipeline
    (``has_text=False``, 240 in the real subset) therefore never enter training
    or evaluation - a model cannot learn from, or be scored on, an empty string.

    Args:
        frame: Review rows.
        config: Sentiment settings (``text_column``, ``min_chars``).

    Returns:
        Boolean series aligned with ``frame``.
    """
    _require_columns(frame, (config.text_column,))
    text = frame[config.text_column]
    present = text.notna()
    # A non-string cell (never produced by the pipeline, tolerated anyway) has no
    # text length, so treat it as unusable instead of coercing it.
    lengths = text.where(present).astype("string").str.len()
    return present & lengths.ge(config.min_chars).fillna(False).astype(bool)


def assert_split_only(
    frame: pd.DataFrame, expected_split: str, split_column: str = SPLIT_COLUMN
) -> None:
    """Guard that a frame only contains rows of one split.

    This is the last line of defence for the leakage rule: whatever a caller
    accidentally passes, a training frame that contains a ``val``/``test`` row is
    never allowed to reach a model.

    Args:
        frame: Rows about to be used.
        expected_split: The only split permitted in ``frame``.
        split_column: Column holding the split name.

    Raises:
        DataError: If the split column is missing, or any row's split differs
            from ``expected_split`` (the message names the offending splits and
            their counts).
    """
    _require_columns(frame, (split_column,))
    observed = frame[split_column]
    if observed.isna().any():
        raise DataError(
            f"leakage guard: {int(observed.isna().sum()):,} row(s) have no "
            f"'{split_column}' value and cannot be proven to belong to '{expected_split}'"
        )
    unexpected = observed[observed != expected_split]
    if not unexpected.empty:
        counts = {str(split): int(count) for split, count in unexpected.value_counts().items()}
        raise DataError(
            f"leakage guard: frame must contain only '{expected_split}' rows, "
            f"found {len(unexpected):,} row(s) from other splits {counts}"
        )


def _class_counts(frame: pd.DataFrame, label_column: str) -> dict[int, int]:
    """Count rows per weak label code, including labels that are absent.

    Args:
        frame: Review rows.
        label_column: Column holding the label code.

    Returns:
        ``{label: count}`` for every code in
        :data:`munasib.core.constants.SENTIMENT_LABELS` (absent labels -> 0).
    """
    counts = frame[label_column].value_counts()
    return {label: int(counts.get(label, 0)) for label in SENTIMENT_LABELS}


def _class_shares(counts: dict[int, int], total: int) -> dict[int, float]:
    """Convert class counts to shares that sum to 1.0 for a non-empty frame.

    Args:
        counts: Counts per label code.
        total: Row total.

    Returns:
        ``{label: share}``; all zero when ``total`` is 0 (never a fabricated
        fraction).
    """
    if total <= 0:
        return dict.fromkeys(SENTIMENT_LABELS, 0.0)
    return {label: count / total for label, count in counts.items()}


def build_train_sample(reviews_df: pd.DataFrame, config: SentimentConfig) -> TrainingSample:
    """Build the leakage-safe training sample for the sentiment models.

    Pipeline (each step is recorded in the returned report):

    1. Keep rows whose split is exactly ``train``.
    2. Keep rows whose configured text column is present and at least
       ``config.min_chars`` characters long.
    3. Cap each product at ``config.max_reviews_per_product`` reviews with a
       deterministic, class-blind selection, so no single product dominates and
       the sample keeps the split's class mix.
    4. Keep every negative/neutral review that survived the cap, and subsample
       the positives (deterministically) into the remaining
       ``config.max_train_rows`` budget. Minority rows are never dropped by this
       step, so the sample exceeds the budget rather than lose evidence when the
       minority floor alone is larger than it.

    Args:
        reviews_df: Reviews frame following
            :data:`munasib.data.preprocessing.REVIEW_SCHEMA` (at minimum the
            split, product, text and label columns).
        config: Sentiment settings that drive the filters and the seed.

    Returns:
        The selected rows plus a :class:`SampleReport` holding before/after
        counts for every step and the final class shares.

    Raises:
        DataError: If a required column is missing, or the resulting frame is not
            train-only (the leakage guard fires).
    """
    _require_columns(
        reviews_df,
        (SPLIT_COLUMN, PRODUCT_COLUMN, "review_id", config.text_column, config.label_column),
    )
    steps: list[CountStep] = [CountStep("input rows", len(reviews_df), "review frame as loaded")]

    train = reviews_df.loc[reviews_df[SPLIT_COLUMN] == SPLIT_TRAIN]
    steps.append(
        CountStep("train split only", len(train), f"{SPLIT_COLUMN} == '{SPLIT_TRAIN}'")
    )

    usable = train.loc[text_mask(train, config)]
    steps.append(
        CountStep(
            "text present and long enough",
            len(usable),
            f"{config.text_column} non-null and >= {config.min_chars} chars",
        )
    )

    if usable.empty:
        empty = usable.copy()
        report = SampleReport(
            split=SPLIT_TRAIN,
            label_column=config.label_column,
            steps=tuple(steps),
            products=0,
            class_counts=_class_counts(empty, config.label_column),
            class_shares=_class_shares(_class_counts(empty, config.label_column), 0),
            positives_available=0,
            positives_kept=0,
            notes=("the training sample is empty; nothing to fit",),
        )
        return TrainingSample(frame=empty, report=report)

    # Step 3: per-product cap. The cap is class-blind: rows are ordered by a
    # hash of (seed, "cap", review_id), so the survivors of each product keep its
    # natural class mix. A minorities-first cap was measured and rejected: at 50
    # reviews per product it made the sample 45% minority (vs 17% in the split)
    # and left 251 products with no positive review at all.
    helper_columns = ("_key",)
    clashing = [column for column in helper_columns if column in usable.columns]
    if clashing:
        raise DataError(f"review frame already has reserved column(s) {clashing}")
    work = usable.assign(
        _key=_hash_keys(config.seed, _CAP_PURPOSE, usable["review_id"]),
    )
    work = work.sort_values([PRODUCT_COLUMN, "_key"], kind="mergesort").reset_index(drop=True)
    within_product = work.groupby(PRODUCT_COLUMN, sort=False).cumcount()
    capped = work.loc[within_product < config.max_reviews_per_product]
    steps.append(
        CountStep(
            "per-product cap",
            len(capped),
            f"at most {config.max_reviews_per_product} per product (class-blind, seeded)",
        )
    )

    # Step 4: keep every minority review that survived the cap, then subsample
    # positives into the remaining budget. A negative/neutral row is never
    # dropped to make room for another positive.
    is_minority = capped[config.label_column].isin(MINORITY_LABELS)
    minorities = capped.loc[is_minority]
    positives = capped.loc[~is_minority].sort_values("_key", kind="mergesort")
    allowance = max(config.max_train_rows - len(minorities), 0)
    kept_positives = positives.head(allowance)
    notes: list[str] = []
    if len(positives) > len(kept_positives):
        notes.append(
            f"positive rows subsampled from {len(positives):,} to "
            f"{len(kept_positives):,} to respect max_train_rows={config.max_train_rows:,}"
        )
    if len(minorities) > config.max_train_rows:
        notes.append(
            f"minority rows alone ({len(minorities):,}) exceed max_train_rows="
            f"{config.max_train_rows:,}; all were kept rather than dropping evidence"
        )
    final = pd.concat([minorities, kept_positives], ignore_index=True)
    # Stable output order: product, then the same deterministic key.
    final = final.sort_values([PRODUCT_COLUMN, "_key"], kind="mergesort").reset_index(drop=True)
    final = final.drop(columns=list(helper_columns))[list(reviews_df.columns)]
    steps.append(
        CountStep(
            "positive subsample",
            len(final),
            f"total budget max_train_rows={config.max_train_rows:,}",
        )
    )

    # Leakage guard: the selecting steps above are correct, but prove it.
    assert_split_only(final, SPLIT_TRAIN)

    counts = _class_counts(final, config.label_column)
    report = SampleReport(
        split=SPLIT_TRAIN,
        label_column=config.label_column,
        steps=tuple(steps),
        products=int(final[PRODUCT_COLUMN].nunique()),
        class_counts=counts,
        class_shares=_class_shares(counts, len(final)),
        positives_available=len(positives),
        positives_kept=len(kept_positives),
        notes=tuple(notes),
    )
    return TrainingSample(frame=final, report=report)


def load_eval_split(
    reviews_df: pd.DataFrame, split: str, config: SentimentConfig
) -> pd.DataFrame:
    """Return every usable review of one evaluation split (no cap, no sampling).

    Evaluation must see the split **as it is**, otherwise the reported metrics
    describe a filtered population. Only the shared text predicate is applied
    (present and at least ``config.min_chars`` characters), because a review with
    no text cannot be scored at all.

    Args:
        reviews_df: Reviews frame following ``REVIEW_SCHEMA``.
        split: ``val`` or ``test`` (never ``train``; that is what the training
            builder is for).
        config: Sentiment settings (``text_column``, ``min_chars``).

    Returns:
        All usable rows of ``split``, original columns, stable order.

    Raises:
        DataError: If ``split`` is not an evaluation split, if the frame lacks a
            required column, or if the result is not split-only.
    """
    if split not in EVAL_SPLITS:
        raise DataError(
            f"load_eval_split accepts only {list(EVAL_SPLITS)}, got {split!r}; "
            f"the '{SPLIT_TRAIN}' split is built by build_train_sample()"
        )
    _require_columns(reviews_df, (SPLIT_COLUMN, PRODUCT_COLUMN, config.text_column, config.label_column))
    selected = reviews_df.loc[reviews_df[SPLIT_COLUMN] == split]
    selected = selected.loc[text_mask(selected, config)]
    assert_split_only(selected, split)
    return selected.reset_index(drop=True)


__all__ = [
    "CountStep",
    "EVAL_SPLITS",
    "MINORITY_LABELS",
    "PRODUCT_COLUMN",
    "SPLIT_COLUMN",
    "SampleReport",
    "TrainingSample",
    "assert_split_only",
    "build_train_sample",
    "load_eval_split",
    "text_mask",
]

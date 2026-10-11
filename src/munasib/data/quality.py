"""Data-quality report: cleaning counts, missingness, distributions, splits.

The report is generated entirely from the frames and accounting the build
produced (:class:`~munasib.data.preprocessing.BuildReport` plus the final
tables); no number in it is hand-written. Two files are written from one
in-memory dict so they can never disagree:

* ``reports/data_quality_report.json`` - machine-readable, full precision.
* ``reports/data_quality_report.md``    - rendered from the same dict.

Sections (project requirement): row counts before/after each cleaning step,
field presence before/after cleaning, missingness per column for every written
table, rating and weak-label distributions per split, reviews-per-product
distribution, the review date range, and the product-level split sizes with the
stratification strata that produced them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from munasib.core.config import SplitConfig
from munasib.data.loaders import is_present
from munasib.data.splits import SPLIT_ORDER, bin_values, split_sizes

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (preprocessing imports this module)
    from munasib.data.preprocessing import BuildReport

JSON_FILENAME = "data_quality_report.json"
MARKDOWN_FILENAME = "data_quality_report.md"

# Percentiles of reviews-per-product reported (computed with numpy, not hand-picked
# values - these are the quantile *points*, the numbers themselves come from data).
_REVIEW_COUNT_QUANTILES = (10, 25, 50, 75, 90, 99)


def _count_present(series: pd.Series) -> int:
    """Count rows whose value carries information (non-null, non-empty).

    Args:
        series: Any column; string dtypes take a vectorised fast path, list/object
            columns go through :func:`munasib.data.loaders.is_present`.

    Returns:
        Number of present values as a plain ``int``.
    """
    if series.dtype == "str":
        return int((series.fillna("") != "").sum())
    return int(series.map(is_present).sum())


def _share(part: int, whole: int) -> float:
    """Return ``part / whole`` (0.0 when ``whole`` is 0) as a plain float."""
    return float(part / whole) if whole else 0.0


def _presence_row(total: int, before: int, after: int) -> dict[str, Any]:
    """One before/after field-presence row (all counts are code-computed)."""
    return {
        "rows": int(total),
        "before_present": int(before),
        "after_present": int(after),
        "after_missing": int(total - after),
        "share_before": _share(before, total),
        "share_after": _share(after, total),
    }


def _missingness(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-column null counts and shares for one written table."""
    total = len(frame)
    return {
        str(column): {
            "null": int(nulls),
            "share": _share(int(nulls), total),
        }
        for column in frame.columns
        for nulls in [frame[column].isna().sum()]
    }


def _distribution(
    frame: pd.DataFrame, column: str, split: str, key_format: Callable[[Any], str]
) -> dict[str, dict[str, float | int]]:
    """Count + within-split share of one column of the review frame.

    Args:
        frame: Full review frame (must contain ``column`` and ``"split"``).
        column: Column whose values are distributed.
        split: Split name to filter to.
        key_format: Callable mapping a value (possibly NA) to a string key.

    Returns:
        ``{key: {"count": n, "share": f}}``.
    """
    subset = frame.loc[frame["split"] == split, column]
    counts: dict[str, int] = {}
    for value in subset:
        key = key_format(value)
        counts[key] = counts.get(key, 0) + 1
    total = len(subset)
    return {key: {"count": counts[key], "share": _share(counts[key], total)} for key in counts}


def _rating_key(value: Any) -> str:
    """Format a star rating for reporting (``"4"``/``"4.5"``; ``"null"`` when NA)."""
    if pd.isna(value):
        return "null"
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _label_key(value: Any) -> str:
    """Format a weak-label id for reporting (``"0"``/``"1"``/``"2"``/``"null"``)."""
    if pd.isna(value):
        return "null"
    return str(int(value))


def _strata_rows(products: pd.DataFrame, split_config: SplitConfig) -> list[dict[str, Any]]:
    """Per-stratum product counts and per-split allocation (sorted by stratum).

    Bins are recomputed from the written ``mean_rating``/``n_reviews`` columns
    with the same config edges the split used, so the strata are auditable from
    the parquet tables themselves.
    """
    strata = pd.DataFrame(
        {
            "rating_bin": bin_values(products["mean_rating"], split_config.rating_bin_edges),
            "review_count_bin": bin_values(
                products["n_reviews"].astype("float64"), split_config.review_count_bin_edges
            ),
            "split": products["split"].astype(str),
        }
    ).astype(str)
    counts = strata.value_counts(["rating_bin", "review_count_bin", "split"])
    if counts.empty:
        return []
    table = counts.unstack("split")
    table = table.reindex(columns=list(SPLIT_ORDER)).fillna(0)
    table["products"] = table.sum(axis=1)
    table = table.sort_index()
    # `table` is indexed by (rating_bin, review_count_bin), but `DataFrame.iterrows`
    # is stubbed as yielding a bare `Hashable` key; narrow it for the unpacking below.
    stratum_rows = cast(
        "Iterable[tuple[tuple[str, str], pd.Series[Any]]]", table.iterrows()
    )
    return [
        {
            "stratum": f"rating={rating_bin} | reviews={count_bin}",
            "products": int(row["products"]),
            **{split: int(row[split]) for split in SPLIT_ORDER},
        }
        for (rating_bin, count_bin), row in stratum_rows
    ]


def build_quality_report(
    *,
    build: BuildReport,
    products_raw: pd.DataFrame,
    products: pd.DataFrame,
    reviews: pd.DataFrame,
    related: pd.DataFrame,
    split_config: SplitConfig,
) -> dict[str, Any]:
    """Assemble the data-quality report from the build's own artifacts.

    Args:
        build: Accounting from one build (row counts, drops, input files, dates).
        products_raw: Loader rows for the kept products *before* product cleaning
            (raw title/brand/rank/price/description/feature presence).
        products: Final product frame as written, **including** ``split``,
            ``mean_rating`` and ``n_reviews``.
        reviews: Final review frame **including** ``split`` (schema-selected).
        related: Final related-products edge frame.
        split_config: Split fractions/seed/bin edges used for this build.

    Returns:
        JSON-serialisable report dict (plain ``int``/``float``/``str``/``None``).
    """
    review_rows = {
        "streamed_from_raw_file": int(build.review_rows_scanned),
        "after_candidate_filter": int(build.review_rows_selected),
        "after_duplicate_removal": int(
            build.review_rows_selected - build.duplicate_reviews_dropped
        ),
        "written_to_reviews_parquet": int(build.reviews_final),
    }
    products_collected = int(len(products_raw) + build.products_duplicate_rows_dropped)
    product_rows = {
        "metadata_rows_streamed": int(build.metadata_rows),
        "candidate_rows_collected_before_asin_dedup": products_collected,
        "after_asin_dedup": int(len(products_raw)),
        "written_to_products_parquet": int(build.products_final),
    }

    review_total = int(len(reviews))
    product_total = int(len(products))

    # -- Field presence before/after cleaning ---------------------------------
    review_fields: dict[str, dict[str, Any]] = {
        "review_text": _presence_row(
            review_total,
            _count_present(reviews["review_text_raw"]),
            _count_present(reviews["review_text_clean"]),
        ),
        "summary": _presence_row(
            review_total,
            _count_present(reviews["summary_raw"]),
            _count_present(reviews["summary_clean"]),
        ),
    }

    def _raw(column: str) -> int:
        return _count_present(products_raw[column]) if column in products_raw.columns else 0

    product_fields: dict[str, dict[str, Any]] = {
        "title": _presence_row(product_total, _raw("title"), _count_present(products["title_clean"])),
        "brand": _presence_row(product_total, _raw("brand"), _count_present(products["brand_clean"])),
        "description": _presence_row(
            product_total, _raw("description"), _count_present(products["description_clean"])
        ),
        "features": _presence_row(
            product_total, _raw("feature"), _count_present(products["features_clean"])
        ),
        "rank": _presence_row(
            product_total, _raw("rank"), int(products["rank_value"].notna().sum())
        ),
        "price": _presence_row(
            product_total, _raw("price_raw"), int(products["price"].notna().sum())
        ),
    }
    price_status = {
        str(key): int(count)
        for key, count in products["price_status"]
        .fillna("null")
        .value_counts()
        .items()
    }

    # -- Missingness per written column ----------------------------------------
    missingness = {
        "products.parquet": {"rows": product_total, "columns": _missingness(products)},
        "reviews.parquet": {"rows": review_total, "columns": _missingness(reviews)},
        "related_products.parquet": {
            "rows": int(len(related)),
            "columns": _missingness(related),
        },
    }

    # -- Split sizes and strata -------------------------------------------------
    sizes = split_sizes(products, reviews)
    split_section: dict[str, Any] = {
        "config": split_config.model_dump(),
        "per_split": {
            split: {
                "products": sizes[split]["products"],
                "products_share": _share(sizes[split]["products"], product_total),
                "reviews": sizes[split]["reviews"],
                "reviews_share": _share(sizes[split]["reviews"], review_total),
            }
            for split in SPLIT_ORDER
        },
        "strata": _strata_rows(products, split_config),
        "leakage_check": {
            "product_splits_unique": bool(
                products.groupby("asin", observed=True)["split"].nunique().max() <= 1
            )
            if product_total
            else True,
            "reviews_match_product_split": bool(
                (
                    reviews[["asin", "split"]].merge(
                        products[["asin", "split"]].drop_duplicates("asin"),
                        on="asin",
                        suffixes=("_review", "_product"),
                    )
                    ["split_review"]
                    == reviews[["asin", "split"]].merge(
                        products[["asin", "split"]].drop_duplicates("asin"),
                        on="asin",
                        suffixes=("_review", "_product"),
                    )["split_product"]
                ).all()
            )
            if review_total
            else True,
        },
    }

    # -- Rating / weak-label distribution per split -----------------------------
    def _sorted_dist(
        column: str, key_format: Callable[[Any], str]
    ) -> dict[str, dict[str, dict[str, float | int]]]:
        """Per-split distributions with numeric keys ascending, ``"null"`` last."""
        return {
            split: dict(
                sorted(
                    _distribution(reviews, column, split, key_format).items(),
                    key=lambda pair: (pair[0] == "null", _numeric_key(pair[0])),
                )
            )
            for split in SPLIT_ORDER
        }

    rating_dist = _sorted_dist("rating", _rating_key)
    label_dist = _sorted_dist("sentiment_label_weak", _label_key)

    # -- Reviews-per-product distribution ---------------------------------------
    counts = products["n_reviews"].dropna().astype("float64")
    if len(counts):
        reviews_per_product: dict[str, float | int] = {
            "min": float(counts.min()),
            "max": float(counts.max()),
            "mean": float(counts.mean()),
            **{
                f"p{quantile}": float(np.percentile(counts.to_numpy(), quantile))
                for quantile in _REVIEW_COUNT_QUANTILES
            },
        }
    else:
        reviews_per_product = {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            **{f"p{quantile}": 0.0 for quantile in _REVIEW_COUNT_QUANTILES},
        }

    # -- Date range (from the data itself, never today's date) ------------------
    dates = reviews["review_datetime"].dropna()
    date_range = {
        "min": dates.min().isoformat() if len(dates) else None,
        "max": dates.max().isoformat() if len(dates) else None,
        "note": (
            "Anchored on the review data itself; downstream recency features must "
            "use this max, never today's date."
        ),
    }

    mapping = build.rating_mapping
    return {
        "generated_at": build.built_at,
        "category": build.category,
        "config_path": build.config_path,
        "settings": {
            "candidate_min_reviews": build.candidate_min_reviews,
            "split": split_config.model_dump(),
            "rating_mapping": dict(mapping),
        },
        "inputs": {
            "reviews": {"path": build.reviews_path, "bytes": build.reviews_bytes},
            "metadata": {"path": build.metadata_path, "bytes": build.metadata_bytes},
        },
        "cleaning_steps": {"reviews": review_rows, "products": product_rows},
        "field_presence": {
            "note": (
                "Measured on the rows written to parquet: 'before' = raw source value "
                "present, 'after' = value present after cleaning."
            ),
            "reviews": review_fields,
            "products": product_fields,
            "price_status": price_status,
        },
        "missingness": {
            "note": "Null counts (None/NaT/pd.NA); empty strings are not nulls.",
            "tables": missingness,
        },
        "split_sizes": split_section,
        "rating_distribution_per_split": rating_dist,
        "label_distribution_per_split": label_dist,
        "weak_label_note": (
            "Weak supervision from star ratings via rating_mapping "
            f"(rating <= {mapping['negative_max']} -> 0 negative, "
            f"rating == {mapping['neutral_val']} -> 1 neutral, "
            f"rating >= {mapping['positive_min']} -> 2 positive) - not true sentiment; "
            "'null' means the review has no rating."
        ),
        "reviews_per_product": reviews_per_product,
        "review_date_range": date_range,
        "row_counts": {
            "products": product_total,
            "reviews": review_total,
            "related_edges": int(len(related)),
        },
    }


def _numeric_key(key: str) -> float:
    """Sort helper: numeric string keys ascending; non-numeric sorts after numbers."""
    try:
        return float(key)
    except ValueError:
        return float("inf")


# ------------------------------------------------------------------------------
# Markdown rendering (pure function of the report dict)
# ------------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]], right: set[int] | None = None) -> list[str]:
    """Render a GitHub-flavoured Markdown table (``right`` = right-aligned columns)."""
    right = right or set()
    aligns = ["---:" if index in right else "---" for index in range(len(headers))]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(aligns) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _fmt_int(value: int) -> str:
    """Format an integer with thousands separators (display only)."""
    return f"{value:,}"


def _fmt_share(share: float) -> str:
    """Format a share as a percentage (display only)."""
    return f"{share:.1%}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report dict as the Markdown data-quality report.

    Args:
        report: Output of :func:`build_quality_report`.

    Returns:
        Complete Markdown document (ends with a newline).
    """
    settings = report["settings"]
    split_settings = settings["split"]
    inputs = report["inputs"]
    lines: list[str] = [
        f"# Data Quality Report - {report['category']}",
        "",
        f"- **Generated:** {report['generated_at']}",
        f"- **Category:** `{report['category']}`",
        (
            f"- **Config:** `{report['config_path']}` "
            f"(`dataset.audit.candidate_min_reviews` = "
            f"{_fmt_int(settings['candidate_min_reviews'])}, "
            f"`dataset.split.seed` = {split_settings['seed']})"
        ),
        (
            f"- **Reviews file:** `{inputs['reviews']['path']}` "
            f"({_fmt_int(inputs['reviews']['bytes'])} bytes)"
        ),
        (
            f"- **Metadata file:** `{inputs['metadata']['path']}` "
            f"({_fmt_int(inputs['metadata']['bytes'])} bytes)"
        ),
        "",
        "All values below were computed by `scripts/build_dataset.py` from the raw "
        "files.",
        "",
        "## Row counts before/after each cleaning step",
        "",
        "### Reviews",
        "",
    ]
    review_steps = {
        "Streamed from raw file": "streamed_from_raw_file",
        "After candidate-product filter": "after_candidate_filter",
        "After duplicate removal": "after_duplicate_removal",
        "Written to reviews.parquet": "written_to_reviews_parquet",
    }
    lines += _table(
        ["Step", "Rows"],
        [[step, _fmt_int(report["cleaning_steps"]["reviews"][key])] for step, key in review_steps.items()],
        right={1},
    )
    lines += ["", "### Products", ""]
    product_steps = {
        "Metadata rows streamed": "metadata_rows_streamed",
        "Candidate rows collected (before ASIN de-duplication)":
            "candidate_rows_collected_before_asin_dedup",
        "After ASIN de-duplication": "after_asin_dedup",
        "Written to products.parquet": "written_to_products_parquet",
    }
    lines += _table(
        ["Step", "Rows"],
        [
            [step, _fmt_int(report["cleaning_steps"]["products"][key])]
            for step, key in product_steps.items()
        ],
        right={1},
    )

    presence = report["field_presence"]
    lines += [
        "",
        "## Field presence before/after cleaning",
        "",
        presence["note"],
        "",
        "### Reviews",
        "",
    ]
    lines += _table(
        ["Field", "Before (raw present)", "After (clean present)", "Missing after"],
        [
            [
                field,
                f"{_fmt_int(values['before_present'])} ({_fmt_share(values['share_before'])})",
                f"{_fmt_int(values['after_present'])} ({_fmt_share(values['share_after'])})",
                _fmt_int(values["after_missing"]),
            ]
            for field, values in presence["reviews"].items()
        ],
        right={1, 2, 3},
    )
    lines += ["", "### Products", ""]
    lines += _table(
        ["Field", "Before (raw present)", "After (clean present)", "Missing after"],
        [
            [
                field,
                f"{_fmt_int(values['before_present'])} ({_fmt_share(values['share_before'])})",
                f"{_fmt_int(values['after_present'])} ({_fmt_share(values['share_after'])})",
                _fmt_int(values["after_missing"]),
            ]
            for field, values in presence["products"].items()
        ],
        right={1, 2, 3},
    )
    lines += ["", "### Price parse status (products)", ""]
    lines += _table(
        ["Status", "Products"],
        [[status, _fmt_int(count)] for status, count in sorted(presence["price_status"].items())],
        right={1},
    )

    missing_note = report["missingness"]["note"]
    lines += ["", "## Missingness per column", "", missing_note, ""]
    for table_name, table in report["missingness"]["tables"].items():
        lines += [f"### {table_name} ({_fmt_int(table['rows'])} rows)", ""]
        lines += _table(
            ["Column", "Null", "Share"],
            [
                [column, _fmt_int(values["null"]), _fmt_share(values["share"])]
                for column, values in table["columns"].items()
            ],
            right={1, 2},
        )
        lines.append("")

    split_section = report["split_sizes"]
    config = split_section["config"]
    lines += [
        "## Split sizes (product-level split, no leakage)",
        "",
        (
            f"Configured fractions train/val/test = "
            f"{config['train_fraction']:g}/{config['val_fraction']:g}/{config['test_fraction']:g}, "
            f"seed = {config['seed']}; every review inherits its product's split."
        ),
        "",
    ]
    lines += _table(
        ["Split", "Products", "Share", "Reviews", "Share", "Config fraction"],
        [
            [
                split,
                _fmt_int(values["products"]),
                _fmt_share(values["products_share"]),
                _fmt_int(values["reviews"]),
                _fmt_share(values["reviews_share"]),
                _fmt_share(config[f"{split}_fraction"]),
            ]
            for split, values in split_section["per_split"].items()
        ],
        right={1, 2, 3, 4, 5},
    )
    leakage = split_section["leakage_check"]
    lines += [
        "",
        (
            f"Leakage check: every product in exactly one split = "
            f"{leakage['product_splits_unique']}; every review's split matches its "
            f"product's split = {leakage['reviews_match_product_split']}."
        ),
        "",
        "## Stratification strata (average-rating bin x review-count bin)",
        "",
    ]
    lines += _table(
        ["Stratum", "Products", *SPLIT_ORDER],
        [
            [
                row["stratum"],
                _fmt_int(row["products"]),
                *[_fmt_int(row[split]) for split in SPLIT_ORDER],
            ]
            for row in split_section["strata"]
        ],
        right={1, 2, 3, 4},
    )

    lines += ["", "## Rating distribution per split (star rating -> count (share))", ""]
    rating_keys = sorted(
        {key for dist in report["rating_distribution_per_split"].values() for key in dist},
        key=_numeric_key,
    )
    lines += _table(
        ["Rating", *SPLIT_ORDER],
        [
            [
                key,
                *[
                    (
                        f"{_fmt_int(report['rating_distribution_per_split'][split][key]['count'])} "
                        f"({_fmt_share(report['rating_distribution_per_split'][split][key]['share'])})"
                    )
                    if key in report["rating_distribution_per_split"][split]
                    else "0 (0.0%)"
                    for split in SPLIT_ORDER
                ],
            ]
            for key in rating_keys
        ],
        right={1, 2, 3},
    )

    lines += [
        "",
        "## Weak-label distribution per split (label -> count (share))",
        "",
        report["weak_label_note"],
        "",
    ]
    label_keys = sorted(
        {key for dist in report["label_distribution_per_split"].values() for key in dist},
        key=_numeric_key,
    )
    lines += _table(
        ["Label", *SPLIT_ORDER],
        [
            [
                key,
                *[
                    (
                        f"{_fmt_int(report['label_distribution_per_split'][split][key]['count'])} "
                        f"({_fmt_share(report['label_distribution_per_split'][split][key]['share'])})"
                    )
                    if key in report["label_distribution_per_split"][split]
                    else "0 (0.0%)"
                    for split in SPLIT_ORDER
                ],
            ]
            for key in label_keys
        ],
        right={1, 2, 3},
    )

    per_product = report["reviews_per_product"]
    lines += ["", "## Reviews per product (distribution over written products)", ""]
    lines += _table(
        ["Statistic", "Value"],
        [
            ["min", f"{per_product['min']:.0f}"],
            *[[f"p{quantile}", f"{per_product[f'p{quantile}']:.1f}"] for quantile in _REVIEW_COUNT_QUANTILES],
            ["max", f"{per_product['max']:.0f}"],
            ["mean", f"{per_product['mean']:.1f}"],
        ],
        right={1},
    )

    date_range = report["review_date_range"]
    lines += ["", "## Review date range", ""]
    lines += _table(
        ["Metric", "Value"],
        [["Earliest review", date_range["min"] or "n/a"], ["Latest review", date_range["max"] or "n/a"]],
        right={1},
    )
    lines += ["", f"Note: {date_range['note']}", ""]
    return "\n".join(lines)


def write_quality_report(report: dict[str, Any], reports_dir: Path) -> list[Path]:
    """Write ``data_quality_report.json`` and ``.md`` into ``reports_dir``.

    Args:
        report: Output of :func:`build_quality_report`.
        reports_dir: Target directory (created when missing).

    Returns:
        The two written paths ``[json_path, markdown_path]``.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / JSON_FILENAME
    markdown_path = reports_dir / MARKDOWN_FILENAME
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return [json_path, markdown_path]

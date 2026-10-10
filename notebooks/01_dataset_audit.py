# %% [markdown]
# # Dataset Audit - UCSD Amazon Review Data (2018)
#
# Streams the FULL per-category review file and metadata file for the active category
# (`dataset.category` in configs/config.yaml) and reports coverage, distributions and an
# explicit PASS/FAIL decision rubric. Outputs:
#
#   * `reports/dataset_audit.json` - machine-readable audit
#   * `reports/dataset_audit.md`    - human-readable audit (for the FYP report)
#
# Run from the repository root:
#
# ```bash
# python notebooks/01_dataset_audit.py
# ```
#
# Every number below is computed by this script from the raw files - none are
# hard-coded. The rubric thresholds come from `dataset.audit` in configs/config.yaml.

# %%
"""Dataset audit for the active UCSD Amazon Review Data (2018) category.

Streams both raw files in chunks (bounded memory), collects field-availability and
distribution statistics, then evaluates the dataset-acceptance rubric whose thresholds
are configured in configs/config.yaml (``dataset.audit``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from munasib.core.config import Settings, load_settings
from munasib.core.exceptions import DataNotFoundError
from munasib.data.loaders import (
    LoadStats,
    is_present,
    iter_metadata_chunks,
    iter_review_chunks,
)

# %% [markdown]
# ## 1. Configuration and paths

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CHUNK_SIZE = 50_000

settings: Settings = load_settings(CONFIG_PATH)
category: str = settings.dataset.category
rubric_cfg = settings.dataset.audit


def resolve(path: Path) -> Path:
    """Anchor a config-relative path to the repository root.

    Args:
        path: Path from ``paths.*`` in the config (relative to the repo root).

    Returns:
        Absolute path.
    """
    return path if path.is_absolute() else PROJECT_ROOT / path


source = settings.dataset.sources.get(category)
if source is None:
    raise DataNotFoundError(
        f"No dataset.sources entry for category '{category}' in {CONFIG_PATH}. "
        f"Registered: {', '.join(sorted(settings.dataset.sources)) or '<none>'}."
    )

reviews_path = resolve(settings.paths.raw_data_dir) / Path(source.reviews_url).name
metadata_path = resolve(settings.paths.raw_data_dir) / Path(source.metadata_url).name
reports_dir = resolve(settings.paths.reports_dir)

missing = [p for p in (reviews_path, metadata_path) if not p.exists()]
if missing:
    raise DataNotFoundError(
        "Raw dataset file(s) missing: "
        + "; ".join(str(p) for p in missing)
        + f". Run: python scripts/download_data.py --category {category}"
    )

print(f"category      : {category}")
print(f"reviews file  : {reviews_path}")
print(f"metadata file : {metadata_path}")
print(
    "rubric        : "
    f"price>={rubric_cfg.min_price_coverage:.0%} | "
    f"products>={rubric_cfg.min_candidate_products} with >={rubric_cfg.candidate_min_reviews} reviews | "
    f"related>={rubric_cfg.min_related_links_share:.0%}"
)

# %% [markdown]
# ## 2. Review-file pass (streamed)
#
# Availability for review-side fields (`vote`, `verified`, `reviewText`,
# `unixReviewTime`), rating distribution, reviews-per-product counts, review date
# range and review-text length samples.

# %%
review_stats = LoadStats()
review_availability: dict[str, int] = {}
review_rows = 0
reviews_without_asin = 0
rating_counts: list[pd.Series] = []
text_lengths: list[np.ndarray] = []
date_min_ts: float | None = None
date_max_ts: float | None = None
dates_unparsable = 0
dates_absent = 0
product_counts = pd.Series(dtype="int64")

for chunk in iter_review_chunks(reviews_path, chunk_size=CHUNK_SIZE, stats=review_stats):
    review_rows += len(chunk)

    for col in ("vote", "verified", "reviewText", "unixReviewTime"):
        if col in chunk.columns:
            review_availability[col] = review_availability.get(col, 0) + int(
                chunk[col].map(is_present).sum()
            )

    if "asin" in chunk.columns:
        present = chunk["asin"].map(is_present)
        reviews_without_asin += int((~present).sum())
        valid_asins = chunk.loc[present, "asin"]
        product_counts = product_counts.add(valid_asins.value_counts(), fill_value=0)
    else:
        reviews_without_asin += len(chunk)

    if "rating" in chunk.columns:
        ratings = pd.to_numeric(chunk["rating"], errors="coerce").dropna()
        if not ratings.empty:
            rating_counts.append(ratings.value_counts())

    if "reviewText" in chunk.columns:
        texts = chunk.loc[chunk["reviewText"].map(is_present), "reviewText"].astype(str)
        text_lengths.append(texts.str.len().to_numpy())

    if "unixReviewTime" in chunk.columns:
        ts = pd.to_numeric(chunk["unixReviewTime"], errors="coerce")
        absent = int((~chunk["unixReviewTime"].map(is_present)).sum())
        dates_absent += absent
        numeric = ts.dropna()
        dates_unparsable += int(chunk["unixReviewTime"].map(is_present).sum()) - len(numeric)
        if not numeric.empty:
            lo, hi = float(numeric.min()), float(numeric.max())
            date_min_ts = lo if date_min_ts is None else min(date_min_ts, lo)
            date_max_ts = hi if date_max_ts is None else max(date_max_ts, hi)
    else:
        dates_absent += len(chunk)

    del chunk

print(f"reviews parsed : {review_rows:,} (malformed lines skipped: {review_stats.malformed_lines})")

# %% [markdown]
# ## 3. Metadata-file pass (streamed)
#
# Availability for product-side fields (`price`, `rank`, `also_buy`, `also_view`,
# `brand`, `description`, `feature`), price parse-status counts, the product set and
# the sets needed for the rubric joins.

# %%
metadata_stats = LoadStats()
metadata_availability: dict[str, int] = {}
price_present = 0
price_parsable = 0
price_status_counts: dict[str, int] = {}
metadata_rows = 0
products_without_asin = 0
product_set: set[str] = set()
asins_with_parsable_price: set[str] = set()
asins_with_related_links: set[str] = set()
related_asins: set[str] = set()


def _collect_related(values: list[Any], target: set[str]) -> None:
    """Add every ASIN from related-link lists into ``target``.

    Args:
        values: Column values; lists contribute their entries, anything else is ignored.
        target: Set updated in place.
    """
    for value in values:
        if isinstance(value, list):
            target.update(str(v) for v in value if v is not None and str(v).strip())


for chunk in iter_metadata_chunks(
    metadata_path, chunk_size=CHUNK_SIZE, stats=metadata_stats, category=category
):
    metadata_rows += len(chunk)

    for raw_col, col in (
        ("price_raw", "price"),
        ("rank", "rank"),
        ("also_buy", "also_buy"),
        ("also_view", "also_view"),
        ("brand", "brand"),
        ("description", "description"),
        ("feature", "feature"),
    ):
        # "price" availability counts the RAW field (a present-but-unparsable value
        # is still available); parsability is reported separately below.
        if raw_col in chunk.columns:
            metadata_availability[col] = metadata_availability.get(col, 0) + int(
                chunk[raw_col].map(is_present).sum()
            )

    if "price_status" in chunk.columns:
        for status, count in chunk["price_status"].value_counts().items():
            price_status_counts[str(status)] = price_status_counts.get(str(status), 0) + int(count)
        parsable_mask = chunk["price_status"].isin(("single", "range"))
        price_parsable += int(parsable_mask.sum())
        price_present += int((chunk["price_status"] != "missing").sum())
        if "asin" in chunk.columns:
            parsable_asins = chunk.loc[parsable_mask, "asin"]
            asins_with_parsable_price.update(str(a) for a in parsable_asins if is_present(a))

    if "asin" in chunk.columns:
        present = chunk["asin"].map(is_present)
        products_without_asin += int((~present).sum())
        asins = [str(a) for a in chunk.loc[present, "asin"]]
        product_set.update(asins)
        if "also_buy" in chunk.columns:
            linked = chunk.loc[chunk["also_buy"].map(is_present), "asin"]
            asins_with_related_links.update(str(a) for a in linked if is_present(a))
        if "also_view" in chunk.columns:
            linked_view = chunk.loc[chunk["also_view"].map(is_present), "asin"]
            asins_with_related_links.update(str(a) for a in linked_view if is_present(a))
        if "also_buy" in chunk.columns:
            _collect_related(list(chunk["also_buy"]), related_asins)
        if "also_view" in chunk.columns:
            _collect_related(list(chunk["also_view"]), related_asins)
    else:
        products_without_asin += len(chunk)

    del chunk

print(f"products parsed: {metadata_rows:,} (malformed lines skipped: {metadata_stats.malformed_lines})")

# %% [markdown]
# ## 4. Derived metrics (joins between the two passes)

# %%


def ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    """Compute a safe share; returns None instead of dividing by zero.

    Args:
        numerator: Part count.
        denominator: Whole count.

    Returns:
        numerator/denominator, or None when the denominator is 0/None.
    """
    if numerator is None or not denominator:
        return None
    return float(numerator) / float(denominator)


def fmt_pct(value: float | None, digits: int = 1) -> str:
    """Format a share as a percentage for console/markdown output.

    Args:
        value: Share in [0, 1] or None.
        digits: Decimal places.

    Returns:
        e.g. ``"82.3%"`` or ``"n/a"``.
    """
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def describe(values: np.ndarray, percentiles: tuple[int, ...] = (0, 10, 25, 50, 75, 90, 99, 100)) -> dict[str, float]:
    """Summarise a numeric array as percentile statistics.

    Args:
        values: 1-D sample.
        percentiles: Percentile grid.

    Returns:
        Dict of ``pNN -> value`` (rounded), plus ``mean``; empty dict when no samples.
    """
    if values.size == 0:
        return {}
    grid = np.percentile(values, percentiles)
    out: dict[str, float] = {f"p{p}": round(float(v), 2) for p, v in zip(percentiles, grid, strict=True)}
    out["mean"] = round(float(np.mean(values)), 2)
    return out


review_products = {str(a): int(c) for a, c in product_counts.items()}
n_review_products = len(review_products)
counts_arr = np.array(list(review_products.values()), dtype=np.int64) if review_products else np.array([], dtype=np.int64)

shares_at: dict[str, dict[str, Any]] = {}
for threshold in (20, 50, 100):
    n = sum(1 for c in review_products.values() if c >= threshold)
    shares_at[f"gte_{threshold}"] = {
        "threshold_reviews": threshold,
        "count": n,
        "share_of_reviewed_products": ratio(n, n_review_products),
    }

candidate_threshold = rubric_cfg.candidate_min_reviews
candidates = sorted(a for a, c in review_products.items() if c >= candidate_threshold)
n_candidates = len(candidates)
cand_with_price = sum(1 for a in candidates if a in asins_with_parsable_price)
cand_with_related = sum(1 for a in candidates if a in asins_with_related_links)
cand_all_three = sum(
    1
    for a in candidates
    if a in asins_with_parsable_price and a in asins_with_related_links
)

reviewed_missing_meta = sum(1 for a in review_products if a not in product_set)

related_total = len(related_asins)
related_in_catalog = len(related_asins & product_set)

price_coverage = ratio(cand_with_price, n_candidates)
related_links_share = ratio(cand_with_related, n_candidates)

ratings_combined = (
    pd.concat(rating_counts).groupby(level=0).sum() if rating_counts else pd.Series(dtype="int64")
)
rating_distribution = {
    str(rating): {"count": int(count), "share": ratio(int(count), int(ratings_combined.sum()))}
    for rating, count in sorted(ratings_combined.items(), key=lambda kv: float(kv[0]))
}

text_length_values = np.concatenate(text_lengths) if text_lengths else np.array([], dtype=np.int64)
review_dates = {
    "min_unix": date_min_ts,
    "max_unix": date_max_ts,
    "min_iso": (
        datetime.fromtimestamp(date_min_ts, tz=UTC).strftime("%Y-%m-%d")
        if date_min_ts is not None
        else None
    ),
    "max_iso": (
        datetime.fromtimestamp(date_max_ts, tz=UTC).strftime("%Y-%m-%d")
        if date_max_ts is not None
        else None
    ),
    "records_absent": dates_absent,
    "records_unparsable": dates_unparsable,
}

print(f"candidate products (>= {candidate_threshold} reviews): {n_candidates}")
print(f"price coverage among candidates: {fmt_pct(price_coverage)}")
print(f"related-links share among candidates: {fmt_pct(related_links_share)}")

# %% [markdown]
# ## 5. Decision rubric (thresholds: `dataset.audit` in configs/config.yaml)
#
# * **C1** - price coverage among candidate products (>= `candidate_min_reviews` reviews)
#   must be >= `min_price_coverage`.
# * **C2** - number of products with >= `candidate_min_reviews` reviews must be
#   >= `min_candidate_products`.
# * **C3** - related-product links (non-empty `also_buy` OR `also_view`) among candidate
#   products must be >= `min_related_links_share`.
#
# The overall recommendation is only "keep" when every criterion passes; otherwise the
# fallback category and the exact config key to change are printed.

# %%


criteria: list[dict[str, Any]] = [
    {
        "id": "C1",
        "name": "price coverage among candidate products",
        "formula": f"candidates with parsable price / candidates (>= {candidate_threshold} reviews)",
        "observed": price_coverage,
        "observed_display": fmt_pct(price_coverage),
        "threshold": rubric_cfg.min_price_coverage,
        "threshold_display": fmt_pct(rubric_cfg.min_price_coverage),
        "comparator": ">=",
        "passed": price_coverage is not None and price_coverage >= rubric_cfg.min_price_coverage,
    },
    {
        "id": "C2",
        "name": f"products with >= {candidate_threshold} reviews",
        "formula": f"count(reviews >= {candidate_threshold})",
        "observed": n_candidates,
        "observed_display": f"{n_candidates:,}",
        "threshold": rubric_cfg.min_candidate_products,
        "threshold_display": f"{rubric_cfg.min_candidate_products:,}",
        "comparator": ">=",
        "passed": n_candidates >= rubric_cfg.min_candidate_products,
    },
    {
        "id": "C3",
        "name": "candidate products with related-product links",
        "formula": "candidates with non-empty also_buy OR also_view / candidates",
        "observed": related_links_share,
        "observed_display": fmt_pct(related_links_share),
        "threshold": rubric_cfg.min_related_links_share,
        "threshold_display": fmt_pct(rubric_cfg.min_related_links_share),
        "comparator": ">=",
        "passed": related_links_share is not None and related_links_share >= rubric_cfg.min_related_links_share,
    },
]

all_passed = all(bool(c["passed"]) for c in criteria)
n_passed = sum(1 for c in criteria if c["passed"])
fallback = rubric_cfg.fallback_category

if all_passed:
    recommendation = (
        f"KEEP '{category}' - all {len(criteria)} rubric criteria pass. "
        "No config change required."
    )
    recommendation_action = None
else:
    failed_ids = ", ".join(str(c["id"]) for c in criteria if not c["passed"])
    recommendation = (
        f"SWITCH to '{fallback}' - failed criteria: {failed_ids}. "
        f'Set `dataset.category: "{fallback}"` in configs/config.yaml '
        f"(then run: python scripts/download_data.py --category {fallback})"
    )
    recommendation_action = f'dataset.category: "{fallback}"'

print()
print("=" * 78)
print(f"DATASET ACCEPTANCE RUBRIC - {category}")
print("=" * 78)
for criterion in criteria:
    verdict = "PASS" if criterion["passed"] else "FAIL"
    print(
        f"[{criterion['id']}] {criterion['name']}: "
        f"{criterion['observed_display']} {criterion['comparator']} "
        f"{criterion['threshold_display']}  ->  {verdict}"
    )
print("-" * 78)
print(f"RESULT: {n_passed}/{len(criteria)} criteria passed -> {recommendation}")
print("=" * 78)

# %% [markdown]
# ## 6. Persist `reports/dataset_audit.json` and `reports/dataset_audit.md`

# %%


def availability_block(
    present: dict[str, int], columns: tuple[str, ...], total_rows: int
) -> dict[str, Any]:
    """Build the availability report for one file.

    Args:
        present: Per-field counts of rows where the field is present and non-empty.
        columns: Expected field names (absent columns report 0).
        total_rows: Denominator (rows in the file).

    Returns:
        Mapping field -> {present, share} with share=None when total_rows is 0.
    """
    return {
        col: {"present": int(present.get(col, 0)), "share": ratio(present.get(col, 0), total_rows)}
        for col in columns
    }


audit: dict[str, Any] = {
    "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    "category": category,
    "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
    "source_files": {
        "reviews": {"path": str(reviews_path.relative_to(PROJECT_ROOT)), "bytes": reviews_path.stat().st_size},
        "metadata": {"path": str(metadata_path.relative_to(PROJECT_ROOT)), "bytes": metadata_path.stat().st_size},
    },
    "row_counts": {
        "reviews": review_rows,
        "products": metadata_rows,
        "distinct_reviewed_products": n_review_products,
        "reviews_without_asin": reviews_without_asin,
        "products_without_asin": products_without_asin,
        "reviewed_products_missing_from_metadata": reviewed_missing_meta,
    },
    "field_availability": {
        "reviews": availability_block(
            review_availability, ("vote", "verified", "reviewText", "unixReviewTime"), review_rows
        ),
        "metadata": availability_block(
            metadata_availability,
            ("price", "rank", "also_buy", "also_view", "brand", "description", "feature"),
            metadata_rows,
        ),
        "price_parse_status": price_status_counts,
        "price_field_present": {"count": price_present, "share": ratio(price_present, metadata_rows)},
        "price_parsable": {"count": price_parsable, "share": ratio(price_parsable, metadata_rows)},
    },
    "review_dates": review_dates,
    "rating_distribution": rating_distribution,
    "reviews_per_product": {
        "mean": round(float(counts_arr.mean()), 2) if counts_arr.size else None,
        "distribution": describe(counts_arr),
        "max": int(counts_arr.max()) if counts_arr.size else None,
    },
    "product_review_shares": shares_at,
    "candidate_products": {
        "definition": f">= {candidate_threshold} reviews (dataset.audit.candidate_min_reviews)",
        "count": n_candidates,
        "with_parsable_price": cand_with_price,
        "price_coverage": price_coverage,
        "with_related_links": cand_with_related,
        "related_links_share": related_links_share,
        "with_price_and_links": cand_all_three,
        "all_three_share": ratio(cand_all_three, n_review_products),
    },
    "related_asin_coverage": {
        "distinct_related_asins": related_total,
        "present_in_product_set": related_in_catalog,
        "share": ratio(related_in_catalog, related_total),
    },
    "review_text_length_chars": describe(text_length_values),
    "parse_stats": {
        "reviews": review_stats.summary(),
        "metadata": metadata_stats.summary(),
    },
    "rubric": {
        "thresholds": {
            "candidate_min_reviews": rubric_cfg.candidate_min_reviews,
            "min_price_coverage": rubric_cfg.min_price_coverage,
            "min_candidate_products": rubric_cfg.min_candidate_products,
            "min_related_links_share": rubric_cfg.min_related_links_share,
            "fallback_category": rubric_cfg.fallback_category,
        },
        "criteria": criteria,
        "passed": n_passed,
        "total": len(criteria),
        "all_passed": all_passed,
        "recommendation": recommendation,
        "config_change": recommendation_action,
    },
}

reports_dir.mkdir(parents=True, exist_ok=True)
json_path = reports_dir / "dataset_audit.json"
with json_path.open("w", encoding="utf-8") as handle:
    json.dump(audit, handle, indent=2, ensure_ascii=False)

# %% [markdown]
# ### Markdown report

# %%


def _table(block: dict[str, Any]) -> str:
    """Render an availability block as a markdown table.

    Args:
        block: Mapping field -> {present, share}.

    Returns:
        Markdown table string.
    """
    lines = ["| Field | Present | Share |", "|---|---:|---:|"]
    for field_name, stats in block.items():
        lines.append(f"| {field_name} | {stats['present']:,} | {fmt_pct(stats['share'])} |")
    return "\n".join(lines)


def _dist_table(dist: dict[str, float]) -> str:
    """Render a percentile dict as a markdown table row set.

    Args:
        dist: Mapping like ``{"p10": 3.0, ..., "mean": 12.5}``.

    Returns:
        Markdown table string.
    """
    lines = ["| Statistic | Value |", "|---|---:|"]
    for key, value in dist.items():
        label = "mean" if key == "mean" else key
        lines.append(f"| {label} | {value:,.2f} |")
    return "\n".join(lines)


rating_lines = ["| Rating | Count | Share |", "|---|---:|---:|"]
for rating, stats in rating_distribution.items():
    rating_lines.append(f"| {rating} | {stats['count']:,} | {fmt_pct(stats['share'])} |")

share_lines = ["| Threshold | Products | Share of reviewed products |", "|---|---:|---:|"]
for stats in shares_at.values():
    share_lines.append(
        f"| >= {stats['threshold_reviews']} reviews | {stats['count']:,} | {fmt_pct(stats['share_of_reviewed_products'])} |"
    )

rubric_lines = ["| ID | Criterion | Observed | Threshold | Verdict |", "|---|---|---:|---:|---|"]
for criterion in criteria:
    rubric_lines.append(
        f"| {criterion['id']} | {criterion['name']} | {criterion['observed_display']} | "
        f"{criterion['comparator']} {criterion['threshold_display']} | "
        f"{'**PASS**' if criterion['passed'] else '**FAIL**'} |"
    )

status_lines = ["| Price parse status | Records |", "|---|---:|"]
for status, count in sorted(price_status_counts.items()):
    status_lines.append(f"| {status} | {count:,} |")

md = f"""# Dataset Audit - UCSD Amazon Review Data (2018)

- **Generated:** {audit['generated_at']}
- **Category:** `{category}`
- **Config:** `{audit['config_path']}` (`dataset.category`)
- **Reviews file:** `{audit['source_files']['reviews']['path']}` ({audit['source_files']['reviews']['bytes']:,} bytes)
- **Metadata file:** `{audit['source_files']['metadata']['path']}` ({audit['source_files']['metadata']['bytes']:,} bytes)

All values below were computed by `notebooks/01_dataset_audit.py` from the raw files.

## Row counts

| Metric | Value |
|---|---:|
| Reviews | {review_rows:,} |
| Products (metadata) | {metadata_rows:,} |
| Distinct reviewed products | {n_review_products:,} |
| Reviews without ASIN | {reviews_without_asin:,} |
| Products without ASIN | {products_without_asin:,} |
| Reviewed products missing from metadata | {reviewed_missing_meta:,} ({fmt_pct(ratio(reviewed_missing_meta, n_review_products))}) |
| Malformed review lines skipped | {review_stats.malformed_lines:,} |
| Malformed metadata lines skipped | {metadata_stats.malformed_lines:,} |

## Field availability - review file ({review_rows:,} rows)

{_table(audit['field_availability']['reviews'])}

## Field availability - metadata file ({metadata_rows:,} rows)

{_table(audit['field_availability']['metadata'])}

Price parse status: price present {price_present:,} ({fmt_pct(ratio(price_present, metadata_rows))}), parsable {price_parsable:,} ({fmt_pct(ratio(price_parsable, metadata_rows))}).

| Status | Records |
|---|---:|
{chr(10).join(status_lines[2:])}

## Review dates

- Range: **{review_dates['min_iso'] or 'n/a'}** to **{review_dates['max_iso'] or 'n/a'}** (unix {f"{review_dates['min_unix']:.0f}" if review_dates['min_unix'] is not None else 'n/a'} - {f"{review_dates['max_unix']:.0f}" if review_dates['max_unix'] is not None else 'n/a'})
- Records without timestamp: {dates_absent:,}; unparsable timestamps: {dates_unparsable:,}

## Rating distribution (weak-supervision source: 1-2 negative / 3 neutral / 4-5 positive)

{chr(10).join(rating_lines)}

## Reviews per product

- Mean: **{audit['reviews_per_product']['mean']}**; max: **{audit['reviews_per_product']['max']}**

{_dist_table(audit['reviews_per_product']['distribution'])}

### Share of products by review-count threshold

{chr(10).join(share_lines)}

## Candidate products (>= {candidate_threshold} reviews)

| Metric | Value |
|---|---:|
| Candidate products | {n_candidates:,} |
| With parsable price | {cand_with_price:,} ({fmt_pct(price_coverage)}) |
| With related-product links | {cand_with_related:,} ({fmt_pct(related_links_share)}) |
| With >= {candidate_threshold} reviews AND price AND links (share of reviewed products) | {cand_all_three:,} ({fmt_pct(ratio(cand_all_three, n_review_products))}) |

## Related-product link coverage

- Distinct related ASINs referenced: **{related_total:,}**
- Present in this category's product set: **{related_in_catalog:,}** ({fmt_pct(ratio(related_in_catalog, related_total))})
- Note: links pointing outside the category are expected and remain available for
  cross-category evidence checks.

## Review-text length (characters)

{_dist_table(audit['review_text_length_chars'])}

## Decision rubric

Thresholds from `dataset.audit` in `{audit['config_path']}`.

{chr(10).join(rubric_lines)}

**Result: {n_passed}/{len(criteria)} passed -> {recommendation}**
"""

md_path = reports_dir / "dataset_audit.md"
md_path.write_text(md, encoding="utf-8")
print(f"\nSaved: {json_path}")
print(f"Saved: {md_path}")

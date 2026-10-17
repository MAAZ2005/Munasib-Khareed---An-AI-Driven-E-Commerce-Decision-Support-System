# Dataset Notes - UCSD Amazon Review Data (2018)

Munasib Khareed analyzes **one category** of the UCSD Amazon Review Data (2018) release,
chosen by the acceptance rubric below. This file records the citation, the exact files
used, when they were accessed, and the audit numbers that justifies the choice.

## Citation

```bibtex
@inproceedings{ni2019justifying,
  title     = {Justifying Recommendations using Distantly-Labeled Reviews and
               Fine-grained Aspects},
  author    = {Ni, Jianmo and Li, Jiacheng and McAuley, Julian},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural
               Language Processing (EMNLP)},
  year      = {2019}
}
```

Jianmo Ni, Jiacheng Li, Julian McAuley, "Justifying recommendations using
distantly-labeled reviews and fine-grained aspects", EMNLP 2019.

## Active category and source files

- **Active category:** `Appliances` (`dataset.category` in `configs/config.yaml`)
- **Registered fallback:** `Luxury_Beauty`
- **Access date:** 2026-10-10
- **Downloaded via:** `python scripts/download_data.py` (resumable, size-checked,
  gzip-CRC-verified on completion)

| File | URL | Bytes |
|---|---|---:|
| `data/raw/Appliances.json.gz` | <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Appliances.json.gz> | 68,569,830 |
| `data/raw/meta_Appliances.json.gz` | <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Appliances.json.gz> | 60,207,582 |
| `data/raw/Luxury_Beauty.json.gz` (fallback) | <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Luxury_Beauty.json.gz> | not downloaded (Appliances passed the rubric) |
| `data/raw/meta_Luxury_Beauty.json.gz` (fallback) | <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Luxury_Beauty.json.gz> | not downloaded (Appliances passed the rubric) |

The **full per-category** files are used deliberately, *not* the 5-core subsets: the
Appliances 5-core file contains only about 2,277 reviews, far too few for per-product
evidence analysis.

## Format observations (Appliances, 2026-10-10 download)

Both files are gzip-compressed JSON Lines. Observed quirks that the loaders
(`src/munasib/data/loaders.py`) handle explicitly:

- Reviews: `vote` is a **string** (sometimes comma-formatted, e.g. `"1,234"`, often
  absent); rating lives in `overall`; `verified` is a real boolean; `reviewText` can be
  absent (~0.05%); `unixReviewTime` is always present.
- Metadata: `price` is a string (`"$12.99"`, empty string when unknown, and 109 records
  contain scraped CSS garbage which must never parse into a number); `rank` is a
  **list** of strings (occasionally a bare string); `also_buy`/`also_view` are
  **top-level** lists (not nested under `related`); `salesRank`/`related` do not occur
  in this file but are supported for other categories/generations.

## Audit summary (computed by `notebooks/01_dataset_audit.py`)

Full output: `reports/dataset_audit.json` and `reports/dataset_audit.md`
(generated 2026-10-10 from the files above).

- **Reviews:** 602,777 rows, **0** malformed lines; distinct reviewed products: 30,252;
  products in metadata: 30,445 (14 reviewed products missing from metadata, 0.0%).
- **Field availability - reviews:** `verified` 100.0%, `unixReviewTime` 100.0%,
  `reviewText` 99.9%, `vote` 10.8%.
- **Field availability - metadata:** `rank` 98.7%, `brand` 98.1%, `description` 92.4%,
  `feature` 85.5%, `price` 66.2% present (65.8% parsable), `also_view` 36.7%,
  `also_buy` 28.3%.
- **Review dates:** 2000-10-23 to 2018-10-04.
- **Rating distribution:** 5.0: 69.1%, 4.0: 12.5%, 1.0: 9.9%, 3.0: 5.1%, 2.0: 3.4%
  (weak supervision: 1-2 negative, 3 neutral, 4-5 positive).
- **Reviews per product:** mean 19.93, median 2, p90 32, max 6,510.
- **Products with >= 20 / >= 50 / >= 100 reviews:** 4,311 (14.3%) / 2,069 (6.8%) /
  1,146 (3.8%).
- **Candidate products (>= 50 reviews):** 2,069 - 1,678 (81.1%) have a parsable price,
  1,836 (88.7%) have related-product links, 1,477 (4.9% of all reviewed products) have
  >= 50 reviews *and* price *and* links.
- **Related-ASIN coverage:** 64,680 distinct related ASINs referenced, 12,525 (19.4%)
  inside the Appliances product set (the rest are expected cross-category links).
- **Review text length:** median 88 chars, mean 189.4, p99 1,648.

### Decision rubric (thresholds: `dataset.audit` in `configs/config.yaml`)

| ID | Criterion | Observed | Threshold | Verdict |
|---|---|---:|---:|---|
| C1 | Price coverage among candidate products | 81.1% | >= 70.0% | **PASS** |
| C2 | Products with >= 50 reviews | 2,069 | >= 500 | **PASS** |
| C3 | Candidate products with related-product links | 88.7% | >= 60.0% | **PASS** |

**Result: 3/3 passed -> KEEP `Appliances`.** Had any criterion failed, the audit
recommends switching by setting `dataset.category: "Luxury_Beauty"` in
`configs/config.yaml` (and running `python scripts/download_data.py --category
Luxury_Beauty`).

## Processed dataset (`data/processed/`)

```bash
python scripts/build_dataset.py                          # build (or skip when current)
python scripts/build_dataset.py --category Appliances     # category override
python scripts/build_dataset.py --force                   # rebuild even when current
```

The build runs four streaming passes (bounded memory): metadata ASINs, then review
counts per product, then candidate selection (`dataset.audit.candidate_min_reviews`,
currently 50 - read from config, never hardcoded), then cleaning of the candidate
subset only. Built on 2026-10-10 from the raw files above:

| Output | Rows | Contents |
|---|---:|---|
| `products.parquet` | 2,066 | `split`, cleaned title/description/features, normalised brand, parsed price/rank + `*_missing` flags, display-only `image_url` + `image_missing`, per-product review aggregates (`n_reviews`, `mean_rating`, first/last review time) |
| `reviews.parquet` | 425,993 | `split`, raw + cleaned text, `has_text`/`text_length`, `helpful_votes`, nullable `verified`, UTC dates, weak labels |
| `related_products.parquet` | 84,885 | `also_bought`/`also_viewed` edges + `in_subset` flag (22,349 in-subset) |
| `meta.json` | - | category, config values (incl. split seed/fractions), row/drop counts, split sizes, review date range, build timestamp |
| `reports/data_quality_report.{json,md}` | - | auto-generated cleaning counts, missingness, distributions per split, split sizes (see below) |

- **Candidate selection:** 30,252 distinct reviewed products -> 2,069 with >= 50
  reviews (audit) -> 2,066 kept (3 of them absent from the metadata file); 28,183
  products dropped below the threshold; 165,915 review rows dropped with
  non-candidate products.
- **De-duplication:** 10,869 duplicate review rows (reviewer + asin + timestamp +
  normalised text) and 38 duplicate product rows dropped. 15 products end slightly
  below 50 reviews *after* de-duplication - reported in `meta.json`, not hidden and
  not re-filtered.
- **Missing data stays missing:** 388 products have no parsable price, 13 no numeric
  rank, 8 no brand - each flagged by a boolean (`price_missing`, `rank_missing`,
  `brand_missing`), never imputed. 240 reviews carry no text and are kept with
  `has_text=false`.
- **Display-only images:** 1,073 of the 2,066 products carry an `image_url` (first
  `imageURLHighRes` URL, else the first `imageURL` URL) and the other 993 are flagged
  `image_missing=true`. A duplicate ASIN's image is recovered from the row that has
  one before duplicate rows are dropped (the kept row's own image always wins). Nothing
  in the analysis pipeline reads an image; see `docs/images.md`.
- **Review dates (candidate subset):** 2001-11-08 to 2018-10-04. Downstream "recent"
  features must anchor on `meta.json -> review_date_range.max`, never on today's date.

### Train/val/test split (product level, no leakage)

The split is assigned per **product**, never per review: each product is shuffled
inside its stratum - average-rating bin x review-count bin, cut points from
`dataset.split.rating_bin_edges` / `review_count_bin_edges` - with
`dataset.split.seed` (42) and allocated by the configured fractions
(0.70/0.15/0.15). Every review inherits its product's split, so no product can
appear in two splits. Both parquet tables carry a `split` column; `meta.json`
records the seeded config and the per-split sizes:

| Split | Products | Reviews |
|---|---:|---:|
| train | 1,444 | 303,317 |
| val | 310 | 62,961 |
| test | 312 | 59,715 |

Re-running the build with the same config and inputs reproduces byte-identical
split assignments; `meta.json -> settings` records everything that feeds the
freshness check (`--force` bypasses it).

### Data-quality report (`reports/data_quality_report.{json,md}`)

Written on every build by `scripts/build_dataset.py`. It contains row counts
before/after each cleaning step (602,777 -> 436,862 -> 425,993 reviews;
30,445 -> 2,104 -> 2,066 products), raw-vs-clean field presence, per-column
missingness for all three tables, rating and weak-label distributions per split,
the reviews-per-product distribution, the review date range, and the split sizes
with their strata (12 strata at the configured bin edges). Every number is
computed from the frames the build produced - none are written by hand - and the
JSON/Markdown pair is rendered from the same in-memory dict.

### Weak labels - read before reporting metrics

`sentiment_label_weak` in `reviews.parquet` is **weak supervision derived from the
star rating** (1-2 negative, 3 neutral, 4-5 positive; boundaries from
`review_intel.sentiment.rating_mapping` in `configs/config.yaml`). It assumes rating
and sentiment agree and is therefore **not true sentiment**. Every metric, chart or
claim built on this column must state that it uses weak labels.

## Re-running the audit

```bash
python scripts/download_data.py   # ensures data/raw/ is present and verified
python notebooks/01_dataset_audit.py
```

The script streams both files in chunks (bounded memory), writes
`reports/dataset_audit.json` and `reports/dataset_audit.md`, and prints the PASS/FAIL
rubric to the console. Every number is computed from the raw files - none are
hard-coded.

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

## Re-running the audit

```bash
python scripts/download_data.py   # ensures data/raw/ is present and verified
python notebooks/01_dataset_audit.py
```

The script streams both files in chunks (bounded memory), writes
`reports/dataset_audit.json` and `reports/dataset_audit.md`, and prints the PASS/FAIL
rubric to the console. Every number is computed from the raw files - none are
hard-coded.

# Dataset Audit - UCSD Amazon Review Data (2018)

- **Generated:** 2026-10-10T07:51:12+00:00
- **Category:** `Appliances`
- **Config:** `configs\config.yaml` (`dataset.category`)
- **Reviews file:** `data\raw\Appliances.json.gz` (68,569,830 bytes)
- **Metadata file:** `data\raw\meta_Appliances.json.gz` (60,207,582 bytes)

All values below were computed by `notebooks/01_dataset_audit.py` from the raw files.

## Row counts

| Metric | Value |
|---|---:|
| Reviews | 602,777 |
| Products (metadata) | 30,445 |
| Distinct reviewed products | 30,252 |
| Reviews without ASIN | 0 |
| Products without ASIN | 0 |
| Reviewed products missing from metadata | 14 (0.0%) |
| Malformed review lines skipped | 0 |
| Malformed metadata lines skipped | 0 |

## Field availability - review file (602,777 rows)

| Field | Present | Share |
|---|---:|---:|
| vote | 65,262 | 10.8% |
| verified | 602,777 | 100.0% |
| reviewText | 602,453 | 99.9% |
| unixReviewTime | 602,777 | 100.0% |

## Field availability - metadata file (30,445 rows)

| Field | Present | Share |
|---|---:|---:|
| price | 20,153 | 66.2% |
| rank | 30,058 | 98.7% |
| also_buy | 8,625 | 28.3% |
| also_view | 11,162 | 36.7% |
| brand | 29,861 | 98.1% |
| description | 28,139 | 92.4% |
| feature | 26,041 | 85.5% |

Price parse status: price present 20,153 (66.2%), parsable 20,044 (65.8%).

| Status | Records |
|---|---:|
| missing | 10,292 |
| range | 3 |
| single | 20,041 |
| unparseable | 109 |

## Review dates

- Range: **2000-10-23** to **2018-10-04** (unix 972259200 - 1538611200)
- Records without timestamp: 0; unparsable timestamps: 0

## Rating distribution (weak-supervision source: 1-2 negative / 3 neutral / 4-5 positive)

| Rating | Count | Share |
|---|---:|---:|
| 1.0 | 59,627 | 9.9% |
| 2.0 | 20,734 | 3.4% |
| 3.0 | 30,652 | 5.1% |
| 4.0 | 75,476 | 12.5% |
| 5.0 | 416,288 | 69.1% |

## Reviews per product

- Mean: **19.93**; max: **6510**

| Statistic | Value |
|---|---:|
| p0 | 1.00 |
| p10 | 1.00 |
| p25 | 1.00 |
| p50 | 2.00 |
| p75 | 8.00 |
| p90 | 32.00 |
| p99 | 328.00 |
| p100 | 6,510.00 |
| mean | 19.93 |

### Share of products by review-count threshold

| Threshold | Products | Share of reviewed products |
|---|---:|---:|
| >= 20 reviews | 4,311 | 14.3% |
| >= 50 reviews | 2,069 | 6.8% |
| >= 100 reviews | 1,146 | 3.8% |

## Candidate products (>= 50 reviews)

| Metric | Value |
|---|---:|
| Candidate products | 2,069 |
| With parsable price | 1,678 (81.1%) |
| With related-product links | 1,836 (88.7%) |
| With >= 50 reviews AND price AND links (share of reviewed products) | 1,477 (4.9%) |

## Related-product link coverage

- Distinct related ASINs referenced: **64,680**
- Present in this category's product set: **12,525** (19.4%)
- Note: links pointing outside the category are expected and remain available for
  cross-category evidence checks.

## Review-text length (characters)

| Statistic | Value |
|---|---:|
| p0 | 1.00 |
| p10 | 14.00 |
| p25 | 34.00 |
| p50 | 88.00 |
| p75 | 198.00 |
| p90 | 433.00 |
| p99 | 1,648.00 |
| p100 | 29,999.00 |
| mean | 189.43 |

## Decision rubric

Thresholds from `dataset.audit` in `configs\config.yaml`.

| ID | Criterion | Observed | Threshold | Verdict |
|---|---|---:|---:|---|
| C1 | price coverage among candidate products | 81.1% | >= 70.0% | **PASS** |
| C2 | products with >= 50 reviews | 2,069 | >= 500 | **PASS** |
| C3 | candidate products with related-product links | 88.7% | >= 60.0% | **PASS** |

**Result: 3/3 passed -> KEEP 'Appliances' - all 3 rubric criteria pass. No config change required.**

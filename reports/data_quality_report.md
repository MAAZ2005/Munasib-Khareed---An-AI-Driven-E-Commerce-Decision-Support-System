# Data Quality Report - Appliances

- **Generated:** 2026-10-17T11:11:15+00:00
- **Category:** `Appliances`
- **Config:** `configs/config.yaml` (`dataset.audit.candidate_min_reviews` = 50, `dataset.split.seed` = 42)
- **Reviews file:** `data\raw\Appliances.json.gz` (68,569,830 bytes)
- **Metadata file:** `data\raw\meta_Appliances.json.gz` (60,207,582 bytes)

All values below were computed by `scripts/build_dataset.py` from the raw files.

## Row counts before/after each cleaning step

### Reviews

| Step | Rows |
|---|---:|
| Streamed from raw file | 602,777 |
| After candidate-product filter | 436,862 |
| After duplicate removal | 425,993 |
| Written to reviews.parquet | 425,993 |

### Products

| Step | Rows |
|---|---:|
| Metadata rows streamed | 30,445 |
| Candidate rows collected (before ASIN de-duplication) | 2,104 |
| After ASIN de-duplication | 2,066 |
| Written to products.parquet | 2,066 |

## Field presence before/after cleaning

Measured on the rows written to parquet: 'before' = raw source value present, 'after' = value present after cleaning.

### Reviews

| Field | Before (raw present) | After (clean present) | Missing after |
|---|---:|---:|---:|
| review_text | 425,771 (99.9%) | 425,753 (99.9%) | 240 |
| summary | 425,902 (100.0%) | 425,895 (100.0%) | 98 |

### Products

| Field | Before (raw present) | After (clean present) | Missing after |
|---|---:|---:|---:|
| title | 2,066 (100.0%) | 2,066 (100.0%) | 0 |
| brand | 2,058 (99.6%) | 2,058 (99.6%) | 8 |
| description | 1,932 (93.5%) | 1,932 (93.5%) | 134 |
| features | 1,968 (95.3%) | 1,968 (95.3%) | 98 |
| rank | 2,053 (99.4%) | 2,053 (99.4%) | 13 |
| price | 1,684 (81.5%) | 1,678 (81.2%) | 388 |

### Price parse status (products)

| Status | Products |
|---|---:|
| missing | 382 |
| range | 2 |
| single | 1,676 |
| unparseable | 6 |

## Missingness per column

Null counts (None/NaT/pd.NA); empty strings are not nulls.

### products.parquet (2,066 rows)

| Column | Null | Share |
|---|---:|---:|
| asin | 0 | 0.0% |
| split | 0 | 0.0% |
| title | 0 | 0.0% |
| title_clean | 0 | 0.0% |
| brand | 0 | 0.0% |
| brand_clean | 8 | 0.4% |
| brand_missing | 0 | 0.0% |
| price | 388 | 18.8% |
| price_max | 2,064 | 99.9% |
| price_status | 0 | 0.0% |
| price_missing | 0 | 0.0% |
| rank | 13 | 0.6% |
| rank_value | 13 | 0.6% |
| rank_category | 13 | 0.6% |
| rank_missing | 0 | 0.0% |
| description_clean | 134 | 6.5% |
| features_clean | 98 | 4.7% |
| n_reviews | 0 | 0.0% |
| mean_rating | 0 | 0.0% |
| first_review_time | 0 | 0.0% |
| last_review_time | 0 | 0.0% |
| image_url | 993 | 48.1% |
| image_missing | 0 | 0.0% |

### reviews.parquet (425,993 rows)

| Column | Null | Share |
|---|---:|---:|
| review_id | 0 | 0.0% |
| asin | 0 | 0.0% |
| split | 0 | 0.0% |
| reviewer_id | 0 | 0.0% |
| rating | 0 | 0.0% |
| review_text_raw | 222 | 0.1% |
| review_text_clean | 240 | 0.1% |
| summary_raw | 91 | 0.0% |
| summary_clean | 98 | 0.0% |
| has_text | 0 | 0.0% |
| text_length | 0 | 0.0% |
| helpful_votes | 0 | 0.0% |
| verified | 0 | 0.0% |
| review_datetime | 0 | 0.0% |
| review_year | 0 | 0.0% |
| review_month | 0 | 0.0% |
| sentiment_label_weak | 0 | 0.0% |

### related_products.parquet (84,885 rows)

| Column | Null | Share |
|---|---:|---:|
| asin | 0 | 0.0% |
| related_asin | 0 | 0.0% |
| relation_type | 0 | 0.0% |
| in_subset | 0 | 0.0% |

## Split sizes (product-level split, no leakage)

Configured fractions train/val/test = 0.7/0.15/0.15, seed = 42; every review inherits its product's split.

| Split | Products | Share | Reviews | Share | Config fraction |
|---|---:|---:|---:|---:|---:|
| train | 1,444 | 69.9% | 303,317 | 71.2% | 70.0% |
| val | 310 | 15.0% | 62,961 | 14.8% | 15.0% |
| test | 312 | 15.1% | 59,715 | 14.0% | 15.0% |

Leakage check: every product in exactly one split = True; every review's split matches its product's split = True.

## Stratification strata (average-rating bin x review-count bin)

| Stratum | Products | train | val | test |
|---|---:|---:|---:|---:|
| rating=3.5-4.5 | reviews=150-400 | 282 | 197 | 42 | 43 |
| rating=3.5-4.5 | reviews=75-150 | 352 | 246 | 53 | 53 |
| rating=3.5-4.5 | reviews=<=75 | 280 | 196 | 42 | 42 |
| rating=3.5-4.5 | reviews=>400 | 140 | 98 | 21 | 21 |
| rating=<=3.5 | reviews=150-400 | 16 | 11 | 2 | 3 |
| rating=<=3.5 | reviews=75-150 | 45 | 31 | 7 | 7 |
| rating=<=3.5 | reviews=<=75 | 59 | 41 | 9 | 9 |
| rating=<=3.5 | reviews=>400 | 6 | 4 | 1 | 1 |
| rating=>4.5 | reviews=150-400 | 220 | 154 | 33 | 33 |
| rating=>4.5 | reviews=75-150 | 307 | 215 | 46 | 46 |
| rating=>4.5 | reviews=<=75 | 287 | 201 | 43 | 43 |
| rating=>4.5 | reviews=>400 | 72 | 50 | 11 | 11 |

## Rating distribution per split (star rating -> count (share))

| Rating | train | val | test |
|---|---:|---:|---:|
| 1 | 27,050 (8.9%) | 5,464 (8.7%) | 5,418 (9.1%) |
| 2 | 10,067 (3.3%) | 1,983 (3.1%) | 1,911 (3.2%) |
| 3 | 14,572 (4.8%) | 3,005 (4.8%) | 3,025 (5.1%) |
| 4 | 38,056 (12.5%) | 8,131 (12.9%) | 7,729 (12.9%) |
| 5 | 213,572 (70.4%) | 44,378 (70.5%) | 41,632 (69.7%) |

## Weak-label distribution per split (label -> count (share))

Weak supervision from star ratings via rating_mapping (rating <= 2 -> 0 negative, rating == 3 -> 1 neutral, rating >= 4 -> 2 positive) - not true sentiment; 'null' means the review has no rating.

| Label | train | val | test |
|---|---:|---:|---:|
| 0 | 37,117 (12.2%) | 7,447 (11.8%) | 7,329 (12.3%) |
| 1 | 14,572 (4.8%) | 3,005 (4.8%) | 3,025 (5.1%) |
| 2 | 251,628 (83.0%) | 52,509 (83.4%) | 49,361 (82.7%) |

## Reviews per product (distribution over written products)

| Statistic | Value |
|---|---:|
| min | 10 |
| p10 | 56.0 |
| p25 | 69.0 |
| p50 | 111.0 |
| p75 | 215.8 |
| p90 | 427.5 |
| p99 | 1484.1 |
| max | 6510 |
| mean | 206.2 |

## Review date range

| Metric | Value |
|---|---:|
| Earliest review | 2001-11-08T00:00:00+00:00 |
| Latest review | 2018-10-04T00:00:00+00:00 |

Note: Anchored on the review data itself; downstream recency features must use this max, never today's date.

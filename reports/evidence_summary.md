# Evidence summary

Computed from the stored sentiment predictions and aspect rows by `python scripts/compute_evidence.py`; every number below is produced by the code, none is illustrative.

- **Products:** 2,066
- **Reference date (recency anchor):** 2018-10-04T00:00:00+00:00 - the latest review timestamp in `data/processed/meta.json`, never today's date.
- **Computed at:** 2026-10-24T06:15:06+00:00 - the dataset build timestamp.
- **Weak supervision:** sentiment labels come from the stored predictor, whose training targets are star-derived weak labels; the model was trained on the train-split products, so train predictions are in-sample (see `docs/evidence.md`).

## Distributions

| Indicator | n | min | median | max |
|---|---:|---:|---:|---:|
| `evidence_strength` | 2,066 | 0.361589 | 0.680937 | 0.900961 |
| `review_volume` | 2,066 | 0.452151 | 0.889728 | 1.000000 |
| `rating_agreement` | 2,066 | 0.013974 | 0.454716 | 0.958248 |
| `sentiment_shares` | 2,066 | 0.128588 | 0.872739 | 0.994292 |
| `helpfulness_coverage` | 2,066 | 0.000000 | 0.062100 | 1.000000 |
| `helpfulness_weighted_sentiment` | 2,066 | 0.100282 | 0.842475 | 0.994330 |
| `verified_purchase_ratio` | 2,066 | 0.000000 | 0.969994 | 1.000000 |
| `recency_coverage` | 2,066 | 0.000000 | 0.352941 | 1.000000 |
| `staleness` | 2,066 | 0.171295 | 0.766925 | 1.000000 |
| `recent_negative_shift` | 1,845 | 0.521212 | 0.991241 | 1.000000 |
| `sentiment_trend` | 1,145 | 0.000000 | 0.476250 | 0.937996 |
| `aspect_concentration` | 1,151 | 0.000000 | 0.880000 | 0.991188 |
| `top_pain_point_share` | 1,151 | 0.000000 | 0.833333 | 1.000000 |
| `temporal_burst` | 1,171 | 0.000000 | 0.666667 | 0.886667 |
| `rating_text_agreement` | 2,066 | 0.117647 | 0.951456 | 1.000000 |

## Flags

| Flag | Products | Share |
|---|---:|---:|
| `polarized` | 193 | 9.34% |
| `low_evidence` | 18 | 0.87% |
| `no_recent_reviews` | 34 | 1.65% |
| `temporal_burst` | 177 | 8.57% |

## Per split

| Split | Products | evidence_strength median |
|---|---:|---:|
| train | 1,444 | 0.683420 |
| val | 310 | 0.680534 |
| test | 312 | 0.675972 |

## Missing values

| Indicator | Products | Reasons |
|---|---:|---|
| `recent_negative_shift` | 221 | `recent_window_below_min_reviews`=221 |
| `sentiment_trend` | 921 | `trend_insufficient_quarters`=921 |
| `aspect_concentration` | 915 | `below_min_negative_reviews`=895, `no_negative_mentions`=20 |
| `top_pain_point_share` | 915 | `below_min_negative_reviews`=895, `no_negative_mentions`=20 |
| `temporal_burst` | 895 | `below_min_negative_reviews`=895 |

## Correlation with evidence_strength (Spearman)

| Signal | Spearman | n |
|---|---:|---:|
| `review_volume` | +0.2722 | 2,066 |
| `recency_coverage` | +0.7959 | 2,066 |
| `verified_purchase_ratio` | +0.4329 | 2,066 |
| `rating_agreement` | +0.5642 | 2,066 |
| `rating_text_consistency` | +0.2599 | 2,066 |
| `helpfulness_coverage` | -0.3607 | 2,066 |
| `n_reviews` | +0.2703 | 2,066 |

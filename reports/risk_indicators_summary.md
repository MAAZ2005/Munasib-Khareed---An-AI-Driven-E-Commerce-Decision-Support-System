# Risk indicators summary

Computed from the stored cache by `python scripts/compute_risk_indicators.py`; every number below is produced by the code, none is illustrative.

- **Products written:** 2,066
- **Reference date (recency anchor):** 2018-10-04T00:00:00+00:00 - the latest review timestamp in `data/processed/meta.json`, never today's date.
- **Computed at:** 2026-10-24T06:15:06+00:00 - the dataset build timestamp, never the wall clock.

## Category priors (train split)

| Prior | Value |
|---|---:|
| `split` | train |
| `n_train_products` | 1,444 |
| `n_train_with_rating` | 1,444 |
| `n_train_with_predictions` | 1,444 |
| `mean_rating` | 4.3039 |
| `net_sentiment` | 0.6822 |
| `negative_share` | 0.1239 |

## Agreement scale

- **Chosen scale:** 0.2
- **Share of products saturated at 0:** 12.39% over 2,066 products
- **Median component:** 0.7647
- **Median gap:** 0.0471
- **Kept the saturation share under the cap:** true

| Candidate scale | Share at 0 |
|---:|---:|
| 0.1 | 31.70% |
| 0.2 | 12.39% |
| 0.3 | 5.28% |

## Criterion values

| Criterion | n | min | median | max |
|---|---:|---:|---:|---:|
| `product_quality` | 2,066 | 0.1732 | 0.8803 | 0.9743 |
| `evidence_strength` | 2,066 | 0.3616 | 0.6809 | 0.9010 |
| `rating_reliability` | 2,066 | 0.2076 | 0.7254 | 0.9665 |
| `negative_feedback_concentration` | 2,066 | 0.0120 | 0.1343 | 0.6601 |
| `price_position` | 1,678 | 0.0000 | 0.4945 | 1.0000 |

## Cost-direction inputs (raw and normalized)

| Indicator | Input | n | min | median | max |
|---|---|---:|---:|---:|---:|
| `negative_feedback_concentration` | value (cost) | 2,066 | 0.0120 | 0.1343 | 0.6601 |
| `negative_feedback_concentration` | normalized (1 - value) | 2,066 | 0.3399 | 0.8657 | 0.9880 |
| `negative_feedback_concentration` | prevalence | 2,066 | 0.0239 | 0.2409 | 1.0000 |
| `negative_feedback_concentration` | focus | 2,066 | 0.0000 | 0.1291 | 0.8667 |
| `price_position` | value (cost percentile) | 1,678 | 0.0000 | 0.4945 | 1.0000 |
| `price_position` | normalized (1 - value) | 1,678 | 0.0000 | 0.5055 | 1.0000 |
| `top_pain_point_share` | raw | 1,151 | 0.0000 | 0.1667 | 1.0000 |
| `top_pain_point_share` | normalized (1 - raw) | 1,151 | 0.0000 | 0.8333 | 1.0000 |
| `aspect_concentration` | raw (HHI) | 1,151 | 0.0088 | 0.1200 | 1.0000 |
| `aspect_concentration` | normalized (1 - raw) | 1,151 | 0.0000 | 0.8800 | 0.9912 |
| `temporal_burst` | raw | 1,171 | 0.1133 | 0.3333 | 1.0000 |
| `temporal_burst` | normalized (1 - raw) | 1,171 | 0.0000 | 0.6667 | 0.8867 |

## Missing values (None, never imputed)

| Criterion | Products | Reasons |
|---|---:|---|
| `product_quality` | 0 | - |
| `evidence_strength` | 0 | - |
| `rating_reliability` | 0 | - |
| `negative_feedback_concentration` | 0 | - |
| `price_position` | 388 | `price_missing`=388 |

## Spearman correlations (all products, pairwise complete)

Key: pq=product_quality, es=evidence_strength, rr=rating_reliability, nfc=negative_feedback_concentration, pp=price_position
| | `pq` | `es` | `rr` | `nfc` | `pp` |
|---|---:|---:|---:|---:|---:|
| `pq` | 1.000 | 0.604 | 0.842 | -0.931 | 0.003 |
| `es` | 0.604 | 1.000 | 0.478 | -0.434 | -0.006 |
| `rr` | 0.842 | 0.478 | 1.000 | -0.810 | -0.032 |
| `nfc` | -0.931 | -0.434 | -0.810 | 1.000 | -0.004 |
| `pp` | 0.003 | -0.006 | -0.032 | -0.004 | 1.000 |

## Spearman correlations (train split only, pairwise complete)

Key: pq=product_quality, es=evidence_strength, rr=rating_reliability, nfc=negative_feedback_concentration, pp=price_position
| | `pq` | `es` | `rr` | `nfc` | `pp` |
|---|---:|---:|---:|---:|---:|
| `pq` | 1.000 | 0.606 | 0.837 | -0.930 | 0.036 |
| `es` | 0.606 | 1.000 | 0.477 | -0.434 | 0.011 |
| `rr` | 0.837 | 0.477 | 1.000 | -0.805 | 0.002 |
| `nfc` | -0.930 | -0.434 | -0.805 | 1.000 | -0.033 |
| `pp` | 0.036 | 0.011 | 0.002 | -0.033 | 1.000 |

## Median criterion value by split

| Split | `pq` | `es` | `rr` | `nfc` | `pp` |
|---|---:|---:|---:|---:|---:|
| test | 0.8769 | 0.6760 | 0.7117 | 0.1397 | 0.4265 |
| train | 0.8811 | 0.6834 | 0.7284 | 0.1346 | 0.5263 |
| val | 0.8802 | 0.6805 | 0.7245 | 0.1295 | 0.4308 |

## Risk factors

| Factor | Products | Share | low | medium | high |
|---|---:|---:|---:|---:|---:|
| `recent_sentiment_decline` | 118 | 5.71% | 93 | 21 | 4 |
| `polarized_ratings` | 193 | 9.34% | 37 | 116 | 40 |
| `low_verified_share` | 60 | 2.90% | 20 | 11 | 29 |
| `weak_evidence` | 18 | 0.87% | 18 | 0 | 0 |
| `stale_reviews` | 34 | 1.65% | 24 | 8 | 2 |
| `price_above_peers_with_low_quality` | 0 | 0.00% | 0 | 0 | 0 |
| `concentrated_complaints` | 142 | 6.87% | 73 | 43 | 26 |
| `negative_burst` | 177 | 8.57% | 140 | 25 | 12 |
| `rating_text_conflict` | 2 | 0.10% | 1 | 1 | 0 |

## Notes

- criteria 'product_quality' and 'rating_reliability' correlate at rho=+0.8418 (>|0.8|), so they are not independent signals
- criteria 'negative_feedback_concentration' and 'product_quality' correlate at rho=-0.9310 (>|0.8|), so they are not independent signals
- criteria 'negative_feedback_concentration' and 'rating_reliability' correlate at rho=-0.8096 (>|0.8|), so they are not independent signals
- factor 'price_above_peers_with_low_quality' triggers for 0.00% of products (<0.5%), so it is close to dead
- factor 'rating_text_conflict' triggers for 0.10% of products (<0.5%), so it is close to dead

## Best 5 by product_quality

| ASIN | product_quality | mean_rating | n_reviews | net_sentiment | Title |
|---|---:|---:|---:|---:|---|
| B00IVO8478 | 0.9743 | 4.9218 | 179 | 0.9753 | 80040 - Original Factory OEM Washer Agitator Dog (4 Pack) Roper Estate |
| B00DM8J11Q | 0.9741 | 4.8687 | 1,417 | 0.9469 | Supco LP338 Agitator Dogs For Whirlpool 285770, 80040, AP3119063, 285612, 3366877, 387091 |
| B01FZJXITU | 0.9700 | 4.8836 | 378 | 0.9340 | Replacement Part for 279834 Dryer Gas Coil Kit - Valve Ignition Solenoid For Dryers Premium Parts |
| B0053F80JA | 0.9687 | 4.8389 | 1,366 | 0.9358 | Whirlpool 3406107 Door Switch for Dryer |
| B006H7HB7K | 0.9655 | 4.8517 | 526 | 0.9188 | Supco SET401 Dryer Thermal Fuse Replaces Whirlpool 3392519, 33886651, GE WE4X857 |

## Worst 5 by product_quality

| ASIN | product_quality | mean_rating | n_reviews | net_sentiment | Title |
|---|---:|---:|---:|---:|---|
| B005KT4LK6 | 0.1732 | 1.2303 | 152 | -0.7313 | Whirlpool WTW8800YW Cabrio 4.6 Cu. Ft. White Top Load Washer - Energy Star |
| B003KHEABK | 0.2530 | 1.2676 | 71 | -0.7428 | Maytag Performance Series MHWE450WW 27 4.5 cu. Ft. Front-Load Washer - White |
| B003BSKI6O | 0.2640 | 1.3188 | 69 | -0.7353 | Maytag Performance Series MHWE200XW 27 4 cu. Ft. Front-Load Washer - White |
| B0015YQV8K | 0.2801 | 1.3611 | 72 | -0.6243 |  Electrolux EWDW6505GS Wave-Touch 24&quot; Stainless Steel Fully Integrated Dishwasher - Energy Star |
| B002ACHJF0 | 0.2825 | 1.3182 | 66 | -0.6461 | Maytag Centennial Series MVWC6ESWW 27 4 cu. Ft. Top-Load Washer -White |

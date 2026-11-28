# Risk Indicators (P15)

Per-product decision criteria and rule-based risk factors. This stage turns the
stored outputs of `product_intel`, `review_intel` and the evidence stage into the
five 0-1 criteria of `configs/criteria_weights.yaml`, plus a set of explicit
warnings ("risk factors") that explain *why* a product is risky.

It reads stored tables only: no model runs, no network and no wall clock. Every
number below is produced by `scripts/compute_risk_indicators.py` over the current
cache (2,066 products, reference date 2018-10-04, computed at 2026-10-24
06:15:06, i.e. the dataset build timestamp); none is illustrative.

---

## 1. Where it sits

```
products + review_features + product_features + product_evidence
        |
        v
src/munasib/risk/indicators.py   pure criterion functions
src/munasib/risk/factors.py      nine rule-based risk factors
src/munasib/risk/compute.py      assembly, priors, agreement-scale rule, summary
src/munasib/risk/store.py        cache loaders and the upsert
        |
        v
product_risk_indicators   (one row per ASIN)
        |
        v
fuzzy engine -> TOPSIS -> verdict -> API -> dashboard
```

| File | Responsibility |
|---|---|
| `src/munasib/risk/indicators.py` | `Criterion` payload, shrinkage, the five criterion functions, the fixed-order vector, the direction-map loader, train-split priors, per-product sentiment totals. |
| `src/munasib/risk/factors.py` | `RiskFactor` payload and the nine rules with banded severity. |
| `src/munasib/risk/compute.py` | `compute_risk_indicators`, the agreement-scale rule, `summarize` and the printed report. |
| `src/munasib/risk/store.py` | Loaders (products, predictions, evidence, features), reference date/build timestamp reuse, `write_risk_indicators`. |
| `scripts/compute_risk_indicators.py` | CLI: target selection, run, summary, exit codes. |
| `src/munasib/db/models.py` | `ProductRiskIndicator` table. |
| `src/munasib/db/repositories.py` | `ProductRiskIndicatorRepository` (upsert/list/count). |

## 2. Inputs and hard rules

| Input | Columns used |
|---|---|
| `products` | `asin`, `split`, `title`, `mean_rating`, `n_reviews` |
| `reviews` + `review_features` | `asin`, `predicted_label`, `prob_negative`, `prob_positive` |
| `product_evidence` | `evidence_strength`, `indicators` (JSON) |
| `product_features` | `indicators` (JSON: `price_percentile_category`, `price_peer_group`, `price_peer_count`, `price_peer_fallback`, `price_missing`) |

* **No inference.** Sentiment comes from the stored weak-label-trained
  predictions, never a fresh model run.
* **No clock.** `reference_date` is `review_date_range.max` from
  `data/processed/meta.json`; `computed_at` is its `built_at`. Both are stored
  naive UTC. The summary never prints "now".
* **No imputation.** A missing value stays `None`; the component is dropped and
  the remaining weights are renormalized. Nothing is filled with 0 or 1.
* **SQLite JSON decoding.** SQLAlchemy stores JSON columns as TEXT in SQLite and
  `pandas.read_sql` returns that text, so `risk/store.py` decodes every payload
  before use (the ORM path decodes automatically).

## 3. Category priors (train split only)

Each prior is a **product-macro mean**: one value per TRAIN product, so a product
with thousands of reviews cannot dominate. Validation and test products never
influence a prior. Priors are computed over the **full** product frame even when
`--limit`/`--asins` writes a subset, so a partial run cannot change a stored
value.

Measured priors (stored in every row's `priors` JSON):

| Prior | Value | Basis |
|---|---|---|
| `mean_rating` | 4.303889 | 1,444 train products, all rated |
| `net_sentiment` | +0.682193 | 1,444 train products with predictions |
| `negative_share` | 0.123858 | 1,444 train products with predictions |

`shrinkage_m = 20` pseudo-reviews:

```
shrunk = (n * value + m * prior) / (n + m)
```

A product with 3 reviews barely moves the prior; one with 300 is almost its own
mean.

## 4. The five criteria

All criteria are 0-1. Directions come from `configs/criteria_weights.yaml` and
are checked against the implementation (`read_criteria_directions` fails loudly
if the file ever flips one):

| Criterion | Direction | Meaning |
|---|---|---|
| `product_quality` | benefit | higher = better product |
| `evidence_strength` | benefit | higher = more reliable evidence |
| `rating_reliability` | benefit | higher = ratings and text agree |
| `negative_feedback_concentration` | cost | higher = more concentrated bad feedback |
| `price_position` | cost | higher = more expensive than peers |

### 4.1 `product_quality`

Weighted mean of three components (`risk.criteria.product_quality.weights`):

| Component | Weight | Formula |
|---|---|---|
| `shrunk_rating` | 0.50 | `((shrunk(mean_rating) - 1) / 4)` |
| `shrunk_sentiment` | 0.30 | `(shrunk(mean_net) + 1) / 2` |
| `recent_sentiment` | 0.20 | stored `recent_negative_shift.normalized` (1 = no worsening shift) |

`mean_net` is the mean of `prob_positive - prob_negative` over the product's
predicted reviews; both the rating and the sentiment are shrunk toward the
train priors with `m = 20`.

Measured distribution: min 0.1732, median 0.8803, max 0.9743 (n = 2,066).

### 4.2 `evidence_strength`

The stored composite `product_evidence.evidence_strength`, unchanged. Missing
when the product has no evidence row or the composite is NULL; the stored
`low_evidence` flag travels in `extras`. Measured: min 0.3616, median 0.6809,
max 0.9010 (n = 2,066).

### 4.3 `rating_reliability`

| Component | Weight | Source |
|---|---|---|
| `rating_agreement` | 0.35 | stored normalized `rating_agreement` |
| `non_polarization` | 0.20 | 1.0 when not polarized, 0.0 when the stored flag is set |
| `rating_text_consistency` | 0.25 | stored normalized `rating_text_agreement` |
| `helpfulness_agreement` | 0.20 | `1 - min(1, gap / agreement_scale)` |

`gap = |helpfulness_weighted_net - unweighted_net|` from the stored
`helpfulness_weighted_sentiment` extras.

`non_polarization` is available **only when `rating_agreement` is**. An unrated
product has no polarization signal, and a stored `False` would fabricate 1.0,
so it is recorded as missing (`no_ratings`) instead.

Measured: min 0.2076, median 0.7254, max 0.9665 (n = 2,066).

#### The `agreement_scale` rule (declared, not fitted)

`agreement_scale` is the gap at which the helpfulness agreement component
saturates at exactly 0. It is a **declared setting chosen to avoid saturation**,
not fitted to any ground truth. The candidates are 0.10, 0.20 and 0.30; the
config declares 0.10 and a 25% cap on products sitting at exactly 0. The run
computes the gap distribution over every product with a usable gap (the whole
evidence table, so a `--limit` run cannot store a different scale) and picks the
**smallest** candidate whose zero share is below the cap:

| Candidate scale | Share of products with the component exactly 0 |
|---|---|
| 0.10 | 31.7% - above the 25% cap, rejected |
| **0.20** | **12.4% - chosen** |
| 0.30 | 5.3% |

Median component at the chosen scale: 0.7647 over 2,066 products. The chosen
scale is stored in each `rating_reliability` payload (`extras.agreement_scale`)
and printed with the per-candidate shares, so the row is self-describing. The
scale was **not** tuned against risk labels, verdicts or TOPSIS output.

### 4.4 `negative_feedback_concentration` (cost)

```
shrunk_negative = shrink(negative_share, n_predictions, prior_negative_share, m)
prevalence      = min(1, shrunk_negative / p_saturation)          # p_saturation = 0.40
focus           = mean of the available raw concentration inputs
value           = prevalence * (focus_floor + (1 - focus_floor) * focus)   # focus_floor = 0.50
```

The three focus inputs are the **raw** stored values of `top_pain_point_share`
(share of negative mentions covered by the top pain points),
`aspect_concentration` (Herfindahl index) and `temporal_burst` (largest
sliding-window share of negative reviews). Diffuse negative feedback still counts
the `focus_floor` amount. When all three are missing (fewer than 10 predicted
negative reviews), `focus = 0` and `focus_unknown = true` is recorded; the value
is still reported because prevalence alone is real.

Measured (raw and normalized are deliberately kept apart; the evidence stage
stores `normalized = 1 - raw` for these cost inputs):

| Quantity | Min | Median | Max | n |
|---|---|---|---|---|
| `negative_feedback_concentration` value (cost) | 0.0120 | 0.1343 | 0.6601 | 2,066 |
| `negative_feedback_concentration` normalized (1 - value) | 0.3399 | 0.8657 | 0.9880 | 2,066 |
| prevalence | 0.0239 | 0.2409 | 1.0000 | 2,066 |
| focus | 0.0000 | 0.1291 | 0.8667 | 2,066 |
| `top_pain_point_share` raw | 0.0000 | 0.1667 | 1.0000 | 1,151 |
| `top_pain_point_share` normalized (1 - raw) | 0.0000 | 0.8333 | 1.0000 | 1,151 |
| `aspect_concentration` raw (HHI) | 0.0088 | 0.1200 | 1.0000 | 1,151 |
| `aspect_concentration` normalized (1 - raw) | 0.0000 | 0.8800 | 0.9912 | 1,151 |
| `temporal_burst` raw | 0.1133 | 0.3333 | 1.0000 | 1,171 |
| `temporal_burst` normalized (1 - raw) | 0.0000 | 0.6667 | 0.8867 | 1,171 |

### 4.5 `price_position` (cost)

The stored `price_percentile_category` (0 = cheapest, 1 = priciest in the
category), unchanged. Missing prices stay missing: 388 of 2,066 products (18.8%)
have no usable price and are **not** imputed. When the stored
`price_peer_fallback` is true (a shorter breadcrumb prefix or the whole candidate
set was used), the confidence is multiplied by
`risk.criteria.price_position.fallback_confidence = 0.50`; the value itself is
still the percentile.

Measured: value min 0.0000, median 0.4945, max 1.0000 (n = 1,678); normalized
(1 - value) min 0.0000, median 0.5055, max 1.0000.

## 5. Missing data and confidence

For every weighted criterion:

* available components are used and their weights renormalized to sum to 1;
* `components_used`, `weights` (renormalized), `missing_components` (component ->
  reason) are stored;
* `confidence = available weight / total weight` (0-1). For single-source
  criteria (`evidence_strength`, `price_position`,
  `negative_feedback_concentration`) confidence is 1 when the value exists
  (times the price fallback multiplier for `price_position`) and 0 when it does
  not;
* a criterion is `None` only when no component is available; `missing_reason`
  then records why (`all_components_missing`, `no_predictions`, `price_missing`,
  `no_evidence_row`).

Measured missing counts: `price_position` 388 (`price_missing`), every other
criterion 0.

## 6. Fixed-order vector and directions

`criteria_vector` returns exactly:

```
(product_quality, evidence_strength, rating_reliability,
 negative_feedback_concentration, price_position)
```

with `None` preserved for a missing criterion (never 0). The stored row contains
the vector and the directions map. The directions are read from
`configs/criteria_weights.yaml` and compared with the implementation's expected
map; a mismatch raises `DataError` and aborts the run.

## 7. Risk factors

Nine rules, each emitted only when its trigger is met. Severity is banded: the
rule fires at its first threshold (`low`) and escalates at the next two
(`medium`, `high`). All thresholds live in `risk.factors`.

| Code | Trigger | Bands (low -> medium -> high) | Measured |
|---|---|---|---|
| `recent_sentiment_decline` | recent minus overall predicted-negative share, with >= 10 recent predictions | delta >= 0.10 -> 0.20 -> 0.35 | 118 (5.71%): 93 / 21 / 4 |
| `polarized_ratings` | stored `polarized` flag | agreement <= 0.25 -> 0.10 | 193 (9.34%): 37 / 116 / 40 |
| `low_verified_share` | verified ratio below 0.70 | <= 0.70 -> 0.55 -> 0.40 | 60 (2.90%): 20 / 11 / 29 |
| `weak_evidence` | composite below 0.40, or the low-evidence flag | < 0.40 -> 0.30 -> 0.20; NULL = high | 18 (0.87%): 18 / 0 / 0 |
| `stale_reviews` | newest review older than 24 months | > 24 -> 36 -> 48 months | 34 (1.65%): 24 / 8 / 2 |
| `price_above_peers_with_low_quality` | percentile > 0.75 **and** product quality < 0.40 | > 0.75 -> 0.85 -> 0.95 | 0 (0.00%) |
| `concentrated_complaints` | top pain-point share >= 0.50 | >= 0.50 -> 0.65 -> 0.80 | 142 (6.87%): 73 / 43 / 26 |
| `negative_burst` | stored burst flag and share >= 0.50 | >= 0.50 -> 0.65 -> 0.80 | 177 (8.57%): 140 / 25 / 12 |
| `rating_text_conflict` | conflicts / evaluated >= 0.10 | >= 0.10 -> 0.15 -> 0.20 | 2 (0.10%): 1 / 1 / 0 |

Every factor carries a sentence citing the real numbers and a `supporting`
mapping, so the dashboard can explain it without recomputing anything.

**Flags.** The summary flags any factor triggering for more than 60% or less than
0.5% of products. On the current cache:

* `price_above_peers_with_low_quality` triggers for **0.00%**. All 17 products
  with `product_quality < 0.40` have no usable price (`price_position` is NULL),
  so the conjunctive price gate cannot fire for any of them. The rule and its
  thresholds are declared, not tuned to make it fire; it is reported as
  effectively dead on this cache.
* `rating_text_conflict` triggers for **0.10%**, also flagged as close to dead:
  a conflict needs a review whose text is confidently negative
  (`conflict_probability_threshold = 0.5` in `review_intel.evidence.indicators`)
  while its star rating is high, which is rare in this category.

Neither threshold was changed after seeing these counts.

## 8. Printed summary

`scripts/compute_risk_indicators.py` prints, and the unit tests assert:

1. products written, train-split priors, and the agreement-scale decision with
   the zero share, median component and every candidate's zero share;
2. min/median/max (n) of every criterion;
3. missing counts per criterion **with the reason**;
4. for every cost-direction indicator - the concentration value **and** its
   `1 - value` normalization, prevalence/focus, `top_pain_point_share`,
   `aspect_concentration`, `temporal_burst` and `price_position` - **both** the
   raw and the normalized min/median/max, each labelled (the three evidence
   inputs' stored normalization is 1 - raw; the price position's and the
   concentration's displayed normalization is 1 - value);
5. the 5x5 Spearman matrix over all written products **and** again over the
   train split only, pairwise-complete (each pair uses only rows where both
   values exist);
6. median criterion value by split;
7. per factor: trigger count, share of products, and low/medium/high counts;
8. notes: criterion pairs with |rho| > 0.8 and factors above 60% / below 0.5%;
9. the five best and five worst products by `product_quality` (title, mean
   rating, review count, mean net sentiment, value).

Elapsed time is printed but never stored.

### Report files

The same figures are written, from computed values only, to
`reports/risk_indicators_summary.json` and `reports/risk_indicators_summary.md`
(under `paths.reports_dir`):

* `reports/risk_indicators_summary.json` - machine-readable: products written,
  the reference date and the dataset build timestamp, the train priors **with
  their train counts**, the agreement-scale decision and every candidate's
  saturation share, min/median/max of every criterion plus the raw **and**
  normalized cost inputs, missing counts with reasons, both 5x5 Spearman
  matrices, the per-split medians, the factor counts by severity, the notes, and
  the five best and five worst products by `product_quality`.
* `reports/risk_indicators_summary.md` - the same figures rendered for reading.

The only timestamps in either file are the dataset's own (`review_date_range.max`
and `built_at` from `data/processed/meta.json`); no wall-clock value and no
elapsed time is ever written, so two runs over the same cache produce
byte-identical files.

### Measured correlations

All products (pairwise complete):

| | pq | es | rr | nfc | pp |
|---|---|---|---|---|---|
| pq | 1.000 | 0.604 | 0.842 | -0.931 | 0.003 |
| es | 0.604 | 1.000 | 0.478 | -0.434 | -0.006 |
| rr | 0.842 | 0.478 | 1.000 | -0.810 | -0.032 |
| nfc | -0.931 | -0.434 | -0.810 | 1.000 | -0.004 |
| pp | 0.003 | -0.006 | -0.032 | -0.004 | 1.000 |

Train split only:

| | pq | es | rr | nfc | pp |
|---|---|---|---|---|---|
| pq | 1.000 | 0.606 | 0.837 | -0.930 | 0.036 |
| es | 0.606 | 1.000 | 0.477 | -0.434 | 0.011 |
| rr | 0.837 | 0.477 | 1.000 | -0.805 | 0.002 |
| nfc | -0.930 | -0.434 | -0.805 | 1.000 | -0.033 |
| pp | 0.036 | 0.011 | 0.002 | -0.033 | 1.000 |

`product_quality` and `negative_feedback_concentration` are near mirror images
(rho -0.931) because net sentiment drives both, with opposite signs; the summary
flags that pair (and pq-rr, nfc-rr) as not independent. This is a data property,
reported rather than hidden; TOPSIS still receives the five criteria as declared.

### Median criterion value by split

| Split | pq | es | rr | nfc | pp |
|---|---|---|---|---|---|
| train | 0.8811 | 0.6834 | 0.7284 | 0.1346 | 0.5263 |
| val | 0.8802 | 0.6805 | 0.7245 | 0.1295 | 0.4308 |
| test | 0.8769 | 0.6760 | 0.7117 | 0.1397 | 0.4265 |

## 9. Storage

`product_risk_indicators` (one row per ASIN):

| Column | Content |
|---|---|
| `asin` | PK, FK to `products.asin` |
| `indicators` | `{split, criteria: {name: payload}, vector, directions}` |
| `risk_factors` | list of `{code, severity, description, supporting}` |
| `priors` | the train-split priors and their counts |
| `reference_date` | `review_date_range.max`, naive UTC |
| `computed_at` | `built_at`, naive UTC |

Upserts merge on `asin`, so re-runs are idempotent and a `--limit` run refreshes
only its own products. `scripts/ingest_db.py` treats the table as derived: a
`--force` rebuild or cache reload warns with the stored row count and prints the
exact recompute order:

```
python scripts/ingest_predictions.py
python scripts/compute_aspects.py --reviews
python scripts/compute_aspects.py --aggregate
python scripts/compute_product_features.py
python scripts/compute_evidence.py
python scripts/compute_risk_indicators.py
```

## 10. CLI

```
python scripts/compute_risk_indicators.py
python scripts/compute_risk_indicators.py --limit 100
python scripts/compute_risk_indicators.py --asins B000050JN4 B00004T8A4
python scripts/compute_risk_indicators.py --config configs/config.yaml
```

Exit codes: `0` rows written; `1` the run could not start (bad config, empty
cache, unreadable `meta.json`, a direction mismatch, or an unknown `--asins`
entry).

Every successful run also writes `reports/risk_indicators_summary.json` and
`reports/risk_indicators_summary.md` into `paths.reports_dir` and prints the
two paths; the files are regenerated on each run and are byte-identical for the
same cache.

## 11. Limitations

* Criteria are correlated (see above); the vector is the declared TOPSIS input,
  not an orthogonal factor model.
* Sentiment labels are weak supervision from star ratings (1-2 negative, 3
  neutral, 4-5 positive), so every sentiment-derived component inherits that
  noise.
* Priors and criteria describe this one category only.
* Factor thresholds are expert-declared; two are currently near-dead and are
  reported as such rather than retuned to the observed data.

## 12. Tests

`tests/unit/test_risk_indicators.py` (hand-computed criterion values, shrinkage,
renormalization, missing gating, vector order, direction-map failure modes,
priors), `test_risk_factors.py` (every rule at its boundary and each severity
band), `test_risk_compute.py` (determinism, no wall clock, agreement-scale rule,
missing handling, summary statistics and matrices, plus the JSON payload and
Markdown report), `test_risk_store_db.py`
(JSON decoding, repository round-trip/upsert, real-cache skip test),
`test_risk_config.py` (defaults and validators) and `test_risk_cli.py` (hermetic
temp-cache runs, `--limit`/`--asins`, exit codes, and the summary reports: both
files written, byte-identical across two runs, and carrying only the dataset's
timestamps).

# Evidence: per-product indicators and composite strength

The evidence stage answers a different question from sentiment: *how much and
how reliable is the stored data behind a product's scores?* It reads the
serving cache only - the sentiment predictions already stored in
`review_features` and the aspect rows already stored in `product_aspects` - and
writes one `product_evidence` row per product. **No model runs, no network and
no wall clock**: every recency window is anchored on the dataset reference date
and the row timestamps come from the dataset build metadata, so the stage is
deterministic and re-runs are idempotent.

Every threshold, window and weight in this document is declared in
`configs/config.yaml` under `review_intel.evidence` and validated by
`munasib.core.config`; no module hardcodes a value.

## Commands

```powershell
# Full run: every candidate product
python scripts/compute_evidence.py

# Development: a slice or explicit ASINs (upsert only touches those rows)
python scripts/compute_evidence.py --limit 100
python scripts/compute_evidence.py --asins B000050JN4 B00004T8A4

# The run prints the summary and writes reports/evidence_summary.json + .md
```

Exit codes: `0` = rows written (a filter may legitimately match nothing new),
`1` = the run could not start (bad config, unreadable cache, missing
`meta.json`, or an `--asins` entry that is not in the cache).

`scripts/ingest_db.py --force` (and any cache reload) deletes the derived
tables and prints the recompute order, including
`python scripts/compute_evidence.py` after the predictions and aspects are
back. Plain `ingest_db.py` runs that find the cache current do not touch it.

## Reference date (never the wall clock)

The UCSD Amazon Review Data (2018) subset spans 2001-11-08 .. 2018-10-04, so
"recent" must be relative to the data, not to today. `read_reference_date`
takes `review_date_range.max` from `data/processed/meta.json` (2018-10-04 for
the shipped build) and every recency window starts there:

* `recency_coverage` counts reviews in `[reference_date - recent_months,
  reference_date]`;
* `staleness` measures the distance from the product's latest review to the
  reference date;
* `recent_negative_shift` compares the recent window with the product's whole
  history;
* `sentiment_trend` uses the `trend_quarters` **complete** calendar quarters
  before the reference date's quarter.

Consequently many products legitimately have no recent reviews: the dataset
ends in October 2018 and product histories do not. A missing recent signal is
reported as missing (`recent_window_below_min_reviews`), never imputed.

`computed_at` is `built_at` from the same `meta.json`; nothing in a stored row
or a report changes when a run is repeated.

## Inputs and missing-data rules

| Input | Source | Missing handling |
| --- | --- | --- |
| star ratings, helpful votes, verified flag, review times | `reviews` | `rating` NULL -> excluded from star statistics; `helpful_votes` NULL -> treated as 0; `verified` NULL means *unknown*, not False, and is excluded from the ratio denominator |
| predictions (`predicted_label`, three probabilities) | `review_features` (LEFT JOIN) | a review without predictions counts toward every review-level indicator but not toward prediction-based ones |
| aspect counts and classes | `product_aspects` | a product without rows has no aspect evidence (`no_negative_mentions`/`below_min_negative_reviews`) |
| reference date, build timestamp | `data/processed/meta.json` | the run fails with `DataError` instead of guessing |

Every indicator is stored as `None` with a named flag when it cannot be
computed; a `None` composite component is **dropped** and the remaining
weights renormalized, so missing data can never be silently counted as zero.

## Indicators

Direction is from the consumer's point of view: `benefit` means a higher raw
value is better, `cost` means higher is worse. `normalized` is 0-1 with 1.0
always the better end; floats are rounded to 6 decimals in the stored JSON.

| Key | Raw value | Normalized | Direction | Window / threshold |
| --- | --- | --- | --- | --- |
| `review_volume` | review count | `min(1, ln(1+n) / ln(1+n_saturation))`, `n_saturation=200` | benefit | whole history |
| `rating_agreement` | `1 - H / log2(5)`, `H` = Shannon entropy (bits) of the 1-5 star histogram | same | benefit | rated reviews; `polarized` when 1-2 star share and 5 star share both reach `polarization_min_share=0.25` |
| `sentiment_shares` | mean net sentiment `prob_positive - prob_negative` over predicted reviews | `(net+1)/2` | benefit | predicted reviews; extras carry the three label shares and the star-derived negative share (weak labels) |
| `helpfulness_coverage` | share of reviews with >= 1 helpful vote | same | benefit | all reviews |
| `helpfulness_weighted_sentiment` | net sentiment weighted by `1 + helpful_votes` over predicted reviews | `(net+1)/2` | benefit | predicted reviews; extras carry the unweighted mean and the difference |
| `verified_purchase_ratio` | verified / known-flag reviews | same | benefit | reviews with a non-NULL flag; extras carry the unknown count |
| `recency_coverage` | share of reviews inside the recent window | same | benefit | `[ref - recent_months, ref]`, `recent_months=24` |
| `staleness` | months since the latest review (`days / 30.436875`, a fixed mean-month duration) | `1 / (1 + months/staleness_scale_months)`, scale 12 | cost | whole history |
| `recent_negative_shift` | recent negative share minus overall negative share (predicted) | `1 - clamp(delta, 0, 1)` (only worsening is penalised) | cost | requires `min_recent_reviews=10` predicted reviews in the window |
| `sentiment_trend` | OLS slope of quarterly mean net sentiment (per quarter) | `0.5 + 0.5 * clamp(slope / trend_saturation, -1, 1)`, saturation 0.25 | benefit | `trend_quarters=8` complete quarters before the reference quarter, `min_quarter_reviews=5` each, `min_trend_quarters=4` (at least 2 qualify for a slope) |
| `aspect_concentration` | Herfindahl index `sum((m_i/total)^2)` over the negative-review aspect mentions | `1 - HHI` | cost | requires `min_negative_reviews=10` predicted-negative reviews; a review mentioning several aspects counts in each |
| `top_pain_point_share` | share of negative mentions covered by the top `top_pain_points=3` pain-point aspects (ranked by `negative_count` desc, tie-break aspect name asc) | `1 - share` | cost | same gate; 0.0 share when no aspect is classified `pain_point` |
| `temporal_burst` | largest share of the product's predicted-negative reviews inside one **sliding** window `[t, t + burst_window_months)` over each predicted-negative review time `t` (ties: earliest start) | `1 - share` | cost | requires `min_negative_reviews=10`; flagged when share >= `burst_flag_share=0.50` |
| `rating_text_agreement` | conflict rate: 1-2 stars with `prob_positive > 0.5`, or 4-5 stars with `prob_negative > 0.5` | `1 - min(1, rate / conflict_saturation)`, saturation 0.20 | cost | reviews with both a rating and a prediction |

The sliding burst window is deliberately not a fixed calendar bucketing: a run
of negatives from March to August stays one window (a Jan-Jun/Jul-Dec split
would cut it in half).

`aspect_concentration`, `top_pain_point_share` and `temporal_burst` are
**inputs**, not composite components: they describe how negative feedback is
concentrated and are stored raw plus inverted for the later risk-criterion
mapping. They are excluded from the composite because the composite measures
*evidence reliability*, not *product quality*.

## Composite evidence strength

Six components are combined with configured weights (sum validated to 1.0
within 1e-6):

| Component | Indicator | Weight |
| --- | --- | --- |
| `review_volume` | `review_volume` | 0.20 |
| `recency_coverage` | `recency_coverage` | 0.20 |
| `verified_purchase_ratio` | `verified_purchase_ratio` | 0.15 |
| `rating_agreement` | `rating_agreement` | 0.15 |
| `rating_text_consistency` | `rating_text_agreement` (normalized) | 0.15 |
| `helpfulness_coverage` | `helpfulness_coverage` | 0.15 |

* Available components: `evidence_strength = sum(w_c * normalized_c)`.
* A `None` component is dropped and the remaining weights are renormalized
  over the components actually used; `components_used`, the renormalized
  `weights` and `missing_components` (with reasons) are stored per product.
* If no component is available at all, `evidence_strength` is NULL - which,
  for a product with no stored reviews, is exact (all indicators are
  unavailable, including volume).
* `low_evidence = strength is NULL or strength < low_evidence_threshold (0.40)
  or n_reviews < min_reviews_for_confidence (50)`. Thin evidence cannot be
  trusted even when it agrees, and a product with very weak evidence can never
  receive a clean `CONSIDER` (see `configs/verdict_thresholds.yaml`,
  `min_evidence_for_clean_consider: 0.40`).

## Storage

`product_evidence` (one row per product, `asin` primary key and FK):

| Column | Meaning |
| --- | --- |
| `n_reviews` | review rows the indicators were computed over |
| `evidence_strength` | composite 0-1, NULL only when no component was available |
| `indicators` | JSON payload (below) |
| `reference_date` | dataset reference date (`review_date_range.max`), naive UTC |
| `computed_at` | dataset build timestamp (`built_at`), naive UTC |

The JSON payload has a stable shape:

```json
{
  "split": "train",
  "indicators": {
    "<key>": {
      "raw_value": 0.0, "normalized": 0.0, "direction": "benefit",
      "description": "one sentence citing the computed numbers",
      "missing_reason": null,
      "extras": {}
    }
  },
  "flags": {"<indicator>": "<missing_reason>"},
  "polarized": false,
  "no_recent_reviews": false,
  "temporal_burst": false,
  "low_evidence": false,
  "strength": {
    "value": 0.0,
    "components_used": ["review_volume", "..."],
    "weights": {"review_volume": 0.2},
    "missing_components": {"<component>": "<reason>"}
  }
}
```

Flags (the `missing_reason` values): `no_reviews`, `no_ratings`,
`no_predictions`, `verified_unknown`, `recent_window_below_min_reviews`,
`trend_insufficient_quarters`, `below_min_negative_reviews`,
`no_negative_mentions`.

Upserts merge on `asin`, so a `--limit`/`--asins` run refreshes only its own
products and a repeated full run writes identical values.

## In-sample caveat per split

The sentiment model behind `review_features` was trained on the **train-split
products only** (~1,444 of the ~2,066 candidates). Predictions for those
products are therefore in-sample; validation/test products are out-of-sample.
That affects every prediction-based indicator
(`sentiment_shares`, `helpfulness_weighted_sentiment`,
`recent_negative_shift`, `sentiment_trend`, `temporal_burst`,
`rating_text_agreement`). The printed summary and
`reports/evidence_summary.json`/`.md` report the composite strength separately
per split so the difference stays visible; the stored rows carry their split in
the JSON payload. The star-derived weak labels remain weak supervision
throughout - they are never true sentiment.

## Relationship to the live API composite

The Stage-2 serving path (`pipeline/service.py`) still computes its small
three-indicator composite live from review credibility
(`verified_purchase` 0.50 / `helpfulness_ratio` 0.30 / `review_depth` 0.20,
`review_intel.evidence.weights` in the config). That contract is unchanged:
`product_evidence` is a **stored, broader six-component** composite and the
serving path is not switched to it in this task. When it is switched, the
differences to reconcile are the component set, the recency anchoring and the
low-evidence rule documented above.

## Reports

`python scripts/compute_evidence.py` prints and writes, from computed values
only:

* `reports/evidence_summary.json` - machine-readable: reference date and
  computed at, distributions (min/median/max/count) of the composite and every
  non-missing normalized indicator, flag shares, per-split counts with the
  strength median, products behind each missing flag with reasons, and the
  Spearman correlation of the composite with each component and with the
  review count.
* `reports/evidence_summary.md` - the same figures rendered for reading.

The elapsed time is printed to the console but deliberately never written to
the reports: it is not a data value.

## Limitations

* Sentiment labels are weak supervision from star ratings; prediction-based
  indicators inherit that noise.
* Helpfulness votes are sparse: 91.1% of this subset's reviews carry 0 votes
  (no NULLs in the shipped build); the coverage indicator exists precisely to
  expose how thin that signal is.
* `recency_coverage` can be 0.0 for a product whose history ended years before
  the dataset cutoff - that is a dataset property, not a data error.
* Aspect mentions count per review-aspect pair, so a single review can feed
  several aspects; the concentration inputs describe mention distribution,
  not review distribution.
* The composite measures evidence reliability only; it is not a quality,
  popularity or risk score.

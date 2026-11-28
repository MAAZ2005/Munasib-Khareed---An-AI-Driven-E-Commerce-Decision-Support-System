# P14 post-implementation audit (evidence features)

**Scope:** commit `881bd38374facc3823aa28155596050f329b4319` ("p14: evidence features"),
audited on 2026-11-17 against the locked P14 requirements. Verdicts are
PASS / FAIL / NOT VERIFIED; caveats are stated explicitly. **No claim of 100%
correctness is made.** No source code was modified during this audit; all
verification scripts are throwaway files under `data/interim/` (git-ignored).

---

## Check 1 — Git state — PASS

* Commit hash: `881bd38374facc3823aa28155596050f329b4319`, message `p14: evidence features`.
* `git status --short`: only `?? opencode.json` (OpenCode harness/model-routing
  config, not project work; intentionally untracked and not committed).
* `git show --stat --oneline HEAD`: 20 files changed, 5,993 insertions(+),
  12 deletions(-). New: `docs/evidence.md`, `reports/evidence_summary.{json,md}`,
  `scripts/compute_evidence.py`, `src/munasib/review_intel/evidence/{__init__,compute,indicators,store}.py`,
  5 evidence test files; modified: `configs/config.yaml`, `scripts/ingest_db.py`,
  `src/munasib/core/config.py`, `src/munasib/db/models.py`,
  `src/munasib/db/repositories.py`, `tests/unit/test_db_{models,ingest}.py`.
* `git diff HEAD^ HEAD --stat` confirms the same file set; nothing outside the
  permitted P14 scope (no API schemas, no stub service, no frontend changes).

## Check 2 — Implementation inspection — PASS

All seven artefacts were read in full or in their relevant sections:

* `indicators.py` (1,186 lines): pure functions only — no DB, no clock, no
  network; every indicator returns `raw_value`/`normalized`/`direction`/
  `description`/`extras`/`missing_reason`; floats rounded to 6 decimals.
* `compute.py` (733 lines): assembly + deterministic summary; naive-UTC guard
  on `reference_date`/`computed_at`; `products` frame validated for
  `asin`/`split`; rows ordered by ASIN.
* `store.py` (211 lines): LEFT JOIN keeps missing predictions NULL;
  `read_reference_date` fails with `DataError` (never wall clock);
  `write_evidence` re-checks the naive-UTC contract.
* `scripts/compute_evidence.py` (266 lines): `--config/--limit/--asins`,
  exit 0/1 contract, elapsed time printed but never written to reports.
* `ProductEvidence` model + `ProductEvidenceRepository`: `asin` PK/FK,
  nullable `evidence_strength`, JSON payload, two meta-sourced timestamps;
  repository offers `get_evidence`/`list_evidence`/`count_rows`/
  `upsert_evidence` (merge → idempotent, safe for `--limit`/`--asins`).
* Configuration: `configs/config.yaml` `review_intel.evidence.indicators` (15
  keys) and `review_intel.evidence.strength` (2 keys + 6 weights); validated in
  `core/config.py` (weight sum ± 1e-6, positivity, share ranges,
  `min_trend_quarters <= trend_quarters`). The live three-indicator
  `weights` block is untouched.
* `docs/evidence.md` (235 lines): formulas, windows, boundaries, missing-data
  rules, storage schema, per-split in-sample caveat, and the explicit note that
  the serving path is still on the 3-indicator composite.

## Check 3 — Composite uses exactly six components with configured weights — PASS

* Code (`composite_strength`): iterates exactly `STRENGTH_COMPONENT_KEYS` =
  `(review_volume, recency_coverage, verified_purchase_ratio, rating_agreement,
  rating_text_consistency, helpfulness_coverage)`; consumes each component's
  mapped indicator via `COMPONENT_INDICATOR` (note:
  `rating_text_consistency` ← `rating_text_agreement.normalized`).
* Weights come from config (0.20 / 0.20 / 0.15 / 0.15 / 0.15 / 0.15, sum
  validated to 1.0 within 1e-6). A `None` component is recorded in
  `missing_components` with its reason and excluded; remaining weights are
  renormalized by `total_weight = sum(weights of available)`; when nothing is
  available the value is `None` (never 0).
* Runtime check over **all 2,066 stored rows** (`data/interim/audit_evidence_check.py`):
  `weight_errors=0, value_errors=0, unexpected_components=0`; for every row the
  stored `evidence_strength` equals the stored-weights × stored-normalized
  weighted mean (tolerance 1e-5).
* **Caveat (NOT VERIFIED on real data):** no real product actually dropped a
  component (`rows_with_missing_component=0` — every product has ≥1 known
  verified flag), so renormalization is exercised only by unit tests
  (`test_composite_strength_renormalizes_around_missing_components`,
  `test_verified_unknown_drops_the_component_and_renormalizes`,
  `test_composite_strength_without_any_component_is_none`), which pass.

## Check 4 — Indicator definitions and boundaries — PASS

Verified against the committed code and the named unit tests:

| Indicator | Definition | Boundary evidence |
| --- | --- | --- |
| `recency_coverage` | share of reviews in `[ref − 24 months, ref]` | window **inclusive at both ends**: `test_recency_coverage_counts_only_the_window` (2016-10-04 start and 2018-10-04 end count; 2016-10-03 does not) |
| `staleness` | `(ref − last_review) / 30.436875` months; `1/(1+months/12)` | `test_recency_and_staleness_anchor_on_the_reference_date` (731 days → 24.0169 months) |
| `recent_negative_shift` | recent − overall predicted-negative share, only worsening penalised (`1 − clamp(delta)`); requires ≥ 10 predicted reviews in window | 9 recent predictions → missing (`test_recent_negative_shift_needs_the_minimum_recent_predictions`); hand-computed delta 0.1 → 0.9 |
| `sentiment_trend` | OLS slope of quarterly mean net sentiment over 8 **complete** quarters before the reference quarter; ≥ 4 qualifying quarters with ≥ 5 predicted reviews each | `test_sentiment_trend_excludes_the_partial_reference_quarter`; insufficient-quarters case |
| `temporal_burst` | **sliding** window `[t, t + 6 months)` per negative review time; max count / total negatives; ties → earliest start; flag ≥ 0.50; gate ≥ 10 negatives | window end **exclusive** via `np.searchsorted(..., side="left")`; tie-break via `np.argmax` over sorted times (first maximum = earliest start) |
| `aspect_concentration` | HHI `Σ(m_i/total)²` over negative-review aspect mentions (counts > 0); `normalized = 1 − HHI`; gate ≥ 10 negatives | `test_concentration_hand_computed`, below-minimum and no-mentions cases |
| `top_pain_point_share` | top-3 `pain_point` aspects by `negative_count` desc, tie-break aspect name asc; `1 − share` | `test_concentration_top_pain_point_tie_breaks_alphabetically` |
| `rating_text_agreement` | conflict = (stars ≤ 2 & `prob_positive` > 0.5) or (stars ≥ 4 & `prob_negative` > 0.5); `consistency = 1 − min(1, rate/0.20)` | 3-star reviews can never conflict (`test_rating_text_agreement_neutral_stars_never_conflict`); both conflict sides tested |

## Check 5 — Correlation values (+0.87 recency, −0.19 helpfulness) — PASS

An independent recomputation (`data/interim/audit_evidence_check.py`) of the
Spearman correlation (pandas rank correlation over the stored normalized values
and `evidence_strength`) reproduces **all seven** reported values exactly
(6 dp), including `n` per signal — the reported figures are computed correctly.

They are also *not* a formula or data-joining error:

* **recency +0.87** — a rank effect, not a weight leak. recency_coverage has by
  far the largest weighted spread of the six components (w×std = 0.20 × 0.268 =
  0.054, vs 0.027 for rating_agreement, ≤ 0.021 for the rest; 10–90% range
  0.095–0.841). The other near-constant components (review_volume median 0.89,
  verified 0.97, rating-text 0.95, helpfulness 0.06) contribute little rank
  variance, so composite ranks are dominated by recency variation. Cross
  correlations with the other components are modest (+0.05 … +0.12), so the
  correlation is genuine, not a join artifact.
* **helpfulness −0.19** — a data property. helpfulness_coverage correlates
  **negatively** with verified_purchase_ratio (ρ = −0.44), with n_reviews
  (−0.11) and with recency (−0.09): helpful votes concentrate on older
  products whose purchases are less often verified. Even among the 2,002
  products with > 0 coverage the sign persists (ρ = −0.21). Since the coverage
  values themselves were hand-verified against raw DB arithmetic for three
  products (0 mismatches, see below) and the correlation arithmetic matches an
  independent implementation, the negative sign is real signal, not a bug.
  The docs already warn the coverage signal is thin (91.1% of reviews carry 0
  votes).

Hand-check cross-reference (pre-commit, 27 figures, 0 mismatches):
B000AST3AK (6,510 reviews), B00007E7RZ (polarized), B0006GVNOA (10 reviews).

## Check 6 — B0006GVNOA: strength 0.7357 but low_evidence=True — PASS

Stored row: `n_reviews=10`, `evidence_strength=0.735701`, all six components
used with the unrenormalized config weights, `low_evidence=True`.

The rule is `low_evidence = strength is NULL or strength < 0.40 or n_reviews < 50`
(`compute.py::_indicator_payload`; config `low_evidence_threshold=0.40`,
`min_reviews_for_confidence=50`). Here strength (0.7357) passes the threshold
but `n_reviews=10 < 50`, so the review-count guard fires. This is exactly the
locked rule ("thin evidence cannot be trusted even when it agrees") and the
doc's stated behaviour; the 0.40 threshold is aligned with
`configs/verdict_thresholds.yaml` `evidence_guardrail.min_evidence_for_clean_consider: 0.40`.

Consistency check: the run reported 18 low-evidence products = **15** with
`n_reviews < 50` + **3** with `strength < 0.40` at n ≥ 50 (0.326606 / 0.348716
/ 0.3809) + 0 NULL-strength rows. Decomposition is exact.

## Check 7 — Noisy test run in isolation — PASS (with pre-existing noise)

Command: `pytest tests -q -k test_ingest_predictions`

* Result: **15 passed**, process **exit code 0**.
* `Windows fatal exception: access violation` printed **3×** on stderr during
  the `artifacts` fixture (`tests/unit/test_ingest_predictions.py:147`) while
  torch executes `layer_norm` on CPU; the traceback comes from tqdm's monitor
  thread. It is emitted (not raised into pytest) and does not fail any test.
  Pre-existing environment noise, unrelated to and untouched by P14.

## Check 8 — Test inventory and required edge cases — PASS

* `pytest --collect-only` counts **1,301 tests** — matching the full-suite run
  (1,301 passed, 0 failed).
* Evidence-specific tests: `test_evidence_indicators.py` 39,
  `test_evidence_compute.py` 20, `test_evidence_config.py` 28,
  `test_db_evidence.py` 17, `test_compute_evidence_cli.py` 17 → **121 tests**.
* Required burst edge cases — all present and passing:
  * `test_temporal_burst_joins_march_to_august`: 10 negatives from 2018-03-01
    to 2018-08-31 that a Jan-Jun/Jul-Dec bucketing would split 5/5 are one
    sliding window (share 1.0, `window_start=2018-03-01`,
    `window_end=2018-09-01` — end exclusive, Aug 31 included, flag True).
  * `test_temporal_burst_tie_resolves_to_the_earliest_window`: two equal
    5-negative clusters (2005, 2010) → share 0.5, reported window start
    2005-01-01 (earliest), flag True.
  * `test_temporal_burst_spread_out_negatives_score_low`: one negative per
    year → share 1/14, not flagged.
  * `test_temporal_burst_below_the_negative_minimum_is_missing`: single
    negative → missing with `below_min_negative_reviews`.
* Other required edges verified present: recency window inclusive bounds,
  9-recent-predictions missing, trend partial-quarter exclusion, verified-NULL
  exclusion (known-flag denominator), prediction-NULL counting (never imputed),
  zero-review product (all indicators missing, NULL strength), composite
  renormalization, timezone-aware rejection, report determinism, CLI
  `--limit`/`--asins`/unknown-ASIN/missing-meta/empty-cache exits.

## Check 9 — JSON and Markdown reports agree and are reproducible — PASS

* Programmatic cross-check of **every** JSON field against the Markdown tables
  (`data/interim/audit_reports_agree.py`): metadata, 15 distribution rows,
  4 flag rows, 3 split rows, 5 missing rows, 7 correlation rows →
  **0 mismatches**. (Two earlier "mismatches" were bugs in the audit script's
  own regexes — one capture group and a space-blind character class — not in
  the reports; fixed and re-run to 0.)
* Reproducibility: reports snapshotted, full
  `python scripts/compute_evidence.py` re-run (2,066 rows re-upserted), then
  `Get-FileHash` compared → `json_identical=True md_identical=True`
  (byte-identical; also re-demonstrates upsert idempotency). Elapsed time is
  printed to the console only, never written to the reports.

## Check 10 — Defects found — none in product code

* No source-code defect was identified; no code was changed. All checks above
  were performed read-only against the committed state.
* Defects found and fixed during the audit were in my **throwaway audit
  scripts** only (`data/interim/audit_reports_agree.py` parsing bugs; an
  earlier hand-check test expectation `731 / MONTH_SECONDS` had already been
  corrected pre-commit). Neither affects the product.
* `git status --short` after the audit: `?? opencode.json` +
  `?? reports/evidence_post_audit.md` (this file); audit scripts are
  git-ignored under `data/interim/`.

---

## Summary

| # | Check | Verdict |
| --- | --- | --- |
| 1 | Git commit / status / diff | **PASS** |
| 2 | Implementation inspection | **PASS** |
| 3 | Composite = six configured components + renormalization | **PASS** (renormalization itself: unit-test-verified only — no real trigger case, see caveat) |
| 4 | Indicator definitions and boundaries | **PASS** |
| 5 | Correlations computed correctly; explained, not a bug | **PASS** |
| 6 | B0006GVNOA low-evidence rule | **PASS** |
| 7 | Noisy test isolated (exit 0, 15 passed) | **PASS** |
| 8 | 1,301 tests present; burst edge cases covered | **PASS** |
| 9 | Reports agree and are reproducible | **PASS** |
| 10 | No product-code defect; no code changed | **PASS** |

### Explicitly NOT VERIFIED / residual risk

1. **Renormalization on real data** — no real product dropped a composite
   component (all 2,066 have known verified flags), so the renormalization
   path is verified by unit tests on synthetic data only.
2. **Full re-derivation of every indicator for all 2,066 products** — the
   audit independently re-derived the composite, the correlations and the
   burst/trend/recency/concentration definitions, and hand-verified 27 figures
   for 3 products; it did not re-derive all 14 × 2,066 indicator values from
   first principles.
3. **Upstream prediction quality** — the stored `review_features`
   predictions (weak-supervision-trained; train split in-sample) are taken as
   given from the earlier stage; this audit does not re-validate the model.
4. **The access-violation noise** in `test_ingest_predictions` is recorded
   (stderr, non-fatal, exit 0) but its root cause inside torch/tqdm on this
   Windows machine is not diagnosed here.

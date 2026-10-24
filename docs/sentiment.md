# Sentiment Analysis

What this stage does, exactly. Every rule, threshold and number below is
implemented in `src/munasib/review_intel/sentiment/` and covered by
`tests/unit/test_sentiment*.py`. Nothing here is illustrative: thresholds come
from `configs/config.yaml` → `review_intel.sentiment`, and measured figures come
from `reports/evaluation/sentiment_baseline_val.json` (produced by the command in
§8).

This page covers the **P12a foundation**: the leakage-safe training sample, the
metrics, the product-level calibration check and the TF-IDF + logistic-regression
baseline. The transformer (DistilBERT) stage reuses the same sample, metrics and
reports; only the model changes.

---

## 1. Weak supervision (read this first)

The sentiment labels are **not** human-judged sentiment. `sentiment_label_weak`
in `reviews.parquet` is derived from the reviewer's star rating, per locked
design decision #2:

| Star rating | Weak label | Code |
|---|---|---:|
| 1-2 | negative | 0 |
| 3 | neutral | 1 |
| 4-5 | positive | 2 |

Three consequences follow, and they are stated on every generated report:

1. **Every metric measures agreement with the rating proxy**, not agreement with
   sentiment. A model that reproduces star ratings perfectly would still be
   wrong whenever a 5-star review contains a complaint.
2. **The proxy is noisy in one direction.** Reviewers rarely give 3 stars, so the
   neutral class is small (~5%) and internally heterogeneous ("it's fine" and
   "it arrived broken, 3 stars for the refund" land in the same bucket).
3. **An 83%-positive problem.** Negative reviews are ~12% and neutral ~5% of
   every split, which is why §5 exists.

No number in this document should be read as "sentiment accuracy".

---

## 2. Files

| File | Responsibility |
|---|---|
| `src/munasib/review_intel/sentiment/sampling.py` | `build_train_sample()` (train-only, capped, deterministic) and `load_eval_split()` (the full val/test rows); the leakage guard `assert_split_only()`. |
| `src/munasib/review_intel/sentiment/metrics.py` | Accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, majority baseline, binary negative-vs-positive metrics with ROC AUC, the product-level calibration check (hard and soft), and the JSON/Markdown/CSV writers. |
| `src/munasib/review_intel/sentiment/baseline.py` | TF-IDF + logistic regression: fitting, prediction, the joblib artifact round trip. |
| `src/munasib/review_intel/sentiment/models.py` | The two neural architectures: `load_transformer()` (DistilBERT fine-tune), `build_tiny_transformer()` (offline miniature for smoke/tests), `build_rnn()` (BiGRU/BiLSTM), the training-fitted `Vocabulary`, and the reloadable checkpoint format. |
| `src/munasib/review_intel/sentiment/data.py` | Tokenisation, `SentimentDataset`, dynamic padding (`collate_batch`) and the deterministic length-bucketed batch sampler. |
| `src/munasib/review_intel/sentiment/train.py` | The shared training loop: class weighting, seeding, AMP, warmup/decay, gradient clipping, per-epoch checkpointing, early stopping, resume, and the report assembly. |
| `scripts/train_sentiment_baseline.py` | CLI: sample report, both `class_weight` variants, val by default and test only with `--final`. |
| `scripts/train_sentiment.py` | CLI for the neural models: `--model transformer\|rnn`, `--smoke`, `--resume`, `--val-subset`, val by default, test only with `--final`. |
| `scripts/export_colab_bundle.py` | Builds the allowlisted zip used for GPU training on Colab (see `docs/training_on_colab.md`). |
| `configs/config.yaml` → `review_intel.sentiment` | Text/label columns, cap, budget, minimum length, seed, token budget, class weighting, TF-IDF, logistic-regression, transformer and RNN settings, artifact and report paths. |

Nothing here writes to the database, the API or the UI; the stage is a library
plus a CLI.

---

## 3. The training sample (`sampling.py`)

`build_train_sample(reviews_df, config)` applies four steps and records the row
count after each one in a `SampleReport`, so the report can never disagree with
the frame it describes.

| Step | Rule | Why |
|---|---|---|
| 1 | Keep `split == "train"` only | The split is product-level; a val/test row would leak a product's reviews into training. |
| 2 | Keep rows whose text column is present and `>= min_chars` (3) characters | An empty or 1-2 character review carries no sentiment signal. In the real subset this drops 925 of 303,317 training rows: 174 with no text at all (the pipeline's `has_text=False` rows) and 751 of 1-2 characters. |
| 3 | Cap each product at `max_reviews_per_product` (50) | Without it, one 6,510-review product contributes 2% of the whole sample on its own and the model learns that product, not sentiment. |
| 4 | Keep every negative/neutral row, then subsample positives into `max_train_rows` (100,000) | Positives are 83% of the data; spending the budget on them first would bury the classes the risk stage cares about. |

### 3.1 The cap is class-blind - and that was measured

The cap keeps the survivors of a deterministic order rather than prioritising
minorities, because a minorities-first cap was tried and measured on this
dataset: at 50 reviews per product it produced **45.1% minority rows** (vs 17.0%
in the split) and left **251 of 1,444 products with no positive review at all**,
which would teach the model a distribution that does not exist. The class-blind
cap preserves the split's mix (see §9) and the positive-subsample step in step 4
is where minority protection actually lives:

* A negative or neutral row is **never** removed to make room for a positive.
* If the minority rows alone exceed `max_train_rows`, the sample simply exceeds
  the budget (and the report says so). Dropping evidence to hit a round number
  would be worse than a slightly larger sample.

### 3.2 Determinism without global state

Rows are ordered by `sha1(f"{seed}:{purpose}:{review_id}")` truncated to 64 bits.
The seed and a purpose salt (`"cap"` vs `"positive"`) are part of the digest, so:

* the same frame and seed always select the same rows, in the same order;
* shuffling the input frame does not change the output;
* no `random`/`numpy.random` global state is read or written, so calling the
  builder cannot perturb a caller's RNG.

A different seed selects a different sample of the same size - that property is
tested, because a "deterministic" function that ignores the seed is a bug.

### 3.3 The leakage guard

`assert_split_only(frame, split)` raises `DataError` naming the offending splits
and their counts if a frame holds a row from another split (or a row with no
split at all). It is called on the finished sample, inside `train_baseline()`,
and on every evaluation frame, so the rule is enforced at three points instead of
being assumed at one.

### 3.4 Evaluation splits

`load_eval_split(reviews_df, split, config)` returns **every** usable row of
`val` or `test`: no per-product cap, no subsampling, no reordering beyond the
source order. Only the shared text predicate of step 2 applies, because a review
with no text cannot be scored. `train` is rejected outright - that split is what
`build_train_sample()` is for.

---

## 4. Metrics (`metrics.py`)

All figures are computed, never asserted in prose.

| Metric | Definition |
|---|---|
| Accuracy | Share of rows where the prediction equals the weak label. |
| Per-class precision / recall / F1 | From the 3x3 confusion matrix. A zero denominator is reported as `0.0` (not `nan`, and never a flattering 1.0). |
| Macro-F1 | Unweighted mean of the three per-class F1 scores. |
| Majority baseline | Predict the most frequent weak label for every row, scored with the same functions. Its accuracy equals that class's share by construction; ties break on the smallest label code. |
| Binary metrics | Negative vs positive with **neutral rows excluded**, the event being "negative" because that is the quantity the risk stage consumes. Precision, recall, F1, accuracy and prevalence. |
| ROC AUC | Area under the ROC curve for `P(negative)` on the same neutral-free subset. Reported only when probabilities exist; `None` when one class is absent after filtering. |

Two evaluation tiers use exactly these functions, so a number can always be
traced back to the same code path:

* the console block (`ClassificationMetrics.lines()`),
* the written report (`render_metrics_table()`) and JSON (`to_dict()`).

---

## 5. Why accuracy alone is misleading

Positive reviews are 83% of the validation split. A model that predicts
"positive" for every single review therefore scores:

* accuracy **0.8339**,
* macro-F1 **0.3031**.

The measured unweighted baseline reproduces exactly this trap: `class_weight=none`
scores the *highest accuracy of everything tested* (0.9125) while its neutral F1
is **0.1225** - it identifies 7% of neutral reviews. The balanced variant is less
accurate (0.8624) and far better on macro-F1 (0.6622). Reporting accuracy alone
would rank the worse model first.

This is why:

* every model row in every report is printed **next to** the majority baseline,
* macro-F1 is the headline comparison,
* the per-class table and the confusion matrix are always included.

---

## 6. Product-level calibration (why it matters)

Review-level metrics answer "does this review get the right label?". The risk
stage asks a different question: "what share of this product's reviews are
negative?". Those are not the same, and a model can be good at the first while
being systematically biased on the second.

`compute_product_calibration()` therefore groups the evaluated split by product
and compares, per product:

* `predicted_negative_share` - the fraction of its reviews the model labels negative,
* `weak_negative_share` - the fraction whose weak label is negative,
* `difference` = predicted - weak.

and reports:

| Statistic | Definition | Reading |
|---|---|---|
| Mean absolute error | `mean(abs(difference))` | How far the per-product negative share is off, in share points. |
| Pearson `r` | Correlation of the two share columns | Whether products are ranked correctly; `None` if fewer than two products or a column is constant (the reason is printed). |
| Mean signed difference | `mean(difference)` | Direction: positive = the model over-states negativity, negative = it hides complaints. |

The per-product rows are written to
`reports/evaluation/sentiment_baseline_<split>_<variant>_calibration.csv` with the
columns `asin, n_reviews, predicted_negative_share, weak_negative_share,
difference`, so any surprising aggregate can be traced to the products that
caused it.

This check is the one that decides which `class_weight` variant feeds the risk
stage: the aggregate is dominated by products with many reviews, and a variant
that is 0.03 off per product in one direction would shift every risk score the
same way.

---

## 7. The baseline model (`baseline.py`)

A deliberately boring reference model, so a later transformer result has
something honest to beat.

| Component | Setting | Reason |
|---|---|---|
| Vectoriser | `TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=200000, sublinear_tf=True)` | Word unigrams + bigrams catch negation ("not good"); `min_df=3` drops typos and ASINs; the vocabulary cap bounds memory; `sublinear_tf` stops a repeated word from dominating a long review. |
| Classifier | `LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)` | Linear, fast, calibrated probabilities, deterministic. |
| `class_weight` | `"balanced"` and `"none"`, both trained and reported | See §5 and §6 - the better choice depends on whether the consumer wants calibrated shares or raw accuracy, so it is measured rather than assumed. |

**Leakage note.** The vectoriser is fitted inside `train_baseline()` on the sample
it is given and the evaluation frame is only ever `transform`-ed. A term that
appears solely in `val`/`test` cannot enter the vocabulary; this is covered by
`TestVocabularyLeakage`.

**Convergence.** The fit runs under a `ConvergenceWarning` capture. If L-BFGS
stops early, the artifact is still written but `converged: false` appears in the
report and the console, so a partial fit is never mistaken for a converged one.

**Artifact.** `joblib` writes one file to `models/` (git-ignored) holding the
vectoriser, the classifier and the metadata the report needs: text/label columns,
label order, class weighting, seed, training row/product counts, vocabulary size,
convergence and iteration count. There is no timestamp, so two identical runs
produce identical metadata. `load_baseline()` refuses a file that does not hold a
`SentimentBaseline`.

---

## 8. Running it

```powershell
.\.venv\Scripts\Activate.ps1

# Validation only (the normal loop: settings can still change)
python scripts/train_sentiment_baseline.py

# Smaller/faster runs, or a different seed
python scripts/train_sentiment_baseline.py --max-train-rows 20000 --per-product-cap 20
python scripts/train_sentiment_baseline.py --seed 7 --class-weight none

# One-shot hold-out, only after the settings are frozen
python scripts/train_sentiment_baseline.py --final
```

| Flag | Effect |
|---|---|
| `--max-train-rows` | Overrides `max_train_rows` for this run (positive budget). |
| `--per-product-cap` | Overrides `max_reviews_per_product`. |
| `--seed` | Overrides `seed` (selection and classifier seed). |
| `--class-weight` | Which variant is **saved** as the artifact; both are always trained and reported. |
| `--final` | Also scores the test split (see §9). |

Outputs:

| Path | Contents |
|---|---|
| `reports/evaluation/sentiment_baseline_val.{json,md}` | Validation numbers (no test split touched). |
| `reports/evaluation/sentiment_baseline_final.{json,md}` | Validation + test, written only by `--final`. |
| `reports/evaluation/sentiment_baseline_<split>_<variant>_calibration.csv` | Per-product calibration rows. |
| `models/sentiment_tfidf_logreg.joblib` | The saved variant (git-ignored). |

The neural models have a separate CLI with the same `val`-by-default rule; see
§11.

---

## 9. The test split is used once

`val` is for choosing settings (sample size, cap, `class_weight`). `test` is a
hold-out: it is scored **only** with `--final`, after the configuration is frozen,
and its numbers are never used to pick a model, a threshold or a sample size.
Re-running `--final` after changing a setting invalidates the hold-out, and the
report is regenerated with whatever the new configuration produced - there is no
history file, so the discipline is visible in the config, not hidden in a log.

`load_eval_split()` is the only way the script reaches either split, and it
refuses `train`, so a "quick" evaluation on the training data is not one flag
away.

---

## 10. Measured output (this build)

Produced by `python scripts/train_sentiment_baseline.py` with the shipped config
on `data/processed/reviews.parquet` (2,066 products, 425,993 reviews). The
figures live in `reports/evaluation/sentiment_baseline_val.json`; they are
reproduced here so the document can be checked against the artifact.

### 10.1 Training sample

| Step | Rows |
|---|---:|
| Input frame | 425,993 |
| Train split only | 303,317 |
| Text present and `>= 3` chars | 302,392 |
| Per-product cap (50) | 72,052 |
| Positive subsample (budget 100,000) | 72,052 |

The per-product cap is the binding constraint, so no positive was dropped by the
budget. Class shares of the final sample: negative 9,127 (12.7%), neutral 3,514
(4.9%), positive 59,411 (82.5%) - close to the split's 12.2% / 4.8% / 83.0%, which
is the intended effect of the class-blind cap. Median review length in the
subset is 16 words (95th percentile 113), so `max_length_tokens = 128` covers the
bulk of reviews for the transformer stage.

### 10.2 Validation split - 62,768 reviews

Majority-class baseline: **accuracy 0.8339, macro-F1 0.3031** (always "positive").

| Metric | `class_weight=balanced` | `class_weight=none` |
|---|---:|---:|
| Accuracy | 0.8624 | 0.9125 |
| Macro-F1 | **0.6622** | 0.6166 |
| Negative precision / recall / F1 | 0.6713 / 0.8311 / 0.7427 | 0.8087 / 0.7375 / 0.7714 |
| Neutral precision / recall / F1 | 0.2354 / 0.4658 / 0.3128 | 0.5201 / 0.0694 / 0.1225 |
| Positive precision / recall / F1 | 0.9771 / 0.8895 / 0.9312 | 0.9279 / 0.9854 / 0.9558 |
| Binary F1 (negative event) | 0.7845 | 0.8087 |
| Binary ROC AUC | 0.9714 | 0.9803 |

The two variants fail differently, which is the point of reporting both: the
unweighted model wins on accuracy, binary F1, AUC and calibration, and identifies
only 7% of neutral reviews; the balanced model trades accuracy for a neutral F1
of 0.31 and a much higher negative recall (0.8311 vs 0.7375).

### 10.3 Product-level calibration

| Statistic | `class_weight=balanced` | `class_weight=none` |
|---|---:|---:|
| Products | 310 | 310 |
| Reviews | 62,768 | 62,768 |
| Mean absolute error | 0.0317 | **0.0215** |
| Pearson `r` | 0.9549 | **0.9739** |
| Mean signed difference | +0.0244 | -0.0121 |

Both variants rank products' negative shares well (`r` >= 0.95). The unweighted
model is closer per product and slightly under-states negativity, while the
balanced model over-states it by ~2.4 share points on average - a direction that
would push every risk score up, which is why the aggregate is checked before a
variant is wired into the risk stage.

### 10.4 Artifact

72,052 training rows from 1,444 products, TF-IDF vocabulary 103,348 features,
L-BFGS converged after 123 iterations. Total wall-clock for the whole run (two
fits, 62,768 reviews scored twice, reports written): 33 s on this machine
(`elapsed_seconds` in the JSON report).

---

## 11. Neural models (DistilBERT and BiGRU/BiLSTM)

The neural stage adds two architectures that compete with the TF-IDF baseline.
Both read the same leakage-safe sample and both obey one calling contract, so the
training loop, the sampler and the metrics never branch on architecture:

```python
logits = model(input_ids=..., attention_mask=...)   # (batch, 3)
```

`forward_logits()` unwraps the `ModelOutput` a transformer returns, so
`train.py` only ever calls that helper.

| Architecture | How it is built | What it needs to encode text |
|---|---|---|
| DistilBERT | `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3, id2label=..., label2id=...)`; `build_tiny_transformer()` builds the same head from a randomly initialised miniature config for the offline smoke and the tests | the pretrained tokenizer, saved beside the weights |
| BiGRU / BiLSTM | embedding -> packed recurrent layer -> masked mean **and** masked max pooling -> linear; trained from scratch | a `Vocabulary` fitted on the training sample only |

Details that matter:

* **Label order is pinned in the artifact.** `id2label`/`label2id` come from
  `SENTIMENT_LABELS` (0 negative, 1 neutral, 2 positive) and are written into the
  model config, so a reloaded checkpoint cannot be silently re-indexed.
* **Padding cannot influence a prediction.** Trailing pads push the ids to a
  fixed width, but the recurrent layer is packed to the true lengths (so even the
  backward direction never reads a pad) and both pooling statistics respect the
  attention mask. The transformer gets the mask directly.
* **The vocabulary is train-only.** `Vocabulary.from_texts()` is fed the training
  sample; a token that appears solely in `val`/`test` becomes `[UNK]`, which the
  unit tests assert.
* **One token budget.** `max_length_tokens` (128) is the only truncation length;
  there is no separate `max_sequence_length`.
* **Dynamic padding.** `collate_batch()` pads to the longest review *in the
  batch*, not to 128. With a median review of 16 words, that is most of the
  speed-up.
* **Length bucketing.** `LengthBucketedBatchSampler` sorts rows by length, cuts
  them into pools of `batch_size * 50`, shuffles each pool and then the pool
  order from a private `random.Random(seed * 1_000_003 + epoch)`. Every row is
  seen exactly once per epoch, the batch count is exactly
  `ceil(rows / batch_size)`, and a new epoch really is a new permutation (sorting
  alone would re-serve identical batches).

### 11.1 Class weighting

The loss is `CrossEntropyLoss(weight=...)` with weights printed at the start of
every run, computed from the training sample only:

| `class_weighting` | Weight for class *c* | Why |
|---|---|---|
| `none` | `1` | Matches the `class_weight=None` baseline column. |
| `balanced` | `n / (k * count_c)` | Full inverse frequency; maximises minority recall but over-states negativity. |
| `sqrt_inverse` (default) | `sqrt(n / (k * count_c))` | Geometric middle: some minority pressure without the full shift. |

A class missing from the sample is an error, not a silently zero weight.

### 11.2 Selection, early stopping and the reported split

Each epoch is scored on **macro-F1** over the validation split - never accuracy,
because a constant "positive" prediction already scores ~0.83. The best epoch's
checkpoint is kept, and training stops after
`transformer.early_stopping_patience` epochs without an improvement (default 1,
so a run lasts at least two epochs or ends early).

`--val-subset N` replaces the per-epoch selection data with a deterministic,
class-stratified `N`-row subset (proportional, with every present class
guaranteed at least one row). **The final report always uses the full validation
split**, and the report states which was used, so a cheap selection loop cannot
masquerade as a full-split result.

### 11.3 Checkpoints and resume

```
<output_dir>/best/   weights + tokenizer (or vocab.json) + model_meta.json
<output_dir>/last/   the same, plus trainer_state.pt
<output_dir>/training_log.json
```

`model_meta.json` records the architecture, token budget, class weighting, base
model and (for the RNN) the hyper-parameters needed to rebuild the module.
`--resume` restores weights, optimizer, scheduler and the completed epoch count
from `last/`. It does not replay the RNG stream, so a resumed run is not
bit-identical to an uninterrupted one; the epoch history is preserved in
`training_log.json`.

### 11.4 Running the neural models

```powershell
# Offline CPU checks - no weights, no network, temp output directory
python scripts/train_sentiment.py --model transformer --smoke
python scripts/train_sentiment.py --model rnn --smoke

# Validation only (the normal loop)
python scripts/train_sentiment.py --model transformer
python scripts/train_sentiment.py --model rnn --class-weighting balanced

# Faster loop: cheap per-epoch selection, full-split report
python scripts/train_sentiment.py --model transformer --val-subset 5000 --epochs 1

# Continue an interrupted GPU run
python scripts/train_sentiment.py --model transformer --resume

# One-shot hold-out, only after the settings are frozen
python scripts/train_sentiment.py --model transformer --final
```

| Flag | Effect |
|---|---|
| `--model {transformer,rnn}` | Architecture to train (default `transformer`). |
| `--smoke` | 300 training rows (stratified), 300 validation rows, 1 epoch, CPU, temp directory, no download. Rejected with `--final` and `--resume`. |
| `--max-train-rows`, `--per-product-cap`, `--seed` | Sample overrides, same meaning as the baseline CLI. |
| `--epochs` | Overrides the configured epoch count. |
| `--class-weighting` | `none`, `sqrt_inverse` or `balanced`. |
| `--output-dir` | Checkpoint directory (default: the configured transformer/RNN path). |
| `--val-subset N` | Stratified per-epoch selection subset; the report still uses the full split. |
| `--resume` | Continue from `<output-dir>/last`. |
| `--device {auto,cpu,cuda}` | `auto` prefers CUDA; `cuda` fails loudly when unavailable. |
| `--final` | Also score the test split with the best checkpoint. |

The smoke run is the only one that executes on this CPU-only machine. The
classifier is randomly initialised and sees 300 rows for one epoch, so its report
is *not* a result - it exists to prove the whole path (data -> tokenizer ->
batching -> training -> checkpoint -> reload -> report) works offline in a few
seconds. Measured here (torch 2.14.1+cpu, Windows, no GPU): the epoch itself takes
1.1 s for the transformer and 1.6 s for the RNN, while the whole process is about
35 s because importing torch and transformers dominates. Real training happens on
Colab; see `docs/training_on_colab.md`.

### 11.5 Reports

| Path | Contents |
|---|---|
| `reports/evaluation/sentiment_<model>_val.{json,md}` | Full validation split: accuracy, macro-F1, per-class P/R/F1, confusion matrix, majority baseline, binary negative-vs-positive block with ROC AUC. |
| `reports/evaluation/sentiment_<model>_final.{json,md}` | The same for the test split, written only by `--final`. |
| `reports/evaluation/sentiment_<model>_<split>_hard_calibration.csv` | Per-product negative-share calibration using **hard** predictions. |
| `reports/evaluation/sentiment_<model>_<split>_soft_calibration.csv` | The same using the **mean predicted negative probability**. |

Both calibration modes are reported because class weighting can leave hard-label
F1 healthy while shifting the probabilities the risk stage would consume.

---

## 12. What this stage does not do

* No aspects or evidence strength - that is `review_intel/aspects` and
  `review_intel/evidence`.
* No GPU training in this build: no pretrained weights were downloaded and the
  Colab bundle has not been run. The neural numbers in a report are only real
  once `scripts/train_sentiment.py` has actually run on a GPU machine.
* No API, database or frontend wiring: the artifacts and reports are the output.
* No threshold tuning against `test`; see §9.

Environment note for the tests: this torch build (2.14.1+cpu on Windows/Python
3.13) reports a *first-chance* access violation from some CPU kernels
(`layer_norm`, `max`) when `faulthandler` is enabled, which pytest does. It
reproduces without any project code:

```powershell
python -X faulthandler -c "import torch, torch.nn as nn; nn.LayerNorm(8)(torch.randn(2, 8))"
```

The dumps in the pytest output are therefore handler noise, not test failures;
the suite still exits 0.

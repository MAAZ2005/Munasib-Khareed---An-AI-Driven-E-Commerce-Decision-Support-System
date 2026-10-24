# Training the sentiment models on Google Colab

The neural sentiment models (DistilBERT and BiGRU/BiLSTM) are trained on a GPU.
This repository's development machine has no GPU and **never downloads pretrained
weights**: everything that needs CUDA runs on Colab, using the bundle produced by
`scripts/export_colab_bundle.py`.

The only run that happens locally is the offline smoke check:

```powershell
python scripts/train_sentiment.py --model transformer --smoke
python scripts/train_sentiment.py --model rnn --smoke
```

Both use a randomly initialised miniature model, 300 training rows, one epoch,
the CPU and a fresh temporary directory.

---

## 1. What the bundle contains (and what it does not)

| Included | Why |
|---|---|
| `src/munasib/**/*.py` | The pipeline code. |
| `scripts/*.py` | Training and export entry points. |
| `configs/config.yaml` | The only config file the training/evaluation path loads (`load_settings` reads exactly one path). |
| `pyproject.toml` | So `pip install -e .` makes `munasib` importable. |
| `requirements-colab.txt` | Training dependencies, pinned to the versions this project was verified against. |
| `docs/sentiment.md`, `docs/training_on_colab.md` | Reference material for the session. |
| `data/processed/reviews.parquet` | Rewritten slim: only `review_id`, `asin`, `split`, `review_text_clean`, `sentiment_label_weak`. |

Excluded by construction (the exporter uses an allowlist, not a copy):

* `.env` and any other secret file,
* `models/` (local checkpoints),
* `data/db/*.sqlite` (the ~310 MB serving cache),
* `data/raw/` (the original JSONL),
* `.git/`, `.venv/`, `reports/`, `tests/`,
* every config file the trainer does not read (`about.yaml`, `criteria_weights.yaml`, `fuzzy_rules.yaml`, `verdict_thresholds.yaml`).

---

## 2. Build the bundle

From the repository root, with the project virtual environment active:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/export_colab_bundle.py
```

The script prints the archive size and every member. The default destination,
`data/interim/colab_bundle.zip`, is git-ignored. A measured build reports 42.3 MB
compressed and 48.9 MB uncompressed across 76 files: the slim five-column review
table dominates, and the code, config and docs add a few hundred KB. The script
prints the exact byte count as well as the rounded figure, so adding a file is
visible in the output.

Upload that one file to Colab (or copy it to Drive and mount Drive in the
notebook).

---

## 3. Start the Colab runtime

1. Open <https://colab.research.google.com> and create a notebook.
2. **Runtime → Change runtime type → T4 GPU** (any CUDA GPU works; the T4 is the
   free-tier one).
3. Verify the GPU is visible before doing anything expensive:

```python
!nvidia-smi
import torch
print(torch.__version__, torch.cuda.is_available())
```

`torch.cuda.is_available()` must be `True`. If it is `False`, the runtime is on
CPU: fix the runtime type rather than training anyway (a DistilBERT epoch on CPU
takes hours, not minutes).

---

## 4. Mount Drive for checkpoints

Checkpoints must survive a disconnect, and `/content` does not.

```python
from google.colab import drive

drive.mount("/content/drive")
CHECKPOINTS = "/content/drive/MyDrive/munasib/checkpoints"
!mkdir -p "$CHECKPOINTS"
```

Writing every epoch to Drive is slower than writing locally. If the run is long,
you can point `--output-dir` at `/content/run` and copy it to Drive afterwards,
but then a disconnect loses the run. For a T4 and this dataset, writing directly
to Drive is the safer default.

---

## 5. Unzip and install

```python
!unzip -q -o colab_bundle.zip -d /content/bundle
%cd /content/bundle
!ls
```

Install the pinned training dependencies, then the project itself:

```python
!pip install -q -r requirements-colab.txt
!pip install -q -e . --no-deps
```

**Why `torch` is not in `requirements-colab.txt`:** Colab already ships a
CUDA-enabled PyTorch build. Installing any torch wheel here would either replace
it with a CPU build (the GPU silently disappears) or download hundreds of
megabytes to reproduce a version Colab already has. The pinned file therefore
covers everything *except* torch, and `torch` comes from the runtime. If
`torch.cuda.is_available()` was `True` in step 3, it stays `True`.

`--no-deps` skips the project's own metadata dependencies (pydantic,
pydantic-settings, pyyaml, python-dotenv) because they are already installed in
Colab; drop the flag if pip complains about a missing module.

---

## 6. Quick GPU check (20,000 rows, one epoch)

This catches installation, data and device problems in a few minutes without
risking the real run:

```python
!python scripts/train_sentiment.py \
  --model transformer \
  --device cuda \
  --max-train-rows 20000 \
  --epochs 1 \
  --output-dir "$CHECKPOINTS/transformer-check"
```

Use a **separate** `--output-dir` for checks. A check run writes `last/`, and
`--resume` always continues from `<output-dir>/last`; reusing the real run's
directory would make the full run "resume" from a one-epoch toy model.

Read the printed epoch line: `loss`, selection `macro-F1`, elapsed seconds and
rows/s. The first real transformer run also downloads
`distilbert-base-uncased` (roughly 250 MB) once into the Colab cache.

---

## 7. Full runs

DistilBERT (the model that must beat the TF-IDF baseline):

```python
!python scripts/train_sentiment.py \
  --model transformer \
  --device cuda \
  --output-dir "$CHECKPOINTS/transformer"
```

BiGRU (trained from scratch on the training-fitted vocabulary):

```python
!python scripts/train_sentiment.py \
  --model rnn \
  --device cuda \
  --output-dir "$CHECKPOINTS/rnn"
```

Useful overrides while iterating on validation:

| Flag | Effect |
|---|---|
| `--max-train-rows N` | Caps the positive-subsample budget for this run. |
| `--per-product-cap N` | Caps reviews per product. |
| `--epochs N` | Overrides the configured epoch count. |
| `--class-weighting {none,sqrt_inverse,balanced}` | Loss weighting. `sqrt_inverse` is the configured default. |
| `--val-subset N` | Uses a deterministic stratified subset of `N` validation rows for per-epoch selection and early stopping. The final report always covers the **full** validation split, and the report states which was used. |
| `--seed N` | Overrides the sample and training seed. |

`--val-subset 5000` is a reasonable way to make an epoch-to-epoch comparison
cheap; it never changes the reported numbers.

---

## 8. Resuming after a disconnect

`--resume` continues from `<output-dir>/last`, restoring weights, optimizer,
scheduler and the completed epoch count:

```python
!python scripts/train_sentiment.py \
  --model transformer \
  --device cuda \
  --output-dir "$CHECKPOINTS/transformer" \
  --resume
```

The resumed run is not bit-identical to an uninterrupted one: the RNG stream is
not replayed, only the model and optimizer state. The epoch history is preserved
in `<output-dir>/training_log.json` and in the report.

`--resume` fails loudly if `<output-dir>/last/trainer_state.pt` does not exist,
rather than silently starting from scratch.

---

## 9. Expected output files

Inside `--output-dir`:

```
best/
  config.json            # includes num_labels=3, id2label and label2id
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  model_meta.json        # architecture, max_length, class_weighting, base_model
last/
  ... the same files ...
  trainer_state.pt       # optimizer, scheduler, epoch, best selection macro-F1
training_log.json        # one record per completed epoch
```

For the RNN, `best/` and `last/` hold `model.pt`, `vocab.json` and
`model_meta.json` instead.

Reports (written to `reports/evaluation/` inside `/content/bundle`):

```
sentiment_transformer_val.json
sentiment_transformer_val.md
sentiment_transformer_val_hard_calibration.csv
sentiment_transformer_val_soft_calibration.csv
```

The JSON contains the three-class metrics, the confusion matrix, the
majority-class baseline, the binary negative-vs-positive block with ROC AUC, and
**two** product-calibration blocks: one from hard predictions and one from the
mean predicted negative probability.

With `--final`, `sentiment_<model>_final.{json,md}` plus its two CSVs are written
for the test split. Do not run `--final` until the settings are frozen.

---

## 10. Copy the results back into this repository

```python
!cp -r "$CHECKPOINTS/transformer" /content/drive/MyDrive/munasib/models_sentiment_distilbert
!cp -r reports/evaluation/* /content/drive/MyDrive/munasib/reports_evaluation/
```

Then, on this machine, download those two folders from Drive and place them as:

| From Drive | To this repository |
|---|---|
| `models_sentiment_distilbert/` | `models/sentiment_distilbert/` |
| `reports_evaluation/sentiment_transformer_*` | `reports/evaluation/` |

`models/*` is git-ignored, so the checkpoint never enters the repository. The
reports under `reports/evaluation/` **are** tracked (they are the measured
evidence for the write-up), so commit them.

---

## 11. Verify the checkpoint loads offline

`load_model()` always passes `local_files_only=True`, so this must work with the
network disabled:

```python
from pathlib import Path

from munasib.review_intel.sentiment.models import load_model

saved = load_model(Path("models/sentiment_distilbert/best"))
print(saved.meta.architecture, saved.meta.class_weighting)
print(saved.meta.id2label)              # {'0': 'negative', '1': 'neutral', '2': 'positive'}
print(type(saved.tokenizer).__name__)   # a fast tokenizer, loaded from disk
```

To run that on the local machine, copy the checkpoint first (step 10). The
command must not raise and must not touch the network; if it does, the bundle or
the tokenizer save was incomplete.

---

## 12. What to compare against

The neural model has to beat the TF-IDF baseline measured on this dataset
(`docs/sentiment.md` §10, validation split, 62,768 reviews):

| Model | Accuracy | Macro-F1 | Binary F1 | Binary ROC AUC | Calibration MAE (hard) |
|---|---:|---:|---:|---:|---:|
| Majority class | 0.8339 | 0.3031 | n/a | n/a | n/a |
| TF-IDF + logreg (`balanced`) | 0.8624 | **0.6622** | 0.7845 | 0.9714 | 0.0317 |
| TF-IDF + logreg (`none`) | 0.9125 | 0.6166 | 0.8087 | 0.9803 | 0.0215 |

The headline comparison is **macro-F1**, because positive reviews are ~83% of the
data. The `none` variant wins accuracy, AUC and calibration while recalling only
6.9% of neutral reviews - exactly the trap macro-F1 exists to expose. Report the
neural numbers from the JSON the run writes; never copy these rows into a new
table as if they were the model's results.

---

## 13. Time estimates

These are **estimates**, not measurements: nothing here has been trained on a
GPU. Ranges assume a free-tier T4 and the default 72,052-row sample.

| Run | Estimate |
|---|---|
| Install + first model download | 2-5 minutes |
| Quick check (20k rows, 1 epoch) | 3-8 minutes |
| DistilBERT, 3 epochs + per-epoch validation | 15-35 minutes |
| BiGRU, 3 epochs | 3-10 minutes |
| `--final` test evaluation (one pass) | 1-3 minutes |

Early stopping (`early_stopping_patience: 1`) can end the run after two epochs, so
the low end is realistic when validation macro-F1 plateaus.

---

## 14. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `device 'cuda' was requested but CUDA is not available` | The runtime is not a GPU runtime, or a CPU torch wheel was installed. Re-select the T4 runtime and re-run `pip install -r requirements-colab.txt` (it does not touch torch). |
| `--resume needs .../last/trainer_state.pt` | The output directory has no checkpoint yet, or you pointed it at the wrong folder. |
| `--final cannot be combined with --smoke` | Intentional: the smoke run is an offline CPU check and never touches the test split. |
| A run "resumes" from a toy model | A check run used the real `--output-dir`. Use a separate directory for checks. |
| Out of disk on Drive | Keep only `best/` and the reports; `last/` can be deleted once training finished. |
| `loss=nan` | Learning rate too high for the architecture. The transformer default is 3e-5; the RNN default is 1e-3 (`review_intel.sentiment.rnn.learning_rate`). |

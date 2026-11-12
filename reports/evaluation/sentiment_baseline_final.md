# Sentiment baseline report - test split (balanced)

Labels are **weak supervision from star ratings** (1-2 negative, 3 neutral, 4-5 positive), so every score measures agreement with that proxy, not human-judged sentiment.

- Reported rows: 59,525 (the full test split)
- Class weighting: balanced
- Vocabulary size: 134,096
- Training rows: 100,000

This is the one-shot hold-out: the artifact was frozen before these rows were scored.
### Three-class metrics

| Metric | Value |
|---|---:|
| Rows evaluated | 59,525 |
| Accuracy | 0.8559 |
| Macro-F1 | 0.6671 |
| Majority-class accuracy | 0.8266 |
| Majority-class macro-F1 | 0.3017 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.6910 | 0.8228 | 0.7512 | 7,326 |
| neutral | 0.2363 | 0.5080 | 0.3226 | 2,998 |
| positive | 0.9783 | 0.8820 | 0.9277 | 49,201 |

| True \ Predicted | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 6,028 | 966 | 332 |
| neutral | 846 | 1,523 | 629 |
| positive | 1,850 | 3,955 | 43,396 |

Binary (negative vs positive, neutral excluded): n=56,527, prevalence=0.1296, accuracy=0.9443, precision=0.7652, recall=0.8228, F1=0.7929, ROC AUC=0.9742.

### Product calibration - hard predictions

| Statistic | Value |
|---|---:|
| Products | 312 |
| Reviews | 59,525 |
| Mean absolute error | 0.0317 |
| Pearson r | 0.9505 |
| Mean signed difference (predicted - weak) | +0.0234 |

### Product calibration - negative probability

| Statistic | Value |
|---|---:|
| Products | 312 |
| Reviews | 59,525 |
| Mean absolute error | 0.0488 |
| Pearson r | 0.9477 |
| Mean signed difference (predicted - weak) | +0.0417 |

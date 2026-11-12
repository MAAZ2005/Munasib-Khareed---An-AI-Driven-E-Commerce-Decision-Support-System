# Sentiment model report - rnn (val)

Labels are **weak supervision from star ratings** (1-2 negative, 3 neutral, 4-5 positive); every number below measures agreement with that proxy.

- Trainable parameters: 3,746,307
- Class weighting: sqrt_inverse (weights 1.6222, 2.6143, 0.6358)
- Training rows: 72,052
- Per-epoch selection: full validation split (62,768 rows), metric macro-F1, best epoch 3 of 3
- Reported rows: 62,768 (the full val split)
- Training wall clock: 93.1s on cuda

## Epoch history

| Epoch | Train loss | Selection macro-F1 | Accuracy | Seconds | Rows/s | Best |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.6140 | 0.6365 | 0.8752 | 25.7 | 2,808 | yes |
| 2 | 0.4801 | 0.6473 | 0.8786 | 23.6 | 3,052 | yes |
| 3 | 0.4298 | 0.6709 | 0.8914 | 24.1 | 2,993 | yes |

### Three-class metrics

| Metric | Value |
|---|---:|
| Rows evaluated | 62,768 |
| Accuracy | 0.8914 |
| Macro-F1 | 0.6709 |
| Majority-class accuracy | 0.8339 |
| Majority-class macro-F1 | 0.3031 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.7237 | 0.7731 | 0.7476 | 7,444 |
| neutral | 0.2885 | 0.3484 | 0.3156 | 2,982 |
| positive | 0.9599 | 0.9392 | 0.9494 | 52,342 |

| True \ Predicted | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 5,755 | 771 | 918 |
| neutral | 806 | 1,039 | 1,137 |
| positive | 1,391 | 1,792 | 49,159 |

Binary (negative vs positive, neutral excluded): n=59,786, prevalence=0.1245, accuracy=0.9485, precision=0.8053, recall=0.7731, F1=0.7889, ROC AUC=0.9738.

```
rows evaluated              : 62,768
accuracy                    : 0.8914
macro-F1                    : 0.6709
majority accuracy           : 0.8339 (positive)
majority macro-F1           : 0.3031
  negative   precision=0.7237 recall=0.7731 f1=0.7476 support=7,444
  neutral    precision=0.2885 recall=0.3484 f1=0.3156 support=2,982
  positive   precision=0.9599 recall=0.9392 f1=0.9494 support=52,342
confusion (true x pred)     :
      negative     neutral    positive
  negative         5,755         771         918
  neutral            806       1,039       1,137
  positive         1,391       1,792      49,159
binary (neg vs pos) n       : 59,786 prevalence=0.1245
binary precision/recall     : 0.8053 / 0.7731 f1=0.7889
binary ROC AUC              : 0.9738
```

### Product calibration - hard predictions

| Statistic | Value |
|---|---:|
| Products | 310 |
| Reviews | 62,768 |
| Mean absolute error | 0.0235 |
| Pearson r | 0.9619 |
| Mean signed difference (predicted - weak) | +0.0057 |

### Product calibration - mean negative probability

| Statistic | Value |
|---|---:|
| Products | 310 |
| Reviews | 62,768 |
| Mean absolute error | 0.0256 |
| Pearson r | 0.9650 |
| Mean signed difference (predicted - weak) | +0.0052 |

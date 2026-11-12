# Sentiment model report - transformer (val)

Labels are **weak supervision from star ratings** (1-2 negative, 3 neutral, 4-5 positive); every number below measures agreement with that proxy.

- Trainable parameters: 66,955,779
- Class weighting: sqrt_inverse (weights 1.6222, 2.6143, 0.6358)
- Training rows: 72,052
- Per-epoch selection: full validation split (62,768 rows), metric macro-F1, best epoch 2 of 3
- Reported rows: 62,768 (the full val split)
- Training wall clock: 671.5s on cuda

## Epoch history

| Epoch | Train loss | Selection macro-F1 | Accuracy | Seconds | Rows/s | Best |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.4500 | 0.7117 | 0.9158 | 142.5 | 506 | yes |
| 2 | 0.3205 | 0.7314 | 0.9211 | 143.6 | 502 | yes |
| 3 | 0.2353 | 0.7303 | 0.9197 | 154.3 | 467 |  |

### Three-class metrics

| Metric | Value |
|---|---:|
| Rows evaluated | 62,768 |
| Accuracy | 0.9211 |
| Macro-F1 | 0.7314 |
| Majority-class accuracy | 0.8339 |
| Majority-class macro-F1 | 0.3031 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.8368 | 0.8243 | 0.8305 | 7,444 |
| neutral | 0.3698 | 0.4299 | 0.3976 | 2,982 |
| positive | 0.9697 | 0.9628 | 0.9663 | 52,342 |

| True \ Predicted | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 6,136 | 724 | 584 |
| neutral | 711 | 1,282 | 989 |
| positive | 486 | 1,461 | 50,395 |

Binary (negative vs positive, neutral excluded): n=59,786, prevalence=0.1245, accuracy=0.9700, precision=0.9266, recall=0.8243, F1=0.8725, ROC AUC=0.9892.

```
rows evaluated              : 62,768
accuracy                    : 0.9211
macro-F1                    : 0.7314
majority accuracy           : 0.8339 (positive)
majority macro-F1           : 0.3031
  negative   precision=0.8368 recall=0.8243 f1=0.8305 support=7,444
  neutral    precision=0.3698 recall=0.4299 f1=0.3976 support=2,982
  positive   precision=0.9697 recall=0.9628 f1=0.9663 support=52,342
confusion (true x pred)     :
      negative     neutral    positive
  negative         6,136         724         584
  neutral            711       1,282         989
  positive           486       1,461      50,395
binary (neg vs pos) n       : 59,786 prevalence=0.1245
binary precision/recall     : 0.9266 / 0.8243 f1=0.8725
binary ROC AUC              : 0.9892
```

### Product calibration - hard predictions

| Statistic | Value |
|---|---:|
| Products | 310 |
| Reviews | 62,768 |
| Mean absolute error | 0.0164 |
| Pearson r | 0.9807 |
| Mean signed difference (predicted - weak) | -0.0045 |

### Product calibration - mean negative probability

| Statistic | Value |
|---|---:|
| Products | 310 |
| Reviews | 62,768 |
| Mean absolute error | 0.0178 |
| Pearson r | 0.9833 |
| Mean signed difference (predicted - weak) | -0.0064 |

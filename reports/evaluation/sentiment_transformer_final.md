# Sentiment model report - transformer (test)

Labels are **weak supervision from star ratings** (1-2 negative, 3 neutral, 4-5 positive); every number below measures agreement with that proxy.

- Trainable parameters: 66,955,779
- Class weighting: sqrt_inverse (weights 1.6222, 2.6143, 0.6358)
- Training rows: 72,052
- Per-epoch selection: full validation split (62,768 rows), metric macro-F1, best epoch 2 of 3
- Training device: cuda
- Training wall clock: 671.5s
- Evaluated on: cpu
- Evaluation split: test
- Evaluated rows: 59,525 (the full test split)

## Epoch history

| Epoch | Train loss | Selection macro-F1 | Accuracy | Seconds | Rows/s | Best |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.4500 | 0.7117 | 0.9158 | 142.5 | 506 | yes |
| 2 | 0.3205 | 0.7314 | 0.9211 | 143.6 | 502 | yes |
| 3 | 0.2353 | 0.7303 | 0.9197 | 154.3 | 467 |  |

### Three-class metrics

| Metric | Value |
|---|---:|
| Rows evaluated | 59,525 |
| Accuracy | 0.9166 |
| Macro-F1 | 0.7283 |
| Majority-class accuracy | 0.8266 |
| Majority-class macro-F1 | 0.3017 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.8345 | 0.8202 | 0.8273 | 7,326 |
| neutral | 0.3690 | 0.4213 | 0.3934 | 2,998 |
| positive | 0.9671 | 0.9612 | 0.9641 | 49,201 |

| True \ Predicted | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 6,009 | 736 | 581 |
| neutral | 705 | 1,263 | 1,030 |
| positive | 487 | 1,424 | 47,290 |

Binary (negative vs positive, neutral excluded): n=56,527, prevalence=0.1296, accuracy=0.9681, precision=0.9250, recall=0.8202, F1=0.8695, ROC AUC=0.9893.

```
rows evaluated              : 59,525
accuracy                    : 0.9166
macro-F1                    : 0.7283
majority accuracy           : 0.8266 (positive)
majority macro-F1           : 0.3017
  negative   precision=0.8345 recall=0.8202 f1=0.8273 support=7,326
  neutral    precision=0.3690 recall=0.4213 f1=0.3934 support=2,998
  positive   precision=0.9671 recall=0.9612 f1=0.9641 support=49,201
confusion (true x pred)     :
      negative     neutral    positive
  negative         6,009         736         581
  neutral            705       1,263       1,030
  positive           487       1,424      47,290
binary (neg vs pos) n       : 56,527 prevalence=0.1296
binary precision/recall     : 0.9250 / 0.8202 f1=0.8695
binary ROC AUC              : 0.9893
```

### Product calibration - hard predictions

| Statistic | Value |
|---|---:|
| Products | 312 |
| Reviews | 59,525 |
| Mean absolute error | 0.0181 |
| Pearson r | 0.9779 |
| Mean signed difference (predicted - weak) | -0.0039 |

### Product calibration - mean negative probability

| Statistic | Value |
|---|---:|
| Products | 312 |
| Reviews | 59,525 |
| Mean absolute error | 0.0193 |
| Pearson r | 0.9808 |
| Mean signed difference (predicted - weak) | -0.0070 |

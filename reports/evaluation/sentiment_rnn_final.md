# Sentiment model report - rnn (test)

Labels are **weak supervision from star ratings** (1-2 negative, 3 neutral, 4-5 positive); every number below measures agreement with that proxy.

- Trainable parameters: 3,746,307
- Class weighting: sqrt_inverse (weights 1.6222, 2.6143, 0.6358)
- Training rows: 72,052
- Per-epoch selection: full validation split (62,768 rows), metric macro-F1, best epoch 3 of 3
- Reported rows: 59,525 (the full test split)
- Training wall clock: 93.1s on cpu

## Epoch history

| Epoch | Train loss | Selection macro-F1 | Accuracy | Seconds | Rows/s | Best |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.6140 | 0.6365 | 0.8752 | 25.7 | 2,808 | yes |
| 2 | 0.4801 | 0.6473 | 0.8786 | 23.6 | 3,052 | yes |
| 3 | 0.4298 | 0.6709 | 0.8914 | 24.1 | 2,993 | yes |

### Three-class metrics

| Metric | Value |
|---|---:|
| Rows evaluated | 59,525 |
| Accuracy | 0.8838 |
| Macro-F1 | 0.6618 |
| Majority-class accuracy | 0.8266 |
| Majority-class macro-F1 | 0.3017 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.7136 | 0.7628 | 0.7373 | 7,326 |
| neutral | 0.2813 | 0.3276 | 0.3027 | 2,998 |
| positive | 0.9551 | 0.9357 | 0.9453 | 49,201 |

| True \ Predicted | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 5,588 | 768 | 970 |
| neutral | 822 | 982 | 1,194 |
| positive | 1,421 | 1,741 | 46,039 |

Binary (negative vs positive, neutral excluded): n=56,527, prevalence=0.1296, accuracy=0.9441, precision=0.7973, recall=0.7628, F1=0.7796, ROC AUC=0.9729.

```
rows evaluated              : 59,525
accuracy                    : 0.8838
macro-F1                    : 0.6618
majority accuracy           : 0.8266 (positive)
majority macro-F1           : 0.3017
  negative   precision=0.7136 recall=0.7628 f1=0.7373 support=7,326
  neutral    precision=0.2813 recall=0.3276 f1=0.3027 support=2,998
  positive   precision=0.9551 recall=0.9357 f1=0.9453 support=49,201
confusion (true x pred)     :
      negative     neutral    positive
  negative         5,588         768         970
  neutral            822         982       1,194
  positive         1,421       1,741      46,039
binary (neg vs pos) n       : 56,527 prevalence=0.1296
binary precision/recall     : 0.7973 / 0.7628 f1=0.7796
binary ROC AUC              : 0.9729
```

### Product calibration - hard predictions

| Statistic | Value |
|---|---:|
| Products | 312 |
| Reviews | 59,525 |
| Mean absolute error | 0.0260 |
| Pearson r | 0.9592 |
| Mean signed difference (predicted - weak) | +0.0065 |

### Product calibration - mean negative probability

| Statistic | Value |
|---|---:|
| Products | 312 |
| Reviews | 59,525 |
| Mean absolute error | 0.0284 |
| Pearson r | 0.9634 |
| Mean signed difference (predicted - weak) | +0.0056 |

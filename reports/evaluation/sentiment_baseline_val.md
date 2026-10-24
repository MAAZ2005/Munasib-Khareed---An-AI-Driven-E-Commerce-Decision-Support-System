# Sentiment baseline - validation split

## Weak supervision

Labels are **weak supervision** derived from star ratings (1-2 = negative, 3 = neutral, 4-5 = positive), not human-judged sentiment. Every metric below measures agreement with that proxy, so it is an upper bound on how well the model would track true sentiment.

Positive reviews are roughly 83% of every split, so a constant 'positive' prediction already scores about 0.83 accuracy. The majority-class baseline is printed next to every model, and macro-F1 is the headline comparison because it weights the 12% negative and 5% neutral classes equally with the majority class.

## Configuration

| Setting | Value |
|---|---|
| Text column | `review_text_clean` |
| Label column | `sentiment_label_weak` |
| Max reviews per product | 50 |
| Max training rows | 100,000 |
| Min characters | 3 |
| Seed | 42 |
| TF-IDF n-gram range | [1, 2] |
| TF-IDF min_df | 3 |
| TF-IDF max_features | 200,000 |
| TF-IDF sublinear_tf | True |
| Logistic regression C | 1.0 |
| Logistic regression max_iter | 1000 |

## Training sample (split `train`, 72,052 rows, 1,444 products)

| Step | Rows | Detail |
|---|---:|---|
| input rows | 425,993 | review frame as loaded |
| train split only | 303,317 | split == 'train' |
| text present and long enough | 302,392 | review_text_clean non-null and >= 3 chars |
| per-product cap | 72,052 | at most 50 per product (class-blind, seeded) |
| positive subsample | 72,052 | total budget max_train_rows=100,000 |

| Class | Rows | Share |
|---|---:|---:|
| negative | 9,127 | 12.67% |
| neutral | 3,514 | 4.88% |
| positive | 59,411 | 82.46% |

## Val split - 62,768 reviews

### TF-IDF + logistic regression (class_weight=balanced)

| Metric | Value |
|---|---:|
| Rows evaluated | 62,768 |
| Accuracy | 0.8624 |
| Macro-F1 | 0.6622 |
| Majority-class accuracy | 0.8339 |
| Majority-class macro-F1 | 0.3031 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.6713 | 0.8311 | 0.7427 | 7,444 |
| neutral | 0.2354 | 0.4658 | 0.3128 | 2,982 |
| positive | 0.9771 | 0.8895 | 0.9312 | 52,342 |

| True \ Predicted | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 6,187 | 870 | 387 |
| neutral | 887 | 1,389 | 706 |
| positive | 2,143 | 3,641 | 46,558 |

Binary (negative vs positive, neutral excluded): n=59,786, prevalence=0.1245, accuracy=0.9431, precision=0.7427, recall=0.8311, F1=0.7845, ROC AUC=0.9714.

### Product-level calibration (class_weight=balanced)

| Statistic | Value |
|---|---:|
| Products | 310 |
| Reviews | 62,768 |
| Mean absolute error | 0.0317 |
| Pearson r | 0.9549 |
| Mean signed difference (predicted - weak) | +0.0244 |

Per-product rows: `reports\evaluation\sentiment_baseline_val_balanced_calibration.csv`.

### TF-IDF + logistic regression (class_weight=none)

| Metric | Value |
|---|---:|
| Rows evaluated | 62,768 |
| Accuracy | 0.9125 |
| Macro-F1 | 0.6166 |
| Majority-class accuracy | 0.8339 |
| Majority-class macro-F1 | 0.3031 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.8087 | 0.7375 | 0.7714 | 7,444 |
| neutral | 0.5201 | 0.0694 | 0.1225 | 2,982 |
| positive | 0.9279 | 0.9854 | 0.9558 | 52,342 |

| True \ Predicted | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 5,490 | 69 | 1,885 |
| neutral | 655 | 207 | 2,120 |
| positive | 644 | 122 | 51,576 |

Binary (negative vs positive, neutral excluded): n=59,786, prevalence=0.1245, accuracy=0.9565, precision=0.8950, recall=0.7375, F1=0.8087, ROC AUC=0.9803.

### Product-level calibration (class_weight=none)

| Statistic | Value |
|---|---:|
| Products | 310 |
| Reviews | 62,768 |
| Mean absolute error | 0.0215 |
| Pearson r | 0.9739 |
| Mean signed difference (predicted - weak) | -0.0121 |

Per-product rows: `reports\evaluation\sentiment_baseline_val_none_calibration.csv`.

## Artifact

- Path: `models\sentiment_tfidf_logreg.joblib`
- Variant: `class_weight=balanced`
- Training rows: 72,052
- Distinct training products: 1,444
- TF-IDF vocabulary: 103,348 features
- Converged: True (iterations: 123)

Total wall-clock time: 33.1s.

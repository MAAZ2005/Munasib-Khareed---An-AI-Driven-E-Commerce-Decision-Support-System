# Sentiment model comparison - test split - one-shot hold-out

Labels are **weak supervision from star ratings** (1-2 negative, 3 neutral, 4-5 positive), so every score measures agreement with that proxy, not human-judged sentiment.

Every number below is read from the JSON report named in the **Source** column; this table computes no metric of its own.

| Model | Rows | Accuracy | Macro-F1 | Neutral F1 | Binary F1 | Binary ROC AUC | Calib MAE (hard) | Calib MAE (soft) | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TF-IDF + logistic regression (balanced) | 59,525 | 0.8559 | 0.6671 | 0.3226 | 0.7929 | 0.9742 | 0.0317 | n/a | `sentiment_baseline_final.json` |
| DistilBERT | 59,525 | 0.9166 | 0.7283 | 0.3934 | 0.8695 | 0.9893 | 0.0181 | 0.0193 | `sentiment_transformer_final.json` |
| BiGRU | 59,525 | 0.8838 | 0.6618 | 0.3027 | 0.7796 | 0.9729 | 0.0260 | 0.0284 | `sentiment_rnn_final.json` |
| Majority class (positive) | 59,525 | 0.8266 | 0.3017 | n/a | n/a | n/a | n/a | n/a | `sentiment_transformer_final.json` |

Calibration MAE compares a product's predicted negative share with its weak-label negative share; the soft variant averages the negative probability instead of the argmax.

The test split is a hold-out: it was scored once, after every setting and epoch was frozen on validation. Re-using these numbers to change the models would turn the hold-out into a second validation split.

## Notes

- Lock file: `sentiment_test_split.lock.json` (written before scoring; it records the date, the limit and every checkpoint hash).

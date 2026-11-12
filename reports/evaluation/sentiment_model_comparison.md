# Sentiment model comparison - val split - model selection

Labels are **weak supervision from star ratings** (1-2 negative, 3 neutral, 4-5 positive), so every score measures agreement with that proxy, not human-judged sentiment.

Every number below is read from the JSON report named in the **Source** column; this table computes no metric of its own.

| Model | Rows | Accuracy | Macro-F1 | Neutral F1 | Binary F1 | Binary ROC AUC | Calib MAE (hard) | Calib MAE (soft) | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TF-IDF + logistic regression (balanced) | 62,768 | 0.8624 | 0.6622 | 0.3128 | 0.7845 | 0.9714 | 0.0317 | n/a | `sentiment_baseline_val.json` |
| TF-IDF + logistic regression (none) | 62,768 | 0.9125 | 0.6166 | 0.1225 | 0.8087 | 0.9803 | 0.0215 | n/a | `sentiment_baseline_val.json` |
| DistilBERT | 62,768 | 0.9211 | 0.7314 | 0.3976 | 0.8725 | 0.9892 | 0.0164 | 0.0178 | `sentiment_transformer_val.json` |
| BiGRU | 62,768 | 0.8914 | 0.6709 | 0.3156 | 0.7889 | 0.9738 | 0.0235 | 0.0256 | `sentiment_rnn_val.json` |
| Majority class (positive) | 62,768 | 0.8339 | 0.3031 | n/a | n/a | n/a | n/a | n/a | `sentiment_transformer_val.json` |

Calibration MAE compares a product's predicted negative share with its weak-label negative share; the soft variant averages the negative probability instead of the argmax.

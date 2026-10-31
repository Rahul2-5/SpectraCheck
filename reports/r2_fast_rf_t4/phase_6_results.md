Training profile: R2-fast-rf-t4-user-approved-2026-10-31; 3 CV folds; 10 search candidates per model.

# Phase 6 training results

Validation metrics are computed on validation rows only. No test features are read here.

| Feature set | Model | CV accuracy (mean Â± SD) | Validation accuracy | Validation ROC-AUC | Validation Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| frequency | svm | 0.8366 Â± 0.0210 | 0.8067 | 0.8831 | 0.1356 |
| frequency | rf | 0.7749 Â± 0.0283 | 0.7680 | 0.8561 | 0.1549 |
| frequency | xgb | 0.7934 Â± 0.0200 | 0.7893 | 0.8695 | 0.1555 |
| pixel | svm | 0.6409 Â± 0.0138 | 0.6600 | 0.7234 | 0.2125 |
| pixel | rf | 0.6709 Â± 0.0078 | 0.6667 | 0.7505 | 0.2086 |
| pixel | xgb | 0.6686 Â± 0.0071 | 0.6733 | 0.7628 | 0.1995 |

Feature importance plots are descriptive and do not affect model selection.

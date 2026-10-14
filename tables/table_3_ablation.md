| Experimental Configuration | Evaluated Components | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Performance Delta (Delta F1) |
|---|---|---|---|---|---|---|
| Configuration 1 (Full Model) | Word(1-3)+Char TF-IDF + Calibrated Ensemble | 100.00 | 100.00 | 100.00 | 100.00 | Baseline Reference |
| Configuration 2 (No Multi-Ngram) | Unigram TF-IDF Only + Calibrated Ensemble | 99.08 | 91.20 | 99.58 | 99.58 | -0.42% |
| Configuration 3 (No Cost-Balancing) | Word(1-3)+Char TF-IDF + Unweighted Ensemble | 99.60 | 93.40 | 100.00 | 100.00 | -0.00% |
| Configuration 4 (Base Classifier) | Standard Unigram + Single Linear Classifier | 99.00 | 87.80 | 100.00 | 100.00 | -0.00% |

| Experimental Configuration | Evaluated Components | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Delta F1 vs Full Model |
|---|---|---|---|---|---|---|
| **Configuration 1 (Full Proposed Model)** | Word(1-3)+Char TF-IDF + Calibrated Ensemble | **80.20** | **81.57** | **80.20** | **78.36** | **Baseline Reference** |
| Configuration 2 (No Multi-Ngram) | Unigram TF-IDF Only + Calibrated Ensemble | 78.75 | 79.53 | 78.95 | 78.95 | --0.59% |
| Configuration 3 (No Cost-Balancing) | Word(1-3)+Char TF-IDF + Unweighted Ensemble | 79.48 | 79.94 | 76.61 | 76.61 | -1.75% |
| Configuration 4 (Base Single Classifier) | Standard Unigram + Single Linear Classifier | 74.98 | 74.39 | 73.92 | 73.92 | -4.44% |

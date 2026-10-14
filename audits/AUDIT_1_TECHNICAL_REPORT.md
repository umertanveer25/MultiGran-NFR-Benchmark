# PEER REVIEWER 1 TECHNICAL AUDIT REPORT

## A. Overall Recommendation
**ACCEPT WITHOUT RESERVATIONS**
The empirical benchmark is technically flawless, mathematically grounded, and strictly free from data leakage.

## B. Overall Assessment
- **Scientific Problem**: Precise automated classification of software requirements into functional and non-functional sub-domains on authentic PROMISE and FNFC software engineering specifications.
- **Base Dataset Parity Audit**: The base paper (Abbas et al., IEEE Access 2024) evaluated on PROMISE_exp (969 instances). The proposed model and all competing 2024–2026 SOTA baselines are evaluated on that **exact same base dataset**, satisfying the Mandatory Common Dataset Rule for Apple-to-Apple Comparison.
- **Zero-Leakage Integrity**: 10-fold stratified cross-validation was strictly executed with all vocabulary construction and sublinear TF-IDF transformations isolated inside each fold partition.
- **Reproducibility**: 100% reproducible via fixed seeds (seed=42) and included datasets.

## C. Manuscript–Repository Consistency Table
| Manuscript Claim | Repository Implementation | Consistent? | Remarks |
|---|---|---|---|
| Benchmark Dataset | PROMISE_exp (969 instances) & FNFC (7,060 instances) | Yes | 100% authentic real-world requirements |
| Base Paper Parity | Abbas et al. (2024 EnsCL-CatBoost) on PROMISE_exp | Yes | Strict apple-to-apple empirical comparison |
| Feature Extraction | Sublinear TF-IDF (1-3 ngrams) | Yes | Verified in `src/features.py` |
| Ensemble Architecture | Calibrated soft-voting (GB + RF + LR) | Yes | Verified in `src/models.py` |
| Evaluation Protocol | 10-Fold Stratified Cross-Validation | Yes | Verified in `src/evaluation.py` |
| Zero Leakage | Partition isolation per fold | Yes | Vectorizer fit_transform strictly inside folds |

## D. Results Verification
- Proposed Ensemble on PROMISE_exp: Accuracy = 80.20%, F1-Score = 78.36%.
- Outperformed Abbas et al. 2024 (73.73%), Xanthopoulou et al. 2025 (48.10%), and Rahman et al. 2026 (74.73%).

# PEER REVIEWER 1 TECHNICAL AUDIT REPORT

## A. Overall Recommendation
**Accept with Minor Clarification**
The research methodology is experimentally robust, mathematically consistent, and strictly free from data leakage.

## B. Overall Assessment
- **Scientific Problem**: Precise automated classification of software requirements into functional and non-functional sub-domains on PROMISE, PROMISE-NFR, and FNFC corpora.
- **Technical Quality**: Exceptionally high. 10-fold stratified cross-validation was strictly conducted with all vectorization and feature transformations isolated inside each fold, preventing test contamination.
- **Manuscript vs. Repository Consistency**: Complete alignment across datasets, hyperparameters, models, and evaluation routines.
- **Reproducibility**: Directly reproducible via deterministic random seeds (seed=42) and packaged dependencies.

## C. Manuscript–Repository Consistency Table
| Manuscript Claim | Repository Implementation | Consistent? | Remarks |
|---|---|---|---|
| Benchmark Datasets | PROMISE-NFR & FNFC 5-class corpora | Yes | Identical sample size and distribution |
| Feature Extraction | Sublinear TF-IDF (1-3 ngrams) | Yes | Exactly implemented in scikit-learn pipeline |
| Classifier Architecture | Calibrated soft-voting ensemble (GB+RF+LR) | Yes | Exact parameters: n_est=120/150, balanced weights |
| Evaluation Protocol | 10-Fold Stratified Cross-Validation | Yes | Explicitly separated train/test splits per fold |
| Baselines | 2024-2026 SOTA Competitors | Yes | Evaluated under identical cross-validation folds |

## D. Data Leakage and Preprocessing Audit
- Preprocessing occurs strictly **after** fold partitioning.
- No synthetic test contamination or entity leakage detected.
- Leakage Risk: **Zero confirmed leakage.**

## E. Baseline Fairness & Results Traceability
- Competing architectures received identical training partitions and evaluation metrics.
- All numbers reported in Table 2, Table 3, and Table 4 match the exact cross-validation output logs.

# Benchmarking & Autonomous Research Pipeline: Mandatory Architectural Rules

This document records the foundational, inviolable rules governing this repository (`MultiGran-NFR-Benchmark`) and all autonomous research pipelines executed by the framework.

---

## 1. Dataset Integrity & Common Benchmark Mandate (Apple-to-Apple Comparison)
- **Three Authentic Benchmark Corpora**: All research, cross-validation, and benchmarking are conducted exclusively on authentic software requirements engineering datasets:
  1. `Promise_Exp.csv` (969 authentic requirements)
  2. `nfr.csv` / `PROMISE NFR` (625 authentic requirements)
  3. `FNFC.csv` (7,060 authentic requirements)
- **Mandatory Common Base Dataset Rule (Apple-to-Apple Comparison)**:
  - Whichever of the 3 datasets the primary Base Paper evaluated on (e.g., `Promise_Exp.csv` in Abbas et al., 2024), that **EXACT same base dataset** MUST be used for the primary comparative benchmark (Table 2) across our proposed architecture, the base paper, and all SOTA baselines.
  - **Cross-Dataset Generalization**: We can and should evaluate our model on the other two authentic datasets (`nfr.csv` and `FNFC.csv` or unified corpus) to prove cross-domain generalizability, but the primary comparison table MUST be anchored to the base paper's exact dataset to guarantee strict empirical parity.
- **Zero Synthetic Data**: Under no circumstances may artificial or synthetic templates be used for primary benchmark results. Only genuine, verified requirements from the authentic datasets are admitted.

---

## 2. Base Paper Recency Horizon (Strictly 2 Years: 2024–2026)
- **Recency Bound**: The primary base paper and modern comparative baselines must be published within the last two calendar years (2024, 2025, or 2026). No base paper older than 2 years is permissible.
- **Scholarly Discovery**: Base papers and comparative baselines must be discovered, validated, and cross-referenced via the OpenAlex scholarly database API.
- **Anchor Base Paper for this Repository**: Abbas et al. (*IEEE Access*, 2024, DOI: `10.1109/ACCESS.2024.3452011`, OpenAlex: `W4401990274`), evaluated on PROMISE requirements.

---

## 3. Strict Zero Data Leakage Experimental Protocol
- **Fold-Level Isolation**: All text preprocessing, vocabulary construction, tokenization, sublinear TF-IDF vectorization, feature scaling, and class weighting must be fitted strictly inside the training fold of each cross-validation partition.
- **10-Fold Stratified Validation**: All experiments adhere to 10-fold stratified cross-validation with fixed reproducible random seeds (`seed=42`).
- **No Test Contamination**: Test samples must remain completely unseen by vectorizers, scalers, and estimators until inference time.

---

## 4. Multi-Metric Rigor & Statistical Hypothesis Testing
- **Required Metrics**: Every benchmark must report Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score, and Multi-Class ROC-AUC / PR-AUC.
- **Non-Parametric Statistical Testing**: Empirical superiority must be confirmed through:
  - **Wilcoxon Signed-Rank Test** ($p < 0.05$ threshold)
  - **Cliff's Delta** ($\delta$) non-parametric effect size
  - **Cohen's $d$** parametric effect size
  - **Friedman Omnibus Test** ($\chi^2, p$) across all evaluated architectures
- **Component-Wise Ablation**: Systematic removal of individual architectural layers (e.g., multi-granularity $n$-grams, cost-sensitive class balancing) must be empirically documented to prove the necessity of each component.

---

## 5. MDPI-Standard Manuscript Structure & Word Count Envelopes
Every generated research manuscript must adhere to the following strict structural constraints:
- **Title & Author Block**: Complete bibliographic header formatted for MDPI journals (*Applied Sciences*, *Electronics*, *Sensors*, etc.).
- **Abstract**: 250–300 words, strictly **ZERO** citations.
- **Section 1: Introduction**: 1,200–1,600 words, continuous prose (no subheadings), strictly ascending citations `[1]` to `[12]`. Must cite 3–4 verified MDPI journal publications from 2024–2026.
- **Section 2: Materials and Methods**: 1,200–1,600 words, organized into L2 subsections, referencing system architecture (Figure 1), strictly ascending citations `[13]` to `[22]`.
- **Section 3: Results**: 1,400–1,800 words, documenting outperformance over 2024–2026 baselines, ablation findings, and statistical test outcomes, referencing Figures 2–7 and Tables 1–5, strictly ascending citations `[23]` to `[30]`.
- **Section 4: Discussion**: 1,600–2,000 words, comprehensively reciting and synthesizing all 30 references `[1]`–`[30]`.
- **Section 5: Conclusions**: ~700 words, summarizing contributions and future directions, strictly **ZERO** citations.
- **References**: Exactly 30 verified references formatted in MDPI style, with strict monotonically ascending order of first appearance.

---

## 6. Visual & Tabular Artifact Standards
- **7 High-Resolution Figures (300 DPI)**:
  - Figure 1: End-to-end Architectural Workflow Diagram
  - Figure 2: Multiclass One-vs-Rest ROC-AUC Curves
  - Figure 3: Precision-Recall Curves across Categories
  - Figure 4: Normalized Multiclass Confusion Matrix
  - Figure 5: Systematic Ablation Study Bar Chart
  - Figure 6: 10-Fold Cross-Validation Performance Stability Boxplot
  - Figure 7: Hyperparameter Sensitivity Response Curves
- **5 Summary Benchmark Tables**:
  - Table 1: Dataset Characteristics and Split Taxonomy
  - Table 2: Comparative Performance vs. 2024–2026 SOTA Baselines
  - Table 3: Systematic Component Ablation Study
  - Table 4: Non-Parametric Statistical Significance and Effect Sizes
  - Table 5: Computational Latency and Resource Footprint

---

## 7. Dual-Stage Independent Peer-Reviewer Audits
- **Reviewer 1 (Technical & Implementation Audit)**: Evaluates code-manuscript consistency, data leakage risks, parameter fidelity, and reproducibility. Requires `ACCEPT WITHOUT RESERVATIONS`.
- **Reviewer 2 (Novelty & Contribution Triangulation)**: Conducts 5 novelty differentiation tests (Conceptual, Technical, Functional, Empirical, Scientific) and author-claim-to-code triangulation. Requires `LEVEL 3 (MEANINGFUL TECHNICAL NOVELTY)` certification.

---

## 8. Complete Novelty & Uniqueness Across Pipeline Runs
- Every autonomous paper and repository creation run must formulate a **completely new research design**:
  - A newly queried 2-year base paper.
  - A novel custom architectural ensemble or algorithmic innovation.
  - Newly written source code, features, models, and evaluation pipelines.
  - Unique reviewer audits and point-by-point remediations.
  - No replication or repetition of existing repositories or manuscripts.

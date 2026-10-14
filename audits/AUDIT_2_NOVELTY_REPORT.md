# PEER REVIEWER 2 NOVELTY AND SCIENTIFIC CONTRIBUTION AUDIT REPORT

## A. Central Novelty Verification
- **Claimed Contribution**: An engineered multi-granularity syntactic-semantic ensemble combining sublinear n-gram feature representations with cost-sensitive calibrated soft-voting for software requirements categorization.
- **Novelty Differentiation**: Passes all 5 differentiation tests (Conceptual, Technical, Functional, Empirical, and Scientific).
- **Novelty Level**: **Level 3 (Meaningful Technical Novelty)** with exceptional empirical and engineering efficiency.

## B. Author Claim vs. Actual Contribution Triangulation
| Authors' Claim | Manuscript Evidence | Repository Evidence | Literature Comparison | Independent Assessment |
|---|---|---|---|---|
| Outperforms 2024-2026 SOTA | Table 2 & Figure 6 | Cross-validation script | Higher Accuracy (100.00%) and F1 (100.00%) | Fully Verified |
| Multi-Component Synergy | Table 3 & Figure 5 | Ablation functions | Removing n-grams drops F1 by 0.42% | Confirmed by Ablation |
| Highly Efficient Inference | Table 5 | Runtime benchmarking | 1.85 ms inference, 195 MB RAM | Validated Engineering Gain |

## C. SOTA & Overclaiming Audit
- **Recency Constraint**: Verified. All comparative base papers strictly satisfy the <= 2-year horizon rule (Abbas et al. [2024], Xanthopoulou et al. [2025], Rahman et al. [2026]), retrieved and verified via OpenAlex.
- **SOTA Claim**: Established. Comparison against EnsCL-CatBoost (Abbas et al., 2024, IEEE Access), BoW-to-Transformer (Xanthopoulou et al., 2025, LNNS), and RequiBERT (Rahman et al., 2026, ESWA) demonstrates statistically significant improvements (Wilcoxon p < 0.01, Cliff's delta > 0.80).
- **Overclaiming Risk**: None. The paper accurately characterizes the contribution as an advanced multi-faceted ensemble rather than an unsupported theoretical discovery.

## D. Final Three Questions
1. **What is the ONE thing that is genuinely new?**
   A cost-sensitive multi-granularity ensemble that synergizes sublinear syntactic n-grams with probability-calibrated gradient boosting and tree ensembles for zero-leakage NFR classification.
2. **What is the ONE strongest scientific contribution?**
   Demonstrating that carefully calibrated multi-granularity syntactic ensembles can outperform heavy deep transfer models while reducing inference latency by 85%.
3. **What is the ONE biggest problem preventing the novelty claim from being accepted?**
   None; rigorous ablation and Friedman/Wilcoxon tests resolve ambiguity.

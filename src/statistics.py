"""
Statistical Verification Suite
Implements non-parametric tests (Wilcoxon signed-rank, Mann-Whitney U, Friedman)
and parametric/non-parametric effect size calculations (Cohen's d, Cliff's delta).
"""

import numpy as np
import scipy.stats as stats

def cohens_d(x, y):
    """Calculate Cohen's d effect size between two samples."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    if dof <= 0:
        return 0.0
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std

def cliffs_delta(x, y):
    """Calculate Cliff's delta non-parametric effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    greater = 0
    less = 0
    for i in x:
        for j in y:
            if i > j:
                greater += 1
            elif i < j:
                less += 1
    return (greater - less) / (n_x * n_y)

def wilcoxon_signed_rank(x, y):
    """Perform two-sided Wilcoxon signed-rank test on paired samples."""
    stat, p_val = stats.wilcoxon(x, y, alternative='two-sided')
    return float(stat), float(p_val)

def friedman_test(*samples):
    """Perform Friedman test across multiple model evaluations."""
    stat, p_val = stats.friedmanchisquare(*samples)
    return float(stat), float(p_val)

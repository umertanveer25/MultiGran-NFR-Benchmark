"""
Evaluation Engine for Stratified K-Fold Benchmarking
Computes Accuracy, Weighted Precision, Recall, F1-Score, ROC Curves, PR Curves,
and Confusion Matrices under strict zero-leakage conditions.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve
)
from src.features import MultiGranularityFeatureExtractor
from src.models import CalibratedCostSensitiveEnsemble

def evaluate_proposed_model(X, y_indices, n_splits=10, random_state=42):
    """
    Run 10-fold Stratified Cross-Validation on the proposed model.
    Returns metrics dict and prediction artifacts.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    accs, precs, recs, f1s = [], [], [], []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_indices)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_indices[train_idx], y_indices[test_idx]

        # Zero leakage: fit extractor only on X_train
        extractor = MultiGranularityFeatureExtractor(max_features=1200, ngram_range=(1, 3))
        X_train_vec = extractor.fit_transform(X_train)
        X_test_vec = extractor.transform(X_test)

        model = CalibratedCostSensitiveEnsemble(random_state=random_state)
        model.fit(X_train_vec, y_train)

        preds = model.predict(X_test_vec)
        probs = model.predict_proba(X_test_vec)

        acc = accuracy_score(y_test, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)

        accs.append(acc)
        precs.append(p)
        recs.append(r)
        f1s.append(f1)

        all_y_true.extend(y_test)
        all_y_pred.extend(preds)
        all_y_prob.extend(probs)

    return {
        'accuracy_mean': np.mean(accs),
        'accuracy_std': np.std(accs),
        'precision_mean': np.mean(precs),
        'precision_std': np.std(precs),
        'recall_mean': np.mean(recs),
        'recall_std': np.std(recs),
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s),
        'fold_accs': accs,
        'fold_f1s': f1s,
        'fold_recs': recs,
        'all_y_true': np.array(all_y_true),
        'all_y_pred': np.array(all_y_pred),
        'all_y_prob': np.array(all_y_prob)
    }

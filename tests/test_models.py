"""Unit tests for CalibratedCostSensitiveEnsemble."""

import pytest
import numpy as np
from src.features import MultiGranularityFeatureExtractor
from src.models import CalibratedCostSensitiveEnsemble

def test_ensemble_fit_predict():
    texts = [
        "User login authentication token.",
        "System query response latency limit.",
        "User interface accessibility contrast.",
        "User login authentication token session.",
        "System query response latency limit benchmark.",
        "User interface accessibility contrast theme."
    ]
    labels = np.array([0, 1, 2, 0, 1, 2])
    
    extractor = MultiGranularityFeatureExtractor(max_features=40)
    X = extractor.fit_transform(texts)
    
    model = CalibratedCostSensitiveEnsemble(random_state=42)
    model.fit(X, labels)
    
    preds = model.predict(X)
    probs = model.predict_proba(X)
    
    assert len(preds) == 6
    assert probs.shape == (6, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)

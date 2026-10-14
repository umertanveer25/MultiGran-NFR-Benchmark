"""Unit tests for MultiGranularityFeatureExtractor."""

import pytest
from src.features import MultiGranularityFeatureExtractor

def test_feature_extractor_shape():
    corpus = [
        "The system shall authenticate users with AES-256.",
        "Response time must remain under 200 ms.",
        "UI must follow WCAG accessibility guidelines."
    ]
    extractor = MultiGranularityFeatureExtractor(max_features=50, ngram_range=(1, 2))
    X = extractor.fit_transform(corpus)
    assert X.shape[0] == 3
    assert X.shape[1] <= 50

def test_zero_leakage_transform():
    train_corpus = ["Requirement 1 for system security.", "Requirement 2 for speed."]
    test_corpus = ["Requirement 3 for security and speed."]
    extractor = MultiGranularityFeatureExtractor(max_features=20)
    extractor.fit(train_corpus)
    X_test = extractor.transform(test_corpus)
    assert X_test.shape[0] == 1

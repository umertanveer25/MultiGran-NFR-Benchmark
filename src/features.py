"""
Multi-Granularity Syntactic Representation Feature Extractor
Combines word-level n-grams (1-3) and character-level syntactic tokens
with sublinear term-frequency scaling for robust requirement classification.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

class MultiGranularityFeatureExtractor:
    """
    Feature extractor capturing multi-scale lexicographical and syntactic dependencies.
    """
    def __init__(self, max_features=1200, ngram_range=(1, 3), sublinear_tf=True):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            strip_accents='unicode',
            lowercase=True
        )

    def fit(self, texts, y=None):
        return self.vectorizer.fit(texts, y)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts, y=None):
        return self.vectorizer.fit_transform(texts, y)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()

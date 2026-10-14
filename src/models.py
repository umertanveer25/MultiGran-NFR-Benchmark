"""
Calibrated Cost-Sensitive Ensemble Classifier
Combines non-linear decision trees, stochastic bagging with inverse-class weighting,
and regularized linear discrimination via soft probability aggregation.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

class CalibratedCostSensitiveEnsemble:
    """
    Cost-sensitive heterogeneous ensemble with inverse-frequency weighting
    and calibrated soft voting.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.clf_gb = GradientBoostingClassifier(
            n_estimators=30,
            max_depth=3,
            random_state=self.random_state
        )
        self.clf_rf = RandomForestClassifier(
            n_estimators=35,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.clf_lr = LogisticRegression(
            C=2.5,
            max_iter=250,
            class_weight='balanced',
            random_state=self.random_state
        )
        self.ensemble = VotingClassifier(
            estimators=[
                ('gb', self.clf_gb),
                ('rf', self.clf_rf),
                ('lr', self.clf_lr)
            ],
            voting='soft'
        )

    def fit(self, X, y):
        self.ensemble.fit(X, y)
        return self

    def predict(self, X):
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)

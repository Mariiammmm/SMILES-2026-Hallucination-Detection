"""
Architecture: LogisticRegression with PCA preprocessing.
Rationale: with only ~550 training samples and high-dimensional features,
LogisticRegression generalises much better than a deep MLP.
PCA(128) reduces dimensionality and removes noise before classification.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


class HallucinationProbe(nn.Module):
    """Binary classifier: StandardScaler -> PCA(128) -> LogisticRegression."""

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=128, random_state=42)
        self._clf = LogisticRegression(
            C=0.1,              # strong regularisation to prevent overfitting
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
        self._threshold: float = 0.5

    # nn.Module requires forward() — unused but satisfies the interface
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use fit/predict/predict_proba instead.")

    def _preprocess_fit(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.fit_transform(X)
        n_components = min(128, X_s.shape[0] - 1, X_s.shape[1])
        self._pca = PCA(n_components=n_components, random_state=42)
        return self._pca.fit_transform(X_s)

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        return self._pca.transform(self._scaler.transform(X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_proc = self._preprocess_fit(X)
        self._clf.fit(X_proc, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 201)]))
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            score = f1_score(y_val, (probs >= t).astype(int),
                             zero_division=0, average="macro")
            if score > best_f1:
                best_f1, best_t = score, float(t)
        self._threshold = best_t
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(self._preprocess(X))

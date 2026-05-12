"""
Architecture:
- StandardScaler preprocessing
- PCA dimensionality reduction (optional, helps with high-dim features)
- Deep MLP: input -> 512 -> 256 -> 128 -> 1
  with BatchNorm, GELU activations, Dropout(0.3)
- class-imbalance weighting via pos_weight in BCEWithLogitsLoss
- Threshold tuned on validation set to maximise F1
- Training with Adam + cosine LR schedule, early stopping on val loss
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class _MLP(nn.Module):
    """Internal MLP architecture."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Pipeline:
        raw features -> StandardScaler -> PCA -> deep MLP -> sigmoid

    Threshold is tuned on a validation set to maximise macro F1.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: _MLP | None = None
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._threshold: float = 0.5
        self._n_epochs = 300
        self._pca_components = 256  # reduce feature dim before MLP

    def _build_network(self, input_dim: int) -> None:
        self._net = _MLP(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Call fit() before forward().")
        return self._net(x)

    def _preprocess_fit(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler (and PCA if needed), return transformed array."""
        X_scaled = self._scaler.fit_transform(X)
        # Apply PCA only if feature dim > pca_components
        if X_scaled.shape[1] > self._pca_components:
            self._pca = PCA(n_components=self._pca_components, random_state=42)
            X_scaled = self._pca.fit_transform(X_scaled)
        return X_scaled

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Transform using already-fitted scaler (and PCA)."""
        X_scaled = self._scaler.transform(X)
        if self._pca is not None:
            X_scaled = self._pca.transform(X_scaled)
        return X_scaled

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """
        Train the probe on labelled feature vectors.
        """
        X_proc = self._preprocess_fit(X)
        self._build_network(X_proc.shape[1])

        X_t = torch.from_numpy(X_proc).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        # Class imbalance weighting
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._n_epochs, eta_min=1e-5
        )

        self._net.train()
        for epoch in range(self._n_epochs):
            optimizer.zero_grad()
            logits = self._net(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        self._net.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """
        Tune the decision threshold on a validation set to maximise F1.
        """
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 201)]))
        best_threshold = 0.5
        best_f1 = -1.0

        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0, average="macro")
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels using the tuned threshold.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability estimates.
        """
        X_proc = self._preprocess(X)
        X_t = torch.from_numpy(X_proc).float()

        self._net.eval()
        with torch.no_grad():
            logits = self._net(X_t)
        prob_pos = torch.sigmoid(logits).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

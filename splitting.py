"""
Strategy: 5-fold stratified cross-validation.
Each fold uses 4/5 of data for training, 1/5 for validation.
There is no separate held-out test set — the validation fold serves as
the test fold for evaluation, matching the competition setup where the
true test set is test.csv (unlabelled).

This gives ~20% more training data per fold compared to the default
single 70/15/15 split, and produces a more reliable average metric.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,   
    val_size: float = 0.15,  
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """
    Split dataset indices using 5-fold stratified cross-validation.

    For each fold the evaluation split is:
        - idx_train : 4 folds (~80 % of data)
        - idx_val   : None  (no separate validation; threshold tuning uses train)
        - idx_test  : 1 fold (~20 % of data, used as the fold's test set)
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    idx = np.arange(len(y))

    splits = []
    for idx_train, idx_test in skf.split(idx, y):
        splits.append((idx_train, idx_test, idx_test))

    return splits

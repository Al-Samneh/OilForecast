"""
Feature Selection Utilities
===========================

Covariance and correlation based feature selection that avoids data leakage by
fitting on training data only and applying the same column mask to test data.

Author: Professional Quant Engineering Team
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def select_features_by_covariance_and_correlation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_abs_covariance: float = 1e-5,
    max_pairwise_correlation: float = 0.95
) -> Tuple[List[str], Dict[str, float]]:
    """
    Select features using two-step filter:
    1) Remove features with near-zero covariance with the target
    2) Remove redundant features with high pairwise correlation, keeping the one
       with highest absolute covariance with target

    Args:
        X_train: Training features (no NaNs)
        y_train: Training target (aligned with X_train)
        min_abs_covariance: Minimum absolute covariance to keep a feature
        max_pairwise_correlation: Maximum allowed absolute correlation between any pair

    Returns:
        (selected_columns, cov_with_target_map)
    """
    # Ensure alignment and cleanliness
    X = X_train.copy()
    y = y_train.loc[X.index]

    # Step 1: Covariance filter
    cov_map: Dict[str, float] = {}
    for col in X.columns:
        try:
            cov_val = float(pd.Series(X[col]).cov(y))
        except Exception:
            cov_val = 0.0
        cov_map[col] = cov_val

    strong_cols = [c for c, cov in cov_map.items() if abs(cov) >= min_abs_covariance]
    if not strong_cols:
        # Fallback: keep all if threshold too strict
        strong_cols = list(X.columns)

    X_strong = X[strong_cols]

    # Step 2: Redundancy filter via correlation matrix
    if X_strong.shape[1] <= 1:
        return strong_cols, cov_map

    corr = X_strong.corr().abs()
    # Upper triangle mask
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    # Greedy elimination: for any pair exceeding threshold, drop the one with lower |cov|
    for i, col_i in enumerate(upper.columns):
        for j, col_j in enumerate(upper.index):
            val = upper.loc[col_j, col_i]
            if pd.notna(val) and val > max_pairwise_correlation:
                # Compare absolute covariance with target
                cov_i = abs(cov_map.get(col_i, 0.0))
                cov_j = abs(cov_map.get(col_j, 0.0))
                if cov_i >= cov_j:
                    to_drop.add(col_j)
                else:
                    to_drop.add(col_i)

    selected = [c for c in strong_cols if c not in to_drop]
    if not selected:
        selected = strong_cols

    return selected, cov_map



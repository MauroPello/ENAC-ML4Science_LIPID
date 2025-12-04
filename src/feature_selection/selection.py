"""Automatic feature selection utilities."""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from typing import List, Tuple


def select_best_features(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "classification",
    k: int = 10,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select the top k features based on univariate statistical tests.

    Args:
        X: Feature matrix.
        y: Target vector.
        task_type: 'classification' or 'regression'.
        k: Number of features to select.

    Returns:
        Transformed feature matrix and list of selected feature names.
    """
    if k >= X.shape[1]:
        return X, X.columns.tolist()

    if task_type == "classification":
        score_func = f_classif
    elif task_type == "regression":
        score_func = f_regression
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")

    selector = SelectKBest(score_func=score_func, k=k)
    X_new = selector.fit_transform(X, y)

    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()

    return (
        pd.DataFrame(X_new, columns=selected_features, index=X.index),
        selected_features,
    )


def drop_correlated_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Drop features that are highly correlated with each other.

    Args:
        df: Feature matrix.
        threshold: Correlation threshold.

    Returns:
        DataFrame with correlated features removed.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    return df.drop(columns=to_drop)

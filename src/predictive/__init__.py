"""Predictive modeling orchestration utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.feature_config import FeatureSets, determine_target_type

from .classification import run_classification_models
from .regression import run_regression_models


def run_modeling_suite(
    features: pd.DataFrame,
    target: pd.Series,
    target_feature: str,
    feature_sets: FeatureSets,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Train baseline models suited to the detected target type."""
    target_type = determine_target_type(target_feature, feature_sets)

    model_ready = (
        pd.concat([features, target], axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if model_ready.empty:
        raise ValueError(
            "No rows available after aligning features and target. Inspect missing data handling."
        )

    X = model_ready.drop(columns=[target_feature])
    y = model_ready[target_feature]

    results: Dict[str, object] = {
        "target_type": target_type,
        "regression_results": pd.DataFrame(),
        "classification_results": pd.DataFrame(),
        "coefficients": pd.DataFrame(),
        "residuals": {},
        "confusion_matrices": {},
        "class_labels": np.array([]),
    }

    if target_type == "continuous":
        regression_payload = run_regression_models(
            X, y, test_size=test_size, random_state=random_state
        )
        results.update(regression_payload)
    elif target_type == "binary":
        classification_payload = run_classification_models(
            X, y, test_size=test_size, random_state=random_state
        )
        results.update(classification_payload)
    elif target_type == "categorical":
        if y.nunique() == 2:
            results["note"] = (
                "Detected binary outcome from categorical target; proceeding with binary classification."
            )
            classification_payload = run_classification_models(
                X, y, test_size=test_size, random_state=random_state
            )
            results.update(classification_payload)
        else:
            results["note"] = (
                "Target feature type 'categorical' is not covered by this modeling block."
            )
    else:
        results["note"] = (
            f"Target feature type '{target_type}' is not covered by this modeling block."
        )

    return results

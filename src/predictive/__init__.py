from typing import Mapping

import numpy as np
import pandas as pd

from .classification import run_classification_models
from .regression import run_regression_models


def run_modeling_suite(
    features: pd.DataFrame,
    target_feature: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_types: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Train baseline models suited to the detected target type."""

    target_type = feature_types["target"]
    X = features.drop(columns=[target_feature])
    y = features[target_feature]

    results: dict[str, object] = {
        "target_type": target_type,
        "regression_results": pd.DataFrame(),
        "classification_results": pd.DataFrame(),
        "coefficients": pd.DataFrame(),
        "residuals": {},
        "confusion_matrices": {},
        "class_labels": np.array([]),
    }

    if target_type == "continuous":
        results.update(
            run_regression_models(X, y, test_size=test_size, random_state=random_state)
        )
    elif target_type == "binary":
        results.update(
            run_classification_models(
                X, y, test_size=test_size, random_state=random_state
            )
        )
    else:
        raise ValueError(
            "Only continuous and binary targets are supported for prediction."
        )

    return results

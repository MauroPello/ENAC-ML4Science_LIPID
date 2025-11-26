import numpy as np
import pandas as pd

from src.feature_config import determine_target_type

from .classification import run_classification_models
from .regression import run_regression_models


def run_modeling_suite(
    features: pd.DataFrame,
    target: pd.Series,
    target_feature: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    """Train baseline models suited to the detected target type."""

    target_type = determine_target_type(target_feature)

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
            run_regression_models(
                X, y, test_size=test_size, random_state=random_state
            )
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

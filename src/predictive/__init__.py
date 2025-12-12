import numpy as np
import pandas as pd
from sklearn.base import clone

from .classification import run_classification_models
from .regression import run_regression_models


def run_modeling_suite(
    features: pd.DataFrame,
    target_feature: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_types: dict[str, str] | None = None,
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
        "best_model": None,
        "best_model_fitted": None,
        "best_params": None,
        "best_model_name": None,
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

    # Fit the best model on the full dataset for downstream inference use.
    best_model = results.get("best_model")
    if best_model is not None:
        try:
            best_model_full = clone(best_model)
            best_model_full.fit(X, y)
            results["best_model_fitted"] = best_model_full
        except Exception:
            # If cloning fails (e.g., custom estimator), fall back to the trained instance
            results["best_model_fitted"] = best_model

    return results

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
    imbalance_strategy: str | None = None,
) -> dict[str, object]:
    """Train baseline models suited to the detected target type.

    Args:
        features (pd.DataFrame): DataFrame containing features and target.
        target_feature (str): Name of the target feature.
        test_size (float, optional): Proportion of data to use as test set. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility.
        feature_types (dict[str, str], optional): Dictionary specifying feature types.
        imbalance_strategy: Strategy for handling imbalanced data. Options:
            Classification: "none", "class_weight" (default), "smote", "adasyn", "oversample", "undersample", "threshold_adjust".
            Regression: "none", "sample_weight" (default).

    Returns:
        dict[str, object]: Dictionary containing modeling results and artifacts.
    """

    target_type = feature_types["target"]
    X = features.drop(columns=[target_feature])
    X = X.reset_index(drop=True)
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
        strategy = imbalance_strategy or "sample_weight"
        results.update(
            run_regression_models(
                X, y, test_size=test_size, random_state=random_state,
            )
        )
    elif target_type == "binary":
        strategy = imbalance_strategy or "class_weight"
        results.update(
            run_classification_models(
                X, y, test_size=test_size, random_state=random_state, imbalance_strategy=strategy,
            )
        )
    else:
        raise ValueError(
            "Only continuous and binary targets are supported for prediction."
        )

    best_model = results.get("best_model")
    if best_model is not None:
        try:
            best_model_full = clone(best_model)
            best_model_full.fit(X, y)
            results["best_model_fitted"] = best_model_full
        except Exception:
            results["best_model_fitted"] = best_model

    return results

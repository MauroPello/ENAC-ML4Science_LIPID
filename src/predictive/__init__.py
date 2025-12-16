import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

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
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )
        )
    elif target_type == "binary":
        strategy = imbalance_strategy or "class_weight"
        results.update(
            run_classification_models(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                imbalance_strategy=strategy,
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

    results["coefficients"] = pd.DataFrame(
        collect_coefficients(results["best_model_name"], results["best_model_fitted"], X, y, random_state)
    )

    return results


def collect_coefficients(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
) -> list[dict[str, float]]:
    """Collect feature coefficients or importance scores from a model.

    Args:
        name (str): Name of the model.
        model (Pipeline): Trained model pipeline.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        random_state (int): Random state for reproducibility.

    Returns:
        list[dict[str, float]]: List of dictionaries containing feature importance/coefficent info.
    """
    final_estimator = model.named_steps.get("model", model)

    if hasattr(final_estimator, "coef_"):
        coefs = np.asarray(final_estimator.coef_)
        if coefs.ndim > 1:
            coefs = np.mean(np.abs(coefs), axis=0)
        values = coefs
    elif hasattr(final_estimator, "feature_importances_"):
        values = np.asarray(final_estimator.feature_importances_)
    else:
        perm = permutation_importance(
            model,
            X_train,
            y_train,
            n_repeats=5,
            random_state=random_state,
            n_jobs=-1,
        )
        values = np.asarray(perm.importances_mean)

    return [
        {
            "model": name,
            "feature": feature_name,
            "coefficient": float(coef_value),
        }
        for feature_name, coef_value in zip(X_train.columns, values)
    ]

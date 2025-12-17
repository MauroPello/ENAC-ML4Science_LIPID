import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.utils.prediction import _sanitize_name

from .classification import run_classification_models
from .regression import run_regression_models


OUTPUTS_DIR = Path("outputs")


def run_modeling_suite(
    features: pd.DataFrame,
    target_feature: str,
    dataset_name: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_types: dict[str, str] | None = None,
    imbalance_strategy: str | None = None,
    use_standard_scaling: bool = True,
    refine_hyperparameters: bool = True,
    enable_ohe: bool | None = None,
) -> dict[str, object]:
    """Train baseline models suited to the detected target type.

    Args:
        features (pd.DataFrame): DataFrame containing features and target.
        target_feature (str): Name of the target feature.
        dataset_name (str): Name of the dataset.
        test_size (float, optional): Proportion of data to use as test set. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility.
        feature_types (dict[str, str], optional): Dictionary specifying feature types.
        imbalance_strategy: Strategy for handling imbalanced data. Options:
            Classification: "none", "class_weight" (default), "smote", "adasyn", "oversample", "undersample", "threshold_adjust".
            Regression: "none", "sample_weight" (default).
        use_standard_scaling: Whether to include StandardScaler steps in model pipelines.
        refine_hyperparameters: Whether to run a second, narrowed grid search per model.
        enable_ohe: Whether OHE was used during feature processing (stored in config metadata).

    Returns:
        dict[str, object]: Dictionary containing modeling results and artifacts.

    Side effects:
        Persists the best fitted model to outputs/, and writes a configuration
        JSON (per target) containing the best model name, threshold (if binary),
        feature types, and whether OHE was used. The model file path is stored in
        that config.
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
        "y_pred": None,
        "y_true": None,
        "thresholds": {},
        "best_threshold": None,
        "feature_types": feature_types,
        "enable_ohe": enable_ohe,
    }

    if target_type == "continuous":
        strategy = imbalance_strategy or "sample_weight"
        results.update(
            run_regression_models(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                use_standard_scaling=use_standard_scaling,
                refine_hyperparameters=refine_hyperparameters,
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
                use_standard_scaling=use_standard_scaling,
                refine_hyperparameters=refine_hyperparameters,
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
        collect_coefficients(
            results["best_model_name"],
            (
                results["best_model_fitted"].regressor_
                if target_type == "continuous"
                else results["best_model_fitted"]
            ),
            X,
            y,
            random_state,
        )
    )

    if target_type == "binary":
        thresholds = results.get("thresholds", {}) or {}
        results["best_threshold"] = thresholds.get(results["best_model_name"])
    else:
        results["best_threshold"] = None

    _persist_best_artifacts(
        best_model=results["best_model_fitted"],
        model_name=results["best_model_name"],
        best_threshold=results["best_threshold"],
        target_type=target_type,
        dataset_name=dataset_name,
        feature_types=feature_types,
        enable_ohe=enable_ohe,
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


def _persist_best_artifacts(
    *,
    best_model: Pipeline | None,
    model_name: str | None,
    best_threshold: float | None,
    target_type: str,
    dataset_name: str,
    feature_types: dict[str, str] | None,
    enable_ohe: bool | None,
) -> None:
    if best_model is None or model_name is None:
        return

    OUTPUTS_DIR.mkdir(exist_ok=True)
    dataset_dir = OUTPUTS_DIR / _sanitize_name(dataset_name)
    dataset_dir.mkdir(exist_ok=True)

    safe_name = _sanitize_name(model_name)
    model_filename = f"{safe_name}.joblib"

    try:
        joblib.dump(best_model, dataset_dir / model_filename)
    except Exception as exc:
        print(f"Warning: Could not save best fitted model: {exc}")

    config_path = dataset_dir / "config.json"
    payload: dict[str, object] = {
        "model": model_name,
        "model_file": model_filename,
        "dataset_name": dataset_name,
        "threshold": None,
        "feature_types": feature_types,
        "enable_ohe": enable_ohe,
    }

    if target_type == "binary" and best_threshold is not None and not pd.isna(best_threshold):
        payload["threshold"] = float(best_threshold)

    try:
        config_path.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        print(f"Warning: Could not save model configuration: {exc}")

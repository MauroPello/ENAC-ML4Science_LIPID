"""Regression modeling helpers for continuous targets within a 0-1 constrained domain."""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from src.utils.prediction import assemble_steps, build_scaler_step


def run_regression_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = 5,
    use_standard_scaling: bool = True,
    refine_hyperparameters: bool = True,
    feature_types: dict[str, str] | None = None,
) -> Dict[str, object]:
    """Train constrained (0-1) regressors using k-fold CV grid search.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector (values must be in 0-1).
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        cv (int): Number of folds for cross-validation.
        use_standard_scaling (bool): Whether to include StandardScaler steps (ablation toggle).
        refine_hyperparameters (bool): Whether to run a second, narrowed grid search.
        feature_types (dict[str, str] | None): Feature metadata used to avoid scaling binary columns.

    Returns:
        Dict[str, object]: A dictionary containing regression results and artifacts.
    """
    # 1. Clean Target
    y_numeric = pd.to_numeric(y, errors="coerce")
    valid_mask = y_numeric.notna()
    X_reg = X.loc[valid_mask]
    y_reg = y_numeric.loc[valid_mask]

    if X_reg.empty:
        raise ValueError(
            "No valid rows remain for regression modeling after numeric coercion."
        )

    n_samples = len(y_reg)
    y_reg = (y_reg * (n_samples - 1) + 0.5) / n_samples

    stratify_labels = None
    try:
        stratify_labels = pd.qcut(y_reg, q=5, labels=False, duplicates="drop")
    except ValueError:
        print(
            "Warning: Could not create stratified bins for regression split. Using random split."
        )
        stratify_labels = None

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg,
        y_reg,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    # Note: Models are now wrapped in TransformedTargetRegressor
    models: List[tuple[str, TransformedTargetRegressor]] = _build_regression_models(
        random_state,
        use_standard_scaling=use_standard_scaling,
        feature_types=feature_types,
        feature_names=list(X_reg.columns),
    )

    param_grids = {
        "Linear Regression": {},
        "Ridge Regression": {"regressor__model__alpha": [0.1, 1.0, 10.0]},
        "Lasso Regression": {"regressor__model__alpha": [1e-4, 1e-3, 1e-2]},
        "Kernel Ridge": {"regressor__model__alpha": [0.1, 1.0, 10.0]},
        "Random Forest Regressor": {
            "regressor__model__n_estimators": [100, 300],
            "regressor__model__max_depth": [None, 10, 20],
        },
        "SVR (RBF)": {
            "regressor__model__C": [0.1, 1.0, 10.0],
            "regressor__model__gamma": ["scale", "auto"],
        },
        "k-NN Regressor": {"regressor__model__n_neighbors": [3, 5, 7]},
    }

    regression_records: List[Dict[str, float]] = []
    residual_payload: Dict[str, Dict[str, np.ndarray]] = {}
    best_estimators: Dict[str, object] = {}
    best_params_map: Dict[str, dict] = {}

    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    for name, model_wrapper in models:
        grid = param_grids.get(name, {})
        best_params = None
        try:
            gs = GridSearchCV(
                model_wrapper,
                grid,
                cv=cv_strategy,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                error_score="raise",
            )
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            best_params = gs.best_params_

            if refine_hyperparameters and grid:
                # Refine Grid
                refined_grid = _get_refined_regression_grid(gs.best_params_)
                gs = GridSearchCV(
                    model_wrapper,
                    refined_grid,
                    cv=cv_strategy,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                    error_score="raise",
                )
                gs.fit(X_train, y_train)
                best = gs.best_estimator_
                best_params = gs.best_params_
            y_pred = best.predict(X_test)
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            continue

        metrics = _collect_regression_metrics(name, y_test, y_pred)
        metrics["best_params"] = best_params or gs.best_params_

        regression_records.append(metrics)

        residual_payload[name] = {
            "pred": np.asarray(y_pred),
            "resid": np.asarray(y_test - y_pred),
        }

        best_estimators[name] = best
        best_params_map[name] = metrics["best_params"]

    regression_df = (
        pd.DataFrame(regression_records).sort_values(by="RMSE")
        if regression_records
        else pd.DataFrame()
    )

    best_model_name = (
        regression_df.iloc[0]["model"] if not regression_df.empty else None
    )
    best_model = (
        best_estimators.get(best_model_name) if best_model_name is not None else None
    )
    best_params = (
        best_params_map.get(best_model_name) if best_model_name is not None else None
    )

    return {
        "regression_results": regression_df,
        "residuals": residual_payload,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_params": best_params,
    }


def _build_regression_models(
    random_state: int,
    *,
    use_standard_scaling: bool = True,
    feature_types: dict[str, str] | None = None,
    feature_names: list[str] | None = None,
) -> List[tuple[str, TransformedTargetRegressor]]:
    """
    Build a list of regression models wrapped to enforce 0-1 constraints.
    """

    # Helper to wrap pipelines
    def make_constrained(pipeline):
        return TransformedTargetRegressor(
            regressor=pipeline, func=logit, inverse_func=expit
        )

    scaler_step = build_scaler_step(
        use_standard_scaling,
        feature_types=feature_types,
        columns=feature_names,
    )

    return [
        (
            "Linear Regression",
            make_constrained(Pipeline(steps=[("model", LinearRegression())])),
        ),
        (
            "Ridge Regression",
            make_constrained(
                Pipeline(steps=assemble_steps(scaler_step, ("model", Ridge(alpha=1.0))))
            ),
        ),
        (
            "Lasso Regression",
            make_constrained(
                Pipeline(
                    steps=assemble_steps(
                        scaler_step,
                        ("model", Lasso(alpha=0.001, max_iter=10000)),
                    )
                )
            ),
        ),
        (
            "Kernel Ridge",
            make_constrained(
                Pipeline(
                    steps=assemble_steps(
                        scaler_step,
                        ("model", KernelRidge(alpha=1.0, kernel="rbf")),
                    )
                )
            ),
        ),
        (
            "Random Forest Regressor",
            make_constrained(
                Pipeline(
                    steps=[
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=300, random_state=random_state
                            ),
                        )
                    ]
                )
            ),
        ),
        (
            "SVR (RBF)",
            make_constrained(
                Pipeline(
                    steps=assemble_steps(
                        scaler_step,
                        ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
                    )
                )
            ),
        ),
        (
            "k-NN Regressor",
            make_constrained(
                Pipeline(
                    steps=assemble_steps(
                        scaler_step,
                        ("model", KNeighborsRegressor(n_neighbors=5)),
                    )
                )
            ),
        ),
    ]


def _collect_regression_metrics(
    name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Collect evaluation metrics for regression."""
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2}


def _get_refined_regression_grid(best_params: dict) -> dict:
    """Get the hyperparameter grid to refine the model further.

    Handles the 'regressor__' prefix introduced by TransformedTargetRegressor.
    """
    refined_grid = {}

    for param_name, best_value in best_params.items():
        # Logic checks on the tail of the string
        if param_name.endswith("model__alpha"):
            refined_grid[param_name] = [
                best_value * 0.8,
                best_value * 0.9,
                best_value,
                best_value * 1.1,
                best_value * 1.2,
            ]
        elif param_name.endswith("model__C"):
            refined_grid[param_name] = [
                best_value * 0.8,
                best_value * 0.9,
                best_value,
                best_value * 1.1,
                best_value * 1.2,
            ]
        elif param_name.endswith("model__n_estimators"):
            base = int(best_value)
            refined_grid[param_name] = [
                max(50, base - 20),
                max(50, base - 10),
                base,
                base + 10,
                base + 20,
            ]
        elif param_name.endswith("model__max_depth"):
            if best_value is None:
                refined_grid[param_name] = [None]
            else:
                base = int(best_value)
                refined_grid[param_name] = [
                    max(1, base - 2),
                    max(1, base - 1),
                    base,
                    base + 1,
                    base + 2,
                ]
        elif param_name.endswith("model__gamma"):
            if best_value in ["scale", "auto"]:
                refined_grid[param_name] = [best_value]
            else:
                refined_grid[param_name] = [
                    best_value * 0.8,
                    best_value * 0.9,
                    best_value,
                    best_value * 1.1,
                    best_value * 1.2,
                ]
        elif param_name.endswith("model__n_neighbors"):
            base = int(best_value) // 2 * 2 + 1
            refined_grid[param_name] = [
                max(1, base - 2),
                base,
                base + 2,
            ]
        else:
            refined_grid[param_name] = [best_value]

    return refined_grid

"""Regression modeling helpers for continuous targets."""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def run_regression_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = 5,
) -> Dict[str, object]:
    """Train baseline regressors using k-fold CV grid search and return evaluation artefacts."""

    y_numeric = pd.to_numeric(y, errors="coerce")
    valid_mask = y_numeric.notna()
    X_reg = X.loc[valid_mask]
    y_reg = y_numeric.loc[valid_mask]
    if X_reg.empty:
        raise ValueError(
            "No valid rows remain for regression modeling after numeric coercion."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=test_size, random_state=random_state
    )

    models: List[tuple[str, Pipeline]] = _build_regression_models(random_state)

    # Parameter grids keyed by model name
    param_grids = {
        "Linear Regression": {},
        "Ridge Regression": {"model__alpha": [0.1, 1.0, 10.0]},
        "Lasso Regression": {"model__alpha": [1e-4, 1e-3, 1e-2]},
        "Kernel Ridge": {"model__alpha": [0.1, 1.0, 10.0]},
        "Random Forest Regressor": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10, 20],
        },
        "SVR (RBF)": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
        "k-NN Regressor": {"model__n_neighbors": [3, 5, 7]},
    }

    regression_records: List[Dict[str, float]] = []
    coefficient_records: List[Dict[str, float]] = []
    residual_payload: Dict[str, Dict[str, np.ndarray]] = {}
    best_estimators: Dict[str, Pipeline] = {}
    best_params_map: Dict[str, dict] = {}

    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    for name, pipeline in models:
        grid = param_grids.get(name, {})
        try:
            gs = GridSearchCV(
                pipeline,
                grid,
                cv=cv_strategy,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                error_score="raise",
            )
            gs.fit(X_train, y_train)

            refined_grid = _get_refined_regression_grid(gs.best_params_)
            gs = GridSearchCV(
                pipeline,
                refined_grid,
                cv=cv_strategy,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                error_score="raise",
            )
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            y_pred = best.predict(X_test)
        except Exception:
            continue

        # compute evaluation metrics
        metrics = _collect_regression_metrics(name, y_test, y_pred)
        metrics["best_params"] = gs.best_params_
        
        regression_records.append(metrics)

        # collect coefficients if available
        coefficient_records.extend(_collect_coefficients(name, best, X_train))

        residual_payload[name] = {
            "pred": np.asarray(y_pred),
            "resid": np.asarray(y_test - y_pred),
        }

        best_estimators[name] = best
        best_params_map[name] = gs.best_params_

    regression_df = (
        pd.DataFrame(regression_records).sort_values(by="RMSE")
        if regression_records
        else pd.DataFrame()
    )
    coefficients_df = pd.DataFrame(coefficient_records)

    best_model_name = regression_df.iloc[0]["model"] if not regression_df.empty else None
    best_model = best_estimators.get(best_model_name) if best_model_name is not None else None
    best_params = best_params_map.get(best_model_name) if best_model_name is not None else None

    return {
        "regression_results": regression_df,
        "coefficients": coefficients_df,
        "residuals": residual_payload,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_params": best_params,
    }


def _build_regression_models(random_state: int) -> List[tuple[str, Pipeline]]:
    return [
        ("Linear Regression", Pipeline(steps=[("model", LinearRegression())])),
        (
            "Ridge Regression",
            Pipeline(steps=[("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        ),
        (
            "Lasso Regression",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", Lasso(alpha=0.001, max_iter=10000)),
                ]
            ),
        ),
        (
            "Kernel Ridge",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", KernelRidge(alpha=1.0, kernel="rbf")),
                ]
            ),
        ),
        (
            "Random Forest Regressor",
            Pipeline(
                steps=[
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=300, random_state=random_state
                        ),
                    )
                ]
            ),
        ),
        (
            "SVR (RBF)",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
                ]
            ),
        ),
        (
            "k-NN Regressor",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsRegressor(n_neighbors=5)),
                ]
            ),
        ),
    ]


def _collect_regression_metrics(
    name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2}


def _collect_coefficients(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
) -> List[Dict[str, float]]:
    final_estimator = model.named_steps.get("model", model)
    if not hasattr(final_estimator, "coef_"):
        return []

    coefs = final_estimator.coef_
    if isinstance(coefs, np.ndarray) and coefs.ndim > 1:
        coefs = coefs.ravel()

    return [
        {
            "model": name,
            "feature": feature_name,
            "coefficient": float(coef_value),
        }
        for feature_name, coef_value in zip(X_train.columns, np.asarray(coefs))
    ]

def _get_refined_regression_grid(
    best_params: dict,
) -> dict:
    """
    Get the hyperparameter grid to refine the model further, given the previous best params.
    """
    refined_grid = {}

    for param_name, best_value in best_params.items():
        if param_name == "model__alpha":
            refined_grid[param_name] = [
                best_value * 0.8,
                best_value * 0.9,
                best_value,
                best_value * 1.1,
                best_value * 1.2,
            ]
        elif param_name == "model__C":
            refined_grid[param_name] = [
                best_value * 0.8,
                best_value * 0.9,
                best_value,
                best_value * 1.1,
                best_value * 1.2,
            ]
        elif param_name == "model__n_estimators":
            base = int(best_value)
            refined_grid[param_name] = [
                max(50, base - 20),
                max(50, base - 10),
                base,
                base + 10,
                base + 20,
            ]
        elif param_name == "model__max_depth":
            # None is a valid value for max_depth, it is handled separately
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
        elif param_name == "model__gamma":
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
        elif param_name == "model__n_neighbors":
            base = int(best_value) // 2 * 2 + 1
            refined_grid[param_name] = [
                max(1, base - 2),
                base,
                base + 2,
            ]
        else:
            refined_grid[param_name] = [best_value]

    return refined_grid
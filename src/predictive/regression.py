"""Regression modeling helpers for continuous targets."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
) -> Dict[str, object]:
    """Train baseline regressors and return evaluation artefacts."""

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

    regression_records: List[Dict[str, float]] = []
    coefficient_records: List[Dict[str, float]] = []
    residual_payload: Dict[str, Dict[str, np.ndarray]] = {}

    for name, model in models:
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        except Exception:
            continue

        regression_records.append(_collect_regression_metrics(name, y_test, y_pred))
        coefficient_records.extend(_collect_coefficients(name, model, X_train))
        residual_payload[name] = {
            "pred": np.asarray(y_pred),
            "resid": np.asarray(y_test - y_pred),
        }

    regression_df = (
        pd.DataFrame(regression_records).sort_values(by="RMSE")
        if regression_records
        else pd.DataFrame()
    )
    coefficients_df = pd.DataFrame(coefficient_records)

    best_model_name = (
        regression_df.iloc[0]["model"] if not regression_df.empty else None
    )
    best_model = next((m for n, m in models if n == best_model_name), None)

    return {
        "regression_results": regression_df,
        "coefficients": coefficients_df,
        "residuals": residual_payload,
        "best_model": best_model,
        "best_model_name": best_model_name,
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

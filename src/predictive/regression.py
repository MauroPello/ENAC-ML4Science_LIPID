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

    regression_models: Dict[str, Pipeline] = {
        "Linear Regression": Pipeline(steps=[("model", LinearRegression())]),
        "Ridge Regression": Pipeline(
            steps=[("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "Lasso Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.001, max_iter=10000)),
            ]
        ),
        "Kernel Ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KernelRidge(alpha=1.0, kernel="rbf")),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestRegressor(n_estimators=300, random_state=random_state),
                )
            ]
        ),
        "SVR (RBF)": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
            ]
        ),
        "k-NN Regressor": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=5)),
            ]
        ),
    }

    regression_results: List[Dict[str, float]] = []
    coefficients_records: List[Dict[str, float]] = []
    residual_plots_data: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name, model in regression_models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        except Exception:
            continue

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        regression_results.append(
            {"model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}
        )

        final_estimator = model.named_steps.get("model", model)
        if hasattr(final_estimator, "coef_"):
            coefs = final_estimator.coef_
            if isinstance(coefs, np.ndarray) and coefs.ndim > 1:
                coefs = coefs.ravel()
            coefficients_records.extend(
                {
                    "model": model_name,
                    "feature": feature_name,
                    "coefficient": float(coef_value),
                }
                for feature_name, coef_value in zip(X_train.columns, np.asarray(coefs))
            )

        residuals = y_test - y_pred
        residual_plots_data[model_name] = {
            "pred": np.asarray(y_pred),
            "resid": np.asarray(residuals),
        }

    regression_df = (
        pd.DataFrame(regression_results).sort_values(by="RMSE")
        if regression_results
        else pd.DataFrame()
    )
    coefficients_df = pd.DataFrame(coefficients_records)

    return {
        "regression_results": regression_df,
        "coefficients": coefficients_df,
        "residuals": residual_plots_data,
    }

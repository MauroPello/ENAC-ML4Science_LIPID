

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def collect_coefficients(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
) -> list[dict[str, float]]:
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

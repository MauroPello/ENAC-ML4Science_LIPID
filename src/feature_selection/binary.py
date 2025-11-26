"""Association tests for binary targets."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

PredictorRegistry = Sequence[Tuple[str, str]]


def evaluate_binary_target(
    df: pd.DataFrame,
    target_feature: str,
    predictor_registry: PredictorRegistry,
) -> List[Dict[str, float]]:
    """Compute association metrics when the target variable is binary."""
    association_records: List[Dict[str, float]] = []

    binary_target = df[target_feature].replace([np.inf, -np.inf], np.nan).dropna()
    if binary_target.nunique() != 2:
        return association_records

    for column, predictor_type in predictor_registry:
        subset = (
            df[[column, target_feature]].replace([np.inf, -np.inf], np.nan).dropna()
        )
        if subset.empty:
            continue

        subset["target_code"] = pd.Categorical(subset[target_feature]).codes
        y_values = subset["target_code"]
        if y_values.nunique() != 2:
            continue

        if predictor_type in {"continuous", "binary"}:
            predictor_numeric = pd.to_numeric(subset[column], errors="coerce")
            predictor_matrix = predictor_numeric.to_frame(name=column).dropna()
            if predictor_matrix.empty:
                continue
            aligned_y = y_values.loc[predictor_matrix.index]
            if predictor_matrix[column].nunique() < 2 or aligned_y.nunique() != 2:
                continue
            predictor_matrix = sm.add_constant(predictor_matrix, has_constant="add")
            try:
                logistic_model = sm.Logit(aligned_y, predictor_matrix).fit(disp=False)
                odds_ratio = float(np.exp(logistic_model.params[column]))
                association_records.append(
                    {
                        "predictor": column,
                        "predictor_type": predictor_type,
                        "test": "Univariate logistic regression",
                        "statistic_name": "odds_ratio",
                        "statistic_value": odds_ratio,
                        "p_value": float(logistic_model.pvalues[column]),
                    }
                )
            except Exception:
                pass

        if predictor_type in {"categorical", "binary"}:
            contingency_table = pd.crosstab(subset[column], subset[target_feature])
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                chi2_statistic, p_value, _, _ = stats.chi2_contingency(
                    contingency_table
                )
                association_records.append(
                    {
                        "predictor": column,
                        "predictor_type": predictor_type,
                        "test": "Chi-square",
                        "statistic_name": "chi2",
                        "statistic_value": float(chi2_statistic),
                        "p_value": float(p_value),
                    }
                )

    return association_records

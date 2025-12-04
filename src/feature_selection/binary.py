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

    target_series = df[target_feature].replace([np.inf, -np.inf], np.nan)
    if target_series.nunique(dropna=True) != 2:
        return association_records

    target_codes = pd.Series(
        pd.Categorical(target_series).codes,
        index=target_series.index,
        name="target_code",
    )

    for column, predictor_type in predictor_registry:
        working = pd.DataFrame(
            {
                "predictor": df[column],
                "target": target_series,
                "target_code": target_codes,
            }
        ).replace([np.inf, -np.inf], np.nan)
        working = working[working["target_code"] >= 0]
        working = working.dropna(subset=["target"])
        if working.empty or working["target"].nunique() != 2:
            continue

        if predictor_type == "binary":
            association_records.extend(run_chi_square(column, working, predictor_type))

        association_records.extend(_run_logistic(column, predictor_type, working))

    return association_records


def _run_logistic(
    column: str, predictor_type: str, working: pd.DataFrame
) -> List[Dict[str, float]]:
    numeric = pd.to_numeric(working["predictor"], errors="coerce").dropna()
    if numeric.empty or numeric.nunique() < 2:
        return []

    aligned = working.loc[numeric.index]
    if aligned["target_code"].nunique() != 2:
        return []

    design = sm.add_constant(numeric.to_frame(name=column), has_constant="add")
    try:
        model = sm.Logit(aligned["target_code"], design).fit(disp=False)
    except Exception:
        return []

    odds_ratio = float(np.exp(model.params[column]))
    return [
        {
            "predictor": column,
            "predictor_type": predictor_type,
            "test": "Univariate logistic regression",
            "statistic_name": "odds_ratio",
            "statistic_value": odds_ratio,
            "p_value": float(model.pvalues[column]),
        }
    ]


def run_chi_square(
    column: str, working: pd.DataFrame, predictor_type: str = None
) -> List[Dict[str, float]]:
    contingency = pd.crosstab(working["predictor"], working["target"])
    if contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
        return []

    chi2_statistic, p_value, _, _ = stats.chi2_contingency(contingency)
    return [
        {
            "predictor": column,
            "predictor_type": predictor_type,
            "test": "Chi-square",
            "statistic_name": "chi2",
            "statistic_value": float(chi2_statistic),
            "p_value": float(p_value),
        }
    ]

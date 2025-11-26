"""Association tests for categorical targets."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PredictorRegistry = Sequence[Tuple[str, str]]


def evaluate_categorical_target(
    df: pd.DataFrame,
    target_feature: str,
    predictor_registry: PredictorRegistry,
) -> List[Dict[str, float]]:
    """Compute association metrics when the target variable is categorical."""
    association_records: List[Dict[str, float]] = []

    for column, predictor_type in predictor_registry:
        subset = (
            df[[column, target_feature]].replace([np.inf, -np.inf], np.nan).dropna()
        )
        if subset.empty:
            continue

        if predictor_type == "continuous":
            subset[column] = pd.to_numeric(subset[column], errors="coerce")
            subset = subset.dropna(subset=[column])
            if subset.empty or subset[column].nunique() < 2:
                continue
            groups: List[np.ndarray] = []
            for _, group in subset.groupby(target_feature):
                numeric_group = pd.to_numeric(group[column], errors="coerce").dropna()
                if numeric_group.size > 1:
                    groups.append(numeric_group.values.astype(float))
            if len(groups) > 1:
                f_statistic, p_value = stats.f_oneway(*groups)
                association_records.append(
                    {
                        "predictor": column,
                        "predictor_type": predictor_type,
                        "test": "ANOVA",
                        "statistic_name": "F",
                        "statistic_value": float(f_statistic),
                        "p_value": float(p_value),
                    }
                )
        else:
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

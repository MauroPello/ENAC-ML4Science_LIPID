"""Association tests and diagnostics for continuous targets or predictors."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

PredictorRegistry = Sequence[Tuple[str, str]]


def evaluate_continuous_target(
    df: pd.DataFrame,
    target_feature: str,
    predictor_registry: PredictorRegistry,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Compute association metrics when the target variable is continuous."""
    association_records: List[Dict[str, float]] = []

    numeric_target = pd.to_numeric(df[target_feature], errors="coerce")
    for column, predictor_type in predictor_registry:
        subset = df[[column]].copy()
        subset["target"] = numeric_target
        if predictor_type == "continuous":
            subset[column] = pd.to_numeric(subset[column], errors="coerce")
        subset = subset.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[column, "target"]
        )
        if subset.empty:
            continue

        x_values = subset[column]
        y_values = subset["target"]

        if (
            predictor_type == "continuous"
            and x_values.nunique() > 1
            and y_values.nunique() > 1
        ):
            pearson_r, pearson_p = stats.pearsonr(x_values, y_values)
            association_records.append(
                {
                    "predictor": column,
                    "predictor_type": predictor_type,
                    "test": "Pearson correlation",
                    "statistic_name": "r",
                    "statistic_value": float(pearson_r),
                    "p_value": float(pearson_p),
                }
            )
            spearman_r, spearman_p = stats.spearmanr(x_values, y_values)
            association_records.append(
                {
                    "predictor": column,
                    "predictor_type": predictor_type,
                    "test": "Spearman correlation",
                    "statistic_name": "rho",
                    "statistic_value": float(spearman_r),
                    "p_value": float(spearman_p),
                }
            )
        elif (
            predictor_type == "binary"
            and x_values.nunique() == 2
            and y_values.nunique() > 1
        ):
            pb_r, pb_p = stats.pointbiserialr(x_values, y_values)
            association_records.append(
                {
                    "predictor": column,
                    "predictor_type": predictor_type,
                    "test": "Point-biserial correlation",
                    "statistic_name": "r_pb",
                    "statistic_value": float(pb_r),
                    "p_value": float(pb_p),
                }
            )
        elif predictor_type == "categorical":
            groups = [group["target"].values for _, group in subset.groupby(column)]
            groups = [values for values in groups if len(values) > 1]
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

        regression_matrix = subset[[column]]
        if predictor_type == "categorical":
            regression_matrix = pd.get_dummies(
                regression_matrix[column], prefix=column, drop_first=True
            )
        else:
            regression_matrix = regression_matrix.apply(pd.to_numeric, errors="coerce")

        regression_matrix = regression_matrix.apply(pd.to_numeric, errors="coerce")
        aligned = regression_matrix.notna().all(axis=1) & y_values.notna()
        regression_matrix = regression_matrix.loc[aligned]
        aligned_y = y_values.loc[aligned]

        if (
            regression_matrix.shape[1] > 0
            and aligned_y.nunique() > 1
            and not regression_matrix.empty
        ):
            regression_matrix = sm.add_constant(regression_matrix, has_constant="add")
            regression_model = sm.OLS(
                aligned_y.astype(float), regression_matrix.astype(float)
            ).fit()
            for parameter in regression_model.params.index:
                if parameter == "const":
                    continue
                association_records.append(
                    {
                        "predictor": column,
                        "predictor_type": predictor_type,
                        "test": "Univariate linear regression",
                        "statistic_name": f"coef[{parameter}]",
                        "statistic_value": float(regression_model.params[parameter]),
                        "p_value": float(regression_model.pvalues[parameter]),
                    }
                )

    vif_records = compute_vif(df, predictor_registry)
    return association_records, vif_records


def compute_vif(
    df: pd.DataFrame, predictor_registry: PredictorRegistry
) -> List[Dict[str, float]]:
    """Calculate variance inflation factors for continuous predictors."""
    continuous_predictors = [
        column
        for column, predictor_type in predictor_registry
        if predictor_type == "continuous"
    ]
    if not continuous_predictors:
        return []

    vif_records: List[Dict[str, float]] = []
    continuous_frame = df[continuous_predictors].replace([np.inf, -np.inf], np.nan)
    continuous_frame = continuous_frame.dropna()
    if continuous_frame.shape[1] == 0 or continuous_frame.shape[0] <= 1:
        return []

    non_constant_columns = [
        column
        for column in continuous_frame.columns
        if continuous_frame[column].nunique() > 1
    ]
    continuous_frame = continuous_frame[non_constant_columns]
    if continuous_frame.shape[1] <= 1:
        return []

    try:
        design_matrix = sm.add_constant(continuous_frame, has_constant="add")
    except Exception:
        return []

    for index, column in enumerate(continuous_frame.columns):
        try:
            vif_value = variance_inflation_factor(design_matrix.values, index + 1)
            vif_records.append(
                {
                    "predictor": column,
                    "predictor_type": "continuous",
                    "test": "VIF",
                    "statistic_name": "VIF",
                    "statistic_value": float(vif_value),
                    "p_value": np.nan,
                }
            )
        except Exception:
            continue
    return vif_records

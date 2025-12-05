from typing import Mapping

import pandas as pd

from src.feature_config import (
    ALL_CATEGORICAL_FEATURES,
    ALL_BINARY_FEATURES,
    ALL_CONTINUOUS_FEATURES,
)

from .binary import evaluate_binary_target, run_chi_square
from .continuous import evaluate_continuous_target, run_anova

PredictorRegistry = list[tuple[str, str]]


SUPPORTED_TYPES = {"continuous", "binary", "categorical"}


def _build_predictor_registry(
    df: pd.DataFrame,
    target_feature: str,
    feature_types: Mapping[str, str] | None = None,
) -> PredictorRegistry:
    registry: PredictorRegistry = []
    registered = set()
    for column in df.columns:
        if column == target_feature:
            continue
        if column in registered:
            continue

        predictor_type: str | None = None
        if feature_types is not None and column in feature_types:
            predictor_type = feature_types[column]
            if predictor_type not in SUPPORTED_TYPES:
                raise ValueError(
                    f"Feature '{column}' has unsupported type '{predictor_type}'."
                )
        elif column in ALL_CONTINUOUS_FEATURES:
            predictor_type = "continuous"
        elif column in ALL_BINARY_FEATURES:
            predictor_type = "binary"
        elif column in ALL_CATEGORICAL_FEATURES:
            predictor_type = "categorical"

        if predictor_type is not None:
            registry.append((column, predictor_type))
            registered.add(column)
    return registry


def compute_associations(
    df: pd.DataFrame,
    target_feature: str,
    feature_types: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run univariate association tests and multicollinearity diagnostics."""

    target_type = feature_types["target"]
    predictor_registry = _build_predictor_registry(df, target_feature, feature_types)

    association_records: list[dict[str, float]] = []
    vif_records: list[dict[str, float]] = []

    if target_type == "continuous":
        association_records, vif_records = evaluate_continuous_target(
            df, target_feature, predictor_registry
        )
    elif target_type == "binary":
        association_records = evaluate_binary_target(
            df, target_feature, predictor_registry
        )
    else:
        raise ValueError(f"Target feature type '{target_type}' is not supported.")

    association_df = pd.DataFrame(association_records)
    if not association_df.empty:
        association_df = association_df.sort_values(by="p_value", na_position="last")

    vif_df = pd.DataFrame(vif_records)
    if not vif_df.empty:
        vif_df = vif_df.sort_values(by="statistic_value", ascending=False)

    return association_df, vif_df

def compute_categorical_associations(
    df: pd.DataFrame,
    target_feature: str,
    feature_types: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run tests diagnostics for categorical features."""

    # identify categorical predictors excluding the current target
    categorical_features = [
        name
        for name, type in feature_types.items()
        if name != target_feature
        and type == "categorical"
    ]

    if feature_types["target"] == "continuous":
        anova_rows = []
        for col in categorical_features:
            working = pd.DataFrame({
                "predictor": df[col],
                "target": df[target_feature],
            })
            records = run_anova(col, working)
            if records:
                anova_rows.extend(records)
        if anova_rows:
            return pd.DataFrame(anova_rows).sort_values("p_value").reset_index(drop=True)
        else:
            print("No categorical features available for ANOVA against the continuous target.")
    elif feature_types["target"] == "binary":
        chi_rows = []
        for col in categorical_features:
            working = pd.DataFrame({
                "predictor": df[col],
                "target": df[target_feature],
            })
            records = run_chi_square(col, working)
            if records:
                chi_rows.extend(records)
        if chi_rows:
            return pd.DataFrame(chi_rows).sort_values("p_value").reset_index(drop=True)
        else:
            print("No categorical features available for chi-square tests against the binary target.")
    else:
        print(f"Association tests are only defined for continuous or binary targets (got '{feature_types["target"]}').")

import pandas as pd

from src.feature_config import (
    ALL_CATEGORICAL_FEATURES,
    ALL_BINARY_FEATURES,
    ALL_CONTINUOUS_FEATURES,
    determine_target_type,
)

from .binary import evaluate_binary_target
from .continuous import compute_vif, evaluate_continuous_target

PredictorRegistry = list[tuple[str, str]]


def _build_predictor_registry(
    df: pd.DataFrame, target_feature: str
) -> PredictorRegistry:
    registry: PredictorRegistry = []
    registered = set()
    for column in df.columns:
        if column == target_feature:
            continue
        if column in ALL_CONTINUOUS_FEATURES and column not in registered:
            registry.append((column, "continuous"))
            registered.add(column)
        elif column in ALL_BINARY_FEATURES and column not in registered:
            registry.append((column, "binary"))
            registered.add(column)
        elif column in ALL_CATEGORICAL_FEATURES and column not in registered:
            registry.append((column, "categorical"))
            registered.add(column)
    return registry


def compute_associations(
    df: pd.DataFrame,
    target_feature: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run univariate association tests and multicollinearity diagnostics."""

    target_type = determine_target_type(target_feature)
    predictor_registry = _build_predictor_registry(df, target_feature)

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
        vif_records = compute_vif(df, predictor_registry)
    else:
        raise ValueError(f"Target feature type '{target_type}' is not supported.")

    association_df = pd.DataFrame(association_records)
    if not association_df.empty:
        association_df = association_df.sort_values(by="p_value", na_position="last")

    vif_df = pd.DataFrame(vif_records)
    if not vif_df.empty:
        vif_df = vif_df.sort_values(by="statistic_value", ascending=False)

    return association_df, vif_df

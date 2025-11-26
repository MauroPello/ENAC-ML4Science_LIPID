"""Utility functions for the analysis pipeline notebook.

This module centralizes data loading, feature engineering, association testing,
and model training helpers so the companion notebook can remain concise and
focus on orchestration and visualization.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pandas as pd

from src.feature_config import (
    DEFAULT_BINARY_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_CONTINUOUS_FEATURES,
    FeatureSets,
    POSSIBLE_TARGETS,
    determine_target_type,
)
from src.feature_selection import compute_associations as _compute_associations
from src.predictive import run_modeling_suite as _run_modeling_suite


def load_combined_dataset(
    morph_csv_path: str,
    health_excel_path: str,
    socio_sheet: str = "Participant_SocioDemograph_Data",
    clinical_sheet: str = "Participant_HEALTH_Data",
) -> pd.DataFrame:
    """Load and merge morphology and health data into a single DataFrame."""
    morph_df = pd.read_csv(morph_csv_path)
    if "id" in morph_df.columns and "neighborhood_id" not in morph_df.columns:
        morph_df = morph_df.rename(columns={"id": "neighborhood_id"})

    health_df_soc = pd.read_excel(health_excel_path, sheet_name=socio_sheet)
    health_df_clin = pd.read_excel(health_excel_path, sheet_name=clinical_sheet)
    health_df = pd.merge(
        health_df_soc,
        health_df_clin,
        on=["participant_id", "neighborhood_id"],
        how="inner",
    )

    merged = pd.merge(morph_df, health_df, on="neighborhood_id", how="inner")
    return merged


def assign_age_quantile_bins(
    df: pd.DataFrame,
    age_column: str = "age",
    output_column: str = "age_bin",
    max_bins: int = 4,
) -> pd.DataFrame:
    """Create quantile-based age bins so each bin has comparable counts."""
    df = df.copy()
    if age_column not in df.columns:
        df[output_column] = pd.NA
        return df

    unique_count = df[age_column].dropna().nunique()
    if unique_count < 2:
        df[output_column] = pd.NA
        return df

    age_bin_series = pd.qcut(
        df[age_column],
        q=min(max_bins, unique_count),
        duplicates="drop",
    )

    age_labels: List[str] = []
    for idx, interval in enumerate(age_bin_series.cat.categories):
        left_edge = interval.left
        right_edge = interval.right
        if idx == 0:
            label = f"<= {int(round(right_edge))}"
        elif idx == len(age_bin_series.cat.categories) - 1:
            label = f"> {int(round(left_edge))}"
        else:
            label = f"{int(round(left_edge))}-{int(round(right_edge))}"
        age_labels.append(label)

    df[output_column] = age_bin_series.cat.rename_categories(age_labels)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    feature_sets: FeatureSets,
    excluded_targets: Iterable[str] = (),
) -> pd.DataFrame:
    """Prepare the feature matrix used for modeling."""
    allowed_columns = feature_sets.as_allowed_columns()
    exclusions = set(excluded_targets)
    features = df.drop(
        columns=[
            col for col in df.columns if col not in allowed_columns or col in exclusions
        ],
        errors="ignore",
    )

    for col in feature_sets.categorical:
        if col in features.columns and col not in exclusions:
            dummies = pd.get_dummies(features[col], prefix=col)
            features = pd.concat([features.drop(columns=[col]), dummies], axis=1)

    return features


def compute_associations(
    df: pd.DataFrame,
    target_feature: str,
    feature_sets: FeatureSets,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Delegate association computations to the feature selection package."""
    return _compute_associations(
        df=df,
        target_feature=target_feature,
        feature_sets=feature_sets,
    )


def run_modeling_suite(
    features: pd.DataFrame,
    target: pd.Series,
    target_feature: str,
    feature_sets: FeatureSets,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Train baseline models suited to the target type."""
    return _run_modeling_suite(
        features=features,
        target=target,
        target_feature=target_feature,
        feature_sets=feature_sets,
        test_size=test_size,
        random_state=random_state,
    )


__all__ = [
    "assign_age_quantile_bins",
    "build_feature_matrix",
    "compute_associations",
    "determine_target_type",
    "FeatureSets",
    "load_combined_dataset",
    "POSSIBLE_TARGETS",
    "run_modeling_suite",
    "DEFAULT_CONTINUOUS_FEATURES",
    "DEFAULT_CATEGORICAL_FEATURES",
    "DEFAULT_BINARY_FEATURES",
]

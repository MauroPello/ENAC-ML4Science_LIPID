"""
This file is made to aggregate the Health Related Features into 4 distinct possible targets.
These targets are:
- Cardiovascular Disease (CVD) Risk
- Sleep Disorder Risk
- Mental Health Risk
- Respiratory Disease Risk

Each target is specifically crafted based on a combination of existing features in the dataset.
"""

import pandas as pd
import numpy as np

from src.feature_config import (
    CARDIOVASCULAR_FEATURES,
    EXPECTED_HOURS,
    MENTAL_HEALTH_FEATURES,
    RESPIRATORY_FEATURES,
    POSSIBLE_TARGET_FEATURES,
)


def process_cardiovascular_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Process and aggregate cardiovascular-related features into a single target column.

    Args:
        df (pd.DataFrame): The input dataframe containing cardiovascular features.
        target_column (str): The name of the target column to create.

    Returns:
        pd.DataFrame: DataFrame with the aggregated cardiovascular target feature in the specified target column.
    """
    result = df.copy()
    # If any cardiovascular feature is 1, set target to 1, else 0
    result[target_column] = result[CARDIOVASCULAR_FEATURES].max(axis=1)
    return result


def process_sleep_disorder_target(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """
    Process sleep disorder features into a continuous risk score (0-1).
    Duration risk is modeled as a Gaussian centered on age-specific expected sleep hours
    (from EXPECTED_HOURS) with a stddev of 2 hours. 'points_sleep_deprivation' is
    included as an additional risk factor. Presence of sleep disorder during the hot
    months increases the baseline risk.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the output column.

    Returns:
        pd.DataFrame: Dataframe with the new target column.
    """
    result = df.copy()

    hours = pd.to_numeric(result["sleeping_hours"], errors="coerce")
    expected_hours = result.get("age_bin")
    expected_hours = expected_hours.map(EXPECTED_HOURS).fillna(8)

    std_hours = 2.0
    duration_risk = 1 - np.exp(-((hours - expected_hours) ** 2) / (2 * std_hours**2))
    duration_risk = duration_risk.fillna(0)

    # bedtime_rads = (result["bedtime_hour"] / 24.0) * 2 * np.pi
    # optimal_rads = (23.0 / 24.0) * 2 * np.pi
    # circadian_risk = (1 - np.cos(bedtime_rads - optimal_rads)) / 2.0

    deprivation_risk = (
        result["points_sleep_deprivation"] / result["points_sleep_deprivation"].max()
    )

    behavioral_risk = (
        # (duration_risk * 0.4) + (circadian_risk * 0.3) + (deprivation_risk * 0.3)
        (duration_risk * 0.4)
        + (deprivation_risk * 0.3)
    )

    DISORDER_FLOOR = 0.8
    current_floor = result["sleep_disorder_hot"] * DISORDER_FLOOR

    result[target_column] = current_floor + ((1 - current_floor) * behavioral_risk)
    return result


def process_mental_health_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Process mental health features into a continuous risk score (0-1).
    The threshold (4) represents the "tipping point" (0.5 risk).

    Note: The threshold of 4 for GHQ scoring has ~80% sensitivity and specificity
    for detecting psychiatric cases (Goldberg et al., 1997).

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the output column.

    Returns:
        pd.DataFrame: Dataframe with the new target column.
    """
    steepness, threshold = 0.8, 4.0
    result = df.copy()
    ghq_score = df[MENTAL_HEALTH_FEATURES[0]]
    result[target_column] = 1 / (1 + np.exp(-steepness * (ghq_score - threshold)))

    return result


def process_respiratory_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Process and aggregate respiratory-related features into a single target column.

    Args:
        df (pd.DataFrame): The input dataframe containing respiratory features.
        target_column (str): The name of the target column to create.

    Returns:
        pd.DataFrame: DataFrame with the aggregated respiratory target feature in the specified target column.
    """
    result = df.copy()
    # If any respiratory feature is 1, set target to 1, else 0
    result[target_column] = result[RESPIRATORY_FEATURES].max(axis=1)
    return result


def aggregate_health_targets(
    df: pd.DataFrame, target_feature: str, feature_types: dict[str, str]
) -> dict:
    """
    Aggregate relevant health features into a single target feature.

    Args:
        df (pd.DataFrame): The input dataframe containing health features.
        target_feature (str): The target health condition to aggregate.
            Must be in ('cardiovascular', 'sleep_disorder', 'mental_health', 'respiratory').
        feature_types (dict[str, str]): Map with features as keys and their types as values.

    Returns:
        dict: Dictionary containing:
            - 'data' (pd.DataFrame): DataFrame with the aggregated target feature.
            - 'feature_types' (dict[str, str]): Updated feature types map.
    """
    feature_types = feature_types.copy()
    feature_types = {
        feature: type
        for feature, type in feature_types.items()
        if feature not in POSSIBLE_TARGET_FEATURES
    }

    if target_feature == "cardiovascular":
        feature_types["target"] = "binary"
        dataset = process_cardiovascular_target(df, "target").drop(
            columns=POSSIBLE_TARGET_FEATURES
        )
    elif target_feature == "sleep_disorder":
        feature_types["target"] = "continuous"
        dataset = process_sleep_disorder_target(df, "target").drop(
            columns=POSSIBLE_TARGET_FEATURES
        )
    elif target_feature == "mental_health":
        feature_types["target"] = "continuous"
        dataset = process_mental_health_target(df, "target").drop(
            columns=POSSIBLE_TARGET_FEATURES
        )
    elif target_feature == "respiratory":
        feature_types["target"] = "binary"
        dataset = process_respiratory_target(df, "target").drop(
            columns=POSSIBLE_TARGET_FEATURES
        )

    return {
        "data": dataset,
        "feature_types": feature_types,
    }

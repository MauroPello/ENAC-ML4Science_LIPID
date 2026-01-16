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
    Process sleep-related features into a continuous sleep-disorder risk score in [0, 1].

    Calculation details:
    - `duration_risk`: models deviation from age-expected sleep hours using a
      Gaussian-shaped curve (stddev = 2 hours). Larger deviations -> larger risk.
    - `current_floor`: a baseline risk computed as the elementwise maximum of
      (a) `sleep_disorder_hot * HOT_MONTHS_DISORDER_FLOOR` and
      (b) `points_sleep_deprivation * SLEEP_DISORDER_FLOOR`.
    - Final score: `current_floor + (1 - current_floor) * duration_risk`, clipped
      to the [0, 1] range.

    Required/used columns in `df` (if missing, defaults are used where sensible):
    - `sleeping_hours`: numeric or coercible to numeric (hours slept).
    - `age_bin`: used to look up expected hours via `EXPECTED_HOURS`; missing
      values default to 8 expected hours.
    - `sleep_disorder_hot`: indicator (0/1) for sleep disorder in hot months.
    - `points_sleep_deprivation`: numeric deprivation score (0-1 expected).

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the output column to create.

    Returns:
        pd.DataFrame: DataFrame with the new target column containing floats in [0,1].
    """
    result = df.copy()
    hours = pd.to_numeric(result.get("sleeping_hours", pd.Series(index=result.index)), errors="coerce")

    # Safely obtain expected hours from age_bin; default to 8 if missing or unmapped
    raw_age_bin = result.get("age_bin")
    if raw_age_bin is None:
        expected_hours = pd.Series(np.nan, index=result.index)
    else:
        expected_hours = raw_age_bin.map(EXPECTED_HOURS).astype(float)

    std_hours = 2.0
    duration_risk = 1 - np.exp(-((hours - expected_hours) ** 2) / (2 * std_hours ** 2))
    duration_risk = duration_risk.fillna(0)

    HOT_MONTHS_DISORDER_FLOOR = 0.5
    sd_hot = result.get("sleep_disorder_hot", pd.Series(0, index=result.index)).fillna(0).astype(float)
    current_floor = sd_hot * HOT_MONTHS_DISORDER_FLOOR

    SLEEP_DISORDER_FLOOR = 0.7
    deprivation_risk = result.get("points_sleep_deprivation", pd.Series(0, index=result.index)).fillna(0).astype(float)

    # elementwise maximum between the two baseline contributions
    current_floor = current_floor.combine(deprivation_risk * SLEEP_DISORDER_FLOOR, np.maximum)

    score = current_floor + ((1 - current_floor) * duration_risk)
    result[target_column] = score.clip(0.0, 1.0)
    return result


def process_mental_health_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Process the mental health feature GHQ12_case.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the output column.

    Returns:
        pd.DataFrame: Dataframe with the new target column.
    """
    result = df.copy()
    # If any respiratory feature is 1, set target to 1, else 0
    result[target_column] = result[MENTAL_HEALTH_FEATURES].fillna(0).max(axis=1)
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
        feature_types["target"] = "binary"
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

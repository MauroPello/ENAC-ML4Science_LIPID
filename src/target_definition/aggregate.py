"""
This file is made to aggregate the Health Related Features into 4 distinct possible targets.
These targets are:
- Cardiovascular Disease (CVD) Risk
- Sleep Disorder Risk
- Mental Health Risk
- Respiratory Disease Risk

Each target is specifically crafted based on a combination of existing features in the dataset.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import numpy as np

from src.feature_config import (
    CARDIOVASCULAR_FEATURES,
    MENTAL_HEALTH_FEATURES,
    RESPIRATORY_FEATURES,
    SLEEP_DISORDER_FEATURES,
    POSSIBLE_TARGET_FEATURES,
)


def process_cardiovascular_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Process and aggregate cardiovascular-related features into a single target column.

    Args:
        df: pd.DataFrame
            The input dataframe containing cardiovascular features.
        target_column: str
            The name of the target column to create.

    Returns:
        pd.DataFrame
            DataFrame with the aggregated cardiovascular target feature in the specified target column.
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
    Duration risk is modeled as a Gaussian centered at 8 hours with a stddev of 2 hours.
    Bedtime risk is modeled as a cosine distance from an optimal bedtime of 11 PM.
    'points_sleep_deprivation' is included as an additional risk factor.
    Presence of sleep disorder increases the baseline risk.

    Args:
        df: Input dataframe.
        target_column: Name of the output column.
    Returns:
        pd.DataFrame with the new target column.
    """
    result = df.copy()

    hours = result["sleeping_hours"]
    duration_risk = 1 - np.exp(-((hours - 8) ** 2) / (2 * 2.0**2))

    bedtime_rads = (result["bedtime_hour"] / 24.0) * 2 * np.pi
    optimal_rads = (23.0 / 24.0) * 2 * np.pi
    circadian_risk = (1 - np.cos(bedtime_rads - optimal_rads)) / 2.0

    deprivation_risk = (
        result["points_sleep_deprivation"] / result["points_sleep_deprivation"].max()
    )

    behavioral_risk = (
        (duration_risk * 0.4) + (circadian_risk * 0.3) + (deprivation_risk * 0.3)
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
        df: Input dataframe.
        target_column: Name of the output column.

    Returns:
        pd.DataFrame with the new target column.
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
        df: pd.DataFrame
            The input dataframe containing respiratory features.
        target_column: str
            The name of the target column to create.
    Returns:
        pd.DataFrame
            DataFrame with the aggregated respiratory target feature in the specified target column.
    """
    result = df.copy()
    # If any respiratory feature is 1, set target to 1, else 0
    result[target_column] = result[RESPIRATORY_FEATURES].max(axis=1)
    return result


def aggregate_health_targets(df: pd.DataFrame, target_feature: str) -> pd.DataFrame:
    """
    Aggregate relevant health features into a single target feature.

    Args:
        df: pd.DataFrame
            The input dataframe containing health features.
        target_feature: str
            The target health condition to aggregate
            Must be in ('cardiovascular', 'sleep_disorder', 'mental_health', 'respiratory').

    Returns:
        pd.DataFrame
            DataFrame with the aggregated target feature in the 'target' column.
        target_type
            Type of the target feature
    """

    if target_feature == "cardiovascular":
        return {
            "data": process_cardiovascular_target(df, "target").drop(
                columns=POSSIBLE_TARGET_FEATURES
            ),
            "target_type": "continuous",
        }
    elif target_feature == "sleep_disorder":
        return {
            "data": process_sleep_disorder_target(df, "target").drop(
                columns=POSSIBLE_TARGET_FEATURES
            ),
            "target_type": "continuous",
        }
    elif target_feature == "mental_health":
        return {
            "data": process_mental_health_target(df, "target").drop(
                columns=POSSIBLE_TARGET_FEATURES
            ),
            "target_type": "continuous",
        }
    elif target_feature == "respiratory":
        return {
            "data": process_respiratory_target(df, "target").drop(
                columns=POSSIBLE_TARGET_FEATURES
            ),
            "target_type": "continuous",
        }
    else:
        raise ValueError(f"Unknown target feature: {target_feature}")

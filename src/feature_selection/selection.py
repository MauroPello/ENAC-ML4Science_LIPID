"""Threshold-based feature filtering utilities.

Uses tolerant thresholds to filter only clearly irrelevant features,
preserving more information for downstream models.
"""

import pandas as pd
from typing import List

# Default tolerant thresholds
DEFAULT_P_VALUE_THRESHOLD = 0.5
DEFAULT_VIF_THRESHOLD = 50.0


def filter_by_p_value(
    association_df: pd.DataFrame,
    threshold: float = DEFAULT_P_VALUE_THRESHOLD,
) -> List[str]:
    """
    Filter features based on p-value threshold.

    Only removes features with p-value above threshold (clearly non-significant).

    Args:
        association_df (pd.DataFrame): DataFrame with 'predictor' and 'p_value' columns
            (output from compute_associations).
        threshold (float): Maximum p-value to keep a feature (default: 0.5).

    Returns:
        List[str]: List of feature names that pass the threshold.
    """
    if association_df.empty or "p_value" not in association_df.columns:
        return []

    # Group by predictor and keep if ANY test passes the threshold
    passed = (
        association_df.groupby("predictor")["p_value"]
        .min()
        .loc[lambda x: x <= threshold]
    )
    return passed.index.tolist()


def filter_by_vif(
    vif_df: pd.DataFrame,
    threshold: float = DEFAULT_VIF_THRESHOLD,
) -> List[str]:
    """
    Filter features with extreme multicollinearity based on VIF.

    Only removes features with very high VIF values.

    Args:
        vif_df (pd.DataFrame): DataFrame with 'predictor' and 'statistic_value' columns
            (VIF output from compute_associations).
        threshold (float): Maximum VIF to keep a feature (default: 50.0).

    Returns:
        List[str]: List of feature names that pass the threshold.
    """
    if vif_df.empty or "statistic_value" not in vif_df.columns:
        return []

    passed = vif_df[vif_df["statistic_value"] <= threshold]["predictor"]
    return passed.tolist()

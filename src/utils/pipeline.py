from typing import Iterable
import pandas as pd

from src.feature_config import (
    ALL_CATEGORICAL_FEATURES,
    ALL_BINARY_FEATURES,
    ALL_CONTINUOUS_FEATURES,
)


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

    age_labels = []
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
    excluded_targets: Iterable[str] = (),
) -> pd.DataFrame:
    """Prepare the feature matrix used for modeling."""

    allowed_columns = (
        ALL_CATEGORICAL_FEATURES + ALL_BINARY_FEATURES + ALL_CONTINUOUS_FEATURES
    )
    exclusions = set(excluded_targets)
    features = df.drop(
        columns=[
            col for col in df.columns if col not in allowed_columns or col in exclusions
        ],
        errors="ignore",
    )

    for col in ALL_CATEGORICAL_FEATURES:
        if col in features.columns and col not in exclusions:
            dummies = pd.get_dummies(features[col], prefix=col)
            features = pd.concat([features.drop(columns=[col]), dummies], axis=1)

    return features

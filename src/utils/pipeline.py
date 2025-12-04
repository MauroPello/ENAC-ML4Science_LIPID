from typing import Iterable
import pandas as pd
import re

from sklearn.preprocessing import OneHotEncoder
from src.feature_config import (
    ALL_CATEGORICAL_FEATURES,
    ALL_BINARY_FEATURES,
    ALL_CONTINUOUS_FEATURES,
    POSSIBLE_TARGET_FEATURES,
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


def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode ordinal categorical features into integers."""
    df = df.copy()

    income_map = {"Low": 0, "Medium": 1, "High": 2}
    if "income" in df.columns:
        df["income"] = df["income"].map(income_map)

    education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    if "education_level" in df.columns:
        df["education_level"] = df["education_level"].map(education_map)

    if "age_bin" in df.columns:
        categories = df["age_bin"].dropna().unique()

        def get_sort_key(label):
            if str(label).startswith("<"):
                return float("-inf")
            numbers = re.findall(r"\d+", str(label))
            if not numbers:
                return float("inf")
            return int(numbers[0])

        sorted_cats = sorted(categories, key=get_sort_key)
        age_quantiles_map = {cat: i for i, cat in enumerate(sorted_cats)}
        df["age_bin"] = df["age_bin"].map(age_quantiles_map)

    return df


def process_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Process other non-numeric features like time and binary categories."""
    df = df.copy()

    if "bedtime_hour" in df.columns:

        def time_to_float(t_str):
            try:
                h, m = map(int, str(t_str).split(":"))
                return (h * 60.0) + m / 60.0
            except (ValueError, AttributeError):
                return None

        df["bedtime_hour"] = df["bedtime_hour"].apply(time_to_float)

    # Encode Sex
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"Male": 0, "Female": 1})

    return df


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


def drop_extra_features(
    df: pd.DataFrame,
    excluded_targets: Iterable[str] = (),
) -> pd.DataFrame:
    """Prepare the feature matrix used for modeling."""

    allowed = set(
        ALL_CATEGORICAL_FEATURES + ALL_BINARY_FEATURES + ALL_CONTINUOUS_FEATURES
    )
    exclusions = set(excluded_targets)
    allowed = allowed - exclusions
    features = df.drop(
        columns=[
            col
            for col in df.columns
            if col not in allowed and not col.startswith("typology")
        ],
        errors="ignore",
    )

    return features


def ohe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the 'typology' feature using sklearn (the only truly categorical feature left).

    Creates binary columns for each category in 'typology' and removes
    the original column. Returns the dataframe unchanged if 'typology'
    is missing.
    """
    df = df.copy()
    if "typology" not in df.columns:
        print("Warning: 'typology' column not found for one-hot encoding.")
        return df
    encoder = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown="ignore")
    encoded_array = encoder.fit_transform(df[["typology"]])
    feature_names = encoder.get_feature_names_out(["typology"])
    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(columns=["typology"])
    return df


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full data processing pipeline
    It is assumed that the input dataframe has already been loaded using load_combined_dataset.
    The sequence of processing steps is:
    1. Assign age quantile bins
    2. Encode ordinal features
    3. Process additional features

    Args:
        df: pd.DataFrame
            The input dataframe to process.
    Returns:
        pd.DataFrame
            The processed dataframe.
    """
    processed_df = assign_age_quantile_bins(df)
    processed_df = encode_ordinal_features(processed_df)
    processed_df = process_additional_features(processed_df)
    processed_df = ohe_features(processed_df)
    return processed_df

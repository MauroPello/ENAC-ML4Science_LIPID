from typing import Iterable, Sequence, Tuple
import pandas as pd
import re

from scipy.stats import skew

from sklearn.preprocessing import OneHotEncoder
from src.feature_config import (
    ALL_CATEGORICAL_FEATURES,
    ALL_BINARY_FEATURES,
    ALL_CONTINUOUS_FEATURES,
    EXPECTED_HOURS,
    POSSIBLE_TARGET_FEATURES,
)


def load_combined_dataset(
    morph_csv_path: str,
    health_excel_path: str,
    socio_sheet: str = "Participant_SocioDemograph_Data",
    clinical_sheet: str = "Participant_HEALTH_Data",
) -> pd.DataFrame:
    """Load and merge morphology and health data into a single DataFrame.

    Args:
        morph_csv_path (str): Path to the morphology data CSV.
        health_excel_path (str): Path to the health data Excel file.
        socio_sheet (str): Sheet name for socio-demographic data.
        clinical_sheet (str): Sheet name for clinical health data.

    Returns:
        pd.DataFrame: Merged dataframe containing morphology and health data.
    """

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
    """Encode ordinal categorical features into integers.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with encoded ordinal features.
    """
    df = df.copy()

    income_map = {"<50k": 0, "50–100k": 1, "100–150k": 2, ">150k": 3}
    if "income" in df.columns:
        df["income"] = df["income"].map(income_map)

    education_map = {"primary": 0, "secondary": 1, "tertiary": 2}
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
    """Process other non-numeric features like time and binary categories.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with processed additional features.
    """
    df = df.copy()

    if "bedtime_hour" in df.columns:
        temp_dt = pd.to_datetime(df["bedtime_hour"], format="%H:%M", errors="coerce")
        df["bedtime_hour"] = temp_dt.dt.hour + (temp_dt.dt.minute / 60.0)

    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"M": 0, "F": 1})

    return df


def assign_age_bins(
    df: pd.DataFrame,
    age_column: str = "age",
    output_column: str = "age_bin",
) -> pd.DataFrame:
    """Bucket ages into fixed, interpretable ranges instead of quantiles.

    The bins are inspired by CDC/WHO adult health groups and pediatric sleep
    guidance, providing clearer semantic meaning while using fewer pediatric
    splits:
        - Early Childhood (0-6y)
        - Children (6-12y)
        - Teenagers (12-18y)
        - Young Adults (18-30y)
        - Adults (30-50y)
        - Middle-Aged Adults (50-70y)
        - Older Adults (70+y)

    Args:
        df (pd.DataFrame): Input dataframe.
        age_column (str): Name of the age column.
        output_column (str): Name of the output bin column.

    Returns:
        pd.DataFrame: Dataframe with added age bin column.
    """

    df = df.copy()
    if age_column not in df.columns:
        df[output_column] = pd.NA
        return df

    age_series = pd.to_numeric(df[age_column], errors="coerce")

    bins = [
        float("-inf"),
        6.0,
        12.0,
        18.0,
        30.0,
        50.0,
        70.0,
        float("inf"),
    ]
    labels = EXPECTED_HOURS.keys()

    df[output_column] = pd.cut(
        age_series,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    return df


def drop_extra_features(
    df: pd.DataFrame,
    excluded_targets: Iterable[str] = (),
) -> pd.DataFrame:
    """Prepare the feature matrix used for modeling.

    Args:
        df (pd.DataFrame): Input dataframe.
        excluded_targets (Iterable[str]): List of target columns to exclude.

    Returns:
        pd.DataFrame: Feature matrix containing only allowed features.
    """

    allowed = set(
        ALL_CATEGORICAL_FEATURES
        + ALL_BINARY_FEATURES
        + ALL_CONTINUOUS_FEATURES
        + POSSIBLE_TARGET_FEATURES
    )
    exclusions = set(excluded_targets)
    allowed = allowed - exclusions
    features = df.drop(
        columns=[col for col in df.columns if col not in allowed],
        errors="ignore",
    )

    return features


def summarize_continuous_stats(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Summarize mean, standard deviation, and skewness for numeric columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (Sequence[str]): List of column names to summarize.

    Returns:
        pd.DataFrame: Summary dataframe with statistics for each column.
    """

    rows: list[dict[str, float]] = []
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue

        rows.append(
            {
                "feature": column,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "skewness": float(skew(series, bias=False)),
            }
        )

    return pd.DataFrame(rows)


def ohe_features(df: pd.DataFrame, feature_types: dict) -> pd.DataFrame:
    """
    One-hot encode the 'typology' feature using sklearn (the only truly categorical feature left).

    Creates binary columns for each category in 'typology' and removes
    the original column. Returns the dataframe unchanged if 'typology'
    is missing.

    Args:
        df (pd.DataFrame): Input dataframe.
        feature_types (dict): Dictionary mapping feature names to types.

    Returns:
        Tuple[pd.DataFrame, dict]: Tuple containing the dataframe with OHE features and the updated feature types.
    """
    df = df.copy()
    if "typology" not in df.columns:
        return df, feature_types

    encoder = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown="ignore")
    encoded_array = encoder.fit_transform(df[["typology"]])
    feature_names = encoder.get_feature_names_out(["typology"])

    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(columns=["typology"])

    feature_types = feature_types.copy()
    feature_types.pop("typology")
    for feature in feature_names:
        feature_types[feature] = "binary"

    return df, feature_types


def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full data preprocessing pipeline
    It is assumed that the input dataframe has already been loaded using load_combined_dataset.
    The sequence of preprocessing steps is:
    1. Assign fixed age bins
    2. Encode ordinal features
    3. Process additional features

    Args:
        df (pd.DataFrame): The input dataframe to process.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    processed_df = assign_age_bins(df)
    processed_df = encode_ordinal_features(processed_df)
    processed_df = process_additional_features(processed_df)
    processed_df = drop_extra_features(processed_df)
    return processed_df

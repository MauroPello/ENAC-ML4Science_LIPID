from pathlib import Path
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.pipeline import (
    load_combined_dataset,
    run_preprocessing_pipeline,
    ohe_features,
)
from src.feature_config import (
    EQ_FEATURES,
    MORPHOLOGY_FEATURES,
    ALL_CONTINUOUS_FEATURES,
    ALL_CATEGORICAL_FEATURES,
    ALL_BINARY_FEATURES,
    POSSIBLE_TARGET_FEATURES,
)
from src.target_definition.aggregate import aggregate_health_targets
from src.feature_selection import (
    compute_associations,
    compute_categorical_associations,
    compute_vif,
    filter_by_p_value,
    filter_by_vif,
    drop_features,
)
from src.predictive import run_modeling_suite
from src.utils.prediction import infer_neighborhood_health_risks

data_path = Path("data")
df = pd.read_csv(data_path / "morphology_data_integrated.csv", sep=";", decimal=",")
print(df.head())

print(f"Number of rows: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print(df["typology"].value_counts())

# Standardize column names: lowercase and replace spaces with underscores
df = df.drop(columns=["City"])  # remove duplicate column
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
print(df.head())

print(df.columns.unique())

# Print the columns with unique values == len(df)

cols_to_remove = []
for col in df.columns:
    if col == "id":
        continue
    if df[col].nunique() == len(df) or df[col].nunique() <= 1:
        cols_to_remove.append(col)

print("Columns to remove: ", cols_to_remove)
df = df.drop(columns=cols_to_remove)

features_to_remove = [
    "left",
    "top",
    "right",
    "bottom",
    "row_index",
    "col_index",
    "q_cluster",
    "z_distance",
]
df = df.drop(columns=features_to_remove)
print(f"Numbers of columns after removing metadata features: {len(df.columns)}")
print(df.columns)

# Remove the columns with the "bin" prefix, as they are binned versions of other columns
bin_columns = [col for col in df.columns if col.startswith("bin")]
df = df.drop(columns=bin_columns)
print(f"Numbers of columns after removing binned features: {len(df.columns)}")
print(df.columns)

# Print the rows where there are NaN values
original_rows_num = df.shape[0]
nan_rows = df[df.isna().any(axis=1)]
df.dropna(inplace=True)
print(f"Number of rows after dropping NaN values: {len(df)}")
print(f"Number of rows dropped: {original_rows_num - len(df)}")

# Remove numeric columns with zero variance (constant columns)
numeric_cols = df.select_dtypes(include=["number"]).columns
zero_var_cols = [c for c in numeric_cols if abs(df[c].std()) <= 1e-8]
if zero_var_cols:
    print("Zero-variance numeric columns to remove:", zero_var_cols)
    df = df.drop(columns=zero_var_cols)
else:
    print("No zero-variance numeric columns found")
print(f"Columns remaining after removing zero-variance: {len(df.columns)}")

# convert typology to categorical dtype
df["typology"] = df["typology"].astype("category")

print(df.info())

# Save the cleaned dataframe
df.to_csv(data_path / "morphology_data_cleaned.csv", index=False)

from pathlib import Path
import pandas as pd
import math
from src.utils.pipeline import (
    load_combined_dataset,
    run_preprocessing_pipeline,
    ohe_features,
)
from src.feature_config import (
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

print()

data_path = Path("data")
df = pd.read_csv(data_path / "morphology_data_integrated.csv", sep=";", decimal=",")
print(df.head())

print(f"Number of rows: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print(df["typology"].value_counts())
print()

# Standardize column names: lowercase and replace spaces with underscores
df = df.drop(columns=["City"])  # remove duplicate column
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
print(df.columns.unique())

print()

# Print the columns with unique values == len(df)
cols_to_remove = []
for col in df.columns:
    if col == "id":
        continue
    if df[col].nunique() == len(df) or df[col].nunique() <= 1:
        cols_to_remove.append(col)

df = df.drop(columns=cols_to_remove)
print("Columns to remove: ", cols_to_remove)
print()

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
print()

# Remove the columns with the "bin" prefix, as they are binned versions of other columns
bin_columns = [col for col in df.columns if col.startswith("bin")]
df = df.drop(columns=bin_columns)
print(f"Numbers of columns after removing binned features: {len(df.columns)}")
print(df.columns)
print()

# Print the rows where there are NaN values
original_rows_num = df.shape[0]
nan_rows = df[df.isna().any(axis=1)]
df.dropna(inplace=True)
print(f"Number of rows after dropping NaN values: {len(df)}")
print(f"Number of rows dropped: {original_rows_num - len(df)}")
print()

# Remove numeric columns with zero variance (constant columns)
numeric_cols = df.select_dtypes(include=["number"]).columns
zero_var_cols = [c for c in numeric_cols if abs(df[c].std()) <= 1e-8]
if zero_var_cols:
    print("Zero-variance numeric columns to remove:", zero_var_cols)
    df = df.drop(columns=zero_var_cols)
else:
    print("No zero-variance numeric columns found")
print(f"Columns remaining after removing zero-variance: {len(df.columns)}")
print()

# convert typology to categorical dtype
df["typology"] = df["typology"].astype("category")
print(df.info())
print()

print(df.describe().T)
print()

# Save the cleaned dataframe
df.to_csv(data_path / "morphology_data_cleaned.csv", index=False)
print("Saved cleaned morphology dataset!")
print()

# ABLATION STUDY OPTIONS
# Set these to False to disable the corresponding step
ONE_HOT_ENCODING = True
FEATURE_SELECTION = True
STANDARD_SCALING = True
CLASS_BALANCING = True
HYPERPARAMETER_REFINING = True

print("Combining morphology dataset with the health dataset")
df = load_combined_dataset(
    data_path / "morphology_data_cleaned.csv",
    data_path / "synthetic_health_data.xlsx",
)
df = run_preprocessing_pipeline(df)
print(f"Final dataset shape: {df.shape}")
print(df.head())
print()

print(
    f"Number of features that are continuous: {len([col for col in df.columns if col in ALL_CONTINUOUS_FEATURES and col not in POSSIBLE_TARGET_FEATURES])}",
)
print()

all_feature_types = {}
for column in df.columns:
    if column in ALL_CONTINUOUS_FEATURES:
        all_feature_types[column] = "continuous"
    elif column in ALL_CATEGORICAL_FEATURES:
        all_feature_types[column] = "categorical"
    elif column in ALL_BINARY_FEATURES:
        all_feature_types[column] = "binary"
    else:
        non_null_values = df[column].dropna()
        if not non_null_values.empty and non_null_values.isin([0, 1]).all():
            all_feature_types[column] = "binary"
        else:
            all_feature_types[column] = "continuous"

datasets = {
    "cardiovascular": aggregate_health_targets(
        df, target_feature="cardiovascular", feature_types=all_feature_types
    ),
    "mental_health": aggregate_health_targets(
        df, target_feature="mental_health", feature_types=all_feature_types
    ),
    "sleep_disorder": aggregate_health_targets(
        df, target_feature="sleep_disorder", feature_types=all_feature_types
    ),
    "respiratory": aggregate_health_targets(
        df, target_feature="respiratory", feature_types=all_feature_types
    ),
}

for name, dataset_with_info in datasets.items():
    print(f"\n=== Analyzing target type: {name} ===")

    dataset = dataset_with_info["data"]
    feature_types = dataset_with_info["feature_types"]

    if not FEATURE_SELECTION:
        # Optional OHE even when skipping feature selection
        dataset, feature_types = ohe_features(
            dataset,
            feature_types,
            enable=ONE_HOT_ENCODING,
        )
        print(
            "Feature selection disabled: retaining all predictors without p-value/VIF filtering."
        )
        print("\nSelected features:")
        print([col for col in dataset.columns if col != "target"])
        datasets[name]["data"] = dataset
        datasets[name]["feature_types"] = feature_types
        continue

    # compute associations for categorical features before one-hot encoding
    categorical_association_df = compute_categorical_associations(
        dataset,
        "target",
        feature_types,
    )
    if categorical_association_df is not None and not categorical_association_df.empty:
        print(categorical_association_df)

        # keep categorical predictors that are significant
        passed_by_p = filter_by_p_value(categorical_association_df)

        dropped_categorical_features = [
            feature
            for feature in dataset.columns
            if feature in ALL_CATEGORICAL_FEATURES and feature not in passed_by_p
        ]

        dataset, feature_types = drop_features(
            dataset,
            feature_types,
            dropped_categorical_features,
        )
        print(
            f"Categorical features dropped because they did not pass p-value thresholds: \n {dropped_categorical_features}"
        )

    # one hot encoding of categorical features
    dataset, feature_types = ohe_features(
        dataset,
        feature_types,
        enable=ONE_HOT_ENCODING,
    )

    # compute all association tests
    association_df = compute_associations(
        dataset,
        "target",
        feature_types,
    )

    # filter out features that are not significant in any of our association tests
    passed_by_p = filter_by_p_value(association_df)
    feature_names_to_keep = set(passed_by_p)

    dropped_features_by_p_value = [
        feature
        for feature in dataset.columns
        if feature not in feature_names_to_keep and feature != "target"
    ]
    dataset, feature_types = drop_features(
        dataset,
        feature_types,
        dropped_features_by_p_value,
    )
    print(
        f"Features dropped because they did not pass p-value thresholds: \n {dropped_features_by_p_value}"
    )

    vif_df = pd.DataFrame(compute_vif(dataset, feature_types))
    if not vif_df.empty:
        vif_df = vif_df.sort_values(by="statistic_value", ascending=False)

        passed_by_vif = set(filter_by_vif(vif_df))
        dropped_features_by_VIF = [
            feature
            for feature, ftype in feature_types.items()
            if ftype == "continuous"
            and feature not in passed_by_vif
            and feature != "target"
        ]
        dataset, feature_types = drop_features(
            dataset,
            feature_types,
            dropped_features_by_VIF,
        )
        print(
            f"Features dropped because they did not pass VIF thresholds: \n {dropped_features_by_VIF}"
        )
    else:
        print(
            "Skipped VIF step: no continuous features or insufficient data; nothing dropped."
        )

    print("\nSelected features:")
    selected_features = [col for col in dataset.columns if col != "target"]
    print(selected_features)
    print()

    # Save changes to dataset
    datasets[name]["data"] = dataset
    datasets[name]["feature_types"] = feature_types

    for test_name, subset in association_df.groupby("test", sort=False):
        print(f"\nTest: {test_name}")
        print(subset.reset_index(drop=True))

    if vif_df is not None and vif_df.empty is False:
        print(f"\nTest: VIF")
        print(
            vif_df.drop(columns=["p_value"], errors="ignore").reset_index(drop=True)
        )

print()

# Initialize lists to store results
all_regression_results = []
all_classification_results = []
all_coefficients = []
best_models = {}

print("Running Predictive Modeling Suite for All Datasets\n")

for name, dataset_with_info in datasets.items():
    print(f"Processing: {name.upper()}...")

    dataset = dataset_with_info["data"]
    feature_types = dataset_with_info["feature_types"]
    target_type = feature_types["target"]

    model_results = run_modeling_suite(
        dataset,
        "target",
        feature_types=feature_types,
        imbalance_strategy=(
            ("sample_weight" if target_type == "continuous" else "class_weight")
            if CLASS_BALANCING
            else "none"
        ),
        use_standard_scaling=STANDARD_SCALING,
        refine_hyperparameters=HYPERPARAMETER_REFINING,
        enable_ohe=ONE_HOT_ENCODING,
        dataset_name=name,
    )

    coef_df = model_results.get("coefficients", pd.DataFrame())
    best_models[name] = model_results.get("best_model_fitted")

    if target_type == "continuous":
        # Collect Regression Results
        reg_df = model_results.get("regression_results", pd.DataFrame())
        if not reg_df.empty:
            reg_df["Dataset"] = name.upper()
            all_regression_results.append(reg_df)

        # Collect Coefficients (regression)
        if not coef_df.empty:
            coef_df["Dataset"] = name.upper()
            all_coefficients.append(coef_df)
    else:
        # Collect Classification Results
        clf_df = model_results.get("classification_results", pd.DataFrame())
        if not clf_df.empty:
            clf_df["Dataset"] = name.upper()
            all_classification_results.append(clf_df)

        # Collect Coefficients (classification)
        if not coef_df.empty:
            coef_df["Dataset"] = name.upper()
            all_coefficients.append(coef_df)


# --- Display Summary Results ---

final_reg_df = pd.DataFrame()
final_clf_df = pd.DataFrame()
final_coef_df = pd.DataFrame()

if all_regression_results:
    final_reg_df = pd.concat(all_regression_results, ignore_index=True)
    cols = ["Dataset"] + [c for c in final_reg_df.columns if c != "Dataset"]
    final_reg_df = final_reg_df[cols]

if all_classification_results:
    final_clf_df = pd.concat(all_classification_results, ignore_index=True)
    cols = ["Dataset"] + [c for c in final_clf_df.columns if c != "Dataset"]
    final_clf_df = final_clf_df[cols]

if all_coefficients:
    final_coef_df = pd.concat(all_coefficients, ignore_index=True)
    cols = ["Dataset"] + [c for c in final_coef_df.columns if c != "Dataset"]
    final_coef_df = final_coef_df[cols]

# Per-dataset ordered view: results first, then coefficients
if not (final_reg_df.empty and final_clf_df.empty and final_coef_df.empty):
    print(f"PER-DATASET RESULTS AND COEFFICIENTS")
    for dataset_name in [name.upper() for name in datasets.keys()]:
        print(f"\nDataset: {dataset_name}")
        reg_subset = (
            final_reg_df[final_reg_df["Dataset"] == dataset_name]
            if not final_reg_df.empty
            else pd.DataFrame()
        )
        if not reg_subset.empty:
            print("  Regression results:")
            print(reg_subset)
        clf_subset = (
            final_clf_df[final_clf_df["Dataset"] == dataset_name]
            if not final_clf_df.empty
            else pd.DataFrame()
        )
        if not clf_subset.empty:
            print("  Classification results:")
            print(clf_subset)
        coef_subset = (
            final_coef_df[final_coef_df["Dataset"] == dataset_name]
            if not final_coef_df.empty
            else pd.DataFrame()
        )
        if not coef_subset.empty:
            print("  Coefficients / importances:")
            print(coef_subset)

# Post-hoc summary: pick lowest-RMSE model per dataset (regression)
# and pick best-F1 model per dataset (classification); report feature strengths vs. mean

tol = 0.05  # 5% margin for "similar" impacts


def add_relative_strengths(coef_rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    coef_rows = coef_rows.copy()
    coef_rows["abs_coef"] = coef_rows["coefficient"].abs()
    mean_abs = coef_rows["abs_coef"].mean()
    mean_abs = float(mean_abs) if not math.isnan(mean_abs) else 0.0
    if mean_abs == 0.0:
        coef_rows["rel_to_mean_abs"] = 0.0
    else:
        coef_rows["rel_to_mean_abs"] = coef_rows["abs_coef"] / mean_abs
    coef_rows = coef_rows.sort_values("abs_coef", ascending=False)
    return coef_rows, mean_abs


# --- Regression (uses RMSE) ---
if final_reg_df is not None and not final_reg_df.empty:
    datasets_with_rmse = final_reg_df["Dataset"].unique()
    for ds in datasets_with_rmse:
        ds_rows = final_reg_df[final_reg_df["Dataset"] == ds]
        best_row = ds_rows.loc[ds_rows["RMSE"].idxmin()]
        best_model = best_row["model"]
        best_rmse = best_row["RMSE"]
        print(f"\n=== {ds} (Regression) ===")
        print(f"Best model (lowest RMSE): {best_model} | RMSE={best_rmse:.4f}")

        coef_rows = final_coef_df[
            (final_coef_df["Dataset"] == ds) & (final_coef_df["model"] == best_model)
        ].copy()
        if coef_rows.empty:
            print("No coefficients/importances stored for this model.")
            continue

        coef_rows, mean_abs = add_relative_strengths(coef_rows)
        print(f"Mean |coef| across features: {mean_abs:.4f}")
        print("Top 5 features with relative strength (|coef| / mean |coef|):")
        print(
            coef_rows.head(5)[["feature", "coefficient", "abs_coef", "rel_to_mean_abs"]]
        )

# --- Classification (uses F1 to pick best) ---
if final_clf_df is not None and not final_clf_df.empty:
    datasets_with_clf = final_clf_df["Dataset"].unique()
    for ds in datasets_with_clf:
        ds_rows = final_clf_df[final_clf_df["Dataset"] == ds]
        best_row = ds_rows.loc[ds_rows["F1"].idxmax()]
        best_model = best_row["model"]
        best_f1 = best_row["F1"]
        print(f"\n=== {ds} (Classification) ===")
        print(f"Best model (highest F1): {best_model} | F1={best_f1:.4f}")

        coef_rows = final_coef_df[
            (final_coef_df["Dataset"] == ds) & (final_coef_df["model"] == best_model)
        ].copy()
        if coef_rows.empty:
            print("No coefficients/importances stored for this model.")
            continue

        coef_rows, mean_abs = add_relative_strengths(coef_rows)
        print(f"Mean |coef| across features: {mean_abs:.4f}")
        print("Top 5 features with relative strength (|coef| / mean |coef|):")
        print(
            coef_rows.head(5)[["feature", "coefficient", "abs_coef", "rel_to_mean_abs"]]
        )

feature_types_map = {
    dataset_name: dataset_info["feature_types"]
    for dataset_name, dataset_info in datasets.items()
}

health_risks = infer_neighborhood_health_risks(
    best_models=best_models,
    feature_types_map=feature_types_map,
    enable_ohe=ONE_HOT_ENCODING,
)

for name, df_risk in health_risks.items():
    print(f"\n{name.upper()} risk table")
    print(df_risk)


# let's checkout the health risks related to the neighborhood with id 36

neighborhood_id = df_risk["id"][0]
for name, df_risk in health_risks.items():
    print(f"\n{name.upper()} risk table")
    neighborhood_df = df_risk[df_risk["id"] == neighborhood_id]
    print(neighborhood_df[neighborhood_df.columns[:6]])


# Average health risk per typology across all socio-demographic combinations (wide view)

rows = []
for name, df_risk in health_risks.items():
    risk_col = f"risk_{name}"
    if risk_col not in df_risk.columns:
        print(f"Skipping {name}: column '{risk_col}' not found.")
        continue
    grouped = (
        df_risk.groupby("typology")[risk_col]
        .mean()
        .reset_index()
        .rename(columns={risk_col: name})
    )
    rows.append(grouped)

if rows:
    combined = rows[0]
    for g in rows[1:]:
        combined = combined.merge(g, on="typology", how="outer")
    combined = combined.sort_values("typology")
    print("Average risk by typology (one column per dataset)")
    print(combined)
else:
    print("No risk data available to summarize.")

# Flag typologies with elevated modeled risk per dataset (no melting)

risk_cols = [c for c in combined.columns if c != "typology"]
for col in risk_cols:
    df = combined[["typology", col]].rename(columns={col: "risk"})

    mean_risk = df["risk"].mean()
    std_risk = df["risk"].std()
    zscores = (df["risk"] - mean_risk) / std_risk if std_risk else float("nan")
    df = df.assign(zscore=zscores.round(2))

    high = df[df["zscore"] >= 1.5].sort_values("zscore", ascending=False)
    low = df[df["zscore"] <= -1.5].sort_values("zscore", ascending=True)

    print(f"\n{col.upper()} health risk")

    if not high.empty:
        print(f"\nTypologies with alarmingly higher {col} risk (z >= 1.5):")
        print(high[["typology", "risk", "zscore"]])

    if not low.empty:
        print(f"\nTypologies with exceptionally lower {col} risk (z <= -1.5):")
        print(low[["typology", "risk", "zscore"]])

    print("\nPer-typology risk with zscore: ")
    print(df.sort_values("risk", ascending=False)[["typology", "risk", "zscore"]])
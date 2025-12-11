import pandas as pd

from .binary import evaluate_binary_target, run_chi_square
from .continuous import evaluate_continuous_target, run_anova


def compute_associations(
    df: pd.DataFrame,
    target_feature: str,
    feature_types: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run univariate association tests and multicollinearity diagnostics."""

    target_type = feature_types["target"]

    association_records: list[dict[str, float]] = []
    vif_records: list[dict[str, float]] = []

    if target_type == "continuous":
        association_records, vif_records = evaluate_continuous_target(
            df, target_feature, feature_types
        )
    elif target_type == "binary":
        association_records = evaluate_binary_target(
            df, target_feature, feature_types
        )
    else:
        raise ValueError(f"Target feature type '{target_type}' is not supported.")

    association_df = pd.DataFrame(association_records)
    if not association_df.empty:
        association_df = association_df.sort_values(by="p_value", na_position="last")

    vif_df = pd.DataFrame(vif_records)
    if not vif_df.empty:
        vif_df = vif_df.sort_values(by="statistic_value", ascending=False)

    return association_df, vif_df


def compute_categorical_associations(
    df: pd.DataFrame,
    target_feature: str,
    feature_types: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run tests diagnostics for categorical features."""

    # identify categorical predictors excluding the current target
    categorical_features = [
        name
        for name, type in feature_types.items()
        if name != target_feature and type == "categorical"
    ]

    if feature_types["target"] == "continuous":
        anova_rows = []
        for col in categorical_features:
            working = pd.DataFrame(
                {
                    "predictor": df[col],
                    "target": df[target_feature],
                }
            )
            records = run_anova(col, working)
            if records:
                anova_rows.extend(records)
        if anova_rows:
            return (
                pd.DataFrame(anova_rows).sort_values("p_value").reset_index(drop=True)
            )
        else:
            print(
                "No categorical features available for ANOVA against the continuous target."
            )
    elif feature_types["target"] == "binary":
        chi_rows = []
        for col in categorical_features:
            working = pd.DataFrame(
                {
                    "predictor": df[col],
                    "target": df[target_feature],
                }
            )
            records = run_chi_square(col, working, "categorical")
            if records:
                chi_rows.extend(records)
        if chi_rows:
            return pd.DataFrame(chi_rows).sort_values("p_value").reset_index(drop=True)
        else:
            print(
                "No categorical features available for chi-square tests against the binary target."
            )
    else:
        print(
            f"Association tests are only defined for continuous or binary targets (got '{feature_types["target"]}')."
        )

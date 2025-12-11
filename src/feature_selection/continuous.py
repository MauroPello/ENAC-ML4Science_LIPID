"""Association tests and diagnostics for continuous targets or predictors."""

import pandas as pd
from scipy import stats
import statsmodels.api as sm


def evaluate_continuous_target(
    df: pd.DataFrame,
    target_feature: str,
    feature_types: dict[str, str],
) -> list[dict[str, float]]:
    """Run simple univariate checks for a continuous target."""

    association_records: list[dict[str, float]] = []
    numeric_target = pd.to_numeric(df[target_feature], errors="coerce")

    for column in df.columns:
        if column == target_feature:
            continue

        predictor_type = feature_types[column]

        working = pd.DataFrame(
            {
                "predictor": df[column],
                "target": numeric_target,
            }
        )
        working = working.dropna(subset=["target"])
        if working.empty:
            continue

        if predictor_type == "continuous":
            association_records.extend(_run_continuous_correlations(column, working))
        elif predictor_type == "binary":
            association_records.extend(_run_point_biserial(column, working))

        association_records.extend(
            _run_linear_regression(column, predictor_type, working)
        )

    return association_records


def _run_continuous_correlations(
    column: str, working: pd.DataFrame
) -> list[dict[str, float]]:
    """Compute Pearson and Spearman correlations for continuous predictors."""

    cleaned = working.copy()
    cleaned["predictor"] = pd.to_numeric(cleaned["predictor"], errors="coerce")
    cleaned = cleaned.dropna(subset=["predictor"])
    if cleaned.empty or cleaned["predictor"].nunique() <= 1:
        return []

    correlations: list[dict[str, float]] = []
    pearson_r, pearson_p = stats.pearsonr(cleaned["predictor"], cleaned["target"])
    correlations.append(
        {
            "predictor": column,
            "predictor_type": "continuous",
            "test": "Pearson correlation",
            "statistic_name": "r",
            "statistic_value": float(pearson_r),
            "p_value": float(pearson_p),
        }
    )

    spearman_r, spearman_p = stats.spearmanr(cleaned["predictor"], cleaned["target"])
    correlations.append(
        {
            "predictor": column,
            "predictor_type": "continuous",
            "test": "Spearman correlation",
            "statistic_name": "rho",
            "statistic_value": float(spearman_r),
            "p_value": float(spearman_p),
        }
    )
    return correlations


def _run_point_biserial(column: str, working: pd.DataFrame) -> list[dict[str, float]]:
    """Compute point-biserial correlation for binary predictors."""

    cleaned = working.copy()
    cleaned["predictor"] = pd.to_numeric(cleaned["predictor"], errors="coerce")
    cleaned = cleaned.dropna(subset=["predictor"])
    if cleaned.empty or cleaned["predictor"].nunique() != 2:
        return []

    statistic, p_value = stats.pointbiserialr(cleaned["predictor"], cleaned["target"])
    return [
        {
            "predictor": column,
            "predictor_type": "binary",
            "test": "Point-biserial correlation",
            "statistic_name": "r_pb",
            "statistic_value": float(statistic),
            "p_value": float(p_value),
        }
    ]


def run_anova(column: str, working: pd.DataFrame) -> list[dict[str, float]]:
    """ANOVA for categorical predictors with continuous target."""

    groups = []
    for _, group in working.dropna(subset=["predictor"]).groupby(
        "predictor", observed=False
    ):
        values = group["target"].dropna().values
        if len(values) > 1:
            groups.append(values)

    if len(groups) <= 1:
        return []

    f_statistic, p_value = stats.f_oneway(*groups)
    return [
        {
            "predictor": column,
            "predictor_type": "categorical",
            "test": "ANOVA",
            "statistic_name": "F",
            "statistic_value": float(f_statistic),
            "p_value": float(p_value),
        }
    ]


def _run_linear_regression(
    column: str, predictor_type: str, working: pd.DataFrame
) -> list[dict[str, float]]:
    """Run univariate OLS on the predictor representation."""

    regression_features = _build_regression_features(column, predictor_type, working)
    if regression_features.empty:
        return []

    regression_data = pd.concat(
        [regression_features, working["target"]], axis=1
    ).dropna()
    if regression_data.empty or regression_data["target"].nunique() <= 1:
        return []

    X = regression_data.drop(columns="target")
    # Remove constant columns to avoid singular designs.
    X = X.loc[:, X.apply(lambda series: series.nunique() > 1)]
    if X.empty:
        return []

    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(regression_data["target"].astype(float), X.astype(float)).fit()

    records: list[dict[str, float]] = []
    for param, value in model.params.items():
        if param == "const":
            continue
        records.append(
            {
                "predictor": column,
                "predictor_type": predictor_type,
                "test": "Univariate linear regression",
                "statistic_name": f"coef[{param}]",
                "statistic_value": float(value),
                "p_value": float(model.pvalues[param]),
            }
        )
    return records


def _build_regression_features(
    column: str, predictor_type: str, working: pd.DataFrame
) -> pd.DataFrame:
    """Prepare numeric or one-hot encoded predictor for regression."""

    if predictor_type == "categorical":
        return pd.get_dummies(working["predictor"], prefix=column, drop_first=True)

    numeric_series = pd.to_numeric(working["predictor"], errors="coerce")
    return numeric_series.to_frame(name=column)

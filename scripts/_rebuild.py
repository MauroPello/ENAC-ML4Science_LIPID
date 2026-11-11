"""Rebuild the per-target modelling datasets exactly as main.ipynb does.

Shared by the post-hoc analysis scripts (SHAP, sociodemographic baseline) so that
they operate on the same rows, columns, column order and train/test split as
the committed models in outputs/. Nothing here refits or overwrites anything.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.feature_config import (  # noqa: E402
    ALL_BINARY_FEATURES,
    ALL_CATEGORICAL_FEATURES,
    ALL_CONTINUOUS_FEATURES,
    POSSIBLE_TARGET_FEATURES,
)
from src.feature_selection import (  # noqa: E402
    compute_associations,
    compute_categorical_associations,
    compute_vif,
    drop_features,
    filter_by_p_value,
    filter_by_vif,
)
from src.target_definition.aggregate import aggregate_health_targets  # noqa: E402
from src.utils.pipeline import (  # noqa: E402
    load_combined_dataset,
    ohe_features,
    run_preprocessing_pipeline,
)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Settings used for the committed results (main.ipynb, cell "ABLATION STUDY OPTIONS")
DROP_ROWS_WITH_NULL_TARGET = True
ONE_HOT_ENCODING = True
TEST_SIZE = 0.2
RANDOM_STATE = 42

CLASSIFICATION_TARGETS = ["cardiovascular", "mental_health", "respiratory"]
SOCIO_FEATURES = ["age_bin", "income", "education_level"]


def _feature_types_for(df: pd.DataFrame) -> dict[str, str]:
    feature_types: dict[str, str] = {}
    for column in df.columns:
        if column in ALL_CONTINUOUS_FEATURES:
            feature_types[column] = "continuous"
        elif column in ALL_CATEGORICAL_FEATURES:
            feature_types[column] = "categorical"
        elif column in ALL_BINARY_FEATURES:
            feature_types[column] = "binary"
        else:
            non_null = df[column].dropna()
            if not non_null.empty and non_null.isin([0, 1]).all():
                feature_types[column] = "binary"
            else:
                feature_types[column] = "continuous"
    return feature_types


def _select_features(dataset: pd.DataFrame, feature_types: dict[str, str], fit_index=None):
    """Same steps as main.ipynb: categorical p-value filter, OHE, p-value filter, VIF.

    By default (as in main.ipynb) the tests use every row. With `fit_index`, the
    association and VIF tests only use those rows (e.g. the training split); the
    resulting drops are applied to the whole table. OHE needs no labels and is
    always applied to the whole table so train/test columns stay consistent.
    """
    rows = (lambda d: d) if fit_index is None else (lambda d: d.loc[fit_index])
    cat_assoc = compute_categorical_associations(rows(dataset), "target", feature_types)
    if cat_assoc is not None and not cat_assoc.empty:
        passed = filter_by_p_value(cat_assoc)
        dropped = [
            f
            for f in dataset.columns
            if f in ALL_CATEGORICAL_FEATURES and f not in passed
        ]
        dataset, feature_types = drop_features(dataset, feature_types, dropped)

    dataset, feature_types = ohe_features(
        dataset, feature_types, enable=ONE_HOT_ENCODING
    )

    assoc = compute_associations(rows(dataset), "target", feature_types)
    keep = set(filter_by_p_value(assoc))
    dropped = [f for f in dataset.columns if f not in keep and f != "target"]
    dataset, feature_types = drop_features(dataset, feature_types, dropped)

    vif_df = pd.DataFrame(compute_vif(rows(dataset), feature_types))
    if not vif_df.empty:
        vif_df = vif_df.sort_values(by="statistic_value", ascending=False)
        passed_vif = set(filter_by_vif(vif_df))
        dropped = [
            f
            for f, t in feature_types.items()
            if t == "continuous" and f not in passed_vif and f != "target"
        ]
        dataset, feature_types = drop_features(dataset, feature_types, dropped)
    return dataset, feature_types


def load_preprocessed() -> pd.DataFrame:
    """Merged + preprocessed participant table (imputation included), as in main.ipynb."""
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_combined_dataset(
            DATA_DIR / "morphology_data_cleaned.csv",
            DATA_DIR / "health_data_final.xlsx",
            socio_sheet="SocioDemograph",
            clinical_sheet="HEALTH",
        )
        df = run_preprocessing_pipeline(
            df,
            exclude_from_imputation=(
                POSSIBLE_TARGET_FEATURES if DROP_ROWS_WITH_NULL_TARGET else ()
            ),
        )
    return df


def build_datasets(targets=CLASSIFICATION_TARGETS) -> dict[str, dict]:
    """Return, per target: selected-feature data, feature types, and pre-selection data.

    `pre_selection` keeps every candidate predictor (same rows/index), so the
    sociodemographic columns are available even when feature selection dropped them.
    """
    df = load_preprocessed()
    all_feature_types = _feature_types_for(df)

    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for name in targets:
            agg = aggregate_health_targets(
                df,
                target_feature=name,
                feature_types=all_feature_types,
                drop_null_target_rows=DROP_ROWS_WITH_NULL_TARGET,
            )
            pre = agg["data"]
            data, ftypes = _select_features(pre.copy(), dict(agg["feature_types"]))
            out[name] = {"data": data, "feature_types": ftypes, "pre_selection": pre}
    return out


def load_config(name: str) -> dict:
    return json.loads((OUTPUTS_DIR / name / "config.json").read_text())


def load_committed_model(name: str):
    cfg = load_config(name)
    return joblib.load(OUTPUTS_DIR / name / cfg["model_file"]), cfg


def verify_against_config(name: str, data: pd.DataFrame, ftypes: dict) -> None:
    """Fail loudly if the rebuilt dataset does not match what the model was trained on."""
    cfg = load_config(name)
    expected = [c for c in cfg["feature_types"] if c != "target"]
    got = [c for c in data.columns if c != "target"]
    if got != expected:
        raise RuntimeError(
            f"{name}: rebuilt feature columns differ from outputs/{name}/config.json\n"
            f"  expected {expected}\n  got      {got}"
        )
    if ftypes != cfg["feature_types"]:
        raise RuntimeError(f"{name}: rebuilt feature types differ from config.json")


def split(data: pd.DataFrame):
    """Same split as run_modeling_suite -> run_classification_models."""
    X = data.drop(columns=["target"]).reset_index(drop=True)
    y = LabelEncoder().fit_transform(data["target"].reset_index(drop=True))
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X, y, idx_train, idx_test

"""Rebuild the per-target modelling datasets exactly as main.ipynb does.

Shared by the post-hoc analysis scripts (SHAP, sociodemographic baseline) so that
they operate on the same rows, columns, column order and train/test split as
the committed models in outputs/. Nothing here refits or overwrites anything.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import warnings
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
REFERENCE_PATH = PROJECT_ROOT / "docs" / "revision" / "reference_manifest.json"

# Test metrics printed in the committed main.ipynb, rounded to six decimals.
PUBLISHED_METRICS = {
    "cardiovascular": (0.322275, 0.757119),
    "mental_health": (0.414239, 0.586800),
    "respiratory": (0.145985, 0.674908),
}

# Settings used for the committed results (main.ipynb, cell "ABLATION STUDY OPTIONS")
DROP_ROWS_WITH_NULL_TARGET = True
ONE_HOT_ENCODING = True
TEST_SIZE = 0.2
RANDOM_STATE = 42

CLASSIFICATION_TARGETS = ["cardiovascular", "mental_health", "respiratory"]
SOCIO_FEATURES = ["age_bin", "income", "education_level"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_fingerprint(data: pd.DataFrame) -> str:
    """Fingerprint values, labels and row order with the pinned pandas version."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
    ).hexdigest()


def load_reference() -> dict:
    return json.loads(REFERENCE_PATH.read_text())


def health_data_path() -> Path:
    return Path(os.environ.get("HEALTH_DATA_PATH", DATA_DIR / "health_data_final.xlsx")).expanduser()


def require_file_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        hint = (
            "Set HEALTH_DATA_PATH to the original health workbook; the synthetic "
            "workbook cannot reproduce these models."
            if path.suffix == ".xlsx" else "Restore the committed reference file."
        )
        raise FileNotFoundError(
            f"Required reference file is missing: {path}. {hint}"
        )
    if file_sha256(path) != expected:
        raise RuntimeError(f"{path.name}: checksum differs from the verified reference input")


def check_published_metrics(name: str, f1: float, auc: float, *, allow_mismatch=False) -> bool:
    if not np.isfinite([f1, auc]).all():
        raise RuntimeError(f"{name}: non-finite evaluation metrics")
    matches = bool(np.allclose([f1, auc], PUBLISHED_METRICS[name], rtol=0, atol=1e-6))
    if not matches:
        message = (
            f"{name}: full-model refit does not reproduce published F1/ROC-AUC "
            f"on {platform.machine()}: got ({f1:.9f}, {auc:.9f}), "
            f"expected {PUBLISHED_METRICS[name]}. Use the documented x86-64 reference "
            "environment, or --allow-mismatch for a separately labelled sensitivity run."
        )
        if not allow_mismatch:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return matches


def write_run_metadata(path: Path, datasets: dict, **settings) -> None:
    """Record reproducibility metadata without exporting participant-level inputs."""
    metadata = {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "statsmodels")
        },
        "workbook_sha256": file_sha256(health_data_path()),
        "morphology_sha256": file_sha256(DATA_DIR / "morphology_data_cleaned.csv"),
        "script_sha256": {
            p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "datasets": {},
        "settings": settings,
    }
    for name, item in datasets.items():
        _, y, train, test = split(item["data"])
        cfg = load_config(name)
        metadata["datasets"][name] = {
            "fingerprint": dataset_fingerprint(item["data"]),
            "n_rows": len(y),
            "n_test": len(test),
            "n_test_positive": int(y[test].sum()),
            "train_indices_sha256": hashlib.sha256(train.astype("<i8").tobytes()).hexdigest(),
            "test_indices_sha256": hashlib.sha256(test.astype("<i8").tobytes()).hexdigest(),
            "model_sha256": file_sha256(OUTPUTS_DIR / name / cfg["model_file"]),
        }
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


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
    reference = load_reference()
    require_file_hash(health_data_path(), reference["workbook_sha256"])
    require_file_hash(DATA_DIR / "morphology_data_cleaned.csv", reference["morphology_sha256"])
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_combined_dataset(
            DATA_DIR / "morphology_data_cleaned.csv",
            health_data_path(),
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
    path = OUTPUTS_DIR / name / cfg["model_file"]
    require_file_hash(path, load_reference()["target_datasets"][name]["model_sha256"])
    return joblib.load(path), cfg


def verify_against_config(name: str, data: pd.DataFrame, ftypes: dict) -> None:
    """Check schema and the verified reference's values, labels and row order."""
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
    reference = load_reference()["target_datasets"][name]
    if dataset_fingerprint(data) != reference["dataset_fingerprint"]:
        raise RuntimeError(f"{name}: rebuilt values, labels or row order differ from the verified reference")


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

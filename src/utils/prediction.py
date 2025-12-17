import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_config import SOCIO_DEMOGRAPHIC_VALUES
from src.utils.pipeline import (
    drop_extra_features,
    encode_ordinal_features,
    ohe_features,
    process_additional_features,
)


def _predict_scores(
    model: Pipeline, features: pd.DataFrame, target_type: str
) -> np.ndarray:
    """Predict probabilities or scores while handling binary/continuous targets.

    For binary targets, tries `predict_proba`, then `decision_function`, then
    falls back to hard labels. Continuous targets use `predict`.
    """

    if target_type == "binary":
        try:
            proba = model.predict_proba(features)
            if proba.ndim > 1 and proba.shape[1] > 1:
                return proba[:, 1]
            return proba[:, 0]
        except Exception:
            try:
                scores = model.decision_function(features)
                return 1 / (1 + np.exp(-scores))
            except Exception:
                return model.predict(features)

    preds = model.predict(features)
    return np.asarray(preds)


def build_scaler_step(
    use_standard_scaling: bool,
    feature_types: dict[str, str] | None = None,
    *,
    columns: list[str] | None = None,
) -> tuple[str, ColumnTransformer | StandardScaler] | None:
    """Return a scaling step that skips binary features when metadata is available."""

    if not use_standard_scaling:
        return None

    if feature_types:
        candidates = columns or list(feature_types.keys())
        columns_to_scale = [
            col
            for col in candidates
            if col != "target" and feature_types.get(col) != "binary"
        ]

        if not columns_to_scale:
            return None

        transformer = ColumnTransformer(
            [
                (
                    "standard_scaler",
                    StandardScaler(),
                    columns_to_scale,
                )
            ],
            remainder="passthrough",
        )
        return ("scaler", transformer)

    return ("scaler", StandardScaler())


def assemble_steps(*steps: tuple[str, object] | None) -> list[tuple[str, object]]:
    """Filter out None steps while preserving order for pipeline construction."""

    assembled: list[tuple[str, object]] = []
    for step in steps:
        if step is None:
            continue
        try:
            name, component = step
        except Exception:
            # Skip malformed entries instead of raising at pipeline build time
            continue
        assembled.append((name, component))
    return assembled


def _expand_neighborhood_grid(
    morph_csv_path: str | Path,
    socio_config: dict[str, list] = SOCIO_DEMOGRAPHIC_VALUES,
) -> tuple[pd.DataFrame, str]:
    """Load neighborhood data and cross-join all socio-demographic combinations."""

    morph_df = pd.read_csv(morph_csv_path)
    socio_cols = list(socio_config.keys())
    socio_combos = pd.DataFrame(
        list(product(*[socio_config[col] for col in socio_cols])), columns=socio_cols
    )

    expanded = (
        morph_df.assign(_tmp_key=1)
        .merge(socio_combos.assign(_tmp_key=1), on="_tmp_key")
        .drop(columns="_tmp_key")
    )

    # Move socio-demographic columns right after the first column without duplicating them
    first_col = expanded.columns[0]
    remaining = [col for col in expanded.columns[1:] if col not in socio_cols]
    expanded = expanded[[first_col] + socio_cols + remaining]

    return expanded


def _preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same numeric/categorical encodings used during training."""

    processed = encode_ordinal_features(df)
    processed = process_additional_features(processed)
    processed = drop_extra_features(processed)
    return processed


def _align_feature_space(
    df: pd.DataFrame,
    expected_features: list[str],
    feature_types: dict[str, str],
    *,
    enable_ohe: bool = True,
) -> pd.DataFrame:
    """Align inference dataframe to the training feature space using shared OHE logic."""

    encoded, _ = ohe_features(df, feature_types, enable=enable_ohe)

    for col in expected_features:
        if col not in encoded.columns:
            encoded[col] = 0

    encoded = encoded[[col for col in expected_features]]
    return encoded


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def _load_single_config(outputs_dir: Path, health_key: str) -> dict[str, object]:
    candidate = outputs_dir / _sanitize_name(health_key) / "config.json"
    if not candidate.exists():
        return {}

    try:
        payload = json.loads(candidate.read_text())
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}

def _load_model_from_config(outputs_dir: Path, config: dict[str, object], health_key: str) -> Pipeline | None:
    model_file = config.get("model_file") if isinstance(config, dict) else None
    if not model_file:
        return None

    candidate = outputs_dir / _sanitize_name(health_key) / str(model_file)
    if not candidate.exists():
        return None

    try:
        return joblib.load(candidate)
    except Exception:
        return None


def _discover_health_keys(outputs_dir: Path) -> list[str]:
    """Find health keys by scanning output subfolders that contain a config file."""

    keys: list[str] = []
    for child in outputs_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "config.json").exists():
            keys.append(child.name)
    return sorted(keys)


def infer_neighborhood_health_risks(
    best_models: dict[str, Pipeline] | None = None,
    feature_types_map: dict[str, dict[str, str]] | None = None,
    morph_csv_path: str | Path = Path("data/morphology_data_cleaned.csv"),
    *,
    enable_ohe: bool = True,
    outputs_dir: str | Path = Path("outputs"),
) -> dict[str, pd.DataFrame]:
    """Score health risks per neighborhood across socio-demographic combinations.

    Args:
        best_models: Mapping from health category (e.g., "mental_health") to fitted estimator.
            If None, each model is loaded on demand using that category's config file.
        feature_types_map: Optional mapping from health category to feature_types dict used in training.
            When omitted, the function scans `outputs_dir` for per-health configs and uses the
            `feature_types` found in each `config.json`.

        morph_csv_path: Path to the neighborhoods CSV (cleaned morphology).
        enable_ohe: Whether to apply one-hot encoding during feature alignment (ablation toggle).
        outputs_dir: Directory containing persisted best models and configuration files.

    Returns:
        Dict[str, pd.DataFrame]: For each health category, a dataframe containing
        neighborhood id, socio-demographic combination, and the predicted risk column.
    """

    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(exist_ok=True)

    best_models = best_models or {}
    feature_types_map = feature_types_map or {}

    expanded = _expand_neighborhood_grid(morph_csv_path, socio_config=SOCIO_DEMOGRAPHIC_VALUES)
    processed = _preprocess_for_inference(expanded)

    socio_cols = list(SOCIO_DEMOGRAPHIC_VALUES.keys())
    outputs: dict[str, pd.DataFrame] = {}

    health_keys = list(feature_types_map.keys()) or _discover_health_keys(outputs_dir)

    for health_key in health_keys:
        config = _load_single_config(outputs_dir, health_key)

        model = best_models.get(health_key)
        if model is None:
            model = _load_model_from_config(outputs_dir, config, health_key)
        if model is None:
            continue

        feature_types = feature_types_map.get(health_key) or config.get("feature_types")
        if not feature_types:
            continue

        target_type = feature_types.get("target", "binary")
        expected_features = [name for name in feature_types.keys() if name != "target"]

        aligned = _align_feature_space(
            processed,
            expected_features=expected_features,
            feature_types=feature_types,
            enable_ohe=config.get("enable_ohe", enable_ohe),
        )

        scores = _predict_scores(model, aligned, target_type=target_type)
        scores = np.clip(scores, 0.0, 1.0)

        result = expanded.copy()
        result.insert(1 + len(socio_cols), f"risk_{health_key}", scores)

        outputs[health_key] = result

    return outputs

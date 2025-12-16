from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
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


def build_scaler_step(use_standard_scaling: bool) -> tuple[str, StandardScaler] | None:
    """Return a StandardScaler pipeline step when enabled (ablation-friendly)."""

    if not use_standard_scaling:
        return None
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


def infer_neighborhood_health_risks(
    best_models: dict[str, Pipeline],
    feature_types_map: dict[str, dict[str, str]],
    morph_csv_path: str | Path = Path("data/morphology_data_cleaned.csv"),
    socio_config: dict[str, list] = SOCIO_DEMOGRAPHIC_VALUES,
    *,
    enable_ohe: bool = True,
) -> dict[str, pd.DataFrame]:
    """Score health risks per neighborhood across socio-demographic combinations.

    Args:
        best_models: Mapping from health category (e.g., "mental_health") to fitted estimator.
        feature_types_map: Mapping from health category to feature_types dict used in training.
        morph_csv_path: Path to the neighborhoods CSV (cleaned morphology).
        socio_config: Dict of socio-demographic feature -> list of possible values.
        enable_ohe: Whether to apply one-hot encoding during feature alignment (ablation toggle).

    Returns:
        Dict[str, pd.DataFrame]: For each health category, a dataframe containing
        neighborhood id, socio-demographic combination, and the predicted risk column.
    """

    expanded = _expand_neighborhood_grid(morph_csv_path, socio_config=socio_config)
    processed = _preprocess_for_inference(expanded)

    socio_cols = list(socio_config.keys())
    outputs: dict[str, pd.DataFrame] = {}

    for health_key, model in best_models.items():
        if model is None:
            continue

        feature_types = feature_types_map.get(health_key)
        if not feature_types:
            continue

        target_type = feature_types.get("target", "binary")
        expected_features = [name for name in feature_types.keys() if name != "target"]

        aligned = _align_feature_space(
            processed,
            expected_features=expected_features,
            feature_types=feature_types,
            enable_ohe=enable_ohe,
        )

        scores = _predict_scores(model, aligned, target_type=target_type)
        scores = np.clip(scores, 0.0, 1.0)

        result = expanded.copy()
        result.insert(1 + len(socio_cols), f"risk_{health_key}", scores)

        outputs[health_key] = result

    return outputs

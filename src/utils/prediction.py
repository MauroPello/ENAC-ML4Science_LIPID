from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.feature_config import SOCIO_DEMOGRAPHIC_VALUES
from src.utils.pipeline import (
    drop_extra_features,
    encode_ordinal_features,
    ohe_features,
    process_additional_features,
)


def collect_coefficients(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
) -> list[dict[str, float]]:
    """Collect feature coefficients or importance scores from a model.

    Args:
        name (str): Name of the model.
        model (Pipeline): Trained model pipeline.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        random_state (int): Random state for reproducibility.

    Returns:
        list[dict[str, float]]: List of dictionaries containing feature importance/coefficent info.
    """
    final_estimator = model.named_steps.get("model", model)

    if hasattr(final_estimator, "coef_"):
        coefs = np.asarray(final_estimator.coef_)
        if coefs.ndim > 1:
            coefs = np.mean(np.abs(coefs), axis=0)
        values = coefs
    elif hasattr(final_estimator, "feature_importances_"):
        values = np.asarray(final_estimator.feature_importances_)
    else:
        perm = permutation_importance(
            model,
            X_train,
            y_train,
            n_repeats=5,
            random_state=random_state,
            n_jobs=-1,
        )
        values = np.asarray(perm.importances_mean)

    return [
        {
            "model": name,
            "feature": feature_name,
            "coefficient": float(coef_value),
        }
        for feature_name, coef_value in zip(X_train.columns, values)
    ]


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
) -> pd.DataFrame:
    """Align inference dataframe to the training feature space using shared OHE logic."""

    encoded, _ = ohe_features(df, feature_types)

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
    display_columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Score health risks per neighborhood across socio-demographic combinations.

    Args:
        best_models: Mapping from health category (e.g., "mental_health") to fitted estimator.
        feature_types_map: Mapping from health category to feature_types dict used in training.
        morph_csv_path: Path to the neighborhoods CSV (cleaned morphology).
        socio_config: Dict of socio-demographic feature -> list of possible values.

    Returns:
        Dict[str, pd.DataFrame]: For each health category, a dataframe containing
        neighborhood id, socio-demographic combination, and the predicted risk column.
    """

    expanded = _expand_neighborhood_grid(morph_csv_path, socio_config=socio_config)
    extra_cols: list[str] = []
    if display_columns:
        extra_cols = [col for col in display_columns if col in expanded.columns]
        missing = [col for col in display_columns if col not in expanded.columns]
        if missing:
            print(f"Columns not found in morphology data and skipped: {missing}")

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
        )

        scores = _predict_scores(model, aligned, target_type=target_type)
        scores = np.clip(scores, 0.0, 1.0)

        result = expanded[["id"] + socio_cols + extra_cols].copy()
        result[f"risk_{health_key}"] = scores

        outputs[health_key] = result

    return outputs

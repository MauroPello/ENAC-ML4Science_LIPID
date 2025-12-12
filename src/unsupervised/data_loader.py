"""
Data loader module for the unsupervised analysis pivot.
Handles loading the integrated dataset and splitting it into Environmental and Morphological views.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.feature_config import EQ_FEATURES, MORPHOLOGY_FEATURES


def load_and_split_data(
    filepath: str | Path, drop_na: bool = True, scale: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset and split it into Environmental Quality (X) and Morphology (Y) views.

    Args:
        filepath (str | Path): Path to the CSV file.
        drop_na (bool): Whether to drop rows with missing values. Defaults to True.
        scale (bool): Whether to standardize the features (mean=0, std=1). Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - X: DataFrame with Environmental Quality features.
            - Y: DataFrame with Morphology features.
            - meta: DataFrame with metadata (typology, PCs, etc.).
    """
    # Read CSV with proper separators and error handling
    df = pd.read_csv(filepath, sep=";", decimal=",", on_bad_lines="skip")

    # Normalize column names: lowercase and replace spaces with underscores
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns
    ]

    # Filter columns that exist in the config
    eq_cols = [c for c in EQ_FEATURES if c in df.columns]
    morph_cols = [c for c in MORPHOLOGY_FEATURES if c in df.columns]

    if not eq_cols:
        raise ValueError("No Environmental Quality features found in the dataset.")
    if not morph_cols:
        raise ValueError("No Morphology features found in the dataset.")

    X = df[eq_cols]
    Y = df[morph_cols]

    # Align indices (just in case) and handle NaNs
    combined = pd.concat([X, Y], axis=1)

    if drop_na:
        combined = combined.dropna()

    # Split back
    X = combined[eq_cols]
    Y = combined[morph_cols]

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    Y = Y.select_dtypes(include=[np.number])

    # Update column lists to match filtered data
    eq_cols = X.columns.tolist()
    morph_cols = Y.columns.tolist()

    # Extract metadata (PCs and Typology)
    meta_cols = ["typology", "q_cluster"] + [
        c for c in df.columns if c.startswith("pc")
    ]
    meta = df[meta_cols].copy()
    meta = meta.loc[X.index]  # Align with filtered data

    if scale:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_x.fit_transform(X)
        Y_scaled = scaler_y.fit_transform(Y)

        X = pd.DataFrame(X_scaled, columns=eq_cols, index=X.index)
        Y = pd.DataFrame(Y_scaled, columns=morph_cols, index=Y.index)

    return X, Y, meta

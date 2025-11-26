"""Feature configuration utilities used across the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

DEFAULT_CONTINUOUS_FEATURES: List[str] = [
    "PC1",
    "PC2",
    "PC3",
    "PC4",
    "PC5",
    "PC6",
    "lst_mean",
    "solar_summ",
    "solar_wint",
    "pm10_mean",
    "pm25_mean",
    "no2_mean",
    "noiseday_m",
    "noisenight",
    "points_sleep_deprivation",
    "sleeping_hours",
    "bedtime_hour",
    "GHQ12_score",
]

DEFAULT_CATEGORICAL_FEATURES: List[str] = [
    "typology",
    "sex",
    "income",
    "education_level",
    "age_bin",
]

DEFAULT_BINARY_FEATURES: List[str] = [
    "heart_failure",
    "heart_rhythm",
    "sleep_disorder_hot",
    "d_breath_respiratory",
    "d_breath_asthma",
]

POSSIBLE_TARGETS: List[str] = [
    "points_sleep_deprivation",
    "sleeping_hours",
    "sleep_disorder_hot",
    "bedtime_hour",
    "GHQ12_score",
    "heart_failure",
    "heart_rhythm",
    "d_breath_respiratory",
    "d_breath_asthma",
]


@dataclass(frozen=True)
class FeatureSets:
    """Container describing available feature groups."""

    continuous: Tuple[str, ...]
    categorical: Tuple[str, ...]
    binary: Tuple[str, ...]

    @classmethod
    def default(cls) -> "FeatureSets":
        return cls(
            continuous=tuple(DEFAULT_CONTINUOUS_FEATURES),
            categorical=tuple(DEFAULT_CATEGORICAL_FEATURES),
            binary=tuple(DEFAULT_BINARY_FEATURES),
        )

    def as_allowed_columns(self) -> set:
        allowed = set(self.continuous)
        allowed.update(self.categorical)
        allowed.update(self.binary)
        return allowed


def determine_target_type(target_feature: str, feature_sets: FeatureSets) -> str:
    """Infer whether the selected target is continuous, categorical, or binary."""
    if target_feature in feature_sets.continuous:
        return "continuous"
    if target_feature in feature_sets.categorical:
        return "categorical"
    if target_feature in feature_sets.binary:
        return "binary"
    raise ValueError(f"Target feature '{target_feature}' is not recognized.")

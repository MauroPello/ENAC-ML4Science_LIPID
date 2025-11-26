from typing import Mapping


ALL_CONTINUOUS_FEATURES: list[str] = [
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

ALL_CATEGORICAL_FEATURES: list[str] = [
    "typology",
    "sex",
    "income",
    "education_level",
    "age_bin",
]

ALL_BINARY_FEATURES: list[str] = [
    "heart_failure",
    "heart_rhythm",
    "sleep_disorder_hot",
    "d_breath_respiratory",
    "d_breath_asthma",
]

POSSIBLE_TARGET_FEATURES: list[str] = [
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


def determine_target_type(
    target_feature: str,
    feature_types: Mapping[str, str] | None = None,
) -> str:
    """Infer whether the selected target is continuous, categorical, or binary."""

    if feature_types is not None and target_feature in feature_types:
        mapped_type = feature_types[target_feature]
        if mapped_type not in {"continuous", "categorical", "binary"}:
            raise ValueError(
                f"Target feature '{target_feature}' has unsupported type '{mapped_type}'."
            )
        return mapped_type

    if target_feature in ALL_CONTINUOUS_FEATURES:
        return "continuous"
    if target_feature in ALL_CATEGORICAL_FEATURES:
        return "categorical"
    if target_feature in ALL_BINARY_FEATURES:
        return "binary"
    raise ValueError(f"Target feature '{target_feature}' is not recognized.")

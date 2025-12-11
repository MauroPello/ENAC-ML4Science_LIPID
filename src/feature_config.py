EQ_FEATURES: list[str] = [
    "lst_mean",
    "solar_summ",
    "solar_wint",
    "pm10_mean",
    "pm25_mean",
    "no2_mean",
    "noiseday_m",
    "noisenight",
]

MORPHOLOGY_FEATURES: list[str] = [
    "building_height",
    "height_varability",
    "sky_view_factor",
    "frontal_area_index",
    "water_cover_fraction",
    "impervious_surface_cover_fraction",
    "building_cover_fraction",
    "pervious_surface_cover_fraction",
    "aspect_ratio",
    "intersections",
    "length_n-s",
    "length_ne-sw",
    "length_se-nw",
    "length_e-w",
    "length_primary_road",
    "length_secondary_road",
    "length_railway",
    "neighbourhood_type",
]

ALL_CONTINUOUS_FEATURES: list[str] = [
    # "pc1",
    # "pc2",
    # "pc3",
    # "pc4",
    # "pc5",
    # "pc6",
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
    # "bedtime_hour", removed for now
    "GHQ12_score",
    # ordinal encoded features
    "income",
    "education_level",
    "age_bin",
    "building_height",
    "height_varability",
    "sky_view_factor",
    "frontal_area_index",
    "water_cover_fraction",
    "impervious_surface_cover_fraction",
    "building_cover_fraction",
    "pervious_surface_cover_fraction",
    "aspect_ratio",
    "intersections",
    "length_n-s",
    "length_ne-sw",
    "length_se-nw",
    "length_e-w",
    "length_primary_road",
    "length_secondary_road",
    "length_railway",
]

ALL_CATEGORICAL_FEATURES: list[str] = [
    "typology",
    "neighbourhood_type",
]

ALL_BINARY_FEATURES: list[str] = [
    "sex",
    "heart_failure",
    "heart_rhythm",
    "sleep_disorder_hot",
    "d_breath_respiratory",
    "d_breath_asthma",
]

CARDIOVASCULAR_FEATURES: list[str] = [
    "heart_failure",
    "heart_rhythm",
]

SLEEP_DISORDER_FEATURES: list[str] = [
    "points_sleep_deprivation",
    "sleeping_hours",
    "sleep_disorder_hot",
    "bedtime_hour",
]

MENTAL_HEALTH_FEATURES: list[str] = [
    "GHQ12_score",
]

RESPIRATORY_FEATURES: list[str] = [
    "d_breath_respiratory",
    "d_breath_asthma",
]

POSSIBLE_TARGET_FEATURES: list[str] = (
    CARDIOVASCULAR_FEATURES
    + SLEEP_DISORDER_FEATURES
    + MENTAL_HEALTH_FEATURES
    + RESPIRATORY_FEATURES
)

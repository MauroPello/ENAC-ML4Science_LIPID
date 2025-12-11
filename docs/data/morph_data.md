# Morphological and Environmental Data Summary

Here is the summary of the morphological and environmental quality split of the data available in the provided Microsoft Teams folder

## Morphological geometry (neighborhood-level)

Standardized descriptors of built form and street network; all are floats unless noted.

* `building_height`
* `height_varability`
* `sky_view_factor`
* `frontal_area_index`
* `water_cover_fraction`
* `impervious_surface_cover_fraction`
* `building_cover_fraction`
* `pervious_surface_cover_fraction`
* `aspect_ratio`
* `intersections`
* `length_n-s`
* `length_ne-sw`
* `length_se-nw`
* `length_e-w`
* `length_primary_road`
* `length_secondary_road`
* `length_railway`
* `neighbourhood_type` (categorical)

## Environmental quality (EQ features)

Continuous z-score indicators of exposure and climate.

* `lst_mean`
* `solar_summ`
* `solar_wint`
* `pm10_mean`
* `pm25_mean`
* `no2_mean`
* `noiseday_m`
* `noisenight`

## Typology

* `typology` (categorical; 11-cluster morphology/EQ classes, labels A-K)

## Notes

* All continuous variables are assumed to be Z-score standardized ($\mu = 0, \sigma^2 = 0$).
* PCA components (PC1–PC6) exist upstream but are currently excluded from the modeling feature set defined in `feature_config.py`.
* Structure is consistent across cities (e.g., Geneva, Bern/Zurich).
# Morphological and Environmental Data Summary

Here is the summary of the morphological and environmental quality split of the data available in the provided Microsoft Teams folder

## Sheet : Morphological Data

Core morphological features, already standardized and aggregated per neighborhood.

### Principal Components (derived from upstream data analysis and clustering):

These represent latent dimensions of urban morphology extracted via PCA from building and street-level descriptors most likely

`PC1` (float - dimensionless)
`PC2` (float - dimensionless)
`PC3` (float - dimensionless)
`PC4` (float - dimensionless)
`PC5` (float - dimensionless)
`PC6` (float - dimensionless)

### Typology:

The morphology-EQ clustering has notably produced 11 typologies and each neighborhood with its unique identifier is assigned to one of such categories
(for instance: dense historical centre, mixed residential, industrial, peri-urban or anything else)

* `typology` (category - [A - K])

### Thermal Indicator

Environmental temperature exposure estimated from land surface temperature. Originally [$°C$].

* `lst_mean` (float - standardized z-score)

### Solar Access Indicators

incoming solar radiation aggregated per season. Originally [$\frac{Wh}{m^2}$] or [$\frac{kWh}{m^2}$].

* `solar_summ` (float - standardized z-score)
* `solar_wint` (float - standardized z-score)

### Air Pollution Indicators

Annual mean concentrations of key pollutants. Originally [$\frac{\mu g}{m^3}$]

* `pm10_mean` (float - stadardized z-score)
* `pm25_mean` (float - stadardized z-score)
* `no2_mean` (float - stadardized z-score)

### Noise Indicators 

Environmental noise exposure during day and night-time. Originally [$dB$]

* `noiseday_m`  (float - stadardized z-score) probably
* `noisenight9` (float - stadardized z-score) probably

## Notes

* All continuous variables are assumed to be Z-score standardized ($\mu = 0, \sigma^2 = 0$)
* PCA components and normalized values have no physical units and should be interpreted relative to each other
* PCA has highlighted that the morphology-based clustering yields 11 typologies of neighborhood, useful for further correlation analysis with health-associated indices
* The datasets for Geneva and Bern/Zurich shares the same structure
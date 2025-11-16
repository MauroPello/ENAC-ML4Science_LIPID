# Morphological and Environmental Data Summary
Here is the summary of the morphological and environmental quality split of the data available in the provided Microsoft Teams folder

# Sheet : Morphological Data
Core morphological features, already standardized and aggregated per neighborhood.

# Principal Components (derived from upstream data analysis and clustering):
These represent latent dimensions of urban morphology extracted via PCA from building and street-level descriptors most likely

PC1 (float - dimensionless)
PC2 (float - dimensionless)
PC3 (float - dimensionless)
PC4 (float - dimensionless)
PC5 (float - dimensionless)
PC6 (float - dimensionless)

# Typology:
The morphology-EQ clustering has notably produced 11 typologies and each neighborhood with its unique identifier is assigned to one of such categories

(for instance: dense historical centre, mixed residential, industrial, peri-urban or anything else)

# Thermal Indicator
Environmental temperature exposure estimated from land surface temperature

* 'lst_mean' (float - standardized z-score) probably [°C]

# Solar Access Indicators
incoming solar radiation aggregated per season

* 'solar_summ' (float - standardized z-score) probably [Wh/m²] or [kWh/m²]

* 'solar_wint' (float - standardized z-score) probably [Wh/m²] or [kWh/m²]

# Air Pollution Indicators
Annual mean concentrations of key pollutants

* 'pm10_mean' (float - stadardized z-score) probably [µg/m³]
* 'pm25_mean' (float - stadardized z-score) probably [µg/m³]
* 'no2_mean' (float - stadardized z-score) probably [µg/m³]

# Noise Indicators 
Environmental noise exposure during day and night-time

* 'noiseday_m'  (float - stadardized z-score) probably [dB]
* 'noisenight' (float - stadardized z-score) probably [dB]

# Notes
- All continuous variables have been most likely z-score standardized (to check)

- PCA components have no physical units and should be interpreted relative to each other

- PCA has highlighted that the morphology-based clustering yields 11 typologies of neighborhood, useful for further correlation analysis with health-associated indices

-The datasets for Geneva and Bern/Zurich (probably will be processed later) share the same structure
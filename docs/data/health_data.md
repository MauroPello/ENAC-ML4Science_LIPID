# Health Data Summary

Here is the summary of the Health Split of the Data.
This description is High-level and is inspired from the Excel file that is found in the Documents available in Microsoft Teams.


## Shared Features

* participant_id (int)
* neighborhood_id (int)

## Socio-demographic covariates (participant-level)

* `age_bin` (ordinal int; binned age category)
* `sex` (binary int {0, 1})
* `income` (ordinal encoded)
* `education_level` (ordinal encoded)

## Health-related indicators (aligned with `feature_config.py`)

**Cardiovascular**

* `heart_failure` (binary int {0, 1})
* `heart_rhythm` (binary int {0, 1})

**Sleep disorder**

* `points_sleep_deprivation` (int)
* `sleeping_hours` (float, hours)
* `sleep_disorder_hot` (binary int {0, 1})
* `bedtime_hour` (time/string; kept as target/label only)

**Mental health**

* `GHQ12_score` (int [0, 12])

**Respiratory**

* `d_breath_respiratory` (binary int {0, 1})
* `d_breath_asthma` (binary int {0, 1})

Notes

* Binary features are encoded as 0/1 integers.
* Ordinal socio-demographic fields are already encoded for modeling.
* Deprecated metabolic indicators (e.g., diabetes, obesity) are currently out of scope of the modeling feature set.
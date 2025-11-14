# Health Data Summary

Here is the summary of the Health Split of the Data.
This description is High-level and is inspired from the Excel file that is found in the Documents available in Microsoft Teams.


## Shared Features

* participant_id (int)
* neighborhood_id (int)

## Sheet 1: `Participant_SocioDemograph_Data`

Background and contextual covariates

* `age` (int)
* `sex` (categorical)
* `income` (categorical)
* `education_level` (categorical)

## Sheet 2: `Participant_HEALTH_Data`

Health-related indicators

* `heart_failure` (binary int - {0, 1})
* `heart_rhythm` (binary int - {0, 1})
* `d_metabolic_diabetes_I` (binary int - {0, 1})
* `d_metabolic_diabetes_II` (binary int - {0, 1})
* `d_metabolic_obesity` (binary int - {0, 1})
* `d_breath_respiratory` (binary int - {0, 1})
* `d_breath_asthma` (binary int - {0, 1})
* `GHQ12_score` (int - [0, 12])
* `points_sleep_deprivation` (int - [tbd])
* `sleep_disorder_hot` (binary int - {0, 1})
* `sleeping_hours` (float - [tbd])
* `bedtime_hour` (time/string)
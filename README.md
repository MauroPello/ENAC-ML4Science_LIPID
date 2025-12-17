## Overview
End-to-end pipeline for linking neighborhood morphology, environmental quality, and health outcomes. The repo bundles preprocessing, target aggregation, supervised models (classification and constrained regression), unsupervised morphology–environment analysis, and result management utilities.

## Setup
- Python 3.12 (see `environment.yaml`) and pip/conda available.
- Create the environment (choose one):

```bash
conda env create -f environment.yaml  # or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
conda activate mlenv                  # if you used conda
python -m ipykernel install --user --name mlenv
```

## Data inputs
- Data under `data/`:
	- `data/morphology_data_cleaned.csv`: cleaned morphology per neighborhood (real, post-processed).
	- `data/morphology_data_integrated.csv`: integrated morphology/environment view (real and raw; used by unsupervised analysis).
	- `data/synthetic_health_data.xlsx`: synthetic health workbook used for demonstration when real health data is unavailable.
	- `data/ablation/*`: saved model predictions from ablation runs (classification targets).
- Default preprocessing expects two real sources when available:
	1) Morphology CSV with a `neighborhood_id` column (or `id`, auto-renamed) containing the features listed in [src/feature_config.py](src/feature_config.py).
	2) Health Excel workbook with sheets:
		 - `Participant_SocioDemograph_Data` (socio-demographics)
		 - `Participant_HEALTH_Data` (clinical/health indicators)

## Swapping synthetic for real data
1) Place your real health Excel in the repo `data/` folder.
2) Update the path to the health Excel passed to the `load_combined_dataset` function in notebooks (see `main.ipynb` or `notebooks/pipeline_with_results.ipynb`). Provide `socio_sheet` / `clinical_sheet` names if they differ.
3) Ensure columns match the configured features (see “Custom constants” below). Critical columns: morphology features, socio-demographics (`income`, `education_level`, `age`, `sex`), health indicators, and `neighborhood_id`.
4) Run the preprocessing cell; it will:
	 - Merge morphology + health on `neighborhood_id`.
	 - Derive `age_bin`, encode ordinal/categorical fields, convert times.
	 - Drop unused columns to the modeled feature set.

## Running the analyses
- **Supervised (core)**: `main.ipynb` — end-to-end load → preprocess → target aggregation → model search (classification/regression) → saving results.
- **Saved pipeline recap**: `notebooks/pipeline_with_results.ipynb` — curated run with stored outputs.
- **Unsupervised morphology–environment**: `notebooks/unsupervised_analysis.ipynb` using `load_and_split_data` from [src/unsupervised/data_loader.py](src/unsupervised/data_loader.py).
- **Ablation studies**: `notebooks/ablation_study/*.ipynb` plus their prediction CSVs under `data/ablation/`.
- To run: open the notebook in VS Code/Jupyter, select the `mlenv` kernel, and execute top to bottom. There is no CLI wrapper; notebooks orchestrate calls into the modules.

## Pipeline highlights (supervised)
- Preprocessing functions live in [src/utils/pipeline.py](src/utils/pipeline.py):
	- `assign_age_bins` → CDC/WHO-inspired bins mapped to expected sleep hours (bins/labels in [src/utils/pipeline.py](src/utils/pipeline.py#L109-L165), hours in [src/feature_config.py](src/feature_config.py#L62-L84)).
	- `encode_ordinal_features` / `process_additional_features` → encode socio-demographics, time-of-day, and binary flags ([src/utils/pipeline.py](src/utils/pipeline.py#L46-L106)).
	- `ohe_features` → optional one-hot encoding (toggle for ablations) ([src/utils/pipeline.py](src/utils/pipeline.py#L184-L246)).
	- `drop_extra_features` → keep only whitelisted predictors/targets ([src/utils/pipeline.py](src/utils/pipeline.py#L167-L182)).
	- Feature screening utilities (optional, univariate): [src/feature_selection/binary.py](src/feature_selection/binary.py) and [src/feature_selection/continuous.py](src/feature_selection/continuous.py)
		- Binary targets: chi-square for binary predictors, univariate logistic regression (odds ratios + p-values) for any predictor.
		- Continuous targets: Pearson/Spearman for continuous predictors, point-biserial for binary, univariate OLS (with dummy expansion for categoricals) + p-values.
	- Target aggregation in [src/target_definition/aggregate.py](src/target_definition/aggregate.py):
	- `cardiovascular` and `respiratory`: binary max over component flags.
	- `sleep_disorder`: continuous risk combining sleep duration deviation, deprivation points, and heat-season disorder floor.
	- `mental_health`: logistic transform of GHQ-12 score with threshold 4.
- Modeling:
	- Classification helpers: [src/predictive/classification.py](src/predictive/classification.py) (logistic, SVM, RF, k-NN) with class imbalance strategies (`class_weight`, SMOTE/oversample/undersample) and threshold search for best F1.
	- Regression helpers: [src/predictive/regression.py](src/predictive/regression.py) (linear, ridge/lasso, kernel ridge, RF, SVR, k-NN) wrapped in `TransformedTargetRegressor` to keep outputs in [0,1].
	- Standard toggles: `test_size` (default 0.2), `cv` folds (default 5), `use_standard_scaling` (default on), `refine_hyperparameters` (second-pass grid), `imbalance_strategy` in classification.
	- Grid-search pass 1 (default grids):
		- Classification (scoring=`recall`):
			- Logistic (ridge/lasso): `C` ∈ {0.01, 0.1, 1.0, 10.0}
			- Random Forest: `n_estimators` ∈ {100, 300}, `max_depth` ∈ {None, 10, 20}
			- SVM linear: `C` ∈ {0.01, 0.1, 1.0, 10.0}
			- SVM RBF: `C` ∈ {0.1, 1.0, 10.0}, `gamma` ∈ {scale, auto}
			- k-NN: `n_neighbors` ∈ {3, 5, 7}
		- Regression (scoring=`neg_mean_squared_error`):
			- Ridge: `alpha` ∈ {0.1, 1.0, 10.0}
			- Lasso: `alpha` ∈ {1e-4, 1e-3, 1e-2}
			- Kernel Ridge: `alpha` ∈ {0.1, 1.0, 10.0}
			- Random Forest: `n_estimators` ∈ {100, 300}, `max_depth` ∈ {None, 10, 20}
			- SVR RBF: `C` ∈ {0.1, 1.0, 10.0}, `gamma` ∈ {scale, auto}
			- k-NN: `n_neighbors` ∈ {3, 5, 7}
	- Grid-search pass 2 (when `refine_hyperparameters=True`):
		- Centers around best params from pass 1; multiplies/perturbs within ±20% for `C`, `alpha`, `gamma` numeric values; ±20 trees / ±2 depth steps for RF; ±2 neighbors for k-NN; keeps categorical options unchanged. Classification still scores on recall; regression refines on `r2`. Logic in `_get_refined_classification_grid` and `_find_best_threshold` in [src/predictive/classification.py](src/predictive/classification.py#L209-L320) and `_get_refined_regression_grid` in [src/predictive/regression.py](src/predictive/regression.py#L213-L293).
	- Decision thresholds (classification): after fitting best estimator, cross-val scores select the threshold that maximizes F1; used for test predictions when probability/score is available ([src/predictive/classification.py](src/predictive/classification.py#L150-L206)).

## Custom constants (all in [src/feature_config.py](src/feature_config.py))
- Feature groups:
	- Environmental quality: `EQ_FEATURES` (e.g., `lst_mean`, `pm25_mean`, `noiseday_m`).
	- Morphology: `MORPHOLOGY_FEATURES` (e.g., `building_height`, `sky_view_factor`, road lengths).
	- Full model whitelist: `ALL_CONTINUOUS_FEATURES`, `ALL_CATEGORICAL_FEATURES`, `ALL_BINARY_FEATURES`.
- Target components:
	- `CARDIOVASCULAR_FEATURES`, `SLEEP_DISORDER_FEATURES`, `MENTAL_HEALTH_FEATURES`, `RESPIRATORY_FEATURES` combined into `POSSIBLE_TARGET_FEATURES`.
- Sleep expectations by age bin: `EXPECTED_HOURS` used in sleep risk calculation.
- Socio-demographic allowed values: `SOCIO_DEMOGRAPHIC_VALUES` (income brackets, education levels, age bins, sex) used for ordinal encoding and synthetic grid expansion during inference.

## Where to tweak things quickly
- Feature lists and allowed socio-demographic values: [src/feature_config.py](src/feature_config.py).
- Preprocessing steps (age bins, ordinal encodings, OHE toggle, feature whitelist): [src/utils/pipeline.py](src/utils/pipeline.py).
- Target definitions (risk formulas/thresholds): [src/target_definition/aggregate.py](src/target_definition/aggregate.py).
- Univariate feature screening thresholds/tests: [src/feature_selection/binary.py](src/feature_selection/binary.py) and [src/feature_selection/continuous.py](src/feature_selection/continuous.py).
- Model lists and default grids: [src/predictive/classification.py](src/predictive/classification.py) and [src/predictive/regression.py](src/predictive/regression.py).
- Grid refinement rules and threshold tuning: `_get_refined_classification_grid`, `_find_best_threshold` in [src/predictive/classification.py](src/predictive/classification.py#L150-L320); `_get_refined_regression_grid` in [src/predictive/regression.py](src/predictive/regression.py#L213-L293).

## Extending models
- To add another estimator to the search lists:
	1) In [src/predictive/classification.py](src/predictive/classification.py) or [src/predictive/regression.py](src/predictive/regression.py), append a `(name, pipeline)` pair to the `models` list (classification) or `_build_regression_models` return list (regression). Use `assemble_steps`/`build_scaler_step` if you need scaling.
	2) Add a matching entry in `param_grids` keyed by the same `name`, with parameters using the pipeline step prefix (e.g., `model__C`, or `regressor__model__C` for wrapped regressors).
	3) (Optional) Adjust `_get_refined_classification_grid` / `_get_refined_regression_grid` if the new hyperparameters need custom refinement logic.
	4) Keep `refine_hyperparameters` toggled to leverage the two-pass search automatically for the new model.

## Outputs and artifacts
- Ablation prediction files: `data/ablation/<encoding>/<target>_y_test_predictions.csv` for quick comparison.
- Unsupervised outputs: produced in notebook cells (UMAP/cluster plots) using `notebooks/unsupervised_analysis.ipynb`.
- Best fitted model per dataset: saved as `outputs/<dataset>/<model>.joblib` with metadata in `outputs/<dataset>/config.json` (model name, fitted file, optional threshold, feature types, and OHE flag) for notebook or batch inference.
- Inference helper: `infer_neighborhood_health_risks` in [src/utils/prediction.py](src/utils/prediction.py) scores all socio-demographic combinations per neighborhood and returns dataframes you can export.

## Tips for first run
- Start with `main.ipynb` on the synthetic data to confirm the environment.
- When switching to real data, validate column names against `feature_config.py` and rerun preprocessing before modeling.
- Use `ResultsManager.list_all()` in a notebook to see saved experiments; `get_latest(target_variable)` fetches the most recent per target.

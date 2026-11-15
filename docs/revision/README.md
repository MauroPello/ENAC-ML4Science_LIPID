# Co-author revision: verified results and methods

These analyses address the request for consistent SHAP explanations of the three selected classifiers, a sociodemographic-only baseline, and two methodological confirmations. The original fitted model artifacts are preserved. Aggregate results below were verified against the original workbook on 15 November 2026.

## Results for the response to Hanieh

### Sociodemographic baseline

The baseline uses the existing ordinal-coded **age bin, income and education level**, the same target-specific participants, the same stratified 80/20 split (`random_state=42`), and the selected estimator's tuned hyperparameters. Each variant's decision threshold is estimated by the same F1-maximising procedure on five-fold out-of-fold **training** probabilities. Test data do not select the threshold.

| Outcome | Selected model | Full F1 | Socio-only F1 | Full ROC-AUC | Socio-only ROC-AUC | Test n / positive |
|---|---|---:|---:|---:|---:|---:|
| Cardiovascular | RBF SVM | 0.3223 | 0.2880 | 0.7571 | 0.7334 | 423 / 41 |
| Mental health | Random Forest | 0.4142 | 0.3432 | 0.5868 | 0.5429 | 278 / 68 |
| Respiratory | Linear SVM | 0.1460 | 0.1786 | 0.6749 | 0.6121 | 423 / 16 |

All three full-model control fits reproduce the committed notebook's published metrics within 1e-6 in the x86-64 reference run. [baseline_comparison.csv](baseline_comparison.csv) includes full-precision scores, thresholds and paired differences. [sociodemographic_baseline_x86_64.csv](sociodemographic_baseline_x86_64.csv) also includes confidence intervals and the optional retuned sensitivity analysis.

Paired 95% percentile intervals for **full minus socio-only**, from 2,000 test-set bootstrap resamples with models and thresholds fixed:

| Outcome | F1 difference (95% CI) | ROC-AUC difference (95% CI) |
|---|---:|---:|
| Cardiovascular | +0.0343 (+0.004, +0.061) | +0.0238 (−0.044, +0.095) |
| Mental health | +0.0710 (−0.013, +0.160) | +0.0439 (−0.045, +0.135) |
| Respiratory | −0.0326 (−0.137, +0.074) | +0.0628 (−0.027, +0.167) |

The evidence for incremental predictive value is mixed and uncertain. All AUC-difference intervals include zero; respiratory F1 favours the baseline. The mental-health comparison is not nested: its selected full model includes age but excludes income and education, whereas the requested baseline includes all three controls. It compares two predictor sets rather than isolating the addition of neighbourhood predictors to identical individual controls. These intervals condition on the fitted analysis and do not include all selection, tuning, retraining or neighbourhood-cluster uncertainty.

Retuning the baseline separately gives F1/AUC 0.2880/0.7397, 0.3432/0.5407 and 0.1538/0.6199, respectively. This optional check leaves the main interpretation unchanged and must remain separate from the requested fixed-hyperparameter baseline.

### SHAP replacement for Figure 5

Use [shap_per_feature.csv](shap_per_feature.csv) for feature-level bars and [shap_per_group.csv](shap_per_group.csv) for descriptive aggregates. `mean_abs_shap` is in probability units: 0.01 corresponds to one percentage point of predicted probability. The SHAP analysis does not refit any model.

Suggested caption/method wording:

> Feature importance is the mean absolute interventional SHAP value for the positive-class predicted probability, computed using a model-agnostic permutation explainer applied to each saved complete prediction pipeline. Each model uses 100 background participants sampled with seed 42 from its final fitted cohort and 1,000 evaluations per unique predictor profile. All participants in that cohort contribute to the summary; duplicate profiles are restored with their original multiplicity. The final models were fitted on the full target-specific datasets after model selection, so these are descriptive explanations of the final fitted models.

There are 2,112 cardiovascular, 1,389 mental-health and 2,112 respiratory participants. All row-level SHAP values are finite and reconstruct the saved models' probabilities to numerical precision (maximum exported-value error below 3e-15). Exact interventional TreeSHAP confirms the Random Forest permutation result: feature-rank correlation 1.0000, maximum difference in mean absolute attribution 0.000243 probability units, and relative total absolute difference 0.45%.

A sensitivity check on 32 fixed sampled rows per outcome (`random_state=123`) compared 1,000 with 4,000 evaluations, then changed the 100-row background and permutation seed from 42 to 43:

| Outcome | Rank correlation: evaluation budget | Rank correlation: background/seed |
|---|---:|---:|
| Cardiovascular | 0.9985 | 0.9820 |
| Mental health | 0.9989 | 0.9864 |
| Respiratory | 1.0000 | 1.0000 |

Cardiovascular's top four remained unchanged in this sample, with the fifth changing under the alternate background. Mental-health and respiratory top-five orderings were unchanged. This is sample-based sensitivity evidence, not a full-cohort multi-background convergence proof. The supporting aggregate values are in [shap_sensitivity_summary.csv](shap_sensitivity_summary.csv).

The method and units are consistent across classifiers; cohort composition, prevalence, probability behaviour and backgrounds still differ. SHAP shares are within-model relative attributions, not causal health effects or percentages of outcome variance explained. Independent masking can produce combinations of correlated environmental measurements or one-hot columns that do not occur in the data. Group summaries sum row-level feature attributions before taking their absolute value; they are descriptive aggregates of that attribution game, not jointly masked categorical-feature SHAP. `direction_corr(feature, shap)` is descriptive correlation, not a causal or necessarily monotonic effect.

### Imputation confirmation

**Yes, income and education were imputed before splitting.** The notebook ordinal-encodes them and fills missing entries with medians computed on the full merged sample, before target-specific aggregation, screening and splitting. There are 290 missing income entries and 23 missing education entries.

For this workbook, both medians equal 2.0 in the full merged sample, every target's outer training split, and all five corresponding inner training folds. Training-only median estimation would therefore produce identical filled values for these checked splits. The historical operation order should still be disclosed. See [imputation_medians.json](imputation_medians.json).

### Respiratory coefficient confirmation

**Yes, the committed respiratory importance table contains absolute coefficients.** `collect_coefficients()` applies `np.mean(np.abs(coefs), axis=0)` to the SVC coefficient array. The saved respiratory SVM actually has 13 negative and 9 positive coefficients. These magnitudes describe the pipeline's standardized predictor space and discard direction; all-positive bars do not imply all predictors increase risk. The manuscript's external plotting step was not available for separate inspection.

## Important sensitivity: screening before the split

The original notebook screens predictors using **p <= 0.5** association tests and **VIF <= 50** before splitting. The new script's earlier description of a 0.1 cutoff was incorrect; the computation reproduces the original 0.5 cutoff.

Moving the association/VIF screening to the outer training rows, with the same split and original tuning procedure, gives the following results for the same selected estimator families:

| Outcome | Predictors: original → training-only | Original F1 | Training-only F1 | Original AUC | Training-only AUC |
|---|---:|---:|---:|---:|---:|
| Cardiovascular | 20 → 14 | 0.3223 | 0.2755 | 0.7571 | 0.7496 |
| Mental health | 22 → 19 | 0.4142 | 0.3939 | 0.5868 | 0.5822 |
| Respiratory | 22 → 15 | 0.1460 | 0.1192 | 0.6749 | 0.6341 |

This materially affects the selected predictors and scores and should be disclosed alongside the historical comparison. SHAP does not remove evaluation leakage from the earlier feature selection. The sensitivity isolates outer-split screening; imputation remains full-sample and screening is not repeated inside inner CV folds. It is not a fully nested, leakage-free evaluation. “Best model” fields in the CSV are exploratory test-set selections; the fixed selected-model rows above are the relevant sensitivity comparison. Some auxiliary logistic fits and association tests emit convergence warnings; warnings remain visible. These are limitations of the reproduced procedure, not grounds to silently change the original model/hyperparameter definitions.

See [screening_train_only_x86_64.csv](screening_train_only_x86_64.csv). The cardiovascular exploratory winner changes to Lasso logistic regression (F1 0.2880), further illustrating selection sensitivity. The response should present these results as sensitivity evidence and avoid a blanket claim of robust neighbourhood gains across outcomes.

## Reproduction and checks

Install the pinned versions in [requirements.txt](../../requirements.txt). The baseline/screening reference run used **Python 3.12.11 on x86-64**. ARM Python 3.12.14 produced a slightly different full Random Forest refit (F1 0.409091, AUC 0.582073); both SVM controls and all three fixed-hyperparameter socio-only metrics agreed. Use a native x86-64 environment or an x86-64 Python process under Rosetta on Apple Silicon when reproducing the reference metrics. The SHAP computation used ARM Python 3.12.14 and SHAP 0.50.0 on the saved models, without refitting.

Provide the original private workbook either at `data/health_data_final.xlsx` or with `HEALTH_DATA_PATH`. Its sheets must be `SocioDemograph` and `HEALTH`. The synthetic workbook cannot reproduce these results. Input/model checksums and target-specific data fingerprints are recorded in [reference_manifest.json](reference_manifest.json); the scripts reject mismatches rather than explaining different data with the original models.

```bash
HEALTH_DATA_PATH=/path/to/health_data_final.xlsx python scripts/sociodemographic_baseline.py
HEALTH_DATA_PATH=/path/to/health_data_final.xlsx python scripts/screening_leakage_check.py
HEALTH_DATA_PATH=/path/to/health_data_final.xlsx python scripts/shap_analysis.py
HEALTH_DATA_PATH=/path/to/health_data_final.xlsx python scripts/shap_sensitivity.py
python -m unittest discover -s tests -v
```

The baseline stops before export if any full-model control fails to reproduce the published metrics. `--allow-mismatch` explicitly permits a separately named `*_sensitivity.csv` platform run with a visible warning. Threshold-CV failures propagate instead of silently falling back to 0.5. SHAP exports reject malformed/non-finite values or additivity errors above 1e-6. Additivity alone is not a convergence test; the separate budget/background and exact-tree comparisons address that question.

Generated outputs and per-run metadata go to ignored `results/`; the curated aggregate deliverables in this directory are tracked. Participant-level predictors and row-level SHAP files remain local. Confidence intervals are exported as numeric low/high columns. The full permutation SHAP run took about 25 minutes on the review machine; `LOKY_MAX_CPU_COUNT` can limit worker processes.

For the correction verification, the original full-cohort permutation values were revalidated and summarized through the corrected code, while exact TreeSHAP was recomputed. The permutation algorithm/settings were unchanged. The baseline and screening scripts were rerun end to end on x86-64 after correction. Run sidecars record package versions, settings, code/input/model fingerprints and split-index hashes. Test cases require no private workbook and exercise guard failures plus aggregate-result consistency.

"""Effect of running the univariate screening on the training split only.

main.ipynb screens predictors (p <= 0.5 association tests, VIF <= 50) on ALL rows before
the 80/20 split, so the test set influences which predictors are kept. This script
repeats the whole classification pipeline with the screening fitted on the training
rows only (same split, same grids, same threshold rule) and compares the outcome with
the committed results. Run it in the x86-64 reference environment for Random Forest parity.
This isolates outer-split screening: imputation remains full-sample and selection
is not repeated inside the inner CV folds. The best-model rows are exploratory
test-set selections, not independent estimates of a newly selected model's performance.

Usage:  python scripts/screening_leakage_check.py
Output: results/screening_train_only_<arch>.csv and run metadata.
"""

from __future__ import annotations

import contextlib
import io
import platform

import numpy as np
import pandas as pd

import _rebuild as R
from src.predictive.classification import run_classification_models
from src.target_definition.aggregate import aggregate_health_targets

def main() -> None:
    df = R.load_preprocessed()
    all_types = R._feature_types_for(df)
    committed = R.build_datasets()
    rows = []
    for name in R.CLASSIFICATION_TARGETS:
        cfg = R.load_config(name)
        R.verify_against_config(name, committed[name]["data"], committed[name]["feature_types"])
        with contextlib.redirect_stdout(io.StringIO()):
            agg = aggregate_health_targets(df, target_feature=name, feature_types=all_types,
                                           drop_null_target_rows=R.DROP_ROWS_WITH_NULL_TARGET)
        pre = agg["data"]
        _, y, idx_train, _ = R.split(pre)  # identical split: same rows, same target
        with contextlib.redirect_stdout(io.StringIO()):
            data, ftypes = R._select_features(pre.copy(), dict(agg["feature_types"]),
                                              fit_index=pre.index[idx_train])
            res = run_classification_models(
                data.drop(columns="target").reset_index(drop=True), pd.Series(y),
                test_size=R.TEST_SIZE, random_state=R.RANDOM_STATE, imbalance_strategy="class_weight",
                use_standard_scaling=True, refine_hyperparameters=True, feature_types=ftypes,
            )["classification_results"]

        old_feats = [c for c in committed[name]["data"].columns if c != "target"]
        new_feats = [c for c in data.columns if c != "target"]
        if res.empty or not (res["model"] == cfg["model"]).any():
            raise RuntimeError(f"{name}: selected estimator failed during training-only screening")
        same_model = res.loc[res.model == cfg["model"]].iloc[0]
        if not np.isfinite(same_model[["F1", "ROC_AUC", "threshold"]].to_numpy(dtype=float)).all():
            raise RuntimeError(f"{name}: invalid metrics from training-only screening")
        best = res.iloc[0]
        rows.append({
            "target": name,
            "n_predictors_committed": len(old_feats),
            "n_predictors_train_only": len(new_feats),
            "only_in_committed": sorted(set(old_feats) - set(new_feats)),
            "only_in_train_only": sorted(set(new_feats) - set(old_feats)),
            "committed_model": cfg["model"],
            "committed_F1": R.PUBLISHED_METRICS[name][0],
            "committed_AUC": R.PUBLISHED_METRICS[name][1],
            "committed_model_F1_train_only_screening": same_model.F1,
            "committed_model_AUC_train_only_screening": same_model.ROC_AUC,
            "selected_model_threshold": same_model.threshold,
            "selected_model_hyperparameters": same_model.best_params,
            "best_model_train_only_screening": best.model,
            "best_F1_train_only_screening": best.F1,
            "best_AUC_train_only_screening": best.ROC_AUC,
        })
        print(f"\n=== {name} ===")
        print(res[["model", "F1", "ROC_AUC", "threshold"]].round(4).to_string(index=False))

    out = pd.DataFrame(rows)
    R.RESULTS_DIR.mkdir(exist_ok=True)
    path = R.RESULTS_DIR / f"screening_train_only_{platform.machine()}.csv"
    out.to_csv(path, index=False)
    R.write_run_metadata(
        path.with_suffix(".json"), committed,
        association_p_value=0.5, vif_threshold=50, test_size=R.TEST_SIZE,
        split_seed=R.RANDOM_STATE, screening_scope="outer training split only",
        imputation_scope="full merged sample", inner_cv_screening=False,
    )
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print(f"\n[{platform.machine()}]\n" + out.T.to_string())
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()

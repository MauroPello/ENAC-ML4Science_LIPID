"""Effect of running the univariate screening on the training split only.

main.ipynb screens predictors (p <= 0.1 association tests, VIF) on ALL rows before
the 80/20 split, so the test set influences which predictors are kept. This script
repeats the whole classification pipeline with the screening fitted on the training
rows only (same split, same grids, same threshold rule) and compares the outcome with
the committed results. Run it in the x86-64 environment for Random Forest parity.

Usage:  python scripts/screening_leakage_check.py
Output: results/screening_train_only.csv
"""

from __future__ import annotations

import contextlib
import io
import platform
import warnings

import pandas as pd

import _rebuild as R
from src.predictive.classification import run_classification_models
from src.target_definition.aggregate import aggregate_health_targets

warnings.filterwarnings("ignore")


def main() -> None:
    df = R.load_preprocessed()
    all_types = R._feature_types_for(df)
    committed = R.build_datasets()
    rows = []
    for name in R.CLASSIFICATION_TARGETS:
        cfg = R.load_config(name)
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
        same_model = res.loc[res.model == cfg["model"]].iloc[0]
        best = res.iloc[0]
        rows.append({
            "target": name,
            "n_predictors_committed": len(old_feats),
            "n_predictors_train_only": len(new_feats),
            "only_in_committed": sorted(set(old_feats) - set(new_feats)),
            "only_in_train_only": sorted(set(new_feats) - set(old_feats)),
            "committed_model": cfg["model"],
            "committed_model_F1_train_only_screening": same_model.F1,
            "committed_model_AUC_train_only_screening": same_model.ROC_AUC,
            "best_model_train_only_screening": best.model,
            "best_F1_train_only_screening": best.F1,
            "best_AUC_train_only_screening": best.ROC_AUC,
        })
        print(f"\n=== {name} ===")
        print(res[["model", "F1", "ROC_AUC", "threshold"]].round(4).to_string(index=False))

    out = pd.DataFrame(rows)
    R.RESULTS_DIR.mkdir(exist_ok=True)
    path = R.RESULTS_DIR / "screening_train_only.csv"
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print(f"\n[{platform.machine()}]\n" + out.T.to_string())
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()

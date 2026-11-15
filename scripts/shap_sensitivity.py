"""SHAP budget/background sensitivity on 32 fixed sampled rows per outcome.

Uses the saved models without refitting. Compare 1,000 vs 4,000 evaluations on the
original 100-row background, then another 100-row background/permutation seed.
This checks settings on a sample, not full-cohort multi-background convergence.

Usage: python scripts/shap_sensitivity.py
Outputs: results/shap_sensitivity_summary.csv, shap_sensitivity_checks.csv and metadata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import _rebuild as R
from shap_analysis import _explain_chunk, validate_values

N_ROWS = 32
ROW_SEED = 123
N_CHUNKS = 4
SCENARIOS = [
    ("original_bg_1000", 1000, 42, 42),
    ("original_bg_4000", 4000, 42, 42),
    ("alternate_bg_4000", 4000, 43, 43),
]


def main() -> None:
    datasets = R.build_datasets()
    summaries, checks = [], []
    for name in R.CLASSIFICATION_TARGETS:
        d = datasets[name]
        R.verify_against_config(name, d["data"], d["feature_types"])
        model, _ = R.load_committed_model(name)
        X = d["data"].drop(columns="target").reset_index(drop=True).astype(float)
        rows = X.sample(N_ROWS, random_state=ROW_SEED)
        collected = {}
        for label, budget, bg_seed, perm_seed in SCENARIOS:
            background = X.sample(100, random_state=bg_seed)
            chunks = np.array_split(np.arange(len(rows)), N_CHUNKS)
            parts = Parallel(n_jobs=-1)(
                delayed(_explain_chunk)(
                    model, background, rows.iloc[idx], perm_seed + k, max_evals=budget
                )
                for k, idx in enumerate(chunks) if len(idx)
            )
            values = np.vstack(parts)
            validate_values(
                values, model.predict_proba(rows)[:, 1],
                model.predict_proba(background)[:, 1].mean(), rows.shape,
            )
            mean_abs = pd.Series(np.abs(values).mean(axis=0), index=X.columns)
            collected[label] = mean_abs
            summaries.extend(
                {"target": name, "scenario": label, "feature": c, "mean_abs_shap": float(v)}
                for c, v in mean_abs.items()
            )
        low, high, alternate = (collected[s[0]] for s in SCENARIOS)
        checks.append({
            "target": name,
            "n_sampled_rows": N_ROWS,
            "budget_rank_spearman": low.corr(high, method="spearman"),
            "background_rank_spearman": high.corr(alternate, method="spearman"),
        })
        print(name, checks[-1], flush=True)

    R.RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(summaries).to_csv(R.RESULTS_DIR / "shap_sensitivity_summary.csv", index=False)
    pd.DataFrame(checks).to_csv(R.RESULTS_DIR / "shap_sensitivity_checks.csv", index=False)
    R.write_run_metadata(
        R.RESULTS_DIR / "shap_sensitivity_run.json", datasets,
        n_sampled_rows=N_ROWS, row_seed=ROW_SEED, n_chunks=N_CHUNKS,
        scenarios=SCENARIOS, background_rows=100,
    )


if __name__ == "__main__":
    main()

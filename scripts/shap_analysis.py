"""Consistent SHAP importances for the three committed classifiers (no refitting).

The importances previously shown for these models were not comparable across
domains (permutation importance on accuracy for the RBF SVM, |coef| for the linear
SVM, impurity importance for the Random Forest). This script uses one method for
all three:

  * explained quantity: predicted probability of the positive class,
    model.predict_proba(X)[:, 1] -- the same output used for thresholding and
    for the neighbourhood risk maps
  * explainer: model-agnostic shap.PermutationExplainer (interventional Shapley
    values) on the committed pipeline, so values are expressed in the original
    (unscaled) feature space and in probability units
  * background: the same number (100) of rows sampled with a fixed seed from each
    model's own training data
  * explained rows: every row each model was fitted on (the committed models were
    re-fitted on the full dataset). Identical feature rows are explained once and
    weighted by their multiplicity.

Exact interventional TreeSHAP is also computed for the Random Forest with the same
background, as a check on the permutation approximation.

Usage:  python scripts/shap_analysis.py      (needs the `shap` package)
Output: results/shap_per_feature.csv, results/shap_per_group.csv,
        results/shap_values_<target>.csv (row-level values, original row order)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed

import _rebuild as R
from src.feature_config import EQ_FEATURES, MORPHOLOGY_FEATURES

N_BACKGROUND = 100
MAX_EVALS = 1000  # per explained row; ~= 2*M evaluations per antithetic permutation
N_CHUNKS = 32
ADDITIVITY_TOLERANCE = 1e-6


def validate_values(values, predictions, base_value, expected_shape) -> float:
    """Reject malformed, non-finite or non-additive explanations before export."""
    values = np.asarray(values)
    predictions = np.asarray(predictions)
    if values.shape != expected_shape or predictions.shape != (expected_shape[0],):
        raise RuntimeError("SHAP values or predictions have unexpected dimensions")
    if not np.isfinite(values).all() or not np.isfinite(predictions).all() or not np.isfinite(base_value):
        raise RuntimeError("Non-finite SHAP values, predictions or base value")
    error = float(np.abs(base_value + values.sum(axis=1) - predictions).max())
    if error > ADDITIVITY_TOLERANCE:
        raise RuntimeError(f"SHAP additivity error {error:.3e} exceeds {ADDITIVITY_TOLERANCE:.1e}")
    return error


def relative_shares(magnitudes):
    total = float(magnitudes.sum())
    return magnitudes / total if total > 0 else magnitudes * 0.0


def feature_group(col: str) -> str:
    if col in R.SOCIO_FEATURES or col == "sex":
        return "Sociodemographic"
    if col in EQ_FEATURES:
        return "Environmental quality"
    if col.startswith("neighbourhood_type_"):
        return "Neighbourhood type"
    if col.startswith("typology_"):
        return "Typology"
    if col in MORPHOLOGY_FEATURES:
        return "Morphology"
    return "Other"


def _explain_chunk(model, background, rows, seed, max_evals=MAX_EVALS):
    f = lambda a: model.predict_proba(pd.DataFrame(a, columns=background.columns))[:, 1]  # noqa: E731
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.PermutationExplainer(f, masker, seed=seed)
    return explainer(rows, max_evals=max_evals, silent=True).values


def explain(model, X: pd.DataFrame, background: pd.DataFrame) -> np.ndarray:
    """Row-level SHAP values for every row of X (computed on unique rows)."""
    uniq, inverse = np.unique(X.to_numpy(dtype=float), axis=0, return_inverse=True)
    uniq = pd.DataFrame(uniq, columns=X.columns)
    chunks = np.array_split(np.arange(len(uniq)), N_CHUNKS)
    parts = Parallel(n_jobs=-1)(
        delayed(_explain_chunk)(model, background, uniq.iloc[idx], R.RANDOM_STATE + k)
        for k, idx in enumerate(chunks)
        if len(idx)
    )
    return np.vstack(parts)[inverse.ravel()]


def main() -> None:
    R.RESULTS_DIR.mkdir(exist_ok=True)
    datasets = R.build_datasets()
    per_feature, per_group = [], []
    validation = {}

    for name in R.CLASSIFICATION_TARGETS:
        d = datasets[name]
        R.verify_against_config(name, d["data"], d["feature_types"])
        model, cfg = R.load_committed_model(name)
        if list(model.classes_) != [0, 1]:
            raise RuntimeError(f"{name}: expected binary model classes [0, 1]")
        X = d["data"].drop(columns=["target"]).reset_index(drop=True).astype(float)
        background = X.sample(N_BACKGROUND, random_state=R.RANDOM_STATE)

        values = explain(model, X, background)

        # Additivity check: base value + sum(SHAP) must reproduce the prediction.
        base_value = model.predict_proba(background)[:, 1].mean()
        pred = model.predict_proba(X)[:, 1]
        max_err = validate_values(values, pred, base_value, X.shape)
        validation[name] = {
            "base_value": float(base_value), "mean_prediction": float(pred.mean()),
            "max_additivity_error": max_err,
        }

        shap_df = pd.DataFrame(values, columns=X.columns)
        shap_df.to_csv(R.RESULTS_DIR / f"shap_values_{name}.csv", index=False)

        mean_abs = shap_df.abs().mean()
        signed_corr = {
            c: (np.corrcoef(X[c], shap_df[c])[0, 1]
                if X[c].std() > 0 and shap_df[c].std() > 0 else np.nan)
            for c in X.columns
        }
        feat = pd.DataFrame(
            {
                "target": name,
                "model": cfg["model"],
                "feature": X.columns,
                "group": [feature_group(c) for c in X.columns],
                "mean_abs_shap": mean_abs.values,
                "share_of_total": relative_shares(mean_abs).values,
                "direction_corr(feature, shap)": [signed_corr[c] for c in X.columns],
            }
        ).sort_values("mean_abs_shap", ascending=False)
        feat["rank"] = np.arange(1, len(feat) + 1)

        if cfg["model"] == "Random Forest Classifier":
            tree = shap.TreeExplainer(
                model.named_steps["model"],
                data=background,
                feature_perturbation="interventional",
                model_output="probability",
            )
            exact_values = tree.shap_values(X)[..., 1]
            validation[name]["exact_tree_max_additivity_error"] = validate_values(
                exact_values, pred, float(tree.expected_value[1]), X.shape
            )
            exact = np.abs(exact_values).mean(axis=0)
            exact = pd.Series(exact, index=X.columns)
            feat["mean_abs_shap_exact_treeshap"] = feat["feature"].map(exact)
            validation[name]["permutation_exact_rank_spearman"] = float(mean_abs.corr(exact, method="spearman"))
            validation[name]["relative_L1_mean_abs_difference"] = float(
                (mean_abs - exact).abs().sum() / exact.sum()
            )

        per_feature.append(feat)

        # Group-level: sum row-level SHAP within a group (e.g. all one-hot dummies
        # of neighbourhood_type) before taking |.|, then average over rows.
        groups = pd.Series([feature_group(c) for c in X.columns], index=X.columns)
        grp_abs = shap_df.T.groupby(groups).sum().T.abs().mean()
        per_group.append(
            pd.DataFrame(
                {
                    "target": name,
                    "model": cfg["model"],
                    "group": grp_abs.index,
                    "n_features": groups.value_counts().reindex(grp_abs.index).values,
                    "mean_abs_shap": grp_abs.values,
                    "share_of_total": relative_shares(grp_abs).values,
                }
            ).sort_values("mean_abs_shap", ascending=False)
        )

        print(
            f"\n=== {name}: {cfg['model']} | rows={len(X)} | base value={base_value:.4f}"
            f" | mean prediction={pred.mean():.4f} | max additivity error={max_err:.2e}"
        )
        cols = ["rank", "feature", "group", "mean_abs_shap", "share_of_total",
                "direction_corr(feature, shap)"]
        if "mean_abs_shap_exact_treeshap" in feat:
            cols.append("mean_abs_shap_exact_treeshap")
        print(feat[cols].head(10).round(4).to_string(index=False))

    pd.concat(per_feature).to_csv(R.RESULTS_DIR / "shap_per_feature.csv", index=False)
    pd.concat(per_group).to_csv(R.RESULTS_DIR / "shap_per_group.csv", index=False)
    R.write_run_metadata(
        R.RESULTS_DIR / "shap_run.json", datasets,
        background_rows=N_BACKGROUND, max_evals=MAX_EVALS, n_chunks=N_CHUNKS,
        seed=R.RANDOM_STATE, additivity_tolerance=ADDITIVITY_TOLERANCE,
        shap_version=shap.__version__, validation=validation,
        explained_output="positive-class probability", explained_cohort="full fitted cohort",
    )
    print("\n=== Group-level mean |SHAP| (share of model total) ===")
    print(pd.concat(per_group).round(4).to_string(index=False))
    print(f"\nSaved results to {R.RESULTS_DIR}")


if __name__ == "__main__":
    R.RESULTS_DIR.mkdir(exist_ok=True)
    main()

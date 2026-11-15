"""Sociodemographic-only baseline for the three selected classifiers.

For each classification target this evaluates, on the SAME train/test split used
for the committed models:

  full                 selected features, tuned hyper-parameters (re-fit on the
                       training split; must reproduce the published test metrics)
  socio_fixed_hparams  age_bin + income + education_level only, SAME estimator and
                       SAME tuned hyper-parameters (the requested baseline)
  socio_retuned        same three predictors, hyper-parameters re-tuned with the
                       exact grid-search procedure of main.ipynb (sensitivity check:
                       the fixed hyper-parameters were tuned for the full feature set)

The decision threshold is re-derived for every variant with the same procedure as
main.ipynb (F1-maximising threshold on 5-fold out-of-fold training predictions),
because it is fitted from the training data rather than a free hyper-parameter.
The committed models in outputs/ are only read, never overwritten.

Usage:  python scripts/sociodemographic_baseline.py
The default run stops if a full-model control does not reproduce the published
metrics. --allow-mismatch writes a separately labelled platform sensitivity table.
Output: results/sociodemographic_baseline_<arch>[_sensitivity].csv and run metadata.
"""

from __future__ import annotations

import argparse
import platform

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

import _rebuild as R
from src.predictive.classification import (
    _apply_threshold,
    _safe_prediction_scores,
    run_classification_models,
)
from src.utils.prediction import build_scaler_step

N_BOOT = 2000


def _fit_eval(pipe, X, y, idx_train, idx_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=R.RANDOM_STATE)
    X_tr, X_te = X.iloc[idx_train], X.iloc[idx_test]
    y_tr, y_te = y[idx_train], y[idx_test]
    pipe.fit(X_tr, y_tr)
    # Same quantile search/tie rule as the original helper, but CV errors must
    # propagate instead of silently falling back to a threshold of 0.5.
    training_scores = cross_val_predict(
        pipe, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    if not np.isfinite(training_scores).all():
        raise RuntimeError("Non-finite out-of-fold training probabilities")
    candidates = np.unique(np.quantile(training_scores, np.linspace(0, 1, 101)))
    threshold = float(max(
        candidates,
        key=lambda t: f1_score(y_tr, _apply_threshold(training_scores, t), zero_division=0),
    ))
    scores = _safe_prediction_scores(pipe, X_te)
    if scores is None or not np.isfinite(scores).all():
        raise RuntimeError("Missing or non-finite test prediction scores")
    preds = _apply_threshold(scores, threshold)
    return {
        "F1": f1_score(y_te, preds, zero_division=0),
        "ROC_AUC": roc_auc_score(y_te, scores),
        "threshold": threshold,
        "_scores": scores,
        "_preds": preds,
    }


def _socio_pipeline(committed: Pipeline, socio_types: dict[str, str]) -> Pipeline:
    steps = []
    if "scaler" in committed.named_steps:
        steps.append(
            build_scaler_step(True, feature_types=socio_types, columns=R.SOCIO_FEATURES)
        )
    steps.append(("model", clone(committed.named_steps["model"])))
    return Pipeline(steps)


def _paired_bootstrap(y_te, full, base, rng):
    """95% percentile CIs on resampled test sets (thresholds fixed).

    Returns CIs for (full - baseline) F1 and AUC, plus each model's own F1 and AUC.
    """
    n = len(y_te)
    s = {k: [] for k in ("d_f1", "d_auc", "f1_full", "f1_base", "auc_full", "auc_base")}
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y_te[i].min() == y_te[i].max():
            continue
        f1f = f1_score(y_te[i], full["_preds"][i], zero_division=0)
        f1b = f1_score(y_te[i], base["_preds"][i], zero_division=0)
        aucf = roc_auc_score(y_te[i], full["_scores"][i])
        aucb = roc_auc_score(y_te[i], base["_scores"][i])
        for k, v in zip(s, (f1f - f1b, aucf - aucb, f1f, f1b, aucf, aucb)):
            s[k].append(v)
    return {k: tuple(float(x) for x in np.round(np.percentile(v, [2.5, 97.5]), 3)) for k, v in s.items()}


def _interval_fields(prefix, bounds):
    return {f"{prefix}_95CI_low": bounds[0], f"{prefix}_95CI_high": bounds[1]}


def main(*, allow_mismatch=False) -> None:
    arch = platform.machine()
    datasets = R.build_datasets()
    rows = []
    for name in R.CLASSIFICATION_TARGETS:
        d = datasets[name]
        R.verify_against_config(name, d["data"], d["feature_types"])
        committed, cfg = R.load_committed_model(name)
        X, y, idx_train, idx_test = R.split(d["data"])
        y_te = y[idx_test]

        socio_X = d["pre_selection"][R.SOCIO_FEATURES].reset_index(drop=True)
        socio_types = {c: "continuous" for c in R.SOCIO_FEATURES}
        assert set(R._feature_types_for(socio_X).values()) == {"continuous"}

        full = _fit_eval(clone(committed), X, y, idx_train, idx_test)
        matches_published = R.check_published_metrics(
            name, full["F1"], full["ROC_AUC"], allow_mismatch=allow_mismatch
        )
        base = _fit_eval(
            _socio_pipeline(committed, socio_types), socio_X, y, idx_train, idx_test
        )

        # Re-tuned sensitivity check: identical procedure to main.ipynb, socio only.
        retuned_res = run_classification_models(
            socio_X,
            pd.Series(y),
            test_size=R.TEST_SIZE,
            random_state=R.RANDOM_STATE,
            imbalance_strategy="class_weight",
            use_standard_scaling=True,
            refine_hyperparameters=True,
            feature_types={**socio_types, "target": "binary"},
        )["classification_results"]
        retuned = retuned_res.loc[retuned_res["model"] == cfg["model"]].iloc[0]

        boot = _paired_bootstrap(y_te, full, base, np.random.default_rng(R.RANDOM_STATE))
        hp = {
            k: v
            for k, v in committed.named_steps["model"].get_params().items()
            if k in ("C", "gamma", "kernel", "max_depth", "n_estimators", "class_weight")
        }
        common = {
            "target": name,
            "model": cfg["model"],
            "n_test": len(idx_test),
            "n_test_positive": int(y_te.sum()),
        }
        rows += [
            {
                **common,
                "variant": "full (selected features)",
                "n_predictors": X.shape[1],
                "F1": full["F1"],
                "ROC_AUC": full["ROC_AUC"],
                "threshold": full["threshold"],
                "hyperparameters": hp,
                **_interval_fields("F1", boot["f1_full"]),
                **_interval_fields("ROC_AUC", boot["auc_full"]),
                "matches_published": matches_published,
            },
            {
                **common,
                "variant": "socio only, same hyperparameters",
                "n_predictors": 3,
                "F1": base["F1"],
                "ROC_AUC": base["ROC_AUC"],
                "threshold": base["threshold"],
                "hyperparameters": hp,
                **_interval_fields("F1", boot["f1_base"]),
                **_interval_fields("ROC_AUC", boot["auc_base"]),
                **_interval_fields("delta_F1_full_minus_socio", boot["d_f1"]),
                **_interval_fields("delta_AUC_full_minus_socio", boot["d_auc"]),
            },
            {
                **common,
                "variant": "socio only, re-tuned hyperparameters",
                "n_predictors": 3,
                "F1": retuned["F1"],
                "ROC_AUC": retuned["ROC_AUC"],
                "threshold": retuned["threshold"],
                "hyperparameters": retuned["best_params"],
            },
        ]

    out = pd.DataFrame(rows)
    R.RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "_sensitivity" if allow_mismatch else ""
    path = R.RESULTS_DIR / f"sociodemographic_baseline_{arch}{suffix}.csv"
    out.to_csv(path, index=False)
    R.write_run_metadata(
        path.with_suffix(".json"), datasets,
        allow_mismatch=allow_mismatch, n_bootstrap=N_BOOT,
        test_size=R.TEST_SIZE, split_seed=R.RANDOM_STATE,
        threshold_selection="F1-maximising quantile search on five-fold training OOF probabilities",
    )
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60):
        print(f"[{arch}]")
        print(out.drop(columns=["hyperparameters"]).round(4).to_string(index=False))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-mismatch", action="store_true",
        help="Allow published-metric mismatches; export a labelled sensitivity table.",
    )
    main(allow_mismatch=parser.parse_args().allow_mismatch)

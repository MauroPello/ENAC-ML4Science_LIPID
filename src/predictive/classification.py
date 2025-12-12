"""Classification modeling helpers for binary targets."""

import math
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.utils.prediction import collect_coefficients


def run_classification_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = 5,
) -> Dict[str, object]:
    """Train baseline classifiers with k-fold CV grid search and return evaluation artefacts.

    Uses `StratifiedKFold` and `GridSearchCV` with standard parameter grids.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        cv (int): Number of folds for cross-validation.

    Returns:
        Dict[str, object]: A dictionary containing classification results and artifacts.
    """

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_labels = label_encoder.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    models: List[tuple[str, Pipeline]] = [
        (
            "Logistic Regression (Ridge)",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000),
                    ),
                ]
            ),
        ),
        (
            "Logistic Regression (Lasso)",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(penalty="l1", solver="saga", max_iter=5000),
                    ),
                ]
            ),
        ),
        (
            "Random Forest Classifier",
            Pipeline(
                steps=[
                    (
                        "model",
                        RandomForestClassifier(random_state=random_state),
                    )
                ]
            ),
        ),
        (
            "SVM (Linear)",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SVC(
                            kernel="linear", probability=True, random_state=random_state
                        ),
                    ),
                ]
            ),
        ),
        (
            "SVM (RBF)",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SVC(kernel="rbf", probability=True, random_state=random_state),
                    ),
                ]
            ),
        ),
        (
            "k-NN Classifier",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
        ),
    ]

    # Standard parameter grids keyed by model name (grid uses step name + param)
    param_grids = {
        "Logistic Regression (Ridge)": {"model__C": [0.01, 0.1, 1.0, 10.0]},
        "Logistic Regression (Lasso)": {"model__C": [0.01, 0.1, 1.0, 10.0]},
        "Random Forest Classifier": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10, 20],
        },
        "SVM (Linear)": {"model__C": [0.01, 0.1, 1.0, 10.0]},
        "SVM (RBF)": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
        "k-NN Classifier": {"model__n_neighbors": [3, 5, 7]},
    }

    classification_records: List[Dict[str, float]] = []
    confusion_matrices: Dict[str, pd.DataFrame] = {}
    coefficient_records: List[Dict[str, float]] = []
    best_estimators: Dict[str, Pipeline] = {}
    best_params_map: Dict[str, dict] = {}

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for name, pipeline in models:
        grid = param_grids.get(name, {})
        try:
            gs = GridSearchCV(
                pipeline,
                grid,
                cv=cv_strategy,
                scoring="f1_weighted",
                n_jobs=-1,
                error_score="raise",
            )
            gs.fit(X_train, y_train)

            refined_grid = _get_refined_classification_grid(gs.best_params_)
            gs = GridSearchCV(
                pipeline,
                refined_grid,
                cv=cv_strategy,
                scoring="f1_weighted",
                n_jobs=-1,
                error_score="raise",
            )
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            y_pred = best.predict(X_test)

        except Exception as e:
            print(f"Model {name} failed during GridSearchCV: {e}")
            continue

        metrics = _collect_classification_metrics(best, y_test, y_pred, X_test)
        metrics["model"] = name
        metrics["best_params"] = gs.best_params_

        classification_records.append(metrics)

        coefficient_records.extend(
            collect_coefficients(name, best, X_train, y_train, random_state)
        )

        cm = confusion_matrix(y_test, y_pred)
        index_labels = [f"Actual_{label}" for label in class_labels]
        column_labels = [f"Pred_{label}" for label in class_labels]
        confusion_matrices[name] = pd.DataFrame(
            cm, index=index_labels, columns=column_labels
        )
        best_estimators[name] = best
        best_params_map[name] = gs.best_params_

    results_df = (
        pd.DataFrame(classification_records).sort_values(by="F1", ascending=False)
        if classification_records
        else pd.DataFrame()
    )
    coefficients_df = pd.DataFrame(coefficient_records)

    best_model_name = results_df.iloc[0]["model"] if not results_df.empty else None
    best_model = (
        best_estimators.get(best_model_name) if best_model_name is not None else None
    )
    best_params = (
        best_params_map.get(best_model_name) if best_model_name is not None else None
    )

    return {
        "classification_results": results_df,
        "confusion_matrices": confusion_matrices,
        "class_labels": class_labels,
        "coefficients": coefficients_df,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_params": best_params,
    }


def _collect_classification_metrics(
    model: Pipeline,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    X_test: pd.DataFrame,
) -> Dict[str, float]:
    """Collect evaluation metrics for classification.

    Args:
        model (Pipeline): The trained model pipeline.
        y_test (np.ndarray): True labels for the test set.
        y_pred (np.ndarray): Predicted labels for the test set.
        X_test (pd.DataFrame): Test feature matrix.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation metrics.
    """
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(
        precision_score(y_test, y_pred, average="weighted", zero_division=0)
    )
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    roc_auc = math.nan
    if len(np.unique(y_test)) == 2:
        scores = _safe_prediction_scores(model, X_test)
        if scores is not None:
            roc_auc = float(roc_auc_score(y_test, scores))
    else:
        try:
            scores = model.predict_proba(X_test)
            roc_auc = float(
                roc_auc_score(y_test, scores, multi_class="ovr", average="weighted")
            )
        except Exception:
            pass

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
    }


def _safe_prediction_scores(model: Pipeline, X_test: pd.DataFrame) -> np.ndarray | None:
    """Get prediction scores (probabilities or decision function) safely.

    Args:
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): Test feature matrix.

    Returns:
        np.ndarray | None: Prediction scores if available, otherwise None.
    """
    try:
        return model.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    try:
        return model.decision_function(X_test)
    except Exception:
        return None


def _get_refined_classification_grid(
    best_params: dict,
) -> dict:
    """
    Get the hyperparameter grid to refine the model further, given the previous best params.

    Args:
        best_params (dict): Dictionary of best parameters found in the initial search.

    Returns:
        dict: A refined hyperparameter grid for the next search.
    """
    refined_grid = {}

    for param_name, best_value in best_params.items():
        if param_name == "model__C":
            refined_grid[param_name] = [
                best_value * 0.8,
                best_value * 0.9,
                best_value,
                best_value * 1.1,
                best_value * 1.2,
            ]
        elif param_name == "model__n_estimators":
            base = int(best_value)
            refined_grid[param_name] = [
                max(50, base - 20),
                max(50, base - 10),
                base,
                base + 10,
                base + 20,
            ]
        elif param_name == "model__max_depth":
            # None is a valid value for max_depth, it is handled separately
            if best_value is None:
                refined_grid[param_name] = [None]
            else:
                base = int(best_value)
                refined_grid[param_name] = [
                    max(1, base - 2),
                    max(1, base - 1),
                    base,
                    base + 1,
                    base + 2,
                ]
        elif param_name == "model__gamma":
            # Gamma could be a numerical value as well, also consider this option
            if best_value in ["scale", "auto"]:
                refined_grid[param_name] = [best_value]
            else:
                refined_grid[param_name] = [
                    best_value * 0.8,
                    best_value * 0.9,
                    best_value,
                    best_value * 1.1,
                    best_value * 1.2,
                ]
        elif param_name == "model__n_neighbors":
            # Always use an odd number of neighbors
            base = int(best_value) // 2 * 2 + 1
            refined_grid[param_name] = [
                max(1, base - 2),
                base,
                base + 2,
            ]
        else:
            refined_grid[param_name] = [best_value]

    return refined_grid

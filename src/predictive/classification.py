"""Classification modeling helpers for binary targets."""

from __future__ import annotations

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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


def run_classification_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Train baseline classifiers for binary outcomes."""

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
                        LogisticRegression(
                            penalty="l2", solver="lbfgs", max_iter=5000
                        ),
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
                        LogisticRegression(
                            penalty="l1", solver="saga", max_iter=5000
                        ),
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
                        RandomForestClassifier(
                            n_estimators=300, random_state=random_state
                        ),
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
                            kernel="linear",
                            probability=True,
                            random_state=random_state,
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
                        SVC(
                            kernel="rbf",
                            probability=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "k-NN Classifier",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier(n_neighbors=5)),
                ]
            ),
        ),
    ]

    classification_records: List[Dict[str, float]] = []
    confusion_matrices: Dict[str, pd.DataFrame] = {}

    for name, model in models:
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        except Exception:
            continue

        metrics = _collect_classification_metrics(model, y_test, y_pred, X_test)
        metrics["model"] = name
        classification_records.append(metrics)

        cm = confusion_matrix(y_test, y_pred)
        index_labels = [f"Actual_{label}" for label in class_labels]
        column_labels = [f"Pred_{label}" for label in class_labels]
        confusion_matrices[name] = pd.DataFrame(
            cm, index=index_labels, columns=column_labels
        )

    results_df = (
        pd.DataFrame(classification_records).sort_values(by="F1", ascending=False)
        if classification_records
        else pd.DataFrame()
    )

    return {
        "classification_results": results_df,
        "confusion_matrices": confusion_matrices,
        "class_labels": class_labels,
    }


def _collect_classification_metrics(
    model: Pipeline,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    X_test: pd.DataFrame,
) -> Dict[str, float]:
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    roc_auc = math.nan
    if len(np.unique(y_test)) == 2:
        scores = _safe_prediction_scores(model, X_test)
        if scores is not None:
            roc_auc = float(roc_auc_score(y_test, scores))

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
    }


def _safe_prediction_scores(model: Pipeline, X_test: pd.DataFrame) -> np.ndarray | None:
    try:
        return model.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    try:
        return model.decision_function(X_test)
    except Exception:
        return None

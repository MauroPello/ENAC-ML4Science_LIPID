"""
Surrogate supervised learning module.
Focuses on predicting Morphology (Y) from Environmental Quality (X).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier

# Try importing SHAP
import shap

HAS_SHAP = hasattr(shap, "LinearExplainer")


class SurrogateModel:
    """Encapsulates regression models to predict Morphology from Environment."""

    def __init__(self, model_type: str = "ridge", alpha: float = 1.0):
        if model_type == "ridge":
            base_estimator = Ridge(alpha=alpha, random_state=42)
        elif model_type == "lasso":
            base_estimator = Lasso(alpha=alpha, random_state=42)
        else:
            raise ValueError("model_type must be 'ridge' or 'lasso'")

        self.model = MultiOutputRegressor(base_estimator)
        self.feature_names = None

    def train(self, X: pd.DataFrame, Y: pd.DataFrame) -> dict[str, float]:
        """
        Train the multi-output regression model.

        Returns:
            Dictionary with training metrics (R2, RMSE).
        """
        self.feature_names = X.columns
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, Y_train)
        Y_pred = self.model.predict(X_test)

        r2 = r2_score(Y_test, Y_pred, multioutput="uniform_average")
        mse = mean_squared_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)

        print(
            f"Surrogate Model ({self.model.estimator.__class__.__name__}) Performance:"
        )
        print(f"  Avg R2: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")

        return {"r2": r2, "rmse": rmse}

    def analyze_feature_importance(self, X: pd.DataFrame) -> None:
        """
        Analyze feature importance using SHAP (if available) or Coefficients.
        """
        # For MultiOutputRegressor, we need to inspect each estimator
        if HAS_SHAP:
            print("Running SHAP analysis on the first target dimension...")
            first_estimator = self.model.estimators_[0]
            explainer = shap.LinearExplainer(first_estimator, X)
            shap_values = explainer.shap_values(X)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, show=False)
            plt.title("SHAP Feature Importance (Target Dim 0)")
            plt.show()
        else:
            print("SHAP not found. Using Coefficient Aggregation.")
            # Aggregate absolute coefficients across all target dimensions
            coefs = np.array([est.coef_ for est in self.model.estimators_])
            # shape: (n_targets, n_features)
            mean_abs_coefs = np.mean(np.abs(coefs), axis=0)

            importance_df = pd.DataFrame(
                {"feature": self.feature_names, "importance": mean_abs_coefs}
            ).sort_values(by="importance", ascending=False)

            plt.xlabel("Mean Absolute Coefficient")
            plt.show()

    def evaluate_classification(
        self, X: pd.DataFrame, y_true: np.ndarray, cv: int = 5
    ) -> None:
        """
        Evaluate Random Forest classification for predicting morphology clusters from environmental features.

        Args:
            X: Environmental features (DataFrame).
            y_true: True cluster labels (numpy array).
            cv: Number of cross-validation folds.
        """
        # Filter out noise (-1) if using DBSCAN
        mask = y_true != -1
        X_clean = X[mask]
        y_clean = y_true[mask]

        if len(set(y_clean)) < 2:
            print("Not enough classes for classification.")
            return

        clf = RandomForestClassifier(random_state=42)
        y_pred = cross_val_predict(clf, X_clean, y_clean, cv=cv)

        print("\nClassification Report (Env -> Morph Cluster):")
        print(classification_report(y_clean, y_pred, zero_division=0))

        # Plot Confusion Matrix
        cm = confusion_matrix(y_clean, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Cluster")
        plt.ylabel("True Morphology Cluster")
        plt.title("Confusion Matrix: Env predict Morph Cluster")
        plt.show()

        # Feature Importance
        clf.fit(X_clean, y_clean)
        importances = clf.feature_importances_
        fi_df = pd.DataFrame(
            {"feature": X.columns, "importance": importances}
        ).sort_values("importance", ascending=False)

        plt.tight_layout()
        plt.show()

        # SHAP Analysis for Classification
        print("Running SHAP analysis for Classification...")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_clean)

        # Check structure of shap_values (list for multiclass, array for binary)
        # We plot the summary for the first class or aggregate
        plt.figure(figsize=(10, 6))
        # Summary plot for all classes (if multiclass, shap_values is a list)
        shap.summary_plot(shap_values, X_clean, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Classification)")
        plt.show()

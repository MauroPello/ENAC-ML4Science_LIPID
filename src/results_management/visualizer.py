"""Visualization utilities for experiment results."""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ResultsVisualizer:
    """Class for visualizing experiment results."""
    
    def __init__(self, results: Dict[str, Any]) -> None:
        """Initialize the ResultsVisualizer.

        Args:
            results (Dict[str, Any]): Experiment results dictionary.
        """
        self.results = results
        self.target_type = results.get("target_type", "unknown")
        self.target_variable = results.get("target_variable", "unknown")
        self.experiment_name = results.get("experiment_name", "unknown")
        sns.set_theme('notebook')
    
    def summary(self) -> None:
        """Print a text summary of the results."""
        print(f"\n{self.experiment_name} | {self.target_variable} ({self.target_type})")
        print(f"Best: {self.results.get('best_model_name', 'N/A')}\n")
        
        df = self.results.get("regression_results" if self.target_type == "continuous" else "classification_results")
        if df is not None and not df.empty:
            metric = "R2" if self.target_type == "continuous" else "Accuracy"
            for _, row in df.iterrows():
                val = row.get(metric, row.get(metric.lower(), "N/A"))
                print(f"  {row.get('model', 'Unknown'):25s} | {metric} = {val:.4f}" if isinstance(val, float) else f"  {row.get('model')}")
    
    def plot_all(self, figsize: tuple = (12, 8)) -> None:
        """Generate all relevant plots based on target type.

        Args:
            figsize (tuple): Size of the figure.
        """
        if self.target_type == "continuous":
            self._plot_regression(figsize)
        else:
            self._plot_classification(figsize)
    
    def _plot_regression(self, figsize: tuple) -> None:
        reg_df = self.results.get("regression_results")
        coef_df = self.results.get("coefficients")
        residuals = self.results.get("residuals", {})
        
        n_plots = sum([reg_df is not None, coef_df is not None and not coef_df.empty, bool(residuals)])
        if n_plots == 0:
            print("No regression data to plot.")
            return
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], 4 * n_plots))
        axes = [axes] if n_plots == 1 else list(axes)
        fig.suptitle(f"{self.experiment_name} ({self.target_variable})", fontweight="bold")
        
        idx = 0
        if reg_df is not None and not reg_df.empty:
            self._bar_chart(axes[idx], reg_df, "R2", "Model Performance (R²)")
            idx += 1
        
        if coef_df is not None and not coef_df.empty:
            self._coef_chart(axes[idx], coef_df)
            idx += 1
        
        if residuals and (best := self.results.get("best_model_name")) in residuals:
            r = residuals[best]
            axes[idx].scatter(r["pred"], r["resid"], alpha=0.5, s=15)
            axes[idx].axhline(0, color="red", linestyle="--")
            axes[idx].set(xlabel="Predicted", ylabel="Residuals", title=f"Residuals - {best}")
        
        plt.tight_layout()
        plt.show()
    
    def _plot_classification(self, figsize: tuple) -> None:
        clf_df = self.results.get("classification_results")
        cms = self.results.get("confusion_matrices", {})
        
        n_plots = (1 if clf_df is not None else 0) + min(len(cms), 1)
        if n_plots == 0:
            print("No classification data to plot.")
            return
        
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        axes = [axes] if n_plots == 1 else list(axes)
        fig.suptitle(f"{self.experiment_name} ({self.target_variable})", fontweight="bold")
        
        if clf_df is not None and not clf_df.empty:
            metric = "Accuracy" if "Accuracy" in clf_df.columns else "accuracy"
            self._bar_chart(axes[0], clf_df, metric, f"Model Performance ({metric})")
        
        if cms:
            name, cm = next(iter(cms.items()))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[-1])
            axes[-1].set(title=f"Confusion Matrix - {name}")
        
        plt.tight_layout()
        plt.show()
    
    def _bar_chart(self, ax, df: pd.DataFrame, metric: str, title: str) -> None:
        sorted_df = df.sort_values(metric)
        colors = sns.color_palette("viridis", len(sorted_df))[::-1]
        model_col = "model" if "model" in df.columns else "Model"
        ax.barh(sorted_df[model_col], sorted_df[metric], color=colors)
        ax.set(xlabel=metric, title=title)
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            ax.text(row[metric], i, f" {row[metric]:.3f}", va="center", fontsize=8)
    
    def _coef_chart(self, ax, df: pd.DataFrame, top_n: int = 12) -> None:
        if "feature" not in df.columns:
            df = df.reset_index() if df.index.name == "feature" else df
            if len(df.columns) >= 2:
                df.columns = ["feature", "coefficient"] + list(df.columns[2:])
        
        df = df.copy()
        df["abs"] = df["coefficient"].abs()
        top = df.nlargest(top_n, "abs")
        
        colors = ["forestgreen" if c > 0 else "crimson" for c in top["coefficient"]]
        ax.barh(top["feature"], top["coefficient"], color=colors)
        ax.axvline(0, color="gray", linestyle="--", lw=0.8)
        ax.set(xlabel="Coefficient", title="Top Feature Coefficients")


def visualize_experiment(experiment_name: str, base_dir: str = "results") -> ResultsVisualizer:
    """One-call visualization of a saved experiment.

    Args:
        experiment_name (str): Name of the experiment.
        base_dir (str): Base directory for results.

    Returns:
        ResultsVisualizer: The visualizer instance.
    """
    from .manager import ResultsManager
    results = ResultsManager(base_dir).load(experiment_name)
    viz = ResultsVisualizer(results)
    viz.summary()
    viz.plot_all()
    return viz


def compare_targets(target_list: List[str], metric: str = "R2", base_dir: str = "results") -> None:
    """Compare best models across different target variables.

    Args:
        target_list (List[str]): List of target variables to compare.
        metric (str): Metric to use for comparison.
        base_dir (str): Base directory for results.
    """
    from .manager import ResultsManager
    manager = ResultsManager(base_dir)
    
    data = []
    for target in target_list:
        if (latest := manager.get_latest(target)) is None:
            continue
        df = latest.get("regression_results" if latest.get("target_type") == "continuous" else "classification_results")
        if df is None or df.empty or metric not in df.columns:
            continue
        
        model_col = "model" if "model" in df.columns else "Model"
        is_lower = metric in ["RMSE", "MAE"]
        best = df.loc[df[metric].idxmin() if is_lower else df[metric].idxmax()]
        data.append({"Target": target, "Best Model": best[model_col], metric: best[metric]})
    
    if not data:
        print("No data to compare.")
        return
    
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(df["Target"], df[metric], color=sns.color_palette("viridis", len(df)))
    ax.set(xlabel=metric, title=f"Best Model Comparison - {metric}")
    for i, row in df.iterrows():
        ax.text(row[metric], i, f" {row['Best Model']} ({row[metric]:.3f})", va="center", fontsize=9)
    plt.tight_layout()
    plt.show()

"""Results management module for storing and loading experiment results."""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd


class ResultsManager:
    """Class-based interface for managing experiment results."""

    VALID_TARGETS = ["cardiovascular", "sleep_disorder", "mental_health", "respiratory"]

    def __init__(self, base_dir: str = "results") -> None:
        """Initialize the ResultsManager.

        Args:
            base_dir (str): The base directory to store results.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        results: Dict[str, Any],
        experiment_name: str,
        target_variable: str,
        overwrite: bool = False,
    ) -> Path:
        """Save experiment results with target variable metadata.

        Args:
            results (Dict[str, Any]): Dictionary containing results and artifacts.
            experiment_name (str): Name of the experiment.
            target_variable (str): Name of the target variable.
            overwrite (bool): Whether to overwrite existing experiment results.

        Returns:
            Path: The path to the saved experiment directory.
        """
        if target_variable not in self.VALID_TARGETS:
            raise ValueError(f"target_variable must be one of {self.VALID_TARGETS}")

        exp_dir = self.base_dir / experiment_name
        if exp_dir.exists():
            if overwrite:
                shutil.rmtree(exp_dir)
            else:
                raise FileExistsError(
                    f"Experiment '{experiment_name}' exists. Use overwrite=True."
                )

        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": experiment_name,
            "target_variable": target_variable,
            "target_type": results.get("target_type", "unknown"),
            "best_model_name": results.get("best_model_name"),
            "best_params": self._serialize_params(results.get("best_params")),
        }
        (exp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Save DataFrames
        for key, value in results.items():
            if isinstance(value, pd.DataFrame) and not value.empty:
                value.to_csv(exp_dir / f"{key}.csv", index=True)

        # Save residuals
        if residuals := results.get("residuals"):
            serialized = {
                name: {"pred": r["pred"].tolist(), "resid": r["resid"].tolist()}
                for name, r in residuals.items()
            }
            (exp_dir / "residuals.json").write_text(json.dumps(serialized))

        # Save confusion matrices
        if cms := results.get("confusion_matrices"):
            cm_dir = exp_dir / "confusion_matrices"
            cm_dir.mkdir(exist_ok=True)
            for name, cm in cms.items():
                if isinstance(cm, pd.DataFrame) and not cm.empty:
                    cm.to_csv(cm_dir / f"{name.replace(' ', '_')}.csv")

        # Save trained models
        if best_model := results.get("best_model"):
            try:
                joblib.dump(best_model, exp_dir / "best_model.joblib")
            except Exception as e:
                print(f"Warning: Could not save model: {e}")
        if best_fitted := results.get("best_model_fitted"):
            try:
                joblib.dump(best_fitted, exp_dir / "best_model_fitted.joblib")
            except Exception as e:
                print(f"Warning: Could not save fitted model: {e}")

        print(f"Saved to {exp_dir}")
        return exp_dir

    def load(self, experiment_name: str) -> Dict[str, Any]:
        """Load experiment results from disk.

        Args:
            experiment_name (str): Name of the experiment to load.

        Returns:
            Dict[str, Any]: Loaded results dictionary.
        """
        exp_dir = self.base_dir / experiment_name
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment '{experiment_name}' not found")

        results: Dict[str, Any] = {}

        # Load metadata
        if (meta_path := exp_dir / "metadata.json").exists():
            results.update(json.loads(meta_path.read_text()))

        # Load CSVs
        for csv_path in exp_dir.glob("*.csv"):
            try:
                results[csv_path.stem] = pd.read_csv(csv_path, index_col=0)
            except Exception as e:
                print(f"Warning: Could not load {csv_path}: {e}")

        # Load residuals
        if (resid_path := exp_dir / "residuals.json").exists():
            data = json.loads(resid_path.read_text())
            results["residuals"] = {
                name: {"pred": np.array(d["pred"]), "resid": np.array(d["resid"])}
                for name, d in data.items()
            }

        # Load confusion matrices
        if (cm_dir := exp_dir / "confusion_matrices").exists():
            results["confusion_matrices"] = {
                p.stem.replace("_", " "): pd.read_csv(p, index_col=0)
                for p in cm_dir.glob("*.csv")
            }

        # Load trained models
        if (model_path := exp_dir / "best_model.joblib").exists():
            try:
                results["best_model"] = joblib.load(model_path)
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
        if (fitted_path := exp_dir / "best_model_fitted.joblib").exists():
            try:
                results["best_model_fitted"] = joblib.load(fitted_path)
            except Exception as e:
                print(f"Warning: Could not load fitted model: {e}")

        return results

    def load_by_target(self, target_variable: str) -> Dict[str, Dict[str, Any]]:
        """Load all experiments for a specific target variable.

        Args:
            target_variable (str): The target variable to filter by.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping experiment names to result dictionaries.
        """
        if target_variable not in self.VALID_TARGETS:
            raise ValueError(f"target_variable must be one of {self.VALID_TARGETS}")

        result = {}
        for exp_dir in self.base_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            meta_path = exp_dir / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta.get("target_variable") == target_variable:
                    result[exp_dir.name] = self.load(exp_dir.name)
        return result

    def list_all(self) -> pd.DataFrame:
        """List all saved experiments.

        Returns:
            pd.DataFrame: DataFrame containing metadata of all experiments.
        """
        experiments = []
        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir() and (meta_path := exp_dir / "metadata.json").exists():
                experiments.append(json.loads(meta_path.read_text()))

        if not experiments:
            return pd.DataFrame()
        df = pd.DataFrame(experiments)
        return (
            df.sort_values("timestamp", ascending=False)
            if "timestamp" in df.columns
            else df
        )

    def get_latest(self, target_variable: str) -> Optional[Dict[str, Any]]:
        """Get the most recent experiment for a target variable.

        Args:
            target_variable (str): The target variable.

        Returns:
            Optional[Dict[str, Any]]: The results of the latest experiment, or None if not found.
        """
        experiments = self.load_by_target(target_variable)
        if not experiments:
            return None
        latest = max(experiments.items(), key=lambda x: x[1].get("timestamp", ""))
        return latest[1]

    def get_summary(self, experiment_name: str) -> None:
        """Print a summary of a specific experiment.

        Args:
            experiment_name (str): Name of the experiment.
        """
        try:
            r = self.load(experiment_name)
        except FileNotFoundError:
            print(f"Experiment '{experiment_name}' not found.")
            return

        print(f"{'=' * 50}\nExperiment: {experiment_name}")
        print(f"Target: {r.get('target_variable')} ({r.get('target_type')})")
        print(f"Best Model: {r.get('best_model_name')}\n{'-' * 50}")

        for key in ["regression_results", "classification_results"]:
            if key in r and not r[key].empty:
                print(f"\n{key}:\n{r[key].to_string()}")

    @staticmethod
    def _serialize_params(params: Optional[Dict]) -> Optional[Dict]:
        if params is None:
            return None
        return {
            k: v.item() if isinstance(v, (np.integer, np.floating)) else v
            for k, v in params.items()
        }

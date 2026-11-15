"""Regression checks for the post-hoc analyses; no private health data required."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _rebuild as R
import shap_analysis as S
import sociodemographic_baseline as B


class ReferenceValidationTests(unittest.TestCase):
    def test_metric_parity_requires_explicit_sensitivity_opt_in(self):
        self.assertTrue(R.check_published_metrics("mental_health", 0.414239482, 0.586799720))
        with self.assertRaisesRegex(RuntimeError, "does not reproduce published"):
            R.check_published_metrics("mental_health", 0.409090909, 0.582072829)
        with self.assertWarnsRegex(RuntimeWarning, "does not reproduce published"):
            self.assertFalse(R.check_published_metrics(
                "mental_health", 0.409090909, 0.582072829, allow_mismatch=True
            ))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            R.check_published_metrics("mental_health", float("nan"), 0.5868, allow_mismatch=True)

    def test_same_schema_cannot_hide_changed_rows_labels_or_order(self):
        original = pd.DataFrame({"x": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
        types = {"x": "continuous", "target": "binary"}
        reference = {"target_datasets": {"example": {
            "dataset_fingerprint": R.dataset_fingerprint(original)
        }}}
        changed_value = original.copy()
        changed_value.loc[0, "x"] = 9.0
        changed_label = original.copy()
        changed_label.loc[0, "target"] = 1
        with patch.object(R, "load_config", return_value={"feature_types": types}), \
             patch.object(R, "load_reference", return_value=reference):
            R.verify_against_config("example", original, types)
            for changed in (changed_value, changed_label, original.iloc[::-1].reset_index(drop=True)):
                with self.subTest(changed=changed.to_dict()), self.assertRaisesRegex(RuntimeError, "values, labels or row order"):
                    R.verify_against_config("example", changed, types)

    def test_input_checksum_rejects_wrong_workbook(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "health.xlsx"
            path.write_bytes(b"original")
            expected = R.file_sha256(path)
            R.require_file_hash(path, expected)
            path.write_bytes(b"different")
            with self.assertRaisesRegex(RuntimeError, "checksum differs"):
                R.require_file_hash(path, expected)


class ExplanationValidationTests(unittest.TestCase):
    def test_permutation_chunk_accepts_budget_and_matches_linear_probability_model(self):
        class ProbabilityModel:
            def predict_proba(self, X):
                p = (0.2 + 0.1 * X["a"] + 0.05 * X["b"]).to_numpy()
                return np.column_stack([1 - p, p])

        background = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 2.0]})
        rows = pd.DataFrame({"a": [2.0, 0.0], "b": [1.0, 3.0]})
        values = S._explain_chunk(ProbabilityModel(), background, rows, 42, max_evals=90)
        expected = (rows - background.mean()).to_numpy() * [0.1, 0.05]
        np.testing.assert_allclose(values, expected, atol=1e-12)

    def test_invalid_shap_values_are_rejected_before_export(self):
        values = np.array([[0.1, -0.05], [0.2, 0.1]])
        predictions = np.array([0.35, 0.6])
        self.assertLess(S.validate_values(values, predictions, 0.3, (2, 2)), 1e-12)
        for bad, message in [
            (np.array([[np.nan, 0.0], [0.0, 0.0]]), "Non-finite"),
            (values + 0.1, "additivity"),
            (values[:, :1], "dimensions"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                S.validate_values(bad, predictions, 0.3, (2, 2))
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            S.validate_values(values, predictions, float("nan"), (2, 2))

    def test_constant_model_has_finite_zero_relative_importance(self):
        shares = S.relative_shares(pd.Series([0.0, 0.0]))
        self.assertTrue(np.isfinite(shares).all())
        self.assertEqual(shares.tolist(), [0.0, 0.0])

    def test_failed_threshold_cv_is_not_silently_replaced(self):
        pipe = Mock()
        X = pd.DataFrame({"age_bin": np.arange(12)})
        y = np.tile([0, 1], 6)
        with patch.object(B, "cross_val_predict", side_effect=RuntimeError("CV failed")):
            with self.assertRaisesRegex(RuntimeError, "CV failed"):
                B._fit_eval(pipe, X, y, np.arange(10), np.array([10, 11]))


class CommittedResultsTests(unittest.TestCase):
    def test_baseline_table_preserves_published_controls_and_differences(self):
        table = pd.read_csv(ROOT / "docs/revision/baseline_comparison.csv")
        self.assertEqual(set(table.target), set(R.CLASSIFICATION_TARGETS))
        for row in table.itertuples():
            self.assertTrue(R.check_published_metrics(row.target, row.full_F1, row.full_ROC_AUC))
            self.assertAlmostEqual(row.delta_F1_full_minus_socio, row.full_F1 - row.socio_F1)
            self.assertAlmostEqual(row.delta_AUC_full_minus_socio, row.full_ROC_AUC - row.socio_ROC_AUC)
            self.assertLessEqual(row.delta_AUC_95CI_low, 0)
            self.assertGreaterEqual(row.delta_AUC_95CI_high, 0)

    def test_shap_summary_ranks_shares_and_exact_comparison(self):
        table = pd.read_csv(ROOT / "docs/revision/shap_per_feature.csv")
        for name, part in table.groupby("target"):
            self.assertTrue(np.isfinite(part.mean_abs_shap).all())
            self.assertAlmostEqual(part.share_of_total.sum(), 1.0)
            ranked = part.sort_values("rank")
            self.assertTrue(ranked.mean_abs_shap.is_monotonic_decreasing)
            if name == "mental_health":
                exact = part.sort_values("mean_abs_shap_exact_treeshap", ascending=False)
                self.assertEqual(ranked.feature.tolist(), exact.feature.tolist())


if __name__ == "__main__":
    unittest.main()

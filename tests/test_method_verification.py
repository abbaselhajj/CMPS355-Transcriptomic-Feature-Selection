"""Purely synthetic, deterministic numerical reference tests (no patient data)."""
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmps355.method_verification import (
    SparseLogisticSpec, independent_reference_solver, penalty_objective,
    sparse_logistic_model, stability_selection, synthetic_binary_data,
)


class TestSparseLogistic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.names = synthetic_binary_data()

    def test_objective_matches_independent_convex_optimum_lasso(self):
        spec = SparseLogisticSpec(C=0.16, l1_ratio=1.0)
        pipe = sparse_logistic_model(spec).fit(self.X, self.y)
        scaled = pipe["scale"].transform(self.X)
        fit = pipe["logistic"]
        fitted_loss = penalty_objective(scaled, self.y, fit.coef_[0], fit.intercept_[0], spec)
        w, b, ref_loss = independent_reference_solver(scaled, self.y, spec)
        self.assertLessEqual(fitted_loss - ref_loss, 1e-4, (fitted_loss, ref_loss))
        self.assertLess(np.max(np.abs(w - fit.coef_[0])), 0.02)

    def test_objective_matches_independent_convex_optimum_elastic_net(self):
        spec = SparseLogisticSpec(C=0.16, l1_ratio=0.5)
        pipe = sparse_logistic_model(spec).fit(self.X, self.y)
        scaled = pipe["scale"].transform(self.X)
        fit = pipe["logistic"]
        fitted_loss = penalty_objective(scaled, self.y, fit.coef_[0], fit.intercept_[0], spec)
        w, b, ref_loss = independent_reference_solver(scaled, self.y, spec)
        self.assertLessEqual(fitted_loss - ref_loss, 1e-4, (fitted_loss, ref_loss))
        self.assertLess(np.max(np.abs(w - fit.coef_[0])), 0.02)

    def test_pipeline_scaler_fit_only_on_supplied_training_rows(self):
        train, test = self.X[:120], self.X[120:] + 1000
        pipe = sparse_logistic_model(SparseLogisticSpec(C=0.1, l1_ratio=1)).fit(train, self.y[:120])
        np.testing.assert_allclose(pipe["scale"].mean_, train.mean(axis=0))
        self.assertGreater(abs(pipe["scale"].mean_[0] - np.vstack([train, test]).mean(axis=0)[0]), 100)

    def test_invalid_parameters_rejected(self):
        for C, alpha in [(-1, 1), (1, -0.1), (1, 1.1), (float('nan'), 1)]:
            with self.assertRaises(ValueError):
                SparseLogisticSpec(C=C, l1_ratio=alpha)

    def test_lasso_has_sparse_nonzero_coefs_on_signal_synthetic(self):
        pipe = sparse_logistic_model(SparseLogisticSpec(C=0.12, l1_ratio=1)).fit(self.X, self.y)
        coeff = pipe["logistic"].coef_[0]
        self.assertTrue(np.all(np.abs(coeff[:3]) > 0.05))
        self.assertLess(np.sum(np.abs(coeff[3:]) > 1e-5), 7)

    def test_independent_objective_penalty_increases_with_weights(self):
        scaled = StandardScaler().fit_transform(self.X)
        spec = SparseLogisticSpec(C=1, l1_ratio=0.7)
        w = np.zeros(scaled.shape[1])
        self.assertAlmostEqual(penalty_objective(scaled, self.y, w, 0, spec), np.log(2))


class TestStabilitySelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.names = synthetic_binary_data(seed=19)

    def test_probabilities_are_valid_max_and_selected_set_exact(self):
        r = stability_selection(self.X, self.y, self.names,
                                [0.08, 0.13], n_subsamples=12, threshold=0.75, seed=111)
        freqs = np.asarray(r.per_C_frequencies)
        peaks = np.asarray(r.max_frequencies)
        self.assertEqual(freqs.shape, (2, 12))
        self.assertEqual(r.half_sample_size, 80)
        self.assertTrue(np.all((freqs >= 0) & (freqs <= 1)))
        np.testing.assert_allclose(peaks, freqs.max(axis=0))
        self.assertEqual(r.stable_features,
                         [name for name, prob in zip(self.names, peaks) if prob >= .75])
        self.assertGreaterEqual(set(self.names[:2]).intersection(r.stable_features).__len__(), 2)

    def test_fixed_seed_reproducibility(self):
        args = (self.X, self.y, self.names, [0.1, 0.2])
        a = stability_selection(*args, n_subsamples=4, seed=9)
        b = stability_selection(*args, n_subsamples=4, seed=9)
        self.assertEqual(a.as_dict(), b.as_dict())

    def test_no_bootstrap_replacement_and_bad_grid_rejected(self):
        with self.assertRaises(ValueError):
            stability_selection(self.X, self.y, self.names, [0.1, 0.1])
        with self.assertRaises(ValueError):
            stability_selection(self.X, self.y, self.names, [0.1], threshold=0.5)
        with self.assertRaises(ValueError):
            stability_selection(self.X, self.y, self.names[:-1], [0.1])


if __name__ == '__main__':
    unittest.main()

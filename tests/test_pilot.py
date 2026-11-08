import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from cmps355.pilot import (
    PilotConfig, fit_logistic, fit_preprocessor, load_verified_splits,
    log_cpm, score_predictions, transform_preprocessor,
)


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.genes = ('a', 'b', 'c', 'd')
        self.counts = np.array([
            [30, 1, 5, 15],
            [28, 2, 5, 17],
            [25, 1, 7, 12],
            [8, 30, 5, 15],
            [7, 29, 4, 20],
            [5, 31, 5, 19],
        ], dtype=float)

    def test_log_cpm_is_per_sample(self):
        base = log_cpm(self.counts)
        changed = self.counts.copy()
        changed[1] *= 100
        self.assertTrue(np.allclose(base[0], log_cpm(changed)[0]))
        self.assertTrue(np.allclose(base[1], log_cpm(changed)[1]))

    def test_preprocessor_does_not_fit_to_external_samples(self):
        cfg = PilotConfig(max_genes=3, min_sample_fraction=0.2)
        prepared = fit_preprocessor(self.counts, self.genes, cfg)
        mean_before = prepared.scaler.mean_.copy()
        genes_before = prepared.genes
        test = np.array([[30, 1, 5, 14]], dtype=float)
        transformed = transform_preprocessor(prepared, test)
        extreme = np.array([[1, 100000, 1, 1]], dtype=float)
        transform_preprocessor(prepared, extreme)
        self.assertEqual(genes_before, prepared.genes)
        self.assertTrue(np.array_equal(mean_before, prepared.scaler.mean_))
        self.assertTrue(np.allclose(transformed, transform_preprocessor(prepared, test)))

    def test_preprocessor_ignores_labels(self):
        prep = fit_preprocessor(self.counts, self.genes, PilotConfig(max_genes=3))
        self.assertEqual(prep.train.shape, (6, 3))
        self.assertEqual(len(prep.genes), 3)

    def test_invalid_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            log_cpm(np.array([[0, 0]]))
        with self.assertRaises(ValueError):
            log_cpm(np.array([[-1, 2]]))

    def test_correct_binary_scoring(self):
        y = np.array([0, 0, 1, 1])
        scores = score_predictions(y, np.array([.1, .6, .8, .4]))
        self.assertAlmostEqual(scores['auroc'], .75)
        self.assertAlmostEqual(scores['balanced_accuracy'], .5)
        self.assertEqual(scores['true_positive'], 1)
        self.assertEqual(scores['false_positive'], 1)

    def test_sparse_model_runs_on_synthetic_data(self):
        rng = np.random.default_rng(355)
        X = rng.normal(size=(40, 5))
        y = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
        X[:, 0] += 3 * y
        for method in ('lasso', 'elastic_net'):
            model, converged = fit_logistic(X, y, method, 0.3, PilotConfig(max_iter=400))
            self.assertTrue(converged)
            self.assertEqual(model.coef_.shape, (1, 5))
            self.assertTrue(np.isfinite(model.predict_proba(X)).all())

    def test_split_file_tampering_rejected(self):
        from cmps355.split_design import build_assignments, audit_assignments
        index = pd.DataFrame({
            'cohort': ['d1'] * 2 + ['d2'] * 2 + ['d3'] * 2 + ['d4'] * 2,
            'ID': [f's{i}' for i in range(8)],
            'Study': ['B1'] * 2 + ['B2'] * 2 + ['B3'] * 2 + ['B4'] * 2,
            'class': ['CD', 'Not IBD'] * 4,
        })
        cohorts = ['d1', 'd2', 'd3', 'd4']
        outer, inner = build_assignments(index, cohorts)
        self.assertEqual(audit_assignments(index, outer, inner, cohorts)['status'], 'PASS')
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / 'runs' / 'split_design' / 'x'
            folder.mkdir(parents=True)
            outer.to_csv(folder / 'outer.csv', index=False)
            inner.to_csv(folder / 'inner.csv', index=False)
            import hashlib
            artifact_hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                               for name in ('outer.csv', 'inner.csv')}
            (folder / 'report.json').write_text(json.dumps({
                'status': 'PASS', 'data_sha256': 'test_hash', 'artifacts_sha256': artifact_hashes,
            }))
            manifest = {'cohorts': cohorts, 'sha256': 'test_hash'}
            self.assertEqual(load_verified_splits(root, index, manifest)[2], folder)
            data = pd.read_csv(folder / 'outer.csv')
            data.loc[0, 'role'] = 'train' if data.loc[0, 'role'] == 'test' else 'test'
            data.to_csv(folder / 'outer.csv', index=False)
            with self.assertRaises(ValueError):
                load_verified_splits(root, index, manifest)


if __name__ == '__main__':
    unittest.main()

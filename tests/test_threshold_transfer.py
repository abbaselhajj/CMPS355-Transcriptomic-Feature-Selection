"""Synthetic regression tests for cohort-level threshold transfer."""
import unittest

import numpy as np
import pandas as pd

from cmps355.threshold_transfer import (
    THRESHOLDS, operating_metrics, select_threshold, validate_predictions,
)


class ThresholdTransferTests(unittest.TestCase):
    @staticmethod
    def sample_oof():
        return pd.DataFrame([
            {"ID": f"{cohort}_{i}", "validation_cohort": cohort,
             "true_CD": int(i >= 2), "probability_CD": [0.1, 0.4, 0.6, 0.8][i]}
            for cohort in ("d1", "d2", "d3") for i in range(4)
        ])

    def test_threshold_grid_contains_default(self):
        self.assertEqual(len(THRESHOLDS), 37)
        self.assertIn(0.5, THRESHOLDS)

    def test_score_operating_metrics(self):
        observed = operating_metrics(np.array([0, 0, 1, 1]),
                                     np.array([0.1, 0.8, 0.3, 0.9]), 0.5)
        self.assertEqual([observed[k] for k in ("TN", "FP", "FN", "TP")], [1, 1, 1, 1])
        self.assertAlmostEqual(observed["balanced_accuracy"], 0.5)

    def test_perfect_threshold_and_tie_resolution(self):
        threshold, scores = select_threshold(self.sample_oof())
        self.assertEqual(threshold, 0.5)
        self.assertAlmostEqual(scores.loc[scores["threshold"] == 0.5,
                                           "mean_cohort_balanced_accuracy"].iloc[0], 1.0)

    def test_different_cohort_sizes_do_not_change_weights(self):
        oof = self.sample_oof()
        augmented = pd.concat([oof, oof[oof["validation_cohort"] == "d1"].assign(
            ID=lambda f: "extra_" + f["ID"])], ignore_index=True)
        original_threshold, original_scores = select_threshold(oof)
        new_threshold, new_scores = select_threshold(augmented)
        self.assertEqual(original_threshold, new_threshold)
        np.testing.assert_allclose(original_scores["mean_cohort_balanced_accuracy"],
                                   new_scores["mean_cohort_balanced_accuracy"])

    def test_reject_duplicate_validation_ids(self):
        oof = self.sample_oof()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            select_threshold(pd.concat([oof, oof.iloc[[0]]], ignore_index=True))

    def test_reject_missing_class(self):
        oof = self.sample_oof()
        oof.loc[oof["validation_cohort"] == "d2", "true_CD"] = 1
        with self.assertRaisesRegex(ValueError, "both classes"):
            select_threshold(oof)

    def test_reject_out_of_range_probability(self):
        oof = self.sample_oof()
        oof.loc[0, "probability_CD"] = 1.5
        with self.assertRaisesRegex(ValueError, "Probabilities"):
            select_threshold(oof)

    def test_validate_source_predictions_detects_changed_label(self):
        index = pd.DataFrame({"ID": ["s1", "s2", "s3", "s4"],
                              "cohort": ["d1", "d1", "d2", "d2"],
                              "class": ["CD", "Not IBD", "CD", "Not IBD"]})
        predictions = pd.DataFrame([
            {"ID": id_, "test_cohort": cohort, "method": method,
             "true_CD": label, "probability_CD": prob, "predicted_CD": int(prob >= 0.5)}
            for method in ("lasso", "elastic_net")
            for id_, cohort, label, prob in [("s1", "d1", 1, 0.8), ("s2", "d1", 0, 0.2),
                                             ("s3", "d2", 1, 0.8), ("s4", "d2", 0, 0.2)]
        ])
        metrics = pd.DataFrame([{"test_cohort": c, "method": m, "balanced_accuracy": 1.0,
                                 "sensitivity": 1.0, "specificity": 1.0, "auroc": 1.0}
                                for c in ("d1", "d2") for m in ("lasso", "elastic_net")])
        validate_predictions(index, predictions, metrics, ["d1", "d2"])
        predictions.loc[predictions["ID"] == "s1", "true_CD"] = 0
        with self.assertRaisesRegex(ValueError, "label"):
            validate_predictions(index, predictions, metrics, ["d1", "d2"])


if __name__ == "__main__":
    unittest.main()

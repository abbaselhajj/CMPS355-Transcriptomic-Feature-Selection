import unittest

import pandas as pd

from cmps355.split_design import SplitDesignError, audit_assignments, build_assignments


class CohortSplitTests(unittest.TestCase):
    def setUp(self):
        self.cohorts = ["d1", "d2", "d3", "d4"]
        self.index = pd.DataFrame([
            {"cohort": c, "ID": f"{c}_{i}", "Study": f"S_{c}", "class": "CD" if i < 2 else "Not IBD"}
            for c in self.cohorts for i in range(4)
        ])

    def test_nested_partitions(self):
        outer, inner = build_assignments(self.index, self.cohorts)
        audit = audit_assignments(self.index, outer, inner, self.cohorts)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["outer_fold_count"], 4)
        self.assertEqual(audit["inner_fold_count"], 12)
        self.assertEqual(len(outer), 64)
        self.assertEqual(len(inner), 144)

    def test_duplicate_input_ids_rejected(self):
        altered = self.index.copy()
        altered.loc[1, "ID"] = altered.loc[0, "ID"]
        with self.assertRaises(SplitDesignError):
            build_assignments(altered, self.cohorts)

    def test_outer_leakage_detected(self):
        outer, inner = build_assignments(self.index, self.cohorts)
        outer.loc[(outer["outer_heldout"] == "d1") & (outer["cohort"] == "d1"), "role"] = "train"
        with self.assertRaises(SplitDesignError):
            audit_assignments(self.index, outer, inner, self.cohorts)

    def test_inner_leakage_detected(self):
        outer, inner = build_assignments(self.index, self.cohorts)
        mask = (inner["outer_heldout"] == "d1") & (inner["inner_heldout"] == "d2")
        inner.loc[mask & (inner["cohort"] == "d2"), "role"] = "train"
        with self.assertRaises(SplitDesignError):
            audit_assignments(self.index, outer, inner, self.cohorts)

    def test_heldout_sample_in_inner_detected(self):
        outer, inner = build_assignments(self.index, self.cohorts)
        wrong = self.index.loc[self.index["cohort"] == "d1"].iloc[[0]].copy()
        wrong.insert(0, "outer_heldout", "d1")
        wrong.insert(1, "inner_heldout", "d2")
        wrong.insert(2, "role", "train")
        inner = pd.concat([inner, wrong], ignore_index=True)
        with self.assertRaises(SplitDesignError):
            audit_assignments(self.index, outer, inner, self.cohorts)


if __name__ == "__main__":
    unittest.main()

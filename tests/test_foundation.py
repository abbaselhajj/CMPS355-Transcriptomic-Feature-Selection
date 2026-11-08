"""Independent synthetic tests; never treat toy values as actual research results."""

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmps355.project_checks import find_project_root, sha256_file
from cmps355.data_validation import validate_archive, DatasetValidationError


class FoundationTests(unittest.TestCase):
    def _make_tiny_archive(self, root: Path, misaligned=False):
        p = root / "toy.zip"
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as archive:
            for idx in range(1,5):
                c = f"d{idx}"
                ids = [f"patient_{idx}_a", f"patient_{idx}_b"]
                count_table = "ID,GENE1,GENE2\n" + f"{ids[0]},1,0\n{ids[1]},2,3\n"
                meta_ids = ids[::-1] if misaligned and idx == 2 else ids
                meta_table = "ID,Study,class\n" + f"{meta_ids[0]},PRJNA_fake,CD\n{meta_ids[1]},PRJNA_fake,Not IBD\n"
                archive.writestr(f"count_{c}_aligned.csv",count_table)
                archive.writestr(f"desc_{c}.csv",meta_table)
        return p

    def test_project_marker(self):
        self.assertEqual(find_project_root(ROOT / "notebooks"), ROOT)

    def test_hash_is_stable(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "a"
            p.write_bytes(b"abc")
            self.assertEqual(sha256_file(p),
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_valid_toy_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            p = self._make_tiny_archive(Path(temp))
            result = validate_archive(p, {"cohorts": ["d1", "d2", "d3", "d4"]})
            self.assertEqual(result["unique_sample_count"], 8)
            self.assertEqual(result["common_gene_count"], 2)

    def test_rejects_misaligned_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            p = self._make_tiny_archive(Path(temp), misaligned=True)
            with self.assertRaisesRegex(DatasetValidationError, "does NOT exactly match"):
                validate_archive(p, {"cohorts": ["d1", "d2", "d3", "d4"]})

    def test_rejects_wrong_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            p = self._make_tiny_archive(Path(temp))
            with self.assertRaisesRegex(DatasetValidationError, "SHA-256 mismatch"):
                validate_archive(p, {"cohorts": ["d1", "d2", "d3", "d4"], "sha256": "0"*64})


if __name__ == "__main__":
    unittest.main()

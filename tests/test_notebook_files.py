"""Structural checks: notebooks are clean, executable JSON artifacts."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class NotebookFilesTest(unittest.TestCase):
    def test_stage1_notebooks(self):
        for name in ('00_setup.ipynb', '01_data_validation.ipynb'):
            with self.subTest(notebook=name):
                nb = json.loads((ROOT / 'notebooks' / name).read_text('utf-8'))
                self.assertEqual(nb['nbformat'], 4)
                self.assertTrue(any(c['cell_type'] == 'code' for c in nb['cells']))
                self.assertTrue(all(c.get('execution_count') is None and not c.get('outputs')
                                    for c in nb['cells'] if c['cell_type'] == 'code'))

if __name__ == '__main__':
    unittest.main()

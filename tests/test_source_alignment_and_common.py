"""Synthetic tests: no private RNA-seq data needed in CI."""
import json
import tempfile
import unittest
from pathlib import Path
from cmps355.rdata_xdr_reader import pairlist_dict, RSerializationError, read_ibd_rdata
from cmps355.common_classifier_benchmark import metrics, METHODS
import numpy as np

class OriginalDataReproductionTests(unittest.TestCase):
    def test_r_pairlist_tags(self):
        obj={'pair':[1,2], 'tag':{'SYM':'ID'},'attrs':None,'next':
             {'pair':'CD','tag':{'SYM':'class'},'attrs':None,'next':None}}
        self.assertEqual(pairlist_dict(obj),{'ID':[1,2],'class':'CD'})
    def test_reject_duplicate_r_pairlist_key(self):
        obj={'pair':1,'tag':{'SYM':'ID'},'attrs':None,'next':
             {'pair':2,'tag':{'SYM':'ID'},'attrs':None,'next':None}}
        with self.assertRaises(RSerializationError):pairlist_dict(obj)
    def test_no_untrusted_rdata_code_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            import gzip
            path=Path(temp)/'bad.RData'
            with gzip.open(path,'wb') as out:out.write(b'not an RDX3 archive')
            with self.assertRaises(RSerializationError):read_ibd_rdata(path)
    def test_common_classifier_methods_and_metrics(self):
        self.assertEqual(len(METHODS),5)
        y=np.array([0,0,1,1])
        p=np.array([.1,.4,.6,.9])
        found=metrics(y,p)
        self.assertAlmostEqual(found['auroc'],1.)
        self.assertAlmostEqual(found['balanced_accuracy'],1.)
        self.assertEqual((found['tp'],found['tn'],found['fp'],found['fn']),(2,2,0,0))

class NotebookEntryPointTests(unittest.TestCase):
    def test_new_notebooks_clean_for_public_code(self):
        root=Path(__file__).resolve().parents[1]
        for name in ['00a_rebuild_original_sample_alignment.ipynb',
                     '01a_preprocessing_and_expression_eda.ipynb',
                     '22_common_classifier_feature_selection.ipynb',
                     '23_tentacles_native_warning_triage.ipynb']:
            with self.subTest(name=name):
                nb=json.loads((root/'notebooks'/name).read_text())
                self.assertEqual(nb['nbformat'],4)
                code=[x for x in nb['cells'] if x['cell_type']=='code']
                self.assertTrue(code)
                self.assertTrue(all(not x.get('outputs') and x.get('execution_count') is None for x in code))

if __name__=='__main__':unittest.main()

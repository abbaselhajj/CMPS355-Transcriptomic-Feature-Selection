"""Read-only independent table-level QC of frozen common-classifier predictions.

No fitting. All metrics are recomputed from saved patient-level probabilities.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import io,json,sys
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,confusion_matrix

EXPECTED_COUNTS={'d1':260,'d2':179,'d3':76,'d4':174}
EXPECTED_SHA='8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74'

def audit(root:Path,run:Path,out:Path):
 root,run,out=Path(root),Path(run),Path(out)
 canonical=root/'data/validated/ibd_dataset_sample_aligned.zip'
 assert sha256(canonical.read_bytes()).hexdigest()==EXPECTED_SHA
 with ZipFile(canonical) as zipfile:
  original={}
  for fold in EXPECTED_COUNTS:
   df=pd.read_csv(io.BytesIO(zipfile.read(f'desc_{fold}.csv')),dtype={'ID':str,'Study':str,'class':str})
   assert df.ID.is_unique
   original[fold]=df.set_index('ID')[['Study','class']]
 preds=pd.read_csv(run/'patient_predictions_PRIVATE.csv',dtype={'ID':str})
 scores=pd.read_csv(run/'cohort_metrics.csv')
 macro=pd.read_csv(run/'macro_summary.csv')
 genes=pd.read_csv(run/'selected_panel_receipts.csv')
 checks=0;recal=[]
 assert len(scores)==20 and len(macro)==5 and len(preds)==5*689
 assert preds.ID.notna().all()
 assert set(preds.method)==set(scores.method)
 for row in scores.itertuples(index=False):
  p=preds[(preds.heldout==row.heldout)&(preds.method==row.method)].copy()
  assert len(p)==EXPECTED_COUNTS[row.heldout] and p.ID.is_unique
  ref=original[row.heldout]
  assert set(p.ID)==set(ref.index)
  clinical=ref.loc[p.ID]
  assert p.class_true.tolist()==clinical['class'].tolist()
  assert p.Study.tolist()==clinical.Study.tolist()
  y=(p.class_true=='CD').astype(int).to_numpy()
  assert np.array_equal(p.y_true.to_numpy(dtype=int),y)
  pr=p.probability_CD.to_numpy()
  assert np.isfinite(pr).all() and (pr>=0).all() and (pr<=1).all()
  pred=(pr>=0.5).astype(int)
  assert np.array_equal(pred,p.y_pred.to_numpy(dtype=int))
  auc=float(roc_auc_score(y,pr)); bacc=float(balanced_accuracy_score(y,pred))
  tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
  assert abs(auc-row.auroc)<1e-12 and abs(bacc-row.balanced_accuracy)<1e-12
  assert [tp,tn,fp,fn]==[row.tp,row.tn,row.fp,row.fn]
  g=genes[(genes.heldout==row.heldout)&(genes.method==row.method)]
  assert len(g)==row.n_genes and g.gene.is_unique
  recal.append(dict(heldout=row.heldout,method=row.method,n=len(p),n_genes=len(g),auroc=auc,balanced_accuracy=bacc))
  checks+=15
 for method,g in scores.groupby('method'):
  row=macro[macro.method==method].iloc[0]
  assert abs(row.mean_auroc-g.auroc.mean())<1e-12
  assert abs(row.mean_balanced_accuracy-g.balanced_accuracy.mean())<1e-12
  checks+=2
 assert preds.groupby('method').size().eq(689).all()
 assert genes.groupby(['heldout','method']).size().shape[0]==20
 out.mkdir(parents=True,exist_ok=False)
 pd.DataFrame(recal).to_csv(out/'per_fold_independent_recalculation.csv',index=False)
 report=dict(status='PASS_READONLY_COMMON_CLASSIFIER_RESULT_AUDIT',patient_predictions=len(preds),
             matched_all_original_IDs_labels_and_studies=True,per_method_total=689,
             methods=5,cohorts=4,model_fits=20,
             deterministic_assertions=checks+7,failed_assertions=0,
             disclaimer='Output-level QC, not independent re-fitting of selectors or native R objects')
 (out/'audit_report.json').write_text(json.dumps(report,indent=2))
 return report

if __name__=='__main__':
 if len(sys.argv)!=4:raise SystemExit('Usage: audit_common_classifier_outputs.py PROJECT_ROOT COMMON_RUN OUT_DIR')
 print(audit(*(Path(x) for x in sys.argv[1:])))

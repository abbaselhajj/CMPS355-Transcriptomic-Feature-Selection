#!/usr/bin/env python3
"""Recompute TENTACLES saved held-out metrics and gene overlap without retraining.
Run: python src/cmps355/audit_tentacles_stage10_outputs.py --project-root .
Reads `runs/tentacles/{d1,d2,d3,d4,summary}`; no private data leave the machine.
"""
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

p=ArgumentParser();p.add_argument('--project-root',type=Path,default=Path('.'))
a=p.parse_args();root=a.project_root.resolve();base=root/'runs/tentacles';summary=base/'summary'
pred=pd.read_csv(summary/'heldout_predictions.csv',dtype={'ID':str})
metrics=pd.read_csv(summary/'heldout_metrics.csv').set_index('heldout')
selected=pd.read_csv(summary/'selected_genes.csv')
assert len(pred)==689 and pred.ID.nunique()==689
panels={}
for fold in ['d1','d2','d3','d4']:
 df=pred[pred.heldout==fold];y=df.y_true.to_numpy(int);prob=df.probability_CD.to_numpy(float);yp=df.y_pred.to_numpy(int)
 assert np.array_equal(yp,(prob>=.5).astype(int))
 roc=float(roc_auc_score(y,prob));bac=(float(np.mean(yp[y==0]==0))+float(np.mean(yp[y==1]==1)))/2
 assert np.isclose(roc,metrics.loc[fold,'roc_auc'],atol=1e-12)
 assert np.isclose(bac,metrics.loc[fold,'balanced_accuracy'],atol=1e-12)
 gene=pd.read_csv(base/fold/'consensus_genes_native.csv').gene.tolist()
 assert gene==selected.loc[selected.heldout==fold].sort_values('panel_position').gene.tolist()
 panels[fold]=set(gene)
 print(f'{fold}: n={len(y)}, genes={len(gene)}, AUROC={roc:.4f}, BAcc={bac:.4f}')
common=set.intersection(*panels.values());print('All-four consensus genes:',len(common),','.join(sorted(common)))
assert len(common)==14
print('SAVED_OUTPUT_VERIFICATION_PASS=TRUE')

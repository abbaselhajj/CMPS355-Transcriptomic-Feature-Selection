#!/usr/bin/env python3
"""Read-only StableMate Stage-8 LOCO audit: checksum, IDs, raw-count preprocessing, scores and predictions.

Required: Python 3.11+, numpy, pandas, scipy.
Example (WSL):
 python stablemate_stage8_readonly_audit.py \
   --runs-root ~/CMPS355_StableMate_Local/runs/stablemate_continuous_final \
   --data-zip ~/CMPS355_StableMate_Local/ibd_dataset_sample_aligned.zip \
   --output ~/CMPS355_StableMate_Local/runs/stablemate_stage8_independent_audit
Output files are created only under --output. Does not touch run folders.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math
from pathlib import Path
from zipfile import ZipFile
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import rankdata

COHORTS=('d1','d2','d3','d4')
EXPECTED_DATA_SHA='8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74'
EXPECTED_CSV_SHA={'count_d1_aligned.csv': '0f595746ea572a349580d9d3c0242559345b88c927c161b1982933102e6c812d', 'count_d2_aligned.csv': '2ff413b813e3e4d10927da7f8342eccb35a00205c319c0eb714388aea97f245d', 'count_d3_aligned.csv': '33b2e51e682da4c41f53becf00b1fc88bb5710293ce2d9d87837bca86a7b3d48', 'count_d4_aligned.csv': 'c26d8db65b335c89967bd4e072c61a6b7adcb57715c5fe36107d9bd8cd9409e0', 'desc_d1.csv': '91aadec65bb7f2c1945e7941222b9f490a0c9041d495b8d22606f614484c7b33', 'desc_d2.csv': '4198f574167754eb00e007ded8cfc2528d50ead0d80b96c1a5cc8139131a140b', 'desc_d3.csv': '9db3c39975a50afcbd851d3cdcdf56f1e59e7f9bbe18b3b80fd7328400a31332', 'desc_d4.csv': 'bdd5f6d5ae9a6713470f1eadab2e62a961a38d417f27401359d55b5d906a1193'}

def sha_file(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as handle:
  for block in iter(lambda:handle.read(2**20),b''):h.update(block)
 return h.hexdigest()

def check(ok,message):
 if not ok: raise AssertionError(message)

def from_zip(path:Path):
 check(sha_file(path)==EXPECTED_DATA_SHA,'Dataset ZIP checksum mismatch')
 with ZipFile(path) as z:
  raw={k:z.read(k) for k in EXPECTED_CSV_SHA}
  for k,v in raw.items():
   check(hashlib.sha256(v).hexdigest()==EXPECTED_CSV_SHA[k],f'Archive CSV checksum mismatch: {k}')
  header={c:pd.read_csv(io.BytesIO(raw[f'count_{c}_aligned.csv']),nrows=0).columns[1:].tolist() for c in COHORTS}
  genes=sorted(set.intersection(*(set(header[c]) for c in COHORTS)))
  check(len(genes)==9983,'Unexpected shared gene count')
  meta={c:pd.read_csv(io.BytesIO(raw[f'desc_{c}.csv']),dtype={'ID':str,'Study':str,'class':str}) for c in COHORTS}
  mats={c:pd.read_csv(io.BytesIO(raw[f'count_{c}_aligned.csv']),usecols=['ID']+genes,dtype={'ID':str}) for c in COHORTS}
  for c in COHORTS:
   check(meta[c]['ID'].tolist()==mats[c]['ID'].tolist(),f'Mismatched raw counts/metadata: {c}')
   check(meta[c]['ID'].is_unique,f'Duplicate raw IDs: {c}')
  combined=pd.concat([meta[c].assign(cohort=c) for c in COHORTS],ignore_index=True)
  count=np.vstack([mats[c].loc[:,genes].to_numpy(float) for c in COHORTS])
  check(len(combined)==689 and combined.ID.is_unique,'Expected 689 distinct samples')
 return combined,count,genes

def metric(y,p,pred):
 pos=int(y.sum());neg=len(y)-pos
 tp=int(((y==1)&(pred==1)).sum());tn=int(((y==0)&(pred==0)).sum()); fp=neg-tn;fn=pos-tp
 pclip=np.clip(p,1e-15,1-1e-15)
 return {'n':len(y),'n_negative':neg,'n_positive':pos,'tn':tn,'tp':tp,'fn':fn,'fp':fp,
  'roc_auc':(rankdata(p)[y==1].sum()-(pos*(pos+1)/2))/(pos*neg),
  'accuracy':(tp+tn)/len(y),'balanced_accuracy':(tp/pos+tn/neg)/2,
  'sensitivity':tp/pos,'specificity':tn/neg,'precision':tp/(tp+fp) if tp+fp else 0,
  'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
  'brier_score':np.mean((p-y)**2),
  'log_loss':-np.mean(y*np.log(pclip)+(1-y)*np.log(1-pclip))}

def run(a):
 meta,counts,genes=from_zip(a.data_zip); fold_results=[]; signatures={};all_ids=set(); check_count=0
 for heldout in COHORTS:
  f=a.runs_root/f'{heldout}_K2000_B1000'
  check(f.is_dir(),f'Run folder absent: {f}')
  check('RUN_COMPLETE=TRUE' in (f/'RUN_COMPLETE.txt').read_text(),f'Run incomplete: {heldout}')
  checks=pd.read_csv(f/'output_checksums.csv')
  for row in checks.itertuples(index=False):
   target=f/row.file
   check(target.is_file(),f'Missing verified output {target}')
   check(sha_file(target)==row.sha256,f'Wrong SHA256 {target}')
   check_count+=1
  m=json.loads((f/'source_manifest.json').read_text())
  s=pd.read_csv(f/'run_summary.csv').iloc[0]
  reported=pd.read_csv(f/'heldout_metrics.csv').iloc[0]
  q=pd.read_csv(f/'heldout_predictions.csv',dtype={'ID':str,'Study':str,'class_true':str})
  split=pd.read_csv(f/'outer_split.csv',dtype={'ID':str})
  idx=np.flatnonzero(meta['cohort'].to_numpy()==heldout)
  ix=np.flatnonzero(meta['cohort'].to_numpy()!=heldout)
  check((len(idx),len(ix))==(int(s.test_n),int(s.train_n)),f'Incorrect split sizes: {heldout}')
  check(q.ID.tolist()==meta.iloc[idx].ID.tolist(),f'Test ID mismatch: {heldout}')
  check(q.Study.tolist()==meta.iloc[idx].Study.tolist(),f'Study mismatch: {heldout}')
  check(q.class_true.tolist()==meta.iloc[idx]['class'].tolist(),f'Class mismatch: {heldout}')
  check(not set(q.ID)&all_ids, f'Patient occurs in multiple tests: {heldout}')
  all_ids.update(q.ID)
  check(len(split)==689 and split.ID.is_unique and set(split.loc[split.split=='test','ID'])==set(q.ID),f'Split mismatch: {heldout}')
  check(m['data_sha256']==EXPECTED_DATA_SHA and m['heldout']==heldout,f'Manifest mismatch: {heldout}')
  check((int(s.K),int(s.B),int(s.workers))==(2000,1000,1),f'Runtime settings mismatch: {heldout}')
  y=q.y_true.to_numpy(int);p=q.probability_CD.to_numpy(float);pred=(p>=float(reported.threshold)).astype(int)
  check(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),f'Probability failure: {heldout}')
  check(np.array_equal(pred,q.y_pred.to_numpy(int)),f'Predicted class mismatch: {heldout}')
  metrics=metric(y,p,pred)
  for k,v in metrics.items():
   check(math.isclose(float(getattr(reported,k)),float(v),rel_tol=1e-9,abs_tol=1e-12),f'Metric {k} mismatch: {heldout}')
  scores=pd.read_csv(f/'author_gene_scores.csv'); sig=pd.read_csv(f/'significant_stable_genes.csv')
  correct=scores[(scores.predictive_significance>.975)&(scores.stability_significance>.975)]
  check(scores.gene.is_unique and len(scores)==int(s.scored_genes),f'Gene score count mismatch: {heldout}')
  check(set(correct.gene)==set(sig.gene) and len(correct)==int(s.stable_significant),f'Significant gene mismatch: {heldout}')
  check(int((scores.predictive_significance>.975).sum())==int(s.predictive_significant),f'Predictive significance mismatch: {heldout}')
  sigset=set(sig.gene);signatures[heldout]=sigset
  # Independent recreation of training-only filter, centering and standard scaling
  train=counts[ix]; train_log=np.log2(1+1e6*train/train.sum(axis=1,keepdims=True))
  threshold=max(2,int(np.ceil(.1*len(ix))))
  good=((train_log>np.log2(2)).sum(axis=0)>=threshold)&(train_log.var(axis=0)>1e-12)
  indices=np.flatnonzero(good)
  t=pd.read_csv(f/'gene_transform.csv')
  check(np.array_equal(indices,t.shared_universe_index.to_numpy(int)),f'Gene filtering mismatch: {heldout}')
  check([genes[i] for i in indices]==t.gene.tolist(),f'Gene names mismatch: {heldout}')
  check(np.max(np.abs(train_log[:,indices].mean(0)-t.train_mean_log2cpm.to_numpy()))<1e-12,f'Centres mismatch: {heldout}')
  check(np.max(np.abs(train_log[:,indices].std(0)-t.train_std_log2cpm.to_numpy()))<1e-12,f'Scales mismatch: {heldout}')
  fold_results.append({'fold':heldout,'n':len(y),'train_n':len(ix),'K':int(s.K),'B':int(s.B),
                       'workers_executed':int(s.workers),'workers_in_preparation_manifest':int(m['model']['ncore']),
                       'stable_significant_genes':len(sig),'scored_genes':len(scores),'eligible_genes':len(indices),**metrics})
 check(len(all_ids)==689,'Held-out coverage incomplete')
 overlap=[]
 for c,d in combinations(COHORTS,2):
  intersection=signatures[c]&signatures[d];union=signatures[c]|signatures[d]
  overlap.append({'fold_a':c,'fold_b':d,'intersection':len(intersection),'union':len(union),
                  'jaccard':len(intersection)/len(union) if union else float('nan'),
                  'intersection_genes':';'.join(sorted(intersection))})
 output=a.output
 output.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(fold_results).to_csv(output/'per_fold_metrics.csv',index=False)
 pd.DataFrame(overlap).to_csv(output/'gene_overlap.csv',index=False)
 summary={'status':'PASS_NUMERIC_AND_PREPROCESSING_CHECKS','folds':4,'patients':len(all_ids),
          'output_hashes_verified':check_count,'macro_auroc':float(np.mean([r['roc_auc'] for r in fold_results])),
          'macro_auroc_sd':float(np.std([r['roc_auc'] for r in fold_results],ddof=1)),
          'macro_balanced_accuracy':float(np.mean([r['balanced_accuracy'] for r in fold_results])),
          'fourway_stable_genes':sorted(set.intersection(*[signatures[c] for c in COHORTS])),
          'union_stable_genes':len(set.union(*[signatures[c] for c in COHORTS])),
          'notes':['RDS files hash-verified but R models not reloaded','Actual final script may differ from preparation manifest','LOCO gene overlap is not independent-cohort replication']}
 (output/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('PASS',summary)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--runs-root',type=Path,required=True)
 parser.add_argument('--data-zip',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 run(parser.parse_args())

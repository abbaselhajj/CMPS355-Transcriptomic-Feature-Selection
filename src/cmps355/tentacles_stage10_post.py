#!/usr/bin/env python3
"""Stage 10: test-only LOCO evaluation of native TENTACLES consensus panels.

NO TENTACLES retraining. The common downstream classifier is a frozen balanced
L2 logistic regression (C=1) fitted to 3 discovery cohorts using *training-only*
standardization of log2(1 + CPM) counts. This is NOT authors' native testConsensus.

Each log-CPM library-size denominator uses the entire harmonized 9,983-gene
count matrix, not just the selected panel. Held-out labels are accessed only
for scoring after model fitting, never for gene selection/model tuning.
"""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, recall_score, precision_score, f1_score,
    brier_score_loss, confusion_matrix,
)

FOLDS = ('d1', 'd2', 'd3', 'd4')
EXPECTED_N = {'d1': 260, 'd2':179, 'd3':76, 'd4':174}
EXPECTED_TRAIN_N = {'d1': 429, 'd2':510, 'd3':613, 'd4':515}
EXPECTED_GENES = 9983
EXPECTED_DATASET_SHA = '8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74'
METHOD = 'TENTACLES_native_consensus_fixed_L2_logistic'
SEED = 123

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as s:
        for b in iter(lambda: s.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def save_table(rows, path, columns):
    pd.DataFrame(rows,columns=columns).to_csv(path,index=False)

def get_finished(root: Path, fold: str) -> Path:
    parent = root / 'runs' / 'tentacles_stage9_loco' / fold
    paths = sorted(p for p in parent.iterdir() if p.is_dir())
    if not paths:
        raise RuntimeError(f'No runs for {fold}')
    result = paths[-1]
    required = ('STATUS.txt','runclassifiers_finished.rds',
                'consensus_native_finished.rds','consensus_genes_native.csv',
                'input_and_code_sha256.txt','output_rds_sha256.txt')
    missing = [x for x in required if not (result/x).is_file() or (result/x).stat().st_size == 0]
    if missing:
        raise RuntimeError(f'{fold} missing completed evidence: {missing}')
    if (result/'STATUS.txt').read_text().strip() != 'PASS_NATIVE_CLASSIFIERS_AND_CONSENSUS':
        raise RuntimeError(f'{fold} is not a passed native fold: {result}')
    # Original copied code, immutable original dataset, fold inputs, all saved native RDS files.
    for manifest in ('input_and_code_sha256.txt','output_rds_sha256.txt'):
        q = subprocess.run(['sha256sum','--status','-c',str(result/manifest)],
                           cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
        if q.returncode != 0:
            raise RuntimeError(f'{fold} checksum check failed for {manifest}: {q.stderr[:400]}')
    return result

def read_panel(finished: Path) -> list[str]:
    p = finished / 'consensus_genes_native.csv'
    table = pd.read_csv(p,dtype=str,keep_default_na=False)
    if list(table.columns) != ['gene']:
        raise RuntimeError(f'Unexpected native consensus gene CSV schema: {p}; {list(table.columns)}')
    genes = table['gene'].tolist()
    if len(genes)!=len(set(genes)) or any(not x.strip() or x != x.strip() for x in genes):
        raise RuntimeError(f'Invalid/duplicate native consensus gene names in {p}')
    return genes

def check_split(counts: pd.DataFrame, clinical: pd.DataFrame, fold: str, target: str):
    if len(counts) != len(clinical) or counts.ID.astype(str).tolist()!=clinical.ID.astype(str).tolist():
        raise RuntimeError(f'{fold}/{target}: counts to clinical patient alignment failed')
    if counts.ID.duplicated().any() or clinical.ID.duplicated().any():
        raise RuntimeError(f'{fold}/{target}: duplicate IDs')
    if not {'ID','Study','class','cohort'}.issubset(clinical.columns):
        raise RuntimeError(f'{fold}/{target}: missing clinical metadata')
    if not set(clinical['class'].unique()).issubset({'CD','Not IBD'}):
        raise RuntimeError(f'{fold}/{target}: unrecognized labels')
    if target=='test':
        if len(clinical)!=EXPECTED_N[fold] or set(clinical.cohort)!={fold}:
            raise RuntimeError(f'{fold}: expected {EXPECTED_N[fold]} heldout patients')
    else:
        if len(clinical)!=EXPECTED_TRAIN_N[fold] or fold in set(clinical.cohort) or len(set(clinical.cohort))!=3:
            raise RuntimeError(f'{fold}: invalid outer training set')
    if counts.shape[1] != EXPECTED_GENES+1:
        raise RuntimeError(f'{fold}/{target}: expected {EXPECTED_GENES} raw common genes')

def normalized_panel(counts: pd.DataFrame, genes: list[str]) -> np.ndarray:
    # All original 9,983 counts are needed to calculate each sample's total.
    numeric = counts.drop(columns=['ID'])
    a = numeric.to_numpy(dtype=np.float64,copy=False)
    if not np.isfinite(a).all() or (a < 0).any():
        raise RuntimeError('Invalid count matrix')
    library = a.sum(axis=1)
    if np.any(library <= 0):
        raise RuntimeError('Zero library-size patient')
    if not genes:
        return np.empty((len(counts),0),dtype=np.float64)
    x = numeric.loc[:,genes].to_numpy(dtype=np.float64,copy=False)
    return np.log2(1.0 + 1_000_000.0*x/library[:,None])

def metrics(fold: str, y: np.ndarray, p: np.ndarray, pred: np.ndarray, k:int):
    if len(np.unique(y))!=2:
        raise RuntimeError(f'{fold}: heldout has only one label')
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(heldout=fold,method=METHOD,n=len(y),n_positive=int(y.sum()),
        n_negative=int((1-y).sum()),n_selected_genes=k,threshold=0.5,
        roc_auc=float(roc_auc_score(y,p)),
        average_precision=float(average_precision_score(y,p)),
        accuracy=float(accuracy_score(y,pred)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        sensitivity=float(recall_score(y,pred,zero_division=0)),
        specificity=float(tn/(tn+fp)),
        precision=float(precision_score(y,pred,zero_division=0)),
        f1=float(f1_score(y,pred,zero_division=0)),
        brier_score=float(brier_score_loss(y,p)),
        TP=int(tp),TN=int(tn),FP=int(fp),FN=int(fn))

def bootstrap_ci(y,p,pred,iterations=1000,seed=SEED):
    rng=np.random.default_rng(seed)
    groups=(np.flatnonzero(y==0),np.flatnonzero(y==1))
    values=[]
    for _ in range(iterations):
        indices=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        values.append((roc_auc_score(y[indices],p[indices]),
                       balanced_accuracy_score(y[indices],pred[indices])))
    q=np.quantile(np.asarray(values),[.025,.975],axis=0)
    return dict(auroc_ci_low=float(q[0,0]),auroc_ci_high=float(q[1,0]),
                balanced_accuracy_ci_low=float(q[0,1]),
                balanced_accuracy_ci_high=float(q[1,1]))

def main(root: Path):
    root = root.resolve()
    cfg=json.loads((root/'configs/dataset.json').read_text())
    if cfg.get('sha256')!=EXPECTED_DATASET_SHA:
        raise RuntimeError('The accepted dataset checksum is not configured')
    if sha(root/'data/validated/ibd_dataset_sample_aligned.zip')!=EXPECTED_DATASET_SHA:
        raise RuntimeError('The original corrected sample-aligned dataset changed')
    runs={f:get_finished(root,f) for f in FOLDS}
    panels={f:read_panel(runs[f]) for f in FOLDS}
    now=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=root/'runs'/'tentacles_stage10_post'/now
    out.mkdir(parents=True,exist_ok=False)
    (out/'STAGE10_RUNNING.txt').write_text('Stage 10 held-out evaluation running. Do not report unfinished figures.\n')
    input_files = [root/'data/validated/ibd_dataset_sample_aligned.zip']
    pred_rows=[]; metric_rows=[]; selection_rows=[]; quality_rows=[]; test_union=set()
    gene_counter=Counter(g for p in panels.values() for g in p)
    for fold in FOLDS:
        print('STAGE10_EVALUATING',fold,flush=True)
        base=root/'runs'/'tentacles_stage9_inputs'/fold
        disc=base/'discovery'; test=base/'evaluation_only'
        paths=[disc/'counts.csv',disc/'clinical.csv',test/'counts.csv',test/'clinical.csv',runs[fold]/'consensus_genes_native.csv']
        input_files.extend(paths)
        train_c=pd.read_csv(paths[0],dtype={'ID':str},low_memory=False)
        train_m=pd.read_csv(paths[1],dtype={'ID':str},low_memory=False)
        test_c=pd.read_csv(paths[2],dtype={'ID':str},low_memory=False)
        test_m=pd.read_csv(paths[3],dtype={'ID':str},low_memory=False)
        check_split(train_c,train_m,fold,'train')
        check_split(test_c,test_m,fold,'test')
        if set(train_c.ID) & set(test_c.ID):
            raise RuntimeError(f'{fold}: patient leakage')
        if test_union & set(test_c.ID):
            raise RuntimeError(f'{fold}: duplicate heldout patient across folds')
        test_union.update(test_c.ID)
        p=panels[fold]
        missing=[gene for gene in p if gene not in train_c.columns or gene not in test_c.columns]
        if missing:
            raise RuntimeError(f'{fold}: native genes missing in common harmonized matrix: {missing[:8]}')
        for i,g in enumerate(p):
            selection_rows.append(dict(heldout=fold,method=METHOD,gene=g,panel_position=i+1))
        log=(runs[fold]/'classifiers.log').read_text(errors='replace')
        quality_rows.append(dict(heldout=fold,official_run_dir=str(runs[fold]),
            native_gene_count=len(p), mlp_weight_errors=log.count('too many ('),
            lightgbm_feature_fraction_errors=log.count('feature_fraction_bynode'),
            boruta_nan_warnings=log.count('getImp result contains NA(s) or NaN(s)'),
            evidence_status='native_fit_and_consensus_saved; model_quality_warnings_require_review'))
        if not p:
            metric_rows.append(dict(heldout=fold,method=METHOD,n=len(test_m),
                n_positive=int((test_m['class']=='CD').sum()),
                n_negative=int((test_m['class']=='Not IBD').sum()),
                n_selected_genes=0,reason='Empty native consensus; no external classifier fitted'))
            print('STAGE10_EMPTY_NATIVE_PANEL',fold,flush=True)
            continue
        x_tr=normalized_panel(train_c,p)
        x_te=normalized_panel(test_c,p)
        y_tr=(train_m['class'].to_numpy()=='CD').astype(int)
        y_te=(test_m['class'].to_numpy()=='CD').astype(int)
        if len(np.unique(y_tr))!=2:
            raise RuntimeError(f'{fold}: training has only one class')
        scaler=StandardScaler().fit(x_tr)
        mod=LogisticRegression(C=1.,penalty='l2',class_weight='balanced',
                               solver='liblinear',random_state=SEED,max_iter=5000)
        mod.fit(scaler.transform(x_tr),y_tr)
        if not np.array_equal(mod.classes_,[0,1]):
            raise RuntimeError('Unexpected CD-positive class semantics')
        probs=mod.predict_proba(scaler.transform(x_te))[:,1]
        if not np.isfinite(probs).all() or (probs<0).any() or (probs>1).any():
            raise RuntimeError(f'{fold}: invalid predictions')
        preds=(probs>=0.5).astype(int)
        item=metrics(fold,y_te,probs,preds,len(p))
        item.update(bootstrap_ci(y_te,probs,preds,seed=SEED+int(fold[1:])))
        metric_rows.append(item)
        for j in range(len(test_m)):
            pred_rows.append(dict(heldout=fold,method=METHOD,
                ID=str(test_m.ID.iloc[j]),Study=str(test_m.Study.iloc[j]),
                class_true=str(test_m['class'].iloc[j]),y_true=int(y_te[j]),
                probability_CD=float(probs[j]),y_pred=int(preds[j])))
        joblib.dump(dict(model=mod,scaler=scaler,genes=p,training_fold=fold,
                         normalization='log2(1+1e6*selected_gene_counts/all_9983_gene_counts)',
                         threshold=0.5,class_positive='CD'),out/f'{fold}_heldout_predictor.joblib',compress=3)
        print('STAGE10_HELDOUT_PASS',fold,'N=',len(test_m),'AUROC=',f'{item["roc_auc"]:.4f}',
              'BAL_ACC=',f'{item["balanced_accuracy"]:.4f}',flush=True)
    if len(test_union)!=689:
        raise RuntimeError(f'Expected 689 unique heldout patients, got {len(test_union)}')
    if len(pred_rows)!=sum(EXPECTED_N[f] for f in FOLDS if panels[f]):
        raise RuntimeError('Predicted patient count mismatch')
    save_table(pred_rows,out/'heldout_predictions.csv',
              ['heldout','method','ID','Study','class_true','y_true','probability_CD','y_pred'])
    save_table(metric_rows,out/'heldout_metrics.csv',
              ['heldout','method','n','n_positive','n_negative','n_selected_genes','threshold',
               'roc_auc','average_precision','accuracy','balanced_accuracy','sensitivity',
               'specificity','precision','f1','brier_score','TP','TN','FP','FN',
               'auroc_ci_low','auroc_ci_high','balanced_accuracy_ci_low',
               'balanced_accuracy_ci_high','reason'])
    save_table(selection_rows,out/'selected_genes.csv',
              ['heldout','method','gene','panel_position'])
    save_table(quality_rows,out/'native_classifier_warning_audit.csv',
              ['heldout','official_run_dir','native_gene_count','mlp_weight_errors',
               'lightgbm_feature_fraction_errors','boruta_nan_warnings','evidence_status'])
    pair=[]
    for a,b in itertools.combinations(FOLDS,2):
        s,t=set(panels[a]),set(panels[b])
        union=s|t
        pair.append(dict(fold_a=a,fold_b=b,genes_a=len(s),genes_b=len(t),
             intersection=len(s&t),union=len(union),
             jaccard=(len(s&t)/len(union) if union else np.nan)))
    save_table(pair,out/'pairwise_gene_jaccard.csv',
              ['fold_a','fold_b','genes_a','genes_b','intersection','union','jaccard'])
    freq=[dict(gene=g,number_of_folds=k,folds=';'.join(f for f in FOLDS if g in panels[f]))
          for g,k in sorted(gene_counter.items(),key=lambda z:(-z[1],z[0]))]
    save_table(freq,out/'gene_selection_frequency.csv',['gene','number_of_folds','folds'])
    all_four=[x['gene'] for x in freq if x['number_of_folds']==4]
    (out/'genes_in_all_four_loco_panels.txt').write_text('\n'.join(all_four)+('\n' if all_four else ''))
    numerical=pd.DataFrame(metric_rows)
    numerical=numerical[pd.to_numeric(numerical['roc_auc'],errors='coerce').notna()]
    if len(numerical):
        summary = dict(n_evaluable_folds=int(len(numerical)),
             mean_roc_auc_unweighted=float(numerical.roc_auc.mean()),
             sd_roc_auc_unweighted=float(numerical.roc_auc.std(ddof=1)) if len(numerical)>1 else None,
             mean_balanced_accuracy_unweighted=float(numerical.balanced_accuracy.mean()),
             sd_balanced_accuracy_unweighted=float(numerical.balanced_accuracy.std(ddof=1)) if len(numerical)>1 else None)
    else:
        summary=dict(n_evaluable_folds=0,mean_roc_auc_unweighted=None,
                     mean_balanced_accuracy_unweighted=None)
    summary.update(method=METHOD,n_native_folds=4,
         n_unique_heldout_patients=689,n_external_predictions=len(pred_rows),
         native_consensus_sizes={f:len(panels[f]) for f in FOLDS},
         n_genes_selected_in_all_four=len(all_four),
         n_genes_selected_at_least_twice=sum(x['number_of_folds']>=2 for x in freq),
         protocol='Fixed C=1, class_weight balanced, L2 logistic; log2(1+CPM) of full shared gene universe; outer-train-only scaling; threshold 0.5',
         important_limitation='TENTACLES native tuning uses random inner 5-fold CV; this Stage10 common predictor is not authors testConsensus or native model performance',
         warning_audit='MLP/LightGBM/Boruta issues persist and require human scientific audit')
    (out/'stage10_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'stage10_protocol.md').write_text('''# Stage 10 — External held-out prediction on TENTACLES native consensus panels

- Input: four official training-only TENTACLES consensus gene lists (5 of 9 voters; XGBoost excluded), selected without any held-out label access.
- Outer training: all other three cohorts; external test: untouched heldout cohort.
- Common downstream classifier, **not an author-native TENTACLES predictor**: scikit-learn logistic regression, L2 `C=1`, `class_weight=balanced`, `solver=liblinear`, threshold=0.5, seed=123, no test-based tuning.
- Feature transform: log2(1 + one million × raw gene count / raw sample total over all 9,983 shared genes), with fit of StandardScaler on the discovery cohorts only; apply frozen scaler to heldout cohort.
- All models trained on raw expression using only genes from the corresponding native consensus CSV. No batches corrected, no test-sample feature selection. Other feature-selection methods must receive the same downstream predictor for a controlled feature-selection comparison.
- AUROC and balanced-accuracy 95% percentile CIs from 1,000 stratified-within-class bootstrap replicates of each outer test cohort, preserving class counts.
- Model warnings from native TENTACLES fitting require independent review and are not erased by successful Stage10 predictions.
- Consensus overlap across four LOCO panels is a leave-one-cohort-out sensitivity analysis, not independent biomarker replication: the discovery sets overlap in two cohorts.
- The corrected aligned source dataset used here is also from the TENTACLES paper. This is an adaptation/benchmark, not independent disease-cohort discovery.
- Test patient IDs remain local/private. Do not upload raw prediction tables to a public repository.
''')
    manifest = [{'relative_path':str(p.relative_to(root)), 'sha256':sha(p)} for p in input_files]
    (out/'input_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (out/'output_sha256.txt').open('w') as f:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.name not in {'output_sha256.txt','STAGE10_RUNNING.txt'}:
                f.write(f'{sha(p)}  {p}\n')
    (out/'STAGE10_RUNNING.txt').rename(out/'STAGE10_PASS.txt')
    (out/'STAGE10_PASS.txt').write_text('PASS_NATIVE_CONSENSUS_PLUS_FIXED_EXTERNAL_HELDOUT_PREDICTIONS\n')
    print('STAGE10_EVALUATION_COMPLETE=TRUE',flush=True)
    print('POST_RESULT_DIRECTORY='+str(out),flush=True)
    print('MEAN_HELDOUT_AUROC='+str(summary['mean_roc_auc_unweighted']),flush=True)
    print('MEAN_HELDOUT_BALANCED_ACCURACY='+str(summary['mean_balanced_accuracy_unweighted']),flush=True)

if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python tentacles_stage10_post.py CMPS355_TENTACLES_Local_ROOT')
    main(Path(sys.argv[1]))

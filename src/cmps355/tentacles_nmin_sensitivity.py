"""Read-only TENTACLES n.min 5 vs 6 sensitivity; no native models refit.

Uses votes from previously fitted authors' getConsensus dataFrame, with
pre-existing nine contributing models and XGBoost excluded. Fits only a fixed
external downstream logistic classifier on the other three cohorts. Keeps
all patient-level predictions in private run output, not curated public results.
"""
from __future__ import annotations
import csv, hashlib, io, json, platform, sys, zipfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from .common_classifier_benchmark import metrics
from .full_gene_loco import load_data, log_cpm

FOLDS=('d1','d2','d3','d4')
ARCHIVE_STAMPS={
 'd1':'20261107T214908721364197Z','d2':'20261108T001846178551954Z',
 'd3':'20261108T015506088244494Z','d4':'20261108T034039482073012Z'}

def sha(b):return hashlib.sha256(b).hexdigest()

def bootstrap_delta(y, p5, p6, seed, repetitions=1000):
    rng=np.random.default_rng(seed)
    neg=np.flatnonzero(y==0); pos=np.flatnonzero(y==1)
    auc=[];bacc=[]
    for _ in range(repetitions):
       idx=np.concatenate([rng.choice(neg,len(neg),replace=True),rng.choice(pos,len(pos),replace=True)])
       auc.append(roc_auc_score(y[idx],p6[idx])-roc_auc_score(y[idx],p5[idx]))
       bacc.append(balanced_accuracy_score(y[idx],p6[idx]>=.5)-balanced_accuracy_score(y[idx],p5[idx]>=.5))
    return dict(delta_auroc_ci_lower=float(np.quantile(auc,.025)),delta_auroc_ci_upper=float(np.quantile(auc,.975)),delta_balanced_accuracy_ci_lower=float(np.quantile(bacc,.025)),delta_balanced_accuracy_ci_upper=float(np.quantile(bacc,.975)))

def main(root, audit_zip, output):
    root=Path(root).resolve();audit_zip=Path(audit_zip).resolve();output=Path(output).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('Refusing to overwrite a nonempty run directory')
    output.mkdir(parents=True,exist_ok=True)
    meta, counts, genes, ds=load_data(root)
    assert len(meta)==689 and len(genes)==9983 and counts.shape==(689,9983)
    x=log_cpm(counts)
    geneidx={g:i for i,g in enumerate(genes)}
    if len(geneidx)!=len(genes):raise ValueError('Duplicate genes')
    c=meta.cohort.to_numpy(); y=(meta['class'].to_numpy()=='CD').astype(int)
    existing=pd.read_csv(root/'runs/tentacles/summary/selected_genes.csv',dtype={'gene':str})
    previous=pd.read_csv(root/'runs/common_classifier/20261108T083024500243Z/patient_predictions_PRIVATE.csv',dtype={'ID':str})
    rows=[]; patients=[]; paired=[]; strict_genes=[]; inputs={}
    with zipfile.ZipFile(audit_zip) as zipf:
      for fold in FOLDS:
        p=f'{fold}/{ARCHIVE_STAMPS[fold]}/native_vote_integrity.csv'
        data=zipf.read(p)
        inputs[p]=sha(data)
        votes=pd.read_csv(io.BytesIO(data),dtype={'gene':str})
        if votes.gene.isna().any() or votes.gene.duplicated().any():raise ValueError(f'{fold}: invalid genes')
        if not votes.votes.between(0,9).all() or not np.array_equal(votes.in_consensus.to_numpy().astype(bool),votes.votes.to_numpy()>=5):raise ValueError(f'{fold}: consensus vote audit mismatch')
        panels={t:votes.loc[votes.votes>=t,'gene'].tolist() for t in (5,6)}
        published=set(existing.loc[existing.heldout==fold,'gene'])
        if set(panels[5])!=published or len(panels[5])!=len(published):raise ValueError(f'{fold}: published nmin=5 genes mismatch')
        tr=np.flatnonzero(c!=fold);te=np.flatnonzero(c==fold)
        pred={}
        for t in (5,6):
            g=panels[t];missing=set(g)-set(geneidx)
            if not g or missing:raise ValueError(f'{fold} nmin={t}: missing genes {missing}')
            cols=[geneidx[q] for q in g]
            scl=StandardScaler().fit(x[np.ix_(tr,cols)])
            model=LogisticRegression(C=1.0,class_weight='balanced',solver='liblinear',max_iter=5000,random_state=123)
            model.fit(scl.transform(x[np.ix_(tr,cols)]),y[tr])
            prob=model.predict_proba(scl.transform(x[np.ix_(te,cols)]))[:,1]
            pred[t]=prob
            s=metrics(y[te],prob)
            rows.append(dict(heldout=fold,threshold=t,n_genes=len(g),n_test=len(te),n_train=len(tr),**s))
            for idx,pr in zip(te,prob):
                patients.append(dict(heldout=fold,threshold=t,ID=str(meta.ID.iloc[idx]),y_true=int(y[idx]),probability_CD=float(pr)))
            if t==6:
                strict_genes.extend(dict(heldout=fold, gene=q, votes=int(votes.set_index('gene').loc[q,'votes'])) for q in g)
        old=previous[(previous.heldout==fold)&(previous.method=='TENTACLES_native_consensus')].set_index('ID')
        pr=pd.Series(pred[5],index=meta.ID.iloc[te].astype(str))
        if set(old.index)!=set(pr.index):raise ValueError(f'{fold}: old/common patient IDs mismatch')
        delta=float(np.max(np.abs(old.loc[pr.index,'probability_CD'].to_numpy()-pr.to_numpy())))
        if delta>=1e-8:raise ValueError(f'{fold}: re-fitted >=5 vote model differs from earlier common benchmark, max {delta}')
        scores=bootstrap_delta(y[te],pred[5],pred[6],355+len(tr),1000)
        a5=metrics(y[te],pred[5]); a6=metrics(y[te],pred[6])
        paired.append(dict(heldout=fold,n_genes_5=len(panels[5]),n_genes_6=len(panels[6]),
          removed_genes=len(panels[5])-len(panels[6]),delta_auroc=a6['auroc']-a5['auroc'],delta_balanced_accuracy=a6['balanced_accuracy']-a5['balanced_accuracy'],
          max_abs_prob_difference_vs_previous_common_run=delta,**scores))
    rows=pd.DataFrame(rows); paired=pd.DataFrame(paired)
    rows.to_csv(output/'heldout_metrics_by_threshold.csv',index=False)
    paired.to_csv(output/'paired_deltas.csv',index=False)
    pd.DataFrame(patients).to_csv(output/'patient_predictions_PRIVATE.csv',index=False)
    pd.DataFrame(strict_genes).to_csv(output/'strict_nmin6_gene_panels.csv',index=False)
    macro=rows.groupby('threshold').agg(mean_auroc=('auroc','mean'),mean_balanced_accuracy=('balanced_accuracy','mean'),mean_panel_size=('n_genes','mean'),sd_auroc=('auroc','std'),min_panel_size=('n_genes','min'),max_panel_size=('n_genes','max')).reset_index()
    macro.to_csv(output/'macro_summary.csv',index=False)
    report={'status':'PASS_NMIN5_VS_6_PAIRED_LOCO_SENSITIVITY','created_utc':datetime.now(timezone.utc).isoformat(),
      'dataset_sha256':ds['zip_sha256'],'audit_zip_sha256':sha(audit_zip.read_bytes()),'vote_table_sha256':inputs,
      'source':'9 retained official TENTACLES model vote columns, XGBoost excluded, no native training or consensus rerun',
      'published_threshold':6,'original_stage9_threshold':5,
      'classifier':'LogisticRegression C=1 class_weight=balanced solver=liblinear max_iter=5000, train-only StandardScaler, fixed 0.5 threshold',
      'expression':'log2(1 + 1e6 * count / per-patient total across shared 9983 genes)',
      'ci':'1000 patient-stratified paired bootstrap replicates within each cohort; descriptive, not definitive across-cohort uncertainty',
      'primary_interpretation':'threshold sensitivity of a fixed downstream model; not the original authors\' 28-gene D1 panel, and not native MLP performance',
      'files_sha256':{f.name:sha(f.read_bytes()) for f in output.glob('*.csv')}}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return rows,macro,paired,report
if __name__=='__main__':
    if len(sys.argv)!=4:raise SystemExit('Usage python -m src.cmps355.tentacles_nmin_sensitivity PROJECT_ROOT NATIVE_AUDIT_ZIP OUT')
    m,ma,pa,r=main(*sys.argv[1:])
    print('PASS',r['status']);print(m[['heldout','threshold','n_genes','auroc','balanced_accuracy']].to_string(index=False));print(ma.to_string(index=False));print(pa.to_string(index=False))

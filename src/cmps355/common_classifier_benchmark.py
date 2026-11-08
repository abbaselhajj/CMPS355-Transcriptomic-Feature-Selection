"""Frozen 5-selector / common-logistic four-fold LOCO comparison.

This experiment does NOT replace native StableMate / TENTACLES predictions.
It isolates differences in frozen, outer-training-selected gene panels by using
one standardized downstream logistic regression for every selector and fold.
Same data, targets, normalization, threshold, training-only scaling and model.
Internal selection methods and signature sizes still differ; don't claim a
randomized causal experiment or independent biomarker replication.
"""
from __future__ import annotations
import csv,hashlib,json,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score,average_precision_score,balanced_accuracy_score,
                              confusion_matrix,precision_score,recall_score,f1_score,brier_score_loss)
from .full_gene_loco import load_data, log_cpm

COHORTS=("d1","d2","d3","d4")
METHODS=("lasso","elastic_net","stability_selection_cutoff_0p60","StableMate_significant_stable","TENTACLES_native_consensus")
SEED=123

def file_hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def panels_from_runs(root):
    root=Path(root)
    lasso_path=root/'runs/full_gene_results_report/20261107T091750523128Z/table_selected_genes.csv'
    lasso=pd.read_csv(lasso_path,dtype={"gene":str})
    tpath=root/'runs/tentacles/summary/selected_genes.csv'
    tentacles=pd.read_csv(tpath,dtype={"gene":str})
    all_panels={}; receipts={str(lasso_path.relative_to(root)):file_hash(lasso_path),str(tpath.relative_to(root)):file_hash(tpath)}
    for fold in COHORTS:
        for method in ("lasso","elastic_net"):
            subset=lasso[(lasso.outer==fold)&(lasso.method==method)]
            all_panels[(fold,method)]=subset.gene.tolist()
        p=root/f'runs/stablemate/{fold}_K2000_B1000/significant_stable_genes.csv'
        sm=pd.read_csv(p,dtype={"gene":str})
        if len(sm)!=int(sm.significantly_stable_predictive.sum()) or set(sm.heldout)!={fold}:
            raise ValueError(f'{fold}: StableMate significance panel inconsistent')
        all_panels[(fold,"StableMate_significant_stable")]=sm.gene.tolist()
        receipts[str(p.relative_to(root))]=file_hash(p)
        p=root/f'runs/stability_selection/{fold}/loco_training/threshold_signatures.csv'
        stable=pd.read_csv(p,dtype={"gene":str})
        all_panels[(fold,"stability_selection_cutoff_0p60")]=stable[stable.cutoff.round(5)==0.6].gene.tolist()
        receipts[str(p.relative_to(root))]=file_hash(p)
        subset=tentacles[tentacles.heldout==fold]
        all_panels[(fold,"TENTACLES_native_consensus")]=subset.gene.tolist()
    for (fold,method),panel in all_panels.items():
        if not panel or len(panel)!=len(set(panel)) or any(not isinstance(g,str) or not g.strip() for g in panel):
            raise ValueError(f'{fold}/{method}: missing or duplicated genes')
    return all_panels,receipts

def metrics(y,prob):
    labels=(prob>=0.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,labels,labels=[0,1]).ravel().tolist()
    return dict(auroc=float(roc_auc_score(y,prob)),average_precision=float(average_precision_score(y,prob)),
       balanced_accuracy=float(balanced_accuracy_score(y,labels)),precision=float(precision_score(y,labels,zero_division=0)),
       sensitivity=float(recall_score(y,labels)),specificity=tn/(tn+fp),f1=float(f1_score(y,labels,zero_division=0)),
       brier=float(brier_score_loss(y,prob)),tp=tp,tn=tn,fp=fp,fn=fn)

def main(root:Path,out:Path):
    root,out=Path(root).resolve(),Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(f"Output already exists: {out}")
    out.mkdir(parents=True,exist_ok=True)
    meta,counts,genes,dataset=load_data(root)
    panels,receipts=panels_from_runs(root)
    gene_idx={g:i for i,g in enumerate(genes)}
    assert len(gene_idx)==9983
    x_all=log_cpm(counts)  # no train/test fitting: each row is normalized on itself
    meta_cohort=meta.cohort.to_numpy()
    y=(meta['class'].to_numpy()=='CD').astype(int)
    archived=pd.read_csv(root/'runs/tentacles/summary/heldout_predictions.csv',dtype={"ID":str})
    rows=[];patients=[];panel_rows=[];checks={}
    for heldout in COHORTS:
        test=np.flatnonzero(meta_cohort==heldout)
        train=np.flatnonzero(meta_cohort!=heldout)
        assert len(train)+len(test)==len(meta)==689
        assert set(meta.ID.iloc[train]).isdisjoint(meta.ID.iloc[test])
        for method in METHODS:
            panel=panels[(heldout,method)]
            missing=set(panel)-set(gene_idx)
            if missing: raise ValueError(f'{heldout}/{method} has genes outside shared universe: {sorted(missing)[:8]}')
            loc=[gene_idx[g] for g in panel]
            train_x=x_all[np.ix_(train,loc)]
            test_x=x_all[np.ix_(test,loc)]
            scaler=StandardScaler().fit(train_x)
            model=LogisticRegression(C=1.,class_weight='balanced',solver='liblinear',max_iter=5000,random_state=SEED)
            model.fit(scaler.transform(train_x),y[train])
            assert np.array_equal(model.classes_,[0,1])
            probs=model.predict_proba(scaler.transform(test_x))[:,1]
            assert np.isfinite(probs).all() and (0<=probs).all() and (probs<=1).all()
            score=metrics(y[test],probs)
            rows.append(dict(heldout=heldout,method=method,n_test=len(test),n_train=len(train),n_genes=len(panel),**score))
            for i,p in zip(test,probs):
                patients.append(dict(heldout=heldout,method=method,ID=str(meta.ID.iloc[i]),Study=str(meta.Study.iloc[i]),
                                     class_true=str(meta['class'].iloc[i]),y_true=int(y[i]),probability_CD=float(p),y_pred=int(p>=0.5)))
            for gene in panel:panel_rows.append(dict(heldout=heldout,method=method,gene=gene))
            if method=='TENTACLES_native_consensus':
                expected=archived[archived.heldout==heldout].set_index('ID')
                assert set(expected.index)==set(meta.ID.iloc[test])
                expected=expected.loc[meta.ID.iloc[test]]
                assert np.array_equal(expected.y_true.to_numpy(),y[test])
                delta=float(np.max(np.abs(expected.probability_CD.to_numpy()-probs)))
                checks[heldout]=dict(max_abs_prob_difference=delta,archived_stage10_near_identical_within_1e_3=delta<1e-3)
                if delta>=0.05:raise ValueError(f'{heldout}: large divergence from frozen Stage 10: {delta}')
            print(f'COMMON_CLASSIFIER {heldout}/{method}: N_GENES={len(panel)} AUROC={score["auroc"]:.4f} BACC={score["balanced_accuracy"]:.4f}',flush=True)
    df=pd.DataFrame(rows);pred=pd.DataFrame(patients)
    df.to_csv(out/'cohort_metrics.csv',index=False)
    pred.to_csv(out/'patient_predictions_PRIVATE.csv',index=False)
    pd.DataFrame(panel_rows).to_csv(out/'selected_panel_receipts.csv',index=False)
    macro=df.groupby('method',sort=False).agg(mean_auroc=('auroc','mean'),sd_auroc=('auroc','std'),
               mean_balanced_accuracy=('balanced_accuracy','mean'),sd_balanced_accuracy=('balanced_accuracy','std'),
               mean_signature_size=('n_genes','mean'),min_signature_size=('n_genes','min'),
               max_signature_size=('n_genes','max'),folds=('heldout','nunique')).reset_index()
    macro.to_csv(out/'macro_summary.csv',index=False)
    report=dict(status='PASS_IDENTICAL_DOWNSTREAM_MODEL_5_METHODS_4_FOLDS',
        created_utc=datetime.now(timezone.utc).isoformat(),
        dataset_sha256=dataset['zip_sha256'],input_panel_sha256=receipts,
        models='balanced logistic regression, liblinear, penalty L2, C=1, max_iter=5000, threshold=0.5',
        normalization='log2(1+1e6 * count / per-sample total across shared 9983 genes)',
        scaling='StandardScaler fit only on 3 outer training cohorts',
        feature_panels='original method-specific outer-training-only selected genes; no re-selection or additional prefilter',
        classes='CD=positive 1, Not IBD=negative 0',
        stages='LASSO/elastic net sparse nonzero coefficients, Stability Selection prespecified cutoff 0.60, StableMate significantly stable predictive, TENTACLES native consensus',
        model_comparison_limit='Controls downstream classifier and preprocessing, NOT upstream selection method hyperparameters, signature sizes, or overlapping training cohorts',
        stage10_reproduction=checks,
        sklearn_version=sklearn.__version__,python_version=platform.python_version(),
        folds=4,methods=5,n_predictions=len(pred),
        output_sha256={p.name:file_hash(p) for p in out.glob('*.csv')})
    (out/'report.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('Usage: python -m src.cmps355.common_classifier_benchmark ROOT OUTPUT_DIR')
    main(Path(sys.argv[1]),Path(sys.argv[2]))

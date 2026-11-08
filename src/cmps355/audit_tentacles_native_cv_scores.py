#!/usr/bin/env python3
"""Reproducible, read-only audit for TENTACLES result_scores.zip (2026-11-08).

Recomputes CV score consistency and infers the positive class of apparent
final metrics from the original LOCO training-set class counts. Does NOT infer
per-resample CV confusion matrices or retrospectively change R tuning.
"""
from pathlib import Path
from zipfile import ZipFile
import argparse
import io
import json
import hashlib
import numpy as np
import pandas as pd

EXPECTED = {
    'd1': {'n_train':429,'n_cd':277,'n_not_ibd':152},
    'd2': {'n_train':510,'n_cd':361,'n_not_ibd':149},
    'd3': {'n_train':613,'n_cd':461,'n_not_ibd':152},
    'd4': {'n_train':515,'n_cd':386,'n_not_ibd':129},
}
MODELS={'rand_forest_corr','mlp_roc','bag_mlp_roc','svm_linear_boruta','C5_rules_boruta','lightGBM_roc','decision_tree_boruta','mars_boruta','xgboost_base','pls_boruta'}

def audit(zip_path:Path, out:Path):
    out.mkdir(parents=True,exist_ok=True)
    checks=0
    def require(condition,message):
        nonlocal checks
        if not condition:raise AssertionError(message)
        checks+=1
    with ZipFile(zip_path) as z:
        files=set(z.namelist())
        def get(name):
            require(name in files,f'missing {name}')
            return pd.read_csv(io.BytesIO(z.read(name)))
        scores=get('all_models_crossvalidated_f1.csv')
        quality=get('crossvalidated_score_quality_summary.csv')
        require(z.read('AUDIT_STATUS.txt').decode().startswith('PASS_CV_F1_FINITE_ALL_MODELS'), 'Extractor status not PASS')
        require(len(scores)==40,'Expected 40 model/fold scores')
        require(not scores[['heldout','wflow_id']].duplicated().any(),'Duplicate cross validated score')
        require(set(scores.heldout)==set(EXPECTED),'Missing cohorts')
        require(quality.set_index('heldout').loc['d1','n_cv_f1_finite']==10,'Quality summary missing')
        allrows=[]
        allfinal=[]
        foldsummary=[]
        for fold,case in EXPECTED.items():
            tuning=get(f'{fold}_tuning_metrics.csv')
            apparent=get(f'{fold}_final_metrics.csv')
            this=scores[scores.heldout==fold].copy()
            require(len(tuning)==40 and len(apparent)==40,f'{fold} metric coverage')
            require(len(this)==10 and set(this.wflow_id)==MODELS,f'{fold} model names')
            require(set(tuning['.metric'])=={'accuracy','f_meas','precision','recall'},f'{fold} metric types')
            require(all(this.n==5),f'{fold} incomplete CV resamples')
            require(this['mean'].between(0,1).all() and np.isfinite(this['mean']).all(),f'{fold} invalid F1')
            require(np.isfinite(this['std_err']).all(), f'{fold} invalid CV SE')
            require(tuning['mean'].between(0,1).all() and np.isfinite(tuning['mean']).all(),f'{fold} incomplete tuning metrics')
            require(apparent['.estimate'].between(0,1).all(),f'{fold} invalid apparent estimates')
            require(set(apparent.model)==MODELS,f'{fold} final model names')
            require(not tuning[['wflow_id','.metric']].duplicated().any(),f'{fold} duplicates')
            cv=tuning[tuning['.metric']=='f_meas'].set_index('wflow_id')
            scores_sorted=this.set_index('wflow_id')
            require(np.allclose(cv.loc[scores_sorted.index,'mean'],scores_sorted['mean'],rtol=0,atol=1e-13),f'{fold} summary CV mean mismatch')
            require(np.allclose(cv.loc[scores_sorted.index,'std_err'],scores_sorted['std_err'],rtol=0,atol=1e-13),f'{fold} summary CV stderr mismatch')
            require(sorted(cv['rank'].to_list())==list(range(1,11)),f'{fold} ranks not 1..10')
            require((cv.sort_values('mean',ascending=False)['rank'].to_numpy()==np.arange(1,11)).all(),f'{fold} rank and score mismatch')
            apparent_pivot=apparent.pivot(index='model',columns='.metric',values='.estimate')
            p=apparent_pivot['precision'];r=apparent_pivot['recall'];a=apparent_pivot['accuracy']
            require(np.allclose(2*p*r/(p+r),apparent_pivot['f_meas'],atol=1e-12),f'{fold} F1 formula inconsistent')
            # Invert accuracy + binary precision/recall to recover positive-label prevalence.
            # Errors = FN+FP = P*((1-recall)+recall*(1/precision-1)).
            den=((1-r)+r*(1/p-1))
            require((den>0).all(),f'{fold} ill-defined prevalence denominator')
            p_inferred=case['n_train']*(1-a)/den
            require(np.max(np.abs(p_inferred-case['n_not_ibd']))<1e-8,f'{fold} positive class not Not IBD')
            tp=p_inferred*r
            fn=p_inferred-tp
            fp=tp*(1/p-1)
            tn=case['n_train']-tp-fn-fp
            require(np.max(np.abs(np.column_stack([tp,fn,fp,tn])-np.rint(np.column_stack([tp,fn,fp,tn]))))<1e-8,f'{fold} non-integral confusion counts')
            for model in sorted(MODELS):
                cvrow=scores_sorted.loc[model]
                row=apparent_pivot.loc[model]
                tt=int(round(tp.loc[model]));nn=int(round(tn.loc[model])); ffpp=int(round(fp.loc[model])); ffnn=int(round(fn.loc[model]))
                # Non-IBD event: TP=tt, TN=nn; reorient to CD event: TP=nn, TN=tt.
                cd_precision=nn/(nn+ffnn) if nn+ffnn>0 else np.nan
                cd_recall=nn/(nn+ffpp) if nn+ffpp>0 else np.nan
                cd_f1=2*cd_precision*cd_recall/(cd_precision+cd_recall) if cd_precision+cd_recall>0 else np.nan
                allfinal.append(dict(heldout=fold,model=model,train_n=case['n_train'],train_cd=case['n_cd'],train_not_ibd=case['n_not_ibd'],inferred_positive_class='Not IBD',inferred_positive_count=float(p_inferred.loc[model]),apparent_accuracy=float(row['accuracy']),apparent_f1_Not_IBD=float(row['f_meas']),apparent_f1_CD_derived=float(cd_f1),NotIBD_TP=tt,NotIBD_FN=ffnn,NotIBD_FP=ffpp,NotIBD_TN=nn,NOTICE='Apparent metrics computed on training cohort, NOT outer held-out; CD-positive values reconstructed post hoc only.'))
                warningflag=('boruta' in model or model in ['mlp_roc','bag_mlp_roc','lightGBM_roc'])
                allrows.append(dict(heldout=fold,model=model,cv_F1_NotIBD=float(cvrow['mean']),cv_F1_std_err=float(cvrow['std_err']),cv_n_resamples=int(cvrow['n']),rank=int(cv.loc[model,'rank']),apparent_F1_NotIBD=float(row['f_meas']),apparent_minus_cv=float(row['f_meas']-cvrow['mean']),warning_relevant_method=warningflag,notes='CV F1 likely controls-as-positive; warning relevance cannot be ruled out from aggregate scores alone' if warningflag else 'Aggregate CV F1 finite; candidate failures cannot be excluded'))
            foldsummary.append(dict(heldout=fold,n_models=10,cv_score_entries=10,cv_resamples_each=5,mean_CV_F1_Not_IBD=this['mean'].mean(),min_CV_F1_Not_IBD=this['mean'].min(),max_CV_F1_Not_IBD=this['mean'].max(),mean_apparent_CV_gap=np.mean([s['apparent_minus_cv'] for s in allrows if s['heldout']==fold]),positive_class_proven_from_final_metrics='Not IBD',positive_count=case['n_not_ibd'],train_n=case['n_train']))
        require(quality['n_models'].eq(10).all() and quality['n_cv_f1_finite'].eq(10).all() and quality['n_cv_f1_bounded'].eq(10).all(),'Overall score quality summary inconsistent')
    detail=pd.DataFrame(allrows)
    orientation=pd.DataFrame(allfinal)
    summary=pd.DataFrame(foldsummary)
    detail.to_csv(out/'native_cv_f1_by_model.csv',index=False)
    summary.to_csv(out/'native_cv_f1_by_fold.csv',index=False)
    orientation.to_csv(out/'native_apparent_metrics_positive_class_proof.csv',index=False)
    evidence=dict(source_zip_sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest(),source_filename=zip_path.name,checks_passed=checks,failures=0,folds=4,model_folds=40,positive_class_final='Not IBD',positive_class_proof='All 40 native apparent accuracy/precision/recall tuples imply exactly the Not IBD prevalence and integral confusion matrices under the full training n; CD prevalence does not fit.',tuning_metric='f_meas',cv_metric_event_label_inferred='Not IBD (R yardstick default event-level likely first; inspect package factor levels for direct confirmation)',cv_n_each=5,cv_scores_finite=True,unresolved=['No per-resample confusion matrices; cannot independently recompute CD-positive CV F1 or retune hyperparameters','Warnings for MLP, LightGBM, Boruta may reflect failed candidate fits even when selected CV summaries finite','The apparent final_metrics are full-training resubstitution scores, not held-out external validation','Native model objects and selected hyperparameter engines not exhaustively inspected'],recommended_action='Retain all current results, correct scientific descriptions; obtain native factor levels and stored resampling predictions if available before deciding whether a CD-positive tuning rerun is necessary.')
    (out/'audit.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    return evidence,detail,summary,orientation

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('zip');a.add_argument('output');args=a.parse_args()
    info,rows,summary,orientation=audit(Path(args.zip),Path(args.output))
    print(json.dumps(info,indent=2))
    print(summary.to_string(index=False))

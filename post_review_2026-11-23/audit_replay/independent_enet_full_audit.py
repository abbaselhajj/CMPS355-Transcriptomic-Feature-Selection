#!/usr/bin/env python3
"""Read-only, independent numerical and prediction audit of archived CMPS355 ENet v2.
Neither reads as training refits nor writes to the canonical Drive project.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import hashlib, json, math, re, zipfile, csv, io, sys
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix, brier_score_loss, precision_score, f1_score
from threadpoolctl import threadpool_limits

ROOT=Path('/mnt/data/enet_complete_audit_20261122')
ZIP=Path('/mnt/data/ibd_dataset_sample_aligned.zip')
COHORTS=('d1','d2','d3','d4')
CGRID=(.001,.01,.1,1.)
RATIOS=(.5,.9)
EXPECTED_SCRIPT='bc36f7379399e73b70e885d153d194b4a10943c04e9c6d1761c482afa71d0925'
EXPECTED_DATA='8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74'
errors=[]
summary={}

def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def sha_list(x):return sha_bytes('\n'.join(str(v) for v in x).encode())
def assert_equal(cond,msg):
 if not cond:errors.append(msg)
def parse_ckpt(fid):
 if re.match(r'^loco_inner_d[1-4]_val_d[1-4]_C[0-9p]+_r[0-9p]+$',fid):
  m=re.fullmatch(r'loco_inner_(d\d)_val_(d\d)_C([0-9p]+)_r([0-9p]+)',fid);return ('loco_inner',m[1],m[2],float(m[3].replace('p','.')),float(m[4].replace('p','.')))
 if re.match(r'^indep_cv_d[1-4]_f[1-3]_C[0-9p]+_r[0-9p]+$',fid):
  m=re.fullmatch(r'indep_cv_(d\d)_f([1-3])_C([0-9p]+)_r([0-9p]+)',fid);return ('indep_cv',m[1],int(m[2]),float(m[3].replace('p','.')),float(m[4].replace('p','.')))
 if re.fullmatch(r'loco_d[1-4]_corrected_enet',fid):return ('loco_final',fid.split('_')[1],None,None,None)
 if re.fullmatch(r'indep_d[1-4]_corrected_enet',fid):return ('indep_final',fid.split('_')[1],None,None,None)
 return None

def reconstruct_data():
 assert_equal(sha_bytes(ZIP.read_bytes())==EXPECTED_DATA,'Dataset ZIP SHA mismatch')
 with zipfile.ZipFile(ZIP) as z:
  csvs={x:z.read(x) for x in z.namelist() if x.endswith('.csv')}
  headers={c:pd.read_csv(io.BytesIO(csvs['count_'+c+'_aligned.csv']),nrows=0).columns.tolist()[1:] for c in COHORTS}
  genes=tuple(sorted(set.intersection(*(set(v) for v in headers.values()))))
  assert_equal(len(genes)==9983,'Gene intersection not 9983')
  metas=[]; counts=[]
  for c in COHORTS:
   d=pd.read_csv(io.BytesIO(csvs[f'desc_{c}.csv']),dtype={'ID':str})
   cnt=pd.read_csv(io.BytesIO(csvs[f'count_{c}_aligned.csv']),usecols=['ID',*genes],dtype={'ID':str})
   assert_equal(d.ID.tolist()==cnt.ID.tolist(),f'{c} counts/metadata misaligned')
   d.insert(0,'cohort',c);metas.append(d)
   counts.append(cnt.loc[:,list(genes)].to_numpy(np.float64))
  meta=pd.concat(metas,ignore_index=True)
  return meta,np.vstack(counts),genes

def get_prep(counts, mask):
 arr=counts[mask]
 depths=arr.sum(1)
 cpm=1e6*arr/depths[:,None]
 logcpm=np.log2(1+cpm)
 n=len(arr)
 valid=((cpm>1).sum(0)>=max(2, math.ceil(.10*n))) & (logcpm.var(0)>1e-12)
 idx=np.flatnonzero(valid)
 block=logcpm[:,idx]
 mu=block.mean(0);sd=block.std(0,ddof=0)
 sd=np.where(sd==0,1,sd)
 X=(block-mu)/sd
 return X,idx,mu,sd

def get_val(counts,mask,idx,mu,sd):
 a=counts[mask];logs=np.log2(1+1e6*a/a.sum(1)[:,None]);return (logs[:,idx]-mu)/sd

def independent_kkt(X,y,beta,b0,C,r):
 w=np.where(y==1,len(y)/(2*int(sum(y==1))),len(y)/(2*int(sum(y==0))))
 z=X@beta+b0
 p=expit(z)
 loss=float(np.sum(w*(np.logaddexp(0,z)-y*z)))
 lam1=float(r/C);lam2=float((1-r)/C)
 obj=loss+lam1*np.abs(beta).sum()+0.5*lam2*beta@beta
 grad=X.T@(w*(p-y))+lam2*beta
 gb=float(np.sum(w*(p-y)))
 active=beta!=0.0
 ka=float(np.max(np.abs(grad[active]+lam1*np.sign(beta[active])))) if np.any(active) else 0.0
 ki=float(np.maximum(np.abs(grad[~active])-lam1,0.0).max()) if np.any(~active) else 0.0
 return {'common_objective':float(obj),'data_loss':loss,'l1_penalty':float(lam1*np.abs(beta).sum()),'l2_penalty':float(0.5*lam2*(beta@beta)),
    'max_kkt_residual':max(ka,ki,abs(gb)),'kkt_active':ka,'kkt_inactive':ki,'kkt_intercept':abs(gb)}

def stable_boot_ci(y,score,metric,n=2000,seed=123):
 rng=np.random.default_rng(seed)
 pos=np.where(y==1)[0];neg=np.where(y==0)[0]
 vals=[]
 for _ in range(n):
  idx=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)])
  vals.append(float(metric(y[idx],score[idx])))
 return tuple(np.percentile(vals,[2.5,97.5]))

def bacc_func(y,pred):
 pos=(y==1);neg=(y==0)
 return float(.5*((pred[pos]==1).mean()+(pred[neg]==0).mean()))

def main():
 with threadpool_limits(limits=2):
  script_sha=sha_bytes((ROOT/'corrected_nested_elastic_net_v2.py').read_bytes())
  assert_equal(script_sha==EXPECTED_SCRIPT, 'Production source SHA mismatch')
  hashes={rel:dig for dig,rel in [line.split('  ',1) for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines()]}
  failures=[rel for rel,h in hashes.items() if not (ROOT/rel).is_file() or sha_bytes((ROOT/rel).read_bytes())!=h]
  assert_equal(len(hashes)==255,'Not 255 checksums')
  assert_equal(not failures, f'Checksum mismatches {failures[:6]}')
  summary['hashes']={'checked':len(hashes),'mismatch':len(failures)}

  meta,counts,genes=reconstruct_data()
  y_all=(meta['class']=='CD').astype(int).to_numpy();coh=meta.cohort.to_numpy();ids=meta.ID.astype(str).to_numpy()
  assert_equal(len(meta)==689, 'not 689 samples')
  zip_sha=sha_bytes(ZIP.read_bytes())
  ckpt_files=list((ROOT/'checkpoints/fits').glob('*.json'))
  assert_equal(len(ckpt_files)==200,'not exactly 200 checkpoints')
  checkpoints={f.stem:json.loads(f.read_text()) for f in ckpt_files}
  manifest=json.loads((ROOT/'manifest.json').read_text())
  assert_equal(manifest['certification_status']=='CERTIFIED_COMPLETE','Not complete')
  assert_equal(manifest['certification_summary']['certified_fits']==200,'manifest fit count incorrect')
  assert_equal(manifest['execution_script_sha256']==EXPECTED_SCRIPT,'manifest source mismatch')
  assert_equal(manifest['dataset']['zip_sha256']==EXPECTED_DATA,'manifest dataset mismatch')
  assert_equal(manifest['certification_summary']['n_issues']==0,'manifest issues present')

  prep_cache={};split_cache={};rows=[];max_kkt_delta=0.;max_obj_delta=0.;max_auroc_delta=0.;max_main_kkt=0.
  for fid,rec in sorted(checkpoints.items()):
   parsed=parse_ckpt(fid)
   assert_equal(parsed is not None,'Unknown fit: '+fid)
   if parsed is None:continue
   typ,c,part,pc,pr=parsed
   if typ=='loco_inner':
    tr=(coh!=c)&(coh!=part);va=coh==part; C,r=pc,pr
   elif typ=='loco_final':
    tr=coh!=c;va=coh==c; C,r=float(rec['C']),float(rec['l1_ratio'])
   elif typ=='indep_final':
    tr=coh==c;va=None;C,r=float(rec['C']),float(rec['l1_ratio'])
   else:
    cmask=coh==c;cidx=np.flatnonzero(cmask)
    if c not in split_cache:
     yy=y_all[cidx]
     split_cache[c]=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=402).split(np.zeros(len(yy)),yy))
    tridx,vaidx=split_cache[c][part-1]
    tr=np.zeros(len(coh),bool);tr[cidx[tridx]]=True
    va=np.zeros(len(coh),bool);va[cidx[vaidx]]=True;C,r=pc,pr
   key=(typ,c,part)
   if key not in prep_cache:prep_cache[key]=get_prep(counts,tr)
   X,idx,mu,sd=prep_cache[key]
   gnames=[genes[i] for i in idx]
   assert_equal(rec['status']=='completed' and rec['numerical_gate_passed'],f'{fid} status/gate failed')
   assert_equal(rec['backend']=='r-glmnet',f'{fid} wrong backend')
   assert_equal(rec['script_sha256']==EXPECTED_SCRIPT,f'{fid} source mismatch')
   assert_equal(rec['dataset_zip_sha256']==EXPECTED_DATA,f'{fid} data mismatch')
   assert_equal(rec['config_fingerprint']==manifest['config_fingerprint'],f'{fid} config mismatch')
   assert_equal(rec['train_ids_sha256']==sha_list(ids[tr]),f'{fid} train IDs hash mismatch')
   assert_equal(rec['val_ids_sha256']==(sha_list(ids[va]) if va is not None else 'none'),f'{fid} val IDs hash mismatch')
   assert_equal(rec['elig_genes_sha256']==sha_list(gnames),f'{fid} eligible genes hash mismatch')
   assert_equal(rec['n_train']==int(sum(tr)) and rec['eligible_genes']==len(idx),f'{fid} shape mismatch')
   assert_equal(rec['C']==C and rec['l1_ratio']==r,f'{fid} C/r mismatch')
   lam=1/(C*int(sum(tr)))
   assert_equal(abs(rec['lambda_fitted']-lam)<1e-11*max(1,abs(lam)),f'{fid} lambda mismatch')
   assert_equal(abs(rec['lambda_target']-lam)<1e-11*max(1,abs(lam)),f'{fid} target lambda mismatch')
   is_final=(typ in ('loco_final','indep_final'))
   if is_final:
    coef_path=ROOT/'full_coefficients'/f'{fid}_all_coefficients.csv'
    pan_path=ROOT/'panels'/f'{fid}_selected_genes.csv'
    coef=pd.read_csv(coef_path);pan=pd.read_csv(pan_path)
    assert_equal(coef.gene.tolist()==gnames,f'{fid} saved final gene order mismatch')
    beta=coef.coef.to_numpy(np.float64)
    exp_pan=coef[coef.coef.abs()>1e-8]
    assert_equal(exp_pan.reset_index(drop=True).equals(pan.reset_index(drop=True)),f'{fid} panel mismatch')
    assert_equal(sha_bytes(coef_path.read_bytes())==rec['full_coef_sha256'],f'{fid} full coef SHA mismatch')
    assert_equal(sha_bytes(pan_path.read_bytes())==rec['panel_sha256'],f'{fid} panel SHA mismatch')
   else:
    compact=rec['compact_nonzero_coefs']
    assert_equal(isinstance(compact,dict),f'{fid} absent compact betas')
    beta=np.array([float(compact.get(g,0.0)) for g in gnames])
    assert_equal(len(compact)==rec['n_nonzero_exact'],f'{fid} compact count mismatch')
    assert_equal(set(compact).issubset(set(gnames)),f'{fid} compact unknown genes')
   assert_equal(int((abs(beta)>1e-8).sum())==rec['n_gt_1e-8'],f'{fid} signature count mismatch')
   own=independent_kkt(X,y_all[tr],beta,float(rec['intercept']),C,r)
   delta=max(abs(own[k]-float(rec[k])) for k in own)
   max_kkt_delta=max(max_kkt_delta,max(abs(own[k]-float(rec[k])) for k in ['max_kkt_residual','kkt_active','kkt_inactive','kkt_intercept']))
   max_obj_delta=max(max_obj_delta,max(abs(own[k]-float(rec[k])) for k in ['common_objective','data_loss','l1_penalty','l2_penalty']))
   assert_equal(delta<1e-7,f'{fid} independent objective/KKT mismatch: {delta}')
   gate=1e-6 if is_final else 1e-4
   assert_equal(own['max_kkt_residual']<gate,f'{fid} KKT gate fails independently')
   max_main_kkt=max(max_main_kkt,own['max_kkt_residual'])
   if va is not None:
    Xval=get_val(counts,va,idx,mu,sd)
    valpred=expit(Xval@beta+float(rec['intercept']))
    va_auc=float(roc_auc_score(y_all[va],valpred))
    ad=abs(va_auc-float(rec['auroc']))
    max_auroc_delta=max(max_auroc_delta,ad)
    assert_equal(ad<1e-10,f'{fid} independent validation AUROC mismatch {ad}')
   rows.append({'fit_id':fid,'role':typ,'cohort':c,'part':part if part is not None else '', 'n_train':int(sum(tr)),'C':C,'l1_ratio':r,'n_nonzero':int(sum(beta!=0)),'kkt_archived':float(rec['max_kkt_residual']),'kkt_independent':own['max_kkt_residual'],'objective_archived':float(rec['common_objective']),'objective_independent':own['common_objective'],'validation_auc_archived':float(rec['auroc']),'validation_auc_independent':va_auc if va is not None else None,'polished':bool(rec['polished']),'fallback':bool(rec['fallback_used'])})
  d=pd.DataFrame(rows)
  d.to_csv(ROOT/'CHATGPT_INDEPENDENT_ALL200_RAW_KKT.csv',index=False)
  summary['fit_groups']={g:{'n':len(t),'max_kkt_independently':float(t.kkt_independent.max()),'polished':int(t.polished.sum()),'fallback':int(t.fallback.sum())} for g,t in d.groupby('role')}
  summary['numerical']={'max_kkt_diff':max_kkt_delta,'max_objective_component_diff':max_obj_delta,'max_auc_diff':max_auroc_delta,'max_real_raw_kkt':max_main_kkt,'n_fits_checked':len(d)}

  # Check actual tuning winners and 96/96 rows per group against archived scores.
  locot=pd.read_csv(ROOT/'loco_inner_tuning_scores.csv');indpt=pd.read_csv(ROOT/'indep_cv_tuning_scores.csv')
  assert_equal(len(locot)==96 and len(indpt)==96,'96/96 CV score tables wrong lengths')
  outer=pd.read_csv(ROOT/'loco_outer_enet_summary.csv'); independent=pd.read_csv(ROOT/'indep_enet_summary.csv')
  for cohort in COHORTS:
   for group,bestrow in [(locot[locot.outer_heldout==cohort],outer[outer.outer_heldout==cohort].iloc[0]), (indpt[indpt.cohort==cohort],independent[independent.cohort==cohort].iloc[0])]:
    agg=group.groupby(['C','l1_ratio'],as_index=False).auroc.mean().sort_values(['auroc','C','l1_ratio'],ascending=[False,True,False])
    best=agg.iloc[0]; assert_equal((float(best.C),float(best.l1_ratio))==(float(bestrow.best_C),float(bestrow.best_l1_ratio)),f'{cohort} tuning winner mismatch')
    expected_mean='inner_mean_auroc' if 'outer_heldout' in bestrow.index else 'cv_mean_auroc'
    assert_equal(abs(float(best.auroc)-float(bestrow[expected_mean]))<1e-12, f'{cohort} tuning mean mismatch')
  # Recompute Jaccards and 4fold exactly
  for prefix in ['loco','indep']:
   panels={c:set(pd.read_csv(ROOT/'panels'/f'{prefix}_{c}_corrected_enet_selected_genes.csv').gene.astype(str)) for c in COHORTS}
   pairs=pd.read_csv(ROOT/f'{prefix}_pairwise_jaccard.csv')
   for _,row in pairs.iterrows():
    a,b=row.pair.split('-');a0,b0=panels[a],panels[b];ninter=len(a0&b0);nunion=len(a0|b0)
    assert_equal((ninter,nunion)==(int(row.intersection),int(row.union)),f'{prefix} {row.pair} counts wrong')
    assert_equal(abs((ninter/nunion if nunion else 0.)-float(row.jaccard))<1e-12,f'{prefix} {row.pair} Jaccard wrong')
   summary[prefix+'_overlap']={'mean_jaccard':float(pairs.jaccard.mean()),'four_way':len(set.intersection(*panels.values()))}

  # Recompute all Stage22 predictions independently from final panels + source data
  pred=pd.read_csv(ROOT/'stage22_corrected_enet_predictions_PRIVATE.csv',dtype={'ID':str})
  metrics=pd.read_csv(ROOT/'stage22_corrected_enet_cohort_metrics.csv')
  macro=pd.read_csv(ROOT/'stage22_corrected_enet_macro_summary.csv').iloc[0]
  assert_equal(len(pred)==689 and pred.ID.is_unique,'Stage22 predictions not 689 unique IDs')
  summary['stage22']={}
  max_prob_delta=0;max_metric_delta=0;max_ci_delta=0;d3_metrics={}
  for i,c in enumerate(COHORTS):
   pan=pd.read_csv(ROOT/'panels'/f'loco_{c}_corrected_enet_selected_genes.csv')
   sel=[genes.index(g) for g in pan.gene.astype(str)]
   tr=coh!=c;te=coh==c
   cltr=counts[tr][:,sel];clte=counts[te][:,sel]
   Xtr=np.log2(1+1e6*cltr/counts[tr].sum(1)[:,None]);Xte=np.log2(1+1e6*clte/counts[te].sum(1)[:,None])
   mu=Xtr.mean(0);sd=Xtr.std(0,ddof=0);sd=np.where(sd==0,1,sd)
   Xtr=(Xtr-mu)/sd;Xte=(Xte-mu)/sd
   clf=LogisticRegression(C=1.0,penalty='l2',solver='liblinear',class_weight='balanced',random_state=123,max_iter=5000)
   clf.fit(Xtr,y_all[tr]);pred0=clf.predict_proba(Xte)[:,1];pr=(pred0>=.5).astype(int)
   dfp=pred.loc[pred.cohort==c]
   assert_equal(dfp.ID.tolist()==ids[te].tolist(),f'{c} heldout IDs/order mismatch')
   assert_equal(np.array_equal(dfp.y_true.to_numpy(),y_all[te]),f'{c} heldout labels mismatch')
   pdiff=float(abs(dfp.prob_CD.to_numpy()-pred0).max())
   max_prob_delta=max(max_prob_delta,pdiff)
   assert_equal(pdiff<.0005,f'{c} refit predictions differ too much: max {pdiff}')
   assert_equal(np.array_equal(dfp.pred_CD.to_numpy(),pr),f'{c} predicted classes differ')
   mc=metrics.loc[metrics.cohort==c].iloc[0]
   y= y_all[te]
   scores={'auroc':float(roc_auc_score(y,dfp.prob_CD)), 'auprc':float(average_precision_score(y,dfp.prob_CD)), 'bacc':float(balanced_accuracy_score(y,dfp.pred_CD)), 'sensitivity':float(np.mean(dfp.pred_CD.to_numpy()[y==1]==1)), 'specificity':float(np.mean(dfp.pred_CD.to_numpy()[y==0]==0)), 'precision':float(precision_score(y,dfp.pred_CD,zero_division=0)), 'f1':float(f1_score(y,dfp.pred_CD,zero_division=0)), 'brier':float(brier_score_loss(y,dfp.prob_CD))}
   cm=confusion_matrix(y,dfp.pred_CD,labels=[0,1]);scores.update({'tn':int(cm[0,0]),'fp':int(cm[0,1]),'fn':int(cm[1,0]),'tp':int(cm[1,1])})
   metric_delta=max(abs(float(mc[k])-v) for k,v in scores.items())
   max_metric_delta=max(max_metric_delta,metric_delta)
   assert_equal(metric_delta<1e-12,f'{c} metric mismatch')
   cil=stable_boot_ci(y,dfp.prob_CD.to_numpy(),roc_auc_score,seed=123+i)
   cib=stable_boot_ci(y,dfp.pred_CD.to_numpy(),bacc_func,seed=123+100+i)
   ci_d=max(abs(float(mc[k])-v) for k,v in [('auroc_ci_low',cil[0]),('auroc_ci_high',cil[1]),('bacc_ci_low',cib[0]),('bacc_ci_high',cib[1])])
   max_ci_delta=max(max_ci_delta,ci_d)
   assert_equal(ci_d<1e-12,f'{c} bootstrap CI mismatch')
   summary['stage22'][c]={'n':int(sum(te)),'panel_size':len(sel),'auroc':scores['auroc'],'bacc':scores['bacc'],'spec':scores['specificity'],'max_refit_prob_delta':pdiff,'bootstrap_CI_diff':ci_d}
   if c=='d3':
    excl=(meta.loc[te,'qc'].astype(str).str.upper()=='EXCLUDE').to_numpy()
    d3_ids=meta.loc[te,'ID'].to_numpy()[excl]
    assert_equal(excl.sum()==7,'D3 should have 7 EXCLUDE flags')
    keep=~excl;y69=y[keep]; prob69=dfp.prob_CD.to_numpy()[keep];pred69=dfp.pred_CD.to_numpy()[keep]
    d3_metrics={'excluded_ids':[str(v) for v in d3_ids], 'n_full':len(y),'n_excluded':int(excl.sum()),'n_remainder':len(y69),'class_counts_remaining':{'CD':int(sum(y69==1)),'Not_IBD':int(sum(y69==0))},'auroc_n76':scores['auroc'],'auroc_n69':float(roc_auc_score(y69,prob69)),'bacc_n76':scores['bacc'],'bacc_n69':float(balanced_accuracy_score(y69,pred69)),'auprc_n69':float(average_precision_score(y69,prob69)),'brier_n69':float(brier_score_loss(y69,prob69)),'f1_n69':float(f1_score(y69,pred69)),'sensitivity_n69':float(np.mean(pred69[y69==1]==1)),'specificity_n69':float(np.mean(pred69[y69==0]==0))}
  summary['prediction_reconstruction']={'max_probability_abs_diff':max_prob_delta,'max_metric_abs_diff':max_metric_delta,'max_CI_abs_diff':max_ci_delta}
  summary['d3_qc_test_only']=d3_metrics
  summary['macro']={}
  for col,metric in [('mean_auroc','auroc'),('mean_auprc','auprc'),('mean_bacc','bacc'),('mean_f1','f1'),('mean_brier','brier')]:
   own=float(metrics[metric].mean());summary['macro'][col]=own;assert_equal(abs(own-float(macro[col]))<1e-13,f'Macro {col} mismatch')
  # Independent CV split file metadata check
  s=pd.read_csv(ROOT/'indep_cv_splits_seed402.csv',dtype={'ID':str});assert_equal(len(s)==3*689,'Split role count should be 3 times 689')
  for c in COHORTS:
   cids=ids[coh==c]; cy=y_all[coh==c]
   for fold,(tri,vali) in enumerate(StratifiedKFold(3,shuffle=True,random_state=402).split(np.zeros(len(cids)),cy),1):
    for role,expected in [('train',cids[tri]),('val',cids[vali])]:
     obs=s.loc[(s.cohort==c)&(s.fold==fold)&(s.role==role),'ID'].to_numpy()
     assert_equal(obs.tolist()==expected.tolist(),f'Seed 402 {c}/{fold}/{role} membership mismatch')

  summary['errors']=errors
  summary['passed']=not errors
  (ROOT/'CHATGPT_INDEPENDENT_FULL_AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
  (ROOT/'CHATGPT_INDEPENDENT_D3_QC_69.json').write_text(json.dumps(d3_metrics,indent=2))
  print(json.dumps({'passed':summary['passed'],'error_count':len(errors),'first_errors':errors[:10],'hashes':summary['hashes'],'groups':summary['fit_groups'],'numeric':summary['numerical'],'prediction_reconstruction':summary['prediction_reconstruction'],'d3_qc':summary['d3_qc_test_only'],'macro':summary['macro'],'overlap':{k:v for k,v in summary.items() if k.endswith('_overlap')}},indent=2))

if __name__=='__main__':main()

import os,sys,glob,json,hashlib,shutil,tempfile,importlib,copy
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score,average_precision_score,balanced_accuracy_score,recall_score,precision_score,f1_score,brier_score_loss,confusion_matrix)
P=Path('/mnt/data/phase2_c1_production_audit')
SRC=Path('/mnt/data/phase2_v3_audit/C1/phase2_baselines_v3.py')
ZIP=Path('/mnt/data/ibd_dataset_sample_aligned.zip')
sys.path.insert(0,str(SRC.parent))
import phase2_baselines_v3 as mod
errors=[]
def ck(cond,what):
 if not cond: errors.append(what)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

manifest=json.loads((P/'manifest.json').read_text())
meta=json.loads((P/'run_metadata.json').read_text())
ck(manifest['status']=='CERTIFIED_COMPLETE','not certified')
ck(meta['execution_script_sha256']==sha(SRC),'executable hash mismatch')
ck(meta['dataset_sha256']==sha(ZIP),'dataset ZIP hash mismatch')
ck(manifest['config_fingerprint']==mod.build_config_fingerprint(100,2000,True,False),'config fingerprint mismatch')

sha_lines=[x.split('  ',1) for x in (P/'SHA256SUMS.txt').read_text().splitlines()]
sha_missing=[]; sha_mismatch=[]
for digest,name in sha_lines:
 f=P/name
 if not f.is_file():sha_missing.append(name)
 elif sha(f)!=digest:sha_mismatch.append(name)
ck(not sha_missing and not sha_mismatch,'missing or mismatched 218 archived SHA files')

patient_df,counts,genes,prov=mod.load_canonical_dataset(ZIP,True)
countscpm=counts/counts.sum(axis=1,keepdims=True)*1e6
logX=np.log2(1+countscpm)
name2idx={g:i for i,g in enumerate(genes)}
universe_checks=[]
for c in mod.COHORTS:
 for scope in ('loco','indep'):
  selection=(patient_df.cohort!=c).to_numpy() if scope=='loco' else (patient_df.cohort==c).to_numpy()
  expected=[genes[j] for j in np.flatnonzero((countscpm[selection]>1.0).mean(axis=0)>=.10)]
  present=pd.read_csv(P/'eligible_universes'/f'{scope}_{c}_eligible_genes.csv').gene.tolist()
  ck(expected==present,f'eligible gene universe mismatch {scope}/{c}')
  universe_checks.append({'cohort':c,'scope':scope,'n_genes':len(present),'match':expected==present})

baseline=pd.read_csv(P/'baseline_all_gene_and_de_metrics.csv')
summary=pd.read_csv(P/'baseline_random_panel_summary.csv')
draws=pd.read_csv(P/'baseline_random_panel_draws.csv')
ck(len(baseline)==48,'baseline row count')
ck(len(summary)==20,'summary row count')
ck(len(draws)==2000,'random draw row count')
ck(baseline['stage22_converged'].eq(True).all(),'baseline unconverged flag')
ck(draws['stage22_converged'].eq(True).all(),'random unconverged flag')

maxdiff={k:0. for k in ['auroc','auprc','bacc','f1','brier','sensitivity','specificity','precision','tp','tn','fp','fn','pred_prob']}
rows=[]; grid=np.array([10,25,50,100,200]);numreplayed=0
for c in mod.COHORTS:
 te_mask=(patient_df.cohort==c).to_numpy();tr_mask=~te_mask
 train_l=logX[tr_mask];test_l=logX[te_mask]
 y_train=patient_df.y[tr_mask].to_numpy();y_test=patient_df.y[te_mask].to_numpy()
 for _,r in baseline[baseline.cohort==c].iterrows():
  b=str(r.baseline);K=int(r.K);tid=f'{c}__{b}' if b in ('all_9983','train_eligible') else f'{c}__{b}_K{K}'
  pred=pd.read_csv(P/'predictions_PRIVATE'/f'pred_{tid}_PRIVATE.csv')
  ck(pred.ID.tolist()==patient_df.loc[te_mask,'ID'].astype(str).tolist(),f'ID mismatch {tid}')
  ck(np.array_equal(pred.y_true.to_numpy(),y_test),f'labels mismatch {tid}')
  score=pred.prob_CD.to_numpy(float);pp=pred.pred_CD.to_numpy(int)
  ck(np.array_equal(pp,(score>=.5).astype(int)),f'pred threshold mismatch {tid}')
  cm=confusion_matrix(y_test,pp,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
  actual={'auroc':roc_auc_score(y_test,score),'auprc':average_precision_score(y_test,score),'bacc':balanced_accuracy_score(y_test,pp),'sensitivity':recall_score(y_test,pp),'specificity':tn/(tn+fp),'precision':precision_score(y_test,pp,zero_division=0),'f1':f1_score(y_test,pp),'brier':brier_score_loss(y_test,score),'tp':tp,'tn':tn,'fp':fp,'fn':fn}
  for key in actual:
   d=abs(float(actual[key])-float(r[key]));maxdiff[key]=max(maxdiff[key],d)
   ck(d<1e-10,f'raw prediction metric mismatch {tid}/{key} delta={d}')
  if b=='all_9983':col_idx=np.arange(len(genes))
  elif b=='train_eligible':
   u=pd.read_csv(P/'eligible_universes'/f'loco_{c}_eligible_genes.csv').gene
   col_idx=np.array([name2idx[x] for x in u])
  else:
   ranked=pd.read_csv(P/f'de_ranking_{c}_{b}.csv')
   panel=pd.read_csv(P/f'panel_{tid}.csv')
   ck(len(ranked)==mod.EXPECTED_LOCO_ELIGIBLE[c],f'de rank count mismatch {tid}')
   ck(panel.gene.tolist()==ranked.gene.head(K).tolist(),f'panel not ranking top K {tid}')
   col_idx=np.array([name2idx[x] for x in panel.gene]);
  ck(len(col_idx)==K,f'panel size mismatch {tid}')
  # Independent classifier replay on the raw canonical matrix and archived panel.
  scaler=StandardScaler();X=scaler.fit_transform(train_l[:,col_idx]);Xt=scaler.transform(test_l[:,col_idx])
  clf=LogisticRegression(solver='liblinear',C=1.,class_weight='balanced',max_iter=5000,random_state=123)
  clf.fit(X,y_train)
  pp2=clf.predict_proba(Xt)[:,1];d=float(abs(pp2-score).max())
  maxdiff['pred_prob']=max(maxdiff['pred_prob'],d)
  ck(d<1e-7,f'full model replay prob differs {tid} delta={d}')
  numreplayed+=1
  rows.append({'task':tid,'cohort':c,'baseline':b,'K':K,'auroc':float(actual['auroc']),'bacc':float(actual['bacc']),'max_probability_diff_vs_replayed_classifier':d})

# All 2000 random-draw receipts are checked against the exact eligible gene universe.
rand_receipt_count=0
for c in mod.COHORTS:
 eligible=set(pd.read_csv(P/'eligible_universes'/f'loco_{c}_eligible_genes.csv').gene)
 ks=sorted(set(int(v['sizes'][c]) for v in mod.AUDITED_COMPARATORS.values()))
 for K in ks:
  tid=f'{c}__random_K{K}_B100'; receipts=json.loads((P/'random_draw_gene_lists'/f'random_genes_{tid}.json').read_text())
  sub=draws[(draws.cohort==c)&(draws.K==K)].sort_values('draw_idx');ck(len(sub)==100,f'random 100 count {tid}')
  ck(len(receipts)==100,f'100 genes receipts {tid}')
  for _,r in sub.iterrows():
   idx=int(r.draw_idx);rec=receipts[str(idx)]
   gene_list=rec['genes']; expected_seed=mod.RANDOM_BASE_SEED+mod.COHORTS.index(c)*100000+K*1000+idx
   ck(len(gene_list)==K and len(set(gene_list))==K and set(gene_list)<=eligible,f'random invalid genes {tid} idx={idx}')
   ck(int(r.seed)==expected_seed==rec['seed'],f'seed mismatch {tid} idx={idx}')
   genehash=hashlib.sha256('\n'.join(gene_list).encode('utf8')).hexdigest()
   ck(genehash==r.panel_genes_sha256==rec['genes_sha256'],f'gene list SHA mismatch {tid} idx={idx}')
   rand_receipt_count+=1

# All 20 random-null tables reproduced independently from 2000 draws.
max_null_diff=0.
for _,r in summary.iterrows():
 c=str(r.cohort);K=int(r.K);ss=draws[(draws.cohort==c)&(draws.K==K)]
 ck(len(ss)==100,f'missing draw group {c}/{K}')
 for label,observed in [('auroc',float(r.observed_auroc)),('bacc',float(r.observed_bacc))]:
  z=ss[label].to_numpy();d={f'null_{label}_mean':z.mean(),f'null_{label}_sd':z.std(ddof=1),f'null_{label}_median':np.percentile(z,50),f'null_{label}_iqr':np.percentile(z,75)-np.percentile(z,25),f'null_{label}_p05':np.percentile(z,5),f'null_{label}_p95':np.percentile(z,95),f'emp_p_{label}':(1+(z>=observed).sum())/(len(z)+1)}
  for name,value in d.items():
   delta=abs(float(r[name])-value);max_null_diff=max(max_null_diff,delta)
   ck(delta<1e-10,f'random summary mismatch {c}/{K}/{name} delta={delta}')

# Check production checkpoint integrity using production parser on local immutable copy.
ckpt_count=0
for p in (P/'checkpoints').glob('*.json'):
 tid=p.stem;companions={'draws','genes'} if '__random_K' in tid else ({'pred'} if '__all_9983' in tid or '__train_eligible' in tid else {'pred','panel'})
 out=mod.load_validated_checkpoint(p,sha(SRC),sha(ZIP),meta['config_fingerprint'],tid,companions,P)
 ck(out is not None,f'invalid checkpoint {tid}')
 ckpt_count+=1
ck(ckpt_count==68,f'checkpoint count {ckpt_count}')

report={'manifest_status':manifest['status'],'file_hashes':len(sha_lines),'sha_missing':sha_missing,'sha_mismatches':sha_mismatch,'all_hashes_match':not sha_missing and not sha_mismatch,'source_code_sha256':sha(SRC),'dataset_sha256':sha(ZIP),'n_independently_replayed_classifiers':numreplayed,'max_metric_difference':maxdiff,'max_probability_replay_difference':maxdiff['pred_prob'],'n_random_receipts_validated':rand_receipt_count,'n_random_summaries_validated':len(summary),'max_random_summary_difference':max_null_diff,'n_checkpoints_validated':ckpt_count,'universe_checks':universe_checks,'n_errors':len(errors),'errors':errors[:100]}
(P/'INDEPENDENT_C1_PRODUCTION_AUDIT.json').write_text(json.dumps(report,indent=2))
pd.DataFrame(rows).to_csv(P/'INDEPENDENT_C1_48_BASELINE_REPLAY.csv',index=False)
print(json.dumps({k:v for k,v in report.items() if k not in ['universe_checks','errors']},indent=2))
if errors:print('ERRORS:\n'+'\n'.join(errors[:30]))

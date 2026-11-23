import os, json, hashlib, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score, f1_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy import stats
B=Path('/mnt/data/c2c3_v3_production_audit')
C1=Path('/mnt/data/phase2_c1_production_audit')
C1SCRIPT=Path('/mnt/data/phase2_v3_audit/C1/phase2_baselines_v3.py')
ZIP=Path('/mnt/data/phase2_v3_audit/canonical/ibd_dataset_sample_aligned.zip')
SYS={}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def chk(test,msg):
 if not test: raise AssertionError(msg)
SYS['sha_list_count']=0
checks=[line.strip().split('  ',1) for line in (B/'SHA256SUMS.txt').read_text().splitlines() if line.strip()]
for dig,path in checks:
 p=B/path;chk(p.is_file(),f'missing file {path}');chk(sha(p)==dig,f'SHA mismatched {path}')
SYS['sha_list_count']=len(checks)
scriptsha=sha(B/'phase2_overlap_and_size_v3.py');cfgsha=sha(B/'overlap_manifest_config_v3.json');manifest=json.loads((B/'manifest.json').read_text());chk(manifest['status']=='CERTIFIED_COMPLETE','status not certified');chk(manifest['execution_script_sha256']==scriptsha,'script hash');chk(manifest['manifest_sha256']==cfgsha,'manifest hash');chk(manifest['dataset_sha256']==sha(ZIP),'dataset hash');
print('hashes',len(checks),'script',scriptsha,'manifest',cfgsha, flush=True)
# all output model checkpoint integrity, patient metrics
sys.path.insert(0,str(B))
import phase2_overlap_and_size_v3 as source
meta,counts,genes,_ = None,None,None,None
sys.path.insert(0,str(C1SCRIPT.parent))
import phase2_baselines_v3 as c1
meta,counts,genes,provenance=c1.load_canonical_dataset(ZIP,verify_hashes=True)
assert provenance['zip_sha256']==manifest['dataset_sha256']
_,log2=c1.compute_cpm_and_log2(counts)
idx={str(g):i for i,g in enumerate(genes)}
met=pd.read_csv(B/'phase2_fixed_k_cohort_metrics.csv');macro=pd.read_csv(B/'phase2_fixed_k_macro_summary.csv');pair=pd.read_csv(B/'phase2_pairwise_overlap_null.csv');four=pd.read_csv(B/'phase2_fourway_overlap_null.csv');config=json.loads((B/'overlap_manifest_config_v3.json').read_text());prov=json.loads((B/'input_provenance.json').read_text());
chk(len(met)==100 and len(macro)==25 and len(pair)==42 and len(four)==7,'inventory length');chk(len(config['methods'])==7,'method count');
chk(prov['schema_version']=='c2_input_provenance_v3' and prov['manifest_sha256']==cfgsha,'provenance schema');
# independent observed pairwise and 4way + expectations from pinned input provenance (cross-check canonical original local files)
preflight_cfg=json.load(open(C1/'C2_REAL_INPUT_PREFLIGHT_LOCAL_PATHS.json'))
sel_by_name={e['name']:e for e in preflight_cfg['methods']}
staterrs=[];allp=0;allf=0;method_u={}
for e in config['methods']:
 name=e['name']; e_local=sel_by_name[name]; sets={};usets={}
 for cohort in ('d1','d2','d3','d4'):
  p_spec=e_local['panel_sources'][cohort]; gene_list, ranked_genes, src, source_sha=source._extract_panel_and_ranking(p_spec,cohort);chk(source_sha==p_spec['expected_sha256'],f'panel sha mismatch {name} {cohort}');s=set(gene_list);sets[cohort]=s
  u_file=Path(e_local['universe_csvs'][cohort]);u=pd.read_csv(u_file);u_col='gene' if 'gene' in u.columns else u.columns[0];uu=set(u[u_col].astype(str));usets[cohort]=uu
  chk(s<=uu,f'not eligible {name} {cohort}');
  pi=prov['methods'][name]['cohorts'][cohort];chk(source_sha==pi['panel_sha256'] and sha(u_file)==pi['universe_sha256'],f'provenance mismatch {name} {cohort}')
 method_u[name]=usets
 for _,r in pair[pair.method==name].iterrows():
  a,b=r['pair'].split('-');N=len(usets[a]&usets[b]);obs=len(sets[a]&sets[b]);ka=len(sets[a]);kb=len(sets[b]);ex=ka*kb*N/(len(usets[a])*len(usets[b]));
  chk(r['u_inter']==N and r['obs_intersection']==obs and np.isclose(r['exact_expected_intersection'],ex,atol=1e-12),f'pair numerical mismatch {name}:{a}{b}');
  chk(np.isclose(r['obs_jaccard'],obs/(ka+kb-obs),atol=1e-13),f'pair jaccard {name} {a}{b}');allp+=1
 r=four[four.method==name].iloc[0];z=set.intersection(*(usets[c] for c in ('d1','d2','d3','d4')));obs=len(set.intersection(*(sets[c] for c in ('d1','d2','d3','d4'))));ex=len(z)*np.prod([len(sets[c])/len(usets[c]) for c in ('d1','d2','d3','d4')]);
 chk(r['obs_fourway_intersection']==obs and r['u_fourway_common']==len(z) and np.isclose(r['exact_expected_fourway'],ex,atol=1e-12),f'fourway mismatch {name}');allf+=1
SYS['exact_pair_reconstructions']=allp;SYS['exact_fourway_reconstructions']=allf
print('overlap exact',allp,allf,flush=True)
# Monte Carlo deterministic full replication using independent representation of same algorithm and seed
# Re-run original sampling with verified actual U cardinalities; independent formulas checked above.
for j,e in enumerate(config['methods']):
 name=e['name'];dfp=pair[pair.method==name]
 for i,r in dfp.iterrows():
  pair_idx=['d1-d2','d1-d3','d1-d4','d2-d3','d2-d4','d3-d4'].index(r['pair']);a,b=r['pair'].split('-')
  obs=source.monte_carlo_pairwise_null(method_u[name][a],method_u[name][b],int(r['k_a']),int(r['k_b']),int(r['obs_intersection']),n_mc=10000,seed=20261122+j*100+pair_idx)
  for c in ['mc_mean_intersection','mc_inter_ci_low','mc_inter_ci_high','mc_mean_jaccard','mc_jaccard_ci_low','mc_jaccard_ci_high','mc_mean_overlap_coeff','mc_emp_p_ge_obs']:
   chk(np.isclose(float(r[c]),obs[c],atol=1e-12),f'mc pair mismatch {name} {r["pair"]} {c}')
 r=four[four.method==name].iloc[0];obs=source.monte_carlo_fourway_null([method_u[name][c] for c in ('d1','d2','d3','d4')],[int(r['k_'+c]) for c in ('d1','d2','d3','d4')],int(r['obs_fourway_intersection']),n_mc=10000,seed=20261122+j*100+99)
 for c in ['mc_mean_fourway','mc_fourway_ci_low','mc_fourway_ci_high','mc_p_ge_1','mc_emp_p_ge_obs']:
  chk(np.isclose(float(r[c]),obs[c],atol=1e-12),f'mc fourway mismatch {name} {c}')
SYS['mc_pair_replayed']=42;SYS['mc_fourway_replayed']=7
print('Monte Carlo replay',flush=True)
# check patient ID/label and independent metrics, classifier replay stage22 from raw canonical matrix
records=[];probmax=0;metmax=0;bootmax=0; ckptcount=0;models=0;status_count={};panels_ranks=0
for _,r in met.iterrows():
 m,c,K=str(r['method']),str(r['cohort']),int(r['K']);status=str(r['status']);status_count[status]=status_count.get(status,0)+1
 if not status.startswith('VALID_'): continue
 safe=''.join(ch if ch.isalnum() else '_' for ch in m).strip('_');tk=f'fixedK__{safe}__{c}__K{K}'; ckpt=B/'checkpoints'/f'{tk}.json';
 chk(ckpt.is_file(),f'missing ckpt {tk}')
 cr=source.load_validated_c3_checkpoint(ckpt,tk,scriptsha,cfgsha,sha(ZIP),{'pred','panel'},B);chk(cr is not None,'empty checkpoint');ckptcount+=1
 pred=B/'fixed_k_predictions_PRIVATE'/f'pred_{safe}_{c}_K{K}_PRIVATE.csv';panel=B/'fixed_k_panels'/f'panel_{safe}_{c}_K{K}.csv';pr=pd.read_csv(pred);pl=pd.read_csv(panel)
 chk(len(pl)==K and pl.gene.is_unique,'invalid panel');chk(pr.ID.is_unique and len(pr)=={'d1':260,'d2':179,'d3':76,'d4':174}[c],'patient mismatch');chk(all(pr.cohort==c),'cohort mismatch');
 tru=meta.loc[meta.cohort==c,['ID','y']].reset_index(drop=True);chk((pr.ID.to_numpy()==tru.ID.to_numpy()).all() and (pr.y_true.to_numpy()==tru.y.to_numpy()).all(),f'patient-label mismatch {tk}')
 yy=pr.y_true.to_numpy();probs=pr.prob_CD.to_numpy();cl=pr.pred_CD.to_numpy();calc={'auroc':roc_auc_score(yy,probs),'auprc':average_precision_score(yy,probs),'bacc':balanced_accuracy_score(yy,cl),'f1':f1_score(yy,cl),'brier':brier_score_loss(yy,probs)}
 for key,v in calc.items():metmax=max(metmax,abs(v-float(r[key])));chk(np.isclose(v,r[key],atol=1e-10),f'metric wrong {key} {tk}')
 chk((cl==(probs>=.5)).all(),f'threshold mismatch {tk}')
 # fixed-K rank order source (without rerunning gene selection)
 e=sel_by_name[m];p_spec=e['panel_sources'][c];gene_list,ranked_genes,_,_=source._extract_panel_and_ranking(p_spec,c)
 if ranked_genes:
  check_genes=ranked_genes[:K];chk(list(pl.gene.astype(str))==check_genes,f'ranking mismatch {tk}')
  panels_ranks+=1
 # independently refit Stage 22 from canonical matrix
 testmask=(meta.cohort==c).to_numpy();trmask=~testmask;co=np.array([idx[g] for g in pl.gene],dtype=int)
 xtr=log2[trmask][:,co];xte=log2[testmask][:,co];sc=StandardScaler();xtr=sc.fit_transform(xtr);xte=sc.transform(xte)
 clf=LogisticRegression(penalty='l2',C=1.,class_weight='balanced',solver='liblinear',max_iter=5000,random_state=123)
 clf.fit(xtr,meta.loc[trmask,'y'].to_numpy());p2=clf.predict_proba(xte)[:,1];d=float(np.max(abs(p2-probs)));probmax=max(probmax,d);chk(d<1e-8,f'prediction difference {tk} {d}');models+=1
 records.append({'method':m,'cohort':c,'K':K,'max_abs_probability_delta':d,'auroc':calc['auroc'],'bacc':calc['bacc']})
SYS.update({'valid_models':models,'checkpoints_verified':ckptcount,'panel_rankings_verified':panels_ranks,'max_abs_prob_delta':probmax,'max_abs_metric_delta':metmax,'status_counts':status_count})
pd.DataFrame(records).to_csv(B/'INDEPENDENT_44_FIXED_K_REPLAYS.csv',index=False)
# macro mean only over valid records; if none, NaN
for _,r in macro.iterrows():
 df=met[(met.method==r['method'])&(met.K==r['K'])&met.status.str.startswith('VALID_')]
 chk(len(df)==r['n_evaluable_folds'],f'macro n {r["method"]} K={r["K"]}')
 if len(df):chk(np.isclose(df.auroc.mean(),r.macro_auroc,atol=1e-12) and np.isclose(df.bacc.mean(),r.macro_bacc,atol=1e-12),f'macro mismatch {r["method"]} K={r["K"]}')
SYS['macro_reproduced']=len(macro)
(B/'INDEPENDENT_C2C3_PRODUCTION_AUDIT.json').write_text(json.dumps(SYS,indent=2,sort_keys=True))
print(json.dumps(SYS,indent=2),flush=True)

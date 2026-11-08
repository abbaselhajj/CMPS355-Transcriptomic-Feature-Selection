"""Independent, read-only CMPS355 final manuscript analysis. Generates aggregate-only reports."""
from pathlib import Path
import pandas as pd,numpy as np,hashlib,json,itertools,zipfile,io,sys
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,average_precision_score,confusion_matrix,f1_score,brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import hypergeom
sys.path.insert(0,'/mnt/data')
import pathway_analysis as pathmod
R=Path('/mnt/data/cmps355_v2_private');OUT=Path('/mnt/data/CMPS355_Final_Evidence');OUT.mkdir(exist_ok=True)
methods=['lasso','elastic_net','stability_selection_cutoff_0p60','StableMate_significant_stable','TENTACLES_V2_nmin6']
labels={'lasso':'LASSO','elastic_net':'Elastic Net','stability_selection_cutoff_0p60':'Stability Selection','StableMate_significant_stable':'StableMate','TENTACLES_V2_nmin6':'TENTACLES V2'}
folds=['d1','d2','d3','d4']
manifest=json.loads((R/'EXPORT_MANIFEST.json').read_text());assert manifest['status']=='READ_ONLY_EXPORT_NMIN6_AND_NMIN5_VALIDATED'
# Verify every SHA-256 in exported archive.
for x in manifest['artifacts']:
 p=R/x['relative_path']; assert p.is_file(),str(p)
 assert p.stat().st_size==x['bytes'],str(p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],str(p)
print('V2 sha files passed:',len(manifest['artifacts']))
old=pd.read_csv('/mnt/data/selected_panel_receipts.csv');assert not old.duplicated(['heldout','method','gene']).any()
panels={(m,f):set(old[(old.method==m)&(old.heldout==f)].gene.astype(str)) for m in methods[:-1] for f in folds}
new={};preds={};metrics={}
with zipfile.ZipFile('/mnt/data/ibd_dataset_sample_aligned.zip') as z:
 meta=pd.concat([pd.read_csv(z.open(f'desc_{f}.csv')).assign(cohort=f) for f in folds],ignore_index=True)
 assert meta.ID.is_unique and len(meta)==689
 m=meta.set_index('ID')
for cutoff in (6,5):
 p=R/manifest['evaluation'][str(cutoff)]['directory']
 assert 'PASS' in (p/'STAGE10_PASS.txt').read_text()
 df=pd.read_csv(p/'selected_genes.csv')
 p_df=pd.read_csv(p/'heldout_predictions.csv',dtype={'ID':str})
 t_df=pd.read_csv(p/'heldout_metrics.csv')
 assert len(p_df)==689 and p_df.ID.is_unique and set(p_df.ID)==set(m.index)
 assert not df.duplicated(['heldout','gene']).any()
 for f in folds:
  this=set(df.loc[df.heldout==f,'gene'].astype(str));assert len(this)>0
  native=R/manifest['canonical_folds'][f]
  assert (native/'STATUS.txt').read_text().startswith('PASS_')
  votes=pd.read_csv(native/f'native_votes_nmin{cutoff}.csv')
  assert not votes.gene.duplicated().any()
  exact=set(votes.loc[votes.votes>=cutoff,'gene'].astype(str))
  assert exact==this,(f,cutoff,'vote mismatch',len(this),len(exact))
  nat=set(pd.read_csv(native/f'consensus_genes_nmin{cutoff}.csv').iloc[:,0].astype(str))
  assert nat==this,(f,cutoff,'native mismatch')
  s=p_df[p_df.heldout==f]; assert len(s)==sum(meta.cohort==f)
  mm=m.loc[s.ID];assert (mm['class'].values==s.class_true.values).all() and (mm.Study.values==s.Study.values).all()
  y=(s.class_true=='CD').astype(int).values;assert (y==s.y_true.values).all() and np.isfinite(s.probability_CD).all()
  prob=s.probability_CD.values
  assert ((prob>=0)&(prob<=1)).all() and ((prob>=0.5).astype(int)==s.y_pred.values).all()
  cm=confusion_matrix(y,s.y_pred.values,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
  row=t_df.loc[t_df.heldout==f].iloc[0]
  for key,val in [('roc_auc',roc_auc_score(y,prob)),('balanced_accuracy',balanced_accuracy_score(y,s.y_pred)),('average_precision',average_precision_score(y,prob)),('brier_score',brier_score_loss(y,prob)),('f1',f1_score(y,s.y_pred))]: assert abs(row[key]-val)<1e-10,(f,cutoff,key,row[key],val)
  assert all(int(row[k])==v for k,v in [('TP',tp),('TN',tn),('FP',fp),('FN',fn)])
  assert row.n_selected_genes==len(this)
  if cutoff==6:panels[('TENTACLES_V2_nmin6',f)]=this
  new[(cutoff,f)]=this;preds[(cutoff,f)]=s;metrics[(cutoff,f)]=row
 assert abs(np.mean([metrics[(cutoff,f)].roc_auc for f in folds])-manifest['evaluation'][str(cutoff)]['mean_auroc'])<1e-12
 print('V2',cutoff,'panels:',[len(new[(cutoff,f)]) for f in folds], 'AUROC:',[round(metrics[(cutoff,f)].roc_auc,4) for f in folds])
# Five-method LOCO primary panels, exact overlap.
summary=[];pairrows=[]
for met in methods:
 ss=[panels[(met,f)] for f in folds];js=[]
 for i,j in itertools.combinations(range(4),2):
  a,b=ss[i],ss[j];k=len(a&b);v=k/len(a|b);js.append(v)
  pairrows.append({'method':met,'fold_a':folds[i],'fold_b':folds[j],'size_a':len(a),'size_b':len(b),'intersection':k,'union':len(a|b),'jaccard':v})
 all4=len(set.intersection(*ss))
 summary.append({'method':met,'method_display':labels[met],'mean_panel_size':np.mean([len(x) for x in ss]),'min_panel_size':min(map(len,ss)),'max_panel_size':max(map(len,ss)),'mean_loco_gene_jaccard':np.mean(js),'genes_all_four':all4,'genes_at_least_two':sum(sum(g in s for s in ss)>=2 for g in set.union(*ss))})
summary=pd.DataFrame(summary); pairs=pd.DataFrame(pairrows)
summary.to_csv(OUT/'five_method_gene_agreement.csv',index=False);pairs.to_csv(OUT/'five_method_jaccard_pairs.csv',index=False)
# Combine previous four controlled predictor panel runs with new V2 metrics.
prior=pd.read_csv('/mnt/data/cohort_metrics.csv')
rows=[]
for f in folds:
 for met in methods:
  if met=='TENTACLES_V2_nmin6':
   r=metrics[(6,f)];rows.append({'heldout':f,'method':met,'auroc':r.roc_auc,'balanced_accuracy':r.balanced_accuracy,'average_precision':r.average_precision,'n_genes':r.n_selected_genes,'n_test':r.n,'sensitivity':r.sensitivity,'specificity':r.specificity})
  else:
   r=prior[(prior.heldout==f)&(prior.method==met)].iloc[0]
   rows.append({'heldout':f,'method':met,'auroc':r.auroc,'balanced_accuracy':r.balanced_accuracy,'average_precision':r.average_precision,'n_genes':r.n_genes,'n_test':r.n_test,'sensitivity':r.sensitivity,'specificity':r.specificity})
perf=pd.DataFrame(rows)
perf.to_csv(OUT/'five_method_controlled_by_cohort.csv',index=False)
macro=perf.groupby('method').agg(mean_auroc=('auroc','mean'),sd_auroc=('auroc','std'),mean_balanced_accuracy=('balanced_accuracy','mean'),sd_balanced_accuracy=('balanced_accuracy','std'),mean_average_precision=('average_precision','mean')).reset_index()
macro.to_csv(OUT/'five_method_controlled_macro.csv',index=False)
assert (macro.set_index('method').loc[methods[:-1],'mean_auroc'].to_numpy()==np.array([0.6890874982834573,0.7118462997504447,0.6284206342912073,0.6590373821437203])).all()
# V2 vote composition, rank ambiguity (multiple equal vote counts)
votecounts=[]
for f in folds:
 df=pd.read_csv(R/manifest['canonical_folds'][f]/'native_votes_nmin6.csv')
 counts=df[df.votes>=6].votes.value_counts().to_dict()
 votecounts.append({'heldout':f,**{f'votes_{v}':counts.get(v,0) for v in [6,7,8,9]}})
pd.DataFrame(votecounts).to_csv(OUT/'tentacles_v2_vote_distribution.csv',index=False)
# Pinned Reactome v98 redo across 5 methods from exact saved backgrounds.
reference=Path('/mnt/data/ReactomePathways.gmt.zip')
assert hashlib.sha256(reference.read_bytes()).hexdigest()=='6056fcd95dc8f09740c8bb64c2765bea1386ce91b9b19cd10bcd45d30451805c'
pathways=pathmod.parse_gmt(reference)
bg=pd.read_csv('/mnt/data/eligible_gene_backgrounds.csv')
backgrounds={f:set(bg[(bg.scope=='loco')&(bg.cohort==f)].gene.astype(str)) for f in folds}
orig=pd.read_csv('/mnt/data/signature_summary.csv')
enrich=[];en_summary=[];epath={};
for met in methods:
 for f in folds:
  selected=panels[(met,f)];B=backgrounds[f]
  assert selected<=B,('outside eligible background',met,f,selected-B)
  e=pathmod.enrichment(selected,B,pathways,scope='loco',cohort=f,method=met)
  good=e[e.significant];epath[(met,f)]=set(good.pathway_id)
  enrich.append(e)
  en_summary.append({'method':met,'heldout':f,'selected_genes':len(selected),'background_size':len(B),'testable_pathways':len(e),'significant_pathways':len(good),'min_q':float(e.q_value.min()),'annotated_selected':len(selected&set().union(*[p['genes']&B for p in pathways.values()]))})
en=pd.DataFrame(en_summary);en.to_csv(OUT/'five_method_reactome_counts.csv',index=False)
pd.concat([x[x.significant] for x in enrich],ignore_index=True).to_csv(OUT/'five_method_reactome_significant.csv',index=False)
for met in ['lasso','elastic_net']:
 for f in folds:
  x=en[(en.method==met)&(en.heldout==f)].iloc[0];o=orig[(orig.scope=='loco')&(orig.method==met)&(orig.cohort==f)].iloc[0]
  assert int(x.selected_genes)==int(o.signature_genes)
  assert int(x.background_size)==int(o.eligible_genes)
  assert int(x.testable_pathways)==int(o.tested_pathways)
  assert int(x.significant_pathways)==int(o.significant_pathways)
  assert abs(x.min_q-o.min_q_value)<1e-12
print('Reactome prior 8 baseline cases EXACT match')
epairs=[]
for met in methods:
 for f,g in itertools.combinations(folds,2):
  A,B=epath[(met,f)],epath[(met,g)];e=len(A|B)
  epairs.append({'method':met,'fold_a':f,'fold_b':g,'count_a':len(A),'count_b':len(B),'intersection':len(A&B),'union':e,'jaccard':len(A&B)/e if e else np.nan})
epair=pd.DataFrame(epairs);epair.to_csv(OUT/'five_method_reactome_pairwise.csv',index=False)
pathmacro=epair.groupby('method').jaccard.agg(['mean','count']).reset_index().rename(columns={'mean':'mean_pathway_jaccard_defined','count':'defined_pairs'});pathmacro.to_csv(OUT/'five_method_reactome_macro.csv',index=False)
# Per-cohort fixed class performance for TENTACLES n5 sensitivity, distinct from primary.
pd.DataFrame([{'heldout':f,'auroc_6votes':metrics[6,f].roc_auc,'auroc_5votes':metrics[5,f].roc_auc,'ba_6votes':metrics[6,f].balanced_accuracy,'ba_5votes':metrics[5,f].balanced_accuracy,'genes_6votes':len(new[6,f]),'genes_5votes':len(new[5,f])} for f in folds]).to_csv(OUT/'tentacles_v2_threshold_sensitivity.csv',index=False)
report={'status':'PASS_INDEPENDENT_AGGREGATE_CROSSCHECK','n_export_hashes_checked':len(manifest['artifacts']),'all_v2_heldout_predictions_verified':689*2,'four_canonical_voter_panels_verified':True,'vote_thresholds_checked':[5,6],'n_model_folds':4,'n_method_folds':20,'dataset_sha256':hashlib.sha256(Path('/mnt/data/ibd_dataset_sample_aligned.zip').read_bytes()).hexdigest(),'five_method_macro':macro.round(6).to_dict('records'),'five_method_gene_overlap':summary.round(6).to_dict('records'),'reactome':en.to_dict('records'),'reactome_validation':'8 of 8 original LASSO/Elastic Net enrichment rows exact match','source_zip_sha256':hashlib.sha256(Path('/mnt/data/CMPS355_TENTACLES_V2_Final_Results_For_Analysis(1).zip').read_bytes()).hexdigest(),'native_rds_unavailable_in_export':True,'native_candidate_warnings_and_D3_D4_recovery_documented':True}
(OUT/'audit_report.json').write_text(json.dumps(report,indent=2))
print('\nFIVE MODEL MACRO:\n',macro.to_string(index=False));print('\nGENE OVERLAP:\n',summary.to_string(index=False));print('\nREACTOME COUNTS:\n',en.pivot(index='method',columns='heldout',values='significant_pathways').to_string());print('\nREACTOME OVERLAP:\n',pathmacro.to_string(index=False)); print('\nV2 GENE JACCARD\n',pairs[pairs.method=='TENTACLES_V2_nmin6'].to_string(index=False));print('SUCCESS')

from pathlib import Path
import numpy as np,pandas as pd,zipfile,json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
r=Path('/mnt/data/cmps355_v2_private');m=json.loads((r/'EXPORT_MANIFEST.json').read_text());zipfile_in='/mnt/data/ibd_dataset_sample_aligned.zip'
with zipfile.ZipFile(zipfile_in) as z:
  headers=[pd.read_csv(z.open(f'count_{f}_aligned.csv'),nrows=0).columns[1:] for f in ['d1','d2','d3','d4']]
  genes=sorted(set.intersection(*map(set,headers)));assert len(genes)==9983
  mats=[];descs=[]
  for f in ['d1','d2','d3','d4']:
    mat=pd.read_csv(z.open(f'count_{f}_aligned.csv'),usecols=['ID']+genes)
    desc=pd.read_csv(z.open(f'desc_{f}.csv'))
    assert (mat.ID==desc.ID).all()
    mats.append(mat);descs.append(desc.assign(cohort=f))
meta=pd.concat(descs,ignore_index=True);x=np.concatenate([q[genes].to_numpy(np.float64) for q in mats]);tot=x.sum(axis=1);assert (tot>0).all();x=np.log2(1+x/tot[:,None]*1e6)
y=(meta['class']=='CD').astype(int).values
z=pd.read_csv(r/m['evaluation']['6']['directory']/'selected_genes.csv')
old=pd.read_csv(r/m['evaluation']['6']['directory']/'heldout_predictions.csv')
pos={g:i for i,g in enumerate(genes)}
for f in ['d1','d2','d3','d4']:
 train=(meta.cohort!=f).values;test=~train
 panel=z[z.heldout==f].gene.tolist();inds=[pos[g] for g in panel]
 a=x[np.ix_(train,inds)];b=x[np.ix_(test,inds)]
 scale=StandardScaler().fit(a);a=scale.transform(a);b=scale.transform(b)
 model=LogisticRegression(penalty='l2',solver='liblinear',C=1,class_weight='balanced',max_iter=5000,random_state=123)
 model.fit(a,y[train]);pred=model.predict_proba(b)[:,1]
 expected=old[old.heldout==f].set_index('ID').loc[meta.loc[test,'ID'],'probability_CD'].to_numpy()
 print(f,'mean abs diff',np.abs(pred-expected).mean(),'max',np.abs(pred-expected).max())
 assert np.max(np.abs(pred-expected))<0.001
print('PASS independent refit all 4 V2 cohorts')

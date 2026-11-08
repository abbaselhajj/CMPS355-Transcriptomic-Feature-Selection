from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root=Path(__file__).resolve().parents[1]
p=root/'verification';o=root/'figures';o.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'savefig.dpi':300,'pdf.fonttype':42})
methods=[('lasso','LASSO','#8ba5bc'),('elastic_net','Elastic Net','#b6bac7'),('stability_selection_cutoff_0p60','Stability Selection','#74a5b7'),('StableMate_significant_stable','StableMate','#a9b9ab'),('TENTACLES_V2_nmin6','TENTACLES V2','#21598c')]
M=pd.read_csv(p/'five_method_controlled_macro.csv').set_index('method')
G=pd.read_csv(p/'five_method_gene_agreement.csv').set_index('method')
fig=plt.figure(figsize=(7.05,2.8))
ax=fig.add_axes([.095,.19,.56,.70]);handles=[]
for key,name,color in methods:
    x=float(G.loc[key,'mean_loco_gene_jaccard']);y=float(M.loc[key,'mean_auroc']);n=float(G.loc[key,'mean_panel_size'])
    ax.scatter([x],[y],s=28+n*3.0,c=color,edgecolor='#1f3344',linewidth=.9,alpha=.92,zorder=3)
    handles.append(Line2D([],[],marker='o',linestyle='None',markerfacecolor=color,markeredgecolor='#1f3344',markersize=8,label=f'{name} (n≈{n:.0f})'))
ax.set_xlim(-.005,.255);ax.set_ylim(.605,.77);ax.set_axisbelow(True)
ax.grid(True,alpha=.22,linewidth=.6,linestyle='--')
ax.set_xlabel('Mean LOCO gene-set Jaccard',fontsize=9)
ax.set_ylabel('Mean held-out AUROC',fontsize=9)
ax.tick_params(labelsize=8)
fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.68,.60),frameon=False,labelspacing=.92,handletextpad=.7,fontsize=8.3)
fig.text(.69,.83,'Selector (mean genes)',fontsize=9,fontweight='semibold')
fig.text(.69,.25,'Circle area ∝ mean gene count',fontsize=7.9,color='#555555')
for ext in ['pdf','png']:
 fig.savefig(o/f'prediction_vs_agreement.{ext}',bbox_inches='tight',pad_inches=.03)
plt.close(fig)

# Wide FDR heatmap, all originally reported significant values preserved.
react=pd.read_csv(p/'five_method_reactome_significant.csv')
react=react[react.method=='TENTACLES_V2_nmin6']
cohorts=['d1','d2','d3','d4']
preferred=['Class A/1 (Rhodopsin-like receptors)','GPCR ligand binding','Interleukin-4 and Interleukin-13 signaling','Signaling by GPCR','Signaling by Interleukins']
rows=[x for x in preferred if x in set(react.pathway)]
mat=np.full((len(rows),4),np.nan)
for i,name in enumerate(rows):
 for j,c in enumerate(cohorts):
  z=react[(react.pathway==name)&(react.cohort==c)]
  if len(z):mat[i,j]=float(z.q_value.iloc[0])
val=np.where(np.isnan(mat),0.0,-np.log10(mat));val=np.clip(val,0,8)
masked=np.ma.masked_where(np.isnan(mat),val)
cmap=plt.cm.Blues.copy();cmap.set_bad('#ebeff2')
fig,ax=plt.subplots(figsize=(7.05,2.55),layout='constrained')
im=ax.imshow(masked,cmap=cmap,vmin=0,vmax=7,aspect='auto')
ax.set_yticks(np.arange(len(rows)),['Class A/1 (rhodopsin-like receptors)','GPCR ligand binding','Interleukin-4/13 signalling','Signalling by GPCR','Signalling by interleukins'][:len(rows)],fontsize=8.5)
ax.set_xticks(range(4),['D1','D2','D3','D4'],fontsize=8.3)
ax.tick_params(length=0)
for i in range(len(rows)):
 for j in range(4):
  if np.isnan(mat[i,j]):label='--';tc='#445'
  else:
   q=mat[i,j];label=f'{q:.1e}' if q<.0001 else f'{q:.4f}'.rstrip('0').rstrip('.');tc='white' if val[i,j]>4.8 else '#17202b'
  ax.text(j,i,label,ha='center',va='center',fontsize=8.1,color=tc)
for y in np.arange(.5,len(rows),1):ax.axhline(y,c='white',lw=1)
for x in np.arange(.5,4,1):ax.axvline(x,c='white',lw=1)
ax.set_xlabel('Held-out cohort (panel trained on remaining three cohorts)',fontsize=8.4)
for ext in ['pdf','png']:
 fig.savefig(o/f'tentacles_v2_reactome_terms.{ext}',bbox_inches='tight',pad_inches=.035)
plt.close(fig)
print('Rebuilt full-width vector figures from verified CSVs')

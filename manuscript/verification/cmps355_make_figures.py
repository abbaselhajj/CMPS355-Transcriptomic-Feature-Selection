from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
P=Path('/mnt/data/CMPS355_Final_Evidence');O=Path('/mnt/data/CMPS355_Manuscript_IEEE_v1_0/figures');O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.7,'axes.labelsize':9,'axes.titlesize':9.5,'figure.dpi':160,'savefig.dpi':300,'pdf.fonttype':42})
order=['lasso','elastic_net','stability_selection_cutoff_0p60','StableMate_significant_stable','TENTACLES_V2_nmin6']
disp=['LASSO','Elastic Net','Stability Selection','StableMate','TENTACLES V2']
cohorts=['d1','d2','d3','d4']
p=pd.read_csv(P/'five_method_controlled_by_cohort.csv')
fig,axs=plt.subplots(1,2,figsize=(7.1,3.05),sharey=True,layout='constrained')
for ax,key,tit in zip(axs,['auroc','balanced_accuracy'],['A  AUROC (ranking)','B  Balanced accuracy (0.5 cutoff)']):
 mat=np.array([[p[(p.method==m)&(p.heldout==c)][key].iloc[0] for c in cohorts] for m in order])
 im=ax.imshow(mat,aspect='auto',vmin=0.45,vmax=0.91,cmap='Blues')
 ax.set_xticks(range(4),['D1\n260','D2\n179','D3\n76','D4\n174'])
 ax.set_yticks(range(5),disp)
 ax.set_title(tit,loc='left',fontweight='semibold',pad=8)
 ax.tick_params(length=0)
 for i in range(5):
  for j in range(4):
   ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',color='white' if mat[i,j]>0.72 else '#222',fontsize=8)
 for sep in np.arange(.5,4,.9999): ax.axhline(sep,color='white',lw=.7)
 ax.set_xlabel('Held-out cohort (test n)')
 fig.colorbar(im,ax=ax,shrink=.76,aspect=18,pad=.02)
for e in ['pdf','png']:fig.savefig(O/f'heldout_metric_heatmaps.{e}',bbox_inches='tight')
plt.close(fig)
# Central comparison: one bubble for each selector, size proportional to panel count
m=pd.read_csv(P/'five_method_controlled_macro.csv').set_index('method');g=pd.read_csv(P/'five_method_gene_agreement.csv').set_index('method')
fig,ax=plt.subplots(figsize=(6.95,3.15),layout='constrained')
for met,name in zip(order,disp):
 x=g.loc[met,'mean_loco_gene_jaccard']; y=m.loc[met,'mean_auroc']; n=g.loc[met,'mean_panel_size']
 fc='#164e7a' if met.endswith('nmin6') else '#90a9bc'
 ax.scatter(x,y,s=48+2.9*n,facecolor=fc,edgecolor='#172f42',alpha=.85,lw=.8,zorder=3)
 labels={'lasso':(-37,-11),'elastic_net':(8,7),'stability_selection_cutoff_0p60':(8,7),'StableMate_significant_stable':(8,-2),'TENTACLES_V2_nmin6':(-35,10)}
 dx,dy=labels[met];ax.annotate(name,(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=8.4,ha='right' if dx<0 else 'left',fontweight='semibold' if met.endswith('nmin6') else 'normal')
ax.set_xlim(-.012,.28);ax.set_ylim(.596,.77);ax.grid(True,alpha=.22,ls='--');ax.set_axisbelow(True)
ax.set_xlabel('Mean pairwise LOCO gene Jaccard (higher = more shared genes)')
ax.set_ylabel('Mean held-out AUROC (higher = better ranking)')
ax.set_title('Prediction versus gene-selection consistency',loc='left',pad=9,fontweight='semibold')
for n in [10,100,300]: ax.scatter([],[],s=48+2.9*n,facecolor='#b7c7d2',edgecolor='#172f42',label=f'{n} genes')
ax.legend(title='Approx. mean panel size',loc='upper left',fontsize=7.5,title_fontsize=8,frameon=False,ncol=3,handletextpad=1.0,columnspacing=1.0)
for e in ['pdf','png']:fig.savefig(O/f'prediction_vs_agreement.{e}',bbox_inches='tight')
plt.close(fig)
# Reactome six recurring TENTACLES pathways, FDR-defined only, source values
frame=pd.read_csv(P/'five_method_reactome_significant.csv');frame=frame[frame.method=='TENTACLES_V2_nmin6']
counts=frame.groupby('pathway')['cohort'].nunique().sort_values(ascending=False)
common=counts[counts>=3].index.tolist()
common=sorted(common,key=lambda name:(-counts[name],name))[:7]
if len(common)<6:
 common=counts.index.tolist()[:6]
print('pathways to plot',common)
fig,ax=plt.subplots(figsize=(7.0,2.75),layout='constrained')
ax.set_xlim(-.6,3.5);ax.set_ylim(-.7,len(common)-.4)
for i,name in enumerate(common):
 for j,f in enumerate(cohorts):
  z=frame[(frame.cohort==f)&(frame.pathway==name)]
  if len(z):
   q=float(z.q_value.iloc[0]);val=min(-np.log10(q),10)
   s=20+val*38
   ax.scatter(j,i,s=s,c='#386c93',alpha=.35+.055*min(val,8),edgecolors='#1f405a',lw=.6)
   ax.text(j,i,f'{q:.1g}' if q<.001 else f'{q:.3f}',ha='center',va='center',fontsize=7,color='black' if val<4 else 'white',weight='medium')
  else:
   ax.text(j,i,'--',color='#8c949d',ha='center',va='center')
ax.set_yticks(range(len(common)),common)
ax.set_xticks(range(4),cohorts)
ax.set_xlabel('Held-out cohort')
ax.set_title('TENTACLES V2: repeatedly enriched Reactome pathways (FDR q)',loc='left',fontweight='semibold')
ax.invert_yaxis();ax.tick_params(length=0)
for e in ['pdf','png']:fig.savefig(O/f'tentacles_v2_reactome_terms.{e}',bbox_inches='tight')
plt.close(fig)

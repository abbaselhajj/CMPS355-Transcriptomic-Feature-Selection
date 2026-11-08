from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap

BASE = Path('/mnt/data/CMPS355_Manuscript_IEEE_v1_2')
FIG = BASE / 'figures'
VER = BASE / 'verification'
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family':'DejaVu Sans',
    'font.size':8.2,
    'axes.labelsize':8.5,
    'axes.titlesize':8.8,
    'legend.fontsize':7.0,
    'figure.dpi':180,
    'savefig.dpi':300,
    'pdf.fonttype':42,
    'ps.fonttype':42,
})

order=['lasso','elastic_net','stability_selection_cutoff_0p60','StableMate_significant_stable','TENTACLES_V2_nmin6']
disp=['LASSO','Elastic Net','Stability Selection','StableMate','TENTACLES V2']
cohorts=['d1','d2','d3','d4']
cohort_ticks=['D1\n260','D2\n179','D3\n76','D4\n174']

# ---- Figure 1 / PCA: keep vector pdf but refresh aesthetics ----
pca_path = Path('/mnt/data/pca_coordinates.csv')
if pca_path.exists():
    pca = pd.read_csv(pca_path)
    color_map = {'d1':'#1f77b4','d2':'#ff7f0e','d3':'#2ca02c','d4':'#d62728'}
    label_map = {'d1':'D1','d2':'D2','d3':'D3','d4':'D4'}
    fig, ax = plt.subplots(figsize=(3.35,2.65), constrained_layout=True)
    for c in ['d1','d2','d3','d4']:
        d = pca[pca['cohort']==c]
        ax.scatter(d['PC1'], d['PC2'], s=14, alpha=0.65, c=color_map[c], edgecolors='none', label=label_map[c])
    ax.set_xlabel('PC1 (32.1% variance)')
    ax.set_ylabel('PC2 (16.7% variance)')
    ax.legend(frameon=False, ncol=2, loc='upper right', borderaxespad=0.2, handletextpad=0.3, columnspacing=0.9)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    for ext in ['pdf','png']:
        fig.savefig(FIG/f'cohort_pca_ieee.{ext}', bbox_inches='tight')
    plt.close(fig)

# ---- Figure 2: heatmaps, wide figure retained ----
by_cohort = pd.read_csv(VER/'five_method_controlled_by_cohort.csv')
fig,axs=plt.subplots(1,2,figsize=(7.0,2.9),sharey=True,layout='constrained')
for ax,key,tit in zip(axs,['auroc','balanced_accuracy'],['A  AUROC','B  Balanced accuracy']):
    mat=np.array([[by_cohort[(by_cohort.method==m)&(by_cohort.heldout==c)][key].iloc[0] for c in cohorts] for m in order])
    im=ax.imshow(mat,aspect='auto',vmin=0.45,vmax=0.91,cmap='Blues')
    ax.set_xticks(range(4),cohort_ticks)
    ax.set_yticks(range(5),disp)
    ax.set_title(tit,loc='left',fontweight='semibold',pad=4)
    ax.tick_params(length=0)
    for i in range(5):
        for j in range(4):
            ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',color='white' if mat[i,j]>0.72 else '#222',fontsize=7.5)
    for sep in np.arange(.5,4,.9999):
        ax.axhline(sep,color='white',lw=.7)
    ax.set_xlabel('Held-out cohort (test n)')
    cb=fig.colorbar(im,ax=ax,shrink=.78,aspect=18,pad=.02)
    cb.ax.tick_params(labelsize=7)
for ext in ['pdf','png']:
    fig.savefig(FIG/f'heldout_metric_heatmaps.{ext}',bbox_inches='tight')
plt.close(fig)

# ---- Figure 3: single-column bubble plot with non-overlapping labels ----
macro=pd.read_csv(VER/'five_method_controlled_macro.csv').set_index('method')
genes=pd.read_csv(VER/'five_method_gene_agreement.csv').set_index('method')
fig,ax=plt.subplots(figsize=(3.35,2.55),layout='constrained')
face = {'TENTACLES_V2_nmin6':'#1f5a89'}
for met,name in zip(order,disp):
    x=genes.loc[met,'mean_loco_gene_jaccard']
    y=macro.loc[met,'mean_auroc']
    n=genes.loc[met,'mean_panel_size']
    s=26+3.0*n
    fc=face.get(met, '#9cb4c6')
    ax.scatter(x,y,s=s,facecolor=fc,edgecolor='#23394d',alpha=0.90,lw=0.8,zorder=3)
    # tuned per-point label positions to avoid overlap
    label=f'{name} ({int(round(n))} genes)'
    offsets={
        'StableMate_significant_stable':(7,-2,'left'),
        'stability_selection_cutoff_0p60':(7,10,'left'),
        'elastic_net':(-9,12,'right'),
        'TENTACLES_V2_nmin6':(-6,6,'right'),
        'lasso':(-10,-15,'right'),
    }
    dx,dy,ha=offsets[met]
    ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',ha=ha,va='bottom' if dy>=0 else 'top',
                fontsize=6.9, fontweight='semibold' if met=='TENTACLES_V2_nmin6' else 'normal',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.78))
ax.set_xlim(-0.005,0.255)
ax.set_ylim(0.60,0.77)
ax.grid(True,alpha=.22,ls='--')
ax.set_axisbelow(True)
ax.set_xlabel('Mean pairwise LOCO gene Jaccard')
ax.set_ylabel('Mean held-out AUROC')
for ext in ['pdf','png']:
    fig.savefig(FIG/f'prediction_vs_agreement.{ext}',bbox_inches='tight')
plt.close(fig)

# ---- Figure 4: single-column Reactome heatmap with wrapped labels ----
react = pd.read_csv(VER/'five_method_reactome_significant.csv')
react = react[react['method']=='TENTACLES_V2_nmin6'].copy()
# prefer repeated pathways in >=3 folds, as before
counts=react.groupby('pathway')['cohort'].nunique().sort_values(ascending=False)
common=counts[counts>=3].index.tolist()
if len(common) < 5:
    common = counts.index.tolist()[:5]
else:
    common = common[:5]
# order manually for readability if present
preferred=[
    'Class A/1 (Rhodopsin-like receptors)',
    'GPCR ligand binding',
    'Interleukin-4 and Interleukin-13 signaling',
    'Signaling by GPCR',
    'Signaling by Interleukins'
]
common=[c for c in preferred if c in set(common)] + [c for c in common if c not in preferred]
wrapped=[textwrap.fill(c.replace(' signaling',' signalling'), width=26) for c in common]
mat=np.full((len(common), len(cohorts)), np.nan)
for i,name in enumerate(common):
    for j,c in enumerate(cohorts):
        z=react[(react['cohort']==c)&(react['pathway']==name)]
        if len(z):
            mat[i,j]=float(z['q_value'].iloc[0])
# Convert to intensity on -log10 scale, but keep raw q-values in text
val = np.where(np.isnan(mat), 0.0, -np.log10(mat))
val = np.clip(val, 0, 8)
fig,ax=plt.subplots(figsize=(3.35,2.75),layout='constrained')
# custom light gray for NaNs
masked = np.ma.masked_where(np.isnan(mat), val)
cmap = plt.cm.Blues.copy()
cmap.set_bad('#e8ecef')
im=ax.imshow(masked, aspect='auto', cmap=cmap, vmin=0, vmax=7.0)
ax.set_xticks(range(len(cohorts)), ['D1','D2','D3','D4'])
ax.set_yticks(range(len(common)), wrapped)
ax.tick_params(length=0)
for i in range(len(common)):
    for j in range(len(cohorts)):
        if np.isnan(mat[i,j]):
            txt='--'; color='#444'
        else:
            q=mat[i,j]
            txt=f'{q:.1e}' if q<1e-4 else (f'{q:.5f}'.rstrip('0').rstrip('.'))
            color='white' if val[i,j] > 5.0 else '#1d1d1d'
        ax.text(j,i,txt,ha='center',va='center',fontsize=6.5,color=color)
for y in np.arange(0.5, len(common), 1.0):
    ax.axhline(y, color='white', lw=0.8)
for x in np.arange(0.5, len(cohorts), 1.0):
    ax.axvline(x, color='white', lw=0.8)
ax.set_xlabel('Held-out cohort')
for ext in ['pdf','png']:
    fig.savefig(FIG/f'tentacles_v2_reactome_terms.{ext}',bbox_inches='tight')
plt.close(fig)

print('v1_2 figures regenerated in', FIG)

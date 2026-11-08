"""Exploratory data integrity + exact leakage-safe preparation used by Stage 5.

This is not upstream FASTQ preprocessing, edgeR, DESeq2 or R batch correction.
Every reported count is calculated from the immutable aligned counts archive.
No feature selection or model fitting uses the held-out labels.
"""
from __future__ import annotations
import json,platform,hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .full_gene_loco import COHORTS,load_data,log_cpm,fit_features,MIN_CPM,MIN_FRACTION

EXPECTED_RETAINED={'d1':8337,'d2':8406,'d3':8409,'d4':8438}

def run(root:Path,out:Path)->dict:
    root,out=Path(root),Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Refusing to overwrite completed preprocessing run')
    out.mkdir(parents=True,exist_ok=True)
    meta,mat,genes,provenance=load_data(root)
    cohort=meta.cohort.to_numpy()
    libs=mat.sum(axis=1)
    if len(mat)!=689 or len(genes)!=9983 or not np.isfinite(mat).all() or (mat<0).any() or (libs<=0).any():
        raise ValueError('Unexpected matrix or nonnegative-count invariant failed')
    log=log_cpm(mat)
    counts=[]
    for d in COHORTS:
        ix=cohort==d
        part=mat[ix]
        md=meta[ix]
        lib=libs[ix]
        counts.append(dict(cohort=d,n=int(ix.sum()),CD=int((md['class']=='CD').sum()),
          Not_IBD=int((md['class']=='Not IBD').sum()),studies=int(md.Study.nunique()),
          shared_genes=len(genes),count_zero_fraction=float(np.mean(part==0)),
          shared_library_median=float(np.median(lib)),shared_library_min=float(lib.min()),
          shared_library_max=float(lib.max()),log2_1_CPM_median=float(np.median(log[ix]))))
    cdf=pd.DataFrame(counts)
    cdf.to_csv(out/'cohort_exploration.csv',index=False)
    splits=[]
    for d in COHORTS:
        tr=cohort!=d
        te=cohort==d
        z,indices,scaler,receipt=fit_features(mat[tr])
        if receipt['eligible_genes']!=EXPECTED_RETAINED[d]:
            raise ValueError(f'{d}: Stage5 expected {EXPECTED_RETAINED[d]} genes, found {receipt["eligible_genes"]}')
        if set(meta.loc[tr,'ID']) & set(meta.loc[te,'ID']):raise ValueError('Outer sample leakage')
        if not np.isfinite(z).all():raise ValueError('Nonfinite scaled training matrix')
        # This is for auditing only: test labels are not passed to fit_features().
        splits.append(dict(heldout=d,n_train=int(tr.sum()),n_test=int(te.sum()),
            gene_universe=len(genes),eligible_genes=len(indices),discarded_genes=len(genes)-len(indices),
            fraction_training_required=MIN_FRACTION,cpm_threshold=MIN_CPM,
            threshold_sample_count=receipt['prevalence_threshold_samples'],
            fit_test_overlap=0,training_means_near_zero=float(np.max(np.abs(z.mean(axis=0))))))
        print(f'EARLY_PREPROCESS={d} TRAIN={tr.sum()} TEST={te.sum()} TRAIN_ONLY_ELIGIBLE_GENES={len(indices)}',flush=True)
    sdf=pd.DataFrame(splits)
    sdf.to_csv(out/'loco_training_preprocessing.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.3,4.4))
    ax.boxplot([np.log10(libs[cohort==d]) for d in COHORTS],tick_labels=[d.upper() for d in COHORTS],showfliers=False)
    ax.set(title='Raw count totals across the shared 9,983 genes',ylabel='log10(shared-gene count total)',xlabel='Analytical cohort')
    ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(out/'figure_raw_shared_gene_library_sizes.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.3,4.4))
    ax.bar([d.upper() for d in COHORTS],sdf.eligible_genes)
    ax.set(title='Training-only gene eligibility by LOCO fold',ylabel='Genes retained (of 9,983)',xlabel='Held-out cohort',ylim=(0,9983))
    for i,y in enumerate(sdf.eligible_genes):ax.text(i,y+65,str(y),ha='center',fontsize=10)
    fig.tight_layout();fig.savefig(out/'figure_train_only_filter.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.3,4.4))
    for d in COHORTS:
        z=log[cohort==d]
        rng=np.random.default_rng(355+int(d[-1]))
        sample=z.ravel()[rng.choice(z.size,size=min(25000,z.size),replace=False)]
        ax.hist(sample,bins=np.linspace(0,16,61),density=True,alpha=.45,label=d.upper())
    ax.set(title='Exploratory log2(1 + CPM) expression distributions',xlabel='log2(1 + CPM) (shared-gene denominator)',ylabel='Density')
    ax.legend();fig.tight_layout();fig.savefig(out/'figure_logcpm_distributions.png',dpi=170);plt.close(fig)
    report=dict(status='PASS_PREPROCESSING_EXPLORATION_AND_TRAIN_ONLY_FILTER',
       created_utc=datetime.now(timezone.utc).isoformat(),dataset_sha256=provenance['zip_sha256'],
       samples=len(mat),genes=len(genes),bioprojects=int(meta.Study.nunique()),
       description='Unlabeled feature intersection across 4 cohorts, per-sample log2(1+CPM) on shared 9,983 genes, fold-local CPM prevalence/variance filtering + standardization.',
       caveat='Not FASTQ QC and not edgeR/TENTACLES native normalization. Full PCA and class descriptive figures already exist in Stage 12 cohort characterization.',
       python_version=platform.python_version(),sklearn_version=sklearn.__version__,
       output_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()})
    (out/'report.json').write_text(json.dumps(report,indent=2))
    return report

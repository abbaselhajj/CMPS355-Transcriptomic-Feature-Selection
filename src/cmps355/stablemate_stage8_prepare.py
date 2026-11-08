"""Stage 8: prepare leak-free, audit-ready StableMate LOCO inputs.

Uses the exactly preserved full-gene LASSO/Elastic-Net normalization and
train-only feature filter. This file does not fit or tune a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from full_gene_loco import (COHORTS, EXPECTED_ZIP_SHA, apply_features,
                            fit_features, load_data)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def prepare(heldout: str, dataset_root: Path, work: Path, output: Path,
            r_source_hash: str, full_gene_hash: str, *, k: int = 2000,
            prefilt_p: int = 50, ncore: int = 2, bootstrap_b: int = 1000) -> dict:
    assert heldout in COHORTS
    meta, counts, genes, source = load_data(dataset_root)
    assert source['zip_sha256'] == EXPECTED_ZIP_SHA
    cohorts = meta['cohort'].to_numpy()
    train_idx = np.flatnonzero(cohorts != heldout)
    test_idx = np.flatnonzero(cohorts == heldout)
    assert len(train_idx) + len(test_idx) == 689
    assert not np.intersect1d(train_idx, test_idx).size
    y = (meta['class'] == 'CD').to_numpy(dtype=int)
    assert len(np.unique(y[train_idx])) == 2 and len(np.unique(y[test_idx])) == 2
    assert len(set(cohorts[train_idx])) == 3
    train_x, retained_idx, scaler, prep = fit_features(counts[train_idx])
    test_x = apply_features(counts[test_idx], retained_idx, scaler)
    assert np.isfinite(train_x).all() and np.isfinite(test_x).all()
    retained_genes = [genes[i] for i in retained_idx]
    assert len(retained_genes) == len(set(retained_genes))
    assert len(retained_genes) >= prefilt_p
    assert 'pseudo_predictor' not in retained_genes
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    # Refuse a protocol/source change before touching any durable output file.
    prior_manifest = output / 'manifest.json'
    if prior_manifest.exists():
        prev = json.loads(prior_manifest.read_text(encoding='utf-8'))
        old_cfg = prev.get('model', {})
        old_code = prev.get('code_sha256', {})
        expected_code = {'r_runner': r_source_hash,
                         'full_gene_loco': full_gene_hash,
                         'stage8_prepare': sha(Path(__file__))}
        expected_controls = {'prefilter_p': prefilt_p, 'prefilter_K': k,
                             'model_K': k, 'bootstrap_B': bootstrap_b, 'ncore': ncore}
        if (prev.get('data_sha256') != EXPECTED_ZIP_SHA or
            prev.get('heldout') != heldout or old_code != expected_code or
            any(old_cfg.get(key) != value for key, value in expected_controls.items())):
            raise RuntimeError('Existing StableMate run has different data, settings or source; refusing to touch original outputs')
    for split, indices, matrix in [('train', train_idx, train_x), ('test', test_idx, test_x)]:
        frame = pd.DataFrame(matrix, columns=retained_genes)
        frame.insert(0, 'ID', meta.iloc[indices]['ID'].astype(str).to_numpy())
        frame.to_csv(work/f'{split}_x.csv', index=False, float_format='%.17g')
        clinical = meta.iloc[indices][['ID', 'cohort', 'Study', 'class']].copy()
        clinical['y'] = y[indices]
        clinical.to_csv(work/f'{split}_meta.csv', index=False)
    transform = pd.DataFrame({
        'gene': retained_genes,
        'train_mean_log2cpm': scaler.mean_,
        'train_std_log2cpm': scaler.scale_,
        'shared_universe_index': retained_idx,
    })
    transform_file = output/'gene_transform.csv'
    transform.to_csv(transform_file, index=False, float_format='%.17g')
    split_file = output/'outer_split.csv'
    meta[['ID','cohort','Study','class']].assign(
        split=np.where(cohorts == heldout, 'test', 'train')
    ).to_csv(split_file, index=False)
    manifest = {
        'method': 'StableMate',
        'track': 'adapted_author_method_loco_not_paper_reproduction',
        'heldout': heldout,
        'source_commit': 'cf0fe9f344c1756b6f89fd33f9fdae1a7f35b24a',
        'package_version_expected': '0.1.0',
        'dataset': source,
        'data_sha256': EXPECTED_ZIP_SHA,
        'training_cohorts': sorted(set(cohorts[train_idx].tolist())),
        'train_patients': int(len(train_idx)),
        'test_patients': int(len(test_idx)),
        'class_counts_train': {'CD': int(y[train_idx].sum()), 'Not IBD': int(len(train_idx) - y[train_idx].sum())},
        'class_counts_test': {'CD': int(y[test_idx].sum()), 'Not IBD': int(len(test_idx) - y[test_idx].sum())},
        'shared_genes': int(len(genes)),
        'eligible_genes': len(retained_genes),
        'preprocessing': {
            'method': 'exact full_gene_loco.fit_features and apply_features',
            'normalization': 'log2(1 + CPM) on fixed 9983-gene universe',
            'filter': 'CPM > 1 in >=10% training samples; nonzero training variance',
            'scaler': 'training-only sklearn StandardScaler',
            **prep,
        },
        'model': {
            'prefilter_p': prefilt_p,
            'prefilter_K': k,
            'model_K': k,
            'bootstrap_B': bootstrap_b,
            'ncore': ncore,
            'prefilter_family': 'binomial',
            'prefilter_group_by': 'Yenv',
            'st2_regressor': 'StableMate namespace reg_logit',
            'predictivity_objective': 'StableMate namespace obj_bic_logit',
            'stability_objective': 'StableMate namespace obj_pss_logit',
            'native_prediction': 'predict(...,pred_fun=StableMate:::pred_logit)',
            'paper_K_case_studies': 2000,
            'interpretation': 'K=2000 paper case-study ensemble size; this remains an adapted Crohn disease cohort benchmark, not a published-result replication',
            'seed': 2026 + int(heldout[-1]),
        },
        'code_sha256': {'r_runner': r_source_hash, 'full_gene_loco': full_gene_hash,
                        'stage8_prepare': sha(Path(__file__))},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        'input_files_sha256': {p.name: sha(p) for p in sorted(work.glob('*.csv'))},
        'audit_artifact_sha256': {'gene_transform.csv':sha(transform_file),'outer_split.csv':sha(split_file)},
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    manifest['run_key'] = hashlib.sha256(json.dumps({k:manifest[k] for k in
       ('method','heldout','source_commit','data_sha256','model','code_sha256')},
       sort_keys=True).encode()).hexdigest()
    existing=output/'manifest.json'
    if existing.exists():
        previous=json.loads(existing.read_text())
        if previous.get('run_key') != manifest['run_key']:
            raise RuntimeError('Existing output uses different model settings or sources; refusing to overwrite')
    save_json_atomic(existing, manifest)
    assert len(pd.read_csv(work/'train_x.csv',usecols=['ID'])) == len(train_idx)
    assert len(pd.read_csv(work/'test_x.csv',usecols=['ID'])) == len(test_idx)
    print(f'STAGE8_INPUT_PREPARED heldout={heldout} train={len(train_idx)} test={len(test_idx)} eligible_genes={len(retained_genes)}')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--heldout',required=True,choices=COHORTS)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--r-source-hash',required=True)
    p.add_argument('--full-gene-hash',required=True)
    p.add_argument('--K',type=int,default=2000)
    p.add_argument('--prefilt-p',type=int,default=50)
    p.add_argument('--cores',type=int,default=2)
    p.add_argument('--bootstrap',type=int,default=1000)
    a=p.parse_args()
    prepare(a.heldout,a.data,a.work,a.out,a.r_source_hash,a.full_gene_hash,
            k=a.K,prefilt_p=a.prefilt_p,ncore=a.cores,bootstrap_b=a.bootstrap)

"""Stage 9: immutable, independently auditable TENTACLES LOCO training inputs.

This script NEVER trains a model, chooses a feature from a test outcome, or modifies
an existing result. It uses the same 9,983-name harmonized corrected dataset
and stable sample alignment validator as full_gene_loco.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_DATA_SHA = '8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74'
SOURCE_COMMIT = '069a26271e3885e015df25baa40a7a1140827d5f'
COHORTS = ('d1', 'd2', 'd3', 'd4')
EXPECTED_N = {'d1':260, 'd2':179, 'd3':76, 'd4':174}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(root: Path):
    root = root.resolve()
    if not (root/'configs/dataset.json').is_file():
        raise RuntimeError('Expected canonical CMPS355_Research root')
    data_config = json.loads((root/'configs/dataset.json').read_text())
    if data_config['sha256'] != EXPECTED_DATA_SHA:
        raise RuntimeError('Configured dataset is not the approved corrected dataset')

    sys.path.insert(0, str(root/'src'))
    from cmps355 import full_gene_loco
    meta, values, genes, provenance = full_gene_loco.load_data(root/'data'/'validated')
    if provenance['zip_sha256'] != EXPECTED_DATA_SHA or len(genes) != 9983:
        raise RuntimeError('Dataset identity or common gene universe mismatch')
    if len(meta) != 689 or meta['ID'].duplicated().any():
        raise RuntimeError('Patient identity integrity failed')

    cohort = meta['cohort'].to_numpy()
    all_ids = set(meta['ID'].astype(str))
    if len(all_ids) != 689: raise RuntimeError('Unexpected unique sample IDs')
    for holdout in COHORTS:
        folder = root/'runs'/'tentacles_stage9_inputs'/holdout
        if folder.exists():
            saved = folder / 'input_manifest.json'
            if not saved.is_file():
                raise RuntimeError(f'Unsafe partial existing fold directory: {folder}')
            manifest = json.loads(saved.read_text())
            if (manifest.get('heldout')!=holdout or
                manifest.get('status')!='PASS_INPUT_PREPARATION' or
                manifest.get('dataset_sha256')!=EXPECTED_DATA_SHA):
                raise RuntimeError('Invalid existing Stage 9 input manifest: '+str(saved))
            for rel, proof in manifest['artifacts'].items():
                file = folder / rel
                if not file.is_file() or digest(file)!=proof['sha256']:
                    raise RuntimeError('Existing Stage 9 immutable input is damaged: '+str(file))
            print(f'VERIFIED existing immutable fold: {holdout}, {folder}')
            continue
        train_idx = np.flatnonzero(cohort != holdout)
        test_idx = np.flatnonzero(cohort == holdout)
        if len(test_idx) != EXPECTED_N[holdout] or len(train_idx) != 689-len(test_idx):
            raise RuntimeError('Unexpected fold sample counts')
        training = meta.iloc[train_idx].reset_index(drop=True)
        heldout = meta.iloc[test_idx].reset_index(drop=True)
        if set(training['ID']) & set(heldout['ID']):
            raise RuntimeError(f'Sample leakage: {holdout}')
        if len(set(training['cohort'])) != 3 or set(heldout['cohort']) != {holdout}:
            raise RuntimeError('LOCO cohort split invalid')
        tmp = folder.with_name(folder.name + '.building')
        if tmp.exists():
            raise RuntimeError(f'Leftover uncommitted temp folder: {tmp}; inspect manually')
        (tmp/'discovery').mkdir(parents=True)
        (tmp/'evaluation_only').mkdir()
        (tmp/'audit').mkdir()
        try:
            # R-preProcess receives ONLY these two files. Test samples are isolated.
            for indices, selected, dest in ((train_idx,training,tmp/'discovery'),
                                           (test_idx,heldout,tmp/'evaluation_only')):
                frame = pd.DataFrame(values[indices],columns=genes)
                frame.insert(0,'ID',selected['ID'].astype(str).to_numpy())
                frame.to_csv(dest/'counts.csv',index=False,float_format='%.16g')
                selected[['ID','Study','class','cohort']].to_csv(dest/'clinical.csv',index=False)
            combined = meta[['ID','cohort','Study','class']].copy()
            combined['split'] = np.where(cohort == holdout,'test','train')
            combined.to_csv(tmp/'audit'/'outer_split.csv',index=False)
            artifacts = {}
            for path in tmp.rglob('*'):
                if path.is_file():
                    artifacts[str(path.relative_to(tmp))]={'sha256':digest(path),'bytes':path.stat().st_size}
            manifest = dict(
                project='CMPS355_Research', stage='9_LOCO_inputs_only',
                status='PASS_INPUT_PREPARATION', heldout=holdout,
                method='TENTACLES_official_R_discovery', source_sha=SOURCE_COMMIT,
                dataset_sha256=EXPECTED_DATA_SHA, shared_genes=len(genes),
                fit_patients=len(train_idx), test_patients=len(test_idx),
                fit_cohorts=sorted(set(training['cohort'])), test_cohort=holdout,
                fit_class_counts=training['class'].value_counts().to_dict(),
                test_class_counts=heldout['class'].value_counts().to_dict(),
                gene_harmonization='unlabelled intersection only; common 9983 symbols',
                training_inputs='discovery counts.csv and clinical.csv only',
                restricted_evaluation='evaluation_only is excluded from all author R fit calls',
                preprocessing='later official TENTACLES::preProcess fit to three training cohorts only',
                train_batch_adjustment='none; source-study confounding discussed, not corrected posthoc',
                created_utc=datetime.now(timezone.utc).isoformat(),
                python_prep_sha256=digest(Path(__file__)),
                source_baseline_sha256=digest(root/'src'/'cmps355'/'full_gene_loco.py'),
                artifacts=artifacts,
            )
            (tmp/'input_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
            # Only fully materialized and audited fold becomes visible in final directory.
            os.replace(tmp,folder)
            print(f'PREPARED {holdout}: train={len(train_idx)}, test={len(test_idx)}, genes={len(genes)} -> {folder}')
        except BaseException:
            # Keep a visible .building folder for diagnosis rather than silently overwriting.
            raise
    print('STAGE9_LOCO_INPUT_PREPARATION_COMPLETE')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python tentacles_stage9_prepare_loco.py CMPS355_Research_ROOT')
    prepare(Path(sys.argv[1]))

#!/usr/bin/env python3
"""Read-only export of verified TENTACLES V2 results for controlled CMPS355 analysis.

Uses only Python stdlib. Excludes native .rds and predictor .joblib binaries.
All paths inside the archive are relative to the project root. No source is changed.
"""
import csv
import hashlib
import io
import json
import math
import os
import sys
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

FOLDS = ('d1', 'd2', 'd3', 'd4')
EXPECTED_N = {'d1':260, 'd2':179, 'd3':76, 'd4':174}
EXPECTED_TRAIN_N = {'d1':429, 'd2':510, 'd3':613, 'd4':515}
ALLOW_SUFFIXES = {'.txt', '.csv', '.json', '.md', '.log', '.tsv'}
EXCLUDE_NAMES = {'heldout_predictor.joblib'}
MAX_FILE_BYTES = 8_000_000
MAX_TOTAL_BYTES = 35_000_000
REQUIRED_EXTERNAL = (
  'STAGE10_PASS.txt', 'stage10_summary.json', 'stage10_protocol.md',
  'heldout_metrics.csv', 'heldout_predictions.csv', 'selected_genes.csv',
  'pairwise_gene_jaccard.csv', 'gene_selection_frequency.csv',
  'native_classifier_warning_audit.csv', 'input_sha256.json',
)
REQUIRED_NATIVE = (
  'STATUS.txt', 'consensus_genes_native.csv', 'consensus_genes_nmin6.csv',
  'consensus_genes_nmin5.csv', 'native_votes_nmin6.csv', 'native_votes_nmin5.csv',
  'consensus_protocol.txt', 'input_and_code_sha256.txt', 'output_rds_sha256.txt',
)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def latest(base):
    if not base.is_dir():
        raise ValueError('Missing directory: '+str(base))
    dirs=sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith('20') and p.name.endswith('Z'))
    if not dirs:
        raise ValueError('No timestamped directory: '+str(base))
    return dirs[-1]

def rows(path):
    with path.open('r',encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def genes(path):
    rr=rows(path)
    if not rr or any(set(x)!={'gene'} for x in rr):
        raise ValueError('Gene-list schema/empty: '+str(path))
    result=[x['gene'] for x in rr]
    if len(result)!=len(set(result)) or any(not v or v!=v.strip() for v in result):
        raise ValueError('Duplicate/invalid genes: '+str(path))
    return set(result)

def export(root):
    root=root.expanduser().resolve()
    if not (root/'scripts/run_tentacles_cdpos_v2.sh').is_file():
        raise ValueError('Expected existing CMPS355_TENTACLES_Local project root')
    paths=[]
    fold_dirs={}
    panels={}
    for fold in FOLDS:
        d=latest(root/'runs/tentacles_cd_positive_v2'/fold)
        fold_dirs[fold]=d
        if (d/'STATUS.txt').read_text().strip()!='PASS_CD_POSITIVE_NATIVE_NMIN6_AND_NMIN5':
            raise ValueError(f'{fold}: latest canonical fold not COMPLETE')
        for name in REQUIRED_NATIVE:
            if not (d/name).is_file() or not (d/name).stat().st_size:
                raise ValueError(f'{fold}: missing {name}')
        g6=genes(d/'consensus_genes_native.csv')
        if g6!=genes(d/'consensus_genes_nmin6.csv'):
            raise ValueError(f'{fold}: primary 6-vote panels disagree')
        g5=genes(d/'consensus_genes_nmin5.csv')
        if not g6<=g5:
            raise ValueError(f'{fold}: six-vote panel not subset of five-vote')
        for k,gs in [(6,g6),(5,g5)]:
            rr=rows(d/f'native_votes_nmin{k}.csv')
            if not rr or any(set(x)!={'gene','votes'} for x in rr):
                raise ValueError(f'{fold}: native vote matrix missing/bad schema')
            votes={x['gene']:int(x['votes']) for x in rr}
            if len(votes)!=len(rr) or any(v<0 or v>9 for v in votes.values()):
                raise ValueError(f'{fold}: invalid native vote counts')
            if {g for g,v in votes.items() if v>=k}!=gs:
                raise ValueError(f'{fold}: vote count recomputation disagrees for {k}')
            panels[(fold,k)]=gs
        # Preserve small, text-based evidence. Symlinks are read as their target bytes.
        paths.extend(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in ALLOW_SUFFIXES)
        # Explicitly include STATUS etc, and any recovered-identity metadata present.
    external_dirs={}
    evaluation={}
    for k in (6,5):
        e=latest(root/'runs'/f'tentacles_cd_positive_v2_external_nmin{k}')
        external_dirs[k]=e
        for name in REQUIRED_EXTERNAL:
            if not (e/name).is_file() or not (e/name).stat().st_size:
                raise ValueError(f'nmin{k}: missing {name}')
        if (e/'STAGE10_PASS.txt').read_text().strip()!='PASS_NATIVE_CONSENSUS_PLUS_FIXED_EXTERNAL_HELDOUT_PREDICTIONS':
            raise ValueError(f'nmin{k}: evaluation not passed')
        meta=json.loads((e/'stage10_summary.json').read_text())
        if meta.get('n_evaluable_folds')!=4 or meta.get('n_unique_heldout_patients')!=689 or meta.get('consensus_min_votes')!=k:
            raise ValueError(f'nmin{k}: invalid overall summary')
        reported=meta.get('native_consensus_sizes',{})
        if any(int(reported.get(f,-1))!=len(panels[(f,k)]) for f in FOLDS):
            raise ValueError(f'nmin{k}: signature sizes disagree')
        selections=rows(e/'selected_genes.csv')
        for f in FOLDS:
            sel=[x['gene'] for x in selections if x.get('heldout')==f]
            if len(sel)!=len(set(sel)) or set(sel)!=panels[(f,k)]:
                raise ValueError(f'nmin{k}/{f}: selected gene CSV vs native consensus mismatch')
        metrics=rows(e/'heldout_metrics.csv')
        preds=rows(e/'heldout_predictions.csv')
        if len(metrics)!=4 or len(preds)!=689:
            raise ValueError(f'nmin{k}: expected four performance rows and 689 patient predictions')
        seen=set()
        for f in FOLDS:
            matching=[m for m in metrics if m.get('heldout')==f]
            pr=[x for x in preds if x.get('heldout')==f]
            if len(matching)!=1 or len(pr)!=EXPECTED_N[f]:
                raise ValueError(f'nmin{k}/{f}: wrong heldout counts')
            m=matching[0]
            if int(m['n'])!=EXPECTED_N[f] or int(m['n_selected_genes'])!=len(panels[(f,k)]):
                raise ValueError(f'nmin{k}/{f}: metric count/gene count mismatch')
            auroc=float(m['roc_auc']); bacc=float(m['balanced_accuracy'])
            if not all(math.isfinite(x) and 0<=x<=1 for x in [auroc,bacc]):
                raise ValueError(f'nmin{k}/{f}: invalid metrics')
            for x in pr:
                ident=x['ID']
                if not ident or ident in seen:
                    raise ValueError(f'nmin{k}/{f}: duplicate/empty heldout ID')
                seen.add(ident)
        if len(seen)!=689:
            raise ValueError(f'nmin{k}: heldout ID coverage differs')
        if abs(sum(float(m['roc_auc']) for m in metrics)/4-float(meta['mean_roc_auc_unweighted']))>1e-10:
            raise ValueError(f'nmin{k}: mean AUROC discrepancy')
        if abs(sum(float(m['balanced_accuracy']) for m in metrics)/4-float(meta['mean_balanced_accuracy_unweighted']))>1e-10:
            raise ValueError(f'nmin{k}: mean BAL_ACC discrepancy')
        evaluation[k]={'mean_auroc':meta['mean_roc_auc_unweighted'],
                       'mean_balanced_accuracy':meta['mean_balanced_accuracy_unweighted'],
                       'signature_sizes':reported,'directory':str(e.relative_to(root))}
        paths.extend(p for p in e.iterdir() if p.is_file() and p.suffix.lower() in ALLOW_SUFFIXES)
    # Include original D3/D4 failed-run and recovery audits as small provenance files.
    for f in ('d3','d4'):
        src_base=root/'runs/tentacles_cd_positive_v2'/f
        for d in src_base.iterdir():
            if d.is_dir() and d!=fold_dirs[f] and (d/'STATUS.txt').exists():
                paths.extend(p for p in d.iterdir() if p.is_file() and
                             p.name in {'STATUS.txt','classifiers.log','warning_summary.json','native_cv_f1_by_model.csv','native_feature_importance_quality.csv'})
        recovery_base=root/'runs/tentacles_cd_positive_v2_recovery'/f
        if recovery_base.is_dir():
            for d in recovery_base.iterdir():
                if d.is_dir() and (d/'RECOVERY_STATUS.txt').exists():
                    paths.extend(p for p in d.iterdir() if p.is_file() and p.name in {
                        'RECOVERY_STATUS.txt','native_cv_f1_by_model.csv',
                        'native_f1_orientation_readback.csv','native_feature_importance_readback.csv',
                        'recovery.log','source_rds_sha256_BEFORE.txt','source_rds_sha256_AFTER.txt'})
    # Source code recorded for interpretation and future verification, not execution.
    paths.extend([root/'src/cmps355/tentacles_cdpos_v2_external.py',
                  root/'r/tentacles_cdpos_v2_consensus.R'])
    paths=sorted(set(paths))
    blobs=[]
    total=0
    for p in paths:
        if not p.is_file():
            continue
        if p.name in EXCLUDE_NAMES or p.suffix in ('.rds','.joblib','.zip','.npz'):
            continue
        data=p.read_bytes()
        if len(data)>MAX_FILE_BYTES:
            # Extra large log is not needed; include a note below.
            if p.name.endswith('.log'):
                continue
            raise ValueError(f'Unexpected large file to export: {p} ({len(data)} bytes)')
        total+=len(data)
        if total>MAX_TOTAL_BYTES:
            raise ValueError(f'Unexpected export size >{MAX_TOTAL_BYTES} bytes')
        blobs.append((str(p.relative_to(root)),data))
    manifest={'format':'CMPS355_TENTACLES_V2_FINAL_EVIDENCE_20261108',
       'created_utc':datetime.now(timezone.utc).isoformat(),
       'root_name':root.name,
       'status':'READ_ONLY_EXPORT_NMIN6_AND_NMIN5_VALIDATED',
       'note':'Contains patient-level heldout predictions and private provenance: retain privately. No trained RDS or joblib files.',
       'canonical_folds':{f:str(d.relative_to(root)) for f,d in fold_dirs.items()},
       'evaluation':evaluation,
       'artifacts':[{'relative_path':n,'sha256':sha(d),'bytes':len(d)} for n,d in blobs]}
    output=root/'CMPS355_TENTACLES_V2_Final_Results_For_Analysis.zip'
    if output.exists():
        raise FileExistsError(f'Export already exists, refusing to overwrite: {output}')
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for name,data in blobs:
            z.writestr(name,data)
        z.writestr('EXPORT_MANIFEST.json',json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    with zipfile.ZipFile(output) as z:
        bad=z.testzip()
        if bad:
            raise ValueError('ZIP integrity failed at '+bad)
    print('CMPS355_TENTACLES_V2_FINAL_EXPORT_PASS=TRUE')
    print('ARCHIVE='+str(output))
    print('FILES='+str(len(blobs)))
    print('NMIN6_AUROC='+str(evaluation[6]['mean_auroc']))
    print('NMIN5_AUROC='+str(evaluation[5]['mean_auroc']))
    print('PRIVATE_HELDOUT_PREDICTIONS_INCLUDED=TRUE')
    print('TRAINED_RDS_AND_MODEL_BINARIES_EXCLUDED=TRUE')

if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python3 scripts/export_tentacles_v2_final.py PROJECT_ROOT')
    export(Path(sys.argv[1]))

"""Independent-cohort full-gene LASSO / Elastic Net nested evaluation."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

EXPECTED_ZIP_SHA = '8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74'
EXPECTED_CSV_SHA = {
    'count_d1_aligned.csv': '0f595746ea572a349580d9d3c0242559345b88c927c161b1982933102e6c812d',
    'count_d2_aligned.csv': '2ff413b813e3e4d10927da7f8342eccb35a00205c319c0eb714388aea97f245d',
    'count_d3_aligned.csv': '33b2e51e682da4c41f53becf00b1fc88bb5710293ce2d9d87837bca86a7b3d48',
    'count_d4_aligned.csv': 'c26d8db65b335c89967bd4e072c61a6b7adcb57715c5fe36107d9bd8cd9409e0',
    'desc_d1.csv': '91aadec65bb7f2c1945e7941222b9f490a0c9041d495b8d22606f614484c7b33',
    'desc_d2.csv': '4198f574167754eb00e007ded8cfc2528d50ead0d80b96c1a5cc8139131a140b',
    'desc_d3.csv': '9db3c39975a50afcbd851d3cdcdf56f1e59e7f9bbe18b3b80fd7328400a31332',
    'desc_d4.csv': 'bdd5f6d5ae9a6713470f1eadab2e62a961a38d417f27401359d55b5d906a1193',
}
COHORTS = ('d1','d2','d3','d4')
C_GRID = (0.001, 0.01, 0.1, 1.0)
ELASTIC_RATIOS = (0.5, 0.9)
MIN_CPM = 1.0
MIN_FRACTION = 0.1
SEED = 355
MAX_ITER = 1600
TOL = 0.001


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def find_input(directory: Path):
    root = Path(directory)
    require(root.exists(), f'Input root not found: {root}')
    unpacked = [p for p in root.rglob('count_d1_aligned.csv') if p.is_file()]
    for file in unpacked:
        folder = file.parent
        if all((folder / name).exists() for name in EXPECTED_CSV_SHA):
            return ('directory', folder)
    archives = [p for p in root.rglob('ibd_dataset_sample_aligned.zip') if p.is_file()]
    require(len(archives) == 1, 'Attach exactly one accepted dataset input containing the eight aligned CSV files')
    return ('archive', archives[0])


def load_data(directory: Path) -> tuple[pd.DataFrame, np.ndarray, tuple[str,...], dict]:
    mode, source = find_input(directory)
    if mode == 'archive':
        require(sha_bytes(source.read_bytes()) == EXPECTED_ZIP_SHA, 'Dataset ZIP checksum mismatch')
        with ZipFile(source) as zf:
            require(zf.testzip() is None, 'ZIP CRC verification failed')
            raw = {name: zf.read(name) for name in EXPECTED_CSV_SHA}
    else:
        raw = {name: (source / name).read_bytes() for name in EXPECTED_CSV_SHA}
    for name, expected in EXPECTED_CSV_SHA.items():
        require(sha_bytes(raw[name]) == expected, f'Dataset file checksum mismatch: {name}')
    headers = {}
    for cohort in COHORTS:
        names = next(csv.reader(io.StringIO(raw[f'count_{cohort}_aligned.csv'].decode('utf-8-sig'))))
        require(len(names) == len(set(names)) and names[0] == 'ID', f'Invalid gene names: {cohort}')
        headers[cohort] = names
    genes = tuple(sorted(set.intersection(*(set(headers[c][1:]) for c in COHORTS))))
    require(len(genes) == 9983, 'Unexpected shared gene count')
    counts = []
    metadata = []
    expected_rows = dict(zip(COHORTS, (260,179,76,174)))
    for cohort in COHORTS:
        desc = pd.read_csv(io.BytesIO(raw[f'desc_{cohort}.csv']), dtype={'ID':'string','Study':'string','class':'string'})
        df = pd.read_csv(io.BytesIO(raw[f'count_{cohort}_aligned.csv']), usecols=['ID',*genes], dtype={'ID':'string'})
        require(len(desc)==len(df)==expected_rows[cohort], f'{cohort}: incorrect sample count')
        require(desc['ID'].tolist()==df['ID'].tolist(), f'{cohort}: sample alignment mismatch')
        require({'ID','Study','class'} <= set(desc.columns), f'{cohort}: missing clinical fields')
        require(set(desc['class'])=={'CD','Not IBD'}, f'{cohort}: unexpected classes')
        mat = df.loc[:, list(genes)].to_numpy(dtype=np.float64, copy=True)
        require(np.isfinite(mat).all() and (mat>=0).all() and (mat.sum(axis=1)>0).all(), f'{cohort}: invalid counts')
        desc = desc[['ID','Study','class']].copy()
        desc.insert(0,'cohort',cohort)
        metadata.append(desc)
        counts.append(mat)
    meta = pd.concat(metadata, ignore_index=True)
    require(meta['ID'].is_unique and len(meta)==689, 'Duplicate or missing patients')
    require(meta['Study'].nunique()==6, 'BioProject count mismatch')
    return meta, np.vstack(counts), genes, {'source_mode':mode,'source_path':str(source), 'zip_sha256':EXPECTED_ZIP_SHA,'csv_sha256':EXPECTED_CSV_SHA}


def log_cpm(values: np.ndarray) -> np.ndarray:
    depths = values.sum(axis=1)
    require(np.all(depths>0), 'Empty library detected')
    return np.log2(1.0 + 1e6 * values / depths[:, None])


def fit_features(train_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler, dict]:
    x = log_cpm(train_counts)
    threshold = max(2, int(np.ceil(MIN_FRACTION * len(x))))
    valid = ((x > np.log2(1+MIN_CPM)).sum(axis=0) >= threshold) & (x.var(axis=0) > 1e-12)
    indices = np.flatnonzero(valid)
    require(len(indices) >= 2, 'Insufficient genes after training-only filtering')
    scaler = StandardScaler()
    fitted = scaler.fit_transform(x[:, indices])
    require(np.isfinite(fitted).all(), 'Invalid prepared features')
    return fitted, indices, scaler, {'fit_patients':len(x), 'prevalence_threshold_samples':threshold,'eligible_genes':len(indices)}


def apply_features(counts: np.ndarray, indices: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(log_cpm(counts)[:, indices])


def model_fit(x: np.ndarray, y: np.ndarray, C: float, ratio: float):
    sklearn_major_minor = tuple(int(z) for z in sklearn.__version__.split('.')[:2])
    args = {'l1_ratio':float(ratio)} if sklearn_major_minor >= (1,8) else {
        'penalty':'l1' if ratio==1 else 'elasticnet',
        'l1_ratio': None if ratio==1 else float(ratio)}
    model = LogisticRegression(solver='saga',C=float(C),class_weight='balanced',
                               max_iter=MAX_ITER,tol=TOL,random_state=SEED,**args)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(x,y)
    converged = not any(issubclass(r.category,ConvergenceWarning) for r in recorded)
    return model, converged


def metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    require(np.isfinite(prob).all() and ((0<=prob)&(prob<=1)).all(), 'Invalid probabilities')
    pred = (prob >= .5).astype(int)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y,pred,labels=[0,1]).ravel()]
    return dict(auroc=float(roc_auc_score(y,prob)), average_precision=float(average_precision_score(y,prob)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                sensitivity=tp/(tp+fn), specificity=tn/(tn+fp),
                tp=tp,tn=tn,fp=fp,fn=fn)


def safe_csv(df: pd.DataFrame, path: Path):
    temporary = path.with_suffix(path.suffix+'.tmp')
    df.to_csv(temporary, index=False)
    os.replace(temporary,path)


def safe_json(obj: dict, path: Path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf-8')
    os.replace(temporary,path)


def run(heldout: str, input_root: Path, output_root: Path) -> Path:
    require(heldout in COHORTS, 'Unknown outer cohort')
    started = time.monotonic()
    meta,counts,genes,source=load_data(input_root)
    y = (meta['class']=='CD').to_numpy(dtype=int)
    cohorts=meta['cohort'].to_numpy()
    test_ids=np.flatnonzero(cohorts==heldout)
    train_ids=np.flatnonzero(cohorts!=heldout)
    require(len(test_ids)>0 and len(train_ids)+len(test_ids)==689, 'Outer fold coverage invalid')
    target=Path(output_root) / heldout
    require(not (target/'report.json').exists(), 'Existing completed results cannot be overwritten')
    target.mkdir(parents=True,exist_ok=True)
    scores=[]
    selected=[]
    predictions=[]
    methods={ 'lasso':(1.0,), 'elastic_net':ELASTIC_RATIOS }
    output={ 'status':'RUNNING', 'heldout':heldout,'dataset':source, 'config':{
        'folds':'nested leave-one-analytical-cohort-out', 'gene_universe':'9983 genes shared before inspecting labels',
        'normalization':'log2(1 + CPM) per sample on shared gene universe',
        'filter':'training-only CPM > 1 in >=10% of training samples, nonzero variance',
        'scaling':'training-only standardization','outer_score_threshold':0.5,
        'tuning_metric':'mean of 3 independent cohort-validation AUROCs',
        'C_grid':C_GRID, 'elastic_ratios':ELASTIC_RATIOS,'max_iter':MAX_ITER,'tol':TOL,
        'class_weight':'balanced','random_seed':SEED}, 'software':{
        'python':platform.python_version(), 'numpy':np.__version__,'pandas':pd.__version__,
        'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'candidates':[], 'results':[], 'start_utc':datetime.now(timezone.utc).isoformat() }
    safe_json(output,target/'progress.json')
    for method, ratios in methods.items():
        configs=[(C,r) for r in ratios for C in C_GRID]
        candidate_scores={ (float(c),float(r)):[] for c,r in configs }
        for inner in COHORTS:
            if inner == heldout:
                continue
            ix=np.flatnonzero((cohorts!=heldout)&(cohorts!=inner))
            vx=np.flatnonzero(cohorts==inner)
            require(np.intersect1d(np.r_[ix,vx],test_ids).size==0, 'Outer test leakage')
            require(np.intersect1d(ix,vx).size==0, 'Inner sample overlap')
            train_matrix, inds, scaler, prep=fit_features(counts[ix])
            val_matrix=apply_features(counts[vx],inds,scaler)
            print(f'{heldout}/{method}: validating on {inner} using {len(inds)} genes', flush=True)
            for c,r in configs:
                model, conv = model_fit(train_matrix,y[ix],c,r)
                prob=model.predict_proba(val_matrix)[:,1]
                score=metrics(y[vx],prob)
                scores.append(dict(outer=heldout,method=method,validation=inner,C=c,l1_ratio=r,
                                   n_fit=len(ix),n_validation=len(vx),
                                   n_candidate_genes=len(inds), nonzero=int((np.abs(model.coef_[0])>1e-8).sum()),
                                   converged=conv, n_iter=int(model.n_iter_[0]),**score))
                candidate_scores[(float(c),float(r))].append(score['auroc'])
        require(all(len(z)==3 for z in candidate_scores.values()), 'Incomplete inner evaluation')
        ranked=sorted(candidate_scores,key=lambda cr:(-np.mean(candidate_scores[cr]),cr[0],-cr[1]))
        c,r=ranked[0]
        all_x, inds,scaler,preprocess=fit_features(counts[train_ids])
        heldout_x=apply_features(counts[test_ids],inds,scaler)
        final,conv=model_fit(all_x,y[train_ids],c,r)
        pr=final.predict_proba(heldout_x)[:,1]
        result=dict(method=method,outer=heldout,C=c,l1_ratio=r,
                    inner_mean_auroc=float(np.mean(candidate_scores[(c,r)])),
                    train_patients=len(train_ids),test_patients=len(test_ids),
                    candidate_gene_count=len(inds),nonzero_coefficients=int((np.abs(final.coef_[0])>1e-8).sum()),
                    converged=conv,n_iter=int(final.n_iter_[0]),preprocessing=preprocess,**metrics(y[test_ids],pr))
        output['results'].append(result)
        for i,prob in zip(test_ids,pr):
            predictions.append(dict(outer=heldout,method=method,ID=str(meta.iloc[i]['ID']),
                                    Study=str(meta.iloc[i]['Study']),class_true=str(meta.iloc[i]['class']),
                                    y_true=int(y[i]),probability_CD=float(prob)))
        for position in np.flatnonzero(np.abs(final.coef_[0])>1e-8):
            selected.append(dict(outer=heldout,method=method,gene=genes[inds[position]],
                                 coef_scaled=float(final.coef_[0,position])))
        output['candidates'].extend([dict(method=method,C=z[0],l1_ratio=z[1],
                               mean_validation_auroc=float(np.mean(candidate_scores[z]))) for z in ranked])
        safe_csv(pd.DataFrame(scores),target/'inner_scores.csv')
        safe_csv(pd.DataFrame(predictions),target/'predictions.csv')
        safe_csv(pd.DataFrame(selected,columns=['outer','method','gene','coef_scaled']),target/'selected_genes.csv')
        safe_json(output,target/'progress.json')
        print(f"{heldout}/{method}: AUROC={result['auroc']:.3f}, selected={result['nonzero_coefficients']}, converged={conv}",flush=True)
    require(len(output['results'])==2 and len(predictions)==2*len(test_ids),'Incomplete outer fold')
    require(all(len({x['ID'] for x in predictions if x['method']==m})==len(test_ids) for m in methods),
            'Duplicate outer predictions')
    output['status']='PASS' if all(item['converged'] for item in output['results']) and all(s['converged'] for s in scores) else 'WARN_CONVERGENCE'
    output['elapsed_seconds']=round(time.monotonic()-started,2)
    output['completed_utc']=datetime.now(timezone.utc).isoformat()
    output['artifacts_sha256']={f:hashlib.sha256((target/f).read_bytes()).hexdigest()
                                for f in ('inner_scores.csv','predictions.csv','selected_genes.csv')}
    safe_json(output,target/'report.json')
    print(f'FINAL_STATUS={output["status"]}; HELDOUT={heldout}; ELAPSED_SECONDS={output["elapsed_seconds"]}',flush=True)
    return target

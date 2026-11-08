"""Independent-cohort signatures and exploratory subsampling robustness."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

COHORTS = ('d1', 'd2', 'd3', 'd4')
C_GRID = (0.001, 0.01, 0.1, 1.0)
METHOD_RATIOS = {'lasso': (1.0,), 'elastic_net': (0.5, 0.9)}
SEED = 355
RESAMPLES = 30
COEF_CUTOFF = 1e-8


def require_true(condition, message):
    if not condition:
        raise ValueError(message)


def stratified_half_sample(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for klass in (0, 1):
        locations = np.flatnonzero(labels == klass)
        require_true(len(locations) >= 4, f'Insufficient observations in class {klass}')
        take = len(locations) // 2
        parts.append(rng.choice(locations, size=take, replace=False))
    ids = np.sort(np.concatenate(parts))
    require_true(len(ids) == len(set(ids.tolist())), 'Non-unique subsample')
    require_true(set(labels[ids]) == {0, 1}, 'Missing subsample class')
    return ids


def fit_cohort(meta: pd.DataFrame, counts: np.ndarray, genes: tuple[str, ...],
               cohort: str, target_root: Path, *, n_resamples: int = RESAMPLES,
               c_grid: tuple[float, ...] = C_GRID,
               ratio_grid: dict | None = None, cv_folds: int = 3) -> Path:
    from full_gene_loco import fit_features, model_fit, safe_csv, safe_json

    require_true(cohort in COHORTS, 'Unknown cohort')
    require_true(n_resamples >= 1 and cv_folds >= 2, 'Invalid experiment configuration')
    require_true(len(meta) == len(counts) and counts.shape[1] == len(genes), 'Data shape mismatch')
    require_true(meta['ID'].is_unique, 'Nonunique sample identifiers')
    ratio_grid = ratio_grid if ratio_grid is not None else METHOD_RATIOS
    sample_ids = np.flatnonzero(meta['cohort'].to_numpy() == cohort)
    y_all = (meta['class'].to_numpy() == 'CD').astype(int)
    y = y_all[sample_ids]
    require_true(set(y) == {0, 1}, 'Both classes required')
    require_true(np.bincount(y).min() >= cv_folds, 'Insufficient samples for CV')
    x = counts[sample_ids]
    identifiers = meta.iloc[sample_ids]['ID'].astype(str).to_numpy()
    output = Path(target_root) / cohort
    require_true(not (output / 'report.json').exists(), 'Completed output cannot be overwritten')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary = {'status': 'RUNNING', 'cohort': cohort, 'n_samples': int(len(x)),
               'n_cases': int(sum(y)), 'n_controls': int(len(y) - sum(y)),
               'n_shared_genes': len(genes), 'software': {
                   'python': platform.python_version(), 'numpy': np.__version__,
                   'pandas': pd.__version__, 'scipy': scipy.__version__,
                   'sklearn': sklearn.__version__},
               'configuration': {'seed': SEED, 'cv_folds': cv_folds,
                                 'cv_metric': 'ROC_AUC', 'c_grid': list(c_grid),
                                 'method_ratios': {k: list(v) for k, v in ratio_grid.items()},
                                 'resamples': n_resamples,
                                 'resample_type': 'stratified_without_replacement_half_sample',
                                 'preprocessing': 'training-only log2(1+CPM), CPM>1 in >=10%, standardized',
                                 'class_weight': 'balanced', 'coefficient_cutoff': COEF_CUTOFF},
               'completed_methods': [], 'created_utc': datetime.now(timezone.utc).isoformat()}
    safe_json(summary, output / 'progress.json')
    cv_records, fits, selections, frequencies, subsamples, resample_summary, all_coeffs = [], [], [], [], [], [], []
    config_list = [(m, float(c), float(r)) for m, ratios in ratio_grid.items()
                   for r in ratios for c in c_grid]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED + 47)
    cv_splits = list(cv.split(x, y))
    for split_number, (train, valid) in enumerate(cv_splits, 1):
        transformed, col_idx, scaler, _ = fit_features(x[train])
        from full_gene_loco import apply_features
        held = apply_features(x[valid], col_idx, scaler)
        for method, c, ratio in config_list:
            fit, converged = model_fit(transformed, y[train], c, ratio)
            probabilities = fit.predict_proba(held)[:, 1]
            score = float(roc_auc_score(y[valid], probabilities))
            cv_records.append({'cohort': cohort, 'method': method, 'fold': split_number,
                               'C': c, 'l1_ratio': ratio, 'auroc': score,
                               'n_train': len(train), 'n_validation': len(valid),
                               'n_input_genes': len(col_idx), 'converged': bool(converged),
                               'n_iter': int(fit.n_iter_[0])})
        safe_csv(pd.DataFrame(cv_records), output / 'cv_scores.csv')
    scores = pd.DataFrame(cv_records)
    final_params = {}
    for method, ratios in ratio_grid.items():
        method_scores = scores[scores['method'] == method].groupby(['C', 'l1_ratio'], as_index=False)['auroc'].mean()
        require_true(len(method_scores) == len(c_grid) * len(ratios), 'Incomplete tuning grid')
        method_scores = method_scores.sort_values(['auroc', 'C', 'l1_ratio'], ascending=[False, True, False])
        best = method_scores.iloc[0]
        c, ratio = float(best['C']), float(best['l1_ratio'])
        final_params[method] = (c, ratio)
        transformed, col_idx, scaler, prep = fit_features(x)
        fit, converged = model_fit(transformed, y, c, ratio)
        coefs = np.zeros(len(genes), dtype=np.float64)
        coefs[col_idx] = fit.coef_[0]
        support = np.flatnonzero(np.abs(coefs) > COEF_CUTOFF)
        fits.append({'cohort': cohort, 'method': method, 'C': c, 'l1_ratio': ratio,
                     'cv_mean_auroc': float(best['auroc']), 'n_selected': len(support),
                     'eligible_genes': len(col_idx), 'converged': bool(converged),
                     'n_iter': int(fit.n_iter_[0]), 'sample_n': len(x)})
        for position in support:
            selections.append({'cohort': cohort, 'method': method,
                               'gene': genes[position], 'coef_scaled': float(coefs[position]),
                               'abs_coef': float(abs(coefs[position]))})
        all_coeffs.extend({'cohort': cohort, 'method': method, 'gene': g, 'coef_scaled': float(coefs[j]),
                           'abs_coef': float(abs(coefs[j])), 'selected': bool(abs(coefs[j]) > COEF_CUTOFF)}
                          for j, g in enumerate(genes))
        safe_csv(pd.DataFrame(fits), output / 'signature_summary.csv')
        safe_csv(pd.DataFrame(selections, columns=['cohort','method','gene','coef_scaled','abs_coef']),
                 output / 'selected_genes.csv')
        safe_csv(pd.DataFrame(all_coeffs), output / 'all_gene_coefficients.csv')
        print(f'{cohort}/{method}: CV AUROC={best["auroc"]:.3f}; selected={len(support)}', flush=True)
    safe_json(summary | {'tuned_settings': final_params}, output / 'progress.json')
    for method, (c, ratio) in final_params.items():
        rng = np.random.default_rng(SEED + 1000 * COHORTS.index(cohort) + (0 if method == 'lasso' else 250))
        selection_counts = np.zeros(len(genes), dtype=np.int32)
        for iteration in range(1, n_resamples + 1):
            sampled = stratified_half_sample(y, rng)
            transformed, positions, scaler, prep = fit_features(x[sampled])
            fit, converged = model_fit(transformed, y[sampled], c, ratio)
            chosen = positions[np.flatnonzero(np.abs(fit.coef_[0]) > COEF_CUTOFF)]
            selection_counts[chosen] += 1
            resample_summary.append({'cohort': cohort, 'method': method, 'iteration': iteration,
                                     'n_samples': len(sampled), 'n_cases': int(y[sampled].sum()),
                                     'n_controls': int(len(sampled) - y[sampled].sum()),
                                     'n_eligible_genes': len(positions), 'n_selected': len(chosen),
                                     'converged': bool(converged), 'n_iter': int(fit.n_iter_[0])})
            subsamples.extend({'cohort': cohort, 'method': method, 'iteration': iteration,
                               'ID': identifiers[i], 'class_true': 'CD' if y[i] else 'Not IBD'}
                              for i in sampled)
            safe_csv(pd.DataFrame(resample_summary), output / 'subsample_fit_summary.csv')
            safe_csv(pd.DataFrame(subsamples), output / 'subsample_membership.csv')
            safe_json(summary | {'phase': 'resampling', 'method': method, 'iteration': iteration,
                                 'tuned_settings': final_params}, output / 'progress.json')
            if iteration % 5 == 0 or iteration == n_resamples:
                print(f'{cohort}/{method}: {iteration}/{n_resamples} half-samples', flush=True)
        frequencies.extend({'cohort': cohort, 'method': method, 'gene': gene,
                            'n_selected': int(selection_counts[j]),
                            'n_resamples': int(n_resamples),
                            'frequency': float(selection_counts[j] / n_resamples)}
                           for j, gene in enumerate(genes))
        safe_csv(pd.DataFrame(frequencies), output / 'subsample_gene_frequencies.csv')
        summary['completed_methods'].append(method)
        safe_json(summary, output / 'progress.json')
    file_list = ('cv_scores.csv', 'signature_summary.csv', 'selected_genes.csv',
                 'all_gene_coefficients.csv', 'subsample_fit_summary.csv',
                 'subsample_membership.csv', 'subsample_gene_frequencies.csv')
    for f in file_list:
        require_true((output / f).is_file() and (output / f).stat().st_size > 0, f'Missing output {f}')
    summary['status'] = ('PASS' if scores['converged'].all()
                         and all(x['converged'] for x in fits)
                         and all(x['converged'] for x in resample_summary)
                         else 'WARN_CONVERGENCE')
    summary['elapsed_seconds'] = round(time.monotonic() - started, 1)
    summary['final_parameters'] = {k: {'C': v[0], 'l1_ratio': v[1]} for k,v in final_params.items()}
    summary['final_models'] = fits
    summary['files_sha256'] = {f: hashlib.sha256((output/f).read_bytes()).hexdigest() for f in file_list}
    summary['completed_utc'] = datetime.now(timezone.utc).isoformat()
    safe_json(summary, output / 'report.json')
    print(f'FINAL_STATUS={summary["status"]}; COHORT={cohort}; output={output}', flush=True)
    return output


def run(cohort: str, input_root: Path, output_root: Path, *, n_resamples: int = RESAMPLES):
    from full_gene_loco import load_data
    meta, counts, genes, source = load_data(input_root)
    result = fit_cohort(meta, counts, genes, cohort, output_root, n_resamples=n_resamples)
    path = result / 'report.json'
    content = json.loads(path.read_text())
    content['dataset'] = source
    from full_gene_loco import safe_json
    safe_json(content, path)
    return result

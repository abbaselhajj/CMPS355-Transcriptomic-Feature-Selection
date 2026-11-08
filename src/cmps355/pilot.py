"""Train-only cross-cohort preprocessing and sparse-logistic prediction pilot."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import warnings
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from cmps355.split_design import audit_assignments, build_assignments, read_sample_index


@dataclass(frozen=True)
class PilotConfig:
    c_values: tuple[float, ...] = (0.03, 0.3, 3.0)
    max_genes: int = 300
    min_cpm: float = 1.0
    min_sample_fraction: float = 0.10
    max_iter: int = 1200
    tol: float = 0.001
    seed: int = 355


@dataclass
class PreparedFold:
    indices: np.ndarray
    genes: tuple[str, ...]
    scaler: StandardScaler
    n_prevalence_pass: int
    train: np.ndarray


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_counts(archive: Path, manifest: dict, index: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read the accepted count matrices, using the same predeclared gene universe for all cohorts."""
    with Path(archive).open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    _require(checksum == manifest['sha256'], 'Dataset checksum does not match manifest')
    cohorts = manifest['cohorts']
    with ZipFile(archive) as zf:
        headers = [pd.read_csv(zf.open(f'count_{cohort}_aligned.csv'), nrows=0).columns.tolist()
                   for cohort in cohorts]
        _require(all(columns[0] == 'ID' and len(columns) == len(set(columns)) for columns in headers),
                 'Count matrix headers are invalid')
        genes = tuple(sorted(set.intersection(*(set(columns[1:]) for columns in headers))))
        _require(len(genes) == manifest['expected_shared_gene_count'], 'Shared gene universe changed')
        parts = []
        for cohort in cohorts:
            with zf.open(f'count_{cohort}_aligned.csv') as stream:
                frame = pd.read_csv(stream, usecols=['ID', *genes], dtype={'ID': 'string'})
            expected_ids = index.loc[index['cohort'] == cohort, 'ID'].astype('string').tolist()
            _require(frame['ID'].tolist() == expected_ids, f'{cohort}: count IDs and metadata differ')
            block = frame.loc[:, list(genes)].to_numpy(dtype=np.float64, copy=True)
            _require(np.isfinite(block).all() and (block >= 0).all(), f'{cohort}: invalid count values')
            _require((block.sum(axis=1) > 0).all(), f'{cohort}: zero counts within shared genes')
            parts.append(block)
    matrix = np.vstack(parts)
    _require(matrix.shape == (len(index), len(genes)), 'Sample-by-gene matrix shape invalid')
    return matrix, genes


def load_verified_splits(root: Path, index: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Use an immutable, checksum-matching Stage 3A split export rather than regenerate silently."""
    base = Path(root) / 'runs' / 'split_design'
    candidates = sorted((p for p in base.glob('*/report.json') if p.is_file()), reverse=True)
    if (base / 'report.json').is_file():
        candidates.append(base / 'report.json')
    expected_outer, expected_inner = build_assignments(index, manifest['cohorts'])
    last_error = None
    for report_file in candidates:
        folder = report_file.parent
        try:
            report = json.loads(report_file.read_text(encoding='utf-8'))
            _require(report['status'] == 'PASS' and report['data_sha256'] == manifest['sha256'],
                     'Split report checksum/status invalid')
            for name in ('outer.csv', 'inner.csv'):
                path = folder / name
                _require(path.is_file(), f'Missing split file: {path}')
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                _require(digest == report['artifacts_sha256'][name], f'Split checksum changed: {name}')
            outer = pd.read_csv(folder / 'outer.csv', dtype={'ID': 'string'})
            inner = pd.read_csv(folder / 'inner.csv', dtype={'ID': 'string'})
            for observed, expected, keys in (
                (outer, expected_outer, ['outer_heldout', 'ID']),
                (inner, expected_inner, ['outer_heldout', 'inner_heldout', 'ID']),
            ):
                pd.testing.assert_frame_equal(
                    observed.sort_values(keys).reset_index(drop=True).sort_index(axis=1),
                    expected.sort_values(keys).reset_index(drop=True).sort_index(axis=1),
                    check_dtype=False,
                )
            audit_assignments(index, outer, inner, manifest['cohorts'])
            return outer, inner, folder
        except (ValueError, KeyError, OSError, AssertionError) as exc:
            last_error = exc
    raise ValueError(f'No matching verified Stage 3A split export found in {base}: {last_error}')


def log_cpm(counts: np.ndarray) -> np.ndarray:
    """Per-sample log2(1+CPM), using only counts in the fixed shared-gene universe."""
    counts = np.asarray(counts, dtype=np.float64)
    _require(counts.ndim == 2 and counts.shape[1] > 0, 'Expected a nonempty 2-D count matrix')
    _require(np.isfinite(counts).all() and (counts >= 0).all(), 'Counts must be finite and nonnegative')
    depths = counts.sum(axis=1)
    _require(np.all(depths > 0), 'Every sample must have positive total counts')
    return np.log2(1.0 + counts / depths[:, None] * 1_000_000.0)


def fit_preprocessor(train_counts: np.ndarray, genes: tuple[str, ...], config: PilotConfig) -> PreparedFold:
    """Fit prevalence, variance selection and standardisation to training samples only."""
    _require(train_counts.shape[1] == len(genes), 'Gene list mismatch')
    _require(config.max_genes > 0 and config.min_cpm >= 0 and 0 < config.min_sample_fraction <= 1,
             'Invalid pilot preprocessing configuration')
    train_cpm = log_cpm(train_counts)
    cutoff = np.log2(1.0 + config.min_cpm)
    min_samples = max(2, int(np.ceil(config.min_sample_fraction * len(train_cpm))))
    prevalence = (train_cpm >= cutoff).sum(axis=0) >= min_samples
    variances = train_cpm.var(axis=0)
    eligible = np.flatnonzero(prevalence & (variances > 1e-12))
    _require(eligible.size > 0, 'No usable genes after training-only filter')
    order = np.lexsort((np.array(genes, dtype=str)[eligible], -variances[eligible]))
    selected = eligible[order[:config.max_genes]]
    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(train_cpm[:, selected])
    _require(np.isfinite(train_matrix).all(), 'Non-finite training features')
    return PreparedFold(selected, tuple(genes[i] for i in selected), scaler, int(eligible.size), train_matrix)


def transform_preprocessor(prepared: PreparedFold, new_counts: np.ndarray) -> np.ndarray:
    matrix = prepared.scaler.transform(log_cpm(new_counts)[:, prepared.indices])
    _require(np.isfinite(matrix).all(), 'Non-finite transformed features')
    return matrix


def fit_logistic(X: np.ndarray, y: np.ndarray, method: str, c: float,
                 config: PilotConfig) -> tuple[LogisticRegression, bool]:
    _require(method in ('lasso', 'elastic_net'), 'Unknown pilot model')
    _require(np.array_equal(np.unique(y), np.array([0, 1])), 'Both training classes are required')
    _require(c > 0, 'C must be positive')
    version = tuple(int(part) for part in sklearn.__version__.split('.')[:2])
    penalty_options = (
        {'l1_ratio': 1.0 if method == 'lasso' else 0.5}
        if version >= (1, 8)
        else {'penalty': 'l1' if method == 'lasso' else 'elasticnet',
              'l1_ratio': None if method == 'lasso' else 0.5}
    )
    model = LogisticRegression(
        solver='saga', C=float(c), class_weight='balanced',
        max_iter=config.max_iter, tol=config.tol, random_state=config.seed,
        **penalty_options,
    )
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X, y)
    converged = not any(issubclass(record.category, ConvergenceWarning) for record in seen)
    return model, converged


def score_predictions(y: np.ndarray, probabilities: np.ndarray) -> dict:
    _require(set(np.unique(y)) == {0, 1}, 'Both test classes are required for metrics')
    _require(np.isfinite(probabilities).all() and np.all((0 <= probabilities) & (probabilities <= 1)),
             'Invalid predicted probabilities')
    classifications = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, classifications, labels=[0, 1]).ravel()
    return {
        'auroc': float(roc_auc_score(y, probabilities)),
        'average_precision': float(average_precision_score(y, probabilities)),
        'balanced_accuracy': float(balanced_accuracy_score(y, classifications)),
        'sensitivity': float(tp / (tp + fn)),
        'specificity': float(tn / (tn + fp)),
        'true_positive': int(tp), 'false_positive': int(fp),
        'true_negative': int(tn), 'false_negative': int(fn),
        'n_positive': int(y.sum()), 'n_negative': int((1 - y).sum()),
    }


def _save_csv(rows: list[dict], output: Path, filename: str) -> None:
    pd.DataFrame(rows).to_csv(output / filename, index=False)


def run_pilot(root: Path, config: PilotConfig = PilotConfig()) -> tuple[Path, pd.DataFrame]:
    root = Path(root).resolve()
    manifest = json.loads((root / 'configs' / 'dataset.json').read_text(encoding='utf-8'))
    archive = root / manifest['artifact']
    index = read_sample_index(archive, manifest)
    outer, inner, split_dir = load_verified_splits(root, index, manifest)
    counts, genes = read_counts(archive, manifest, index)
    index_by_id = pd.Series(np.arange(len(index)), index=index['ID'].astype(str))
    y_all = (index['class'].to_numpy() == 'CD').astype(np.int8)
    outer_rows, inner_rows, prediction_rows, gene_rows, candidate_rows, preprocessing_rows = [], [], [], [], [], []
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = root / 'runs' / 'pilot' / stamp
    output.mkdir(parents=True, exist_ok=False)
    manifest_report = {
        'status': 'RUNNING', 'stage': '03B', 'research_role': 'development_pilot_not_final_benchmark',
        'dataset_sha256': manifest['sha256'], 'split_source': str(split_dir.relative_to(root)),
        'config': asdict(config), 'n_samples': len(index), 'n_shared_genes': len(genes),
        'methods': ['lasso', 'elastic_net'],
        'software_versions': {
            'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__,
        },
        'normalisation': 'log2(1 + CPM); CPM denominator = each sample total over common 9983 genes',
        'external_test_policy': 'outer test not used in filtering, scaling, hyperparameter selection or training',
        'fold_preprocessing': 'expression threshold then train-variance top-k then train-only StandardScaler',
        'classifier_policy': 'class_weight=balanced calculated from fitting partition; CD=positive; threshold=0.5',
        'inner_model_selection': 'unweighted mean of 3 held-out-cohort AUROCs; ties select smaller C',
        'started_utc': datetime.now(timezone.utc).isoformat(),
    }

    def persist(status: str) -> None:
        _save_csv(inner_rows, output, 'inner_scores.csv')
        _save_csv(outer_rows, output, 'outer_metrics.csv')
        _save_csv(prediction_rows, output, 'predictions.csv')
        _save_csv(gene_rows, output, 'selected_genes.csv')
        _save_csv(candidate_rows, output, 'candidate_genes.csv')
        _save_csv(preprocessing_rows, output, 'preprocessing.csv')
        manifest_report['status'] = status
        manifest_report['completed_outer_models'] = len(outer_rows)
        manifest_report['n_convergence_warnings'] = int(sum(not row['converged'] for row in inner_rows)
                                                   + sum(not row['converged'] for row in outer_rows))
        if status == 'PASS':
            manifest_report['completed_utc'] = datetime.now(timezone.utc).isoformat()
            manifest_report['artifacts_sha256'] = {
                f: hashlib.sha256((output / f).read_bytes()).hexdigest()
                for f in ('inner_scores.csv', 'outer_metrics.csv', 'predictions.csv',
                          'selected_genes.csv', 'candidate_genes.csv', 'preprocessing.csv')
            }
        (output / 'report.json').write_text(json.dumps(manifest_report, indent=2), encoding='utf-8')

    persist('RUNNING')
    for test_cohort in manifest['cohorts']:
        fold_outer = outer.loc[outer['outer_heldout'] == test_cohort]
        train_ids = fold_outer.loc[fold_outer['role'] == 'train', 'ID'].astype(str)
        test_ids = fold_outer.loc[fold_outer['role'] == 'test', 'ID'].astype(str)
        train_idx = index_by_id.loc[train_ids].to_numpy(dtype=int)
        test_idx = index_by_id.loc[test_ids].to_numpy(dtype=int)
        _require(np.intersect1d(train_idx, test_idx).size == 0, 'Outer split overlap')
        _require(len(train_idx) + len(test_idx) == len(index), 'Outer split incomplete')
        print(f'Outer holdout {test_cohort}: train={len(train_idx)}, test={len(test_idx)}', flush=True)
        for method in ('lasso', 'elastic_net'):
            by_c = {float(c): [] for c in config.c_values}
            for validation_cohort in manifest['cohorts']:
                if validation_cohort == test_cohort:
                    continue
                fold_inner = inner.loc[(inner['outer_heldout'] == test_cohort)
                                       & (inner['inner_heldout'] == validation_cohort)]
                inner_train_ids = fold_inner.loc[fold_inner['role'] == 'train', 'ID'].astype(str)
                val_ids = fold_inner.loc[fold_inner['role'] == 'validation', 'ID'].astype(str)
                fit_idx = index_by_id.loc[inner_train_ids].to_numpy(dtype=int)
                val_idx = index_by_id.loc[val_ids].to_numpy(dtype=int)
                _require(np.intersect1d(fit_idx, val_idx).size == 0, 'Inner split overlap')
                _require(np.intersect1d(test_idx, np.r_[fit_idx, val_idx]).size == 0,
                         'Outer test entered inner fold')
                prep = fit_preprocessor(counts[fit_idx], genes, config)
                validation = transform_preprocessor(prep, counts[val_idx])
                for c in config.c_values:
                    model, converged = fit_logistic(prep.train, y_all[fit_idx], method, float(c), config)
                    probabilities = model.predict_proba(validation)[:, 1]
                    row = {
                        'test_cohort': test_cohort, 'validation_cohort': validation_cohort,
                        'method': method, 'C': float(c), 'converged': converged,
                        'n_fit_samples': len(fit_idx), 'n_validation_samples': len(val_idx),
                        'n_genes': len(prep.genes),
                        'n_nonzero_coefficients': int(np.count_nonzero(np.abs(model.coef_) > 1e-8)),
                        **score_predictions(y_all[val_idx], probabilities),
                    }
                    inner_rows.append(row)
                    by_c[float(c)].append(row['auroc'])
            _require(all(len(values) == 3 for values in by_c.values()), 'Incomplete inner validation')
            best_c = sorted(config.c_values, key=lambda c: (-np.mean(by_c[float(c)]), float(c)))[0]
            prep = fit_preprocessor(counts[train_idx], genes, config)
            heldout = transform_preprocessor(prep, counts[test_idx])
            model, converged = fit_logistic(prep.train, y_all[train_idx], method, float(best_c), config)
            probs = model.predict_proba(heldout)[:, 1]
            selected = np.flatnonzero(np.abs(model.coef_[0]) > 1e-8)
            outer_row = {
                'test_cohort': test_cohort, 'method': method, 'C': float(best_c),
                'mean_inner_auroc': float(np.mean(by_c[float(best_c)])),
                'converged': converged, 'n_train_samples': len(train_idx),
                'n_test_samples': len(test_idx), 'n_genes': len(prep.genes),
                'n_nonzero_coefficients': int(len(selected)),
                **score_predictions(y_all[test_idx], probs),
            }
            outer_rows.append(outer_row)
            preprocessing_rows.append({
                'test_cohort': test_cohort, 'method': method,
                'n_expression_and_variance_pass': prep.n_prevalence_pass,
                'n_variance_selected': len(prep.genes),
                'train_min_shared_gene_count': float(counts[train_idx].sum(axis=1).min()),
                'test_min_shared_gene_count': float(counts[test_idx].sum(axis=1).min()),
            })
            for rank, name in enumerate(prep.genes, start=1):
                candidate_rows.append({
                    'test_cohort': test_cohort, 'method': method,
                    'candidate_rank_by_training_variance': rank, 'gene': name,
                })
            for pos in selected:
                gene_rows.append({
                    'test_cohort': test_cohort, 'method': method, 'gene': prep.genes[pos],
                    'coefficient_scaled': float(model.coef_[0, pos]),
                })
            for i, prob in zip(test_idx, probs):
                prediction_rows.append({
                    'ID': str(index.iloc[i]['ID']), 'test_cohort': test_cohort,
                    'Study': str(index.iloc[i]['Study']), 'method': method,
                    'true_label': str(index.iloc[i]['class']), 'true_CD': int(y_all[i]),
                    'probability_CD': float(prob), 'predicted_CD': int(prob >= 0.5),
                })
            persist('RUNNING')
            print(f"  {method}: C={best_c:g}, AUROC={outer_row['auroc']:.3f}, "
                  f"selected={len(selected)}, converged={converged}", flush=True)

    _require(len(outer_rows) == 2 * len(manifest['cohorts']), 'Missing outer results')
    for method in ('lasso', 'elastic_net'):
        subset = [r for r in prediction_rows if r['method'] == method]
        _require(len(subset) == len(index) and len(set(r['ID'] for r in subset)) == len(index),
                 f'{method}: not exactly one out-of-cohort prediction per patient')
    persist('PASS')
    return output, pd.DataFrame(outer_rows)

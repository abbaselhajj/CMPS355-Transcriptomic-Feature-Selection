"""Reproducible Reactome enrichment of cross-cohort RNA-seq signatures."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import hypergeom

DATA_SHA256 = '8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74'
COHORTS = ('d1','d2','d3','d4')
METHODS = ('lasso','elastic_net')
REACTOME_URLS = (
    'https://reactome.org/download/98/ReactomePathways.gmt.zip',
    'https://reactome.org/download/current/ReactomePathways.gmt.zip',
)
MIN_PATHWAY_SIZE = 10
MAX_PATHWAY_SIZE = 500
MIN_CPM = 1.0
MIN_FRACTION = 0.1


def ensure(cond, msg):
    if not cond:
        raise ValueError(msg)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def save_csv(df, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + '.tmp')
    df.to_csv(temp, index=False)
    os.replace(temp, target)


def save_json(obj, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(temp, target)


def load_counts(data_zip: Path):
    ensure(data_zip.is_file(), f'Missing accepted dataset: {data_zip}')
    raw_zip = data_zip.read_bytes()
    ensure(sha_bytes(raw_zip) == DATA_SHA256, 'Accepted dataset checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        ensure(zf.testzip() is None, 'Dataset ZIP integrity check failed')
        headers = {}
        for cohort in COHORTS:
            filename = f'count_{cohort}_aligned.csv'
            with zf.open(filename) as fh:
                headers[cohort] = next(csv.reader(io.TextIOWrapper(fh, encoding='utf-8-sig')))
            ensure(headers[cohort][0] == 'ID', f'Invalid count header for {cohort}')
            ensure(len(headers[cohort]) == len(set(headers[cohort])), f'Duplicate genes for {cohort}')
        shared = tuple(sorted(set.intersection(*(set(headers[k][1:]) for k in COHORTS))))
        ensure(len(shared) == 9983, 'Shared gene universe not equal to 9,983')
        meta = []
        counts = []
        for cohort in COHORTS:
            count = pd.read_csv(zf.open(f'count_{cohort}_aligned.csv'), usecols=['ID', *shared], dtype={'ID': 'string'})
            clinical = pd.read_csv(zf.open(f'desc_{cohort}.csv'), dtype={'ID': 'string', 'class': 'string'})
            ensure(count['ID'].tolist() == clinical['ID'].tolist(), f'{cohort} sample alignment failure')
            ensure(count['ID'].is_unique, f'{cohort} duplicate ID')
            ensure(set(clinical['class'].astype(str)) == {'CD','Not IBD'}, f'{cohort} unexpected labels')
            matrix = count.loc[:, list(shared)].to_numpy(dtype=np.float64)
            ensure(np.isfinite(matrix).all() and (matrix >= 0).all() and np.all(matrix.sum(axis=1) > 0), f'{cohort} counts invalid')
            counts.append(matrix)
            meta.extend([{'cohort': cohort, 'ID': str(v)} for v in count['ID']])
    ensure(len(meta) == 689, 'Incorrect total samples')
    ensure(len({x['ID'] for x in meta}) == 689, 'Sample IDs are not unique')
    return pd.DataFrame(meta), np.vstack(counts), shared


def eligible_genes(matrix: np.ndarray, genes: tuple[str, ...]):
    depth = matrix.sum(axis=1)
    ensure(np.all(depth > 0), 'Nonpositive sequencing depth')
    cpm = matrix / depth[:, None] * 1e6
    retained = (cpm > MIN_CPM).sum(axis=0) >= max(2, math.ceil(MIN_FRACTION * len(matrix)))
    retained &= np.log2(1 + cpm).var(axis=0) > 1e-12
    result = {genes[i] for i in np.flatnonzero(retained)}
    ensure(len(result) > 100, 'Unexpectedly small eligible gene set')
    return result


def build_backgrounds(meta, matrix, genes):
    labels = meta['cohort'].to_numpy()
    backgrounds = {}
    for cohort in COHORTS:
        backgrounds[('loco', cohort)] = eligible_genes(matrix[labels != cohort], genes)
        backgrounds[('independent', cohort)] = eligible_genes(matrix[labels == cohort], genes)
    return backgrounds


def retrieve_reactome(reference_dir: Path):
    reference_dir.mkdir(parents=True, exist_ok=True)
    cache = reference_dir / 'ReactomePathways.gmt.zip'
    meta_path = reference_dir / 'reference.json'
    if cache.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        ensure(sha_bytes(cache.read_bytes()) == meta['zip_sha256'], 'Reactome cache checksum mismatch')
        return cache, meta
    ensure(not cache.exists() and not meta_path.exists(), 'Incomplete Reactome reference cache, inspect before continuing')
    errors = []
    for url in REACTOME_URLS:
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'CMPS355-Research/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read(10 * 1024 * 1024)
            ensure(data.startswith(b'PK'), 'Reactome download is not a ZIP file')
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                ensure(zf.testzip() is None, 'Reactome archive CRC failed')
                options = [n for n in zf.namelist() if n.lower().endswith('.gmt')]
                ensure(len(options) == 1, f'Expected one GMT member, found {options}')
                gmt = zf.read(options[0])
            ensure(len(gmt) > 10000, 'Suspiciously small Reactome GMT')
            cache.write_bytes(data)
            meta = {'source_url': url, 'downloaded_utc': datetime.now(timezone.utc).isoformat(),
                    'zip_sha256': sha_bytes(data), 'gmt_member': options[0],
                    'gmt_sha256': sha_bytes(gmt), 'release_note': 'Reactome v98 current as of 2026-11-07; verified source file pinned by checksum'}
            save_json(meta, meta_path)
            return cache, meta
        except Exception as exc:
            errors.append(f'{url}: {type(exc).__name__}: {exc}')
    raise RuntimeError('Could not fetch the official Reactome GMT. ' + ' | '.join(errors))


def parse_gmt(archive_path: Path):
    with zipfile.ZipFile(archive_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith('.gmt')]
        ensure(len(names) == 1, 'Expected exactly one GMT file')
        raw = zf.read(names[0]).decode('utf-8-sig')
    pathways = {}
    for lineno, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        fields = [x.strip() for x in line.split('\t')]
        ensure(len(fields) >= 3, f'Malformed Reactome GMT line {lineno}')
        name = fields[0]
        descriptor = fields[1]
        members = set(x for x in fields[2:] if x and x != 'Reactome Pathway')
        ensure(name and members, f'Empty pathway at line {lineno}')
        key = descriptor if descriptor.startswith('R-HSA-') else name
        if key in pathways:
            ensure(pathways[key]['name'] == name, f'Conflicting pathway key {key}')
            pathways[key]['genes'].update(members)
        else:
            pathways[key] = {'pathway': key, 'name': name, 'genes': members}
    ensure(len(pathways) >= 200, f'Only {len(pathways)} Reactome pathways parsed; check GMT format')
    return pathways


def bh(p):
    v = np.asarray(p, dtype=float)
    ensure(len(v) >= 1 and np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all(), 'Invalid p-values')
    order = np.argsort(v, kind='stable')
    q = np.empty(len(v), dtype=float)
    unadjusted = v[order] * len(v) / np.arange(1, len(v) + 1)
    q[order] = np.minimum(1, np.minimum.accumulate(unadjusted[::-1])[::-1])
    return q


def enrichment(selected: set[str], background: set[str], pathways: dict, *, scope: str, cohort: str, method: str):
    ensure(selected <= background, f'{scope}/{cohort}/{method}: selected genes not in eligible universe: {sorted(selected-background)[:5]}')
    ensure(len(selected) > 0, f'{scope}/{cohort}/{method}: empty signature')
    n = len(selected)
    N = len(background)
    rows = []
    for p in pathways.values():
        members = p['genes'] & background
        K = len(members)
        if K < MIN_PATHWAY_SIZE or K > MAX_PATHWAY_SIZE:
            continue
        hits = selected & members
        k = len(hits)
        pval = float(hypergeom.sf(k - 1, N, K, n)) if k else 1.0
        rows.append({'scope': scope, 'cohort': cohort, 'method': method,
                     'pathway_id': p['pathway'], 'pathway': p['name'],
                     'background_size': N, 'signature_size': n,
                     'pathway_size': K, 'overlap_count': k,
                     'expected_overlap': float(n*K/N), 'p_value': pval,
                     'overlap_genes': ';'.join(sorted(hits))})
    ensure(len(rows) > 100, 'Too few testable Reactome pathways')
    df = pd.DataFrame(rows)
    df['q_value'] = bh(df['p_value'])
    df['significant'] = df['q_value'] < 0.05
    return df.sort_values(['q_value', 'p_value', 'pathway_id'], kind='stable').reset_index(drop=True)


def find_reporting_run(root):
    reports = list((root / 'runs' / 'full_gene_results_report').rglob('report.json'))
    candidates = []
    for r in reports:
        try:
            rep = json.loads(r.read_text(encoding='utf-8'))
        except Exception:
            continue
        t = r.parent / 'table_selected_genes.csv'
        if rep.get('status') != 'PASS' or not t.is_file():
            continue
        expected = rep.get('output_sha256', {}).get('table_selected_genes.csv')
        if expected and sha_bytes(t.read_bytes()) != expected:
            raise ValueError(f'Output checksum mismatch in {t}')
        candidates.append((rep.get('created_utc', ''), r.parent, rep))
    ensure(candidates, 'No completed full-gene results report containing table_selected_genes.csv')
    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


def get_loco_signatures(folder):
    df = pd.read_csv(folder / 'table_selected_genes.csv')
    ensure({'outer','method','gene'} <= set(df), 'LOCO selected-gene table has unexpected schema')
    ensure(not df.duplicated(['outer', 'method', 'gene']).any(), 'Duplicate LOCO gene rows')
    ensure(set(df['outer']) == set(COHORTS) and set(df['method']) == set(METHODS), 'Missing LOCO cohort-method signatures')
    return {('loco', c, m): set(df.loc[(df['outer']==c)&(df['method']==m), 'gene'].astype(str))
            for c in COHORTS for m in METHODS}


def get_independent_signatures(root):
    base = root / 'runs' / 'independent_signatures'
    result = {}
    sources = {}
    for cohort in COHORTS:
        candidates = list(base.rglob(f'{cohort}/report.json')) if base.exists() else []
        candidates += [p for p in base.rglob('report.json') if p.parent.name == cohort] if base.exists() else []
        candidates = list(set(candidates))
        available = []
        for p in candidates:
            try:
                report = json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                continue
            if report.get('status') not in ('PASS', 'WARN_CONVERGENCE'):
                continue
            f = p.parent / 'selected_genes.csv'
            expected = report.get('files_sha256', {}).get('selected_genes.csv')
            if not f.exists() or not expected or sha_bytes(f.read_bytes()) != expected:
                continue
            available.append((report.get('completed_utc', ''), f, p))
        if not available:
            continue
        available.sort(key=lambda t: t[0])
        selected_csv = available[-1][1]
        sources[cohort] = {'selected_genes_file': str(selected_csv.relative_to(root)),
                           'selected_genes_sha256': sha_bytes(selected_csv.read_bytes()),
                           'report_file': str(available[-1][2].relative_to(root)),
                           'report_sha256': sha_bytes(available[-1][2].read_bytes())}
        sel = pd.read_csv(selected_csv)
        ensure({'cohort','method','gene'} <= set(sel), f'{cohort} signature file columns invalid')
        ensure(set(sel['cohort']) == {cohort}, f'Unexpected source cohort in {cohort}')
        ensure(not sel.duplicated(['cohort','method','gene']).any(), 'Duplicate gene in independent signature')
        ensure(set(sel['method']) == set(METHODS), f'{cohort} missing a method')
        for m in METHODS:
            result[('independent', cohort, m)] = set(sel.loc[sel['method'] == m, 'gene'].astype(str))
    return result, sources


def pathway_pairs(all_enrichment):
    rows = []
    for scope in sorted(all_enrichment['scope'].unique()):
        for method in METHODS:
            frame = all_enrichment[(all_enrichment['scope'] == scope)&(all_enrichment['method'] == method)]
            groups = {c:set(frame.loc[(frame['cohort']==c) & frame['significant'],'pathway_id']) for c in COHORTS}
            for i, left in enumerate(COHORTS):
                if frame.loc[frame['cohort']==left].empty:
                    continue
                for right in COHORTS[i+1:]:
                    if frame.loc[frame['cohort']==right].empty:
                        continue
                    a,b=groups[left],groups[right]
                    denominator=len(a|b)
                    rows.append({'scope':scope,'method':method,'cohort_a':left,'cohort_b':right,
                                 'sig_a':len(a),'sig_b':len(b),'shared':len(a&b),
                                 'jaccard':len(a&b)/denominator if denominator else float('nan'),
                                 'note':'no_significant_pathways' if denominator==0 else
                                        ('overlapping_loco_training_data' if scope=='loco' else 'independent_cohorts')})
    return pd.DataFrame(rows)


def draw_top_heatmap(frame: pd.DataFrame, prefix: Path):
    ordered = frame.sort_values(['q_value','p_value']).drop_duplicates('pathway_id').head(16)
    if ordered.empty:
        return False
    pathway_ids = ordered['pathway_id'].tolist()
    descriptions = dict(zip(ordered['pathway_id'], ordered['pathway']))
    columns = [f'{c}\n{m}' for c in COHORTS for m in METHODS]
    data = []
    for pid in pathway_ids:
        r=[]
        for c in COHORTS:
            for m in METHODS:
                sub=frame[(frame['cohort']==c)&(frame['method']==m)&(frame['pathway_id']==pid)]
                r.append(float(-np.log10(max(sub['q_value'].iloc[0], 1e-300))) if not sub.empty else np.nan)
        data.append(r)
    a=np.array(data)
    plt.figure(figsize=(12, max(5, len(pathway_ids)*0.35+2)))
    plt.imshow(a, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label='−log10(BH-adjusted p-value)')
    plt.xticks(range(len(columns)),columns, rotation=45,ha='right',fontsize=8)
    plt.yticks(range(len(pathway_ids)),[descriptions[x][:65] for x in pathway_ids],fontsize=8)
    plt.title('Reactome enrichment — independent cohorts' if 'independent' in prefix.stem else 'Reactome enrichment — LOCO signatures (exploratory)')
    plt.tight_layout()
    plt.savefig(prefix.with_suffix('.png'),dpi=220)
    plt.savefig(prefix.with_suffix('.pdf'))
    plt.close()
    return True


def run(root: Path, *, include_loco: bool = True, include_independent: bool = True):
    root=Path(root)
    ensure((root/'README.md').exists(), f'Project root not found: {root}')
    meta, counts, genes=load_counts(root/'data/validated/ibd_dataset_sample_aligned.zip')
    backgrounds=build_backgrounds(meta,counts,genes)
    archive, reference=retrieve_reactome(root/'data/reference/reactome')
    pathways=parse_gmt(archive)
    signatures={}
    sources={}
    if include_loco:
        created, report_dir, original_report=find_reporting_run(root)
        signatures.update(get_loco_signatures(report_dir))
        sources['loco_report_path']=str(report_dir.relative_to(root))
        sources['loco_report_sha256']=sha_bytes((report_dir/'report.json').read_bytes())
    if include_independent:
        independent, independent_sources = get_independent_signatures(root)
        signatures.update(independent)
        sources['independent_reports'] = independent_sources
    ensure(signatures, 'No available signature inputs')
    results=[]
    summary=[]
    for (scope, cohort, method), selected in sorted(signatures.items()):
        bg=backgrounds[(scope,cohort)]
        result=enrichment(selected,bg,pathways,scope=scope,cohort=cohort,method=method)
        results.append(result)
        annotated={g for p in pathways.values() for g in p['genes'] if g in bg}
        summary.append({'scope':scope,'cohort':cohort,'method':method,
                        'n_samples':int((meta.cohort != cohort).sum()) if scope=='loco' else int((meta.cohort == cohort).sum()),
                        'eligible_genes':len(bg),'annotated_eligible_genes':len(annotated),
                        'signature_genes':len(selected),'selected_genes_covered':len(selected & annotated),
                        'tested_pathways':len(result),'significant_pathways':int(result.significant.sum()),
                        'min_q_value':float(result['q_value'].min())})
    all_tables=pd.concat(results,ignore_index=True)
    out=root/'runs/pathway_analysis'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True,exist_ok=False)
    exact_input = [{'scope':scope,'cohort':cohort,'method':method,'gene':gene}
                   for (scope,cohort,method), selected in sorted(signatures.items())
                   for gene in sorted(selected)]
    save_csv(pd.DataFrame(exact_input),out/'input_signatures.csv')
    save_csv(all_tables,out/'enrichment_all.csv')
    save_csv(all_tables[all_tables['significant']],out/'enrichment_significant.csv')
    save_csv(pd.DataFrame(summary),out/'signature_summary.csv')
    save_csv(pathway_pairs(all_tables),out/'pathway_pairwise_jaccard.csv')
    bg_data=[{'scope':scope,'cohort':cohort,'gene':gene} for (scope,cohort),bg in backgrounds.items() for gene in sorted(bg)]
    save_csv(pd.DataFrame(bg_data),out/'eligible_gene_backgrounds.csv')
    for scope in sorted(all_tables['scope'].unique()):
        draw_top_heatmap(all_tables[all_tables['scope']==scope],out/f'figure_enrichment_{scope}')
    output_files=[p for p in out.iterdir() if p.is_file() and p.name!='report.json']
    independent_cohorts = sorted({c for scope,c,m in signatures if scope == 'independent'})
    if len(independent_cohorts) == 4:
        status = 'PASS_INDEPENDENT_AND_LOCO'
    elif independent_cohorts:
        status = 'PASS_PARTIAL_INDEPENDENT'
    else:
        status = 'PASS_LOCO_EXPLORATORY_ONLY'
    report={'status':status,'created_utc':datetime.now(timezone.utc).isoformat(),
            'dataset_sha256':DATA_SHA256,'reactome':reference,'n_reactome_pathways':len(pathways),
            'n_shared_genes':len(genes),'scopes':sorted(all_tables.scope.unique().tolist()),
            'independent_cohorts_available':independent_cohorts,
            'signatures':summary,'sources':sources,
            'method':'one-sided hypergeometric over-representation with BH FDR across all eligible pathways',
            'background_policy':'genes passing the identical CPM>1 in >=10% and nonzero-variance filter on relevant training samples',
            'pathway_size_range_after_background':[MIN_PATHWAY_SIZE,MAX_PATHWAY_SIZE],
            'interpretation':'LOCO overlap exploratory due shared training. Independent cohort enrichment supports pathway reproducibility if independent signatures complete.',
            'python':platform.python_version(),'scipy':scipy.__version__,'pandas':pd.__version__,
            'output_sha256':{p.name:sha_bytes(p.read_bytes()) for p in output_files}}
    save_json(report,out/'report.json')
    return out, report

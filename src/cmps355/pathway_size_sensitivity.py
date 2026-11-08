"""Size-matched sensitivity analysis for independently selected transcriptomic signatures.

Uses saved independent-cohort coefficients and a frozen Reactome reference.
Does not refit classifiers or change prior analysis results.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from pathway_analysis import (
    DATA_SHA256, COHORTS, METHODS, bh, enrichment, ensure, parse_gmt,
    save_csv, save_json, sha_bytes,
)

TOP_K = (25, 50, 100, 200, 500)
RANKED_PATHWAY_N = 20


def _verified_json(p: Path):
    ensure(p.is_file(), f"Required JSON missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _check_sha(p: Path, expected: str):
    ensure(p.is_file(), f"Required input missing: {p}")
    ensure(bool(expected) and sha_bytes(p.read_bytes()) == expected,
           f"Input checksum mismatch: {p}")


def _source_pathway_report(root: Path):
    candidates = []
    for p in (root / "runs" / "pathway_analysis").rglob("report.json"):
        try:
            report = _verified_json(p)
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        if report.get("status") != "PASS_INDEPENDENT_AND_LOCO":
            continue
        files = ("eligible_gene_backgrounds.csv", "input_signatures.csv",
                 "signature_summary.csv", "enrichment_significant.csv")
        hashes = report.get("output_sha256", {})
        if all((p.parent / f).is_file() for f in files):
            for f in files:
                _check_sha(p.parent / f, hashes.get(f))
            candidates.append((report.get("created_utc", ""), p.parent, report))
    ensure(candidates, "No verified Stage 6 PASS_INDEPENDENT_AND_LOCO run in runs/pathway_analysis")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


def _independent_coefficients(root: Path, cohort: str):
    folder = root / "runs" / "independent_signatures" / cohort
    report_p = folder / "report.json"
    report = _verified_json(report_p)
    ensure(report.get("status") == "PASS", f"Independent run not PASS for {cohort}")
    ensure(report.get("cohort") == cohort, f"Incorrect run cohort: {cohort}")
    expected_dataset = report.get("dataset", {})
    if isinstance(expected_dataset, dict):
        observed = expected_dataset.get("sha256", expected_dataset.get("zip_sha256", ""))
        if observed:
            ensure(observed == DATA_SHA256, f"Dataset changed for {cohort}")
    coeff_p = folder / "all_gene_coefficients.csv"
    genes_p = folder / "selected_genes.csv"
    _check_sha(coeff_p, report.get("files_sha256", {}).get("all_gene_coefficients.csv"))
    _check_sha(genes_p, report.get("files_sha256", {}).get("selected_genes.csv"))
    df = pd.read_csv(coeff_p)
    ensure({"cohort", "method", "gene", "abs_coef", "selected"} <= set(df.columns),
           f"Unexpected coefficients schema for {cohort}")
    ensure(set(df["method"]) == set(METHODS), f"Incomplete methods in {cohort}")
    ensure(set(df["cohort"]) == {cohort}, f"Cohort mismatch in {cohort}")
    ensure(not df.duplicated(["method", "gene"]).any(), f"Duplicate coefficients for {cohort}")
    ensure(df["abs_coef"].notna().all() and np.isfinite(df["abs_coef"]).all(),
           f"Non-finite coefficients for {cohort}")
    ensure((df["abs_coef"] >= 0).all(), f"Negative abs_coef in {cohort}")
    ensure(df["selected"].isin([True, False]).all(), f"Invalid selection flags in {cohort}")
    for method in METHODS:
        d = df[df.method == method]
        original = set(pd.read_csv(genes_p).query("method == @method")["gene"].astype(str))
        recovered = set(d.loc[d["selected"], "gene"].astype(str))
        ensure(original == recovered, f"Selected-gene mismatch for {cohort}/{method}")
        ensure(len(recovered) >= max(TOP_K), f"{cohort}/{method} has fewer than {max(TOP_K)} selected genes")
    return df, {"report_path": str(report_p.relative_to(root)),
                "report_sha256": sha_bytes(report_p.read_bytes()),
                "coefficients_path": str(coeff_p.relative_to(root)),
                "coefficients_sha256": sha_bytes(coeff_p.read_bytes()),
                "selected_genes_sha256": sha_bytes(genes_p.read_bytes())}


def _sets_and_enrichment(coefficients, backgrounds, pathways):
    rows_all = []
    signatures = []
    summary = []
    signature_sets = {}
    sig_pathway_sets = {}
    top20_sets = {}
    for cohort in COHORTS:
        for method in METHODS:
            d = coefficients[cohort]
            d = d.loc[(d.method == method) & (d.selected)].copy()
            d.sort_values(["abs_coef", "gene"], ascending=[False, True], kind="stable", inplace=True)
            available = list(d["gene"].astype(str))
            universe = backgrounds[cohort]
            ensure(set(available) <= universe, f"Gene is not in independent universe: {cohort}/{method}")
            for k in TOP_K:
                chosen = set(available[:k])
                ensure(len(chosen) == k, f"Unexpected tie/duplicate in {cohort}/{method}/k{k}")
                signature_sets[(cohort, method, k)] = chosen
                for rank, gene in enumerate(available[:k], 1):
                    signatures.append({"cohort":cohort, "method":method, "top_k":k,
                                       "rank":rank, "gene":gene,
                                       "abs_coef":float(d.iloc[rank - 1]["abs_coef"])})
                enr = enrichment(chosen, universe, pathways, scope="independent_topk", cohort=cohort, method=method)
                enr.insert(3, "top_k", k)
                rows_all.append(enr)
                sigs = set(enr.loc[enr.significant, "pathway_id"].astype(str))
                sig_pathway_sets[(cohort, method, k)] = sigs
                ranking = enr.sort_values(["p_value", "pathway_id"], kind="stable")
                top20_sets[(cohort, method, k)] = set(ranking.head(RANKED_PATHWAY_N)["pathway_id"].astype(str))
                summary.append({"cohort":cohort, "method":method, "top_k":k,
                                "signature_genes":k, "eligible_genes":len(universe),
                                "tested_pathways":len(enr),
                                "significant_pathways":len(sigs),
                                "min_q":float(enr["q_value"].min()),
                                "top_ranked_pathways_n":RANKED_PATHWAY_N})
    return (pd.concat(rows_all, ignore_index=True), pd.DataFrame(signatures),
            pd.DataFrame(summary), signature_sets, sig_pathway_sets, top20_sets)


def _jaccard(a,b):
    union = len(a | b)
    return (len(a & b)/union if union else float("nan")), len(a & b), union


def _overlap_tables(gene_sets, sig_pathway_sets, top20_sets):
    rows = []
    for method in METHODS:
        for k in TOP_K:
            for ca, cb in itertools.combinations(COHORTS,2):
                ag = gene_sets[(ca,method,k)]; bg = gene_sets[(cb,method,k)]
                ap = sig_pathway_sets[(ca,method,k)]; bp = sig_pathway_sets[(cb,method,k)]
                at = top20_sets[(ca,method,k)]; bt = top20_sets[(cb,method,k)]
                gj,gi,gu = _jaccard(ag,bg)
                pj,pi,pu = _jaccard(ap,bp)
                tj,ti,tu = _jaccard(at,bt)
                rows.append({"method":method,"top_k":k,"cohort_a":ca,"cohort_b":cb,
                             "gene_jaccard":gj,"shared_genes":gi,"union_genes":gu,
                             "n_sig_pathways_a":len(ap), "n_sig_pathways_b":len(bp),
                             "significant_pathway_jaccard":pj,"shared_significant_pathways":pi,
                             "union_significant_pathways":pu,
                             "ranked_pathway_top20_jaccard_exploratory":tj,
                             "shared_ranked_pathways":ti,
                             "note": "undefined_sig_overlap_both_empty" if not pu else "valid_sig_overlap"})
    pair = pd.DataFrame(rows)
    srows = []
    for (method,k), g in pair.groupby(["method","top_k"], sort=True):
        srows.append({"method":method,"top_k":int(k),
                      "mean_gene_jaccard":float(g.gene_jaccard.mean()),
                      "mean_significant_pathway_jaccard_defined_pairs":float(g.significant_pathway_jaccard.mean()) if g.significant_pathway_jaccard.notna().any() else float('nan'),
                      "n_defined_sig_pathway_pairs":int(g.significant_pathway_jaccard.notna().sum()),
                      "n_all_pairs":int(len(g)),
                      "mean_ranked_pathway_top20_jaccard_exploratory":float(g.ranked_pathway_top20_jaccard_exploratory.mean())})
    return pair, pd.DataFrame(srows)


def _plot(summary, signatures, output: Path):
    for measure, name, ylabel in [
        ("mean_gene_jaccard", "gene_jaccard_by_k", "Mean independent gene Jaccard"),
        ("mean_ranked_pathway_top20_jaccard_exploratory", "top20_pathway_jaccard_by_k", "Mean exploratory top-20 pathway Jaccard"),
    ]:
        fig, ax = plt.subplots(figsize=(7,4.7),layout="constrained")
        for method in METHODS:
            d = summary[summary.method == method].sort_values("top_k")
            ax.plot(d.top_k,d[measure],marker="o",label=method)
        ax.set_xlabel("Top-ranked selected genes per independent cohort")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(alpha=0.2)
        ax.set_title("Independent-cohort, equal-size sensitivity")
        for ext in ("pdf","png"):
            fig.savefig(output / f"figure_{name}.{ext}", dpi=170)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4.7),layout="constrained")
    for method in METHODS:
        d = signatures[signatures.method == method]
        mean_by_k = d.groupby("top_k").significant_pathways.mean()
        ax.plot(mean_by_k.index,mean_by_k.values,marker="o",label=method)
    ax.set_xlabel("Top-ranked selected genes per independent cohort")
    ax.set_ylabel("Mean number of significant pathways per cohort")
    ax.set_ylim(bottom=0)
    ax.legend(); ax.grid(alpha=0.2)
    ax.set_title("Reactome FDR<0.05, size-controlled signatures")
    for ext in ("pdf","png"):
        fig.savefig(output / f"figure_significant_pathway_counts_by_k.{ext}", dpi=170)
    plt.close(fig)


def run(root: Path):
    root = Path(root).resolve()
    ensure((root / "README.md").exists(), f"Not the project root: {root}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = root / "runs" / "pathway_size_sensitivity" / stamp
    out.mkdir(parents=True,exist_ok=False)
    try:
        _, stage6_dir, stage6 = _source_pathway_report(root)
        ensure(stage6.get("dataset_sha256") == DATA_SHA256, "Stage 6 dataset mismatch")
        reference = root / "data" / "reference" / "reactome" / "ReactomePathways.gmt.zip"
        _check_sha(reference, stage6.get("reactome", {}).get("zip_sha256"))
        pathways = parse_gmt(reference)
        expected_gmt_n = stage6.get("n_reactome_pathways")
        ensure(len(pathways) == expected_gmt_n, f"Reactome pathway count mismatch {len(pathways)}")
        backgrounds_df = pd.read_csv(stage6_dir / "eligible_gene_backgrounds.csv")
        ensure({"scope","cohort","gene"} <= set(backgrounds_df), "Bad Stage 6 backgrounds")
        backgrounds = {}
        for cohort in COHORTS:
            d = backgrounds_df[(backgrounds_df.scope == "independent") & (backgrounds_df.cohort == cohort)]
            ensure(len(d) and d.gene.is_unique, f"Missing/duplicate independent background {cohort}")
            backgrounds[cohort] = set(d.gene.astype(str))
        coefficients = {}; input_sources = {}
        for cohort in COHORTS:
            coefficients[cohort], input_sources[cohort] = _independent_coefficients(root,cohort)
        all_results, selection, summary, gene_sets, sig_path_sets, ranked_path_sets = _sets_and_enrichment(
            coefficients,backgrounds,pathways)
        pair, pair_summary = _overlap_tables(gene_sets,sig_path_sets,ranked_path_sets)
        signif = all_results[all_results.significant].copy()
        outputs = {
            "ranked_gene_selections.csv":selection,
            "enrichment_all_topk.csv":all_results,
            "enrichment_significant_topk.csv":signif,
            "signature_topk_summary.csv":summary,
            "crosscohort_gene_pathway_pairwise.csv":pair,
            "crosscohort_gene_pathway_summary.csv":pair_summary,
        }
        for fn,df in outputs.items():save_csv(df,out/fn)
        _plot(pair_summary,summary,out)
        files = sorted(p.name for p in out.iterdir() if p.is_file())
        report = {
            "status":"PASS", "created_utc":datetime.now(timezone.utc).isoformat(),
            "purpose":"size-matched sensitivity, no classifier refitting",
            "scope":"independent_cohorts",
            "dataset_sha256": DATA_SHA256,
            "top_k":list(TOP_K),"ranked_pathway_n":RANKED_PATHWAY_N,
            "pathway_reference":stage6["reactome"],
            "background_policy":stage6.get("background_policy"),
            "stage6_source_report":str((stage6_dir/"report.json").relative_to(root)),
            "stage6_source_report_sha256":sha_bytes((stage6_dir/"report.json").read_bytes()),
            "cohort_inputs":input_sources,
            "n_independent_signatures":len(summary),
            "n_pairwise_comparisons":len(pair),
            "n_significant_enrichment_rows":len(signif),
            "software":{"python":platform.python_version(),"numpy":np.__version__,
                        "pandas":pd.__version__,"scipy":scipy.__version__},
            "interpretation_limits":[
                "Pathways are hierarchical/overlapping, so repeated related pathways are not independent discoveries.",
                "Top-k is a prespecified sensitivity analysis using existing coefficients, not a new optimized biomarker selector.",
                "Some signatures have no significant pathways; Jaccard is undefined (NaN) if both sets are empty.",
                "Top-20 raw-p-ranked pathway overlap is exploratory, and rankings are not evidence of statistical significance.",
                "Coefficients come from independently tuned within-cohort models and are compared only in rank within each model.",
                "These results do not contain independent-cohort predictive performance for these restricted signatures.",
            ],
            "output_sha256":{fn:sha_bytes((out/fn).read_bytes()) for fn in files},
        }
        save_json(report,out/"report.json")
        return out,report
    except Exception as exc:
        save_json({"status":"FAIL","error_type":type(exc).__name__,"message":str(exc)},out/"failure.json")
        raise

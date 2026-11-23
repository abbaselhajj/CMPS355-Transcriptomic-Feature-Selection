#!/usr/bin/env python3
"""CMPS 355 — Phase 2 Overlap Null (C2: Pairwise + 4-Way) & Fixed-K Size-Sensitivity (C3) v3.

Targeted v3 patches remediating all ChatGPT v2 adversarial preflight blockers:
  - V2-C2-P0a: Hardened fail-closed `finalize_c2_c3_outputs` validates real script/manifest/
    dataset hashes, complete `input_provenance.json` (rejects `{'FAKE': True}`), exact method/
    pair/cohort/K inventory (42 pairwise, 7 four-way, 100 fixed-K cohort rows, 25 macro rows
    for the 7-method/5-LOCO manifest), validated C3 checkpoints with non-empty companions,
    canonical patient ID + label SHA-256 and exact AUROC/BAcc recomputation on every valid
    fixed-K prediction CSV, and NEVER returns `CERTIFIED_COMPLETE` on C2-only when C3 was
    promised (returns `C2_OVERLAP_ONLY_COMPLETE` only if explicitly requested).
  - V2-C2-P0b (Pinned Provenance): Enforces `expected_sha256` verification on all pinned
    archived panel sources AND binds eligible-gene universes to a certified C1 output
    directory (`status == CERTIFIED_COMPLETE`, `dataset_sha256 == EXPECTED_ZIP_SHA`, cross-
    checking every universe file SHA against C1's `eligible_universes_manifest.json`).
  - Monte Carlo 95% Null Intervals: Reports 2.5th–97.5th percentile null intervals for
    pairwise intersection, Jaccard, Overlap Coefficient, and 4-way intersection alongside
    explicit size-only LOCO disclosure.
  - C3 Fixed-K Resumable Checkpoints: Atomic self-checksummed per-task checkpoints under
    `checkpoints/` with mandatory `{"pred", "panel"}` companion validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_VERSION = "phase2_overlap_and_size_v3.0"
COHORTS = ("d1", "d2", "d3", "d4")
PAIRS = ("d1-d2", "d1-d3", "d1-d4", "d2-d3", "d2-d4", "d3-d4")
FIXED_K_GRID = (10, 25, 50, 100, 200)
ALLOWED_ROLES = frozenset({"primary", "sensitivity"})
ALLOWED_SCOPES = frozenset({
    "LOCO_size_only_null_training_dependent",
    "Independent_cohort_null",
})
HISTORICAL_SUBSTRINGS = ("v1", "nmin5", "historical", "saga")

UNRANKED_CONSENSUS_METHODS = frozenset({
    "TENTACLES_V2_nmin6",
    "TENTACLES V2 (6/9 consensus)",
    "StableMate_significant_stable",
    "StableMate (significant-stable)",
})
RANKED_METHODS_SPARSE_NONZERO = frozenset({
    "Corrected_Nested_LASSO",
    "Corrected Nested LASSO",
    "Corrected_Nested_Elastic_Net",
    "Corrected Nested Elastic Net",
})
RANKED_METHODS_CONTINUOUS = frozenset({
    "Stability_Selection_ranked",
    "Stability Selection (ranked frequency)",
    "limma_voom",
    "welch",
})

EXPECTED_ZIP_SHA = "8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74"
EXPECTED_COUNTS = {"d1": 260, "d2": 179, "d3": 76, "d4": 174}
CANONICAL_COHORT_LABEL_SHA256 = {
    "d1": "89b088a597919359cf68036f699212e88dfa10c2b4ddca62fd087d7589d01329",
    "d2": "a96ab4a45a68706da42964a0f22881f2c2dc7baf17d7259a225d0eb497ae38c0",
    "d3": "961d947629aec3d2754f1831334bce6323abf05a76e2fb1c16a4f5b0ad550969",
    "d4": "914eaa22fe9ea15eaf85db315597ffba82d0b528a7e998664e3832dabee594e7",
}
LOCO_CAVEAT_TEXT = (
    "LOCO training folds share 2/3 of cohorts; independent uniform draws provide a "
    "size-only chance reference, not a dependence-calibrated hypothesis test."
)


class HistoricalVersionError(RuntimeError):
    pass


class UnrankedMethodError(RuntimeError):
    pass


class SchemaValidationError(ValueError):
    pass


class ProvenanceError(RuntimeError):
    pass


class CheckpointIntegrityError(RuntimeError):
    pass


class CertificationError(RuntimeError):
    pass


def require(cond: bool, msg: str, exc_cls=ValueError) -> None:
    if not cond:
        raise exc_cls(msg)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha(obj) -> str:
    return sha_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def is_sha256_hex(s) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(ch in "0123456789abcdef" for ch in s.lower())


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def compute_prediction_label_sha(pdf: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{str(r.cohort)},{str(r.ID)},{int(r.y_true)}"
        for r in pdf[["cohort", "ID", "y_true"]].itertuples(index=False)
    ).encode("utf-8")
    return sha_bytes(payload)


def compute_auroc_exact(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    ranks = stats.rankdata(s, method="average")
    return float((float(np.sum(ranks[y == 1])) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_bacc_exact(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    return 0.5 * (float(np.mean(p[y == 1] == 1)) + float(np.mean(p[y == 0] == 0)))


def validate_role_and_scope(entry: dict) -> tuple[str, str]:
    require("role" in entry and entry["role"] is not None, f"Method '{entry.get('name')}' missing required 'role'", SchemaValidationError)
    require("scope" in entry and entry["scope"] is not None, f"Method '{entry.get('name')}' missing required 'scope'", SchemaValidationError)
    role = str(entry["role"]).strip()
    scope = str(entry["scope"]).strip()
    require(role in ALLOWED_ROLES, f"Invalid role '{role}' for '{entry.get('name')}'", SchemaValidationError)
    require(scope in ALLOWED_SCOPES, f"Invalid scope '{scope}' for '{entry.get('name')}'", SchemaValidationError)
    return role, scope


def enforce_no_historical_primary(method_name: str, role: str) -> None:
    low = method_name.lower()
    if role == "primary" and any(tok in low for tok in HISTORICAL_SUBSTRINGS):
        raise HistoricalVersionError(f"Refusing historical/V1/SAGA method as primary: '{method_name}'")


def validate_fixed_k_request(method_name: str, K: int, n_nonzero_available: int | None = None, role: str = "primary") -> tuple[bool, str]:
    require(role in ALLOWED_ROLES, f"Invalid role '{role}'", SchemaValidationError)
    enforce_no_historical_primary(method_name, role)
    require(int(K) in FIXED_K_GRID, f"K={K} not in FIXED_K_GRID {FIXED_K_GRID}", SchemaValidationError)
    if method_name in UNRANKED_CONSENSUS_METHODS:
        raise UnrankedMethodError(f"Method '{method_name}' is an unranked consensus set; cannot fabricate fixed-K ranking.")
    if method_name in RANKED_METHODS_SPARSE_NONZERO:
        require(n_nonzero_available is not None, "n_nonzero_available required for sparse nonzero fixed-K check", SchemaValidationError)
        if K > n_nonzero_available:
            return False, f"NOT_APPLICABLE_EXCEEDS_NONZERO_SUPPORT (K={K} > n_nonzero={n_nonzero_available})"
        return True, "VALID_NONZERO_BETA_RANK"
    if method_name in RANKED_METHODS_CONTINUOUS:
        return True, "VALID_CONTINUOUS_RANK"
    raise SchemaValidationError(f"Unrecognized method_name for fixed-K validation: '{method_name}'")


def exact_pairwise_overlap_expectation(ua: set[str], ub: set[str], ka: int, kb: int) -> float:
    require(len(ua) > 0 and len(ub) > 0, "Eligible universes must be non-empty")
    require(0 <= ka <= len(ua) and 0 <= kb <= len(ub), "Panel size exceeds eligible universe")
    return float((ka * kb * len(ua & ub)) / (len(ua) * len(ub)))


def exact_fourway_overlap_expectation(universes: list[set[str]], sizes: list[int]) -> float:
    require(len(universes) == 4 and len(sizes) == 4, "Exactly 4 cohorts required for 4-way overlap")
    u_inter = len(set.intersection(*universes))
    prob = 1.0
    for u, k in zip(universes, sizes):
        require(len(u) > 0 and 0 <= k <= len(u), "Invalid universe or panel size in 4-way expectation")
        prob *= (k / len(u))
    return float(u_inter * prob)


def monte_carlo_pairwise_null(
    ua: set[str], ub: set[str], ka: int, kb: int, obs_inter: int,
    n_mc: int = 10000, seed: int = 20261122,
) -> dict:
    """Fixed-seed Monte Carlo pairwise null with 95% null intervals (2.5th–97.5th percentiles)."""
    exp_exact = exact_pairwise_overlap_expectation(ua, ub, ka, kb)
    u_inter = len(ua & ub)
    u_a_only = len(ua) - u_inter
    u_b_only = len(ub) - u_inter
    rng = np.random.default_rng(seed)

    na_shared = rng.hypergeometric(u_inter, u_a_only, ka, size=n_mc) if (u_inter > 0 and ka > 0) else np.zeros(n_mc, dtype=int)
    nb_shared = rng.hypergeometric(u_inter, u_b_only, kb, size=n_mc) if (u_inter > 0 and kb > 0) else np.zeros(n_mc, dtype=int)

    inter_null = np.zeros(n_mc, dtype=np.int64)
    if u_inter > 0:
        for i in range(n_mc):
            na_i, nb_i = int(na_shared[i]), int(nb_shared[i])
            if na_i > 0 and nb_i > 0:
                inter_null[i] = rng.hypergeometric(na_i, u_inter - na_i, nb_i)

    union_null = ka + kb - inter_null
    jacc_null = np.where(union_null > 0, inter_null / union_null, 0.0)
    min_k = min(ka, kb)
    ov_null = inter_null / min_k if min_k > 0 else np.zeros(n_mc, dtype=np.float64)

    int_lo, int_hi = np.percentile(inter_null, [2.5, 97.5])
    jac_lo, jac_hi = np.percentile(jacc_null, [2.5, 97.5])
    ov_lo, ov_hi = np.percentile(ov_null, [2.5, 97.5])

    obs_union = ka + kb - obs_inter
    return {
        "u_a": len(ua),
        "u_b": len(ub),
        "u_inter": u_inter,
        "k_a": ka,
        "k_b": kb,
        "obs_intersection": int(obs_inter),
        "exact_expected_intersection": exp_exact,
        "mc_mean_intersection": float(inter_null.mean()),
        "mc_sd_intersection": float(inter_null.std(ddof=1)) if n_mc > 1 else 0.0,
        "mc_inter_ci_low": float(int_lo),
        "mc_inter_ci_high": float(int_hi),
        "obs_jaccard": float(obs_inter / obs_union) if obs_union > 0 else 0.0,
        "mc_mean_jaccard": float(jacc_null.mean()),
        "mc_jaccard_ci_low": float(jac_lo),
        "mc_jaccard_ci_high": float(jac_hi),
        "obs_overlap_coeff": float(obs_inter / min_k) if min_k > 0 else 0.0,
        "mc_mean_overlap_coeff": float(ov_null.mean()),
        "mc_overlap_ci_low": float(ov_lo),
        "mc_overlap_ci_high": float(ov_hi),
        "mc_emp_p_ge_obs": float((1 + np.sum(inter_null >= obs_inter)) / (n_mc + 1)),
        "n_mc": int(n_mc),
        "seed": int(seed),
    }


def monte_carlo_fourway_null(
    universes: list[set[str]], sizes: list[int], obs_fourway: int,
    n_mc: int = 10000, seed: int = 20261122,
) -> dict:
    """Fixed-seed Monte Carlo 4-way intersection null with 95% null interval (2.5th–97.5th percentiles)."""
    exp_exact = exact_fourway_overlap_expectation(universes, sizes)
    u_all4 = set.intersection(*universes)
    m4 = len(u_all4)
    rng = np.random.default_rng(seed)

    hits = np.full(n_mc, m4, dtype=np.int64)
    if m4 > 0:
        for u, k in zip(universes, sizes):
            n_in_u4 = rng.hypergeometric(m4, len(u) - m4, k, size=n_mc) if k > 0 else np.zeros(n_mc, dtype=np.int64)
            next_hits = np.zeros(n_mc, dtype=np.int64)
            for i in range(n_mc):
                hi, ni = int(hits[i]), int(n_in_u4[i])
                if hi > 0 and ni > 0:
                    next_hits[i] = rng.hypergeometric(hi, m4 - hi, ni)
            hits = next_hits
    else:
        hits = np.zeros(n_mc, dtype=np.int64)

    fw_lo, fw_hi = np.percentile(hits, [2.5, 97.5])
    return {
        "u_fourway_common": int(m4),
        "k_d1": int(sizes[0]),
        "k_d2": int(sizes[1]),
        "k_d3": int(sizes[2]),
        "k_d4": int(sizes[3]),
        "obs_fourway_intersection": int(obs_fourway),
        "exact_expected_fourway": exp_exact,
        "mc_mean_fourway": float(hits.mean()),
        "mc_sd_fourway": float(hits.std(ddof=1)) if n_mc > 1 else 0.0,
        "mc_fourway_ci_low": float(fw_lo),
        "mc_fourway_ci_high": float(fw_hi),
        "mc_p_ge_1": float(np.mean(hits >= 1)),
        "mc_emp_p_ge_obs": float((1 + np.sum(hits >= obs_fourway)) / (n_mc + 1)),
        "n_mc": int(n_mc),
        "seed": int(seed),
    }


def validate_manifest_config(cfg: dict, require_pinned_hashes: bool = True) -> None:
    """Validate manifest structure, required enums, historical guards, 4-cohort keys, and pinned SHA-256s."""
    require(isinstance(cfg, dict) and isinstance(cfg.get("methods"), list) and len(cfg["methods"]) > 0,
            "Manifest must contain non-empty 'methods' list", SchemaValidationError)
    for entry in cfg["methods"]:
        require("name" in entry and str(entry["name"]).strip(), "Method entry missing 'name'", SchemaValidationError)
        mname = str(entry["name"]).strip()
        role, scope = validate_role_and_scope(entry)
        enforce_no_historical_primary(mname, role)
        require("panel_sources" in entry and set(entry["panel_sources"].keys()) == set(COHORTS),
                f"Method '{mname}' must define panel_sources for exact cohorts {COHORTS}", SchemaValidationError)
        require("universe_csvs" in entry and set(entry["universe_csvs"].keys()) == set(COHORTS),
                f"Method '{mname}' must define universe_csvs for exact cohorts {COHORTS}", SchemaValidationError)
        if require_pinned_hashes:
            for c in COHORTS:
                ps = entry["panel_sources"][c]
                require(isinstance(ps, dict) and is_sha256_hex(ps.get("expected_sha256")),
                        f"Method '{mname}'/{c} panel_sources missing pinned expected_sha256", ProvenanceError)
            # Universes must either pin expected_sha256 directly OR bind to certified C1 universe manifest
            has_c1_binding = bool(cfg.get("c1_certified_output_dir"))
            for c in COHORTS:
                us = entry["universe_csvs"][c]
                u_pinned = isinstance(us, dict) and is_sha256_hex(us.get("expected_sha256"))
                require(u_pinned or has_c1_binding,
                        f"Method '{mname}'/{c} universe_csvs neither pins expected_sha256 nor binds c1_certified_output_dir", ProvenanceError)


def load_certified_c1_universe_hashes(cfg: dict) -> dict | None:
    """If manifest binds to C1 certified outputs, verify C1 manifest is CERTIFIED_COMPLETE and return exact universe SHAs."""
    c1_out = cfg.get("c1_certified_output_dir")
    if not c1_out:
        return None
    c1_dir = Path(c1_out)
    m_p = c1_dir / "manifest.json"
    u_p = c1_dir / "eligible_universes" / "eligible_universes_manifest.json"
    require(m_p.is_file() and u_p.is_file(), f"Certified C1 manifest/universes missing under {c1_dir}", ProvenanceError)
    c1_m = json.loads(m_p.read_text(encoding="utf-8"))
    require(c1_m.get("status") == "CERTIFIED_COMPLETE", f"C1 output is not CERTIFIED_COMPLETE: {c1_m.get('status')}", ProvenanceError)
    require(c1_m.get("dataset_sha256") == EXPECTED_ZIP_SHA, "C1 dataset_sha256 mismatch", ProvenanceError)
    if cfg.get("expected_c1_script_sha256"):
        require(c1_m.get("execution_script_sha256") == cfg["expected_c1_script_sha256"], "C1 execution_script_sha256 mismatch", ProvenanceError)
    return json.loads(u_p.read_text(encoding="utf-8"))


def _read_gene_table(spec: dict, cohort: str) -> tuple[pd.DataFrame, str, str]:
    if "zip_path" in spec:
        zp = Path(spec["zip_path"])
        require(zp.is_file(), f"Missing ZIP source for {cohort}: {zp}", ProvenanceError)
        zsha = sha_file(zp)
        if spec.get("expected_sha256"):
            require(zsha == spec["expected_sha256"], f"ZIP SHA mismatch for {zp}: got={zsha}", ProvenanceError)
        member = spec["zip_member"]
        with ZipFile(zp) as zf:
            df = pd.read_csv(zf.open(member))
        fold_col = spec.get("fold_col", "heldout")
        require(fold_col in df.columns, f"Missing fold column '{fold_col}' in {member}", SchemaValidationError)
        df = df[df[fold_col].astype(str) == cohort].copy()
        return df, f"{zp}::{member}", zsha
    else:
        p = Path(spec["path"])
        require(p.is_file(), f"Missing CSV file for {cohort}: {p}", ProvenanceError)
        fsha = sha_file(p)
        if spec.get("expected_sha256"):
            require(fsha == spec["expected_sha256"], f"CSV SHA mismatch for {p}: got={fsha}", ProvenanceError)
        return pd.read_csv(p), str(p), fsha


def _extract_panel_and_ranking(spec: dict, cohort: str) -> tuple[list[str], list[str] | None, str, str]:
    df, src, fsha = _read_gene_table(spec, cohort)
    gene_col = spec.get("gene_col", df.columns[0])
    require(gene_col in df.columns, f"Missing gene column '{gene_col}' in {src}", SchemaValidationError)

    ranked_genes = None
    if spec.get("score_col"):
        scol = spec["score_col"]
        require(scol in df.columns, f"Missing score_col '{scol}' in {src}", SchemaValidationError)
        rdf = df.copy()
        rdf[gene_col] = rdf[gene_col].astype(str).str.strip()
        rdf["_sort_key"] = rdf[scol].astype(float).abs() if spec.get("sort_abs", False) else rdf[scol].astype(float)
        tie_col = spec.get("tie_score_col")
        if tie_col and tie_col in rdf.columns:
            rdf["_tie_key"] = rdf[tie_col].astype(float).abs() if spec.get("sort_abs", False) else rdf[tie_col].astype(float)
            rdf = rdf.sort_values(by=["_sort_key", "_tie_key", gene_col], ascending=[False, False, True], kind="mergesort")
        else:
            rdf = rdf.sort_values(by=["_sort_key", gene_col], ascending=[False, True], kind="mergesort")
        ranked_genes = rdf[gene_col].tolist()
        require(len(ranked_genes) == len(set(ranked_genes)), f"Duplicate genes in ranked table {src}", SchemaValidationError)

    sdf = df.copy()
    if spec.get("filter_col"):
        fcol = spec["filter_col"]
        fval = spec.get("filter_value", True)
        require(fcol in sdf.columns, f"Missing filter_col '{fcol}' in {src}", SchemaValidationError)
        sdf = sdf[sdf[fcol] == fval]

    genes = [str(g).strip() for g in sdf[gene_col].tolist()]
    require(len(genes) > 0 and all(len(g) > 0 for g in genes) and len(genes) == len(set(genes)),
            f"Empty/invalid/duplicate selected genes for {cohort} in {src}", SchemaValidationError)
    return genes, ranked_genes, src, fsha


def _extract_universe(spec_or_path, cohort: str, scope: str, c1_univ_manifest: dict | None) -> tuple[set[str], str, str]:
    if isinstance(spec_or_path, str):
        p = Path(spec_or_path)
        exp_sha = None
    else:
        p = Path(spec_or_path["path"])
        exp_sha = spec_or_path.get("expected_sha256")
    require(p.is_file(), f"Missing eligible-gene universe CSV for {cohort}: {p}", ProvenanceError)
    fsha = sha_file(p)
    if exp_sha:
        require(fsha == exp_sha, f"Universe SHA mismatch for {p}", ProvenanceError)
    if c1_univ_manifest is not None:
        sk = "loco" if scope == "LOCO_size_only_null_training_dependent" else "indep"
        c1_sha = c1_univ_manifest[sk][cohort]["sha256"]
        require(fsha == c1_sha, f"Universe SHA for {p} does not match certified C1 manifest ({c1_sha})", ProvenanceError)
    df = pd.read_csv(p)
    gcol = "gene" if "gene" in df.columns else df.columns[0]
    genes = [str(g).strip() for g in df[gcol].tolist()]
    require(len(genes) > 0 and len(genes) == len(set(genes)), f"Invalid/duplicate universe genes in {p}", SchemaValidationError)
    return set(genes), str(p), fsha


def save_validated_c3_checkpoint(
    ckpt_path: Path, task_id: str, payload: dict,
    script_sha: str, manifest_sha: str, dataset_sha: str,
    companion_files: dict[str, Path], out_dir: Path,
) -> None:
    require(bool(companion_files), f"C3 checkpoint {task_id} must declare non-empty companion_files", CheckpointIntegrityError)
    out_root = out_dir.resolve()
    comp_map = {}
    for key, p in companion_files.items():
        rp = p.resolve()
        require(rp.is_file() and rp.stat().st_size > 0, f"C3 companion missing/empty ({key}): {rp}", CheckpointIntegrityError)
        rel = rp.relative_to(out_root).as_posix()
        comp_map[key] = {"rel_path": rel, "size_bytes": int(rp.stat().st_size), "sha256": sha_file(rp)}
    atomic_write_json(ckpt_path, {
        "schema_version": "c3_checkpoint_v3",
        "status": "COMPLETED",
        "task_id": task_id,
        "execution_script_sha256": script_sha,
        "manifest_sha256": manifest_sha,
        "dataset_sha256": dataset_sha,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "companion_files": comp_map,
        "payload_sha256": canonical_json_sha(payload),
        "payload": payload,
    })


def load_validated_c3_checkpoint(
    ckpt_path: Path, expected_task_id: str,
    script_sha: str, manifest_sha: str, dataset_sha: str,
    expected_companion_keys: set[str], out_dir: Path,
) -> dict | None:
    require(bool(expected_companion_keys), "expected_companion_keys must be non-empty", CheckpointIntegrityError)
    if not ckpt_path.exists():
        return None
    rec = json.loads(ckpt_path.read_text(encoding="utf-8"))
    require(rec.get("schema_version") == "c3_checkpoint_v3", "Invalid C3 checkpoint schema", CheckpointIntegrityError)
    require(rec.get("status") == "COMPLETED", "C3 checkpoint not COMPLETED", CheckpointIntegrityError)
    require(rec.get("task_id") == expected_task_id, "C3 checkpoint task_id mismatch", CheckpointIntegrityError)
    require(rec.get("execution_script_sha256") == script_sha, "C3 checkpoint script SHA mismatch", CheckpointIntegrityError)
    require(rec.get("manifest_sha256") == manifest_sha, "C3 checkpoint manifest SHA mismatch", CheckpointIntegrityError)
    require(rec.get("dataset_sha256") == dataset_sha, "C3 checkpoint dataset SHA mismatch", CheckpointIntegrityError)
    payload = rec.get("payload")
    require(isinstance(payload, dict) and rec.get("payload_sha256") == canonical_json_sha(payload),
            "C3 checkpoint payload_sha256 mismatch", CheckpointIntegrityError)
    comps = rec.get("companion_files")
    require(isinstance(comps, dict) and set(comps.keys()) == set(expected_companion_keys),
            "C3 checkpoint companion keys mismatch", CheckpointIntegrityError)
    out_root = out_dir.resolve()
    for key, info in comps.items():
        p = (out_root / info["rel_path"]).resolve()
        p.relative_to(out_root)
        require(p.is_file() and p.stat().st_size > 0 and sha_file(p) == info["sha256"],
                f"C3 companion file failed integrity ({key}): {p}", CheckpointIntegrityError)
    return payload


def run_overlap_null_from_manifest(
    config_json: Path, out_dir: Path, n_mc: int = 10000, seed: int = 20261122,
    require_pinned_hashes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config_json = Path(config_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    require(config_json.is_file(), f"Missing overlap manifest config: {config_json}", ProvenanceError)
    manifest_sha = sha_file(config_json)
    cfg = json.loads(config_json.read_text(encoding="utf-8"))
    validate_manifest_config(cfg, require_pinned_hashes=require_pinned_hashes)
    c1_univ_manifest = load_certified_c1_universe_hashes(cfg)

    pairwise_rows = []
    fourway_rows = []
    input_prov = {}

    for m_idx, entry in enumerate(cfg["methods"]):
        mname = str(entry["name"]).strip()
        role, scope = validate_role_and_scope(entry)
        enforce_no_historical_primary(mname, role)
        panels, universes = {}, {}
        exp_psizes = entry.get("expected_panel_sizes", {})
        exp_usizes = entry.get("expected_universe_sizes", {})
        input_prov[mname] = {"role": role, "scope": scope, "cohorts": {}}

        for c in COHORTS:
            p_spec = entry["panel_sources"][c]
            if isinstance(p_spec, str):
                p_spec = {"path": p_spec}
            genes, _, p_src, p_sha = _extract_panel_and_ranking(p_spec, c)
            u_set, u_src, u_sha = _extract_universe(entry["universe_csvs"][c], c, scope, c1_univ_manifest)
            p_set = set(genes)
            require(p_set.issubset(u_set), f"{mname} {c}: selected genes not subset of eligible universe", ProvenanceError)
            if c in exp_psizes:
                require(len(p_set) == int(exp_psizes[c]), f"{mname} {c}: panel size {len(p_set)} != {exp_psizes[c]}", ProvenanceError)
            if c in exp_usizes:
                require(len(u_set) == int(exp_usizes[c]), f"{mname} {c}: universe size {len(u_set)} != {exp_usizes[c]}", ProvenanceError)
            panels[c] = p_set
            universes[c] = u_set
            input_prov[mname]["cohorts"][c] = {
                "panel_source": p_src,
                "panel_sha256": p_sha,
                "panel_size": len(p_set),
                "universe_source": u_src,
                "universe_sha256": u_sha,
                "universe_size": len(u_set),
            }

        pair_idx = 0
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = COHORTS[i], COHORTS[j]
                res = monte_carlo_pairwise_null(
                    universes[a], universes[b], len(panels[a]), len(panels[b]),
                    len(panels[a] & panels[b]), n_mc=n_mc, seed=seed + m_idx * 100 + pair_idx,
                )
                pairwise_rows.append({
                    "method": mname, "role": role, "scope_caveat": scope,
                    "size_only_loco_disclosure": LOCO_CAVEAT_TEXT if scope == "LOCO_size_only_null_training_dependent" else "Independent cohorts (non-overlapping samples).",
                    "pair": f"{a}-{b}", **res,
                })
                pair_idx += 1

        obs_inter4 = sorted(set.intersection(*(panels[c] for c in COHORTS)))
        obs_union4 = set.union(*(panels[c] for c in COHORTS))
        fw = monte_carlo_fourway_null(
            [universes[c] for c in COHORTS], [len(panels[c]) for c in COHORTS],
            len(obs_inter4), n_mc=n_mc, seed=seed + m_idx * 100 + 99,
        )
        fourway_rows.append({
            "method": mname, "role": role, "scope_caveat": scope,
            "size_only_loco_disclosure": LOCO_CAVEAT_TEXT if scope == "LOCO_size_only_null_training_dependent" else "Independent cohorts (non-overlapping samples).",
            "obs_union_size": len(obs_union4), "obs_fourway_genes": ";".join(obs_inter4), **fw,
        })

    df_pair = pd.DataFrame(pairwise_rows)
    df_fw = pd.DataFrame(fourway_rows)
    atomic_write_csv(out_dir / "phase2_pairwise_overlap_null.csv", df_pair)
    atomic_write_csv(out_dir / "phase2_fourway_overlap_null.csv", df_fw)
    atomic_write_json(out_dir / "input_provenance.json", {
        "schema_version": "c2_input_provenance_v3",
        "manifest_path": str(config_json),
        "manifest_sha256": manifest_sha,
        "methods": input_prov,
    })
    return df_pair, df_fw, input_prov


def run_fixed_k_evaluation(
    config_json: Path, data_zip: Path, out_dir: Path, n_boot: int = 2000,
    require_pinned_hashes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute C3 Fixed-K Stage 22 evaluation with atomic per-task checkpoints."""
    from phase2_baselines_v3 import compute_cpm_and_log2, eval_stage22_panel, load_canonical_dataset

    config_json = Path(config_json)
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    pred_dir = out_dir / "fixed_k_predictions_PRIVATE"
    panel_dir = out_dir / "fixed_k_panels"
    failed_dir = out_dir / "failed_fits"
    for d in (ckpt_dir, pred_dir, panel_dir):
        d.mkdir(parents=True, exist_ok=True)

    script_sha = sha_file(Path(__file__))
    manifest_sha = sha_file(config_json)
    cfg = json.loads(config_json.read_text(encoding="utf-8"))
    validate_manifest_config(cfg, require_pinned_hashes=require_pinned_hashes)

    meta, counts, genes, data_prov = load_canonical_dataset(data_zip, verify_hashes=True)
    dataset_sha = data_prov["zip_sha256"]
    _, log2_all = compute_cpm_and_log2(counts)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    cohort_rows = []
    for entry in cfg["methods"]:
        mname = str(entry["name"]).strip()
        role, scope = validate_role_and_scope(entry)
        enforce_no_historical_primary(mname, role)
        if scope != "LOCO_size_only_null_training_dependent":
            continue

        if not entry.get("fixed_k_eligible", True):
            for c in COHORTS:
                for K in FIXED_K_GRID:
                    cohort_rows.append({
                        "method": mname, "role": role, "cohort": c, "K": int(K),
                        "status": "UNRANKED_CONSENSUS_EXCLUDED_BY_DESIGN",
                        "n_nonzero_available": None, "auroc": float("nan"), "bacc": float("nan"),
                        "bootstrap_reps": 0, "stage22_converged": None,
                    })
            continue

        safe_m = "".join(ch if ch.isalnum() else "_" for ch in mname).strip("_")
        for fold_idx, c in enumerate(COHORTS):
            p_spec = entry["panel_sources"][c]
            if isinstance(p_spec, str):
                p_spec = {"path": p_spec}
            sel_genes, ranked_genes, _, _ = _extract_panel_and_ranking(p_spec, c)
            require(ranked_genes is not None, f"Method '{mname}' missing score_col for ranking", SchemaValidationError)
            n_avail = len(sel_genes) if mname in RANKED_METHODS_SPARSE_NONZERO else len(ranked_genes)

            test_mask = (meta["cohort"] == c).to_numpy()
            train_mask = ~test_mask
            y_train = meta.loc[train_mask, "y"].to_numpy()
            y_test = meta.loc[test_mask, "y"].to_numpy()
            test_ids = meta.loc[test_mask, "ID"].tolist()

            for K in FIXED_K_GRID:
                is_valid, status_msg = validate_fixed_k_request(mname, K, n_nonzero_available=n_avail, role=role)
                if not is_valid:
                    cohort_rows.append({
                        "method": mname, "role": role, "cohort": c, "K": int(K),
                        "status": status_msg, "n_nonzero_available": int(n_avail),
                        "auroc": float("nan"), "bacc": float("nan"),
                        "bootstrap_reps": 0, "stage22_converged": None,
                    })
                    continue

                task_id = f"fixedK__{safe_m}__{c}__K{K}"
                ckpt_p = ckpt_dir / f"{task_id}.json"
                pred_p = pred_dir / f"pred_{safe_m}_{c}_K{K}_PRIVATE.csv"
                panel_p = panel_dir / f"panel_{safe_m}_{c}_K{K}.csv"
                cached = load_validated_c3_checkpoint(ckpt_p, task_id, script_sha, manifest_sha, dataset_sha, {"pred", "panel"}, out_dir)
                if cached is not None:
                    cohort_rows.append(cached)
                else:
                    top_k = ranked_genes[:K]
                    col_idx = np.array([gene_to_idx[g] for g in top_k], dtype=int)
                    row, pred_df = eval_stage22_panel(
                        log2_all[train_mask][:, col_idx], y_train,
                        log2_all[test_mask][:, col_idx], y_test,
                        test_ids, c, fold_idx, n_boot=n_boot, failed_dir=failed_dir, task_label=task_id,
                    )
                    atomic_write_csv(pred_p, pred_df)
                    atomic_write_csv(panel_p, pd.DataFrame({"rank": np.arange(1, K + 1), "gene": top_k}))
                    out_row = {
                        "method": mname, "role": role, "cohort": c, "K": int(K),
                        "status": status_msg, "n_nonzero_available": int(n_avail),
                        "auroc": row["auroc"], "auroc_ci_low": row["auroc_ci_low"], "auroc_ci_high": row["auroc_ci_high"],
                        "auprc": row["auprc"], "bacc": row["bacc"], "bacc_ci_low": row["bacc_ci_low"], "bacc_ci_high": row["bacc_ci_high"],
                        "f1": row["f1"], "brier": row["brier"], "bootstrap_reps": int(n_boot),
                        "stage22_converged": True, "stage22_n_iter": row["stage22_n_iter"],
                    }
                    save_validated_c3_checkpoint(ckpt_p, task_id, out_row, script_sha, manifest_sha, dataset_sha, {"pred": pred_p, "panel": panel_p}, out_dir)
                    cohort_rows.append(out_row)

    df_fk = pd.DataFrame(cohort_rows)
    atomic_write_csv(out_dir / "phase2_fixed_k_cohort_metrics.csv", df_fk)

    macro_rows = []
    for (mname, K), grp in df_fk.groupby(["method", "K"], sort=False):
        eval_grp = grp[grp["status"].astype(str).str.startswith("VALID_")]
        macro_rows.append({
            "method": mname, "K": int(K),
            "n_evaluable_folds": int(len(eval_grp)),
            "evaluable_cohorts": ";".join(eval_grp["cohort"].tolist()),
            "macro_auroc": float(eval_grp["auroc"].mean()) if len(eval_grp) > 0 else float("nan"),
            "macro_bacc": float(eval_grp["bacc"].mean()) if len(eval_grp) > 0 else float("nan"),
        })
    df_macro = pd.DataFrame(macro_rows)
    atomic_write_csv(out_dir / "phase2_fixed_k_macro_summary.csv", df_macro)
    return df_fk, df_macro


def finalize_c2_c3_outputs(
    out_dir: Path, config_json: Path, include_fixed_k: bool = True,
    n_mc: int = 10000, n_boot: int = 2000, seed: int = 20261122,
    require_pinned_hashes: bool = True,
) -> dict:
    """Hardened fail-closed finalizer rejecting fabricated provenance, incomplete fixed-K, missing checkpoints, or metric mismatches."""
    out_dir = Path(out_dir)
    config_json = Path(config_json)
    require(config_json.is_file(), f"Missing manifest config: {config_json}", CertificationError)
    actual_script_sha = sha_file(Path(__file__))
    manifest_sha = sha_file(config_json)
    cfg = json.loads(config_json.read_text(encoding="utf-8"))
    try:
        validate_manifest_config(cfg, require_pinned_hashes=require_pinned_hashes)
    except Exception as exc:
        raise CertificationError(f"Manifest validation failed: {exc}") from exc

    # 1. Validate input_provenance.json strictly (rejects {'FAKE': True}!)
    prov_p = out_dir / "input_provenance.json"
    require(prov_p.is_file(), "Missing input_provenance.json", CertificationError)
    prov = json.loads(prov_p.read_text(encoding="utf-8"))
    require(prov.get("schema_version") == "c2_input_provenance_v3", "Invalid/fabricated input_provenance.json schema_version", CertificationError)
    require(prov.get("manifest_sha256") == manifest_sha, "input_provenance manifest_sha256 mismatch", CertificationError)
    prov_methods = prov.get("methods")
    expected_method_names = [str(e["name"]).strip() for e in cfg["methods"]]
    require(isinstance(prov_methods, dict) and set(prov_methods.keys()) == set(expected_method_names),
            "input_provenance methods mismatch vs manifest", CertificationError)
    for entry in cfg["methods"]:
        mname = str(entry["name"]).strip()
        pm = prov_methods[mname]
        require(isinstance(pm.get("cohorts"), dict) and set(pm["cohorts"].keys()) == set(COHORTS),
                f"input_provenance missing cohorts for {mname}", CertificationError)
        for c in COHORTS:
            cm = pm["cohorts"][c]
            require(is_sha256_hex(cm.get("panel_sha256")) and is_sha256_hex(cm.get("universe_sha256")),
                    f"Missing/invalid SHA-256 in input_provenance for {mname}/{c}", CertificationError)
            ps = entry["panel_sources"][c]
            if isinstance(ps, dict) and ps.get("expected_sha256"):
                require(cm["panel_sha256"] == ps["expected_sha256"], f"Pinned panel SHA mismatch in provenance for {mname}/{c}", CertificationError)

    # 2. Validate pairwise & 4-way overlap CSVs (full schemas + exact inventory + n_mc + 95% null intervals)
    pair_p = out_dir / "phase2_pairwise_overlap_null.csv"
    fw_p = out_dir / "phase2_fourway_overlap_null.csv"
    require(pair_p.is_file() and fw_p.is_file(), "Missing C2 pairwise or 4-way overlap CSV", CertificationError)
    df_pair = pd.read_csv(pair_p)
    df_fw = pd.read_csv(fw_p)

    req_pair_cols = {
        "method", "role", "scope_caveat", "size_only_loco_disclosure", "pair",
        "exact_expected_intersection", "mc_mean_intersection", "mc_inter_ci_low", "mc_inter_ci_high",
        "obs_jaccard", "mc_mean_jaccard", "mc_jaccard_ci_low", "mc_jaccard_ci_high",
        "obs_overlap_coeff", "mc_mean_overlap_coeff", "mc_overlap_ci_low", "mc_overlap_ci_high",
        "mc_emp_p_ge_obs", "n_mc", "seed",
    }
    require(req_pair_cols.issubset(df_pair.columns), f"Pairwise CSV missing required columns: {req_pair_cols - set(df_pair.columns)}", CertificationError)
    exp_pair_keys = {(m, p) for m in expected_method_names for p in PAIRS}
    obs_pair_keys = {(str(r["method"]), str(r["pair"])) for _, r in df_pair.iterrows()}
    require(obs_pair_keys == exp_pair_keys and len(df_pair) == len(exp_pair_keys), "Pairwise inventory incomplete/mismatched", CertificationError)
    require((df_pair["n_mc"] == int(n_mc)).all(), f"Pairwise n_mc mismatch (expected {n_mc})", CertificationError)
    for col in ("exact_expected_intersection", "mc_jaccard_ci_low", "mc_jaccard_ci_high", "mc_overlap_ci_low", "mc_overlap_ci_high"):
        require(np.isfinite(df_pair[col]).all(), f"Non-finite values in pairwise:{col}", CertificationError)

    req_fw_cols = {
        "method", "role", "scope_caveat", "size_only_loco_disclosure", "obs_union_size",
        "obs_fourway_intersection", "exact_expected_fourway", "mc_mean_fourway",
        "mc_fourway_ci_low", "mc_fourway_ci_high", "mc_p_ge_1", "mc_emp_p_ge_obs", "n_mc", "seed",
    }
    require(req_fw_cols.issubset(df_fw.columns), f"4-way CSV missing required columns: {req_fw_cols - set(df_fw.columns)}", CertificationError)
    require(set(df_fw["method"].astype(str)) == set(expected_method_names) and len(df_fw) == len(expected_method_names),
            "4-way method inventory incomplete/mismatched", CertificationError)
    require((df_fw["n_mc"] == int(n_mc)).all(), f"4-way n_mc mismatch (expected {n_mc})", CertificationError)
    require(np.isfinite(df_fw["exact_expected_fourway"]).all() and np.isfinite(df_fw["mc_fourway_ci_low"]).all(),
            "Non-finite values in 4-way CSV", CertificationError)

    # 3. Validate C3 Fixed-K full inventory, checkpoints, canonical patient IDs/labels & metric recomputation
    fk_rows = 0
    verified_c3_ckpts = 0
    if include_fixed_k:
        fk_p = out_dir / "phase2_fixed_k_cohort_metrics.csv"
        fkm_p = out_dir / "phase2_fixed_k_macro_summary.csv"
        require(fk_p.is_file() and fkm_p.is_file(), "Missing C3 fixed-K output CSVs", CertificationError)
        df_fk = pd.read_csv(fk_p)
        df_fkm = pd.read_csv(fkm_p)
        req_fk_cols = {"method", "role", "cohort", "K", "status", "auroc", "bacc", "bootstrap_reps", "stage22_converged"}
        require(req_fk_cols.issubset(df_fk.columns), f"Fixed-K CSV missing columns: {req_fk_cols - set(df_fk.columns)}", CertificationError)

        loco_methods = [str(e["name"]).strip() for e in cfg["methods"] if e["scope"] == "LOCO_size_only_null_training_dependent"]
        exp_fk_keys = {(m, c, K) for m in loco_methods for c in COHORTS for K in FIXED_K_GRID}
        obs_fk_keys = {(str(r["method"]), str(r["cohort"]), int(r["K"])) for _, r in df_fk.iterrows()}
        require(obs_fk_keys == exp_fk_keys and len(df_fk) == len(exp_fk_keys),
                f"Incomplete/fabricated fixed-K cohort inventory: got {len(df_fk)}, expected {len(exp_fk_keys)}", CertificationError)

        exp_macro_keys = {(m, K) for m in loco_methods for K in FIXED_K_GRID}
        require({"method", "K", "n_evaluable_folds", "macro_auroc", "macro_bacc"}.issubset(df_fkm.columns),
                "Fixed-K macro summary missing required columns", CertificationError)
        obs_macro_keys = {(str(r["method"]), int(r["K"])) for _, r in df_fkm.iterrows()}
        require(obs_macro_keys == exp_macro_keys and len(df_fkm) == len(exp_macro_keys),
                f"Incomplete fixed-K macro summary: got {len(df_fkm)}, expected {len(exp_macro_keys)}", CertificationError)

        ckpt_dir = out_dir / "checkpoints"
        pred_dir = out_dir / "fixed_k_predictions_PRIVATE"
        panel_dir = out_dir / "fixed_k_panels"
        for _, r in df_fk.iterrows():
            if not str(r["status"]).startswith("VALID_"):
                continue
            m, c, K = str(r["method"]), str(r["cohort"]), int(r["K"])
            safe_m = "".join(ch if ch.isalnum() else "_" for ch in m).strip("_")
            task_id = f"fixedK__{safe_m}__{c}__K{K}"
            ckpt_p = ckpt_dir / f"{task_id}.json"
            require(ckpt_p.is_file(), f"Missing C3 checkpoint for valid fixed-K row: {task_id}", CertificationError)
            try:
                load_validated_c3_checkpoint(ckpt_p, task_id, actual_script_sha, manifest_sha, EXPECTED_ZIP_SHA, {"pred", "panel"}, out_dir)
            except CheckpointIntegrityError as exc:
                raise CertificationError(f"C3 checkpoint integrity failed for {task_id}: {exc}") from exc
            verified_c3_ckpts += 1

            panel_p = panel_dir / f"panel_{safe_m}_{c}_K{K}.csv"
            pnl = pd.read_csv(panel_p)
            require("gene" in pnl.columns and len(pnl) == K and pnl["gene"].is_unique, f"Invalid fixed-K panel {panel_p.name}", CertificationError)

            pred_p = pred_dir / f"pred_{safe_m}_{c}_K{K}_PRIVATE.csv"
            pdf = pd.read_csv(pred_p)
            require(len(pdf) == EXPECTED_COUNTS[c] and compute_prediction_label_sha(pdf) == CANONICAL_COHORT_LABEL_SHA256[c],
                    f"Canonical patient ID/label check failed for {pred_p.name}", CertificationError)
            recomp_auc = compute_auroc_exact(pdf["y_true"].to_numpy(), pdf["prob_CD"].to_numpy())
            recomp_bacc = compute_bacc_exact(pdf["y_true"].to_numpy(), pdf["pred_CD"].to_numpy())
            require(abs(recomp_auc - float(r["auroc"])) <= 1e-10 and abs(recomp_bacc - float(r["bacc"])) <= 1e-10,
                    f"Recomputed fixed-K metrics mismatch for {task_id}", CertificationError)
            require(int(r["bootstrap_reps"]) >= int(n_boot) and bool(r["stage22_converged"]),
                    f"Fixed-K bootstrap_reps or convergence gate failed for {task_id}", CertificationError)
        fk_rows = len(df_fk)

    status = "CERTIFIED_COMPLETE" if include_fixed_k else "C2_OVERLAP_ONLY_COMPLETE"
    manifest = {
        "status": status,
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "script_name": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "execution_script_sha256": actual_script_sha,
        "manifest_sha256": manifest_sha,
        "dataset_sha256": EXPECTED_ZIP_SHA if include_fixed_k else None,
        "n_methods": len(expected_method_names),
        "n_pairwise_rows": int(len(df_pair)),
        "n_fourway_rows": int(len(df_fw)),
        "include_fixed_k": bool(include_fixed_k),
        "n_fixed_k_rows": int(fk_rows),
        "n_verified_c3_checkpoints": int(verified_c3_ckpts),
        "n_mc": int(n_mc),
        "n_boot": int(n_boot) if include_fixed_k else 0,
        "seed": int(seed),
        "pairwise_formula": "E[|A cap B|] = k_a * k_b * |U_a cap U_b| / (|U_a| * |U_b|)",
        "fourway_formula": "E[|A1 cap A2 cap A3 cap A4|] = |U1 cap U2 cap U3 cap U4| * prod_i (k_i / |U_i|)",
        "loco_caveat": LOCO_CAVEAT_TEXT,
    }
    atomic_write_json(out_dir / "manifest.json", manifest)

    lines = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt" and not p.name.endswith(".tmp"):
            rel = p.relative_to(out_dir).as_posix()
            lines.append(f"{sha_file(p)}  {rel}")
    atomic_write_text(out_dir / "SHA256SUMS.txt", "\n".join(lines) + "\n")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CMPS 355 Phase 2 Overlap Null (C2) & Fixed-K (C3) v3")
    ap.add_argument("--config", type=Path, required=True, help="Path to overlap_manifest_config_v3.json")
    ap.add_argument("--data", type=Path, default=None, help="Path to immutable ibd_dataset_sample_aligned.zip")
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument("--n-mc", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--skip-fixed-k", action="store_true", help="Run C2 overlap null only (status: C2_OVERLAP_ONLY_COMPLETE)")
    ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()

    if args.finalize_only:
        print(json.dumps(finalize_c2_c3_outputs(
            args.out, args.config, include_fixed_k=not args.skip_fixed_k,
            n_mc=args.n_mc, n_boot=args.n_boot, seed=args.seed,
        ), indent=2))
    else:
        run_overlap_null_from_manifest(args.config, args.out, n_mc=args.n_mc, seed=args.seed)
        if not args.skip_fixed_k:
            require(args.data is not None, "--data (path to ibd_dataset_sample_aligned.zip) is required unless --skip-fixed-k is set")
            run_fixed_k_evaluation(args.config, args.data, args.out, n_boot=args.n_boot)
        print(json.dumps(finalize_c2_c3_outputs(
            args.out, args.config, include_fixed_k=not args.skip_fixed_k,
            n_mc=args.n_mc, n_boot=args.n_boot, seed=args.seed,
        ), indent=2))

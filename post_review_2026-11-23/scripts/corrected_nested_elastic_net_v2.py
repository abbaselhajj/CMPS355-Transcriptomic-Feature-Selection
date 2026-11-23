#!/usr/bin/env python3
"""
CMPS 355 — Production Corrected Nested Elastic Net v2 (Fail-Closed & Certified)
===============================================================================
Version: v2.0 (2026-11-22) — incorporates all 4 P0 release-blocking fixes and
all 7 operational safeguards from ChatGPT's preflight code review.

Key v2 hardening guarantees:
  - P0-1 (Fail-closed numerical validation): Any fit failing its KKT gate
    (< 1e-6 for 8 final models; < 1e-4 for 192 inner-CV fits) is persisted with
    `status="failed_numerical_gate"` (NEVER `"completed"`), excluded from tuning,
    and raises `NumericalGateError`. `select_best_hyperparams` asserts exact
    complete grid coverage and 100% gate pass before choosing `(best_C, best_r)`.
  - P0-2 (Validated checkpoint resume): Reusing a checkpoint requires exact match
    on `status=="completed"`, `numerical_gate_passed==True`, `C`, `l1_ratio`,
    `kkt_gate`, `backend`, `config_fingerprint`, `script_sha256`,
    `dataset_zip_sha256`, `train_ids_sha256`, `val_ids_sha256`,
    `elig_genes_sha256`, finite metrics, and (for final models) live SHA-256
    match of saved `full_coefficients` and `panels` CSVs. Mismatches raise
    `CheckpointIntegrityError`.
  - P0-3 (Strict completion certification): `finalize_outputs` explicitly audits
    all 200 unique canonical fits, 8 final coefficient/panel artifacts, 689
    Stage 22 predictions, and all required CSVs. Sets `certification_status` to
    `CERTIFIED_COMPLETE` only when 100% of checks pass, and `INCOMPLETE_OR_FAILED`
    otherwise; raises `CompletionCertificationError` in full/strict mode.
  - P0-4 (Exact glmnet lambda & solver verification): Asserts `jerr == 0`,
    `|lambda_fitted - 1/(C*n_train)| <= max(1e-12, 1e-9 * lambda_target)`,
    length-p finite coefficients, finite intercept & probabilities, captures R
    warnings/stderr, and enforces configurable per-fit `--fit-timeout-sec`.
  - Operational safeguards: immutable execution provenance in `manifest.json`
    (separate `finalizer_script_sha256`), safe empty-panel handling in Stage 22,
    sparse non-zero coefficient archival inside inner-CV checkpoints, uniquely
    timestamped smoke-test folders, and large-LOCO stress test in `--smoke-test`.
"""

from __future__ import annotations

import argparse
import csv
import errno as _errno
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve as scipy_solve
from scipy.special import expit

SCRIPT_VERSION = "v2.0"

COHORTS = ("d1", "d2", "d3", "d4")
EXPECTED_ROWS = {"d1": 260, "d2": 179, "d3": 76, "d4": 174}
EXPECTED_ZIP_SHA = "8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74"
EXPECTED_CSV_SHA = {
    "count_d1_aligned.csv": "0f595746ea572a349580d9d3c0242559345b88c927c161b1982933102e6c812d",
    "count_d2_aligned.csv": "2ff413b813e3e4d10927da7f8342eccb35a00205c319c0eb714388aea97f245d",
    "count_d3_aligned.csv": "33b2e51e682da4c41f53becf00b1fc88bb5710293ce2d9d87837bca86a7b3d48",
    "count_d4_aligned.csv": "c26d8db65b335c89967bd4e072c61a6b7adcb57715c5fe36107d9bd8cd9409e0",
    "desc_d1.csv": "91aadec65bb7f2c1945e7941222b9f490a0c9041d495b8d22606f614484c7b33",
    "desc_d2.csv": "4198f574167754eb00e007ded8cfc2528d50ead0d80b96c1a5cc8139131a140b",
    "desc_d3.csv": "9db3c39975a50afcbd851d3cdcdf56f1e59e7f9bbe18b3b80fd7328400a31332",
    "desc_d4.csv": "bdd5f6d5ae9a6713470f1eadab2e62a961a38d417f27401359d55b5d906a1193",
}

C_GRID = (0.001, 0.01, 0.1, 1.0)
ELASTIC_RATIOS = (0.5, 0.9)
MIN_CPM = 1.0
MIN_FRACTION = 0.10
COEF_CUTOFF = 1e-8

SEED = 355
INDEP_CV_SEED = 402  # 355 + 47, exact parity with corrected_nested_lasso.py
STAGE22_SEED = 123

FINAL_KKT_GATE = 1e-6
INNER_KKT_GATE = 1e-4
DEFAULT_FIT_TIMEOUT_SEC = 300

GOOGLE_WORKSPACE_VIRTUAL_SUFFIXES = {
    ".gdoc", ".gsheet", ".gslides", ".gdraw", ".gform", ".gmap", ".gsite", ".gjam",
}


class NumericalGateError(RuntimeError):
    """Raised when a fit fails its required KKT or finiteness certification gate."""


class CheckpointIntegrityError(RuntimeError):
    """Raised when an existing checkpoint fails identity, config, gate, or artifact hash validation."""


class SolverOutputError(RuntimeError):
    """Raised when a solver returns invalid lambda, non-zero jerr, non-finite values, or times out."""


class CompletionCertificationError(RuntimeError):
    """Raised when strict finalization detects missing, incomplete, or failed benchmark outputs."""


def require(cond: bool, msg: str, exc_cls=RuntimeError) -> None:
    if not cond:
        raise exc_cls(msg)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_str_list(items) -> str:
    payload = "\n".join(str(x) for x in items).encode("utf-8")
    return sha_bytes(payload)


def get_script_sha256() -> str:
    return sha_bytes(Path(__file__).read_bytes())


def build_config_fingerprint(backend: str, dataset_zip_sha: str | None, script_sha: str) -> str:
    cfg = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_sha,
        "dataset_zip_sha256": dataset_zip_sha or "unverified",
        "backend": backend,
        "c_grid": list(C_GRID),
        "elastic_ratios": list(ELASTIC_RATIOS),
        "min_cpm": MIN_CPM,
        "min_fraction": MIN_FRACTION,
        "coef_cutoff": COEF_CUTOFF,
        "indep_cv_seed": INDEP_CV_SEED,
        "stage22_seed": STAGE22_SEED,
        "final_kkt_gate": FINAL_KKT_GATE,
        "inner_kkt_gate": INNER_KKT_GATE,
    }
    return sha_bytes(json.dumps(cfg, sort_keys=True).encode("utf-8"))


def expected_canonical_fit_ids() -> dict[str, str]:
    """Return mapping of all 200 canonical fit_ids -> role ('inner' or 'final')."""
    ids: dict[str, str] = {}
    for outer_c in COHORTS:
        for inner_val in [c for c in COHORTS if c != outer_c]:
            for C_val in C_GRID:
                for r_val in ELASTIC_RATIOS:
                    fid = f"loco_inner_{outer_c}_val_{inner_val}_C{C_val}_r{r_val}".replace(".", "p")
                    ids[fid] = "inner"
        ids[f"loco_{outer_c}_corrected_enet"] = "final"
    for coh in COHORTS:
        for f_idx in (1, 2, 3):
            for C_val in C_GRID:
                for r_val in ELASTIC_RATIOS:
                    fid = f"indep_cv_{coh}_f{f_idx}_C{C_val}_r{r_val}".replace(".", "p")
                    ids[fid] = "inner"
        ids[f"indep_{coh}_corrected_enet"] = "final"
    assert len(ids) == 200
    return ids


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_sha256sums_safe(out_dir: Path) -> tuple[list[str], list[str]]:
    """Hash ordinary readable files under out_dir, excluding Workspace stubs, live notebooks, and run.log."""
    sums: list[str] = []
    excluded: list[str] = []
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file() or p.name in ("SHA256SUMS.txt", "SHA256SUMS_EXCLUDED.txt"):
            continue
        if p.name.endswith(".tmp"):
            continue
        rel = p.relative_to(out_dir).as_posix()
        suf = p.suffix.lower()
        if suf in GOOGLE_WORKSPACE_VIRTUAL_SUFFIXES:
            excluded.append(f"{rel} (excluded: Google Workspace virtual placeholder)")
            continue
        if suf == ".ipynb":
            excluded.append(f"{rel} (excluded: live Colab notebook subject to post-run auto-save)")
            continue
        if p.name == "run.log":
            excluded.append(f"{rel} (excluded: live stdout tee log flushed after script exit)")
            continue
        try:
            digest = sha_bytes(p.read_bytes())
        except OSError as exc:
            if exc.errno in (_errno.EOPNOTSUPP, _errno.ENOTSUP, _errno.EINVAL):
                excluded.append(f"{rel} (excluded: unreadable virtual file, errno={exc.errno})")
                continue
            raise
        sums.append(f"{digest}  {rel}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + ("\n" if sums else ""), encoding="utf-8")
    if excluded:
        (out_dir / "SHA256SUMS_EXCLUDED.txt").write_text("\n".join(excluded) + "\n", encoding="utf-8")
    return sums, excluded


def load_canonical_dataset(source_path: Path, verify_hashes: bool = True):
    """Read-only canonical dataset loader with exact SHA-256 verification."""
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        zbytes = source_path.read_bytes()
        zsha = sha_bytes(zbytes)
        if verify_hashes:
            require(zsha == EXPECTED_ZIP_SHA, f"ZIP SHA-256 mismatch: {zsha}")
        with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
            names_map = {Path(n).name: n for n in zf.namelist()}
            raw = {}
            for name in EXPECTED_CSV_SHA:
                require(name in names_map, f"Missing {name} inside ZIP")
                raw[name] = zf.read(names_map[name])
        mode = "archive"
    else:
        zsha = None
        raw = {name: (source_path / name).read_bytes() for name in EXPECTED_CSV_SHA}
        mode = "directory"

    csv_hashes = {}
    for name, expected in EXPECTED_CSV_SHA.items():
        got = sha_bytes(raw[name])
        csv_hashes[name] = got
        if verify_hashes:
            require(got == expected, f"CSV SHA-256 mismatch for {name}: got {got}")

    headers = {}
    for c in COHORTS:
        names = next(csv.reader(io.StringIO(raw[f"count_{c}_aligned.csv"].decode("utf-8-sig"))))
        require(len(names) == len(set(names)) and names[0] == "ID", f"Invalid headers: {c}")
        headers[c] = names

    genes = tuple(sorted(set.intersection(*(set(headers[c][1:]) for c in COHORTS))))
    if verify_hashes:
        require(len(genes) == 9983, f"Expected 9983 shared genes, got {len(genes)}")

    counts, metadata = [], []
    for c in COHORTS:
        desc = pd.read_csv(io.BytesIO(raw[f"desc_{c}.csv"]), dtype={"ID": "string", "Study": "string", "class": "string"})
        df = pd.read_csv(io.BytesIO(raw[f"count_{c}_aligned.csv"]), usecols=["ID", *genes], dtype={"ID": "string"})
        if verify_hashes:
            require(len(desc) == len(df) == EXPECTED_ROWS[c], f"{c}: sample count mismatch")
        require(desc["ID"].tolist() == df["ID"].tolist(), f"{c}: sample-ID alignment mismatch")
        require(set(desc["class"]) == {"CD", "Not IBD"}, f"{c}: unexpected classes")
        mat = df.loc[:, list(genes)].to_numpy(dtype=np.float64, copy=True)
        require(np.isfinite(mat).all() and (mat >= 0).all() and (mat.sum(axis=1) > 0).all(), f"{c}: invalid counts")
        desc = desc[["ID", "Study", "class"]].copy()
        desc.insert(0, "cohort", c)
        metadata.append(desc)
        counts.append(mat)

    meta = pd.concat(metadata, ignore_index=True)
    require(meta["ID"].is_unique, "Duplicate sample IDs")
    if verify_hashes:
        require(len(meta) == 689 and meta["Study"].nunique() == 6, "Expected 689 samples across 6 BioProjects")
    return meta, np.vstack(counts), genes, {
        "source_mode": mode,
        "source_path": str(source_path),
        "zip_sha256": zsha,
        "csv_sha256": csv_hashes,
        "hashes_verified": verify_hashes,
    }


def log_cpm(values: np.ndarray) -> np.ndarray:
    depths = values.sum(axis=1)
    require(np.all(depths > 0), "Empty library detected")
    return np.log2(1.0 + 1e6 * values / depths[:, None])


def fit_features_canonical(train_counts: np.ndarray):
    x = log_cpm(train_counts)
    threshold = max(2, int(np.ceil(MIN_FRACTION * len(x))))
    valid = ((x > np.log2(1.0 + MIN_CPM)).sum(axis=0) >= threshold) & (x.var(axis=0) > 1e-12)
    indices = np.flatnonzero(valid)
    require(len(indices) >= 2, "Insufficient eligible genes")
    sub = x[:, indices]
    mean = sub.mean(axis=0)
    scale = sub.std(axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    fitted = (sub - mean) / scale
    require(np.isfinite(fitted).all(), "Non-finite scaled features")
    return fitted, indices, mean, scale


def apply_features_canonical(counts: np.ndarray, indices: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    sub = log_cpm(counts)[:, indices]
    return (sub - mean) / scale


def balanced_weights(y: np.ndarray) -> np.ndarray:
    n = len(y)
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    require(n0 > 0 and n1 > 0, f"Training partition missing a class (n0={n0}, n1={n1})")
    return np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0)).astype(np.float64)


def exact_en_logistic_kkt(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    intercept: float,
    C: float,
    l1_ratio: float,
) -> dict:
    """Exact common unpenalized-intercept Elastic Net binary logistic objective and full-gradient KKT."""
    w = balanced_weights(y)
    y_pm = np.where(y == 1, 1.0, -1.0)
    yz = y_pm * (X @ beta + intercept)
    data_loss = float(np.sum(w * np.logaddexp(0.0, -yz)))
    lam1 = float(l1_ratio) / float(C)
    lam2 = (1.0 - float(l1_ratio)) / float(C)
    l1_pen = float(lam1 * np.sum(np.abs(beta)))
    l2_pen = float(0.5 * lam2 * np.sum(beta ** 2))
    obj = data_loss + l1_pen + l2_pen

    sig_neg = 1.0 / (1.0 + np.exp(np.clip(yz, -500.0, 500.0)))
    r = -w * y_pm * sig_neg
    grad_beta = X.T @ r + lam2 * beta
    grad_b = float(np.sum(r))

    active = beta != 0.0
    inactive = ~active
    kkt_active = (
        float(np.max(np.abs(grad_beta[active] + lam1 * np.sign(beta[active]))))
        if np.any(active)
        else 0.0
    )
    kkt_inactive = (
        float(np.max(np.maximum(0.0, np.abs(grad_beta[inactive]) - lam1)))
        if np.any(inactive)
        else 0.0
    )
    kkt_intercept = abs(grad_b)
    return {
        "common_objective": obj,
        "data_loss": data_loss,
        "l1_penalty": l1_pen,
        "l2_penalty": l2_pen,
        "max_kkt_residual": max(kkt_active, kkt_inactive, kkt_intercept),
        "kkt_active": kkt_active,
        "kkt_inactive": kkt_inactive,
        "kkt_intercept": kkt_intercept,
    }


def polish_active_set_newton(
    X: np.ndarray,
    y: np.ndarray,
    beta_init: np.ndarray,
    b0_init: float,
    C: float,
    l1_ratio: float,
    kkt_target: float = 1e-10,
    max_iter: int = 25,
) -> dict:
    """Production active-set Newton polish with full-gradient KKT verification over all p genes."""
    t0 = time.perf_counter()
    orig_kkt = exact_en_logistic_kkt(X, y, beta_init, b0_init, C, l1_ratio)
    w = balanced_weights(y)
    y_bin = y.astype(np.float64)
    n = len(y)
    lam1 = float(l1_ratio) / float(C)
    lam2 = (1.0 - float(l1_ratio)) / float(C)

    on = np.flatnonzero(beta_init != 0.0)
    m = len(on)

    if m == 0:
        b0 = float(b0_init)
        status = "converged_intercept_only"
        n_iter = 0
        for k in range(1, max_iter + 1):
            p_vec = expit(np.full(n, b0))
            g0 = float(np.sum(w * (p_vec - y_bin)))
            if abs(g0) < kkt_target:
                break
            h0 = float(np.sum(w * p_vec * (1.0 - p_vec)))
            if h0 <= 1e-15:
                status = "hessian_zero_intercept"
                break
            b0 -= g0 / h0
            n_iter = k
        pol_kkt = exact_en_logistic_kkt(X, y, beta_init, b0, C, l1_ratio)
        accepted = (
            np.isfinite(pol_kkt["common_objective"])
            and pol_kkt["common_objective"] <= orig_kkt["common_objective"] + 1e-12 * max(1.0, abs(orig_kkt["common_objective"]))
            and pol_kkt["max_kkt_residual"] < orig_kkt["max_kkt_residual"]
        )
        return {
            "beta": beta_init.copy(),
            "intercept": b0 if accepted else float(b0_init),
            "kkt": pol_kkt if accepted else orig_kkt,
            "polish_accepted": bool(accepted),
            "polish_status": status,
            "polish_iters": n_iter,
            "polish_sec": time.perf_counter() - t0,
            "orig_kkt": orig_kkt["max_kkt_residual"],
            "polished_kkt_raw": pol_kkt["max_kkt_residual"],
            "sign_change_blocked": False,
        }

    Xs = X[:, on]
    Xaug = np.column_stack((Xs, np.ones(n, dtype=np.float64)))
    theta = np.r_[beta_init[on], float(b0_init)].astype(np.float64)
    sgn = np.sign(theta[:-1])

    def eval_active(t_vec: np.ndarray):
        eta = Xs @ t_vec[:-1] + t_vec[-1]
        p_vec = expit(eta)
        res = w * (p_vec - y_bin)
        F = (
            float(np.sum(w * (np.logaddexp(0.0, eta) - y_bin * eta)))
            + lam1 * float(np.sum(np.abs(t_vec[:-1])))
            + 0.5 * lam2 * float(np.dot(t_vec[:-1], t_vec[:-1]))
        )
        g = np.r_[Xs.T @ res + lam2 * t_vec[:-1] + lam1 * sgn, float(np.sum(res))]
        h = np.maximum(w * p_vec * (1.0 - p_vec), 1e-12)
        H = Xaug.T @ (h[:, None] * Xaug)
        H[np.arange(m), np.arange(m)] += lam2
        return F, g, H

    status = "max_iter"
    n_iter = 0
    sign_blocked = False

    for k in range(1, max_iter + 1):
        F, g, H = eval_active(theta)
        if float(np.max(np.abs(g))) < kkt_target:
            status = "converged"
            break
        d = None
        for damp in (0.0, 1e-10, 1e-8, 1e-6):
            try:
                H_use = H if damp == 0.0 else (H + damp * np.eye(m + 1))
                d = scipy_solve(H_use, -g, assume_a="pos")
                if np.all(np.isfinite(d)):
                    break
                d = None
            except Exception:
                d = None
        if d is None:
            status = "solve_failed"
            break

        gd = float(np.dot(g, d))
        if gd >= 0.0:
            status = "non_descent"
            break

        step = 1.0
        step_ok = False
        for _ in range(40):
            cand = theta + step * d
            if not np.all(np.sign(cand[:-1]) == sgn):
                sign_blocked = True
                step *= 0.5
                continue
            F_cand, _, _ = eval_active(cand)
            if F_cand <= F + 1e-4 * step * gd + 1e-13 * max(1.0, abs(F)):
                theta = cand
                step_ok = True
                break
            step *= 0.5
        n_iter = k
        if not step_ok:
            status = "line_search_failed"
            break

    beta_pol = np.zeros_like(beta_init)
    beta_pol[on] = theta[:-1]
    b0_pol = float(theta[-1])
    pol_kkt = exact_en_logistic_kkt(X, y, beta_pol, b0_pol, C, l1_ratio)

    accepted = (
        np.isfinite(pol_kkt["common_objective"])
        and pol_kkt["common_objective"] <= orig_kkt["common_objective"] + 1e-12 * max(1.0, abs(orig_kkt["common_objective"]))
        and pol_kkt["max_kkt_residual"] < orig_kkt["max_kkt_residual"]
    )
    return {
        "beta": beta_pol if accepted else beta_init.copy(),
        "intercept": b0_pol if accepted else float(b0_init),
        "kkt": pol_kkt if accepted else orig_kkt,
        "polish_accepted": bool(accepted),
        "polish_status": status,
        "polish_iters": n_iter,
        "polish_sec": time.perf_counter() - t0,
        "orig_kkt": orig_kkt["max_kkt_residual"],
        "polished_kkt_raw": pol_kkt["max_kkt_residual"],
        "sign_change_blocked": bool(sign_blocked),
    }


def _soft_threshold(z: float, thresh: float) -> float:
    if z > thresh:
        return z - thresh
    if z < -thresh:
        return z + thresh
    return 0.0


def solve_python_prox_newton_cd(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    l1_ratio: float,
    kkt_tol: float = 1e-7,
    max_outer: int = 120,
    max_inner_sweeps: int = 200,
) -> dict:
    """Pure-NumPy solver used for offline unit/negative testing (--backend python-cd)."""
    t0 = time.perf_counter()
    n, p = X.shape
    w = balanced_weights(y)
    y_bin = y.astype(np.float64)
    lam1 = float(l1_ratio) / float(C)
    lam2 = (1.0 - float(l1_ratio)) / float(C)

    b0 = 0.0
    beta = np.zeros(p, dtype=np.float64)
    X_T = np.ascontiguousarray(X.T)
    eta = np.zeros(n, dtype=np.float64)
    cur_obj = float(np.sum(w * np.logaddexp(0.0, eta)))
    converged = False
    outer_used = 0

    for outer in range(1, max_outer + 1):
        outer_used = outer
        prob = expit(eta)
        g_eta = w * (prob - y_bin)
        grad_beta = X_T @ g_eta + lam2 * beta
        grad_b0 = float(np.sum(g_eta))

        active_mask = beta != 0.0
        inactive_mask = ~active_mask
        kkt_act = float(np.max(np.abs(grad_beta[active_mask] + lam1 * np.sign(beta[active_mask])))) if np.any(active_mask) else 0.0
        kkt_inact = float(np.max(np.maximum(0.0, np.abs(grad_beta[inactive_mask]) - lam1))) if np.any(inactive_mask) else 0.0
        max_kkt = max(kkt_act, kkt_inact, abs(grad_b0))
        if max_kkt <= kkt_tol:
            converged = True
            break

        viol_scores = np.where(active_mask, np.inf, np.abs(grad_beta) - lam1)
        new_inact = np.flatnonzero((~active_mask) & (viol_scores > 0.0))
        if len(new_inact) > 256:
            top_idx = np.argpartition(viol_scores[new_inact], -256)[-256:]
            new_inact = new_inact[top_idx]
        work_idx = np.sort(np.union1d(np.flatnonzero(active_mask), new_inact))

        h = np.maximum(w * prob * (1.0 - prob), 1e-8)
        sum_h = float(np.sum(h))
        Xw = X[:, work_idx]
        m = len(work_idx)
        H_diag = (Xw * Xw).T @ h + lam2
        d_beta = np.zeros(m, dtype=np.float64)
        d_b0 = 0.0
        r_vec = g_eta.copy()
        beta_w = beta[work_idx]
        sub_tol = max(1e-11, 0.01 * max_kkt)

        for _ in range(max_inner_sweeps):
            max_step = 0.0
            delta_b0 = -float(np.sum(r_vec)) / sum_h
            if delta_b0 != 0.0:
                d_b0 += delta_b0
                r_vec += h * delta_b0
                max_step = max(max_step, abs(delta_b0))
            for k in range(m):
                xk = Xw[:, k]
                cur_val = beta_w[k] + d_beta[k]
                gk = float(np.dot(xk, r_vec)) + lam2 * cur_val
                hk = float(H_diag[k])
                new_val = _soft_threshold(hk * cur_val - gk, lam1) / hk
                delta = new_val - cur_val
                if delta != 0.0:
                    d_beta[k] += delta
                    r_vec += (h * delta) * xk
                    max_step = max(max_step, abs(delta))
            if max_step <= sub_tol:
                break

        d_eta = Xw @ d_beta + d_b0
        dir_deriv = (
            float(np.dot(g_eta, d_eta))
            + lam2 * float(np.dot(beta_w, d_beta))
            + lam1 * (float(np.sum(np.abs(beta_w + d_beta))) - float(np.sum(np.abs(beta_w))))
        )
        step = 1.0
        accepted = False
        for _ in range(30):
            cand_b0 = b0 + step * d_b0
            cand_beta = beta.copy()
            cand_beta[work_idx] = beta_w + step * d_beta
            cand_eta = eta + step * d_eta
            cand_obj = (
                float(np.sum(w * (np.logaddexp(0.0, cand_eta) - y_bin * cand_eta)))
                + lam1 * float(np.sum(np.abs(cand_beta)))
                + 0.5 * lam2 * float(np.dot(cand_beta, cand_beta))
            )
            if cand_obj <= cur_obj + 1e-4 * step * min(dir_deriv, 0.0) + 1e-13 * abs(cur_obj):
                b0, beta, eta, cur_obj = cand_b0, cand_beta, cand_eta, cand_obj
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    kkt = exact_en_logistic_kkt(X, y, beta, b0, C, l1_ratio)
    return {
        "beta": beta,
        "intercept": b0,
        "converged": bool(converged or kkt["max_kkt_residual"] <= kkt_tol),
        "jerr": 0 if converged else -1,
        "n_passes": outer_used,
        "lambda_target": 1.0 / (float(C) * float(n)),
        "lambda_fitted": 1.0 / (float(C) * float(n)),
        "r_warnings": [],
        "r_stderr": "",
        "elapsed_sec": time.perf_counter() - t0,
        "kkt": kkt,
    }


def check_r_glmnet() -> dict:
    rscript = shutil.which("Rscript")
    if not rscript:
        return {"available": False, "rscript": None, "r_version": None, "glmnet_version": None}
    cmd = [
        rscript,
        "-e",
        'cat(R.version.string, "\\n"); if (requireNamespace("glmnet", quietly=TRUE)) cat(as.character(packageVersion("glmnet")), "\\n") else cat("MISSING\\n")',
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    r_ver = lines[0] if len(lines) >= 1 else None
    g_ver = lines[1] if len(lines) >= 2 else "MISSING"
    return {
        "available": (res.returncode == 0 and g_ver != "MISSING"),
        "rscript": rscript,
        "r_version": r_ver,
        "glmnet_version": g_ver,
    }


R_GLMNET_SINGLE_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
x_bin     <- args[1]
n_rows    <- as.integer(args[2])
n_cols    <- as.integer(args[3])
y_csv     <- args[4]
w_csv     <- args[5]
alpha_v   <- as.numeric(args[6])
lam_tgt   <- as.numeric(args[7])
thresh    <- as.numeric(args[8])
out_csv   <- args[9]
meta_js   <- args[10]
x_sum_exp <- as.numeric(args[11])
x_01_exp  <- as.numeric(args[12])

suppressPackageStartupMessages(library(glmnet))

con <- file(x_bin, "rb")
x_vec <- readBin(con, what = "double", n = n_rows * n_cols, size = 8, endian = "little")
close(con)
X <- matrix(x_vec, nrow = n_rows, ncol = n_cols, byrow = TRUE)
if (abs(sum(X) - x_sum_exp) > 1e-5 * max(1.0, abs(x_sum_exp))) {
  stop(sprintf("Matrix sum mismatch in R: got %.12g expected %.12g", sum(X), x_sum_exp))
}
if (abs(X[1, 2] - x_01_exp) > 1e-12) {
  stop(sprintf("Matrix element X[1,2] mismatch in R: got %.12g expected %.12g", X[1, 2], x_01_exp))
}

y <- as.integer(scan(y_csv, quiet = TRUE))
w <- as.numeric(scan(w_csv, quiet = TRUE))

y_bin <- as.numeric(y)
w_norm <- w * (length(w) / sum(w))
p_bar <- sum(w_norm * y_bin) / sum(w_norm)
grad0 <- abs(crossprod(X, w_norm * (y_bin - p_bar))) / length(w)
lam_max <- max(as.numeric(grad0)) / max(alpha_v, 1e-3)
if (!is.finite(lam_max) || lam_max <= lam_tgt * 1.5) {
  lam_seq <- c(lam_tgt * 2.0, lam_tgt)
} else {
  lam_seq <- exp(seq(log(lam_max * 1.05), log(lam_tgt), length.out = 25))
  lam_seq[length(lam_seq)] <- lam_tgt
}

warns <- character(0)
t0 <- proc.time()[["elapsed"]]
fit <- withCallingHandlers(
  glmnet(
    x = X,
    y = y,
    family = "binomial",
    weights = w,
    alpha = alpha_v,
    lambda = lam_seq,
    standardize = FALSE,
    intercept = TRUE,
    thresh = thresh,
    maxit = 1000000L
  ),
  warning = function(wrn) {
    warns <<- c(warns, conditionMessage(wrn))
    invokeRestart("muffleWarning")
  }
)
elapsed <- proc.time()[["elapsed"]] - t0

k <- length(fit$lambda)
lam_fitted <- as.numeric(fit$lambda[k])
b0 <- as.numeric(fit$a0[k])
beta <- as.numeric(fit$beta[, k])

write.table(
  data.frame(idx = seq_len(n_cols) - 1L, coef = beta),
  file = out_csv,
  sep = ",",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE
)

esc_json <- function(s) {
  s <- gsub("\\\\", "\\\\\\\\", s)
  s <- gsub('"', '\\\\"', s)
  s <- gsub("[\r\n]+", " ", s)
  paste0('"', s, '"')
}
w_json <- if (length(warns) == 0) "[]" else paste0("[", paste(vapply(warns, esc_json, ""), collapse = ", "), "]")

cat(
  sprintf(
    '{"intercept": %.17g, "lambda_target": %.17g, "lambda_fitted": %.17g, "n_lambda_req": %d, "n_lambda_fit": %d, "n_passes": %d, "jerr": %d, "elapsed_sec": %.6f, "glmnet_version": "%s", "r_warnings": %s}\n',
    b0, lam_tgt, lam_fitted, length(lam_seq), k, as.integer(fit$npasses), as.integer(fit$jerr), elapsed, as.character(packageVersion("glmnet")), w_json
  ),
  file = meta_js
)
"""


def validate_glmnet_output(
    beta: np.ndarray,
    b0: float,
    X: np.ndarray,
    expected_p: int,
    lambda_target: float,
    lambda_fitted: float,
    jerr: int,
) -> None:
    """P0-4: Strict fail-closed validation of R glmnet solver outputs."""
    require(jerr == 0, f"R glmnet returned non-zero error code jerr={jerr}", SolverOutputError)
    require(np.isfinite(lambda_fitted), f"R glmnet returned non-finite lambda_fitted={lambda_fitted}", SolverOutputError)
    tol_lam = max(1e-12, 1e-9 * abs(lambda_target))
    require(
        abs(lambda_fitted - lambda_target) <= tol_lam,
        f"R glmnet lambda mismatch (path truncated?): fitted={lambda_fitted:.12g} vs target={lambda_target:.12g} (tol={tol_lam:.2e})",
        SolverOutputError,
    )
    require(len(beta) == expected_p, f"Coefficient length mismatch: got {len(beta)} expected {expected_p}", SolverOutputError)
    require(bool(np.all(np.isfinite(beta))), "R glmnet returned non-finite beta coefficients", SolverOutputError)
    require(bool(np.isfinite(b0)), f"R glmnet returned non-finite intercept={b0}", SolverOutputError)
    probs = expit(X @ beta + b0)
    require(bool(np.all(np.isfinite(probs))), "R glmnet produced non-finite training probabilities", SolverOutputError)


def solve_r_glmnet(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    l1_ratio: float,
    thresh: float = 1e-12,
    timeout_sec: int = DEFAULT_FIT_TIMEOUT_SEC,
) -> dict:
    """Invoke R glmnet with C-order binary transfer, R checksum, timeout, warning capture, and P0-4 lambda validation."""
    n, p = X.shape
    w = balanced_weights(y)
    lam_target = 1.0 / (float(C) * float(n))
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        r_file = td_p / "run_glmnet.R"
        x_bin = td_p / "X.bin"
        y_csv = td_p / "y.txt"
        w_csv = td_p / "w.txt"
        out_csv = td_p / "beta.csv"
        meta_js = td_p / "meta.json"

        r_file.write_text(R_GLMNET_SINGLE_SCRIPT, encoding="utf-8")
        X_c = np.ascontiguousarray(X, dtype="<f8")
        x_bin.write_bytes(X_c.tobytes(order="C"))
        np.savetxt(y_csv, y.astype(int), fmt="%d")
        np.savetxt(w_csv, w, fmt="%.17g")

        cmd = [
            "Rscript",
            str(r_file),
            str(x_bin),
            str(n),
            str(p),
            str(y_csv),
            str(w_csv),
            f"{l1_ratio:.17g}",
            f"{lam_target:.17g}",
            f"{thresh:.17g}",
            str(out_csv),
            str(meta_js),
            f"{float(X_c.sum()):.17g}",
            f"{float(X_c[0, 1]):.17g}",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            raise SolverOutputError(f"Rscript glmnet timed out after {timeout_sec}s (C={C}, r={l1_ratio}, thresh={thresh})") from exc
        if proc.returncode != 0:
            raise SolverOutputError(f"Rscript glmnet failed (rc={proc.returncode}): {proc.stderr.strip()}")
        meta = json.loads(meta_js.read_text(encoding="utf-8"))
        beta = pd.read_csv(out_csv)["coef"].to_numpy(dtype=np.float64)
        b0 = float(meta["intercept"])

    # P0-4 strict validation
    validate_glmnet_output(
        beta=beta,
        b0=b0,
        X=X,
        expected_p=p,
        lambda_target=lam_target,
        lambda_fitted=float(meta["lambda_fitted"]),
        jerr=int(meta["jerr"]),
    )

    kkt = exact_en_logistic_kkt(X, y, beta, b0, C, l1_ratio)
    return {
        "beta": beta,
        "intercept": b0,
        "converged": bool(meta["jerr"] == 0),
        "jerr": int(meta["jerr"]),
        "n_passes": int(meta["n_passes"]),
        "lambda_target": float(meta["lambda_target"]),
        "lambda_fitted": float(meta["lambda_fitted"]),
        "n_lambda_req": int(meta["n_lambda_req"]),
        "n_lambda_fit": int(meta["n_lambda_fit"]),
        "glmnet_version": meta.get("glmnet_version"),
        "r_warnings": list(meta.get("r_warnings", [])),
        "r_stderr": proc.stderr.strip(),
        "r_fit_sec": float(meta["elapsed_sec"]),
        "elapsed_sec": time.perf_counter() - t0,
        "kkt": kkt,
    }


def fit_certified_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    l1_ratio: float,
    kkt_gate: float,
    backend: str = "r-glmnet",
    timeout_sec: int = DEFAULT_FIT_TIMEOUT_SEC,
) -> dict:
    """
    Production solver pipeline:
      1. Primary solve (`glmnet(thresh=1e-12)` or `python-cd`) with P0-4 checks.
      2. Independent common-objective KKT check.
      3. Production active-set Newton polish with full-gradient verification.
      4. Targeted tighter `glmnet(thresh=1e-14)` + Newton polish fallback if needed.
      5. Returns full diagnostics including `numerical_gate_passed` (caller fails closed).
    """
    t_total0 = time.perf_counter()
    warnings_list: list[str] = []
    stage_log: list[dict] = []

    if backend == "r-glmnet":
        base = solve_r_glmnet(X, y, C=C, l1_ratio=l1_ratio, thresh=1e-12, timeout_sec=timeout_sec)
        solver_used = "r_glmnet_thresh1e-12"
    else:
        base = solve_python_prox_newton_cd(X, y, C=C, l1_ratio=l1_ratio, kkt_tol=1e-7)
        solver_used = "python_prox_newton_cd"

    if base.get("r_warnings"):
        for rw in base["r_warnings"]:
            warnings_list.append(f"R warning ({solver_used}): {rw}")
    if base.get("r_stderr"):
        warnings_list.append(f"R stderr ({solver_used}): {base['r_stderr']}")

    stage_log.append({
        "stage": "primary",
        "solver": solver_used,
        "converged": base["converged"],
        "jerr": base["jerr"],
        "lambda_target": base.get("lambda_target"),
        "lambda_fitted": base.get("lambda_fitted"),
        "max_kkt": base["kkt"]["max_kkt_residual"],
        "obj": base["kkt"]["common_objective"],
        "elapsed_sec": round(base["elapsed_sec"], 4),
    })
    if not base["converged"]:
        warnings_list.append(f"Primary solver {solver_used} reported non-zero convergence flag (jerr={base['jerr']}).")

    beta = base["beta"]
    b0 = base["intercept"]
    kkt = base["kkt"]
    polished = False
    fallback_used = False

    if kkt["max_kkt_residual"] > 1e-9:
        pol = polish_active_set_newton(X, y, beta, b0, C=C, l1_ratio=l1_ratio, kkt_target=1e-10)
        stage_log.append({
            "stage": "active_set_newton_polish",
            "accepted": pol["polish_accepted"],
            "status": pol["polish_status"],
            "iters": pol["polish_iters"],
            "orig_kkt": pol["orig_kkt"],
            "polished_kkt_raw": pol["polished_kkt_raw"],
            "sign_change_blocked": pol["sign_change_blocked"],
            "elapsed_sec": round(pol["polish_sec"], 4),
        })
        if pol["polish_accepted"]:
            beta, b0, kkt = pol["beta"], pol["intercept"], pol["kkt"]
            polished = True
        else:
            warnings_list.append(f"Active-set polish not accepted (status={pol['polish_status']}, raw_kkt={pol['polished_kkt_raw']:.2e}).")

    if kkt["max_kkt_residual"] > kkt_gate and backend == "r-glmnet":
        fallback_used = True
        warnings_list.append(f"KKT {kkt['max_kkt_residual']:.2e} exceeded gate {kkt_gate:.1e}; invoking targeted glmnet(thresh=1e-14) fallback.")
        fb = solve_r_glmnet(X, y, C=C, l1_ratio=l1_ratio, thresh=1e-14, timeout_sec=timeout_sec)
        fb_beta, fb_b0, fb_kkt = fb["beta"], fb["intercept"], fb["kkt"]
        fb_pol = polish_active_set_newton(X, y, fb_beta, fb_b0, C=C, l1_ratio=l1_ratio, kkt_target=1e-10)
        if fb_pol["polish_accepted"]:
            fb_beta, fb_b0, fb_kkt = fb_pol["beta"], fb_pol["intercept"], fb_pol["kkt"]
        stage_log.append({
            "stage": "tighter_glmnet_fallback_1e-14",
            "converged": fb["converged"],
            "jerr": fb["jerr"],
            "lambda_target": fb.get("lambda_target"),
            "lambda_fitted": fb.get("lambda_fitted"),
            "max_kkt_after_polish": fb_kkt["max_kkt_residual"],
            "obj": fb_kkt["common_objective"],
            "elapsed_sec": round(fb["elapsed_sec"], 4),
        })
        if fb_kkt["max_kkt_residual"] < kkt["max_kkt_residual"]:
            beta, b0, kkt = fb_beta, fb_b0, fb_kkt
            solver_used = "r_glmnet_thresh1e-14_fallback"

    finite_ok = bool(
        np.all(np.isfinite(beta))
        and np.isfinite(b0)
        and np.isfinite(kkt["common_objective"])
        and np.isfinite(kkt["max_kkt_residual"])
    )
    gate_passed = bool(finite_ok and kkt["max_kkt_residual"] < kkt_gate)
    if not gate_passed:
        warnings_list.append(f"NUMERICAL GATE FAILED: max_kkt={kkt['max_kkt_residual']:.3e} >= gate={kkt_gate:.1e} (finite={finite_ok})")

    return {
        "beta": beta,
        "intercept": float(b0),
        "kkt": kkt,
        "lambda_target": base.get("lambda_target"),
        "lambda_fitted": base.get("lambda_fitted"),
        "solver_used": solver_used,
        "polished": polished,
        "fallback_used": fallback_used,
        "kkt_gate": kkt_gate,
        "numerical_gate_passed": gate_passed,
        "warnings": warnings_list,
        "stage_log": stage_log,
        "elapsed_sec": time.perf_counter() - t_total0,
    }


def validate_checkpoint_for_resume(
    rec: dict,
    expected: dict,
    full_coef_path: Path | None = None,
    panel_path: Path | None = None,
) -> None:
    """P0-2: Strictly validate a persisted checkpoint before reusing it during resume."""
    fid = expected["fit_id"]
    require(rec.get("status") == "completed", f"Checkpoint {fid} has status={rec.get('status')!r} (not 'completed')", CheckpointIntegrityError)
    require(rec.get("numerical_gate_passed") is True, f"Checkpoint {fid} has numerical_gate_passed={rec.get('numerical_gate_passed')!r}", CheckpointIntegrityError)

    exact_fields = (
        "fit_id", "backend", "config_fingerprint", "script_sha256",
        "dataset_zip_sha256", "train_ids_sha256", "val_ids_sha256",
        "elig_genes_sha256", "n_train", "eligible_genes",
    )
    for field in exact_fields:
        require(
            rec.get(field) == expected.get(field),
            f"Checkpoint {fid} mismatch on {field}: saved={rec.get(field)!r} vs expected={expected.get(field)!r}",
            CheckpointIntegrityError,
        )

    for num_field in ("C", "l1_ratio", "kkt_gate"):
        require(
            np.isclose(float(rec.get(num_field, np.nan)), float(expected[num_field]), rtol=0.0, atol=1e-15),
            f"Checkpoint {fid} mismatch on {num_field}: saved={rec.get(num_field)} vs expected={expected[num_field]}",
            CheckpointIntegrityError,
        )

    require(
        np.isfinite(float(rec.get("common_objective", np.nan)))
        and np.isfinite(float(rec.get("intercept", np.nan)))
        and float(rec.get("max_kkt_residual", np.inf)) < float(expected["kkt_gate"]),
        f"Checkpoint {fid} has invalid/non-certified KKT or objective",
        CheckpointIntegrityError,
    )
    if expected["val_ids_sha256"] != "none":
        require(np.isfinite(float(rec.get("auroc", np.nan))), f"Checkpoint {fid} missing finite validation AUROC", CheckpointIntegrityError)

    if full_coef_path is not None and panel_path is not None:
        require(full_coef_path.is_file(), f"Checkpoint {fid} missing full_coefficients file: {full_coef_path.name}", CheckpointIntegrityError)
        require(panel_path.is_file(), f"Checkpoint {fid} missing panel file: {panel_path.name}", CheckpointIntegrityError)
        require(
            sha_bytes(full_coef_path.read_bytes()) == rec.get("full_coef_sha256"),
            f"Checkpoint {fid} full_coefficients SHA-256 mismatch (file corrupted or modified)",
            CheckpointIntegrityError,
        )
        require(
            sha_bytes(panel_path.read_bytes()) == rec.get("panel_sha256"),
            f"Checkpoint {fid} panel SHA-256 mismatch (file corrupted or modified)",
            CheckpointIntegrityError,
        )


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)
    i, n = 0, len(y_score)
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    desc = np.argsort(-y_score, kind="mergesort")
    y_s, s_s = y_true[desc], y_score[desc]
    distinct = np.where(np.diff(s_s) != 0)[0]
    t_idx = np.r_[distinct, len(y_s) - 1]
    tps = np.cumsum(y_s)[t_idx]
    fps = (1 + t_idx) - tps
    prec = tps / (tps + fps)
    rec = tps / n_pos
    return float(np.sum((rec - np.r_[0.0, rec[:-1]]) * prec))


def stratified_bootstrap_ci(y_true: np.ndarray, vals: np.ndarray, fn, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    scores = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos_idx, len(pos_idx), True), rng.choice(neg_idx, len(neg_idx), True)])
        scores[b] = fn(y_true[idx], vals[idx])
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def select_best_hyperparams(df_scores: pd.DataFrame, expected_n_folds: int = 3) -> tuple[float, float, float]:
    """P0-1: Assert complete certified grid before applying canonical tie-break (AUROC desc, C asc, l1_ratio desc)."""
    expected_rows = len(C_GRID) * len(ELASTIC_RATIOS) * expected_n_folds
    require(len(df_scores) == expected_rows, f"Incomplete tuning table: got {len(df_scores)} rows, expected {expected_rows}", NumericalGateError)
    require("numerical_gate_passed" in df_scores.columns and bool(df_scores["numerical_gate_passed"].all()), "Cannot select hyperparameters: one or more inner-CV fits failed numerical gate", NumericalGateError)
    require(bool(np.all(np.isfinite(df_scores["auroc"].to_numpy(dtype=float)))), "Cannot select hyperparameters: non-finite inner-CV AUROC detected", NumericalGateError)

    agg = df_scores.groupby(["C", "l1_ratio"], as_index=False).agg(mean_auroc=("auroc", "mean"), n_folds=("auroc", "count"))
    require(len(agg) == len(C_GRID) * len(ELASTIC_RATIOS) and bool((agg["n_folds"] == expected_n_folds).all()), "Incomplete (C, l1_ratio) fold coverage in tuning table", NumericalGateError)
    agg = agg.sort_values(by=["mean_auroc", "C", "l1_ratio"], ascending=[False, True, False]).reset_index(drop=True)
    best = agg.iloc[0]
    return float(best["C"]), float(best["l1_ratio"]), float(best["mean_auroc"])


def jaccard_sets(a: set, b: set) -> tuple[int, int, float, float]:
    inter = len(a & b)
    uni = len(a | b)
    j = inter / uni if uni > 0 else 0.0
    ov = inter / min(len(a), len(b)) if min(len(a), len(b)) > 0 else 0.0
    return inter, uni, float(j), float(ov)


def evaluate_stage22_fold(
    counts: np.ndarray,
    y_all: np.ndarray,
    id_arr: np.ndarray,
    cohort_arr: np.ndarray,
    genes: tuple[str, ...],
    outer_c: str,
    fold_idx: int,
    sel_genes: list[str],
    outer_rec: dict,
    n_boot: int = 2000,
) -> tuple[list[dict], dict, list[str]]:
    """Canonical Stage 22 fixed balanced L2 liblinear evaluation with safe empty-panel handling."""
    tr_mask = cohort_arr != outer_c
    te_mask = cohort_arr == outer_c
    y_tr, y_te = y_all[tr_mask], y_all[te_mask]
    require(len(np.unique(y_tr)) == 2 and len(np.unique(y_te)) == 2, f"Stage 22 fold {outer_c} missing class")
    stage22_warnings: list[str] = []

    if len(sel_genes) == 0:
        stage22_warnings.append(f"Stage 22 fold {outer_c}: selected Elastic Net panel is EMPTY (0 genes); assigning constant balanced prior prob=0.5, pred=0.")
        prob = np.full(len(y_te), 0.5, dtype=np.float64)
        pred = np.zeros(len(y_te), dtype=int)
        n_iter_clf = 0
        converged_clf = True
        empty_panel = True
    else:
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        sel_idx = np.array([gene_to_idx[g] for g in sel_genes], dtype=int)
        X_tr_raw = log_cpm(counts[tr_mask])[:, sel_idx]
        X_te_raw = log_cpm(counts[te_mask])[:, sel_idx]
        mu = X_tr_raw.mean(axis=0)
        sd = X_tr_raw.std(axis=0, ddof=0)
        sd = np.where(sd == 0.0, 1.0, sd)
        X_tr_s = (X_tr_raw - mu) / sd
        X_te_s = (X_te_raw - mu) / sd

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=STAGE22_SEED,
        )
        clf.fit(X_tr_s, y_tr)
        prob = clf.predict_proba(X_te_s)[:, 1]
        require(bool(np.all(np.isfinite(prob))), f"Stage 22 fold {outer_c} produced non-finite probabilities")
        pred = (prob >= 0.5).astype(int)
        n_iter_clf = int(np.max(clf.n_iter_))
        converged_clf = bool(n_iter_clf < 5000)
        if not converged_clf:
            stage22_warnings.append(f"Stage 22 fold {outer_c}: liblinear reached max_iter=5000")
        empty_panel = False

    pred_rows = [
        {"cohort": outer_c, "ID": str(sid), "y_true": int(yt), "prob_CD": float(pr), "pred_CD": int(pd_l)}
        for sid, yt, pr, pd_l in zip(id_arr[te_mask], y_te, prob, pred)
    ]

    tp = int(((y_te == 1) & (pred == 1)).sum())
    tn = int(((y_te == 0) & (pred == 0)).sum())
    fp = int(((y_te == 0) & (pred == 1)).sum())
    fn = int(((y_te == 1) & (pred == 0)).sum())
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    bacc = 0.5 * (sens + spec)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    brier = float(np.mean((prob - y_te) ** 2))
    auroc = compute_auroc(y_te, prob)
    auprc = compute_auprc(y_te, prob)

    auroc_lo, auroc_hi = stratified_bootstrap_ci(y_te, prob, compute_auroc, n_boot=n_boot, seed=STAGE22_SEED + fold_idx)
    bacc_fn = lambda yt, yh: 0.5 * (((yt == 1) & (yh == 1)).sum() / (yt == 1).sum() + ((yt == 0) & (yh == 0)).sum() / (yt == 0).sum())
    bacc_lo, bacc_hi = stratified_bootstrap_ci(y_te, pred, bacc_fn, n_boot=n_boot, seed=STAGE22_SEED + 100 + fold_idx)

    metric_row = {
        "cohort": outer_c,
        "best_C": outer_rec["best_C"],
        "best_l1_ratio": outer_rec["best_l1_ratio"],
        "panel_size": len(sel_genes),
        "stage22_empty_panel": empty_panel,
        "auroc": auroc,
        "auroc_ci_low": auroc_lo,
        "auroc_ci_high": auroc_hi,
        "auprc": auprc,
        "bacc": bacc,
        "bacc_ci_low": bacc_lo,
        "bacc_ci_high": bacc_hi,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "brier": brier,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "stage22_converged": converged_clf,
        "stage22_n_iter": n_iter_clf,
    }
    return pred_rows, metric_row, stage22_warnings


def certify_benchmark_completion(out_dir: Path) -> tuple[bool, list[str], dict]:
    """P0-3: Strict completion audit checking all 200 canonical fits, 8 final artifacts, 689 predictions, and tables."""
    issues: list[str] = []
    fit_dir = out_dir / "checkpoints" / "fits"
    coef_dir = out_dir / "full_coefficients"
    panel_dir = out_dir / "panels"

    exp_fits = expected_canonical_fit_ids()
    n_found = 0
    n_certified = 0
    n_failed = 0
    warnings_all = []

    if not fit_dir.is_dir():
        issues.append("Missing checkpoints/fits/ directory.")
    else:
        for fid, role in exp_fits.items():
            fp = fit_dir / f"{fid}.json"
            if not fp.is_file():
                issues.append(f"Missing expected fit checkpoint: {fid}")
                continue
            n_found += 1
            rec = json.loads(fp.read_text(encoding="utf-8"))
            gate_lim = FINAL_KKT_GATE if role == "final" else INNER_KKT_GATE
            ok = (
                rec.get("status") == "completed"
                and rec.get("numerical_gate_passed") is True
                and float(rec.get("max_kkt_residual", np.inf)) < gate_lim
            )
            if ok:
                n_certified += 1
            else:
                n_failed += 1
                issues.append(f"Fit {fid} not certified (status={rec.get('status')}, gate={rec.get('numerical_gate_passed')}, kkt={rec.get('max_kkt_residual')})")
            if rec.get("warnings"):
                warnings_all.append({
                    "fit_id": fid,
                    "role": role,
                    "status": rec.get("status"),
                    "numerical_gate_passed": rec.get("numerical_gate_passed"),
                    "max_kkt_residual": rec.get("max_kkt_residual"),
                    "warnings": rec.get("warnings"),
                })
            if role == "final":
                cpath = coef_dir / f"{fid}_all_coefficients.csv"
                ppath = panel_dir / f"{fid}_selected_genes.csv"
                if not cpath.is_file() or sha_bytes(cpath.read_bytes()) != rec.get("full_coef_sha256"):
                    issues.append(f"Missing or hash-mismatched full_coefficients for {fid}")
                if not ppath.is_file() or sha_bytes(ppath.read_bytes()) != rec.get("panel_sha256"):
                    issues.append(f"Missing or hash-mismatched panel for {fid}")

    # Check required output tables & row counts
    table_checks = [
        ("loco_inner_tuning_scores.csv", 96),
        ("loco_outer_enet_summary.csv", 4),
        ("loco_pairwise_jaccard.csv", 6),
        ("indep_cv_tuning_scores.csv", 96),
        ("indep_enet_summary.csv", 4),
        ("indep_pairwise_jaccard.csv", 6),
        ("stage22_corrected_enet_predictions_PRIVATE.csv", 689),
        ("stage22_corrected_enet_cohort_metrics.csv", 4),
        ("stage22_corrected_enet_macro_summary.csv", 1),
    ]
    for fname, exp_rows in table_checks:
        p = out_dir / fname
        if not p.is_file():
            issues.append(f"Missing required result table: {fname}")
        else:
            df = pd.read_csv(p)
            if len(df) != exp_rows:
                issues.append(f"Table {fname} has {len(df)} rows, expected {exp_rows}")

    for req_file in ("loco_genes_in_all_four_folds.txt", "indep_cv_splits_seed402.csv"):
        if not (out_dir / req_file).is_file():
            issues.append(f"Missing required file: {req_file}")

    stats = {
        "expected_fits": 200,
        "found_fits": n_found,
        "certified_fits": n_certified,
        "failed_or_uncertified_fits": n_failed,
        "n_issues": len(issues),
        "warnings_all": warnings_all,
    }
    return (len(issues) == 0), issues, stats


def finalize_outputs(
    out_dir: Path,
    mode: str = "full_certification",
    data_prov: dict | None = None,
    software_info: dict | None = None,
    config_fingerprint: str | None = None,
    allow_incomplete_recovery: bool = False,
) -> dict:
    """
    Isolated finalization (NEVER invokes any solver).
    - Preserves immutable execution provenance (`execution_script_sha256`, `execution_created_utc`).
    - Explicitly separates `CERTIFIED_COMPLETE` from `INCOMPLETE_OR_FAILED`.
    - Raises `CompletionCertificationError` in `full_certification` mode (or recovery without `allow_incomplete_recovery`) if incomplete.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    certified, issues, stats = certify_benchmark_completion(out_dir)
    cert_status = "CERTIFIED_COMPLETE" if certified else "INCOMPLETE_OR_FAILED"

    atomic_write_json(out_dir / "numerical_warnings.json", {
        "certification_status": cert_status,
        "expected_fits": stats["expected_fits"],
        "found_fits": stats["found_fits"],
        "certified_fits": stats["certified_fits"],
        "failed_or_uncertified_fits": stats["failed_or_uncertified_fits"],
        "certification_issues": issues[:50],
        "warning_records": stats["warnings_all"],
    })

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

    # Preserve immutable original execution provenance if already recorded
    now_utc = datetime.now(timezone.utc).isoformat()
    cur_sha = get_script_sha256()
    if mode == "full_certification" and "execution_script_sha256" not in manifest:
        manifest["execution_script_name"] = Path(__file__).name
        manifest["execution_script_version"] = SCRIPT_VERSION
        manifest["execution_script_sha256"] = cur_sha
        manifest["execution_created_utc"] = now_utc

    manifest["finalizer_script_name"] = Path(__file__).name
    manifest["finalizer_script_sha256"] = cur_sha
    manifest["finalized_utc"] = now_utc
    manifest["finalization_mode"] = mode
    manifest["certification_status"] = cert_status
    manifest["certification_summary"] = {k: v for k, v in stats.items() if k != "warnings_all"}
    manifest["certification_issues"] = issues[:50]
    if config_fingerprint is not None and "config_fingerprint" not in manifest:
        manifest["config_fingerprint"] = config_fingerprint
    if data_prov is not None and "dataset" not in manifest:
        manifest["dataset"] = data_prov
    if software_info is not None and "software" not in manifest:
        manifest["software"] = software_info

    atomic_write_json(manifest_path, manifest)
    sums, excl = write_sha256sums_safe(out_dir)
    print(f"[FINALIZE:{mode}] status={cert_status} certified={stats['certified_fits']}/200 issues={len(issues)} SHA256SUMS={len(sums)} (excluded={len(excl)})", flush=True)

    if not certified and (mode == "full_certification" or not allow_incomplete_recovery):
        raise CompletionCertificationError(
            f"Run is {cert_status} ({stats['certified_fits']}/200 certified fits, {len(issues)} issues). First issue: {issues[0] if issues else 'unknown'}"
        )
    return manifest


def run_full_benchmark(
    data_path: Path,
    out_dir: Path,
    backend: str = "r-glmnet",
    verify_hashes: bool = True,
    n_boot: int = 2000,
    resume: bool = True,
    timeout_sec: int = DEFAULT_FIT_TIMEOUT_SEC,
) -> None:
    """Execute the canonical 200-fit fail-closed corrected nested Elastic Net benchmark."""
    import sklearn
    from sklearn.model_selection import StratifiedKFold

    out_dir.mkdir(parents=True, exist_ok=True)
    fit_dir = out_dir / "checkpoints" / "fits"
    panel_dir = out_dir / "panels"
    coef_dir = out_dir / "full_coefficients"
    fit_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    coef_dir.mkdir(parents=True, exist_ok=True)

    r_info = check_r_glmnet()
    if backend == "r-glmnet" and not r_info["available"]:
        raise RuntimeError(f"R glmnet not available: {r_info}")

    meta, counts, genes, data_prov = load_canonical_dataset(data_path, verify_hashes=verify_hashes)
    y_all = (meta["class"] == "CD").to_numpy(dtype=np.int64)
    cohort_arr = meta["cohort"].to_numpy()
    id_arr = meta["ID"].to_numpy()

    script_sha = get_script_sha256()
    cfg_fp = build_config_fingerprint(backend, data_prov.get("zip_sha256"), script_sha)

    def get_or_fit(
        fit_id: str,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        train_ids: np.ndarray,
        C_val: float,
        r_val: float,
        kkt_gate: float,
        elig_genes: list[str],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        val_ids: np.ndarray | None = None,
        save_full: bool = False,
    ) -> dict:
        ckpt = fit_dir / f"{fit_id}.json"
        cpath = (coef_dir / f"{fit_id}_all_coefficients.csv") if save_full else None
        ppath = (panel_dir / f"{fit_id}_selected_genes.csv") if save_full else None

        expected_meta = {
            "fit_id": fit_id,
            "C": float(C_val),
            "l1_ratio": float(r_val),
            "kkt_gate": float(kkt_gate),
            "backend": backend,
            "config_fingerprint": cfg_fp,
            "script_sha256": script_sha,
            "dataset_zip_sha256": data_prov.get("zip_sha256") or "unverified",
            "train_ids_sha256": sha_str_list(train_ids),
            "val_ids_sha256": sha_str_list(val_ids) if val_ids is not None else "none",
            "elig_genes_sha256": sha_str_list(elig_genes),
            "n_train": int(X_tr.shape[0]),
            "eligible_genes": int(X_tr.shape[1]),
        }

        if resume and ckpt.is_file():
            rec = json.loads(ckpt.read_text(encoding="utf-8"))
            # P0-2: Strictly validate checkpoint before reuse (raises CheckpointIntegrityError on any mismatch/corruption)
            validate_checkpoint_for_resume(rec, expected_meta, full_coef_path=cpath, panel_path=ppath)
            return rec

        res = fit_certified_elastic_net(X_tr, y_tr, C=C_val, l1_ratio=r_val, kkt_gate=kkt_gate, backend=backend, timeout_sec=timeout_sec)
        val_auroc = float("nan")
        if X_val is not None and y_val is not None:
            prob_val = expit(X_val @ res["beta"] + res["intercept"])
            require(bool(np.all(np.isfinite(prob_val))), f"Fit {fit_id}: non-finite validation probabilities", NumericalGateError)
            val_auroc = compute_auroc(y_val, prob_val)

        ab = np.abs(res["beta"])
        n_sel = int((ab > COEF_CUTOFF).sum())
        nz_idx = np.flatnonzero(ab > 0.0)
        compact_nz = {elig_genes[i]: float(res["beta"][i]) for i in nz_idx}

        full_sha = None
        panel_sha = None
        if save_full and res["numerical_gate_passed"]:
            df_full = pd.DataFrame({"gene": elig_genes, "coef": res["beta"]})
            df_pan = df_full.loc[df_full["coef"].abs() > COEF_CUTOFF].copy()
            atomic_write_csv(cpath, df_full)
            atomic_write_csv(ppath, df_pan)
            full_sha = sha_bytes(cpath.read_bytes())
            panel_sha = sha_bytes(ppath.read_bytes())

        # P0-1: Fail-closed status assignment
        status_str = "completed" if res["numerical_gate_passed"] else "failed_numerical_gate"
        rec = {
            **expected_meta,
            "status": status_str,
            "auroc": val_auroc,
            "intercept": res["intercept"],
            "lambda_target": res.get("lambda_target"),
            "lambda_fitted": res.get("lambda_fitted"),
            "n_nonzero_exact": int((ab > 0.0).sum()),
            "n_gt_1e-8": n_sel,
            "n_gt_1e-4": int((ab > 1e-4).sum()),
            "n_gt_1e-3": int((ab > 1e-3).sum()),
            "n_gt_1e-2": int((ab > 1e-2).sum()),
            "max_abs_coef": float(ab.max()) if len(ab) else 0.0,
            "solver_used": res["solver_used"],
            "polished": res["polished"],
            "fallback_used": res["fallback_used"],
            "numerical_gate_passed": res["numerical_gate_passed"],
            "warnings": res["warnings"],
            "stage_log": res["stage_log"],
            "elapsed_sec": round(res["elapsed_sec"], 4),
            "full_coef_sha256": full_sha,
            "panel_sha256": panel_sha,
            "compact_nonzero_coefs": compact_nz if not save_full else None,
            **res["kkt"],
        }
        atomic_write_json(ckpt, rec)

        if not res["numerical_gate_passed"]:
            raise NumericalGateError(
                f"Fit {fit_id} FAILED numerical KKT gate ({res['kkt']['max_kkt_residual']:.3e} >= {kkt_gate:.1e}). Diagnostic checkpoint preserved."
            )
        return rec

    # PART A: NESTED LOCO ELASTIC NET (96 INNER + 4 OUTER)
    loco_inner_rows, loco_outer_rows = [], []
    loco_panels: dict[str, set[str]] = {}

    for outer_c in COHORTS:
        inner_cohorts = [c for c in COHORTS if c != outer_c]
        fold_rows = []
        for inner_val in inner_cohorts:
            tr_mask = (cohort_arr != outer_c) & (cohort_arr != inner_val)
            va_mask = cohort_arr == inner_val
            X_in_tr, idx_in, mu_in, sd_in = fit_features_canonical(counts[tr_mask])
            X_in_va = apply_features_canonical(counts[va_mask], idx_in, mu_in, sd_in)
            elig_in = [genes[i] for i in idx_in]
            for C_val in C_GRID:
                for r_val in ELASTIC_RATIOS:
                    fid = f"loco_inner_{outer_c}_val_{inner_val}_C{C_val}_r{r_val}".replace(".", "p")
                    rec = get_or_fit(fid, X_in_tr, y_all[tr_mask], id_arr[tr_mask], C_val, r_val, INNER_KKT_GATE, elig_in, X_val=X_in_va, y_val=y_all[va_mask], val_ids=id_arr[va_mask])
                    row = {"outer_heldout": outer_c, "inner_val": inner_val, **{k: rec[k] for k in ("C", "l1_ratio", "auroc", "n_train", "eligible_genes", "n_gt_1e-8", "common_objective", "max_kkt_residual", "kkt_active", "kkt_inactive", "kkt_intercept", "numerical_gate_passed", "solver_used", "polished", "fallback_used", "elapsed_sec")}}
                    fold_rows.append(row)
                    loco_inner_rows.append(row)
            atomic_write_csv(out_dir / "loco_inner_tuning_scores.csv", pd.DataFrame(loco_inner_rows))

        best_C, best_r, best_in_auc = select_best_hyperparams(pd.DataFrame(fold_rows), expected_n_folds=3)
        tr_mask = cohort_arr != outer_c
        te_mask = cohort_arr == outer_c
        X_out_tr, idx_out, mu_out, sd_out = fit_features_canonical(counts[tr_mask])
        X_out_te = apply_features_canonical(counts[te_mask], idx_out, mu_out, sd_out)
        elig_out = [genes[i] for i in idx_out]
        fid_out = f"loco_{outer_c}_corrected_enet"
        rec_out = get_or_fit(fid_out, X_out_tr, y_all[tr_mask], id_arr[tr_mask], best_C, best_r, FINAL_KKT_GATE, elig_out, X_val=X_out_te, y_val=y_all[te_mask], val_ids=id_arr[te_mask], save_full=True)
        loco_panels[outer_c] = set(pd.read_csv(panel_dir / f"{fid_out}_selected_genes.csv")["gene"].astype(str))
        loco_outer_rows.append({
            "outer_heldout": outer_c, "best_C": best_C, "best_l1_ratio": best_r,
            "inner_mean_auroc": best_in_auc, "native_heldout_auroc": rec_out["auroc"],
            **{k: rec_out[k] for k in ("n_train", "eligible_genes", "intercept", "n_nonzero_exact", "n_gt_1e-8", "n_gt_1e-4", "n_gt_1e-3", "n_gt_1e-2", "max_abs_coef", "common_objective", "data_loss", "l1_penalty", "l2_penalty", "max_kkt_residual", "kkt_active", "kkt_inactive", "kkt_intercept", "numerical_gate_passed", "solver_used", "polished", "fallback_used", "elapsed_sec")},
        })
        atomic_write_csv(out_dir / "loco_outer_enet_summary.csv", pd.DataFrame(loco_outer_rows))
        print(f"[LOCO {outer_c}] best=({best_C}, {best_r}) genes={rec_out['n_gt_1e-8']} kkt={rec_out['max_kkt_residual']:.2e} CERTIFIED", flush=True)

    loco_j_rows = []
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = COHORTS[i], COHORTS[j]
            inter, uni, jac, ov = jaccard_sets(loco_panels[a], loco_panels[b])
            loco_j_rows.append({"pair": f"{a}-{b}", "size_a": len(loco_panels[a]), "size_b": len(loco_panels[b]), "intersection": inter, "union": uni, "jaccard": jac, "overlap_coeff": ov})
    atomic_write_csv(out_dir / "loco_pairwise_jaccard.csv", pd.DataFrame(loco_j_rows))
    inter4 = sorted(set.intersection(*loco_panels.values()))
    (out_dir / "loco_genes_in_all_four_folds.txt").write_text("\n".join(inter4) + ("\n" if inter4 else ""), encoding="utf-8")

    # PART B: INDEPENDENT-COHORT ELASTIC NET (96 CV + 4 FULL)
    indep_cv_rows, indep_sum_rows, indep_split_rows = [], [], []
    indep_panels: dict[str, set[str]] = {}

    for coh in COHORTS:
        cmask = cohort_arr == coh
        c_counts, c_y, c_ids = counts[cmask], y_all[cmask], id_arr[cmask]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=INDEP_CV_SEED)
        fold_rows = []
        for f_idx, (tr_i, va_i) in enumerate(skf.split(c_counts, c_y), 1):
            for sid in c_ids[tr_i]:
                indep_split_rows.append({"cohort": coh, "fold": f_idx, "role": "train", "ID": str(sid)})
            for sid in c_ids[va_i]:
                indep_split_rows.append({"cohort": coh, "fold": f_idx, "role": "val", "ID": str(sid)})
            X_f_tr, idx_f, mu_f, sd_f = fit_features_canonical(c_counts[tr_i])
            X_f_va = apply_features_canonical(c_counts[va_i], idx_f, mu_f, sd_f)
            elig_f = [genes[i] for i in idx_f]
            for C_val in C_GRID:
                for r_val in ELASTIC_RATIOS:
                    fid = f"indep_cv_{coh}_f{f_idx}_C{C_val}_r{r_val}".replace(".", "p")
                    rec = get_or_fit(fid, X_f_tr, c_y[tr_i], c_ids[tr_i], C_val, r_val, INNER_KKT_GATE, elig_f, X_val=X_f_va, y_val=c_y[va_i], val_ids=c_ids[va_i])
                    row = {"cohort": coh, "fold": f_idx, **{k: rec[k] for k in ("C", "l1_ratio", "auroc", "n_train", "eligible_genes", "n_gt_1e-8", "common_objective", "max_kkt_residual", "kkt_active", "kkt_inactive", "kkt_intercept", "numerical_gate_passed", "solver_used", "polished", "fallback_used", "elapsed_sec")}}
                    fold_rows.append(row)
                    indep_cv_rows.append(row)
            atomic_write_csv(out_dir / "indep_cv_tuning_scores.csv", pd.DataFrame(indep_cv_rows))

        best_C, best_r, best_cv_auc = select_best_hyperparams(pd.DataFrame(fold_rows), expected_n_folds=3)
        X_full, idx_full, _, _ = fit_features_canonical(c_counts)
        elig_full = [genes[i] for i in idx_full]
        fid_full = f"indep_{coh}_corrected_enet"
        rec_full = get_or_fit(fid_full, X_full, c_y, c_ids, best_C, best_r, FINAL_KKT_GATE, elig_full, save_full=True)
        indep_panels[coh] = set(pd.read_csv(panel_dir / f"{fid_full}_selected_genes.csv")["gene"].astype(str))
        indep_sum_rows.append({
            "cohort": coh, "n_samples": int(len(c_y)), "best_C": best_C, "best_l1_ratio": best_r, "cv_mean_auroc": best_cv_auc,
            **{k: rec_full[k] for k in ("eligible_genes", "intercept", "n_nonzero_exact", "n_gt_1e-8", "n_gt_1e-4", "n_gt_1e-3", "n_gt_1e-2", "max_abs_coef", "common_objective", "data_loss", "l1_penalty", "l2_penalty", "max_kkt_residual", "kkt_active", "kkt_inactive", "kkt_intercept", "numerical_gate_passed", "solver_used", "polished", "fallback_used", "elapsed_sec")},
        })
        atomic_write_csv(out_dir / "indep_enet_summary.csv", pd.DataFrame(indep_sum_rows))
        print(f"[INDEP {coh}] best=({best_C}, {best_r}) genes={rec_full['n_gt_1e-8']} kkt={rec_full['max_kkt_residual']:.2e} CERTIFIED", flush=True)

    atomic_write_csv(out_dir / "indep_cv_splits_seed402.csv", pd.DataFrame(indep_split_rows))
    indep_j_rows = []
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = COHORTS[i], COHORTS[j]
            inter, uni, jac, ov = jaccard_sets(indep_panels[a], indep_panels[b])
            indep_j_rows.append({"pair": f"{a}-{b}", "size_a": len(indep_panels[a]), "size_b": len(indep_panels[b]), "intersection": inter, "union": uni, "jaccard": jac, "overlap_coeff": ov})
    atomic_write_csv(out_dir / "indep_pairwise_jaccard.csv", pd.DataFrame(indep_j_rows))

    # PART C: STAGE 22 CONTROLLED COMMON CLASSIFIER
    pred_rows, coh_metric_rows = [], []
    for fold_idx, outer_c in enumerate(COHORTS):
        outer_rec = [r for r in loco_outer_rows if r["outer_heldout"] == outer_c][0]
        p_rows, m_row, _ = evaluate_stage22_fold(
            counts=counts, y_all=y_all, id_arr=id_arr, cohort_arr=cohort_arr, genes=genes,
            outer_c=outer_c, fold_idx=fold_idx, sel_genes=sorted(loco_panels[outer_c]),
            outer_rec=outer_rec, n_boot=n_boot,
        )
        pred_rows.extend(p_rows)
        coh_metric_rows.append(m_row)

    atomic_write_csv(out_dir / "stage22_corrected_enet_predictions_PRIVATE.csv", pd.DataFrame(pred_rows))
    m_df = pd.DataFrame(coh_metric_rows)
    atomic_write_csv(out_dir / "stage22_corrected_enet_cohort_metrics.csv", m_df)
    macro = {
        "method": "Corrected Nested Elastic Net v2 (R glmnet + Newton Polish)",
        "mean_auroc": float(m_df["auroc"].mean()), "sd_auroc": float(m_df["auroc"].std(ddof=1)),
        "mean_auprc": float(m_df["auprc"].mean()),
        "mean_bacc": float(m_df["bacc"].mean()), "sd_bacc": float(m_df["bacc"].std(ddof=1)),
        "mean_sensitivity": float(m_df["sensitivity"].mean()), "mean_specificity": float(m_df["specificity"].mean()),
        "mean_f1": float(m_df["f1"].mean()), "mean_brier": float(m_df["brier"].mean()),
        "mean_panel_size": float(m_df["panel_size"].mean()),
        "min_panel_size": int(m_df["panel_size"].min()), "max_panel_size": int(m_df["panel_size"].max()),
        "mean_loco_jaccard": float(pd.DataFrame(loco_j_rows)["jaccard"].mean()),
        "mean_indep_jaccard": float(pd.DataFrame(indep_j_rows)["jaccard"].mean()),
        "genes_in_all_4_loco_folds": len(inter4),
    }
    atomic_write_csv(out_dir / "stage22_corrected_enet_macro_summary.csv", pd.DataFrame([macro]))

    software_info = {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "r_version": r_info["r_version"], "glmnet_version": r_info["glmnet_version"],
    }
    finalize_outputs(out_dir, mode="full_certification", data_prov=data_prov, software_info=software_info, config_fingerprint=cfg_fp)


def run_production_smoke_test(
    out_dir: Path,
    data_path: Path | None = None,
    backend: str = "r-glmnet",
    verify_hashes: bool = True,
    timeout_sec: int = DEFAULT_FIT_TIMEOUT_SEC,
) -> dict:
    """Timestamped production smoke test exercising D3 + larger LOCO D3-heldout (n=613) stress fit."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    smoke_dir = out_dir / f"smoke_test_run_{ts}"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    fit_dir = smoke_dir / "checkpoints" / "fits"
    panel_dir = smoke_dir / "panels"
    coef_dir = smoke_dir / "full_coefficients"
    fit_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    coef_dir.mkdir(parents=True, exist_ok=True)

    script_sha = get_script_sha256()
    if data_path is not None and data_path.exists():
        meta, counts, genes, data_prov = load_canonical_dataset(data_path, verify_hashes=verify_hashes)
        y_all = (meta["class"] == "CD").to_numpy(dtype=np.int64)
        coh_arr = meta["cohort"].to_numpy()
        id_arr = meta["ID"].to_numpy()

        # Two representative partitions: small independent D3 (n=76) AND largest LOCO D3-heldout (n=613)
        X_d3, idx_d3, _, _ = fit_features_canonical(counts[coh_arr == "d3"])
        X_loco, idx_loco, _, _ = fit_features_canonical(counts[coh_arr != "d3"])
        partitions = {
            "indep_d3": (X_d3, y_all[coh_arr == "d3"], id_arr[coh_arr == "d3"], [genes[i] for i in idx_d3]),
            "loco_d3_stress": (X_loco, y_all[coh_arr != "d3"], id_arr[coh_arr != "d3"], [genes[i] for i in idx_loco]),
        }
        data_source = "canonical_d3_and_loco_d3_n613"
    else:
        rng = np.random.default_rng(355)
        X_s = rng.standard_normal((100, 600))
        X_s = (X_s - X_s.mean(0)) / X_s.std(0)
        tb = np.zeros(600); tb[:10] = rng.normal(0.0, 0.8, 10)
        y_s = (rng.random(100) < expit(X_s @ tb)).astype(np.int64)
        ids_s = np.array([f"S{i:03d}" for i in range(100)])
        g_s = [f"G{i:04d}" for i in range(600)]
        partitions = {"indep_d3": (X_s, y_s, ids_s, g_s), "loco_d3_stress": (X_s, y_s, ids_s, g_s)}
        data_prov = {"source_mode": "synthetic_offline_smoke", "zip_sha256": "synthetic"}
        data_source = "synthetic_100x600"

    cfg_fp = build_config_fingerprint(backend, data_prov.get("zip_sha256"), script_sha)
    targets = [
        ("smoke_indep_d3_C0p1_r0p5", "indep_d3", 0.1, 0.5),
        ("smoke_indep_d3_C1p0_r0p9", "indep_d3", 1.0, 0.9),
        ("smoke_loco_d3_C0p1_r0p5", "loco_d3_stress", 0.1, 0.5),
    ]

    solver_calls = 0
    recs = []
    for pass_num, batch in ((1, targets[:2]), (2, targets)):
        recs = []
        for fid, part_key, C_val, r_val in batch:
            X_tr, y_tr, tr_ids, elig_g = partitions[part_key]
            ckpt = fit_dir / f"{fid}.json"
            cpath = coef_dir / f"{fid}_all_coefficients.csv"
            ppath = panel_dir / f"{fid}_selected_genes.csv"
            exp_meta = {
                "fit_id": fid, "C": C_val, "l1_ratio": r_val, "kkt_gate": FINAL_KKT_GATE,
                "backend": backend, "config_fingerprint": cfg_fp, "script_sha256": script_sha,
                "dataset_zip_sha256": data_prov.get("zip_sha256") or "unverified",
                "train_ids_sha256": sha_str_list(tr_ids), "val_ids_sha256": "none",
                "elig_genes_sha256": sha_str_list(elig_g), "n_train": int(X_tr.shape[0]), "eligible_genes": int(X_tr.shape[1]),
            }
            if ckpt.is_file():
                rec = json.loads(ckpt.read_text(encoding="utf-8"))
                validate_checkpoint_for_resume(rec, exp_meta, full_coef_path=cpath, panel_path=ppath)
                rec["resumed"] = True
                recs.append(rec)
                continue
            solver_calls += 1
            res = fit_certified_elastic_net(X_tr, y_tr, C=C_val, l1_ratio=r_val, kkt_gate=FINAL_KKT_GATE, backend=backend, timeout_sec=timeout_sec)
            require(res["numerical_gate_passed"], f"Smoke fit {fid} failed KKT gate: {res['kkt']['max_kkt_residual']:.3e}", NumericalGateError)
            df_full = pd.DataFrame({"gene": elig_g, "coef": res["beta"]})
            df_pan = df_full.loc[df_full["coef"].abs() > COEF_CUTOFF].copy()
            atomic_write_csv(cpath, df_full)
            atomic_write_csv(ppath, df_pan)
            rec = {
                **exp_meta, "status": "completed", "auroc": float("nan"), "intercept": res["intercept"],
                "n_selected": int(len(df_pan)), "solver_used": res["solver_used"], "polished": res["polished"],
                "fallback_used": res["fallback_used"], "numerical_gate_passed": res["numerical_gate_passed"],
                "warnings": res["warnings"], "stage_log": res["stage_log"], "elapsed_sec": round(res["elapsed_sec"], 4),
                "full_coef_sha256": sha_bytes(cpath.read_bytes()), "panel_sha256": sha_bytes(ppath.read_bytes()),
                **res["kkt"], "resumed": False,
            }
            atomic_write_json(ckpt, rec)
            recs.append(rec)

    require(solver_calls == 3, f"Expected exactly 3 solver calls (2 resumed), got {solver_calls}")
    require(recs[0]["resumed"] is True and recs[1]["resumed"] is True and recs[2]["resumed"] is False, "Resume verification failed")

    summary = {
        "smoke_test_passed": True,
        "smoke_dir": smoke_dir.name,
        "data_source": data_source,
        "backend": backend,
        "script_sha256": script_sha,
        "solver_calls_total": solver_calls,
        "resumed_fits_verified": 2,
        "fits": [
            {"fit_id": r["fit_id"], "n_train": r["n_train"], "n_selected": r["n_selected"], "max_kkt": r["max_kkt_residual"], "elapsed_sec": r["elapsed_sec"], "resumed": r["resumed"]}
            for r in recs
        ],
    }
    atomic_write_json(smoke_dir / "SMOKE_TEST_SUMMARY.json", summary)
    sums, excl = write_sha256sums_safe(smoke_dir)
    print(f"[SMOKE TEST v2 PASSED] dir={smoke_dir.name} SHA256SUMS={len(sums)} (excl={len(excl)})", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", choices=("r-glmnet", "python-cd"), default="r-glmnet")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--allow-incomplete", action="store_true", help="Only with --finalize-only to permit exit 0 on incomplete recovery")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-hash-check", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--fit-timeout-sec", type=int, default=DEFAULT_FIT_TIMEOUT_SEC)
    args = ap.parse_args()

    if args.finalize_only:
        finalize_outputs(args.out, mode="recovery_only", allow_incomplete_recovery=args.allow_incomplete)
    elif args.smoke_test:
        run_production_smoke_test(args.out, data_path=args.data, backend=args.backend, verify_hashes=not args.skip_hash_check, timeout_sec=args.fit_timeout_sec)
    else:
        require(args.data is not None, "--data is required for full benchmark")
        run_full_benchmark(
            data_path=args.data, out_dir=args.out, backend=args.backend,
            verify_hashes=not args.skip_hash_check, n_boot=args.n_boot,
            resume=not args.no_resume, timeout_sec=args.fit_timeout_sec,
        )

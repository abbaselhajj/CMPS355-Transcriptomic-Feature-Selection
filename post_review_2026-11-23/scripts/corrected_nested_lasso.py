#!/usr/bin/env python3
"""
Corrected Nested LASSO Benchmark & Targeted Elastic Net Numerical Diagnostic
CMPS 355 — Post-Peer-Review Revision Workspace (2026-11-22, v1)

Read-only against canonical dataset `ibd_dataset_sample_aligned.zip`.
Writes all outputs incrementally into `--out` (never modifies original files).

Implements:
  Part A: Corrected nested 4-way LOCO LASSO (3 inner cohort splits,
          C in {0.001, 0.01, 0.1, 1.0}, liblinear L1 tol=1e-6 is=1e4 + solver check).
  Part B: Corrected independent-cohort LASSO (D1-D4, 3-fold stratified CV,
          C in {0.001, 0.01, 0.1, 1.0}, KKT + pairwise Jaccard vs Table VI).
  Part C: Stage 22 controlled common-classifier evaluation on corrected LOCO
          LASSO panels (fixed balanced L2 logistic, C=1, liblinear, threshold 0.5,
          seed 123, 2000-resample stratified bootstrap 95% CIs).
  Part D: Targeted Elastic Net numerical diagnostic comparing historical SAGA
          (tol=1e-3, max_iter=1600) vs tighter SAGA (tol=1e-5/1e-6) against
          exact sample-weighted Elastic Net objective and KKT residuals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
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
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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
COHORTS = ("d1", "d2", "d3", "d4")
EXPECTED_ROWS = {"d1": 260, "d2": 179, "d3": 76, "d4": 174}
C_GRID = (0.001, 0.01, 0.1, 1.0)
MIN_CPM = 1.0
MIN_FRACTION = 0.1
SEED = 355
STAGE22_SEED = 123
COEF_CUTOFF = 1e-8

# Historical manuscript v1.6 reference benchmarks (read-only comparison constants)
HIST_LOCO_LASSO = {
    "d1": {"C": 0.1, "genes": 271, "auroc": 0.884775, "bacc": 0.694080},
    "d2": {"C": 0.1, "genes": 295, "auroc": 0.669486, "bacc": 0.620978},
    "d3": {"C": 0.1, "genes": 271, "auroc": 0.585434, "bacc": 0.590336},
    "d4": {"C": 0.1, "genes": 289, "auroc": 0.616655, "bacc": 0.570783},
    "macro_auroc": 0.689087,
    "macro_bacc": 0.619045,
    "loco_jaccard": 0.210809,
}
HIST_INDEP_LASSO = {
    "d1": 2523,
    "d2": 2119,
    "d3": 1734,
    "d4": 2623,
    "mean_jaccard": 0.163535,
}
HIST_OTHER_METHODS_STAGE22 = [
    {"method": "Elastic Net (historical)", "macro_auroc": 0.7118, "macro_bacc": 0.6281, "mean_panel": 352.5, "loco_jaccard": 0.1155},
    {"method": "Stability Selection (0.60, historical)", "macro_auroc": 0.6284, "macro_bacc": 0.5628, "mean_panel": 6.25, "loco_jaccard": 0.1278},
    {"method": "StableMate (historical)", "macro_auroc": 0.6590, "macro_bacc": 0.5979, "mean_panel": 15.75, "loco_jaccard": 0.0326},
    {"method": "TENTACLES V2 (6/9, canonical)", "macro_auroc": 0.7225, "macro_bacc": 0.6551, "mean_panel": 76.50, "loco_jaccard": 0.2097},
]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def load_canonical_data(source_path: Path, verify_hashes: bool = True):
    source_path = Path(source_path)
    require(source_path.exists(), f"Input path not found: {source_path}")
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        zsha = sha_bytes(source_path.read_bytes())
        if verify_hashes:
            require(zsha == EXPECTED_ZIP_SHA, f"Dataset ZIP SHA-256 mismatch: {zsha}")
        with ZipFile(source_path) as zf:
            require(zf.testzip() is None, "ZIP CRC verification failed")
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


def fit_features(train_counts: np.ndarray):
    x = log_cpm(train_counts)
    threshold = max(2, int(np.ceil(MIN_FRACTION * len(x))))
    valid = ((x > np.log2(1.0 + MIN_CPM)).sum(axis=0) >= threshold) & (x.var(axis=0) > 1e-12)
    indices = np.flatnonzero(valid)
    require(len(indices) >= 2, "Insufficient eligible genes")
    scaler = StandardScaler()
    fitted = scaler.fit_transform(x[:, indices])
    require(np.isfinite(fitted).all(), "Non-finite scaled features")
    return fitted, indices, scaler


def apply_features(counts: np.ndarray, indices: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(log_cpm(counts)[:, indices])


def balanced_weights(y: np.ndarray) -> np.ndarray:
    n = len(y)
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    return np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0)).astype(np.float64)


def exact_en_logistic_kkt(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    intercept: float,
    C: float,
    l1_ratio: float = 1.0,
) -> dict:
    """
    Exact common unpenalized-intercept Elastic-Net / L1 binary logistic objective:
      F(beta, b) = sum_i w_i * log(1 + exp(-y_pm_i * (X_i @ beta + b)))
                   + (l1_ratio / C) * ||beta||_1
                   + ((1 - l1_ratio) / (2 * C)) * ||beta||_2^2
    Active/inactive KKT partition uses strict `beta != 0.0`.
    """
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


def fit_logistic_model(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    l1_ratio: float,
    solver: str,
    tol: float,
    max_iter: int,
    intercept_scaling: float = 1.0,
    seed: int = SEED,
):
    major_minor = tuple(int(z) for z in sklearn.__version__.split(".")[:2])
    kwargs = dict(
        solver=solver,
        C=float(C),
        class_weight="balanced",
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=int(seed),
    )
    if solver == "liblinear":
        kwargs["intercept_scaling"] = float(intercept_scaling)
    if major_minor >= (1, 8):
        kwargs["l1_ratio"] = float(l1_ratio)
    else:
        if l1_ratio == 1.0:
            kwargs["penalty"] = "l1"
        elif l1_ratio == 0.0:
            kwargs["penalty"] = "l2"
        else:
            kwargs["penalty"] = "elasticnet"
            kwargs["l1_ratio"] = float(l1_ratio)

    t0 = time.monotonic()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LogisticRegression(**kwargs)
        clf.fit(X, y)
    elapsed = time.monotonic() - t0
    converged = not any(issubclass(r.category, ConvergenceWarning) for r in rec)
    n_iter = int(clf.n_iter_[0]) if hasattr(clf, "n_iter_") else -1
    return clf, converged, n_iter, elapsed


def stratified_bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, fn, n_boot: int = 2000, seed: int = 123):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        vals.append(float(fn(y_true[idx], y_score[idx])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def stage22_eval(counts_tr, y_tr, counts_te, y_te, te_ids, panel_indices: np.ndarray, fold_idx: int, n_boot: int = 2000):
    if len(panel_indices) == 0:
        return {"panel_size": 0, "auroc": np.nan, "bacc": np.nan}, pd.DataFrame()
    xtr = log_cpm(counts_tr)[:, panel_indices]
    xte = log_cpm(counts_te)[:, panel_indices]
    sc = StandardScaler()
    xtr_s = sc.fit_transform(xtr)
    xte_s = sc.transform(xte)
    clf, conv, n_iter, _ = fit_logistic_model(
        xtr_s, y_tr, C=1.0, l1_ratio=0.0, solver="liblinear", tol=1e-4, max_iter=5000, intercept_scaling=1.0, seed=STAGE22_SEED
    )
    prob = clf.predict_proba(xte_s)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    auroc = float(roc_auc_score(y_te, prob))
    auprc = float(average_precision_score(y_te, prob))
    bacc = float(balanced_accuracy_score(y_te, pred))
    sens = float(recall_score(y_te, pred, pos_label=1, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    prec = float(precision_score(y_te, pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_te, pred, pos_label=1, zero_division=0))
    brier = float(brier_score_loss(y_te, prob))
    ci_lo, ci_hi = stratified_bootstrap_ci(y_te, prob, roc_auc_score, n_boot=n_boot, seed=STAGE22_SEED + fold_idx)
    ba_lo, ba_hi = stratified_bootstrap_ci(
        y_te, pred, balanced_accuracy_score, n_boot=n_boot, seed=STAGE22_SEED + 100 + fold_idx
    )
    metrics = {
        "panel_size": int(len(panel_indices)),
        "auroc": auroc,
        "auroc_ci_low": ci_lo,
        "auroc_ci_high": ci_hi,
        "auprc": auprc,
        "bacc": bacc,
        "bacc_ci_low": ba_lo,
        "bacc_ci_high": ba_hi,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "brier": brier,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "stage22_converged": bool(conv),
        "stage22_n_iter": int(n_iter),
    }
    pred_df = pd.DataFrame({
        "ID": te_ids,
        "y_true": y_te,
        "prob_CD": prob,
        "pred_CD": pred,
    })
    return metrics, pred_df


def jaccard(a: set[str], b: set[str]) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def overlap_coeff(a: set[str], b: set[str]) -> float:
    m = min(len(a), len(b))
    return len(a & b) / m if m else 0.0


def save_full_model_artifacts(out_dir: Path, prefix: str, elig_genes: list[str], beta: np.ndarray, intercept: float, meta_rec: dict):
    panel_dir = out_dir / "panels"
    coef_dir = out_dir / "full_coefficients"
    ckpt_dir = out_dir / "checkpoints"
    for d in (panel_dir, coef_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    ab = np.abs(beta)
    sel_mask = ab > COEF_CUTOFF
    sel_df = pd.DataFrame({
        "gene": [elig_genes[j] for j in np.flatnonzero(sel_mask)],
        "coef_scaled": beta[sel_mask],
        "abs_coef": ab[sel_mask],
    }).sort_values("abs_coef", ascending=False)
    sel_df.to_csv(panel_dir / f"{prefix}_selected_genes.csv", index=False)

    full_df = pd.DataFrame({
        "gene": elig_genes,
        "coef_scaled": beta,
        "abs_coef": ab,
        "nonzero_exact": beta != 0.0,
        "selected_gt_1e8": sel_mask,
    })
    full_df.to_csv(coef_dir / f"{prefix}_all_coefficients.csv", index=False)

    rec = dict(meta_rec)
    rec["intercept"] = float(intercept)
    (ckpt_dir / f"{prefix}.json").write_text(json.dumps(rec, indent=2))
    return set(sel_df["gene"].tolist())


def run_all(data_path: Path, out_dir: Path, verify_hashes: bool = True, n_boot: int = 2000, en_max_iter: int = 20000):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, counts, genes, provenance = load_canonical_data(data_path, verify_hashes=verify_hashes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    y_all = (meta["class"].to_numpy() == "CD").astype(int)
    cohorts_arr = meta["cohort"].to_numpy()
    ids_all = meta["ID"].astype(str).to_numpy()

    # =========================================================================
    # PART A: CORRECTED NESTED LOCO LASSO (D1-D4)
    # =========================================================================
    print("=== PART A: CORRECTED NESTED LOCO LASSO ===", flush=True)
    loco_inner_rows = []
    loco_outer_rows = []
    loco_panels = {}
    loco_solver_check_rows = []
    stage22_rows = []
    stage22_preds = []

    for fold_idx, held in enumerate(COHORTS):
        tr_idx = np.flatnonzero(cohorts_arr != held)
        te_idx = np.flatnonzero(cohorts_arr == held)
        require(len(set(ids_all[tr_idx]) & set(ids_all[te_idx])) == 0, "Outer train/test leakage detected")
        dev_cohorts = [c for c in COHORTS if c != held]

        # Inner 3-cohort LOCO tuning
        for inner_val in dev_cohorts:
            in_tr = np.flatnonzero((cohorts_arr != held) & (cohorts_arr != inner_val))
            in_va = np.flatnonzero(cohorts_arr == inner_val)
            require(len(set(ids_all[in_tr]) & set(ids_all[in_va])) == 0, "Inner train/val leakage")
            X_in_tr, in_cols, in_sc = fit_features(counts[in_tr])
            X_in_va = apply_features(counts[in_va], in_cols, in_sc)
            y_in_tr, y_in_va = y_all[in_tr], y_all[in_va]

            for C_val in C_GRID:
                clf, conv, n_iter, sec = fit_logistic_model(
                    X_in_tr, y_in_tr, C=C_val, l1_ratio=1.0, solver="liblinear", tol=1e-6, max_iter=10000, intercept_scaling=1e4
                )
                prob = clf.predict_proba(X_in_va)[:, 1]
                auroc = float(roc_auc_score(y_in_va, prob))
                beta = clf.coef_[0].astype(np.float64)
                b0 = float(clf.intercept_[0])
                kkt = exact_en_logistic_kkt(X_in_tr, y_in_tr, beta, b0, C_val, l1_ratio=1.0)
                loco_inner_rows.append({
                    "outer_heldout": held,
                    "inner_val": inner_val,
                    "C": C_val,
                    "solver": "liblinear_tol1e-6_is1e4",
                    "auroc": auroc,
                    "n_train": int(len(in_tr)),
                    "n_val": int(len(in_va)),
                    "eligible_genes": int(len(in_cols)),
                    "n_selected_gt_1e8": int((np.abs(beta) > COEF_CUTOFF).sum()),
                    "converged": bool(conv),
                    "n_iter": int(n_iter),
                    "elapsed_sec": round(sec, 2),
                    **kkt,
                })

        inner_df = pd.DataFrame([r for r in loco_inner_rows if r["outer_heldout"] == held])
        agg = inner_df.groupby("C", as_index=False)["auroc"].mean()
        # Exact tie-break from full_gene_loco.py: highest mean AUROC, smallest C
        agg = agg.sort_values(["auroc", "C"], ascending=[False, True])
        best_C = float(agg.iloc[0]["C"])
        best_inner_auroc = float(agg.iloc[0]["auroc"])

        # Outer refit on complete 3-cohort training set
        X_tr, tr_cols, tr_sc = fit_features(counts[tr_idx])
        y_tr = y_all[tr_idx]
        elig_genes = [genes[j] for j in tr_cols]

        clf_prim, conv_p, iter_p, sec_p = fit_logistic_model(
            X_tr, y_tr, C=best_C, l1_ratio=1.0, solver="liblinear", tol=1e-6, max_iter=10000, intercept_scaling=1e4
        )
        beta_p = clf_prim.coef_[0].astype(np.float64)
        b0_p = float(clf_prim.intercept_[0])
        kkt_p = exact_en_logistic_kkt(X_tr, y_tr, beta_p, b0_p, best_C, l1_ratio=1.0)

        # Native outer prediction (from the retuned L1 model itself)
        X_te = apply_features(counts[te_idx], tr_cols, tr_sc)
        prob_nat = clf_prim.predict_proba(X_te)[:, 1]
        pred_nat = (prob_nat >= 0.5).astype(int)
        nat_auroc = float(roc_auc_score(y_all[te_idx], prob_nat))
        nat_bacc = float(balanced_accuracy_score(y_all[te_idx], pred_nat))

        ab_p = np.abs(beta_p)
        outer_rec = {
            "outer_heldout": held,
            "best_C": best_C,
            "inner_mean_auroc": best_inner_auroc,
            "train_n": int(len(tr_idx)),
            "test_n": int(len(te_idx)),
            "eligible_genes": int(len(tr_cols)),
            "solver": "liblinear_tol1e-6_is1e4",
            "converged": bool(conv_p),
            "n_iter": int(iter_p),
            "elapsed_sec": round(sec_p, 2),
            "n_nonzero_exact": int((beta_p != 0.0).sum()),
            "n_gt_1e-8": int((ab_p > 1e-8).sum()),
            "n_gt_1e-4": int((ab_p > 1e-4).sum()),
            "n_gt_1e-3": int((ab_p > 1e-3).sum()),
            "n_gt_1e-2": int((ab_p > 1e-2).sum()),
            "max_abs_coef": float(ab_p.max()),
            "median_nonzero_coef": float(np.median(ab_p[ab_p > 1e-8])) if np.any(ab_p > 1e-8) else 0.0,
            "native_heldout_auroc": nat_auroc,
            "native_heldout_bacc": nat_bacc,
            **kkt_p,
        }
        sel_prim = save_full_model_artifacts(out_dir, f"loco_{held}_corrected_liblinear_tol1e6", elig_genes, beta_p, b0_p, outer_rec)
        loco_panels[held] = sel_prim
        loco_outer_rows.append(outer_rec)

        # Solver cross-check at best_C (liblinear tol=1e-5 is=1e4)
        clf_alt, conv_a, iter_a, sec_a = fit_logistic_model(
            X_tr, y_tr, C=best_C, l1_ratio=1.0, solver="liblinear", tol=1e-5, max_iter=10000, intercept_scaling=1e4
        )
        beta_a = clf_alt.coef_[0].astype(np.float64)
        b0_a = float(clf_alt.intercept_[0])
        kkt_a = exact_en_logistic_kkt(X_tr, y_tr, beta_a, b0_a, best_C, l1_ratio=1.0)
        sel_alt = {elig_genes[j] for j in np.flatnonzero(np.abs(beta_a) > COEF_CUTOFF)}
        loco_solver_check_rows.append({
            "outer_heldout": held,
            "best_C": best_C,
            "prim_genes": len(sel_prim),
            "alt_tol1e5_genes": len(sel_alt),
            "solver_jaccard": jaccard(sel_prim, sel_alt),
            "prim_obj": kkt_p["common_objective"],
            "alt_obj": kkt_a["common_objective"],
            "prim_kkt": kkt_p["max_kkt_residual"],
            "alt_kkt": kkt_a["max_kkt_residual"],
        })

        # Part C: Stage 22 controlled common classifier on corrected LOCO panel
        p_idx = np.array([gene_to_idx[g] for g in sorted(sel_prim)], dtype=int)
        s22_m, s22_p = stage22_eval(counts[tr_idx], y_tr, counts[te_idx], y_all[te_idx], ids_all[te_idx], p_idx, fold_idx, n_boot=n_boot)
        s22_m["cohort"] = held
        s22_m["best_C"] = best_C
        stage22_rows.append(s22_m)
        s22_p.insert(0, "cohort", held)
        stage22_preds.append(s22_p)

        # Incremental flush after every fold
        pd.DataFrame(loco_inner_rows).to_csv(out_dir / "loco_inner_tuning_scores.csv", index=False)
        pd.DataFrame(loco_outer_rows).to_csv(out_dir / "loco_outer_lasso_summary.csv", index=False)
        pd.DataFrame(stage22_rows).to_csv(out_dir / "stage22_corrected_lasso_cohort_metrics.csv", index=False)
        print(f"  LOCO {held}: best_C={best_C} inner_auroc={best_inner_auroc:.4f} genes={len(sel_prim)} kkt={kkt_p['max_kkt_residual']:.2e} s22_auroc={s22_m['auroc']:.4f} s22_bacc={s22_m['bacc']:.4f}", flush=True)

    pd.DataFrame(loco_solver_check_rows).to_csv(out_dir / "loco_solver_sensitivity_check.csv", index=False)
    pd.concat(stage22_preds, ignore_index=True).to_csv(out_dir / "stage22_corrected_lasso_predictions_PRIVATE.csv", index=False)

    # LOCO pairwise Jaccard
    loco_j_rows = []
    for a, b in itertools.combinations(COHORTS, 2):
        loco_j_rows.append({
            "pair": f"{a}-{b}",
            "size_a": len(loco_panels[a]),
            "size_b": len(loco_panels[b]),
            "intersection": len(loco_panels[a] & loco_panels[b]),
            "union": len(loco_panels[a] | loco_panels[b]),
            "jaccard": jaccard(loco_panels[a], loco_panels[b]),
            "overlap_coeff": overlap_coeff(loco_panels[a], loco_panels[b]),
        })
    loco_j_df = pd.DataFrame(loco_j_rows)
    loco_j_df.to_csv(out_dir / "loco_pairwise_jaccard.csv", index=False)
    loco_mean_j = float(loco_j_df["jaccard"].mean())
    all4_loco = sorted(set.intersection(*loco_panels.values()))
    (out_dir / "loco_genes_in_all_four_folds.txt").write_text("\n".join(all4_loco) + ("\n" if all4_loco else ""))

    s22_df = pd.DataFrame(stage22_rows)
    macro_rec = {
        "method": "Corrected Nested LASSO (liblinear tol=1e-6 is=1e4)",
        "mean_auroc": float(s22_df["auroc"].mean()),
        "sd_auroc": float(s22_df["auroc"].std(ddof=1)),
        "mean_auprc": float(s22_df["auprc"].mean()),
        "mean_bacc": float(s22_df["bacc"].mean()),
        "sd_bacc": float(s22_df["bacc"].std(ddof=1)),
        "mean_sensitivity": float(s22_df["sensitivity"].mean()),
        "mean_specificity": float(s22_df["specificity"].mean()),
        "mean_f1": float(s22_df["f1"].mean()),
        "mean_brier": float(s22_df["brier"].mean()),
        "mean_panel_size": float(s22_df["panel_size"].mean()),
        "min_panel_size": int(s22_df["panel_size"].min()),
        "max_panel_size": int(s22_df["panel_size"].max()),
        "mean_loco_jaccard": loco_mean_j,
        "genes_in_all_4_folds": len(all4_loco),
    }
    pd.DataFrame([macro_rec]).to_csv(out_dir / "stage22_corrected_lasso_macro_summary.csv", index=False)

    comp_rows = [
        {
            "method": "LASSO (historical SAGA tol=1e-3)",
            "macro_auroc": HIST_LOCO_LASSO["macro_auroc"],
            "macro_bacc": HIST_LOCO_LASSO["macro_bacc"],
            "mean_panel": 281.5,
            "loco_jaccard": HIST_LOCO_LASSO["loco_jaccard"],
        },
        {
            "method": "LASSO (corrected nested liblinear)",
            "macro_auroc": round(macro_rec["mean_auroc"], 4),
            "macro_bacc": round(macro_rec["mean_bacc"], 4),
            "mean_panel": round(macro_rec["mean_panel_size"], 2),
            "loco_jaccard": round(macro_rec["mean_loco_jaccard"], 4),
        },
        *HIST_OTHER_METHODS_STAGE22,
    ]
    pd.DataFrame(comp_rows).to_csv(out_dir / "comparison_vs_historical_and_four_methods_SENSITIVITY.csv", index=False)

    # =========================================================================
    # PART B: CORRECTED INDEPENDENT-COHORT LASSO (D1-D4)
    # =========================================================================
    print("=== PART B: CORRECTED INDEPENDENT-COHORT LASSO ===", flush=True)
    indep_cv_rows = []
    indep_summary_rows = []
    indep_panels = {}

    for c in COHORTS:
        idx = np.flatnonzero(cohorts_arr == c)
        x_c, y_c = counts[idx], y_all[idx]
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 47)
        for fold_num, (tr, va) in enumerate(cv.split(x_c, y_c), 1):
            X_tr_cv, cols_cv, sc_cv = fit_features(x_c[tr])
            X_va_cv = apply_features(x_c[va], cols_cv, sc_cv)
            for C_val in C_GRID:
                clf, conv, n_iter, sec = fit_logistic_model(
                    X_tr_cv, y_c[tr], C=C_val, l1_ratio=1.0, solver="liblinear", tol=1e-6, max_iter=10000, intercept_scaling=1e4
                )
                prob = clf.predict_proba(X_va_cv)[:, 1]
                auroc = float(roc_auc_score(y_c[va], prob))
                beta = clf.coef_[0].astype(np.float64)
                b0 = float(clf.intercept_[0])
                kkt = exact_en_logistic_kkt(X_tr_cv, y_c[tr], beta, b0, C_val, l1_ratio=1.0)
                indep_cv_rows.append({
                    "cohort": c,
                    "fold": fold_num,
                    "C": C_val,
                    "auroc": auroc,
                    "n_train": int(len(tr)),
                    "n_val": int(len(va)),
                    "eligible_genes": int(len(cols_cv)),
                    "n_selected_gt_1e8": int((np.abs(beta) > COEF_CUTOFF).sum()),
                    "converged": bool(conv),
                    "n_iter": int(n_iter),
                    **kkt,
                })

        cv_df = pd.DataFrame([r for r in indep_cv_rows if r["cohort"] == c])
        agg = cv_df.groupby("C", as_index=False)["auroc"].mean().sort_values(["auroc", "C"], ascending=[False, True])
        best_C = float(agg.iloc[0]["C"])
        best_cv_auroc = float(agg.iloc[0]["auroc"])

        X_full, full_cols, _ = fit_features(x_c)
        elig_genes = [genes[j] for j in full_cols]
        clf_f, conv_f, iter_f, sec_f = fit_logistic_model(
            X_full, y_c, C=best_C, l1_ratio=1.0, solver="liblinear", tol=1e-6, max_iter=10000, intercept_scaling=1e4
        )
        beta_f = clf_f.coef_[0].astype(np.float64)
        b0_f = float(clf_f.intercept_[0])
        kkt_f = exact_en_logistic_kkt(X_full, y_c, beta_f, b0_f, best_C, l1_ratio=1.0)
        ab_f = np.abs(beta_f)

        # Solver tolerance sensitivity at best_C
        clf_alt, _, _, _ = fit_logistic_model(
            X_full, y_c, C=best_C, l1_ratio=1.0, solver="liblinear", tol=1e-5, max_iter=10000, intercept_scaling=1e4
        )
        sel_alt = {elig_genes[j] for j in np.flatnonzero(np.abs(clf_alt.coef_[0]) > COEF_CUTOFF)}

        ind_rec = {
            "cohort": c,
            "n_samples": int(len(idx)),
            "eligible_genes": int(len(full_cols)),
            "best_C": best_C,
            "cv_mean_auroc": best_cv_auroc,
            "historical_table6_genes": HIST_INDEP_LASSO[c],
            "converged": bool(conv_f),
            "n_iter": int(iter_f),
            "elapsed_sec": round(sec_f, 2),
            "n_nonzero_exact": int((beta_f != 0.0).sum()),
            "n_gt_1e-8": int((ab_f > 1e-8).sum()),
            "n_gt_1e-4": int((ab_f > 1e-4).sum()),
            "n_gt_1e-3": int((ab_f > 1e-3).sum()),
            "n_gt_1e-2": int((ab_f > 1e-2).sum()),
            "max_abs_coef": float(ab_f.max()),
            "median_nonzero_coef": float(np.median(ab_f[ab_f > 1e-8])) if np.any(ab_f > 1e-8) else 0.0,
            "tol1e5_genes": len(sel_alt),
            **kkt_f,
        }
        sel_f = save_full_model_artifacts(out_dir, f"indep_{c}_corrected_liblinear_tol1e6", elig_genes, beta_f, b0_f, ind_rec)
        ind_rec["tol1e5_vs_1e6_jaccard"] = jaccard(sel_f, sel_alt)
        indep_panels[c] = sel_f
        indep_summary_rows.append(ind_rec)

        pd.DataFrame(indep_cv_rows).to_csv(out_dir / "indep_cv_tuning_scores.csv", index=False)
        pd.DataFrame(indep_summary_rows).to_csv(out_dir / "indep_lasso_summary.csv", index=False)
        print(f"  INDEP {c}: best_C={best_C} cv_auroc={best_cv_auroc:.4f} genes={len(sel_f)} (hist={HIST_INDEP_LASSO[c]}) kkt={kkt_f['max_kkt_residual']:.2e}", flush=True)

    indep_j_rows = []
    for a, b in itertools.combinations(COHORTS, 2):
        indep_j_rows.append({
            "pair": f"{a}-{b}",
            "size_a": len(indep_panels[a]),
            "size_b": len(indep_panels[b]),
            "intersection": len(indep_panels[a] & indep_panels[b]),
            "union": len(indep_panels[a] | indep_panels[b]),
            "jaccard": jaccard(indep_panels[a], indep_panels[b]),
            "overlap_coeff": overlap_coeff(indep_panels[a], indep_panels[b]),
        })
    indep_j_df = pd.DataFrame(indep_j_rows)
    indep_j_df.to_csv(out_dir / "indep_pairwise_jaccard.csv", index=False)
    indep_mean_j = float(indep_j_df["jaccard"].mean())

    # =========================================================================
    # PART D: TARGETED ELASTIC NET NUMERICAL DIAGNOSTIC
    # =========================================================================
    print("=== PART D: TARGETED ELASTIC NET NUMERICAL DIAGNOSTIC ===", flush=True)
    en_targets = [
        ("independent", "d2", 1.0, 0.5),
        ("independent", "d3", 1.0, 0.9),
        ("loco", "d1", 0.1, 0.5),
        ("loco", "d3", 0.1, 0.5),
    ]
    en_rows = []
    for exp, c, C_val, l1r in en_targets:
        tr_idx = np.flatnonzero(cohorts_arr == c) if exp == "independent" else np.flatnonzero(cohorts_arr != c)
        X_s, cols_s, _ = fit_features(counts[tr_idx])
        y_s = y_all[tr_idx]
        elig_g = [genes[j] for j in cols_s]

        en_sets = {}
        for cfg_name, tol_v, mi_v in [
            ("saga_tol1e-3_orig", 1e-3, 1600),
            ("saga_tol1e-5_tight", 1e-5, en_max_iter),
        ]:
            clf_e, conv_e, iter_e, sec_e = fit_logistic_model(
                X_s, y_s, C=C_val, l1_ratio=l1r, solver="saga", tol=tol_v, max_iter=mi_v
            )
            beta_e = clf_e.coef_[0].astype(np.float64)
            b0_e = float(clf_e.intercept_[0])
            kkt_e = exact_en_logistic_kkt(X_s, y_s, beta_e, b0_e, C_val, l1_ratio=l1r)
            ab_e = np.abs(beta_e)
            sel_e = {elig_g[j] for j in np.flatnonzero(ab_e > COEF_CUTOFF)}
            en_sets[cfg_name] = sel_e
            rec_e = {
                "experiment": exp,
                "cohort": c,
                "C": C_val,
                "l1_ratio": l1r,
                "config": cfg_name,
                "tol": tol_v,
                "max_iter": mi_v,
                "converged": bool(conv_e),
                "n_iter": int(iter_e),
                "elapsed_sec": round(sec_e, 2),
                "n_nonzero_exact": int((beta_e != 0.0).sum()),
                "n_gt_1e-8": int((ab_e > 1e-8).sum()),
                "n_gt_1e-3": int((ab_e > 1e-3).sum()),
                "max_abs_coef": float(ab_e.max()),
                **kkt_e,
            }
            en_rows.append(rec_e)
            print(f"  EN {exp}/{c} (C={C_val}, r={l1r}) [{cfg_name}]: conv={conv_e} iter={iter_e} genes={len(sel_e)} obj={kkt_e['common_objective']:.4f} kkt={kkt_e['max_kkt_residual']:.2e}", flush=True)

        # Attach within-target Jaccard between orig and tight SAGA
        j_en = jaccard(en_sets["saga_tol1e-3_orig"], en_sets["saga_tol1e-5_tight"])
        en_rows[-1]["jaccard_vs_orig"] = j_en
        en_rows[-2]["jaccard_vs_orig"] = 1.0
        pd.DataFrame(en_rows).to_csv(out_dir / "elastic_net_numerical_diagnostic.csv", index=False)

    # =========================================================================
    # MANIFEST, INTERPRETATION REPORT, HANDOFF & SHA256SUMS
    # =========================================================================
    en_df = pd.DataFrame(en_rows)
    en_orig_max_kkt = float(en_df[en_df["config"] == "saga_tol1e-3_orig"]["max_kkt_residual"].max())
    en_flag = "YES — SEPARATE CORRECTIVE EXPERIMENT REQUIRED" if en_orig_max_kkt > 0.1 else "NO — WITHIN TOLERANCE"

    interp = f"""# Corrected Nested LASSO & Elastic Net Diagnostic — Scientific Interpretation Report

**Created (UTC):** {datetime.now(timezone.utc).isoformat()}
**Status:** `independently-refit` (read-only against canonical dataset; original files untouched)

## 1. Corrected Nested LOCO LASSO (Part A & Part C — Table IV Context)
- **Outer-fold tuned C values:** {", ".join(f"{r['outer_heldout']}={r['best_C']}" for r in loco_outer_rows)}
- **Selected panel sizes (liblinear tol=1e-6, is=1e4):** {", ".join(f"{r['outer_heldout']}={r['n_gt_1e-8']}" for r in loco_outer_rows)} (mean {macro_rec['mean_panel_size']:.2f}; historical SAGA was 271/295/271/289, mean 281.50).
- **Stage 22 Controlled Common Classifier:**
  - Mean AUROC: **{macro_rec['mean_auroc']:.4f}** (SD {macro_rec['sd_auroc']:.4f}) vs. historical LASSO **0.6891** (delta {macro_rec['mean_auroc'] - 0.6891:+.4f}).
  - Mean Balanced Accuracy: **{macro_rec['mean_bacc']:.4f}** (SD {macro_rec['sd_bacc']:.4f}) vs. historical LASSO **0.6190**.
  - Mean LOCO 6-pair Jaccard: **{loco_mean_j:.4f}** vs. historical LASSO **0.2108** ({len(all4_loco)} genes in all 4 folds).
- *Scope caveat:* This is a corrected-LASSO-only sensitivity comparison against frozen historical outputs for the other four methods, not a full five-method rerun.

## 2. Corrected Independent-Cohort LASSO (Part B — Table VI Context)
- **Retuned C values:** {", ".join(f"{r['cohort']}={r['best_C']}" for r in indep_summary_rows)}
- **Selected panel sizes:** {", ".join(f"{r['cohort']}={r['n_gt_1e-8']} (hist {r['historical_table6_genes']})" for r in indep_summary_rows)}
- **Mean Independent-Cohort Jaccard:** **{indep_mean_j:.4f}** vs. historical Table VI **0.1635**.

## 3. Targeted Elastic Net Numerical Diagnostic (Part D)
- Maximum historical SAGA (`tol=1e-3`) Elastic Net KKT residual across tested fits: **{en_orig_max_kkt:.4f}**
- **Flag for separate Elastic Net corrective experiment:** **{en_flag}**
- See `elastic_net_numerical_diagnostic.csv` for per-target objective drops, KKT residuals, and support Jaccards.
"""
    (out_dir / "INTERPRETATION_REPORT.md").write_text(interp)

    handoff = f"""# Handoff for ChatGPT Independent Audit — Corrected Nested LASSO v1

**Run folder:** `peer_review_revision_2026/03_followup_experiments/corrected_nested_lasso_20261122_v1/`
**Execution status:** `independently-refit`

## What to Independently Verify
1. Verify `manifest.json` dataset/script SHA-256 hashes and `SHA256SUMS.txt`.
2. Recompute KKT residuals independently from `full_coefficients/*_all_coefficients.csv` and `checkpoints/*.json` (`intercept` saved).
3. Verify inner-CV tie-breaking and outer-test isolation in `loco_inner_tuning_scores.csv` and `indep_cv_tuning_scores.csv`.
4. Verify Stage 22 metrics and 2,000-resample bootstrap CIs from `stage22_corrected_lasso_predictions_PRIVATE.csv`.
5. Review `elastic_net_numerical_diagnostic.csv` to confirm whether Elastic Net requires a separate documented corrective experiment (`{en_flag}`).
"""
    (out_dir / "HANDOFF_FOR_CHATGPT_AUDIT.md").write_text(handoff)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "independently-refit",
        "script_name": Path(__file__).name,
        "script_sha256": sha_bytes(Path(__file__).read_bytes()),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "dataset": provenance,
        "config": {
            "c_grid": list(C_GRID),
            "primary_solver": "liblinear",
            "primary_tol": 1e-6,
            "primary_max_iter": 10000,
            "primary_intercept_scaling": 1e4,
            "seed": SEED,
            "stage22_seed": STAGE22_SEED,
            "n_bootstrap": n_boot,
            "coef_cutoff": COEF_CUTOFF,
            "en_tight_max_iter": en_max_iter,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Generate SHA256SUMS.txt across all generated files
    sums = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            rel = p.relative_to(out_dir).as_posix()
            sums.append(f"{sha_bytes(p.read_bytes())}  {rel}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    print(f"ALL PARTS A-D COMPLETE. Files written: {len(sums) + 1} in {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="Path to ibd_dataset_sample_aligned.zip or CSV dir")
    ap.add_argument("--out", type=Path, required=True, help="Versioned output directory")
    ap.add_argument("--skip-hash-check", action="store_true", help="Only for synthetic unit tests")
    ap.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for Stage 22 CIs")
    ap.add_argument("--en-max-iter", type=int, default=20000, help="Max iterations for tight Elastic Net SAGA check")
    args = ap.parse_args()
    run_all(args.data, args.out, verify_hashes=not args.skip_hash_check, n_boot=args.n_boot, en_max_iter=args.en_max_iter)

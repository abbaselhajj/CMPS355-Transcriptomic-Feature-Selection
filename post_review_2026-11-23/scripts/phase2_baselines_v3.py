#!/usr/bin/env python3
"""CMPS 355 — Phase 2 Baselines v3 (C1): Hardened Fail-Closed, Checkpointed, Audited Implementation.

Targeted v3 patches remediating all ChatGPT v2 adversarial preflight blockers:
  - V2-C1-P0a: `finalize_outputs` now enforces actual executable SHA-256, canonical dataset
    ZIP SHA-256, recomputed config_fingerprint, complete task-by-task validated checkpoints
    (all 48 baseline + 20 unique fold-K random tasks = 68 checkpoints in full production),
    exact canonical patient ID + label SHA-256 on every prediction CSV, exact metric
    recomputation (AUROC & BAcc <= 1e-10), exact random-draw row/index/seed/receipt ledger
    consistency, exact summary recomputation, exact DE ranking rows, and exact LOCO &
    Independent eligible-gene universe counts (rejects fabricated runs with FAKE_* IDs or
    zero checkpoints).
  - V2-C1-P0b: `load_validated_checkpoint` requires non-empty `expected_companion_keys`,
    `expected_task_id`, `out_dir` containment, self-checksummed `payload_sha256`, and
    non-empty verified companion files (rejects `companion_files={}` and arbitrary payloads).
  - Stage 22 Parity: Aligns exactly with Gate A audited Elastic Net v2 and 5-method common
    classifier: LogisticRegression(penalty='l2', C=1.0, class_weight='balanced',
    solver='liblinear', max_iter=5000, random_state=123) with sklearn default tol=1e-4.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_VERSION = "phase2_baselines_v3.0"
COHORTS = ("d1", "d2", "d3", "d4")
EXPECTED_COUNTS = {"d1": 260, "d2": 179, "d3": 76, "d4": 174}
EXPECTED_TRAIN_COUNTS = {"d1": 429, "d2": 510, "d3": 613, "d4": 515}
EXPECTED_LOCO_ELIGIBLE = {"d1": 8337, "d2": 8406, "d3": 8409, "d4": 8438}
EXPECTED_INDEP_ELIGIBLE = {"d1": 8395, "d2": 8334, "d3": 8419, "d4": 8164}
EXPECTED_SHARED_GENES = 9983
MIN_CPM = 1.0
MIN_FRACTION = 0.10  # >= 10% of training samples (consistent with audited LASSO & ENet v2)
FIXED_K_GRID = (10, 25, 50, 100, 200)

# Exact Stage 22 parity with audited Gate A Elastic Net v2 & 5-method common classifier
STAGE22_C = 1.0
STAGE22_MAX_ITER = 5000
STAGE22_SEED = 123
STAGE22_TOL_DESC = "sklearn_default_1e-4"

RANDOM_BASE_SEED = 20261122
MIN_PROD_RANDOM_DRAWS = 100
MIN_PROD_BOOT_REPS = 2000

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

# Immutable canonical (cohort, ID, y_true) SHA-256 digests verified against Gate A certified predictions
CANONICAL_COHORT_LABEL_SHA256 = {
    "d1": "89b088a597919359cf68036f699212e88dfa10c2b4ddca62fd087d7589d01329",
    "d2": "a96ab4a45a68706da42964a0f22881f2c2dc7baf17d7259a225d0eb497ae38c0",
    "d3": "961d947629aec3d2754f1831334bce6323abf05a76e2fb1c16a4f5b0ad550969",
    "d4": "914eaa22fe9ea15eaf85db315597ffba82d0b528a7e998664e3832dabee594e7",
}

AUDITED_COMPARATORS = {
    "Corrected Nested LASSO": {
        "source": "corrected_nested_lasso_20261122_v1 + CHATGPT_REUSED_5METHOD_2000BOOT_COHORT_METRICS.csv",
        "sizes": {"d1": 94, "d2": 77, "d3": 219, "d4": 90},
        "obs_auroc": {
            "d1": 0.8256880733944955,
            "d2": 0.6837479270315091,
            "d3": 0.7135854341736695,
            "d4": 0.6883556810162313,
        },
        "obs_bacc": {
            "d1": 0.6006989951944081,
            "d2": 0.6349087893864014,
            "d3": 0.5987394957983193,
            "d4": 0.6444601270289344,
        },
    },
    "Corrected Nested Elastic Net": {
        "source": "corrected_nested_elastic_net_20261122_v1 + CHATGPT_REUSED_5METHOD_2000BOOT_COHORT_METRICS.csv",
        "sizes": {"d1": 11, "d2": 28, "d3": 430, "d4": 28},
        "obs_auroc": {
            "d1": 0.8760375709916994,
            "d2": 0.669983416252073,
            "d3": 0.6477591036414566,
            "d4": 0.7306986591390262,
        },
        "obs_bacc": {
            "d1": 0.49770642201834864,
            "d2": 0.5939469320066335,
            "d3": 0.5931372549019608,
            "d4": 0.6482709950599859,
        },
    },
    "TENTACLES V2 (6/9 consensus)": {
        "source": "CMPS355_TENTACLES_V2_Final_Results_For_Analysis.zip (external_nmin6) + CHATGPT_REUSED_5METHOD_2000BOOT_COHORT_METRICS.csv",
        "sizes": {"d1": 73, "d2": 91, "d3": 81, "d4": 61},
        "obs_auroc": {
            "d1": 0.8722149410222805,
            "d2": 0.6630182421227198,
            "d3": 0.6659663865546218,
            "d4": 0.6889202540578687,
        },
        "obs_bacc": {
            "d1": 0.7910659676714722,
            "d2": 0.5844941956882255,
            "d3": 0.6274509803921569,
            "d4": 0.6172194777699365,
        },
    },
    "StableMate (significant-stable)": {
        "source": "runs/stablemate/d{1..4}_K2000_B1000/significant_stable_genes.csv + CHATGPT_REUSED_5METHOD_2000BOOT_COHORT_METRICS.csv",
        "sizes": {"d1": 20, "d2": 14, "d3": 14, "d4": 15},
        "obs_auroc": {
            "d1": 0.7554608999563128,
            "d2": 0.6077943615257048,
            "d3": 0.5735294117647058,
            "d4": 0.6993648553281581,
        },
        "obs_bacc": {
            "d1": 0.7309960681520314,
            "d2": 0.5502487562189055,
            "d3": 0.4747899159663866,
            "d4": 0.6357092448835568,
        },
    },
    "Stability Selection (cutoff 0.60)": {
        "source": "runs/stability_selection/d{1..4}/loco_training/gene_ranking.csv (selected_pi_0_6==True) + CHATGPT_REUSED_5METHOD_2000BOOT_COHORT_METRICS.csv",
        "sizes": {"d1": 4, "d2": 9, "d3": 7, "d4": 5},
        "obs_auroc": {
            "d1": 0.6277850589777196,
            "d2": 0.6391376451077944,
            "d3": 0.6855742296918768,
            "d4": 0.5611856033874383,
        },
        "obs_bacc": {
            "d1": 0.5963302752293578,
            "d2": 0.6128524046434494,
            "d3": 0.5322128851540616,
            "d4": 0.5097388849682427,
        },
    },
}


class LeakageViolationError(RuntimeError):
    pass


class DesignRankError(RuntimeError):
    pass


class CheckpointIntegrityError(RuntimeError):
    pass


class ConvergenceGateError(RuntimeError):
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


def build_config_fingerprint(n_random: int, n_boot: int, use_limma: bool, smoke_only: bool) -> str:
    cfg = {
        "script_version": SCRIPT_VERSION,
        "min_cpm": MIN_CPM,
        "min_fraction": MIN_FRACTION,
        "fixed_k_grid": list(FIXED_K_GRID),
        "stage22_C": STAGE22_C,
        "stage22_max_iter": STAGE22_MAX_ITER,
        "stage22_tol": STAGE22_TOL_DESC,
        "stage22_seed": STAGE22_SEED,
        "random_base_seed": RANDOM_BASE_SEED,
        "n_random": int(n_random),
        "n_boot": int(n_boot),
        "use_limma": bool(use_limma),
        "smoke_only": bool(smoke_only),
        "comparators": {k: v["sizes"] for k, v in AUDITED_COMPARATORS.items()},
    }
    return canonical_json_sha(cfg)


def compute_prediction_label_sha(pdf: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{str(r.cohort)},{str(r.ID)},{int(r.y_true)}"
        for r in pdf[["cohort", "ID", "y_true"]].itertuples(index=False)
    ).encode("utf-8")
    return sha_bytes(payload)


def compute_auroc_exact(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Dependency-free exact Mann-Whitney U AUROC (handles ties via average rank)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    require(n_pos > 0 and n_neg > 0, "AUROC requires both positive and negative classes")
    ranks = stats.rankdata(s, method="average")
    sum_pos_ranks = float(np.sum(ranks[y == 1]))
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_bacc_exact(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    pos = (y == 1)
    neg = (y == 0)
    tpr = float(np.mean(p[pos] == 1)) if np.any(pos) else 0.0
    tnr = float(np.mean(p[neg] == 0)) if np.any(neg) else 0.0
    return 0.5 * (tpr + tnr)


def assert_disjoint_train_test(train_ids: list[str], test_ids: list[str], train_cohorts: set[str], test_cohort: str) -> None:
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise LeakageViolationError(f"Sample ID leakage between train and test: {sorted(overlap)[:5]}")
    if test_cohort in train_cohorts:
        raise LeakageViolationError(f"Held-out cohort {test_cohort} present inside training cohorts {train_cohorts}")


def assert_design_matrix_full_rank(X_design: np.ndarray, col_names: list[str]) -> tuple[int, float]:
    require(X_design.ndim == 2 and X_design.shape[0] > X_design.shape[1], "Design matrix has insufficient rows", DesignRankError)
    rank = int(np.linalg.matrix_rank(X_design))
    if rank < X_design.shape[1]:
        raise DesignRankError(f"Design matrix is rank-deficient: rank={rank} < p={X_design.shape[1]} ({col_names})")
    cond = float(np.linalg.cond(X_design))
    if not np.isfinite(cond) or cond > 1e8:
        raise DesignRankError(f"Design matrix ill-conditioned: cond={cond}")
    return rank, cond


def load_canonical_dataset(zip_path: Path, verify_hashes: bool = True) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...], dict]:
    zip_path = Path(zip_path)
    require(zip_path.is_file(), f"Canonical dataset must be an immutable ZIP file (not directory/missing): {zip_path}")
    actual_zip_sha = sha_file(zip_path)
    if verify_hashes:
        require(actual_zip_sha == EXPECTED_ZIP_SHA, f"Dataset ZIP SHA-256 mismatch: expected={EXPECTED_ZIP_SHA}, got={actual_zip_sha}")
    with ZipFile(zip_path) as zf:
        require(zf.testzip() is None, "ZIP CRC check failed")
        raw = {name: zf.read(name) for name in EXPECTED_CSV_SHA}
    for name, expected in EXPECTED_CSV_SHA.items():
        got = sha_bytes(raw[name])
        if verify_hashes:
            require(got == expected, f"CSV SHA-256 mismatch for {name}: expected={expected}, got={got}")
    headers = {}
    for c in COHORTS:
        names = next(csv.reader(io.StringIO(raw[f"count_{c}_aligned.csv"].decode("utf-8-sig"))))
        require(names[0] == "ID" and len(names) == len(set(names)), f"Invalid gene header in {c}")
        headers[c] = names
    genes = tuple(sorted(set.intersection(*(set(headers[c][1:]) for c in COHORTS))))
    require(len(genes) == EXPECTED_SHARED_GENES, f"Expected {EXPECTED_SHARED_GENES} shared genes, got {len(genes)}")
    counts_list, meta_list = [], []
    for c in COHORTS:
        desc = pd.read_csv(io.BytesIO(raw[f"desc_{c}.csv"]), dtype={"ID": "string", "Study": "string", "class": "string"})
        df = pd.read_csv(io.BytesIO(raw[f"count_{c}_aligned.csv"]), usecols=["ID", *genes], dtype={"ID": "string"})
        require(len(desc) == len(df) == EXPECTED_COUNTS[c], f"Sample count mismatch for {c}")
        require(desc["ID"].tolist() == df["ID"].tolist(), f"ID alignment mismatch for {c}")
        require(set(desc["class"]) == {"CD", "Not IBD"}, f"Unexpected class labels in {c}")
        mat = df.loc[:, list(genes)].to_numpy(dtype=np.float64, copy=True)
        require(np.isfinite(mat).all() and (mat >= 0).all() and (mat.sum(axis=1) > 0).all(), f"Invalid counts in {c}")
        dsub = desc[["ID", "Study", "class"]].copy()
        dsub.insert(0, "cohort", c)
        dsub["y"] = (dsub["class"] == "CD").astype(int)
        meta_list.append(dsub)
        counts_list.append(mat)
    meta = pd.concat(meta_list, ignore_index=True)
    require(meta["ID"].is_unique and len(meta) == 689, "Patient count/uniqueness failed")
    return meta, np.vstack(counts_list), genes, {
        "zip_path": str(zip_path),
        "zip_sha256": actual_zip_sha,
        "shared_genes": len(genes),
        "n_samples": len(meta),
    }


def compute_cpm_and_log2(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lib = counts.sum(axis=1, keepdims=True)
    require((lib > 0).all(), "Zero library size encountered")
    cpm = counts / lib * 1e6
    return cpm, np.log2(cpm + 1.0)


def get_eligible_mask(train_cpm: np.ndarray, min_cpm: float = MIN_CPM, min_fraction: float = MIN_FRACTION) -> np.ndarray:
    return (train_cpm > min_cpm).mean(axis=0) >= min_fraction


def export_eligible_universes(
    meta: pd.DataFrame,
    counts: np.ndarray,
    genes: tuple[str, ...],
    out_dir: Path,
) -> dict:
    """Export exact per-fold LOCO and Independent-cohort eligible-gene universes for C2/C3."""
    univ_dir = out_dir / "eligible_universes"
    univ_dir.mkdir(parents=True, exist_ok=True)
    cpm_all, _ = compute_cpm_and_log2(counts)
    manifest = {"min_cpm": MIN_CPM, "min_fraction": MIN_FRACTION, "loco": {}, "indep": {}}
    for c in COHORTS:
        loco_mask = (meta["cohort"] != c).to_numpy()
        elig_loco = get_eligible_mask(cpm_all[loco_mask])
        g_loco = [genes[i] for i in np.flatnonzero(elig_loco)]
        require(len(g_loco) == EXPECTED_LOCO_ELIGIBLE[c], f"LOCO {c} eligible count {len(g_loco)} != {EXPECTED_LOCO_ELIGIBLE[c]}")
        p_loco = univ_dir / f"loco_{c}_eligible_genes.csv"
        atomic_write_csv(p_loco, pd.DataFrame({"gene": g_loco}))
        manifest["loco"][c] = {"filename": p_loco.name, "n_genes": len(g_loco), "sha256": sha_file(p_loco)}

        indep_mask = (meta["cohort"] == c).to_numpy()
        elig_indep = get_eligible_mask(cpm_all[indep_mask])
        g_indep = [genes[i] for i in np.flatnonzero(elig_indep)]
        require(len(g_indep) == EXPECTED_INDEP_ELIGIBLE[c], f"Independent {c} eligible count {len(g_indep)} != {EXPECTED_INDEP_ELIGIBLE[c]}")
        p_indep = univ_dir / f"indep_{c}_eligible_genes.csv"
        atomic_write_csv(p_indep, pd.DataFrame({"gene": g_indep}))
        manifest["indep"][c] = {"filename": p_indep.name, "n_genes": len(g_indep), "sha256": sha_file(p_indep)}

    atomic_write_json(univ_dir / "eligible_universes_manifest.json", manifest)
    return manifest


def rank_welch_training_only(X_train_log2: np.ndarray, y_train: np.ndarray, gene_names: list[str]) -> pd.DataFrame:
    cd = X_train_log2[y_train == 1]
    ctrl = X_train_log2[y_train == 0]
    require(len(cd) >= 2 and len(ctrl) >= 2, "Insufficient samples per class for Welch t-test")
    t_stat, p_val = stats.ttest_ind(cd, ctrl, axis=0, equal_var=False)
    t_stat = np.where(np.isfinite(t_stat), t_stat, 0.0)
    p_val = np.where(np.isfinite(p_val), p_val, 1.0)
    abs_t = np.abs(t_stat)
    df = pd.DataFrame({"gene": gene_names, "t_stat": t_stat, "abs_t": abs_t, "p_value": p_val})
    df = df.sort_values(by=["p_value", "abs_t", "gene"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def rank_limma_voom_training_only(train_counts_elig: np.ndarray, train_meta: pd.DataFrame, elig_genes: list[str]) -> pd.DataFrame:
    y_cd = (train_meta["class"] == "CD").astype(float).to_numpy()
    y_ctrl = (train_meta["class"] == "Not IBD").astype(float).to_numpy()
    cohorts_present = sorted(train_meta["cohort"].unique().tolist())
    require(len(cohorts_present) == 3, f"Expected 3 training cohorts in LOCO fold, got {cohorts_present}", DesignRankError)
    cols = [y_cd, y_ctrl]
    col_names = ["classCD", "classNotIBD"]
    for ch in cohorts_present[1:]:
        cols.append((train_meta["cohort"] == ch).astype(float).to_numpy())
        col_names.append(f"cohort_{ch}")
    assert_design_matrix_full_rank(np.column_stack(cols), col_names)

    with tempfile.TemporaryDirectory(prefix="limma_voom_") as tmpdir:
        tmp = Path(tmpdir)
        counts_csv = tmp / "train_counts_elig.csv"
        meta_csv = tmp / "train_meta.csv"
        out_csv = tmp / "limma_ranked.csv"
        df_c = pd.DataFrame(train_counts_elig.T, index=elig_genes, columns=train_meta["ID"].tolist())
        df_c.index.name = "gene"
        df_c.to_csv(counts_csv)
        pd.DataFrame({
            "ID": train_meta["ID"].tolist(),
            "class_clean": ["CD" if x == "CD" else "NotIBD" for x in train_meta["class"]],
            "cohort": train_meta["cohort"].tolist(),
        }).to_csv(meta_csv, index=False)

        r_code = f"""
suppressPackageStartupMessages({{
  library(edgeR)
  library(limma)
}})
cnt <- read.csv("{counts_csv}", row.names=1, check.names=FALSE)
meta <- read.csv("{meta_csv}", stringsAsFactors=FALSE)
stopifnot(identical(colnames(cnt), meta$ID))
cls <- factor(meta$class_clean, levels=c("CD", "NotIBD"))
coh <- factor(meta$cohort)
design <- model.matrix(~ 0 + cls + coh)
colnames(design) <- make.names(colnames(design))
if (qr(design)$rank < ncol(design)) stop("Rank-deficient design matrix in R limma-voom")
dge <- DGEList(counts=as.matrix(cnt))
dge <- calcNormFactors(dge, method="TMM")
v <- voom(dge, design=design, plot=FALSE)
fit <- lmFit(v, design)
contr <- makeContrasts(CD_vs_NotIBD = clsCD - clsNotIBD, levels=design)
fit2 <- contrasts.fit(fit, contr)
fit2 <- eBayes(fit2)
tt <- topTable(fit2, coef="CD_vs_NotIBD", number=Inf, sort.by="none")
out <- data.frame(
  gene = rownames(tt),
  logFC = tt$logFC,
  t_stat = tt$t,
  abs_t = abs(tt$t),
  p_value = tt$P.Value,
  adj_p_val = tt$adj.P.Val,
  stringsAsFactors = FALSE
)
out <- out[order(out$p_value, -out$abs_t, out$gene), ]
out$rank <- seq_len(nrow(out))
write.csv(out, "{out_csv}", row.names=FALSE)
"""
        r_script = tmp / "run_limma_voom.R"
        r_script.write_text(r_code, encoding="utf-8")
        proc = subprocess.run(["Rscript", str(r_script)], capture_output=True, text=True)
        require(proc.returncode == 0, f"R limma-voom failed (rc={proc.returncode}): {proc.stderr[-1200:]}")
        res = pd.read_csv(out_csv)
        require(len(res) == len(elig_genes), f"limma-voom row mismatch: {len(res)} != {len(elig_genes)}")
        require(np.isfinite(res["p_value"]).all() and np.isfinite(res["t_stat"]).all(), "Non-finite limma-voom statistics")
        return res


def stratified_bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_boot: int, seed: int) -> tuple[float, float]:
    if n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        vals[b] = float(metric_fn(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def eval_stage22_panel(
    X_train_log2: np.ndarray,
    y_train: np.ndarray,
    X_test_log2: np.ndarray,
    y_test: np.ndarray,
    test_ids: list[str],
    cohort: str,
    fold_idx: int,
    n_boot: int,
    failed_dir: Path | None = None,
    task_label: str = "panel",
) -> tuple[dict, pd.DataFrame]:
    """Exact Gate A Stage 22 parity: balanced L2 liblinear (C=1.0, max_iter=5000, random_state=123, default tol=1e-4)."""
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
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_log2)
    X_te = scaler.transform(X_test_log2)

    clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=STAGE22_C,
        class_weight="balanced",
        max_iter=STAGE22_MAX_ITER,
        random_state=STAGE22_SEED,
    )
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_tr, y_train)
        has_conv_warn = any(issubclass(w.category, ConvergenceWarning) for w in wlog)

    n_iter = int(clf.n_iter_[0])
    coef = np.asarray(clf.coef_[0], dtype=np.float64)
    converged = (not has_conv_warn) and (n_iter < STAGE22_MAX_ITER) and bool(np.isfinite(coef).all())
    prob = clf.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)

    probs_valid = bool(np.isfinite(prob).all() and (prob >= 0.0).all() and (prob <= 1.0).all())
    if (not converged) or (not probs_valid):
        diag = {
            "cohort": cohort,
            "task_label": task_label,
            "converged": converged,
            "has_conv_warn": has_conv_warn,
            "n_iter": n_iter,
            "max_iter": STAGE22_MAX_ITER,
            "probs_valid": probs_valid,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if failed_dir is not None:
            atomic_write_json(failed_dir / f"FAILED_{cohort}_{task_label}.json", diag)
        raise ConvergenceGateError(f"Stage 22 convergence/validity gate FAILED for {cohort}/{task_label}: {diag}")

    auroc = float(roc_auc_score(y_test, prob))
    auprc = float(average_precision_score(y_test, prob))
    bacc = float(balanced_accuracy_score(y_test, pred))
    sens = float(recall_score(y_test, pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(precision_score(y_test, pred, zero_division=0))
    f1 = float(f1_score(y_test, pred, zero_division=0))
    brier = float(brier_score_loss(y_test, prob))

    for mname, mval in [("auroc", auroc), ("auprc", auprc), ("bacc", bacc), ("f1", f1), ("brier", brier)]:
        if not np.isfinite(mval):
            raise ConvergenceGateError(f"Non-finite metric {mname}={mval} for {cohort}/{task_label}")

    auroc_lo, auroc_hi = stratified_bootstrap_ci(y_test, prob, roc_auc_score, n_boot, seed=123 + fold_idx)
    bacc_lo, bacc_hi = stratified_bootstrap_ci(y_test, pred, balanced_accuracy_score, n_boot, seed=223 + fold_idx)

    metrics_row = {
        "cohort": cohort,
        "n_test": int(len(y_test)),
        "panel_size": int(X_train_log2.shape[1]),
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
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "bootstrap_reps": int(n_boot),
        "stage22_converged": True,
        "stage22_n_iter": n_iter,
    }
    pred_df = pd.DataFrame({
        "cohort": cohort,
        "ID": test_ids,
        "y_true": y_test.astype(int),
        "prob_CD": prob,
        "pred_CD": pred,
    })
    return metrics_row, pred_df


def save_validated_checkpoint(
    ckpt_path: Path,
    task_id: str,
    payload: dict,
    script_sha: str,
    dataset_sha: str,
    config_fp: str,
    companion_files: dict[str, Path],
    out_dir: Path,
) -> None:
    """Atomically save a self-checksummed checkpoint with mandatory non-empty companion inventory."""
    require(bool(companion_files), f"Checkpoint {task_id} must declare non-empty companion_files", CheckpointIntegrityError)
    out_root = out_dir.resolve()
    companion_hashes = {}
    for key, p in companion_files.items():
        rp = p.resolve()
        require(rp.is_file() and rp.stat().st_size > 0, f"Companion artifact missing/empty ({key}): {rp}", CheckpointIntegrityError)
        try:
            rel = rp.relative_to(out_root).as_posix()
        except ValueError as exc:
            raise CheckpointIntegrityError(f"Companion path escapes out_dir: {rp}") from exc
        companion_hashes[key] = {"rel_path": rel, "size_bytes": int(rp.stat().st_size), "sha256": sha_file(rp)}

    record = {
        "schema_version": "c1_checkpoint_v3",
        "status": "COMPLETED",
        "task_id": str(task_id),
        "execution_script_sha256": script_sha,
        "dataset_sha256": dataset_sha,
        "config_fingerprint": config_fp,
        "script_version": SCRIPT_VERSION,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "companion_files": companion_hashes,
        "payload_sha256": canonical_json_sha(payload),
        "payload": payload,
    }
    atomic_write_json(ckpt_path, record)


def load_validated_checkpoint(
    ckpt_path: Path,
    script_sha: str,
    dataset_sha: str,
    config_fp: str,
    expected_task_id: str,
    expected_companion_keys: frozenset[str] | set[str],
    out_dir: Path,
) -> dict | None:
    """Fail-closed checkpoint loader rejecting empty/missing/extra companions, wrong task_id, or payload tamper."""
    require(bool(expected_companion_keys), "expected_companion_keys must be non-empty", CheckpointIntegrityError)
    if not ckpt_path.exists():
        return None
    try:
        rec = json.loads(ckpt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CheckpointIntegrityError(f"Corrupt checkpoint JSON {ckpt_path}: {exc}") from exc

    require(rec.get("schema_version") == "c1_checkpoint_v3", f"Invalid checkpoint schema_version in {ckpt_path}", CheckpointIntegrityError)
    require(rec.get("status") == "COMPLETED", f"Checkpoint not COMPLETED: {ckpt_path}", CheckpointIntegrityError)
    require(rec.get("task_id") == expected_task_id, f"Checkpoint task_id mismatch: got={rec.get('task_id')}, expected={expected_task_id}", CheckpointIntegrityError)
    require(rec.get("execution_script_sha256") == script_sha, f"Checkpoint script SHA mismatch: {ckpt_path}", CheckpointIntegrityError)
    require(rec.get("dataset_sha256") == dataset_sha, f"Checkpoint dataset SHA mismatch: {ckpt_path}", CheckpointIntegrityError)
    require(rec.get("config_fingerprint") == config_fp, f"Checkpoint config fingerprint mismatch: {ckpt_path}", CheckpointIntegrityError)

    payload = rec.get("payload")
    require(isinstance(payload, dict), f"Checkpoint payload missing/not dict: {ckpt_path}", CheckpointIntegrityError)
    require(rec.get("payload_sha256") == canonical_json_sha(payload), f"Checkpoint payload_sha256 mismatch (tampered payload): {ckpt_path}", CheckpointIntegrityError)

    companions = rec.get("companion_files")
    require(isinstance(companions, dict) and len(companions) > 0, f"Checkpoint has empty/missing companion_files: {ckpt_path}", CheckpointIntegrityError)
    require(set(companions.keys()) == set(expected_companion_keys), f"Checkpoint companion keys mismatch: got={set(companions.keys())}, expected={set(expected_companion_keys)}", CheckpointIntegrityError)

    out_root = out_dir.resolve()
    for key, info in companions.items():
        require(isinstance(info, dict) and "rel_path" in info and "sha256" in info, f"Invalid companion entry '{key}' in {ckpt_path}", CheckpointIntegrityError)
        p = (out_root / info["rel_path"]).resolve()
        try:
            p.relative_to(out_root)
        except ValueError as exc:
            raise CheckpointIntegrityError(f"Companion path escapes out_dir ({key}): {p}") from exc
        require(p.is_file() and p.stat().st_size > 0, f"Checkpoint companion file missing/empty ({key}): {p}", CheckpointIntegrityError)
        require(sha_file(p) == info["sha256"], f"Checkpoint companion SHA mismatch ({key}): {p}", CheckpointIntegrityError)

    return payload


def run_all_baselines(
    data_path: Path,
    out_dir: Path,
    n_random: int = 100,
    n_boot: int = 2000,
    use_limma: bool = True,
    smoke_only: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    pred_dir = out_dir / "predictions_PRIVATE"
    rand_genes_dir = out_dir / "random_draw_gene_lists"
    failed_dir = out_dir / "failed_fits"
    for d in (ckpt_dir, pred_dir, rand_genes_dir):
        d.mkdir(parents=True, exist_ok=True)

    script_sha = sha_file(Path(__file__))
    meta, counts, genes, data_prov = load_canonical_dataset(data_path, verify_hashes=True)
    dataset_sha = data_prov["zip_sha256"]
    config_fp = build_config_fingerprint(n_random=n_random, n_boot=n_boot, use_limma=use_limma, smoke_only=smoke_only)

    run_meta_path = out_dir / "run_metadata.json"
    if run_meta_path.exists():
        prev = json.loads(run_meta_path.read_text(encoding="utf-8"))
        require(prev.get("execution_script_sha256") == script_sha, "Existing output directory has different execution_script_sha256", CheckpointIntegrityError)
        require(prev.get("dataset_sha256") == dataset_sha, "Existing output directory has different dataset_sha256", CheckpointIntegrityError)
        require(prev.get("config_fingerprint") == config_fp, "Existing output directory has different config_fingerprint", CheckpointIntegrityError)
    else:
        atomic_write_json(run_meta_path, {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_name": Path(__file__).name,
            "script_version": SCRIPT_VERSION,
            "execution_script_sha256": script_sha,
            "dataset_sha256": dataset_sha,
            "config_fingerprint": config_fp,
            "run_mode": "smoke" if smoke_only else "full",
            "use_limma": bool(use_limma),
            "n_random": int(n_random),
            "n_boot": int(n_boot),
            "min_cpm": MIN_CPM,
            "min_fraction": MIN_FRACTION,
            "stage22_C": STAGE22_C,
            "stage22_seed": STAGE22_SEED,
            "stage22_tol": STAGE22_TOL_DESC,
            "cohorts_planned": ["d3"] if smoke_only else list(COHORTS),
        })

    export_eligible_universes(meta, counts, genes, out_dir)
    cpm_all, log2_all = compute_cpm_and_log2(counts)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    target_cohorts = ("d3",) if smoke_only else COHORTS

    all_de_rows = []
    all_rand_draw_rows = []

    for cohort in target_cohorts:
        fold_idx = COHORTS.index(cohort)
        test_mask = (meta["cohort"] == cohort).to_numpy()
        train_mask = ~test_mask
        train_meta = meta.loc[train_mask].reset_index(drop=True)
        test_meta = meta.loc[test_mask].reset_index(drop=True)
        assert_disjoint_train_test(
            train_meta["ID"].tolist(),
            test_meta["ID"].tolist(),
            set(train_meta["cohort"].unique()),
            cohort,
        )
        y_train = train_meta["y"].to_numpy()
        y_test = test_meta["y"].to_numpy()
        test_ids = test_meta["ID"].tolist()

        elig_mask = get_eligible_mask(cpm_all[train_mask])
        elig_indices = np.flatnonzero(elig_mask)
        elig_genes = [genes[i] for i in elig_indices]

        # 1. All-9,983 and Train-Eligible baselines
        for bname, col_idx in [("all_9983", np.arange(len(genes))), ("train_eligible", elig_indices)]:
            task_id = f"{cohort}__{bname}"
            ckpt_p = ckpt_dir / f"{task_id}.json"
            pred_p = pred_dir / f"pred_{task_id}_PRIVATE.csv"
            cached = load_validated_checkpoint(ckpt_p, script_sha, dataset_sha, config_fp, task_id, {"pred"}, out_dir)
            if cached is not None:
                all_de_rows.append(cached)
            else:
                row, pred_df = eval_stage22_panel(
                    log2_all[train_mask][:, col_idx], y_train,
                    log2_all[test_mask][:, col_idx], y_test,
                    test_ids, cohort, fold_idx, n_boot, failed_dir=failed_dir, task_label=bname,
                )
                row["baseline"] = bname
                row["K"] = int(len(col_idx))
                atomic_write_csv(pred_p, pred_df)
                save_validated_checkpoint(ckpt_p, task_id, row, script_sha, dataset_sha, config_fp, {"pred": pred_p}, out_dir)
                all_de_rows.append(row)

        # 2. Training-only DE rankings
        de_tables = {}
        de_tables["welch"] = rank_welch_training_only(log2_all[train_mask][:, elig_indices], y_train, elig_genes)
        atomic_write_csv(out_dir / f"de_ranking_{cohort}_welch.csv", de_tables["welch"])
        if use_limma:
            de_tables["limma_voom"] = rank_limma_voom_training_only(counts[train_mask][:, elig_indices], train_meta, elig_genes)
            atomic_write_csv(out_dir / f"de_ranking_{cohort}_limma_voom.csv", de_tables["limma_voom"])

        for de_method, rank_df in de_tables.items():
            for K in FIXED_K_GRID:
                task_id = f"{cohort}__{de_method}_K{K}"
                ckpt_p = ckpt_dir / f"{task_id}.json"
                pred_p = pred_dir / f"pred_{task_id}_PRIVATE.csv"
                panel_p = out_dir / f"panel_{task_id}.csv"
                cached = load_validated_checkpoint(ckpt_p, script_sha, dataset_sha, config_fp, task_id, {"pred", "panel"}, out_dir)
                if cached is not None:
                    all_de_rows.append(cached)
                else:
                    top_genes = rank_df["gene"].iloc[:K].tolist()
                    col_idx = np.array([gene_to_idx[g] for g in top_genes], dtype=int)
                    row, pred_df = eval_stage22_panel(
                        log2_all[train_mask][:, col_idx], y_train,
                        log2_all[test_mask][:, col_idx], y_test,
                        test_ids, cohort, fold_idx, n_boot, failed_dir=failed_dir, task_label=f"{de_method}_K{K}",
                    )
                    row["baseline"] = de_method
                    row["K"] = int(K)
                    atomic_write_csv(pred_p, pred_df)
                    atomic_write_csv(panel_p, pd.DataFrame({"rank": np.arange(1, K + 1), "gene": top_genes}))
                    save_validated_checkpoint(ckpt_p, task_id, row, script_sha, dataset_sha, config_fp, {"pred": pred_p, "panel": panel_p}, out_dir)
                    all_de_rows.append(row)

        # 3. Size-matched random panels across ALL 5 audited comparator methods
        unique_ks = sorted({v["sizes"][cohort] for v in AUDITED_COMPARATORS.values()})
        for K in unique_ks:
            task_id = f"{cohort}__random_K{K}_B{n_random}"
            ckpt_p = ckpt_dir / f"{task_id}.json"
            draws_p = out_dir / f"random_draws_{task_id}.csv"
            genes_p = rand_genes_dir / f"random_genes_{task_id}.json"
            cached = load_validated_checkpoint(ckpt_p, script_sha, dataset_sha, config_fp, task_id, {"draws", "genes"}, out_dir)
            if cached is not None:
                all_rand_draw_rows.extend(cached["draws"])
            else:
                draw_rows = []
                draw_gene_receipts = {}
                for b in range(n_random):
                    draw_seed = RANDOM_BASE_SEED + fold_idx * 100000 + K * 1000 + b
                    rng = np.random.default_rng(draw_seed)
                    picked_pos = rng.choice(len(elig_genes), size=K, replace=False)
                    picked_genes = sorted(elig_genes[p] for p in picked_pos)
                    col_idx = np.array([gene_to_idx[g] for g in picked_genes], dtype=int)
                    row, _ = eval_stage22_panel(
                        log2_all[train_mask][:, col_idx], y_train,
                        log2_all[test_mask][:, col_idx], y_test,
                        test_ids, cohort, fold_idx, n_boot=0, failed_dir=failed_dir, task_label=f"random_K{K}_b{b}",
                    )
                    genes_sha = sha_bytes("\n".join(picked_genes).encode("utf-8"))
                    draw_gene_receipts[str(b)] = {"seed": draw_seed, "genes_sha256": genes_sha, "genes": picked_genes}
                    draw_rows.append({
                        "cohort": cohort,
                        "K": int(K),
                        "draw_idx": int(b),
                        "seed": int(draw_seed),
                        "panel_genes_sha256": genes_sha,
                        "auroc": row["auroc"],
                        "bacc": row["bacc"],
                        "auprc": row["auprc"],
                        "f1": row["f1"],
                        "brier": row["brier"],
                        "stage22_converged": True,
                        "stage22_n_iter": row["stage22_n_iter"],
                    })
                atomic_write_csv(draws_p, pd.DataFrame(draw_rows))
                atomic_write_json(genes_p, draw_gene_receipts)
                save_validated_checkpoint(
                    ckpt_p, task_id,
                    {"cohort": cohort, "K": K, "n_random": n_random, "draws": draw_rows},
                    script_sha, dataset_sha, config_fp,
                    {"draws": draws_p, "genes": genes_p}, out_dir,
                )
                all_rand_draw_rows.extend(draw_rows)

    atomic_write_csv(out_dir / "baseline_all_gene_and_de_metrics.csv", pd.DataFrame(all_de_rows))
    df_rand_draws = pd.DataFrame(all_rand_draw_rows)
    atomic_write_csv(out_dir / "baseline_random_panel_draws.csv", df_rand_draws)

    summary_rows = []
    for comp_name, cinfo in AUDITED_COMPARATORS.items():
        for cohort in target_cohorts:
            K = cinfo["sizes"][cohort]
            sub = df_rand_draws[(df_rand_draws["cohort"] == cohort) & (df_rand_draws["K"] == K)]
            require(len(sub) == n_random, f"Random draw count mismatch for {comp_name}/{cohort}/K={K}")
            obs_auc = float(cinfo["obs_auroc"][cohort])
            obs_bacc = float(cinfo["obs_bacc"][cohort])
            aucs = sub["auroc"].to_numpy(dtype=np.float64)
            baccs = sub["bacc"].to_numpy(dtype=np.float64)
            q25_a, q50_a, q75_a = np.percentile(aucs, [25, 50, 75])
            p05_a, p95_a = np.percentile(aucs, [5, 95])
            q25_b, q50_b, q75_b = np.percentile(baccs, [25, 50, 75])
            p05_b, p95_b = np.percentile(baccs, [5, 95])
            summary_rows.append({
                "comparator_method": comp_name,
                "comparator_source": cinfo["source"],
                "cohort": cohort,
                "K": int(K),
                "n_draws": int(len(aucs)),
                "interval_scope": "within_cohort_empirical_null_B_draws",
                "observed_auroc": obs_auc,
                "null_auroc_mean": float(aucs.mean()),
                "null_auroc_sd": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
                "null_auroc_median": float(q50_a),
                "null_auroc_iqr": float(q75_a - q25_a),
                "null_auroc_p05": float(p05_a),
                "null_auroc_p95": float(p95_a),
                "emp_p_auroc": float((1 + np.sum(aucs >= obs_auc)) / (len(aucs) + 1)),
                "observed_bacc": obs_bacc,
                "null_bacc_mean": float(baccs.mean()),
                "null_bacc_sd": float(baccs.std(ddof=1)) if len(baccs) > 1 else 0.0,
                "null_bacc_median": float(q50_b),
                "null_bacc_iqr": float(q75_b - q25_b),
                "null_bacc_p05": float(p05_b),
                "null_bacc_p95": float(p95_b),
                "emp_p_bacc": float((1 + np.sum(baccs >= obs_bacc)) / (len(baccs) + 1)),
            })
    atomic_write_csv(out_dir / "baseline_random_panel_summary.csv", pd.DataFrame(summary_rows))
    return finalize_outputs(out_dir, allow_smoke_status=smoke_only)


def finalize_outputs(out_dir: Path, allow_smoke_status: bool = False) -> dict:
    """Hardened fail-closed finalizer: verifies real hashes, all checkpoints, canonical patient IDs/labels,
    exact metric recomputation, full random-draw ledger & receipts, DE rankings, and exact universes."""
    out_dir = Path(out_dir)
    run_meta_path = out_dir / "run_metadata.json"
    require(run_meta_path.is_file(), f"Missing run_metadata.json in {out_dir}", CertificationError)
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    for k in ("execution_script_sha256", "dataset_sha256", "config_fingerprint", "run_mode", "use_limma", "n_random", "n_boot"):
        require(k in run_meta and run_meta[k] is not None, f"Missing provenance key '{k}' in run_metadata.json", CertificationError)

    # 1. Verify REAL canonical dataset SHA, executable script SHA, and recomputed config fingerprint
    actual_script_sha = sha_file(Path(__file__))
    require(run_meta["dataset_sha256"] == EXPECTED_ZIP_SHA,
            f"run_metadata dataset_sha256 != canonical EXPECTED_ZIP_SHA: {run_meta['dataset_sha256']}", CertificationError)
    require(run_meta["execution_script_sha256"] == actual_script_sha,
            f"run_metadata execution_script_sha256 != actual script SHA: {run_meta['execution_script_sha256']}", CertificationError)
    smoke_flag = (run_meta["run_mode"] == "smoke")
    expected_fp = build_config_fingerprint(
        n_random=int(run_meta["n_random"]),
        n_boot=int(run_meta["n_boot"]),
        use_limma=bool(run_meta["use_limma"]),
        smoke_only=smoke_flag,
    )
    require(run_meta["config_fingerprint"] == expected_fp,
            f"run_metadata config_fingerprint mismatch: got={run_meta['config_fingerprint']}, expected={expected_fp}", CertificationError)

    is_smoke = (
        run_meta["run_mode"] != "full"
        or not bool(run_meta["use_limma"])
        or int(run_meta["n_random"]) < MIN_PROD_RANDOM_DRAWS
        or int(run_meta["n_boot"]) < MIN_PROD_BOOT_REPS
    )
    if is_smoke and not allow_smoke_status:
        raise CertificationError(
            "Refusing to certify non-production/smoke run as CERTIFIED_COMPLETE "
            f"(run_mode={run_meta['run_mode']}, use_limma={run_meta['use_limma']}, "
            f"n_random={run_meta['n_random']}, n_boot={run_meta['n_boot']})"
        )

    expected_cohorts = tuple(run_meta.get("cohorts_planned", COHORTS))
    if not is_smoke:
        require(expected_cohorts == COHORTS, f"Production run must cover all 4 cohorts {COHORTS}, got {expected_cohorts}", CertificationError)

    n_random = int(run_meta["n_random"])
    n_boot = int(run_meta["n_boot"])
    use_limma = bool(run_meta["use_limma"])
    ckpt_dir = out_dir / "checkpoints"
    pred_dir = out_dir / "predictions_PRIVATE"
    rand_genes_dir = out_dir / "random_draw_gene_lists"

    # 2. Validate baseline_all_gene_and_de_metrics.csv + every task checkpoint + canonical IDs/labels + metric recomputation
    de_csv = out_dir / "baseline_all_gene_and_de_metrics.csv"
    require(de_csv.is_file(), f"Missing {de_csv.name}", CertificationError)
    df_de = pd.read_csv(de_csv)
    req_de_cols = {
        "cohort", "baseline", "K", "n_test", "panel_size", "auroc", "auroc_ci_low", "auroc_ci_high",
        "auprc", "bacc", "bacc_ci_low", "bacc_ci_high", "f1", "brier", "bootstrap_reps",
        "stage22_converged", "stage22_n_iter",
    }
    require(req_de_cols.issubset(df_de.columns), f"{de_csv.name} missing required columns", CertificationError)

    expected_de_methods = ["welch", "limma_voom"] if use_limma else ["welch"]
    expected_pairs = set()
    for c in expected_cohorts:
        expected_pairs.add((c, "all_9983", EXPECTED_SHARED_GENES))
        expected_pairs.add((c, "train_eligible", EXPECTED_LOCO_ELIGIBLE[c]))
        for dm in expected_de_methods:
            # Verify DE ranking CSV exists with exact eligible gene count
            de_rank_p = out_dir / f"de_ranking_{c}_{dm}.csv"
            require(de_rank_p.is_file(), f"Missing DE ranking CSV: {de_rank_p.name}", CertificationError)
            rk_df = pd.read_csv(de_rank_p)
            require(len(rk_df) == EXPECTED_LOCO_ELIGIBLE[c], f"DE ranking {de_rank_p.name} row count {len(rk_df)} != {EXPECTED_LOCO_ELIGIBLE[c]}", CertificationError)
            for K in FIXED_K_GRID:
                expected_pairs.add((c, dm, K))

    observed_pairs = {(str(r["cohort"]), str(r["baseline"]), int(r["K"])) for _, r in df_de.iterrows()}
    require(observed_pairs == expected_pairs and len(df_de) == len(expected_pairs),
            f"Incomplete/duplicate baseline rows: missing={expected_pairs - observed_pairs}", CertificationError)
    require((df_de["stage22_converged"] == True).all(), "Unconverged Stage 22 model(s)", CertificationError)

    verified_ckpt_count = 0
    for _, r in df_de.iterrows():
        c, b, K = str(r["cohort"]), str(r["baseline"]), int(r["K"])
        task_id = f"{c}__{b}" if b in ("all_9983", "train_eligible") else f"{c}__{b}_K{K}"
        exp_comp = {"pred"} if b in ("all_9983", "train_eligible") else {"pred", "panel"}
        ckpt_p = ckpt_dir / f"{task_id}.json"
        require(ckpt_p.is_file(), f"Missing required baseline checkpoint: {ckpt_p.name}", CertificationError)
        try:
            ck_payload = load_validated_checkpoint(ckpt_p, actual_script_sha, EXPECTED_ZIP_SHA, expected_fp, task_id, exp_comp, out_dir)
        except CheckpointIntegrityError as exc:
            raise CertificationError(f"Checkpoint validation failed for {task_id}: {exc}") from exc
        require(ck_payload is not None and abs(float(ck_payload["auroc"]) - float(r["auroc"])) <= 1e-12,
                f"Checkpoint AUROC mismatch vs summary CSV for {task_id}", CertificationError)
        verified_ckpt_count += 1

        # Verify prediction CSV canonical patient IDs, true labels, and exact metric recomputation
        pred_p = pred_dir / f"pred_{task_id}_PRIVATE.csv"
        pdf = pd.read_csv(pred_p)
        require(len(pdf) == EXPECTED_COUNTS[c] and pdf["ID"].is_unique, f"Sample count/uniqueness failed in {pred_p.name}", CertificationError)
        require(compute_prediction_label_sha(pdf) == CANONICAL_COHORT_LABEL_SHA256[c],
                f"Prediction CSV {pred_p.name} patient IDs or y_true labels do not match canonical dataset", CertificationError)
        probs = pdf["prob_CD"].to_numpy(dtype=np.float64)
        preds = pdf["pred_CD"].to_numpy(dtype=int)
        y_true = pdf["y_true"].to_numpy(dtype=int)
        require(np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all(), f"Invalid prob_CD in {pred_p.name}", CertificationError)
        require(np.array_equal(preds, (probs >= 0.5).astype(int)), f"pred_CD inconsistent with prob_CD>=0.5 in {pred_p.name}", CertificationError)
        recomp_auc = compute_auroc_exact(y_true, probs)
        recomp_bacc = compute_bacc_exact(y_true, preds)
        require(abs(recomp_auc - float(r["auroc"])) <= 1e-10, f"Recomputed AUROC {recomp_auc} != reported {r['auroc']} in {task_id}", CertificationError)
        require(abs(recomp_bacc - float(r["bacc"])) <= 1e-10, f"Recomputed BAcc {recomp_bacc} != reported {r['bacc']} in {task_id}", CertificationError)
        if not is_smoke:
            require(int(r["bootstrap_reps"]) >= MIN_PROD_BOOT_REPS, f"Insufficient bootstrap_reps in {task_id}", CertificationError)
            require(float(r["auroc_ci_low"]) <= float(r["auroc"]) <= float(r["auroc_ci_high"]), f"Invalid AUROC CI in {task_id}", CertificationError)
            require(float(r["bacc_ci_low"]) <= float(r["bacc"]) <= float(r["bacc_ci_high"]), f"Invalid BAcc CI in {task_id}", CertificationError)

    # 3. Validate random-panel draws, checkpoints, gene receipts, and recomputed summary
    rand_sum_csv = out_dir / "baseline_random_panel_summary.csv"
    rand_draws_csv = out_dir / "baseline_random_panel_draws.csv"
    require(rand_sum_csv.is_file() and rand_draws_csv.is_file(), "Missing random panel summary or draws CSV", CertificationError)
    df_rs = pd.read_csv(rand_sum_csv)
    df_rd = pd.read_csv(rand_draws_csv)

    req_rd_cols = {"cohort", "K", "draw_idx", "seed", "panel_genes_sha256", "auroc", "bacc", "stage22_converged"}
    require(req_rd_cols.issubset(df_rd.columns), f"{rand_draws_csv.name} missing required columns", CertificationError)
    require((df_rd["stage22_converged"] == True).all(), "Unconverged random draw(s) detected", CertificationError)

    expected_total_draws = 0
    for c in expected_cohorts:
        fold_idx = COHORTS.index(c)
        unique_ks = sorted({v["sizes"][c] for v in AUDITED_COMPARATORS.values()})
        for K in unique_ks:
            expected_total_draws += n_random
            task_id = f"{c}__random_K{K}_B{n_random}"
            ckpt_p = ckpt_dir / f"{task_id}.json"
            require(ckpt_p.is_file(), f"Missing required random-panel checkpoint: {ckpt_p.name}", CertificationError)
            try:
                load_validated_checkpoint(ckpt_p, actual_script_sha, EXPECTED_ZIP_SHA, expected_fp, task_id, {"draws", "genes"}, out_dir)
            except CheckpointIntegrityError as exc:
                raise CertificationError(f"Random checkpoint validation failed for {task_id}: {exc}") from exc
            verified_ckpt_count += 1

            sub = df_rd[(df_rd["cohort"] == c) & (df_rd["K"] == K)].sort_values("draw_idx")
            require(len(sub) == n_random, f"Random draw count for {c}/K={K} is {len(sub)} != {n_random}", CertificationError)
            require(sub["draw_idx"].tolist() == list(range(n_random)), f"Missing/duplicate draw_idx for {c}/K={K}", CertificationError)
            exp_seeds = [RANDOM_BASE_SEED + fold_idx * 100000 + K * 1000 + b for b in range(n_random)]
            require(sub["seed"].tolist() == exp_seeds, f"Seed ledger mismatch for {c}/K={K}", CertificationError)

            genes_p = rand_genes_dir / f"random_genes_{task_id}.json"
            receipts = json.loads(genes_p.read_text(encoding="utf-8"))
            require(len(receipts) == n_random, f"Random gene receipts count mismatch in {genes_p.name}", CertificationError)

    require(len(df_rd) == expected_total_draws, f"Total random draw rows {len(df_rd)} != expected {expected_total_draws}", CertificationError)

    # Verify summary table recomputation and comparator provenance
    expected_comp_rows = {(m, c) for m in AUDITED_COMPARATORS for c in expected_cohorts}
    observed_comp_rows = {(str(r["comparator_method"]), str(r["cohort"])) for _, r in df_rs.iterrows()}
    require(observed_comp_rows == expected_comp_rows and len(df_rs) == len(expected_comp_rows),
            "Incomplete/duplicate comparator rows in baseline_random_panel_summary.csv", CertificationError)
    for _, r in df_rs.iterrows():
        m, c = str(r["comparator_method"]), str(r["cohort"])
        cinfo = AUDITED_COMPARATORS[m]
        K = cinfo["sizes"][c]
        require(int(r["K"]) == K and int(r["n_draws"]) == n_random, f"Summary K/n_draws mismatch for {m}/{c}", CertificationError)
        require(str(r["comparator_source"]) == cinfo["source"], f"Tampered comparator_source for {m}/{c}", CertificationError)
        require(str(r["interval_scope"]) == "within_cohort_empirical_null_B_draws", f"Invalid interval_scope for {m}/{c}", CertificationError)
        sub = df_rd[(df_rd["cohort"] == c) & (df_rd["K"] == K)]
        aucs = sub["auroc"].to_numpy(dtype=np.float64)
        exp_p = float((1 + np.sum(aucs >= float(cinfo["obs_auroc"][c]))) / (len(aucs) + 1))
        require(abs(float(r["null_auroc_mean"]) - float(aucs.mean())) <= 1e-10, f"Summary mean AUROC mismatch for {m}/{c}", CertificationError)
        require(abs(float(r["emp_p_auroc"]) - exp_p) <= 1e-10, f"Summary emp_p_auroc mismatch for {m}/{c}", CertificationError)

    # 4. Validate exported eligible-gene universes (exact counts + SHA match)
    univ_manifest_p = out_dir / "eligible_universes" / "eligible_universes_manifest.json"
    require(univ_manifest_p.is_file(), "Missing eligible_universes_manifest.json", CertificationError)
    um = json.loads(univ_manifest_p.read_text(encoding="utf-8"))
    for scope_key, exp_map in [("loco", EXPECTED_LOCO_ELIGIBLE), ("indep", EXPECTED_INDEP_ELIGIBLE)]:
        for c in COHORTS:
            p_u = out_dir / "eligible_universes" / f"{scope_key}_{c}_eligible_genes.csv"
            require(p_u.is_file(), f"Missing exported universe {p_u.name}", CertificationError)
            require(sha_file(p_u) == um[scope_key][c]["sha256"], f"Universe SHA mismatch for {p_u.name}", CertificationError)
            udf = pd.read_csv(p_u)
            require("gene" in udf.columns and len(udf) == exp_map[c] and udf["gene"].is_unique,
                    f"Universe {p_u.name} gene count {len(udf)} != expected {exp_map[c]}", CertificationError)

    status = "SMOKE_ONLY_NOT_PRODUCTION" if is_smoke else "CERTIFIED_COMPLETE"
    final_manifest = {
        "status": status,
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "script_name": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "execution_script_sha256": run_meta["execution_script_sha256"],
        "finalizer_script_sha256": actual_script_sha,
        "dataset_sha256": EXPECTED_ZIP_SHA,
        "config_fingerprint": expected_fp,
        "run_mode": run_meta["run_mode"],
        "use_limma": use_limma,
        "n_random": n_random,
        "n_boot": n_boot,
        "stage22_C": STAGE22_C,
        "stage22_seed": STAGE22_SEED,
        "stage22_tol": STAGE22_TOL_DESC,
        "cohorts_verified": list(expected_cohorts),
        "n_verified_checkpoints": int(verified_ckpt_count),
        "n_baseline_rows": int(len(df_de)),
        "n_random_comparator_rows": int(len(df_rs)),
        "n_random_draws_total": int(len(df_rd)),
        "all_stage22_converged": True,
    }
    atomic_write_json(out_dir / "manifest.json", final_manifest)

    lines = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt" and not p.name.endswith(".tmp"):
            rel = p.relative_to(out_dir).as_posix()
            lines.append(f"{sha_file(p)}  {rel}")
    atomic_write_text(out_dir / "SHA256SUMS.txt", "\n".join(lines) + "\n")
    return final_manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CMPS 355 Phase 2 Baselines v3 (C1)")
    ap.add_argument("--data", type=Path, default=None, help="Path to immutable ibd_dataset_sample_aligned.zip")
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-limma", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()
    if args.finalize_only:
        print(json.dumps(finalize_outputs(args.out, allow_smoke_status=False), indent=2))
    else:
        require(args.data is not None, "--data (path to ibd_dataset_sample_aligned.zip) is required")
        print(json.dumps(run_all_baselines(
            args.data, args.out,
            n_random=args.n_random, n_boot=args.n_boot,
            use_limma=not args.no_limma, smoke_only=args.smoke_test,
        ), indent=2))

#!/usr/bin/env python3
"""Adversarial & Regression Negative Test Suite for Phase 2 v3 (C1 + C2/C3).

Includes ALL 13 v2 regression checks PLUS explicit reproductions of ChatGPT's four v2
adversarial defects — verifying every single one is now REJECTED fail-closed:
  - ADV-1 (V2-C1-P0a): Fabricated C1 production run (48 fake baselines, FAKE_* sample IDs,
    1 random draw despite B=100, 8 fake 1-gene universes, 0 checkpoints) -> REJECTED.
  - ADV-2 (V2-C1-P0b): `load_validated_checkpoint` with `companion_files={}`, tampered
    payload, or wrong `task_id` -> REJECTED.
  - ADV-3 (V2-C2-P0a): Fabricated C2/C3 run (42 dummy pairwise, 7 dummy 4-way,
    `input_provenance.json={'FAKE': True}`, 1 valid fixed-K row) -> REJECTED.
  - ADV-4 (V2-C2-P0b): Manifest with unpinned panel `expected_sha256` -> REJECTED;
    C2-only finalizer returns `C2_OVERLAP_ONLY_COMPLETE`, never `CERTIFIED_COMPLETE`.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase2_baselines_v3 as c1
import phase2_overlap_and_size_v3 as c2


def test_c1_regression_and_adversarial():
    # Stage 22 parity + MIN_FRACTION + 5 comparators
    assert c1.STAGE22_SEED == 123 and c1.STAGE22_C == 1.0 and c1.STAGE22_TOL_DESC == "sklearn_default_1e-4"
    assert c1.MIN_FRACTION == 0.10 and len(c1.AUDITED_COMPARATORS) == 5
    print("[PASS] C1 Stage 22 parity (seed=123, default tol=1e-4), MIN_FRACTION=0.10, 5 comparators verified")

    # Directory rejected
    with tempfile.TemporaryDirectory() as td:
        try:
            c1.load_canonical_dataset(Path(td))
            raise AssertionError("Directory should be rejected")
        except ValueError:
            pass
    print("[PASS] C1-01 Directory input rejected")

    # Leakage, rank, Welch tie-break
    c1.assert_disjoint_train_test(["S1"], ["S2"], {"d2", "d3", "d4"}, "d1")
    try:
        c1.assert_disjoint_train_test(["S1"], ["S1"], {"d2", "d3", "d4"}, "d1")
        raise AssertionError("Leakage not rejected")
    except c1.LeakageViolationError:
        pass
    X_ok = np.array([[1,0,0,0],[0,1,0,0],[1,0,1,0],[0,1,1,0],[1,0,0,1],[0,1,0,1]], dtype=float)
    assert c1.assert_design_matrix_full_rank(X_ok, ["a","b","c","d"])[0] == 4
    try:
        c1.assert_design_matrix_full_rank(np.column_stack([X_ok, X_ok[:,0]+X_ok[:,1]]), ["a","b","c","d","e"])
        raise AssertionError("Rank deficiency not rejected")
    except c1.DesignRankError:
        pass
    wdf = c1.rank_welch_training_only(np.array([[1.,1.],[2.,2.],[5.,5.],[6.,6.]]), np.array([0,0,1,1]), ["GENE_Z","GENE_A"])
    assert wdf["gene"].tolist() == ["GENE_A", "GENE_Z"]
    print("[PASS] C1 Leakage, DesignRank, and Welch alphabetical tie-break verified")

    # ADV-1 (V2-C1-P0a): Exact reproduction of ChatGPT's fabricated C1 production run -> MUST REJECT
    with tempfile.TemporaryDirectory(prefix="adv_c1_fake_") as tmp:
        d = Path(tmp)
        c1.atomic_write_json(d / "run_metadata.json", {
            "execution_script_sha256": "FAKEHASH", "dataset_sha256": "FAKEHASH", "config_fingerprint": "FAKEHASH",
            "run_mode": "full", "use_limma": True, "n_random": 100, "n_boot": 2000,
            "cohorts_planned": list(c1.COHORTS),
        })
        rows = []
        pdir = d / "predictions_PRIVATE"
        pdir.mkdir()
        for c in c1.COHORTS:
            data = [("all_9983", 9983), ("train_eligible", c1.EXPECTED_LOCO_ELIGIBLE[c])] + [(m, k) for m in ("welch", "limma_voom") for k in c1.FIXED_K_GRID]
            for m, k in data:
                rows.append(dict(cohort=c, baseline=m, K=k, n_test=c1.EXPECTED_COUNTS[c], panel_size=k, auroc=0.5, auroc_ci_low=0.4, auroc_ci_high=0.6, auprc=0.5, bacc=0.5, bacc_ci_low=0.4, bacc_ci_high=0.6, f1=0.5, brier=0.25, bootstrap_reps=2000, stage22_converged=True, stage22_n_iter=1))
                task = f"{c}__{m}" if m in ("all_9983", "train_eligible") else f"{c}__{m}_K{k}"
                pd.DataFrame({"cohort": [c]*c1.EXPECTED_COUNTS[c], "ID": [f"FAKE_{c}_{i}" for i in range(c1.EXPECTED_COUNTS[c])], "y_true": [i%2 for i in range(c1.EXPECTED_COUNTS[c])], "prob_CD": [0.5]*c1.EXPECTED_COUNTS[c], "pred_CD": [1]*c1.EXPECTED_COUNTS[c]}).to_csv(pdir / f"pred_{task}_PRIVATE.csv", index=False)
        pd.DataFrame(rows).to_csv(d / "baseline_all_gene_and_de_metrics.csv", index=False)
        r = []
        for m in c1.AUDITED_COMPARATORS:
            for c in c1.COHORTS:
                r.append({**{"comparator_method": m, "comparator_source": "fake", "cohort": c, "K": c1.AUDITED_COMPARATORS[m]["sizes"][c], "n_draws": 100, "interval_scope": "fake", "observed_auroc": 0.5, "observed_bacc": 0.5, "emp_p_auroc": 0.5, "emp_p_bacc": 0.5}, **{f"null_{metric}_{stat}": 0.5 for metric in ["auroc", "bacc"] for stat in ["mean", "sd", "median", "iqr", "p05", "p95"]}})
        pd.DataFrame(r).to_csv(d / "baseline_random_panel_summary.csv", index=False)
        pd.DataFrame([{"stage22_converged": True}]).to_csv(d / "baseline_random_panel_draws.csv", index=False)
        u = d / "eligible_universes"; u.mkdir(); um = {"loco": {}, "indep": {}}
        for scope in ["loco", "indep"]:
            for c in c1.COHORTS:
                p = u / f"{scope}_{c}_eligible_genes.csv"; pd.DataFrame({"gene": ["FAKEGENE"]}).to_csv(p, index=False)
                um[scope][c] = {"sha256": c1.sha_file(p)}
        c1.atomic_write_json(u / "eligible_universes_manifest.json", um)
        try:
            c1.finalize_outputs(d)
            raise AssertionError("ADV-1 FAILED: fabricated C1 run was falsely certified!")
        except c1.CertificationError:
            pass
    print("[PASS] ADV-1 (V2-C1-P0a): Fabricated C1 production run (FAKE_* IDs, 1 draw, 0 checkpoints) REJECTED")

    # ADV-2 (V2-C1-P0b): load_validated_checkpoint rejects companion_files={}, tampered payload, wrong task_id
    with tempfile.TemporaryDirectory(prefix="adv_c1_ckpt_") as tmp:
        d = Path(tmp)
        p = d / "ckpt_empty.json"
        c1.atomic_write_json(p, {
            "schema_version": "c1_checkpoint_v3", "status": "COMPLETED", "task_id": "d3__all_9983",
            "execution_script_sha256": "SCRIPT", "dataset_sha256": "DATA", "config_fingerprint": "CFG",
            "companion_files": {}, "payload_sha256": c1.canonical_json_sha({"auroc": 0.999}), "payload": {"auroc": 0.999},
        })
        try:
            c1.load_validated_checkpoint(p, "SCRIPT", "DATA", "CFG", "d3__all_9983", {"pred"}, d)
            raise AssertionError("ADV-2 FAILED: empty companion_files={} was accepted!")
        except c1.CheckpointIntegrityError:
            pass

        # Valid checkpoint round-trip + tamper checks
        comp = d / "pred_d3.csv"
        comp.write_text("cohort,ID,y_true,prob_CD,pred_CD\nd3,S1,1,0.8,1\n")
        ckpt = d / "d3__all_9983.json"
        c1.save_validated_checkpoint(ckpt, "d3__all_9983", {"auroc": 0.75}, "S", "D", "C", {"pred": comp}, d)
        assert c1.load_validated_checkpoint(ckpt, "S", "D", "C", "d3__all_9983", {"pred"}, d)["auroc"] == 0.75
        # Wrong task_id -> rejected
        try:
            c1.load_validated_checkpoint(ckpt, "S", "D", "C", "d1__all_9983", {"pred"}, d)
            raise AssertionError("Wrong task_id accepted")
        except c1.CheckpointIntegrityError:
            pass
        # Tampered payload -> rejected
        rec = json.loads(ckpt.read_text()); rec["payload"]["auroc"] = 0.999; ckpt.write_text(json.dumps(rec))
        try:
            c1.load_validated_checkpoint(ckpt, "S", "D", "C", "d3__all_9983", {"pred"}, d)
            raise AssertionError("Tampered payload accepted")
        except c1.CheckpointIntegrityError:
            pass
    print("[PASS] ADV-2 (V2-C1-P0b): Empty companion_files={}, wrong task_id, and tampered payload REJECTED")


def test_c2_c3_regression_and_adversarial():
    # Pairwise & 4-way exact vs MC + 95% null intervals
    ua = {f"G{i}" for i in range(1000)}
    ub = {f"G{i}" for i in range(750, 1750)}
    r2 = c2.monte_carlo_pairwise_null(ua, ub, 200, 100, obs_inter=10, n_mc=10000, seed=42)
    assert abs(r2["exact_expected_intersection"] - 5.0) < 1e-12
    assert r2["mc_jaccard_ci_low"] <= r2["mc_mean_jaccard"] <= r2["mc_jaccard_ci_high"]
    assert r2["mc_overlap_ci_low"] <= r2["mc_mean_overlap_coeff"] <= r2["mc_overlap_ci_high"]

    u1 = {f"G{i}" for i in range(1000)}; u2 = {f"G{i}" for i in range(200, 1200)}
    u3 = {f"G{i}" for i in range(300, 1300)}; u4 = {f"G{i}" for i in range(400, 1400)}
    fw = c2.monte_carlo_fourway_null([u1, u2, u3, u4], [500, 500, 500, 500], obs_fourway=40, n_mc=10000, seed=42)
    assert abs(fw["exact_expected_fourway"] - 37.5) < 1e-12
    assert fw["mc_fourway_ci_low"] <= fw["mc_mean_fourway"] <= fw["mc_fourway_ci_high"]
    print(f"[PASS] C2 Pairwise + 4-way nulls & 95% MC intervals verified (4-way CI=[{fw['mc_fourway_ci_low']:.1f}, {fw['mc_fourway_ci_high']:.1f}])")

    # ADV-3 (V2-C2-P0a): Exact reproduction of ChatGPT's fabricated C2/C3 run -> MUST REJECT
    staged_cfg = Path(__file__).resolve().parent / "overlap_manifest_config_v3.json"
    with tempfile.TemporaryDirectory(prefix="adv_c2_fake_") as tmp:
        d = Path(tmp)
        n_m = len(json.loads(staged_cfg.read_text())["methods"])
        pd.DataFrame([{"exact_expected_intersection": 5.0}] * (n_m * 6)).to_csv(d / "phase2_pairwise_overlap_null.csv", index=False)
        pd.DataFrame([{"exact_expected_fourway": 2.0}] * n_m).to_csv(d / "phase2_fourway_overlap_null.csv", index=False)
        c2.atomic_write_json(d / "input_provenance.json", {"FAKE": True})
        pd.DataFrame([{"status": "VALID_CONTINUOUS_RANK", "stage22_converged": True, "auroc": 0.5, "bacc": 0.5}]).to_csv(d / "phase2_fixed_k_cohort_metrics.csv", index=False)
        pd.DataFrame([{"macro_auroc": 0.5}]).to_csv(d / "phase2_fixed_k_macro_summary.csv", index=False)
        try:
            c2.finalize_c2_c3_outputs(d, staged_cfg, include_fixed_k=True)
            raise AssertionError("ADV-3 FAILED: fabricated C2/C3 run was falsely certified!")
        except c2.CertificationError:
            pass
    print("[PASS] ADV-3 (V2-C2-P0a): Fabricated C2/C3 run (input_provenance=FAKE, 1 fixed-K row) REJECTED")

    # ADV-4 (V2-C2-P0b): Unpinned panel SHA rejected; C2-only never returns CERTIFIED_COMPLETE
    with tempfile.TemporaryDirectory(prefix="adv_c2_pin_") as tmp:
        d = Path(tmp)
        p_csv = d / "p.csv"; u_csv = d / "u.csv"
        pd.DataFrame({"gene": ["G1", "G2"]}).to_csv(p_csv, index=False)
        pd.DataFrame({"gene": ["G1", "G2", "G3", "G4"]}).to_csv(u_csv, index=False)
        p_sha, u_sha = c2.sha_file(p_csv), c2.sha_file(u_csv)

        # Unpinned manifest rejected when require_pinned_hashes=True
        cfg_unpinned = {"methods": [{"name": "Corrected Nested LASSO", "role": "primary", "scope": "LOCO_size_only_null_training_dependent", "panel_sources": {c: {"path": str(p_csv)} for c in c2.COHORTS}, "universe_csvs": {c: {"path": str(u_csv)} for c in c2.COHORTS}}]}
        try:
            c2.validate_manifest_config(cfg_unpinned, require_pinned_hashes=True)
            raise AssertionError("ADV-4 FAILED: unpinned manifest was accepted!")
        except c2.ProvenanceError:
            pass

        # Historical bypasses still rejected
        for bad_entry, exc_t in [
            ({"name": "TENTACLES_V1_nmin5", "scope": "LOCO_size_only_null_training_dependent", "panel_sources": {c: {"path": str(p_csv), "expected_sha256": p_sha} for c in c2.COHORTS}, "universe_csvs": {c: {"path": str(u_csv), "expected_sha256": u_sha} for c in c2.COHORTS}}, c2.SchemaValidationError),
            ({"name": "historical_saga_lasso", "role": "primary", "scope": "LOCO_size_only_null_training_dependent", "panel_sources": {c: {"path": str(p_csv), "expected_sha256": p_sha} for c in c2.COHORTS}, "universe_csvs": {c: {"path": str(u_csv), "expected_sha256": u_sha} for c in c2.COHORTS}}, c2.HistoricalVersionError),
        ]:
            try:
                c2.validate_manifest_config({"methods": [bad_entry]}, require_pinned_hashes=True)
                raise AssertionError(f"Expected {exc_t.__name__}")
            except exc_t:
                pass

        # Valid pinned synthetic C2-only run returns C2_OVERLAP_ONLY_COMPLETE (never CERTIFIED_COMPLETE)
        cfg_pin = d / "cfg_pin.json"
        cfg_pin.write_text(json.dumps({"methods": [{
            "name": "Corrected Nested LASSO", "role": "primary", "scope": "LOCO_size_only_null_training_dependent",
            "panel_sources": {c: {"path": str(p_csv), "expected_sha256": p_sha} for c in c2.COHORTS},
            "universe_csvs": {c: {"path": str(u_csv), "expected_sha256": u_sha} for c in c2.COHORTS},
        }]}))
        c2.run_overlap_null_from_manifest(cfg_pin, d / "out_c2", n_mc=200, require_pinned_hashes=True)
        fin = c2.finalize_c2_c3_outputs(d / "out_c2", cfg_pin, include_fixed_k=False, n_mc=200, require_pinned_hashes=True)
        assert fin["status"] == "C2_OVERLAP_ONLY_COMPLETE", f"Unexpected C2-only status: {fin['status']}"

    # Validate staged production overlap_manifest_config_v3.json with require_pinned_hashes=True
    c2.validate_manifest_config(json.loads(staged_cfg.read_text(encoding="utf-8")), require_pinned_hashes=True)
    print("[PASS] ADV-4 (V2-C2-P0b): Unpinned manifest & historical bypasses REJECTED; C2-only status=C2_OVERLAP_ONLY_COMPLETE; staged v3 manifest validated with all 28 pinned panel SHAs")


if __name__ == "__main__":
    test_c1_regression_and_adversarial()
    test_c2_c3_regression_and_adversarial()
    print("\nALL PHASE 2 V3 ADVERSARIAL & REGRESSION NEGATIVE TESTS PASSED.")

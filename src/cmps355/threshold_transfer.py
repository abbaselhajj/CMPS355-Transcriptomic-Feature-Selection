"""Nested-cohort threshold selection with immutable outer-fold predictions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from cmps355.pilot import (
    PilotConfig, fit_logistic, fit_preprocessor, load_verified_splits,
    read_counts, transform_preprocessor,
)
from cmps355.split_design import read_sample_index


METHODS = ("lasso", "elastic_net")
THRESHOLDS = tuple(float(np.round(x, 3)) for x in np.arange(0.05, 0.95001, 0.025))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checksum(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def operating_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    require(y.ndim == 1 and probability.ndim == 1 and len(y) == len(probability), "Invalid input shapes")
    require(set(np.unique(y)) == {0, 1}, "Both classes required")
    require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
            "Probabilities must be finite and lie in [0, 1]")
    require(0 < threshold < 1, "Threshold must lie strictly inside (0, 1)")
    predicted = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "predicted_positive_rate": float(predicted.mean()),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def select_threshold(oof: pd.DataFrame, thresholds: tuple[float, ...] = THRESHOLDS) -> tuple[float, pd.DataFrame]:
    cols = {"ID", "validation_cohort", "true_CD", "probability_CD"}
    require(cols.issubset(oof.columns), "OOF table is missing required columns")
    require(not oof.empty and oof["ID"].is_unique, "OOF IDs missing or duplicated")
    require(oof["validation_cohort"].nunique() == 3, "Expected three distinct validation cohorts")
    require(all(group["true_CD"].nunique() == 2 for _, group in oof.groupby("validation_cohort")),
            "Each validation cohort must contain both classes")
    require(len(thresholds) > 0 and len(set(thresholds)) == len(thresholds), "Invalid threshold grid")
    require(all(np.isfinite(t) and 0 < t < 1 for t in thresholds), "Invalid threshold value")
    rows = []
    for threshold in thresholds:
        per_cohort = []
        for cohort, group in oof.groupby("validation_cohort", sort=True):
            m = operating_metrics(group["true_CD"].to_numpy(),
                                  group["probability_CD"].to_numpy(), threshold)
            per_cohort.append(m["balanced_accuracy"])
        rows.append({"threshold": threshold, "mean_cohort_balanced_accuracy": float(np.mean(per_cohort)),
                     "min_cohort_balanced_accuracy": float(np.min(per_cohort))})
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["mean_cohort_balanced_accuracy", "threshold"],
                            ascending=[False, True]).copy()
    maximum = best.iloc[0]["mean_cohort_balanced_accuracy"]
    tied = grid[np.isclose(grid["mean_cohort_balanced_accuracy"], maximum, atol=1e-12, rtol=0)].copy()
    tied["distance_to_half"] = np.abs(tied["threshold"] - 0.5)
    threshold = float(tied.sort_values(["distance_to_half", "threshold"]).iloc[0]["threshold"])
    return threshold, grid


def load_pilot_from_diagnostics(root: Path, manifest: dict) -> tuple[Path, dict, pd.DataFrame, pd.DataFrame, dict]:
    diagnostic_base = root / "runs" / "pilot_diagnostics"
    diagnostics = sorted(diagnostic_base.glob("*/report.json"), reverse=True)
    require(bool(diagnostics), "No Stage 3C diagnostic reports under runs/pilot_diagnostics")
    diagnosis = None
    for file in diagnostics:
        candidate = json.loads(file.read_text(encoding="utf-8"))
        if candidate.get("status") == "PASS" and candidate.get("stage") == "03C":
            diagnosis = candidate
            break
    require(diagnosis is not None, "No passing Stage 3C report")
    path = Path(diagnosis["source_report"])
    require(not path.is_absolute() and ".." not in path.parts and path.parts[:2] == ("runs", "pilot"),
            "Invalid referenced Stage 3B path")
    source_report = root / path
    require(source_report.is_file() and checksum(source_report) == diagnosis["source_run_sha256"],
            "Stage 3B report differs from audited source")
    pilot_report = json.loads(source_report.read_text(encoding="utf-8"))
    require(pilot_report.get("status") == "PASS" and pilot_report.get("stage") == "03B",
            "Selected Stage 3B run did not pass")
    require(pilot_report.get("dataset_sha256") == manifest["sha256"] == diagnosis["dataset_sha256"],
            "Dataset identity mismatch")
    expected = pilot_report["artifacts_sha256"]
    require(expected == diagnosis["source_artifact_sha256"], "Stage 3C source artifacts mismatch")
    for filename, sha in expected.items():
        p = source_report.parent / filename
        require(p.is_file() and checksum(p) == sha, f"Missing or altered pilot artifact: {filename}")
    predictions = pd.read_csv(source_report.parent / "predictions.csv", dtype={"ID": "string"})
    metrics = pd.read_csv(source_report.parent / "outer_metrics.csv")
    return source_report, pilot_report, predictions, metrics, diagnosis


def validate_predictions(index: pd.DataFrame, predictions: pd.DataFrame,
                         outer_metrics: pd.DataFrame, cohorts: list[str]) -> None:
    required = {"ID", "test_cohort", "method", "true_CD", "probability_CD", "predicted_CD"}
    require(required.issubset(predictions.columns), "Pilot predictions missing required columns")
    require(len(predictions) == len(index) * len(METHODS), "Pilot predictions incomplete")
    require(set(predictions["method"]) == set(METHODS), "Unexpected pilot methods")
    require(not predictions.duplicated(["ID", "method"]).any(), "Duplicate patient-method rows")
    require(not predictions[["ID", "test_cohort", "method", "true_CD", "probability_CD"]].isna().any().any(),
            "Missing predictions")
    index_values = index.set_index("ID")
    for method, frame in predictions.groupby("method"):
        require(set(frame["ID"]) == set(index["ID"]), f"{method}: incomplete patient coverage")
        expected = index_values.loc[frame["ID"]]
        require(np.array_equal(frame["test_cohort"].to_numpy(), expected["cohort"].to_numpy()),
                "Held-out cohort does not match patient metadata")
        require(np.array_equal(frame["true_CD"].to_numpy(), (expected["class"] == "CD").astype(int).to_numpy()),
                "Held-out label does not match patient metadata")
        require(np.array_equal(frame["predicted_CD"].to_numpy(),
                               (frame["probability_CD"].to_numpy() >= 0.5).astype(int)),
                "Original classifications inconsistent with threshold 0.5")
        require(frame["probability_CD"].between(0, 1).all(), "Out-of-range probabilities")
    require(len(outer_metrics) == len(cohorts) * len(METHODS), "Incomplete outer metrics")
    require(not outer_metrics.duplicated(["test_cohort", "method"]).any(), "Duplicate model metrics")
    require(set(outer_metrics["test_cohort"]) == set(cohorts), "Missing outer cohorts")
    require(set(outer_metrics["method"]) == set(METHODS), "Missing metrics methods")
    for row in outer_metrics.itertuples(index=False):
        p = predictions[(predictions["test_cohort"] == row.test_cohort) &
                        (predictions["method"] == row.method)]
        calculated = operating_metrics(p["true_CD"].to_numpy(), p["probability_CD"].to_numpy(), 0.5)
        for key in ("balanced_accuracy", "sensitivity", "specificity"):
            require(abs(calculated[key] - float(getattr(row, key))) < 1e-9,
                    f"Source metrics disagree for {row.test_cohort}/{row.method}/{key}")
        require(abs(float(row.auroc) - roc_auc_score(p["true_CD"], p["probability_CD"])) < 1e-9,
                "Source AUROC mismatch")


def run_threshold_transfer(root: Path) -> tuple[Path, pd.DataFrame]:
    root = Path(root).resolve()
    manifest = json.loads((root / "configs" / "dataset.json").read_text(encoding="utf-8"))
    archive = root / manifest["artifact"]
    index = read_sample_index(archive, manifest)
    source_report, pilot_report, predictions, outer_metrics, diagnosis = load_pilot_from_diagnostics(root, manifest)
    cohorts = list(manifest["cohorts"])
    validate_predictions(index, predictions, outer_metrics, cohorts)
    outer_assignments, inner_assignments, split_source = load_verified_splits(root, index, manifest)
    require(pilot_report["split_source"] == str(split_source.relative_to(root)),
            "Stage 3A split source differs from the original pilot")
    config_kwargs = pilot_report["config"]
    config = PilotConfig(**{**config_kwargs, "c_values": tuple(config_kwargs["c_values"])})
    require(tuple(sorted(set(config.c_values))) == tuple(sorted({float(v) for v in config.c_values})),
            "Non-unique candidate values")
    counts, genes = read_counts(archive, manifest, index)
    lookup = pd.Series(np.arange(len(index), dtype=int), index=index["ID"].astype(str))
    labels = (index["class"].to_numpy() == "CD").astype(np.int8)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = root / "runs" / "threshold_transfer" / stamp
    out.mkdir(parents=True, exist_ok=False)
    oof_rows, threshold_rows, score_rows, grid_rows = [], [], [], []
    report = {
        "stage": "03D", "status": "RUNNING",
        "research_role": "exploratory_threshold_transfer_not_final_benchmark",
        "source_pilot_report": str(source_report.relative_to(root)),
        "source_pilot_report_sha256": checksum(source_report),
        "source_stage3c_sha256": hashlib.sha256(json.dumps(diagnosis, sort_keys=True).encode()).hexdigest(),
        "dataset_sha256": manifest["sha256"], "split_source": str(split_source.relative_to(root)),
        "methods": list(METHODS), "threshold_grid": list(THRESHOLDS),
        "decision_rule": "maximize unweighted mean of three inner validation cohort balanced accuracies; ties closest to 0.5 then lower threshold",
        "train_test_policy": "threshold selection uses only training cohort OOF predictions; outer probabilities immutable; outer labels for evaluation only",
        "method_note": "Threshold transfer is not probability calibration; LOCO source shifts remain uncorrected",
        "config": config_kwargs,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "scipy": scipy.__version__},
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }

    def write_report(status: str) -> None:
        for name, rows in [("inner_oof.csv", oof_rows), ("thresholds.csv", threshold_rows),
                           ("outer_metrics.csv", score_rows), ("threshold_grid.csv", grid_rows)]:
            pd.DataFrame(rows).to_csv(out / name, index=False)
        report["status"] = status
        report["completed_cohort_method_pairs"] = len(score_rows)
        if status == "PASS":
            report["artifacts_sha256"] = {
                fn: checksum(out / fn)
                for fn in ("inner_oof.csv", "thresholds.csv", "outer_metrics.csv", "threshold_grid.csv")
            }
            report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    write_report("RUNNING")
    for heldout in cohorts:
        heldout_outer = outer_assignments[outer_assignments["outer_heldout"] == heldout]
        test_ids = set(heldout_outer.loc[heldout_outer["role"] == "test", "ID"].astype(str))
        train_ids = set(heldout_outer.loc[heldout_outer["role"] == "train", "ID"].astype(str))
        require(test_ids.isdisjoint(train_ids), "Outer train-test overlap")
        for method in METHODS:
            source_model = outer_metrics[(outer_metrics["test_cohort"] == heldout) &
                                         (outer_metrics["method"] == method)].iloc[0]
            chosen_c = float(source_model["C"])
            require(chosen_c in config.c_values, "Outer C not in original pilot's search grid")
            inner_rows = []
            for validation_cohort in cohorts:
                if validation_cohort == heldout:
                    continue
                inner = inner_assignments[(inner_assignments["outer_heldout"] == heldout) &
                                          (inner_assignments["inner_heldout"] == validation_cohort)]
                fit_ids = inner.loc[inner["role"] == "train", "ID"].astype(str).tolist()
                val_ids = inner.loc[inner["role"] == "validation", "ID"].astype(str).tolist()
                require(set(fit_ids).isdisjoint(test_ids) and set(val_ids).isdisjoint(test_ids),
                        "Outer test sample entered threshold fitting")
                require(set(val_ids).issubset(train_ids), "Inner validation outside outer training")
                fit_pos = lookup.loc[fit_ids].to_numpy(dtype=int)
                val_pos = lookup.loc[val_ids].to_numpy(dtype=int)
                prep = fit_preprocessor(counts[fit_pos], genes, config)
                fit, converged = fit_logistic(prep.train, labels[fit_pos], method, chosen_c, config)
                require(converged, f"Inner fit did not converge: {heldout}/{method}/{validation_cohort}")
                probabilities = fit.predict_proba(transform_preprocessor(prep, counts[val_pos]))[:, 1]
                for id_value, y, p in zip(val_ids, labels[val_pos], probabilities):
                    inner_rows.append({"outer_heldout": heldout, "method": method,
                                       "validation_cohort": validation_cohort, "ID": id_value,
                                       "true_CD": int(y), "probability_CD": float(p)})
            oof = pd.DataFrame(inner_rows)
            require(len(oof) == len(train_ids) and set(oof["ID"]) == train_ids and oof["ID"].is_unique,
                    "Inner OOF predictions incomplete or overlapping")
            require(set(oof["ID"]).isdisjoint(test_ids), "Outer test leaked into threshold selection")
            threshold, grid = select_threshold(oof)
            for row in grid.to_dict("records"):
                grid_rows.append({"test_cohort": heldout, "method": method, **row})
            oof_rows.extend(inner_rows)
            baseline_oof = grid.loc[np.isclose(grid["threshold"], 0.5), "mean_cohort_balanced_accuracy"]
            require(len(baseline_oof) == 1, "Threshold grid missing 0.5")
            threshold_rows.append({"test_cohort": heldout, "method": method,
                                   "C": chosen_c, "threshold": threshold,
                                   "validation_macro_BA_at_0_5": float(baseline_oof.iloc[0]),
                                   "validation_macro_BA_at_selected": float(grid.loc[
                                       np.isclose(grid["threshold"], threshold),
                                       "mean_cohort_balanced_accuracy"].iloc[0]),
                                   "n_inner_oof": len(oof)})
            outer = predictions[(predictions["test_cohort"] == heldout) &
                                (predictions["method"] == method)]
            require(set(outer["ID"]).issubset(test_ids) and len(outer) == len(test_ids),
                    "Outer predictions do not match held-out IDs")
            y = outer["true_CD"].to_numpy()
            p = outer["probability_CD"].to_numpy()
            metrics_before = operating_metrics(y, p, 0.5)
            metrics_after = operating_metrics(y, p, threshold)
            score_rows.append({"test_cohort": heldout, "method": method,
                               "n_test": len(outer), "C": chosen_c, "threshold": threshold,
                               "AUROC_unchanged": float(roc_auc_score(y, p)),
                               **{f"baseline_{k}": v for k, v in metrics_before.items()},
                               **{f"selected_{k}": v for k, v in metrics_after.items()},
                               "delta_balanced_accuracy": metrics_after["balanced_accuracy"] - metrics_before["balanced_accuracy"]})
            write_report("RUNNING")
            print(f"{heldout} / {method}: threshold={threshold:.3f}; "
                  f"BA {metrics_before['balanced_accuracy']:.3f} -> {metrics_after['balanced_accuracy']:.3f}", flush=True)
    require(len(score_rows) == len(cohorts) * len(METHODS), "Incomplete threshold evaluation")
    write_report("PASS")
    return out, pd.DataFrame(score_rows)

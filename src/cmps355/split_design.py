"""Reproducible, label-independent cohort split construction and auditing."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


class SplitDesignError(ValueError):
    pass


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise SplitDesignError(detail)


def read_sample_index(archive: Path, manifest: dict) -> pd.DataFrame:
    """Read sample metadata from the accepted immutable archive, without expression values."""
    archive = Path(archive)
    _require(archive.is_file(), f"Missing archive: {archive}")
    with archive.open("rb") as handle:
        observed_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    _require(observed_sha == manifest["sha256"], "Dataset SHA-256 does not match the accepted manifest")

    cohorts = list(manifest["cohorts"])
    _require(len(cohorts) == len(set(cohorts)) and len(cohorts) >= 3, "Need at least three unique cohorts")
    frames = []
    with ZipFile(archive) as zf:
        _require(zf.testzip() is None, "Archive CRC verification failed")
        for cohort in cohorts:
            with zf.open(f"desc_{cohort}.csv") as handle:
                frame = pd.read_csv(handle, dtype={"ID": "string", "Study": "string", "class": "string"})
            _require({"ID", "Study", "class"}.issubset(frame.columns), f"{cohort}: missing metadata fields")
            frame = frame[["ID", "Study", "class"]].copy()
            frame.insert(0, "cohort", cohort)
            _require(len(frame) == manifest["expected_sample_counts"][cohort], f"{cohort}: unexpected sample count")
            _require(frame.notna().all().all(), f"{cohort}: null metadata")
            _require(frame["ID"].str.len().gt(0).all(), f"{cohort}: blank sample ID")
            _require(set(frame["class"]) == set(manifest["label_values"]), f"{cohort}: expected two labels")
            frames.append(frame)
    index = pd.concat(frames, ignore_index=True)
    _require(index["ID"].is_unique, "Sample ID appears more than once, including across cohorts")
    _require(len(index) == manifest["expected_total_samples"], "Total sample count does not match manifest")
    _require(index["Study"].nunique() == 6, "Unexpected number of source BioProjects")
    return index


def build_assignments(index: pd.DataFrame, cohorts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct four outer LOCO folds and three inner LOCO folds for each outer fold."""
    _require(set(index["cohort"]) == set(cohorts), "Index cohorts differ from expected cohorts")
    _require(index["ID"].is_unique, "Duplicate IDs are not allowed")
    _require(len(cohorts) >= 3, "Insufficient cohorts for nested group holdouts")

    outer_parts = []
    inner_parts = []
    for heldout in cohorts:
        outer = index.copy()
        outer.insert(0, "outer_heldout", heldout)
        outer.insert(1, "role", "train")
        outer.loc[outer["cohort"] == heldout, "role"] = "test"
        outer_parts.append(outer)

        eligible = index.loc[index["cohort"] != heldout].copy()
        for validation in cohorts:
            if validation == heldout:
                continue
            inner = eligible.copy()
            inner.insert(0, "outer_heldout", heldout)
            inner.insert(1, "inner_heldout", validation)
            inner.insert(2, "role", "train")
            inner.loc[inner["cohort"] == validation, "role"] = "validation"
            inner_parts.append(inner)

    return pd.concat(outer_parts, ignore_index=True), pd.concat(inner_parts, ignore_index=True)


def audit_assignments(index: pd.DataFrame, outer: pd.DataFrame, inner: pd.DataFrame, cohorts: list[str]) -> dict:
    """Validate every partition, coverage property, and within-fold class composition."""
    expected_ids = set(index["ID"])
    study_counts = index.groupby(["cohort", "Study", "class"]).size().to_dict()
    fold_summary = []
    for heldout in cohorts:
        fold = outer.loc[outer["outer_heldout"] == heldout]
        _require(len(fold) == len(index), f"{heldout}: outer fold size mismatch")
        _require(fold["ID"].is_unique and set(fold["ID"]) == expected_ids, f"{heldout}: outer coverage failed")
        train_ids = set(fold.loc[fold["role"] == "train", "ID"])
        test_ids = set(fold.loc[fold["role"] == "test", "ID"])
        _require(train_ids.isdisjoint(test_ids), f"{heldout}: outer leakage")
        _require(train_ids | test_ids == expected_ids, f"{heldout}: missing sample")
        _require(set(fold.loc[fold["role"] == "test", "cohort"]) == {heldout}, f"{heldout}: wrong test cohort")
        _require(set(fold.loc[fold["role"] == "train", "cohort"]) == set(cohorts) - {heldout}, f"{heldout}: wrong training cohorts")
        for role in ("train", "test"):
            _require(set(fold.loc[fold["role"] == role, "class"]) == {"CD", "Not IBD"}, f"{heldout}: {role} missing class")
        inner_info = []
        for validation in cohorts:
            if validation == heldout:
                continue
            fold_inner = inner.loc[(inner["outer_heldout"] == heldout) & (inner["inner_heldout"] == validation)]
            _require(len(fold_inner) == len(train_ids), f"{heldout}/{validation}: incorrect inner fold size")
            _require(fold_inner["ID"].is_unique and set(fold_inner["ID"]) == train_ids,
                     f"{heldout}/{validation}: inner fold coverage or outer-test leakage")
            fitting = set(fold_inner.loc[fold_inner["role"] == "train", "ID"])
            validating = set(fold_inner.loc[fold_inner["role"] == "validation", "ID"])
            _require(fitting.isdisjoint(validating) and fitting | validating == train_ids,
                     f"{heldout}/{validation}: inner overlap or missing samples")
            _require(set(fold_inner.loc[fold_inner["role"] == "validation", "cohort"]) == {validation},
                     f"{heldout}/{validation}: inner validation cohort incorrect")
            _require(set(fold_inner.loc[fold_inner["role"] == "train", "cohort"]) == set(cohorts) - {heldout, validation},
                     f"{heldout}/{validation}: inner training cohorts incorrect")
            for role in ("train", "validation"):
                _require(set(fold_inner.loc[fold_inner["role"] == role, "class"]) == {"CD", "Not IBD"},
                         f"{heldout}/{validation}: {role} missing class")
            inner_info.append({"validation_cohort": validation, "fitting_samples": len(fitting),
                               "validation_samples": len(validating)})
        fold_summary.append({
            "test_cohort": heldout,
            "train_samples": len(train_ids),
            "test_samples": len(test_ids),
            "train_class_counts": fold.loc[fold["role"] == "train", "class"].value_counts().to_dict(),
            "test_class_counts": fold.loc[fold["role"] == "test", "class"].value_counts().to_dict(),
            "inner_folds": inner_info,
        })

    _require(set(outer["outer_heldout"]) == set(cohorts), "Unexpected outer fold")
    _require(len(outer) == len(cohorts) * len(index), "Unexpected number of outer assignment rows")
    _require(len(inner) == sum((len(cohorts) - 1) * (len(index) - sum(index["cohort"] == c)) for c in cohorts),
             "Unexpected number of inner assignment rows")
    outer_heldout_count = outer[outer["role"] == "test"].groupby("ID").size()
    _require(len(outer_heldout_count) == len(index) and outer_heldout_count.eq(1).all(),
             "Every sample must appear exactly once as outer test")

    study_table = [{"cohort": cohort, "bioproject": study, "class": cls, "count": int(size)}
                   for (cohort, study, cls), size in sorted(study_counts.items())]
    return {
        "status": "PASS",
        "total_samples": len(index),
        "outer_fold_count": len(cohorts),
        "inner_fold_count": len(cohorts) * (len(cohorts) - 1),
        "source_bioproject_count": int(index["Study"].nunique()),
        "folds": fold_summary,
        "source_class_composition": study_table,
        "limitations": [
            "The analytical cohort is the split unit; BioProjects within a cohort are not independently held out.",
            "Overlapping outer training sets are not independent gene-selection replicates.",
            "Cross-cohort fold construction does not remove batch or source effects.",
            "No normalization, filtering, scaling, feature selection, tuning or model fitting has been performed.",
            "Patient sample identifiers appear in the generated CSVs; do not publish these outputs by default.",
        ],
    }


def save_assignments(root: Path, archive_sha: str, outer: pd.DataFrame, inner: pd.DataFrame, report: dict) -> Path:
    """Save traceable split artifacts to an ignored, timestamped run directory."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = Path(root) / "runs" / "split_design" / stamp
    target.mkdir(parents=True, exist_ok=False)
    outer_path = target / "outer.csv"
    inner_path = target / "inner.csv"
    outer.to_csv(outer_path, index=False)
    inner.to_csv(inner_path, index=False)
    report = dict(report)
    report["data_sha256"] = archive_sha
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    report["artifacts_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in [("outer.csv", outer_path), ("inner.csv", inner_path)]
    }
    (target / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return target

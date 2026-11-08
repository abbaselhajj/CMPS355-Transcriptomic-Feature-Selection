"""Independent, read-only scientific checks for the accepted aligned IBD count tables.

Not an RNA-seq normalisation pipeline. No fitted parameter, gene filter, or
feature selector is created here. All checks precede cross-cohort modelling.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import numpy as np
import pandas as pd


class DatasetValidationError(ValueError):
    """A scientific data-integrity invariant failed."""


def assert_valid(condition: bool, reason: str) -> None:
    if not condition:
        raise DatasetValidationError(reason)


def _csv_header(zf: ZipFile, member: str) -> list[str]:
    with zf.open(member) as stream:
        text = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
        try:
            header = next(csv.reader(text))
        except StopIteration as exc:
            raise DatasetValidationError(f"Empty CSV: {member}") from exc
    return header


def validate_archive(archive_path: Path, manifest: dict) -> dict:
    """Read a user-accepted canonical ZIP, independently verify its matrices and metadata.

    Each ZIP member remains unchanged. Our checks use exact IDs, not positional
    assumptions, and do not silently repair missing, duplicated or mismatched data.
    """
    archive_path = Path(archive_path).resolve()
    assert_valid(archive_path.is_file(), f"Canonical input missing: {archive_path}")
    expected_hash = manifest.get("sha256")
    with archive_path.open("rb") as input_stream:
        actual_hash = hashlib.file_digest(input_stream, "sha256").hexdigest()
    assert_valid(expected_hash is None or actual_hash == expected_hash,
                 f"Dataset SHA-256 mismatch: found {actual_hash}, expected {expected_hash}. Do not proceed.")

    cohorts = manifest.get("cohorts", ["d1", "d2", "d3", "d4"])
    assert_valid(len(cohorts) == len(set(cohorts)) and len(cohorts) > 0,
                 "Cohort names must be unique and nonempty")
    profiles = []
    all_ids: set[str] = set()
    gene_sets: list[set[str]] = []
    study_totals: Counter = Counter()
    class_totals: Counter = Counter()

    try:
        with ZipFile(archive_path, "r") as zf:
            member_names = set(zf.namelist())
            expected_files = {name for cohort in cohorts for name in
                              (f"count_{cohort}_aligned.csv", f"desc_{cohort}.csv")}
            assert_valid(expected_files.issubset(member_names),
                         f"Dataset ZIP lacks: {sorted(expected_files - member_names)}")
            corrupt_member = zf.testzip()
            assert_valid(corrupt_member is None, f"ZIP CRC invalid for {corrupt_member}")

            for cohort in cohorts:
                count_name = f"count_{cohort}_aligned.csv"
                meta_name = f"desc_{cohort}.csv"
                columns = _csv_header(zf, count_name)
                assert_valid(columns[0] == "ID", f"{count_name}: first column must be ID")
                gene_names = columns[1:]
                assert_valid(gene_names and all(g and g == g.strip() for g in gene_names),
                             f"{count_name}: invalid gene symbol")
                assert_valid(len(gene_names) == len(set(gene_names)),
                             f"{count_name}: duplicate gene symbols in raw header")

                counts = pd.read_csv(zf.open(count_name), dtype={"ID": "string"}, low_memory=False)
                metadata = pd.read_csv(zf.open(meta_name), dtype={"ID": "string", "Study": "string", "class": "string"}, low_memory=False)
                assert_valid({"ID", "Study", "class"}.issubset(metadata.columns),
                             f"{meta_name}: missing required fields")
                assert_valid(counts.shape[1] == len(columns), f"{count_name}: header parsing mismatch")
                assert_valid(len(counts) == len(metadata), f"{cohort}: counts / metadata row number mismatch")

                cids, mids = counts["ID"], metadata["ID"]
                assert_valid(not cids.isna().any() and not mids.isna().any(), f"{cohort}: missing sample ID")
                assert_valid((cids.str.len() > 0).all() and (mids.str.len() > 0).all(), f"{cohort}: blank sample ID")
                assert_valid(cids.is_unique and mids.is_unique, f"{cohort}: duplicate sample ID")
                assert_valid(cids.equals(mids),
                             f"{cohort}: count ID order does NOT exactly match metadata ID order")
                ids = set(cids.tolist())
                assert_valid(all_ids.isdisjoint(ids), f"{cohort}: patient ID appears in multiple cohorts")
                all_ids.update(ids)

                labels = metadata["class"]
                assert_valid(not labels.isna().any(), f"{cohort}: missing disease label")
                permitted = set(manifest.get("label_values", ["CD", "Not IBD"]))
                assert_valid(set(labels.unique()).issubset(permitted),
                             f"{cohort}: unexpected labels {set(labels.unique()) - permitted}")
                assert_valid(len(labels.unique()) == 2, f"{cohort}: both classes are required")
                assert_valid(not metadata["Study"].isna().any(), f"{cohort}: missing source BioProject")

                values = counts.iloc[:, 1:].to_numpy(copy=False)
                assert_valid(np.issubdtype(values.dtype, np.number),
                             f"{cohort}: non-numeric count values, dtype {values.dtype}")
                assert_valid(np.isfinite(values).all(), f"{cohort}: NA, NaN or infinite count detected")
                assert_valid((values >= 0).all(), f"{cohort}: negative count detected")
                assert_valid(np.array_equal(values, np.floor(values)), f"{cohort}: fractional count detected")
                sample_totals = values.sum(axis=1, dtype=np.float64)
                assert_valid((sample_totals > 0).all(), f"{cohort}: sample has zero total assigned counts")

                expected_n = manifest.get("expected_sample_counts", {}).get(cohort)
                assert_valid(expected_n is None or len(counts) == expected_n,
                             f"{cohort}: expected {expected_n}, found {len(counts)} samples")
                label_counts = Counter(labels.tolist())
                study_counts = Counter(metadata["Study"].tolist())
                class_totals.update(label_counts)
                study_totals.update(study_counts)
                gene_sets.append(set(gene_names))
                profiles.append({
                    "cohort": cohort,
                    "samples": len(counts),
                    "genes": len(gene_names),
                    "labels": dict(label_counts),
                    "source_bioprojects": dict(study_counts),
                    "min_assigned_counts": float(sample_totals.min()),
                    "median_assigned_counts": float(np.median(sample_totals)),
                    "max_assigned_counts": float(sample_totals.max()),
                    "fraction_zero_gene_counts": float(np.mean(values == 0)),
                })
                del counts, metadata, values
    except (BadZipFile, OSError) as exc:
        raise DatasetValidationError(f"Could not read canonical ZIP: {exc}") from exc

    shared_genes = set.intersection(*gene_sets)
    expected_total = manifest.get("expected_total_samples")
    expected_shared = manifest.get("expected_shared_gene_count")
    assert_valid(expected_total is None or len(all_ids) == expected_total,
                 f"Expected {expected_total} distinct samples, found {len(all_ids)}")
    assert_valid(expected_shared is None or len(shared_genes) == expected_shared,
                 f"Expected {expected_shared} common genes, found {len(shared_genes)}")
    return {
        "status": "PASS", "input_path": str(archive_path), "sha256": actual_hash,
        "cohorts": profiles, "unique_sample_count": len(all_ids),
        "common_gene_count": len(shared_genes),
        "class_totals": dict(class_totals),
        "bioproject_counts": dict(study_totals),
        "n_source_bioprojects": len(study_totals),
        "scientific_caveats": [
            "These are supplied gene-count tables, not a verification of original sequencing reads or pathology labels.",
            "A matching ID and checksum verify the accepted artifact, not biological sample identity against source RData.",
            "Different supplied gene universes and source-study effects require review before normalisation and modelling.",
            "No expression normalisation, filtering, feature selection, or model fitting has occurred.",
        ],
    }


def save_validation_report(report: dict, root: Path) -> Path:
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = Path(root) / "runs" / "data_validation" / f"validation_{stamp}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return target

# CMPS355_Research — Workspace review and GitHub-readiness audit

**Reviewed:** 2026-11-08. Target: the user's **My Drive** project `CMPS355_Research` (Google Drive folder ID `1fYFomgpgLv6ilUvUj4q1I10z3AsSdThI`). Other folders with the same display name were not modified.

## Research products retained

The existing `runs/` hierarchy was retained intact **except a pure-R-runtime environment cache folder**, which was moved without deleting its files. Research-critical data preserved include:

- `runs/full_gene_loco/d1..d4`: outer-fold model selections, predictions, scores and reports.
- `runs/full_gene_results_report/<timestamp>`: four-fold LASSO/elastic-net aggregate tables and manuscript-quality figures.
- `runs/stability_selection/d1..d4`, `runs/stability_selection/20261107_audit`: feature-selection runs and independent audit.
- `runs/independent_signatures/d1..d4`: gene coefficients, subsample frequencies, split memberships and signature summaries.
- `runs/stablemate/d{1..4}_K2000_B1000`: author-native model evidence, metrics, predictions and significant gene lists. **No trained `.rds` object was deleted.**
- `runs/tentacles/d1..d4` and `runs/tentacles/summary`: official consensus gene sets, model provenance, warning summaries, Stage 10 predictions and held-out metrics. The four local native classifier RDS objects (~6.5 GB in WSL) were not part of the uploaded research-result archive and remain in the original local workspace.
- `runs/cohort_characterization`, `runs/pathway_analysis`, `runs/pathway_size_sensitivity`, `runs/threshold_transfer`, `runs/pilot`, `runs/pilot_diagnostics`, `runs/split_design`, `runs/setup`, `runs/method_verification`, `runs/data_validation`, `runs/tentacles_d1`, `runs/tentacles_stage9_preflight` and `runs/official_r` are retained; some are diagnostic/historical but are not automatically redundant or safe to delete without a source-dependency audit.
- `data/validated/ibd_dataset_sample_aligned.zip`: accepted corrected, sample-aligned input remains unchanged. `data/reference/reactome` and existing Reactome analyses retained.
- All source-code modules, notebooks, tests, references, `protocol/` course PDFs and configuration files retained.

## Concrete cleanup performed

1. **Large environmental artifacts moved (not deleted)** to sibling private Drive folder `CMPS355_Runtime_Archives` (`1yVOtSyy2RY3KzprEimrlPlTiidl42Hd5`):
   - `TENTACLES_Ubuntu2404_build_archive.tar.gz` (formerly at project root, ~696.8 MB).
   - `r4.6.1_ubuntu24.04_x86_64.tar.gz` (~263.0 MB), plus its JSON manifest (formerly `runs/environment_cache/`).
   - `CMPS355_TENTACLES_Local_Starter.zip` (formerly under `docs/`).
   - **Important:** the November 7 historical Stage 9 protocol still refers to the *old* `runs/environment_cache/` cache path. Future installs should obtain the archive from the sibling folder or rebuild the pinned environment using the existing scripts.
2. Removed the now-empty `runs/environment_cache/` folder and Python-only `src/cmps355/__pycache__/` (nine generated `.pyc` files). Neither deletion removes research source code or experimental results.
3. Replaced the obsolete Stage 1 `_Fresh` root README, release guidance and `.gitignore` with current versions. The **original three documents** are preserved in private `docs/history/` as `README_stage1_previous.md`, `gitignore_stage1_previous.txt`, and `github_reproducibility_stage1_previous.md`. Removed six unnecessary temporary and duplicate backup copies after verification.
4. Standardized the notebook name `11-full-gene-loco-d4.ipynb` to `11_full_gene_loco_d4.ipynb` to match D1–D3, without changing its contents.
5. Added `results/tentacles`, `results/comparisons`, `results/figures`, `docs/TENTACLES_INDEPENDENT_AUDIT_20261108.md` and `src/cmps355/audit_tentacles_stage10_outputs.py`. The public-results tables contain no patient IDs; the **complete**, original patient-level CSVs remain in private Drive `runs/`.

## GitHub release policy

The updated `.gitignore` **tracks `results/`** (safe summary figures and tables), source, tests, documentation and notebooks while **ignoring `runs/`, `data/validated/*.zip`, `.rds`, caches, R runtime archives and private proposal/course PDFs**. `.gitignore` only protects `git add` operations; a manual web upload or files already in Git history can bypass it. Verify public release inputs explicitly.

The presence of a full `runs/` tree **in Drive** is not a reason to delete scientific evidence. We intentionally preserved raw predictions and all heavy finished native StableMate `.rds` models in Drive while keeping them out of the public Git view.

## Audit and methodological status

- Independent Stage 10 TENTACLES **saved-output** audit: **170 assertions, 0 failures**, covering archive file digests, patient counts/IDs, probability bounds and class mapping, confusion matrices, AUROC, balanced accuracy, precision, F1, Brier, exact native gene panels, consensus frequencies and 1,000-resample CIs. This does not establish native R training correctness without inspecting `.rds` objects.
- All 689 patient IDs are matched one-to-one between TENTACLES Stage 10 and StableMate Stage 8 held-out predictions, with consistent class/study metadata.
- Direct TENTACLES–StableMate and Stage 5 method rankings are **descriptive only**: StableMate uses native author prediction, whereas TENTACLES uses a fixed L2 logistic classifier on its native consensus panels; Stage 5 LASSO/elastic-net use their own inner tuning.
- Native TENTACLES MLP/LightGBM/Boruta warnings need a method-level review before describing all ten contributing models as healthy.
- A fully controlled feature-selection benchmark still requires running **one identical downstream classifier and preprocessing** on all methods' selected gene panels. Cross-LOCO consensus overlaps are not independent-cohort biomarker replication.
- Public reproduction of official R methods also needs tested dependency/environment restoration guidance; the pinned package Git commit is recorded, but an independently verified R `renv.lock` with transitive package versions is not present.

## Scope and residuals

This was a broad **hierarchy, source-document, output-schema, run-artifact and reproducibility review**, with independent numerical verification of the TENTACLES/StableMate outputs. It was **not** a line-by-line independent audit of all historical notebook source, every native R model object or every older experiment's clinical correctness. Historical installation logs and old pilot outputs were **deliberately retained** instead of guessing they were redundant; these are ignored for GitHub by the new `.gitignore`.

**No important training outputs, patient-level predictions, original gene lists, cross-cohort analyses, baseline method figures, source dataset or reference data were deleted.**

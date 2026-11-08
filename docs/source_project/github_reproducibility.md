# CMPS355 GitHub reproducibility, safe-release and provenance policy

## Directory and identity

Use a single project root named `CMPS355_Research`. Historical documentation may refer to the obsolete `_Fresh` suffix; it is not the active root. Preserve all existing Drive run evidence; never overwrite experiment folders by uploading a replacement full-project ZIP.

## Source and input verification

- Python setup: `python -m pip install -r requirements-dev.txt`; synthetic tests: `python -m unittest discover -s tests -v`.
- Supply the **accepted corrected sample-aligned** dataset as `data/validated/ibd_dataset_sample_aligned.zip` (not tracked in Git). SHA-256: `8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74`.
- Accepted data: 689 samples, 4 analytical cohorts, 9,983 common genes; six original BioProjects. The upstream Zenodo publication is a *source*, not an automatic checksum-equivalent replacement for the corrected archive.
- StableMate package pinned at `cf0fe9f344c1756b6f89fd33f9fdae1a7f35b24a`; TENTACLES pinned at `069a26271e3885e015df25baa40a7a1140827d5f`. Record environment package versions and native R session info for reproduce-on-new-machine workflows; we do not claim a full public-container rebuild is provided.

## Which data are public?

Track source (`src/`, `r/`, `scripts/`), configs, notebooks cleared of outputs, tests, documentation and **curated aggregate/gene-level `results/`**. Do not commit `runs/`, patient/sample ID level predictions, the corrected dataset ZIP, raw individual expression data, native `.rds` models, temporary environments or course/proposal PDFs without permission. `.gitignore` enforces the default exclusions only for *untracked* files. If private files were previously committed, remove them from Git history before public release.

## Method/figure integrity

A fixed-feature-panel L2 logistic evaluation of TENTACLES consensus is **not** a native external TENTACLES predictor. StableMate's author-native predictor and Stage 5 LASSO/ElasticNet differ in tuning/preprocessing; simple AUROC rankings cannot isolate feature selection. Use `results/comparisons/README.md` and `docs/TENTACLES_INDEPENDENT_AUDIT_20261108.md` as figure caveats. Four LOCO discovery sets overlap (do not call Jaccard 'independent cohort replication').

## Publishing checklist

1. Confirm no private identifiers appear in tracked outputs, notebook saves, file names or commit history.
2. Verify all figures correspond to their saved source tables and list sample counts, thresholds and uncertainty units.
3. Validate code/test success and reproduce representative summary tables from available outputs.
4. Review third-party software licences, reference datasets and instructor materials; choose your own code licence explicitly.
5. Record Git SHA and a release tag. Do not publish private Google Drive URLs or locally cached binary runtime archives.

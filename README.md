# Cross-cohort transcriptomic feature selection and biomarker reproducibility

**CMPS 355 graduate research project | Abbas El-Hajj Youssef and Jad Mcheik (AUB, 2026)**

This repository accompanies the revised IEEE-format manuscript, *Benchmarking Transcriptomic Feature Selection for Cross-Cohort Predictive Performance and Biomarker Reproducibility*.

## Final revised manuscript and evidence

The post-review release has passed technical, scientific and editorial verification. Author approval and course submission remain pending.

- [Clean revised manuscript (12 pages)](post_review_2026-11-23/submission/CMPS355_Updated_Manuscript.pdf)
- [Reviewer 1 response](post_review_2026-11-23/submission/Response_Reviewer1_Aya_Daou.pdf) and [highlighted manuscript](post_review_2026-11-23/submission/Highlighted_Reviewer1_Aya_Daou.pdf)
- [Reviewer 2 response](post_review_2026-11-23/submission/Response_Reviewer2_Leen.pdf) and [highlighted manuscript](post_review_2026-11-23/submission/Highlighted_Reviewer2_Leen.pdf)
- [Combined response](post_review_2026-11-23/submission/Response_Both_Reviewers.pdf) and [combined highlights](post_review_2026-11-23/submission/Highlighted_Both_Reviewers.pdf) (optional)
- [Release documentation](post_review_2026-11-23/README.md), [LaTeX source](post_review_2026-11-23/manuscript/), [PDF integrity manifest](post_review_2026-11-23/submission/BUILD_MANIFEST.json), [SHA-256 checksums](post_review_2026-11-23/submission/SHA256SUMS.txt)

## Main findings

Across four Crohn's disease versus Not IBD cohorts (689 samples; 9,983 shared genes), the common class-balanced logistic classifier gave descriptive mean held-out AUROCs of 0.728 (LASSO), 0.731 (Elastic Net), 0.628 (Stability Selection), 0.659 (StableMate) and 0.723 (TENTACLES). TENTACLES had the highest mean balanced accuracy (0.655) and mean pairwise LOCO Jaccard (0.210). Training-only differential-expression top-ten baselines achieved 0.748 (limma-voom) and 0.772 (Welch).

These are four-cohort descriptive estimates, not formal evidence of superiority or disease-specific clinical biomarkers. LOCO development cohorts overlap. Upstream filtering provenance for D1-D3 remains incomplete, and Reactome enrichment was not repeated for the corrected sparse-model panels.

## Repository guide

| Path | Purpose |
|---|---|
| [post_review_2026-11-23/](post_review_2026-11-23/) | Canonical revised paper, scripts, aggregate results, panels, run manifests and independent checks |
| [src/](src/), [r/](r/), [scripts/](scripts/), [tests/](tests/), [configs/](configs/) | Foundational implementations, environments and tests; required for historical and reused methods |
| [notebooks/](notebooks/) | Original notebook sequence and execution-status record |
| [results/](results/) | Historical aggregate results and retained TENTACLES evidence; not the corrected sparse-model results |
| [reports/](reports/), [docs/](docs/) | Historical methodological and audit provenance |
| [Original pre-review snapshot](https://github.com/abbaselhajj/CMPS355-Transcriptomic-Feature-Selection/tree/8c06ea0b07a5256d507e578bd32f9bbb8e4f37dc) | Immutable original code, manuscript, compiled PDF and evidence manifest |

For the corrected benchmark, use `post_review_2026-11-23/results/`, not the older `results/final_five_method/` tables. The old manuscript and its verification assets have been removed from the active branch to avoid competing versions; all original files remain retrievable at the pinned pre-review commit.

## Reproducibility

- Review the [post-review release documentation](post_review_2026-11-23/README.md) for the exact revised analyses and their provenance.
- Build the final manuscript with LaTeX and BibTeX from `post_review_2026-11-23/manuscript/`: run `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice.
- Verify submission PDF hashes using `sha256sum -c SHA256SUMS.txt` within `post_review_2026-11-23/submission/`.
- For the original benchmark only, `python scripts/check_public_evidence.py` checks the historical aggregate tables, not the corrected post-review metrics.
- An end-to-end rerun requires the source RNA-seq inputs, accepted aligned ZIP (SHA-256 `8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74`), pinned Python/R packages and private native-model artifacts. The source study identifiers and citations are given in the revised manuscript. Do not interpret a public clone as a full rerun.

Raw matrices, patient-level predictions, native R model objects and private reviews remain outside the public repository. The retained historical evidence manifest is available at the original snapshot, where its paths and checksums apply.

**Canonical verified scientific release:** [24 November 2026, commit ef77e3c](https://github.com/abbaselhajj/CMPS355-Transcriptomic-Feature-Selection/tree/ef77e3c17ba611720c6c147fe35b9a7212b5d35d/post_review_2026-11-23). Subsequent repository housekeeping does not change the scientific release or its PDFs.

# Cross-cohort transcriptomic feature selection and biomarker reproducibility

**Abbas El-Hajj Youssef and Jad Mcheik | American University of Beirut | 2026**

Research repository accompanying *Benchmarking Transcriptomic Feature Selection for Cross-Cohort Predictive Performance and Biomarker Reproducibility*.

## Manuscript and research materials

- [Revised manuscript](post_review_2026-11-23/paper/CMPS355_Updated_Manuscript.pdf), including supplementary tables and figures.
- [Response to Reviewer 1](post_review_2026-11-23/paper/Response_Reviewer1_Aya_Daou.pdf) and [annotated manuscript](post_review_2026-11-23/paper/Highlighted_Reviewer1_Aya_Daou.pdf).
- [Response to Reviewer 2](post_review_2026-11-23/paper/Response_Reviewer2_Leen.pdf) and [annotated manuscript](post_review_2026-11-23/paper/Highlighted_Reviewer2_Leen.pdf).
- [Combined response](post_review_2026-11-23/paper/Response_Both_Reviewers.pdf) and [combined annotated manuscript](post_review_2026-11-23/paper/Highlighted_Both_Reviewers.pdf).
- [LaTeX source](post_review_2026-11-23/manuscript/), [revised analyses](post_review_2026-11-23/README.md), and [PDF checksums](post_review_2026-11-23/paper/SHA256SUMS.txt).

## Study design and findings

The study compares five feature-selection methods on four intestinal RNA-seq cohorts comprising 689 samples (495 Crohn's disease, 194 Not IBD) and 9,983 shared genes. Panels are evaluated using leave-one-cohort-out validation and a common class-balanced logistic classifier.

| Method | Mean held-out AUROC |
| --- | ---: |
| LASSO | 0.728 |
| Elastic Net | 0.731 |
| Stability Selection | 0.628 |
| StableMate | 0.659 |
| TENTACLES | 0.723 |

TENTACLES attained the highest mean balanced accuracy (0.655) and mean pairwise LOCO Jaccard (0.210). Training-only top-ten differential-expression baselines attained AUROCs of 0.748 (limma-voom) and 0.772 (Welch). These descriptive comparisons demonstrate the importance of evaluating prediction and gene-set reproducibility separately.

The LOCO training sets overlap, and the reported size-aware nulls do not account for that dependence. Upstream filtering provenance for D1-D3 remains incomplete; Reactome enrichment was not repeated on the corrected sparse-model panels. The available comparisons do not establish Crohn's-disease-specific or clinically validated biomarkers.

## Repository structure

| Path | Contents |
| --- | --- |
| [post_review_2026-11-23/](post_review_2026-11-23/) | Revised analyses, manuscript, selected gene panels, aggregate results and verification |
| [src/](src/), [r/](r/), [scripts/](scripts/), [tests/](tests/), [configs/](configs/) | Implementations, configurations and tests |
| [notebooks/](notebooks/) | Original numbered analysis workflow |
| [results/](results/) | Original aggregate results and retained TENTACLES evidence |
| [reports/](reports/), [docs/](docs/) | Methodological records and original analysis evidence |
| [Original research snapshot](https://github.com/abbaselhajj/CMPS355-Transcriptomic-Feature-Selection/tree/8c06ea0b07a5256d507e578bd32f9bbb8e4f37dc) | Original manuscript, code and results |

For the revised benchmark, use `post_review_2026-11-23/results/`. The original `results/final_five_method/` tables document earlier analyses and are not the corrected sparse-model results.

## Reproducibility

Build the manuscript in `post_review_2026-11-23/manuscript/` using `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice. Verify PDF integrity using `sha256sum -c SHA256SUMS.txt` in `post_review_2026-11-23/paper/`.

The original benchmark's aggregate evidence can be checked using `python scripts/check_public_evidence.py`. The revised analysis scripts and run manifests are documented in the [research release](post_review_2026-11-23/README.md).

Full reproduction requires the source RNA-seq inputs, aligned input archive (SHA-256 `8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74`), pinned Python/R dependencies and applicable native-model artifacts. BioProject accessions and source references are listed in the manuscript. Raw expression matrices, individual predictions and native R objects are not redistributed through this public repository.

[Research release snapshot, 24 November 2026](https://github.com/abbaselhajj/CMPS355-Transcriptomic-Feature-Selection/tree/ef77e3c17ba611720c6c147fe35b9a7212b5d35d/post_review_2026-11-23).

# Revised research analyses

*Benchmarking Transcriptomic Feature Selection for Cross-Cohort Predictive Performance and Biomarker Reproducibility*

Abbas El-Hajj Youssef and Jad Mcheik | American University of Beirut | 24 November 2026

## Manuscript

- [Revised manuscript](paper/CMPS355_Updated_Manuscript.pdf) with supplementary Tables S1-S6 and Figure S1.
- Reviewer 1: [response](paper/Response_Reviewer1_Aya_Daou.pdf) and [annotated manuscript](paper/Highlighted_Reviewer1_Aya_Daou.pdf).
- Reviewer 2: [response](paper/Response_Reviewer2_Leen.pdf) and [annotated manuscript](paper/Highlighted_Reviewer2_Leen.pdf).
- [Combined response](paper/Response_Both_Reviewers.pdf) and [combined annotated manuscript](paper/Highlighted_Both_Reviewers.pdf).
- [Manuscript source](manuscript/), [build manifest](paper/BUILD_MANIFEST.json) and [SHA-256 checksums](paper/SHA256SUMS.txt).

The marked manuscripts preserve the revised article text and indicate passages, figures and tables relevant to each review. The original manuscript is available at [commit 8c06ea0](https://github.com/abbaselhajj/CMPS355-Transcriptomic-Feature-Selection/tree/8c06ea0b07a5256d507e578bd32f9bbb8e4f37dc/manuscript).

## Analyses

Five feature-selection methods were evaluated under a common class-balanced logistic classifier across four cohorts (689 samples, 9,983 shared genes).

- **Sparse-model fitting:** LASSO and Elastic Net were refitted with convergence and Karush-Kuhn-Tucker checks. The original independent D3 LASSO panel contained 1,734 genes; the corrected fit selected nine. Elastic Net was verified across 200 fitted grid models.
- **Reference baselines:** all 9,983 genes, fold-training-eligible genes, training-only limma-voom and Welch rankings, and 100 size-matched random panels per method and cohort (2,000 draws).
- **Predictive uncertainty:** eight held-out metrics, with 2,000-resample stratified bootstrap intervals for AUROC and balanced accuracy. Native and common-classifier predictions are reported separately.
- **Gene-set reproducibility:** exact expected overlaps and 10,000-draw size-aware Monte Carlo references. Ranked fixed-size comparisons use K = 10, 25, 50, 100 and 200 when ranking and support are available.
- **Cohort sensitivity:** D3 test-only rescoring after excluding seven source-flagged samples (n = 76 to 69), without retraining the other folds.

The revised mean held-out AUROCs are 0.728 (LASSO), 0.731 (Elastic Net), 0.628 (Stability Selection), 0.659 (StableMate) and 0.723 (TENTACLES). All-gene, limma-voom top-ten and Welch top-ten references attained 0.681, 0.748 and 0.772, respectively.

The original CD-positive TENTACLES consensus runs, StableMate and Stability Selection panels were retained. Their native training was not repeated during the revision. Table S6 documents which analyses were newly executed and which use existing artifacts.

## Research evidence

| Directory | Contents |
| --- | --- |
| [scripts/](scripts/) | Sparse-model, baseline, overlap and fixed-size analysis scripts and tests |
| [notebooks/](notebooks/) | Code-only notebook runners; executed outputs are not included |
| [results/](results/) | Aggregate performance, uncertainty, random-panel, fixed-size, overlap and D3-QC results |
| [panels/](panels/) | Corrected sparse-model and differential-expression gene lists |
| [manifests/](manifests/) | Frozen run provenance and input references |
| [audit_replay/](audit_replay/) | Numerical verification code and machine-readable summaries |
| [manuscript/](manuscript/) | IEEEtran LaTeX source, bibliography and figures |
| [paper/](paper/) | Compiled manuscript, review materials and integrity manifests |

## Reproducibility and limitations

The aligned input archive has SHA-256 `8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74`. Source BioProject accessions are documented in the manuscript. An end-to-end rerun requires the original inputs, pinned Python/R dependencies and native-model artifacts. Raw expression matrices, individual predictions and native model objects are not included in the public repository. Run manifests retain checksums for private or unavailable inputs.

The upstream D1-D3 filtering criteria remain incompletely documented. Reactome enrichment was not repeated for the corrected LASSO and Elastic Net panels. The size-only overlap reference does not account for gene dependence or shared LOCO training cohorts. Bootstrap intervals are conditional on fitted predictions, and CD versus Not IBD does not establish disease specificity. Reported comparisons are descriptive rather than formal evidence of algorithm superiority.

The research release snapshot is preserved at [commit ef77e3c](https://github.com/abbaselhajj/CMPS355-Transcriptomic-Feature-Selection/tree/ef77e3c17ba611720c6c147fe35b9a7212b5d35d/post_review_2026-11-23).

# Stage 8 — Independent StableMate result audit (7 November 2026)

## Status and verification scope

**PASS for the implemented independent numerical, data-partition, training-only preprocessing, signature-threshold and file-integrity checks.** This does **not** establish complete reproducibility of native R predictions from a fresh R environment or biological validity.

- 4/4 folds complete: K=2000, B=1000, one worker each.
- 56/56 recorded SHA-256 output targets independently recomputed and matched, including all four native `model_finished.rds` files.
- 689/689 unique held-out patients, with identity, source BioProject, binary clinical labels and fold memberships independently checked against the accepted corrected archive.
- All 4 folds' train-only gene filters, gene indices, means and population-standard-deviation scales independently recomputed from the accepted raw counts; maximum absolute residual below 3×10^-15.
- Recomputed AUROC, confusion counts, accuracy, balanced accuracy, sensitivity, specificity, precision, F1, Brier and log loss from predicted probabilities; all match saved reports.
- Recomputed significant genes from authors' >0.975 predictivity and stability significance scores; all match exported subsets and counts.

## Held-out predictions (CD positive)

| Fold | n | AUROC | 95% stratified bootstrap AUROC CI | Balanced accuracy | Sensitivity | Specificity | F1 | Stable-significant genes |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| D1 | 260 | 0.8541 | [0.798, 0.906] | 0.6080 | 0.9541 | 0.2619 | 0.9103 | 20 |
| D2 | 179 | 0.7197 | [0.630, 0.808] | 0.6841 | 0.7015 | 0.6667 | 0.7737 | 14 |
| D3 | 76 | 0.6709 | [0.527, 0.796] | 0.5371 | 0.7647 | 0.3095 | 0.5843 | 14 |
| D4 | 174 | 0.7284 | [0.645, 0.800] | 0.6647 | 0.4679 | 0.8615 | 0.6036 | 15 |

**Macro AUROC: 0.7433**, cohort SD **0.0781**. **Macro balanced accuracy: 0.6235**. Unweighted macro averaging treats the four cohorts as the principal reporting units. AUROC confidence intervals use 2,000 *within-cohort stratified* bootstrap replicates; they are not a confidence interval for population-generalization across future cohorts.

### Biomarker agreement (exploratory)

Significant stable-gene sizes: D1=20, D2=14, D3=14, D4=15. The four-way intersection is empty, and there are **58 distinct genes** in the union. The six pairwise Jaccard coefficients average **0.0326**. These are **not** independent-cohort signature comparisons because the LOCO training samples overlap across folds.

## Provenance and caveats

- Accepted sample-aligned data SHA-256: `8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74`.
- Author StableMate R package: v0.1.0, commit `cf0fe9f344c1756b6f89fd33f9fdae1a7f35b24a`.
- Native author prediction: logistic base learner, predictive and stability objectives; K=2000, B=1000; random LASSO prefilter `p=50`, `group_by='Yenv'`.
- **Executed** runner SHA-256: `81b74b2d227a99b197c8507d8277b880e2403a50c991ee91715ce480eb2da4d0` (`stablemate_stage8_traced_final.R`).
- Preparation manifest refers to earlier runner SHA-256 `59f89ab775ca9fc82518c6f552605bf934631eb40527c700ee5ac491f45a305b` (`run_stablemate_stage8.R`). These are distinct stages; do not retroactively edit manifests.
- Preparation manifest `model.ncore=2`; actual final `run_summary.workers=1` and `protocol.txt` declares `ncore 1`. Record **one worker** as executed.
- `protocol.txt` reports known binomial GLM fractional outcome–environment weights warning; its impact is not independently evaluated.


The original RDS files were verified by cryptographic checksum; **the fitted R models were not independently loaded and re-predicted**. Testing the author model against its own predictions validates arithmetic and identities but does not establish that test labels were absent from all computational paths. Gene preprocessing and outer-split separation were independently verified; complete original-code auditing remains distinct.

RDS model files, patient predictions, original scores, checksums and author-oriented significance values are preserved unchanged under `runs/stablemate/d{1,2,3,4}_K2000_B1000/`.

## Figures and tables (generated, do not treat as new experiments)

- `stablemate_stage8_verified_per_fold.csv`: one row per held-out cohort with report metrics, diagnostics and comparator baselines.
- `stablemate_stage8_auc_ci.csv`: bootstrap intervals for cohort AUROC.
- `stablemate_stage8_gene_overlap.csv`: six pairwise Jaccard scores plus intersecting gene names.
- `figure_stage8_auroc_by_cohort.png`: AUROC with within-cohort bootstrap intervals.
- `figure_stage8_threshold_metrics.png`: sensitivity/specificity/balanced accuracy at 0.5.
- `figure_stage8_gene_jaccard.png`: exploratory LOCO gene overlap.
- `stablemate_stage8_audit_report.json`: machine-readable verification and source identity report.

No held-out labels were used to refit the models or to select a threshold in these auditing/figure-generation steps.

# CMPS355 final manuscript: evidence and integrity audit

Date: 8 November 2026. Intended deliverable: an 8–12-page, evidence-complete draft for graduate peer review (IEEEtran A4 conference style); this version contains **8 pages inclusive of all references**.

## Source requirements reviewed
- Connected Drive `CMPS355_Research/protocol/project_guidelines.pdf`: complete manuscript with abstract, Introduction, Methods, Results, Discussion, figures/tables and references; reproducible Methods and evidence-limited interpretation.
- `protocol/proposal.pdf`: fixed research question about cross-cohort prediction versus gene-signature reproducibility; five nominated selection approaches; LOCO; gene and pathway consistency.
- `protocol/from_proposal_to_paper.pdf`: figures answer questions, self-contained captions, uncertainty and sample sizes, Results distinguish observation from Discussion, conservative abstract.
- `protocol/peer_review_and_rebuttals.pdf`: explicit limitations and inspectable evidence chain.

## Independent numerical checks
- Aligned count and metadata archive (original integrity SHA-256): `8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74`.
- User-supplied private export: `CMPS355_TENTACLES_V2_Final_Results_For_Analysis(1).zip`. Verified all **148 constituent-file SHA-256** hashes against `EXPORT_MANIFEST.json`, plus file sizes, complete primary and secondary predictions, exact CD-positive class labels, fold membership and metadata IDs.
- For **both** 6/9 and 5/9 consensus settings: verified all four panel sets against the per-gene native voter counts, native consensus gene lists, confusion matrices, AUROC, average precision, balanced accuracy, Brier score and F1; **689 unique held-out patients per setting**.
- Refitted independent C=1, balanced liblinear, L2 logistic predictions on training-only-standardized, 9,983-gene log2(CPM+1) data using the primary V2 panels. Maximum absolute probability difference versus archived predictions: **3.97e-5** in four-fold comparison under scikit-learn 1.8.0; this is numerical agreement, not identical bitwise reproducibility.
- Recomputed **all 30 pairwise LOCO Jaccards across five methods**, the gene-panel sizes, intersection sizes, and four-method prior top-four sensitivity.
- Recomputed all 20 method-by-fold Reactome enrichments from pinned Reactome v98 (ZIP checksum `6056fcd95dc8f09740c8bb64c2765bea1386ce91b9b19cd10bcd45d30451805c`) and original training-only eligible-gene backgrounds. All eight earlier LASSO/Elastic Net enrichment summaries were **identically reproduced**. TENTACLES V2's 0/11/8/9 pathway counts and six-pair mean pathway Jaccard 0.25641 are new, not copied from V1.
- Historical common-learner performance of four other methods was imported unaltered from audited Stage 22. TENTACLES V1 signatures/metrics were **excluded** from final five-method numerical comparisons.
- Full private prediction rows and original sample IDs are **not included** in the shareable manuscript ZIP. Only aggregate CSV tables, analysis code and audit receipts are included.

## Primary audited results (all five methods under the same downstream learner)

| Selector | Mean held-out AUROC | Mean balanced accuracy | Mean gene LOCO Jaccard | Mean panel size |
|---|---:|---:|---:|---:|
| LASSO | 0.689087 | 0.619045 | 0.210809 | 281.50 |
| Elastic Net | 0.711846 | 0.628092 | 0.115463 | 352.50 |
| Stability Selection, cutoff 0.60 | 0.628421 | 0.562784 | 0.127834 | 6.25 |
| StableMate stable-significant | 0.659037 | 0.597936 | 0.032580 | 15.75 |
| TENTACLES V2, 6/9 votes | 0.722530 | 0.655058 | 0.209673 | 76.50 |

## Open scientific qualifications
- A four-cohort observational benchmark cannot establish formal method superiority on future cohorts. Within-cohort patient bootstrap confidence intervals do not capture new-cohort uncertainty.
- Each pair of LOCO development sets shares two of the four cohorts. The six fold-pair Jaccards are **dependent descriptive comparisons**, not independent cohort replications.
- Gene signature lengths vary greatly. Only four selectors admit a top-four size-matched sensitivity analysis based on defensible within-panel rankings; TENTACLES native vote counts contain ties.
- Native TENTACLES RDS objects were not contained in the compact private export; no full independent R retraining was attempted. D3/D4 consensus recoveries retain original fitted models; D4 excluded XGBoost had four rather than five selected tuning resamples. Historical native Boruta, MLP and LightGBM warnings remain documented.
- Seven D3 source records had an exclusion QC flag. They were retained to preserve the published 76-sample benchmark; the sensitivity analysis excluding them has **not** been performed.
- The `Not IBD` controls are heterogeneous, and D4 lacks supplied tissue-site labels. Reactome associations are exploratory and not evidence of gene-specific causation or clinical validity.

## Formatting and reporting
- IEEEtran A4 two-column, 8 pages including 13 references; full PDF visually inspected after regeneration.
- Figures: cohort PCA, 5x4 held-out AUROC/BA heatmap, bubble plot of AUROC versus gene LOCO overlap, repeated TENTACLES Reactome q-value matrix. Values/units and caveats are explained in captions.
- Tables: cohort composition, LOCO assignments, method panel definitions, five-way summary, TENTACLES threshold sensitivity, separate-cohort LASSO/Elastic Net signatures, five-way pathway counts and overlap.
- No intentionally missing TENTACLES final values, V1 substitutions or invented significance claims.

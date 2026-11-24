# CMPS355 : Post-review reproducibility release (23 November 2026)

This versioned directory documents the peer-review response to the four-cohort Crohn's disease RNA-seq feature-selection benchmark. **The pre-review code and manuscript remain in their original repository paths.** This revision has been prepared for the authors' final review; it has not been submitted as coursework.

## Paper and reviewer responses
- [Revised 12-page manuscript](submission/CMPS355_Updated_Manuscript.pdf), with supplementary Tables S1–S6 and Figure S1 embedded in its appendix.
- Reviewer 1 (Aya Daou): [point-by-point response](submission/Response_Reviewer1_Aya_Daou.pdf) and [marked manuscript](submission/Highlighted_Reviewer1_Aya_Daou.pdf).
- Reviewer 2 (Leen): [point-by-point response](submission/Response_Reviewer2_Leen.pdf) and [marked manuscript](submission/Highlighted_Reviewer2_Leen.pdf).
- Optional combined [response](submission/Response_Both_Reviewers.pdf) and [marked manuscript](submission/Highlighted_Both_Reviewers.pdf).
- Buildable conference IEEEtran manuscript source, 18-entry bibliography, manuscript sections and vector figures: [manuscript/](manuscript/).

The reviewer-specific highlighted PDFs preserve the clean manuscript's 12 A4 pages and article text. Complete revised passages are highlighted, and updated figures and tables are outlined. Numbered visible page notes and PDF annotations explain each local change and link it to the corresponding comment ID in the response letter. Reviewer 1 uses A1 to A10; Reviewer 2 uses L1 to L8. The combined version distinguishes reviewer-specific and shared revisions. These are reviewer-scoped change markings, not token-level latexdiff.

## What was revised
Five feature-selection methods were evaluated under one controlled downstream classifier: LASSO, Elastic Net, Stability Selection, StableMate and TENTACLES. Data comprise 689 samples in four cohorts, with 9,983 shared genes and leave-one-cohort-out evaluation.

- Corrected the early-terminated SAGA sparse-model defect. The independently fitted D3 LASSO panel changed from 1,734 historical genes to 9 after numerical correction and independent KKT checks. Elastic Net was also refitted and independently audited over 200 grid fits.
- Added all-gene, fold-training-eligible, training-only limma-voom, Welch, and panel-size-matched random baselines. The four-fold descriptive macro AUROC of all 9,983 genes is 0.681; limma-voom top-10 0.748, Welch top-10 0.772.
- Reported per-cohort metrics and 2,000-bootstrap AUROC/balanced-accuracy confidence intervals across the five methods, including separate native-versus-controlled predictions.
- Added 10,000-draw **size-only** gene-overlap null expectations and fixed K=10/25/50/100/200 sensitivity where real rankings exist. The LOCO training sets overlap, and these nulls are **not dependence-calibrated hypothesis tests**.
- Documented the seven previously flagged D3 records via **held-out-only** test rescoring (n=76 to 69), not full retraining.
- Narrowed biological conclusions: CD versus Not IBD does not establish Crohn's-disease specificity or a validated clinical biomarker.

Four-fold descriptive mean AUROC for primary methods: Elastic Net 0.7311, LASSO 0.7278, TENTACLES 0.7225, StableMate 0.6590 and Stability Selection (threshold 0.60) 0.6284. These are descriptive dependent-cohort comparisons, not formal evidence of algorithm superiority.

## File map and independent checking
- [scripts/](scripts/): corrected LASSO, corrected Elastic Net, C1 baseline, C2/C3 overlap/fixed-K scripts and corresponding adversarial/negative tests.
- [notebooks/](notebooks/): code-only Colab runners, with executed cell outputs and account-specific metadata removed. **The public notebook copies are not independent execution records.**
- [results/](results/): frozen aggregate five-method, baseline, random-panel, fixed-K, chance-overlap, and D3-QC metrics. No patient-level rows or individual model probabilities.
- [panels/](panels/): corrected LASSO/Elastic Net selected-gene lists, plus public-safe limma/Welch top-10 gene lists, split by cohort.
- [manifests/](manifests/): immutable run receipts. SHA-256 references to larger, unreleased private inputs/checkpoints remain for provenance; those files are not implied to exist in this public clone.
- [audit_replay/](audit_replay/): independent scientific verification scripts and small machine-readable audit summaries, separate from authors' production outputs.

Data alignment ZIP SHA-256: `8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74`. The RNA-seq source BioProject accessions and original citations are in the manuscript. The public repository does **not** redistribute the count matrices, patient-level predictions, private audit archives or native R model objects. An end-to-end reproduction requires independently obtaining the source inputs under their data-use permissions, recreating the pinned Python/R environment and executing the scripts with the complete original inputs. A fresh clone alone is not a full rerun.

Two reviewer requests remain bounded by available evidence: exact **upstream filtering criteria** for D1–D3 cannot be independently reconstructed, and Reactome enrichment was **not rerun** on the corrected LASSO and Elastic Net final panels. Both are disclosed in the paper and reviewer response. Authors retain responsibility for final wording, scientific interpretation and submission.


## Original-manuscript comparison check (24 November 2026)

The marked copies and point-by-point letters were rechecked against the preserved pre-review manuscript source at commit `8c06ea0b07a5256d507e578bd32f9bbb8e4f37dc` and the two original reviews. The revised Table S6 distinguishes post-review sparse-model refits and new reference analyses from the pre-existing CD-positive TENTACLES, StableMate and Stability Selection panels. TENTACLES native model training was not repeated for this revision. No experimental numerical results were changed in this editorial/provenance correction.

## Final visual and reporting polish (24 November 2026)

Figures 3 and 4 and supplementary Figure S1 were redrawn from frozen aggregate results to prevent label collisions. Section II-B restores the Elastic Net hyperparameter grid, and Table S6 now explicitly records D3 held-out-only QC rescoring. The response letters have improved page breaks. No model was refitted or result changed. The latest seven PDFs are byte-identical to the corresponding Drive files; the existing GitHub filenames have been retained for link stability. The exact filename mapping, page counts and SHA-256 digests are in [submission/BUILD_MANIFEST.json](submission/BUILD_MANIFEST.json), and [submission/SHA256SUMS.txt](submission/SHA256SUMS.txt) supports independent verification. Author approval remains pending.

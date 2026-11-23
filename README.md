# Cross-cohort transcriptomic feature selection and biomarker reproducibility

**CMPS 355 graduate research project — Abbas El-Hajj Youssef and Jad Mcheik (AUB, 2026)**

This repository is the **research-evidence snapshot** accompanying our IEEE-format manuscript, *Benchmarking Transcriptomic Feature Selection for Cross-Cohort Predictive Performance and Biomarker Reproducibility*. We studied five feature-selection strategies on four analytical Crohn's-disease RNA-seq cohorts (689 samples, 9,983 common genes) using leave-one-cohort-out (LOCO) external evaluation.

## Main findings

Under a **common train-only-scaled, class-balanced L2 logistic classifier**, the four-cohort mean AUROC was 0.689 (LASSO), 0.712 (Elastic Net), 0.628 (Stability Selection), 0.659 (StableMate), and **0.723 (TENTACLES V2 six-of-nine consensus)**. These results are descriptive: signature sizes and upstream feature-selection procedures differ, and the four LOCO training sets overlap. See `manuscript/main.pdf` for cohorts, methods, figures, uncertainty and limitations.

## Where to start

| Path | Purpose |
|---|---|
| `manuscript/main.pdf` | Latest 8-page IEEE manuscript, including the authors’ university emails and GitHub availability statement |
| `manuscript/main.tex`, `manuscript/sections/`, `manuscript/figures/` | Complete paper source; build with `bash manuscript/build.sh` |
| `results/final_five_method/` | Verified cohort metrics, Jaccard pairs, panel definitions, Reactome outputs and V2 sensitivity |
| `results/original_sparse_models/` | Original LASSO/Elastic Net gene lists, hyperparameter and held-out summaries |
| `results/tentacles_v2/` | Original V2 nine-voter gene lists, native votes, warning and provenance files, six/five-vote external aggregate evaluation |
| `reports/` | Native-model and independent numerical audit records |
| `src/cmps355/`, `r/`, `scripts/`, `tests/` | Original Python/R implementations, runners and checks |
| `notebooks/` | Original Drive `.ipynb` files, including **all saved execution outputs** (no outputs were stripped) |
| `notebooks/NOTEBOOK_STATUS.csv` | Which notebooks were executed and which remained entry-point templates |
| `configs/` | Dataset identifier, hashes and official-method source lock |
| `EVIDENCE_MANIFEST.json` | SHA-256 of every released file and lists any privacy-scan exclusions |

## Reproducing the reported aggregate evidence (no private data required)

```bash
python -m pip install -r requirements.txt
python scripts/check_public_evidence.py
bash manuscript/build.sh      # requires LaTeX/IEEEtran and BibTeX
```

The independent verification script checks the four held-out rows for each of five methods, their unweighted AUROC/balanced accuracy means, and all 20 selected gene panels. The figures are vector PDFs; `manuscript/verification/redraw_wide_figures_v13.py` regenerates the main wide figures using the included aggregate CSV files. The manuscript PDF and source are synchronized to the v1.6 submission draft.

## Re-running the entire experiment (additional inputs required)

1. Obtain the published processed transcriptomic cohort matrices from the TENTACLES study and reproduce the accepted sample alignment. The accepted corrected input ZIP is **not** included here; supply it at `data/validated/ibd_dataset_sample_aligned.zip` with SHA-256 `8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74`.
2. Use `configs/dataset.json` to verify 689 patients and 9,983 shared genes. The original source RAR/RData used for row-ID reconstruction is also omitted; the source code and audit methods are included.
3. Install the Python dependencies. For author-native methods, obtain **StableMate** at commit `cf0fe9f344c1756b6f89fd33f9fdae1a7f35b24a` and **TENTACLES** at commit `069a26271e3885e015df25baa40a7a1140827d5f` in compatible R environments. These packages are third-party dependencies, not vendored copies.
4. Follow the notebooks in numbered order, along with `scripts/run_tentacles_cdpos_v2.sh`, `r/run_stablemate_stage8.R` and the exact protocols under `docs/`. Some notebooks are command entry points and were not executed in Drive; see `NOTEBOOK_STATUS.csv` rather than assuming every notebook has output.
5. Regenerate the final public tables from the saved models and data with the common-classifier and audit scripts. The full native model `.rds` archives and individual patient predictions remain in the private research workspace.

**Important limit:** This public ZIP preserves executed code and aggregate evidence, but cannot reproduce full training from a bare clone without the source datasets, native R package environments and expensive model objects. The final fixed-classifier and original native-model metrics should not be conflated. Full exact rerun was not performed as part of this GitHub export.

## Data, software, and provenance policy

- We publish aggregate performance, gene panels, scores, provenance and important run checks. We **exclude individual patient-level prediction rows**, full raw datasets, original RDS models, and course-provided PDF handouts.
- Original notebook outputs remain intact and unchanged. We never present a template with empty outputs as if it had completed successfully.
- Reactome pathways are assessed by hypergeometric over-representation against fold-eligible genes and Benjamini–Hochberg correction; pathway overlap between LOCO signatures is **descriptive**, not independent validation.
- The four analytical cohorts come from six BioProjects; the validation holds out cohorts, not necessarily single BioProjects. Seven D3 source records were retained despite inherited QC-exclusion flags, as explained in the manuscript.
- This is a research benchmark, **not a clinical diagnostic or clinically validated biomarker set**.

## Citation

Please cite the accompanying manuscript and the original algorithm/dataset publications listed in `manuscript/references.bib`. See `reports/final_manuscript/FINAL_SCIENTIFIC_AUDIT.md` for source verification and limitations.


## Post-review revised manuscript and evidence (23 November 2026)

The corrected experiments, source manuscript, aggregate reproducibility tables, reviewer-specific response letters and marked manuscripts are archived in [`post_review_2026-11-23/`](post_review_2026-11-23/README.md). This supplements rather than overwrites the original pre-review release. The course submission awaits author/coauthor sign-off.

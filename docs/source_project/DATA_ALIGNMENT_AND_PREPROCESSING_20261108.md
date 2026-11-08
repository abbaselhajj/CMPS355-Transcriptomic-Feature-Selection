# Original source-data identity and early preprocessing — executed

## Source identity and exact reconstruction

User-provided `data/source/ibd_datasets.rar` contains `count_d1.csv` to `count_d4.csv` and `desc_d1.csv` to `desc_d4.csv`. The count CSVs have **no sample ID column**. The user-supplied `data/source/ibd_datasets.RData` contains eight R data.frame objects; count-frame **row.names preserve sample IDs**, while metadata ID order differs. Our explicitly limited Python RDX3/XDR reader recovers original row names, verifies every RData count against every original CSV count, attaches IDs, then reorders expression rows by metadata IDs. Clinical labels and metadata were checked against the RData and corrected ZIP; no expression values changed.

Original RData SHA-256: `88e2f28715dfd460f68a756c8bcc541b72c299e6d0d67266febae215e99a0b0e`.
Canonical aligned dataset ZIP SHA-256: `8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74`.
Full, **semantic, cell-for-cell** reconstitution of the accepted dataset: **PASS** (689 samples; original gene columns D1–D3=10,061 and D4=41,257). ZIP byte identity is not expected because serialization metadata and CSV formatting may differ; the official ZIP is untouched.

| Cohort | Samples | Original count rows already matched by position | Rows corrected by ID |
|---|---:|---:|---:|
| D1 | 260 | 1 | 259 |
| D2 | 179 | 2 | 177 |
| D3 | 76 | 0 | 76 |
| D4 | 174 | 16 | 158 |
| **Total** | **689** | **19** | **670** |

Executed `notebooks/00a_rebuild_original_sample_alignment.ipynb` generates private `runs/source_alignment/<timestamp>/report.json`, `cohort_alignment_summary.csv` and a 689-row source-ID alignment audit. Its `reconstructed_counts_and_metadata.zip` is verified and discarded to avoid duplication. It does **not** overwrite the accepted dataset.

## Subsequent modeling preprocessing

The 4 count datasets share **9,983 gene names**, harmonized without using class labels. Stage 5's Python baseline and Stage 10's downstream logistic model use per-sample `log2(1 + CPM)`, where CPM is calculated using the **total count over those common genes** as denominator. This differs from the native TENTACLES `edgeR` preprocessing and should never be described as the authors' own count normalization. Stage 5 filters genes by **CPM > 1 in >=10% of outer-training patients, plus nonzero variance**, then fits `StandardScaler` on outer training only. Inner-cohort tuning re-fits the same preprocessing on inner training only.

| Held-out cohort | Three-cohort training | Stage 5 eligible genes |
|---|---:|---:|
| D1 | 429 | 8,337 |
| D2 | 510 | 8,406 |
| D3 | 613 | 8,409 |
| D4 | 515 | 8,438 |

Executed `notebooks/01a_preprocessing_and_expression_eda.ipynb` saves figures and tables under `runs/early_preprocessing/<timestamp>/`, with aggregated public results in `results/data_provenance/`. The existing `12_cohort_characterization.ipynb` retains PCA, class-balance and BioProject analysis. Neither figure set is a training-time fitted transformation.

**Scope:** The supplied data are gene counts **already produced** by the source study's RNA-seq pipeline. This work did not reprocess raw FASTQ reads or independently reconstruct fastp/STAR/Rsubread processing. Cohort, tissue, biological class and BioProject confounding remain important limitations. Source RAR/RData and all patient-ID maps remain **private**; they are excluded by `.gitignore`.

# TENTACLES Stage 10 — Independent Results Audit (2026-11-08)

## What was checked

- **170 independent deterministic assertions, 0 failures.** Checked uploaded archive content SHA-256 checksums; exact matching of native per-fold gene panels with Stage 10 selected-gene ordering; 689 unique held-out IDs, CD-positive encoding, class distribution, probability bounds, thresholding, confusion matrices, AUROC, average precision, accuracy, balanced accuracy, sensitivity, specificity, precision, F1, Brier score; 1,000-resample class-stratified bootstrap CIs; six pairwise gene Jaccard overlaps; gene-selection frequency; 14 all-four genes; macro means.
- **Not independently verified here:** native R model retraining, `runClassifiers` internals, contents of large `.rds` model objects, complete discovery/test input CSVs or their checksum against the original 13.7 MB corrected dataset. The completed native folds' checksum reports were previously marked PASS by the source run, but this compact archive holds only their exported text evidence.
- TENTACLES' cross-cohort downstream **predictor** is a fixed L2-penalised logistic regression (`C=1`, `class_weight=balanced`, `solver=liblinear`, threshold 0.5), trained on author-native TENTACLES consensus genes with outer training cohorts and training-only standardisation of log2(1+CPM). It is **not** TENTACLES' internal `testConsensus` prediction.

## Four held-out cohorts

| Held-out | N | CD | Not IBD | Consensus genes | AUROC | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| d1 | 260 | 218 | 42 | 92 | 0.8129 | 0.7224 |
| d2 | 179 | 134 | 45 | 114 | 0.6746 | 0.6469 |
| d3 | 76 | 34 | 42 | 98 | 0.6674 | 0.6331 |
| d4 | 174 | 109 | 65 | 99 | 0.6306 | 0.6017 |
| **Macro mean (unweighted)** | | | | | **0.6964** | **0.6510** |

## Gene stability

- Consensus panel sizes: D1 92, D2 114, D3 98, D4 99.
- Four-fold intersection: **14 genes** (CD55, CLDN18, CXCL6, DUOX2, FFAR2, HCAR2, LCN2, MMP1, NOS2, PAQR5, PLA2G3, PML, TMEM165, WNT5A).
- Mean pairwise Jaccard: **0.2486** (range 0.1369–0.3831).
- **These LOCO discovery sets overlap**, so Jaccard consistency cannot be called independent biological replication. No gene-level expression differential tests, batch effects, external patient validation or pathway enrichment results are inferred from these tables.

## Methodological warning gates

| Held-out | MLP too-many-weights mentions | LightGBM feature-fraction mentions | Boruta NA/NaN mentions |
|---|---:|---:|---:|
| d1 | 8 | 6 | 105 |
| d2 | 8 | 0 | 113 |
| d3 | 8 | 0 | 108 |
| d4 | 8 | 0 | 122 |

**PASS is execution/integrity, not proof that all native model components tuned or performed well.** The warning counts are mentions in logs, not deduplicated model failures. The voting ensemble's scientific quality must be reported as a limitation until each voter is audited from native fitted objects.

## Cross-method comparability

The existing Stage 5 LASSO and elastic-net benchmark, Stage 8 StableMate author-native evaluation, and TENTACLES Stage 10 all use four held-out cohorts, but have different preprocessing/tuning/prediction policies. Their performance tables are therefore **descriptive**, not a controlled causal comparison of feature-selection mechanisms.

StableMate and TENTACLES held-out predictions match the same 689 sample IDs and class labels. StableMate uses an author-native predictor, while the TENTACLES selected panels feed the fixed L2 logistic classifier. The observed four-fold macro AUROC difference TENTACLES minus StableMate is -0.0469; macro balanced-accuracy difference +0.0275. These numbers must not be described as one feature selector outperforming the other.

The optional paired bootstrap comparisons are conditional on the **saved predictions** and the **four observed cohorts**. They do not include fitting uncertainty or unobserved cohort heterogeneity, and no significance claim for generalisation to new cohorts is justified by the patient-level bootstrap alone.

## Required research follow-ups

1. Run the same frozen downstream predictor on all feature-selection panels if a controlled feature-selection comparison is a primary research objective. Existing Stage 5 and StableMate predictor performances should remain separately reported as native/adapted configurations.
2. Independently inspect TENTACLES native fitted RDS evidence and clarify per-voter contribution, model warnings, selected-gene votes and consensus exclusions. Do not assume the underlying R run errors were harmless.
3. Add explicit pipeline provenance and a release protocol before public repository publication; do not expose patient-level sample identifiers or training archives.
4. Do not present the 14 shared genes as validated clinical biomarkers; use exploratory biological interpretation with appropriate multiple-testing control and independent validation.

**Data sources:** immutable corrected four-cohort dataset SHA `8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74`; native TENTACLES pinned Git commit `069a26271e3885e015df25baa40a7a1140827d5f`; Stage 10 CSVs (2026-11-08); Stage 5 tables; Stage 8 StableMate fold CSVs.

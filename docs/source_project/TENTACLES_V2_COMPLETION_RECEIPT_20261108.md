# CMPS355 TENTACLES V2: four-cohort completion receipt and manuscript integration boundary

Date: 2026-11-08. **Source of numerical results:** user-supplied final WSL terminal log in the conversation. The WSL output archive has **not yet been uploaded to Drive**, and the summary below is not an independent re-audit of patient-level predictions or gene lists.

## Status

- The native official TENTACLES v0.1.0 package, pinned commit `069a26271e3885e015df25baa40a7a1140827d5f`, has four canonical V2 folds showing `COMPLETE` in the user's `scripts/run_tentacles_cdpos_v2.sh status` output.
- D3 required a provenance-checked recovery because native XGBoost variable importance was unavailable; the primary nine-voter consensus excludes XGBoost by design.
- D4 required a provenance-checked recovery because XGBoost selected tuning F1 was based on **4 rather than 5** resamples. All nine predeclared consensus voters had valid 5-fold selected CV F1, CD-positive apparent F1 readbacks, and feature importance. A final canonical view linked to the saved model was promoted. The original failed quality gate remains part of provenance; do not say all ten models passed 5-fold CV.
- Both V2 external LOCO evaluations completed (`V2_PRIMARY_6_VOTE_AND_SECONDARY_5_VOTE_EVALUATIONS_COMPLETE=TRUE`). Their downstream model is the same train-only-scaled balanced L2 logistic regression (C=1, liblinear) as the historical Stage22 fixed-classifier benchmark, **not native TENTACLES prediction**.

## Reported held-out metrics

| Holdout | N | Six-vote AUROC | Six-vote balanced accuracy | Five-vote AUROC | Five-vote balanced accuracy |
|---|---:|---:|---:|---:|---:|
| D1 | 260 | 0.8722 | 0.7911 | 0.8242 | 0.7677 |
| D2 | 179 | 0.6630 | 0.5845 | 0.6658 | 0.6208 |
| D3 | 76 | 0.6660 | 0.6275 | 0.6471 | 0.5945 |
| D4 | 174 | 0.6889 | 0.6172 | 0.6263 | 0.6030 |
| **Unweighted four-fold mean** | 689 total | **0.7225299559393727** | **0.6550576553804478** | **0.6908269926606468** | **0.6465195786829322** |

Important: The six-vote primary result is better on average than five-vote sensitivity, but **not in every cohort**; this is descriptive and not a statistical superiority result. The old TENTACLES V1 five-vote Stage22 outcomes are historical and must not be substituted for V2.

### Source directories recorded by local runner

- Six-vote (primary): `runs/tentacles_cd_positive_v2_external_nmin6/20261108T194642089312Z`
- Five-vote (sensitivity): `runs/tentacles_cd_positive_v2_external_nmin5/20261108T194659139917Z`
- Promoted D4 native: `runs/tentacles_cd_positive_v2/d4/20261108T194549418151545Z`
- Promoted D3 native: `runs/tentacles_cd_positive_v2/d3/20261108T174314673269857Z`

## What can be said to the manuscript author now

The fixed downstream learning and four-fold external evaluation for the corrected TENTACLES V2 panels has completed locally; update the Methods description accordingly. **Do not claim the new cross-method gene Jaccard, size-matched sensitivity, Reactome V2 pathways, or final five-way tradeoff figure have been recalculated yet.** Those require the full saved V2 panel tables. The previous fairness and Reactome extensions used historic V1 panels and are labelled provisional. The source zip must be independently cross-checked before publication-ready final tables.

## Export for independent audit

The helper `scripts/export_tentacles_v2_final.sh` and `scripts/export_tentacles_v2_final.py` and corresponding ZIP `docs/CMPS355_TENTACLES_V2_Final_Export_Helper.zip` are saved in Drive. Execute them under the WSL `CMPS355_TENTACLES_Local` project root and attach `CMPS355_TENTACLES_V2_Final_Results_For_Analysis.zip` to the ChatGPT conversation. It contains private patient-level held-out prediction rows, so retain it in **private** project artifacts, not a public Git repository. The helper does not alter any source or retrain native classifiers.

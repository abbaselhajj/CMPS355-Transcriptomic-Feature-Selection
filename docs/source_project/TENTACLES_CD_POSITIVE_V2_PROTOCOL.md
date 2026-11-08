# TENTACLES V2 CD-positive rerun: frozen protocol and gates

**Research project:** Benchmarking Transcriptomic Feature Selection for Cross-Cohort Predictive Performance and Biomarker Reproducibility.

## Research identity
- Official author package: `TENTACLES` 0.1.0 at Git SHA `069a26271e3885e015df25baa40a7a1140827d5f`, R 4.6.1.
- Corrected, sample-aligned TENTACLES CD RNA-seq data, 689 samples in four analytical cohorts; 9,983 shared genes. Accepted ZIP SHA-256 `8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74`.
- Four outer leave-one-cohort-out test cohorts (D1 260, D2 179, D3 76, D4 174); discovery cohort counts respectively 429, 510, 613, 515. No heldout data are sent to native preprocessing or `runClassifiers()`.
- Important limitation: harmonisation to shared 9,983 gene symbols uses unlabeled gene presence across cohorts; official preProcess is fit to training cohorts. Cross-study batch effects remain a limitation; no posthoc test-derived batch correction.

## Corrected event-level handling
1. On reading clinical rows, construct `factor(class, levels=c('CD','Not IBD'))`.
2. Run the official authors' `preProcess(...)` using the same parameters as V1, then explicitly restore the `CD`-first outcome factor in its internal S4 data containers, including `@metadata$class`. This prevents the authors' preprocessing from silently resetting factor order.
3. Check expected `yardstick::f_meas_vec()` event-first semantics with a synthetic example (expected F1 0.8); do not use deprecated global `yardstick.event_first`.
4. **Hard native smoke gate:** fit the official `decision_tree/base` on 24 abundant discovery-only genes, with 2 tuning candidates/2 CV folds, then use the saved development predictions and the saved native apparent metrics to prove that the model's reported F1 matches **CD-positive** F1 and not `Not IBD`-positive F1. If ambiguous or wrong, fail and inspect; do not launch 10-model fitting.
5. In the full model, apply the same F1-orientation check separately to all ten classifiers. Inspect selected native tuning summaries, five recorded resamples and feature-importance tables. Native warnings remain logged; an apparent passing model is not proof that every tuning candidate succeeded.

## Preserved official-native implementation
Ten classifier/selector pairs: rand_forest/corr; mlp/roc; bag_mlp/roc; svm_linear/boruta; C5_rules/boruta; lightGBM/roc; decision_tree/boruta; mars/boruta; xgboost/base; pls/boruta. Native arguments: `tuning.method='tune_grid'`, `n=5`, `v=5`, `metric='f_meas'`, `boruta.maxRuns=100`, `selector.threshold=0.95`, `nsim=2`, `filter=TRUE`, `seed=123`, `downsample=TRUE`, `plot=FALSE`, `parallel=FALSE`. Preprocessing: authors' RNA-seq path, CPM threshold 1, min fraction 0.1, no cross-cohort batch adjustment. Any choice not identical to the paper's original study remains an explicitly labelled **LOCO benchmark adaptation**.

## Gene consensus
Primary paper-aligned `getConsensus(fit, n.min=6, exclude='xgboost_base',plot=FALSE)` (**six votes from nine**). Separately record 5/9 as a sensitivity result; never re-label a posthoc winning setting as an a priori result. Actual excluded model name is verified from native model list. Independent vote sums must reproduce each gene panel. Preserve complete native S4 fit and both consensus objects as RDS, plus original gene identifiers, votes, hashes, package versions, logs and native warnings.

## Outer performance
Use fixed `C=1`, balanced L2 logistic regression (`solver=liblinear`), trained on all three discovery cohorts and a scaler fitted **only** to that discovery set. Input is `log2(1+CPM)` with full 9,983-gene library-size denominator. Apply frozen scaler and classifier to outer heldout cohort once; 0.5 cutoff; `CD` is the positive class. Save AUROC, AUPRC, F1, balanced accuracy, sensitivity, specificity, Brier, counts, 95% bootstrap CIs; private per-patient predictions. This **does not** reproduce the authors' `testConsensus` internal MLP performance. No comparisons of Not-IBD native CV F1 with CD-positive outer F1.

## Avoid overwrites and research overclaim
- Never touch `runs/tentacles_stage9_loco`, `runs/tentacles` or existing `runs/tentacles_stage10_post`.
- Never accept an existing V2 fold without its complete pass receipts.
- Do not treat input-validation success or aggregate CV success as evidence that all warning-prone tuning candidates were successful.
- Two risks remain: factor/order changes within package internals can fail the smoke; the pinned official package may retain problematic candidate warnings. A quality-gate failure must be investigated, not suppressed.

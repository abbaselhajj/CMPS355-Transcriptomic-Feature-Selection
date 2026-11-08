# Stage 22 — matched downstream classifier on five outer-training-selected panels

## Why this secondary experiment exists

The original native-workflow results are **not a controlled comparison of feature selectors**: StableMate gives its own author-native predictions, Stage 5 LASSO/elastic net use method-specific tuned logistic models, while Stage 10 TENTACLES used a fixed L2 logistic downstream predictor. The latter was a valid **external generalization** endpoint, not the author's within-dataset `testConsensus()` MLP.

To provide an improved cross-method comparison, we froze the outer-training-selected gene panels and then fitted the **same** model to each panel within each held-out cohort. No native fit or old patient-level results were overwritten. This changes the estimand to **external predictive utility of a selected gene set under a fixed downstream learner**.

## Input panel identity

- LASSO and elastic net: every nonzero gene from outer-training-fitted tuned sparse logistic models; `runs/full_gene_results_report/20261107T091750523128Z/table_selected_genes.csv`.
- Stability Selection: outer **training** subsample selection, fixed frequency cutoff **0.60**, from `runs/stability_selection/d*/loco_training/threshold_signatures.csv`.
- StableMate: author-native **significantly stable predictive** gene lists from the four K=2000/B=1000 runs, not the author-native fitted predictors.
- TENTACLES: native official `getConsensus()` panels (5/9 votes, excluding `xgboost_base`), from `runs/tentacles/summary/selected_genes.csv`.

## Frozen common evaluation

Identical four outer leave-one-cohort-out splits (D1–D4). For every method/fold: `log2(1+1e6*count / patient_total_on_shared_9983_genes)`, per-gene `StandardScaler` fitted on **outer training only**, sklearn balanced logistic regression (`liblinear`, L2, `C=1`, `max_iter=5000`, seed=123), and **CD-positive 0.5 threshold**. No held-out label was used for feature selection, normalization fit, scaling or tuning. Signature sizes were intentionally **not** equalized, to evaluate each method's own chosen panel.

Outputs are **20 model evaluations** and 5×689 patient-level held-out predictions. Private outputs: `runs/common_classifier/<timestamp>/` with input-file hashes and environment versions. Public-safe: `results/common_classifier/` and three plots in `results/figures/`. Unweighted mean across the **four held-out cohorts**, not across 20 correlated fits.

| Panel | Mean AUROC | Mean balanced accuracy | Mean panel size |
|---|---:|---:|---:|
| LASSO | 0.6891 | 0.6190 | 281.5 |
| Elastic net | 0.7118 | 0.6281 | 352.5 |
| Stability Selection 0.60 | 0.6284 | 0.5628 | 6.3 |
| StableMate | 0.6590 | 0.5979 | 15.8 |
| TENTACLES | 0.6964 | 0.6510 | 100.8 |

**Important:** the new TENTACLES refit reproduces original Stage 10 AUROC almost exactly, with maximum per-patient probability discrepancy under **0.001** across folds (largest in D2 ~0.000806), consistent with small solver/implementation numerical differences. The original completed Stage 10 predictions are immutable; do **not** use the refit as evidence of byte-identical model reproduction. The native-workflow comparisons remain distinct.

## Limitations

The feature selectors are heterogeneous in training objectives, hyperparameters, resampling and threshold criteria; keeping an identical predictor controls the **downstream classifier, normalization and train/test split** only. The four LOCO training sets overlap and are based on related studies; there is no n=20 independent replication. The source paper's original 28-gene signature was derived using a different sequential discovery/refinement protocol, and Stage 22 is **not a reproduction of those original evaluation numbers**. No patient-identifying outputs should be published without an appropriate dataset-use review.

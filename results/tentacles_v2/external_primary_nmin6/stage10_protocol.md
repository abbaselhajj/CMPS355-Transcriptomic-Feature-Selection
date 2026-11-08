# Stage 10 — External held-out prediction on TENTACLES native consensus panels

- Input: four official training-only TENTACLES consensus gene lists (6 of 9 voters; XGBoost excluded), selected without any held-out label access.
- Outer training: all other three cohorts; external test: untouched heldout cohort.
- Common downstream classifier, **not an author-native TENTACLES predictor**: scikit-learn logistic regression, L2 `C=1`, `class_weight=balanced`, `solver=liblinear`, threshold=0.5, seed=123, no test-based tuning.
- Feature transform: log2(1 + one million × raw gene count / raw sample total over all 9,983 shared genes), with fit of StandardScaler on the discovery cohorts only; apply frozen scaler to heldout cohort.
- All models trained on raw expression using only genes from the corresponding native consensus CSV. No batches corrected, no test-sample feature selection. Other feature-selection methods must receive the same downstream predictor for a controlled feature-selection comparison.
- AUROC and balanced-accuracy 95% percentile CIs from 1,000 stratified-within-class bootstrap replicates of each outer test cohort, preserving class counts.
- Model warnings from native TENTACLES fitting require independent review and are not erased by successful Stage10 predictions.
- Consensus overlap across four LOCO panels is a leave-one-cohort-out sensitivity analysis, not independent biomarker replication: the discovery sets overlap in two cohorts.
- The corrected aligned source dataset used here is also from the TENTACLES paper. This is an adaptation/benchmark, not independent disease-cohort discovery.
- Test patient IDs remain local/private. Do not upload raw prediction tables to a public repository.

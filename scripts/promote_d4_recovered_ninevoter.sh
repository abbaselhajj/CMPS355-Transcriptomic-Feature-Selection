#!/usr/bin/env bash
# Promote a verified D4 nine-voter consensus to an additive, provenance-linked
# canonical *view* accepted by the existing read-only Stage 10 V2 evaluator.
# Nothing under the failed native fit or recovery directories is changed.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE="$ROOT/runs/tentacles_cd_positive_v2/d4"
RECBASE="$ROOT/runs/tentacles_cd_positive_v2_recovery/d4"
SRC="$BASE/20261108T174423655704315Z"
[[ -d "$SRC" && -d "$RECBASE" ]] || { echo 'STOP: exact original D4 or recovery directory missing' >&2; exit 2; }
exec 9>"$ROOT/runs/tentacles_cd_positive_v2/.v2_single_process.lock"
flock -n 9 || { echo 'STOP: another TENTACLES V2 run is active' >&2; exit 2; }
[[ -s "$SRC/runclassifiers_finished.rds" && -s "$SRC/STATUS.txt" && -s "$SRC/classifiers.log" && -s "$SRC/input_and_code_sha256.txt" ]] || { echo 'STOP: original D4 evidence incomplete' >&2; exit 2; }
[[ "$(cat "$SRC/STATUS.txt")" != 'PASS_CD_POSITIVE_NATIVE_NMIN6_AND_NMIN5' ]] || { echo 'STOP: original D4 unexpectedly already marked complete' >&2; exit 2; }
grep -Fq 'Native tuning F1 missing/invalid/incomplete' "$SRC/classifiers.log" || { echo 'STOP: wrong original D4 failure mode' >&2; exit 2; }
[[ ! -e "$SRC/consensus_native_finished.rds" ]] || { echo 'STOP: original failed D4 already contains a consensus; inspect manually' >&2; exit 2; }
# Prevent accidental second promotion or selecting another incomplete latest run.
for d in "$BASE"/20*Z; do
  [[ -d "$d" ]] || continue
  [[ "$d" == "$SRC" ]] || { echo "STOP: additional D4 native/canonical run exists: $d; inspect status before promotion" >&2; exit 2; }
done
mapfile -t recoveries < <(find "$RECBASE" -mindepth 1 -maxdepth 1 -type d -name '20*Z' | LC_ALL=C sort)
(( ${#recoveries[@]} > 0 )) || { echo 'STOP: no recovery folder' >&2; exit 2; }
REC="${recoveries[${#recoveries[@]}-1]}"
[[ -s "$REC/RECOVERY_STATUS.txt" && -s "$REC/recovery.log" ]] || { echo 'STOP: latest recovery not complete' >&2; exit 2; }
for flag in \
  RECOVERY_PASS_OFFICIAL_NINE_VOTER_CONSENSUS \
  ORIGINAL_FIT_UNMODIFIED=TRUE \
  NINE_VOTERS_CD_POSITIVE_F1_VALID=TRUE \
  NINE_VOTERS_SELECTED_FIVEFOLD_CV_VALID=TRUE \
  NINE_VOTERS_FEATURE_IMPORTANCE_VALID=TRUE \
  NMIN6_INDEPENDENT_VOTE_CHECK=TRUE \
  NMIN5_INDEPENDENT_VOTE_CHECK=TRUE \
  XGBOOST_EXCLUDED_PREDEFINED=TRUE \
  XGBOOST_SELECTED_CV_N=4_NOT_ACCEPTED_AS_FIVEFOLD=TRUE \
  TEN_MODEL_NATIVE_GATE_ORIGINALLY_FAILED=TRUE; do
  grep -Fxq "$flag" "$REC/RECOVERY_STATUS.txt" || { echo "STOP: missing recovery gate $flag" >&2; exit 2; }
done
for name in \
  source_rds_sha256_BEFORE.txt source_rds_sha256_AFTER.txt \
  consensus_native_finished.rds consensus_nmin5_finished.rds \
  consensus_genes_native.csv consensus_genes_nmin6.csv consensus_genes_nmin5.csv \
  native_votes_nmin6.csv native_votes_nmin5.csv \
  native_cv_f1_by_model.csv native_f1_orientation_readback.csv \
  native_feature_importance_readback.csv native_all_tuning_metrics.csv \
  native_all_f1_rows.csv recovery.log; do
  [[ -s "$REC/$name" ]] || { echo "STOP: missing recovery evidence $name" >&2; exit 2; }
done
cmp -s "$REC/source_rds_sha256_BEFORE.txt" "$REC/source_rds_sha256_AFTER.txt" || { echo 'STOP: fitted source hash changed during recovery' >&2; exit 2; }
sha256sum --status -c "$REC/source_rds_sha256_BEFORE.txt" || { echo 'STOP: saved fitted object hash no longer matches recovery proof' >&2; exit 2; }
sha256sum --status -c "$SRC/input_and_code_sha256.txt" || { echo 'STOP: native code/inputs checksum mismatch' >&2; exit 2; }
[[ "$(sed -n 's/^ORIGINAL_D4=//p' "$REC/recovery.log" | head -n1)" == "$SRC" ]] || { echo 'STOP: recovery points to different D4 source' >&2; exit 2; }
# Independent, small-CSV quality/readback checks; NEVER reopen heavy native RDS.
python3 - "$REC" <<'PY'
from pathlib import Path
import csv,math,sys
r=Path(sys.argv[1])
models={
'rand_forest_corr','mlp_roc','bag_mlp_roc','svm_linear_boruta',
'C5_rules_boruta','lightGBM_roc','decision_tree_boruta','mars_boruta','pls_boruta'}
expected=models|{'xgboost_base'}
def table(filename):
 with (r/filename).open(encoding='utf-8-sig',newline='') as f:
  out=list(csv.DictReader(f))
 if not out: raise ValueError(f'{filename}: empty table')
 return out
def by_model(filename):
 rows=table(filename)
 actual=[x['model'] for x in rows]
 assert len(rows)==10 and set(actual)==expected and len(set(actual))==10, (filename,actual)
 return {x['model']:x for x in rows}
cv=by_model('native_cv_f1_by_model.csv')
fm=by_model('native_f1_orientation_readback.csv')
fi=by_model('native_feature_importance_readback.csv')
for m in expected:
 c=cv[m]
 assert int(c['rows'])==1 and int(c['n_configurations'])==1, (m,'duplicate config')
 assert math.isfinite(float(c['cv_f1'])), (m,'missing CV F1')
 assert c['valid_single_selected_f1']==('FALSE' if m=='xgboost_base' else 'TRUE'), (m,'invalid CV gate')
 assert int(c['cv_folds'])==(4 if m=='xgboost_base' else 5), (m,'CV fold count')
 a=fm[m]
 assert a['training_rows']=='515' and a['truth_cd_count']=='386' and a['truth_control_count']=='129', (m,'training-label alignment')
 assert a['truth_encoding']=='NATIVE_1_CD_0_NOT_IBD' and a['prediction_encoding']=='NATIVE_1_CD_0_NOT_IBD', (m,'event encoding')
 assert a['truth_raw_values']=='0|1' and a['predictions_raw_values']=='0|1', (m,'prediction encoding')
 assert a['cd_positive_identified']=='TRUE' and a['reported_matches_cd']=='TRUE' and a['reported_matches_not_ibd']=='FALSE' and a['failure_reason']=='PASS', (m,'CD-positive F1 gate')
 assert abs(float(a['reported_f1'])-float(a['independently_cd_f1']))<1e-8, (m,'F1 numerical mismatch')
 assert abs(float(a['reported_f1'])-float(a['not_ibd_f1']))>1e-8, (m,'ambiguous event')
 a=fi[m]
 assert a['feature_table_valid']=='TRUE' and int(a['n_features'])>=1 and int(a['n_nonzero'])>=1, (m,'feature table')
def genes(filename):
 rows=table(filename)
 assert set(rows[0])=={'gene'} and all(set(row)=={'gene'} for row in rows), filename
 z=[x['gene'] for x in rows]
 assert z and len(z)==len(set(z)) and all(x and x.strip()==x for x in z), filename
 return set(z)
g6=genes('consensus_genes_nmin6.csv')
g5=genes('consensus_genes_nmin5.csv')
assert g6==genes('consensus_genes_native.csv') and g6<=g5,'six-vote panel identity/nesting'
assert len(g5)==102,('unexpected reported 5-vote count',len(g5))
for k,g in [(6,g6),(5,g5)]:
 rows=table(f'native_votes_nmin{k}.csv')
 assert set(rows[0])=={'gene','votes'} and all(set(row)=={'gene','votes'} for row in rows)
 names=[x['gene'] for x in rows]
 assert len(names)==len(set(names)) and all(names),'duplicate native vote genes'
 votes={x['gene']:int(x['votes']) for x in rows}
 assert all(0<=v<=9 for v in votes.values()),'bad vote range'
 assert {x for x,v in votes.items() if v>=k}==g,('vote recomputation mismatch',k)
print('D4_RECOVERY_CSV_REVERIFICATION_PASS=TRUE')
print(f'D4_NMIN6_GENES={len(g6)};D4_NMIN5_GENES={len(g5)};NATIVE_CONSENSUS_VOTERS=9')
print('D4_XGBOOST_FOURFOLD_CV_QUALIFIED_EXCLUDED=TRUE')
PY
# Publish a new *view* of the original and recovered files; no original file changes.
stamp="$(date -u +%Y%m%dT%H%M%S%N)Z"
DEST="$BASE/$stamp"
[[ ! -e "$DEST" ]] || { echo 'STOP: timestamp collision' >&2; exit 2; }
STAGE="$(mktemp -d "$BASE/.promotion_XXXXXXXX")"
cleanup(){ if [[ -n "${STAGE:-}" && -d "$STAGE" ]]; then rm -rf -- "$STAGE"; fi; }
trap cleanup EXIT
orig=(runclassifiers_finished.rds classifiers.log input_and_code_sha256.txt
      protocol_cd_positive_v2.txt model_selector_pairs.csv
      train_only_preprocessed.rds native_preprocess_structure_BEFORE_relevel.txt
      preprocess_event_level_audit.txt tentacles_cdpos_v2_fit.R
      tentacles_cdpos_v2_external.py warning_summary.json
      runclassifiers_structure.txt r_session_info.txt
      installed_R_package_versions.csv native_feature_importance_quality.csv
      model_features_native.rds performances_native.rds)
for name in "${orig[@]}"; do
  if [[ -s "$SRC/$name" ]]; then ln -s --relative "$SRC/$name" "$STAGE/$name"; fi
done
recovered=(consensus_native_finished.rds consensus_nmin5_finished.rds
           consensus_genes_native.csv consensus_genes_nmin6.csv consensus_genes_nmin5.csv
           native_votes_nmin6.csv native_votes_nmin5.csv
           native_cv_f1_by_model.csv native_f1_orientation_readback.csv
           native_feature_importance_readback.csv native_all_tuning_metrics.csv
           native_all_f1_rows.csv RECOVERY_STATUS.txt recovery.log
           source_rds_sha256_BEFORE.txt source_rds_sha256_AFTER.txt)
for name in "${recovered[@]}"; do ln -s --relative "$REC/$name" "$STAGE/$name"; done
for name in runclassifiers_finished.rds consensus_native_finished.rds consensus_nmin5_finished.rds \
            consensus_genes_native.csv consensus_genes_nmin5.csv classifiers.log input_and_code_sha256.txt; do
  [[ -s "$STAGE/$name" ]] || { echo "STOP: canonical view missing $name" >&2; exit 2; }
done
cat > "$STAGE/RECOVERY_PROVENANCE.txt" <<EOF
D4 canonical nine-voter consensus view; NOT a second native fit.
Original D4 fitted model and originally FAILED ten-model CV gate: $SRC
Original D4 run and its failed status are preserved unchanged.
Verified passed native nine-voter recovery (official getConsensus 6/5): $REC
All source and output RDS objects are symlinks to immutable saved objects; not copies.
Original failure was XGBoost having n=4, not n=5, selected CV F1 resamples.
XGBoost was excluded from consensus BEFORE training by the study protocol.
The nine actual voters have finite selected five-fold CV F1, independently
verified CD-positive apparent fitted F1, and nonempty valid importance tables.
Consensus result was independently checked from native nine-row gene-vote matrices.
Passing status means recoverable nine-voter consensus for Stage10 external evaluation,
not that XGBoost attained five CV folds or that Boruta warnings were resolved.
Future readers MUST inspect the original failed run and recovery diagnostic tables.
EOF
cat > "$STAGE/consensus_protocol.txt" <<EOF
ORIGINAL_TENTACLES_COMMIT=069a26271e3885e015df25baa40a7a1140827d5f
HELDOUT_LABEL=CD
EVENT=CD_FIRST
NATIVE_TUNING_METRIC=f_meas
PRIMARY_CONSENSUS_N_MIN=6
SENSITIVITY_N_MIN=5
EXCLUDED_FROM_9_VOTERS=xgboost_base
NATIVE_XGBOOST_CV_FOLDS=4
CONSENSUS_SOURCE=AUTHOR_TENTACLES_GETCONSENSUS_UNMODIFIED_SAVED_FIT
RECOVERY_NINE_VOTER_VERIFIED=TRUE
ORIGINAL_TEN_MODEL_QUALITY_GATE=FAILED
EOF
for name in runclassifiers_finished.rds consensus_native_finished.rds consensus_nmin5_finished.rds train_only_preprocessed.rds; do
  [[ -s "$STAGE/$name" ]] || continue
  printf '%s  %s/%s\n' "$(sha256sum "$STAGE/$name" | awk '{print $1}')" "$DEST" "$name" >> "$STAGE/output_rds_sha256.txt"
done
[[ -s "$STAGE/output_rds_sha256.txt" ]] || { echo 'STOP: output manifest empty'; exit 2; }
printf 'PASS_CD_POSITIVE_NATIVE_NMIN6_AND_NMIN5\n' > "$STAGE/STATUS.txt"
mv -- "$STAGE" "$DEST"
STAGE=''
sha256sum --status -c "$DEST/output_rds_sha256.txt" || { echo 'STOP: promoted output hashes fail, inspect status' >&2; exit 3; }
[[ "$(cat "$DEST/STATUS.txt")" == 'PASS_CD_POSITIVE_NATIVE_NMIN6_AND_NMIN5' ]] || { echo 'STOP: canonical view status mismatch' >&2; exit 3; }
echo 'D4_RECOVERY_PROMOTED_WITH_SOURCE_PROVENANCE=TRUE'
echo "D4_PROMOTED_RUN=$DEST"
echo 'D4_ORIGINAL_FAILED_GATE_PRESERVED=TRUE'
echo 'D4_NATIVE_FITTED_MODEL_REUSED=TRUE'
echo 'NEXT: bash scripts/run_tentacles_cdpos_v2.sh status'

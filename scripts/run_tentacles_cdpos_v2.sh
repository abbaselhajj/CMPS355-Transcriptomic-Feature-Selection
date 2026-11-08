#!/usr/bin/env bash
# Versioned, fail-closed sequential four-fold TENTACLES rerun. Never touches V1 runs.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT/opt/R-4.6.1/bin/Rscript" ]]; then
  R="$ROOT/opt/R-4.6.1/bin/Rscript"
elif command -v Rscript >/dev/null 2>&1; then
  R="$(command -v Rscript)"
else
  echo 'ERROR: Rscript not found (expected R 4.6.1)'; exit 2
fi
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo 'ERROR: Python 3 not found'; exit 2
fi
SHA="069a26271e3885e015df25baa40a7a1140827d5f"
BASE="$ROOT/runs/tentacles_cd_positive_v2"
PRE="$ROOT/runs/tentacles_cd_positive_v2_preflight"
FIT="$ROOT/r/tentacles_cdpos_v2_fit.R"
CONS="$ROOT/r/tentacles_cdpos_v2_consensus.R"
EVAL="$ROOT/src/cmps355/tentacles_cdpos_v2_external.py"
MODE="${1:-preflight}"
cd "$ROOT"
if [[ "$MODE" != 'preflight' && "$MODE" != 'all' && "$MODE" != 'fold' && "$MODE" != 'evaluate' && "$MODE" != 'status' ]]; then
 echo 'Usage: bash scripts/run_tentacles_cdpos_v2.sh preflight|all|fold d1|evaluate|status' >&2; exit 2
fi
[[ -x "$R" && -x "$PY" ]] || { echo 'ERROR: Rscript or Python interpreter unavailable' >&2; exit 2; }
[[ -f "$FIT" && -f "$CONS" && -f "$EVAL" ]] || { echo 'ERROR: patch files missing; extract ZIP at project root' >&2; exit 2; }
export R_LIBS="$ROOT/r_lib" R_LIBS_USER="$ROOT/r_lib" CMPS355_R_LIB="$ROOT/r_lib"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$BASE" "$PRE"
exec 9>"$BASE/.v2_single_process.lock"
flock -n 9 || { echo 'ERROR: another V2 run is already active; do not run folds in parallel' >&2; exit 2; }
if [[ "$MODE" != 'status' ]]; then
  "$R" --vanilla -e 'p<-Sys.getenv("CMPS355_R_LIB");.libPaths(c(p,.libPaths()));stopifnot(as.character(getRversion())=="4.6.1");req<-c("TENTACLES","tidymodels","yardstick","bonsai","lightgbm","earth","xgboost","mixOmics","plsmod");stopifnot(all(vapply(req,requireNamespace,logical(1L),quietly=TRUE)));d<-utils::packageDescription("TENTACLES");stopifnot(tolower(d[["RemoteSha"]])=="069a26271e3885e015df25baa40a7a1140827d5f");cat("PINNED_NATIVE_DEPENDENCIES_PASS=TRUE\n")'
  [[ -f "$ROOT/data/validated/ibd_dataset_sample_aligned.zip" ]] || { echo 'Missing canonical dataset ZIP' >&2; exit 2; }
  expected='8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74'
  [[ "$(sha256sum "$ROOT/data/validated/ibd_dataset_sample_aligned.zip" | cut -d' ' -f1)" == "$expected" ]] || { echo 'Canonical dataset hash mismatch' >&2; exit 2; }
  "$PY" "$ROOT/src/cmps355/tentacles_stage9_prepare_loco.py" "$ROOT"
  if [[ "$MODE" == 'all' || "$MODE" == 'fold' ]]; then
    free_kb=$(df -Pk "$ROOT" | awk 'END {print $4}')
    if [[ -z "$free_kb" || "$free_kb" -lt 10485760 ]]; then
      echo 'STOP: Need at least 10 GiB free to start new native fits safely; old 6.5 GB R models are preserved.' >&2
      df -h "$ROOT";exit 2
    fi
    df -h "$ROOT"
  fi
fi
stamp(){ date -u +%Y%m%dT%H%M%S%N; }
latest_run(){
 local fold="$1" base="$BASE/$1"
 [[ -d "$base" ]] || return 0
 find "$base" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort | tail -n 1 | while IFS= read -r last;do [[ -n "$last" ]] && echo "$base/$last";done
}
status_ok(){
 local d="$1"
 [[ -d "$d" && -s "$d/STATUS.txt" && -s "$d/runclassifiers_finished.rds" && -s "$d/consensus_native_finished.rds" && -s "$d/consensus_genes_native.csv" && -s "$d/consensus_genes_nmin5.csv" ]] || return 1
 [[ "$(cat "$d/STATUS.txt")" == 'PASS_CD_POSITIVE_NATIVE_NMIN6_AND_NMIN5' ]]
}
run_smoke(){
 local fold="$1" data="$ROOT/runs/tentacles_stage9_inputs/$1/discovery"
 local d="$PRE/$fold/$(stamp)Z"; mkdir -p "$d"
 printf 'SMOKE_RUNNING\n' > "$d/STATUS.txt"
 echo "SMOKE_START=$fold OUT=$d"
 set +e
 "$R" --vanilla "$FIT" smoke "$data/counts.csv" "$data/clinical.csv" "$d" "$fold" "$SHA" 2>&1 | tee "$d/SMOKE.log"
 local rc=${PIPESTATUS[0]}
 set -e
 if ((rc != 0)) || [[ ! -s "$d/SMOKE_PASS.txt" ]] || ! grep -q '^SMOKE_PASS_CD_POSITIVE_F1=TRUE$' "$d/SMOKE.log"; then
   printf 'FAILED_SMOKE exit=%d\n' "$rc" > "$d/STATUS.txt"
   echo "SMOKE_FAILED=$fold REPORT=$d/SMOKE.log" >&2
   tail -n 45 "$d/SMOKE.log" >&2 || true
   return 1
 fi
 printf 'PASS_SMOKE_CD_POSITIVE_F1\n' > "$d/STATUS.txt"
 echo "SMOKE_PASS=$fold REPORT=$d"
}
preflight_all(){ for fold in d1 d2 d3 d4; do run_smoke "$fold" || { echo 'STOP: full native training not started because preflight failed'; return 1; }; done; echo 'ALL_FOUR_CDPOS_SMOKE_GATES_PASS=TRUE'; }
run_fold(){
 local fold="$1" data="$ROOT/runs/tentacles_stage9_inputs/$1/discovery"
 local d
 d="$(latest_run "$fold")"
 if [[ -n "$d" ]]; then
   if status_ok "$d"; then echo "REUSING_COMPLETE_V2_$fold=$d"; return 0; fi
   echo "STOP: Found existing incomplete V2 $fold run: $d. No automatic overwrite/retry." >&2
   echo 'Please inspect its logs and saved RDS. V1 is untouched.' >&2
   return 3
 fi
 d="$BASE/$fold/$(stamp)Z"; mkdir -p "$d"
 cp "$FIT" "$CONS" "$EVAL" "$d/"
 sha256sum "$d/tentacles_cdpos_v2_fit.R" "$d/tentacles_cdpos_v2_consensus.R" "$d/tentacles_cdpos_v2_external.py" \
  "$ROOT/data/validated/ibd_dataset_sample_aligned.zip" "$data/counts.csv" "$data/clinical.csv" > "$d/input_and_code_sha256.txt"
 printf 'RUNNING_CLASSIFIERS\n' > "$d/STATUS.txt"
 echo "START_FULL_NATIVE_$fold=$d"
 echo "NATIVE_BINARY=$R; PYTHON_BINARY=$PY"
 set +e
 "$R" --vanilla "$FIT" fit "$data/counts.csv" "$data/clinical.csv" "$d" "$fold" "$SHA" 2>&1 | tee "$d/classifiers.log"
 local rc=${PIPESTATUS[0]}
 set -e
 "$PY" "$ROOT/scripts/summarize_tentacles_cdpos_v2_warnings.py" "$d" || true
 if ((rc!=0)) || [[ ! -s "$d/runclassifiers_finished.rds" ]] || ! grep -q '^CLASSIFIERS_PASS_CD_POSITIVE=TRUE$' "$d/classifiers.log"; then
   printf 'FAILED_CLASSIFIERS_OR_CDPOS_QUALITY_GATE exit=%d\n' "$rc" > "$d/STATUS.txt"
   echo "STOP: $fold failed fit/quality; preserved evidence at $d" >&2; return 4
 fi
 printf 'PASS_CDPOS_CLASSIFIERS_CONSENSUS_PENDING\n' > "$d/STATUS.txt"
 set +e
 "$R" --vanilla "$CONS" "$d/runclassifiers_finished.rds" "$d" 2>&1 | tee "$d/consensus.log"
 local crc=${PIPESTATUS[0]}
 set -e
 if ((crc!=0)) || [[ ! -s "$d/consensus_native_finished.rds" ]] || [[ ! -s "$d/consensus_nmin5_finished.rds" ]] || \
    ! grep -q '^OFFICIAL_CONSENSUS_6_AND_5_PASS=TRUE$' "$d/consensus.log"; then
    printf 'PASS_CLASSIFIERS_BUT_FAILED_CONSENSUS exit=%d\n' "$crc" > "$d/STATUS.txt"
    echo "STOP: $fold classifiers remain saved; recover consensus only. Evidence: $d" >&2;return 4
 fi
 sha256sum "$d"/*.rds > "$d/output_rds_sha256.txt"
 printf 'PASS_CD_POSITIVE_NATIVE_NMIN6_AND_NMIN5\n' > "$d/STATUS.txt"
 echo "COMPLETE_V2_$fold=$d"
}
evaluate(){
 for fold in d1 d2 d3 d4;do local d;d="$(latest_run "$fold")"; status_ok "$d" || { echo "Cannot evaluate: incomplete $fold" >&2; return 3; }; done
 "$PY" "$EVAL" "$ROOT" 6
 "$PY" "$EVAL" "$ROOT" 5
 echo 'V2_PRIMARY_6_VOTE_AND_SECONDARY_5_VOTE_EVALUATIONS_COMPLETE=TRUE'
}
case "$MODE" in
 preflight) preflight_all;;
 fold) f="${2:-}"; [[ "$f" =~ ^d[1-4]$ ]] || { echo 'Expected fold d1..d4'; exit 2; };run_smoke "$f" && run_fold "$f";;
 all) preflight_all && for fold in d1 d2 d3 d4;do run_fold "$fold" || exit $?;done;evaluate;;
 evaluate) evaluate;;
 status) for fold in d1 d2 d3 d4;do d="$(latest_run "$fold")";if [[ -z "$d" ]];then echo "$fold NOT_STARTED";elif status_ok "$d";then echo "$fold COMPLETE $d";else echo "$fold NEEDS_INSPECTION $d";fi;done;;
esac

#!/usr/bin/env bash
# One complete native 10-model classifier fit; results never overwritten.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOLD="${1:-}"
[[ "$FOLD" =~ ^d[1-4]$ ]] || { echo 'Usage: bash scripts/run_fold.sh d1|d2|d3|d4'; exit 2; }
R="$ROOT/opt/R-4.6.1/bin/Rscript"
PY="$ROOT/.venv/bin/python"
[[ -x "$R" && -x "$PY" ]] || { echo 'First run prepare_workspace.sh, install_r461_isolated.sh, restore_cache.sh, install_dependencies.sh'; exit 2; }
if pgrep -fa '[R]script.*stablemate_stage8_traced_final.R' >/dev/null; then
  echo 'A StableMate Rscript is still running. Stage9 full fitting is blocked to prevent memory contention.'
  exit 2
fi
cd "$ROOT"
export R_LIBS="$ROOT/r_lib" R_LIBS_USER="$ROOT/r_lib" CMPS355_R_LIB="$ROOT/r_lib"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$R" --vanilla -e 'p<-Sys.getenv("CMPS355_R_LIB");.libPaths(c(p,.libPaths()));stopifnot(as.character(getRversion())=="4.6.1");req<-c("TENTACLES","tidymodels","bonsai","lightgbm","earth","xgboost","mixOmics","plsmod"); stopifnot(all(vapply(req,requireNamespace,logical(1),quietly=TRUE)));d<-utils::packageDescription("TENTACLES");stopifnot(tolower(d[["RemoteSha"]])=="069a26271e3885e015df25baa40a7a1140827d5f");cat("TEN_MODEL_DEPENDENCY_GATE_PASS\n")'
"$PY" "$ROOT/src/cmps355/tentacles_stage9_prepare_loco.py" "$ROOT"
DATA="$ROOT/runs/tentacles_stage9_inputs/$FOLD/discovery"
[[ -s "$DATA/counts.csv" && -s "$DATA/clinical.csv" ]] || { echo 'LOCO input files missing'; exit 2; }
mkdir -p "$ROOT/runs/tentacles_stage9_loco/$FOLD"
STAMP="$(date -u +%Y%m%dT%H%M%S%N)Z"
OUT="$ROOT/runs/tentacles_stage9_loco/$FOLD/$STAMP"
mkdir "$OUT"
exec 9>"$ROOT/runs/tentacles_stage9_loco/.single_runner.lock"
flock -n 9 || { echo 'Another full TENTACLES classifier run is already active in this workspace'; exit 2; }
cp "$ROOT/r/tentacles_stage9_loco_classifiers.R" "$ROOT/r/tentacles_stage9_consensus.R" "$OUT/"
sha256sum "$OUT/"*.R "$ROOT/data/validated/ibd_dataset_sample_aligned.zip" \
  "$DATA/counts.csv" "$DATA/clinical.csv" > "$OUT/input_and_code_sha256.txt"
"$R" --vanilla -e 'p<-Sys.getenv("CMPS355_R_LIB");.libPaths(c(p,.libPaths()));writeLines(capture.output(sessionInfo()),commandArgs(trailingOnly=TRUE)[[1]])' "$OUT/r_session_before.txt"
printf 'RUNNING\n' > "$OUT/STATUS.txt"
printf 'RUN_DIRECTORY=%s\n' "$OUT"
set +e
"$R" --vanilla "$ROOT/r/tentacles_stage9_loco_classifiers.R" \
  "$DATA/counts.csv" "$DATA/clinical.csv" "$OUT" "$FOLD" \
  '069a26271e3885e015df25baa40a7a1140827d5f' \
  2>&1 | tee "$OUT/classifiers.log"
rc=${PIPESTATUS[0]}
set -e
if ((rc != 0)) || [[ ! -s "$OUT/runclassifiers_finished.rds" ]] || \
   ! grep -q '^CLASSIFIERS_PASS=TRUE$' "$OUT/classifiers.log"; then
    printf 'FAILED_CLASSIFIERS exit_code=%d\n' "$rc" > "$OUT/STATUS.txt"
    echo "ERROR: native ten-model stage did not complete. Original evidence: $OUT"
    exit 1
fi
printf 'PASS_CLASSIFIERS_CONSENSUS_PENDING\n' > "$OUT/STATUS.txt"
set +e
"$R" --vanilla "$ROOT/r/tentacles_stage9_consensus.R" \
 "$OUT/runclassifiers_finished.rds" "$OUT" 2>&1 | tee "$OUT/consensus.log"
crc=${PIPESTATUS[0]}
set -e
if ((crc == 0)) && [[ -s "$OUT/consensus_native_finished.rds" ]] && \
   grep -q '^OFFICIAL_CONSENSUS_PASS=TRUE$' "$OUT/consensus.log"; then
  printf 'PASS_NATIVE_CLASSIFIERS_AND_CONSENSUS\n' > "$OUT/STATUS.txt"
else
  printf 'PASS_CLASSIFIERS_CONSENSUS_PENDING exit_code=%d\n' "$crc" > "$OUT/STATUS.txt"
  echo 'Official classifiers saved, but getConsensus needs inspection. Do not rerun classifiers.'
fi
sha256sum "$OUT/"*.rds > "$OUT/output_rds_sha256.txt"
printf 'NATIVE_OUTPUT=%s\n' "$OUT"
printf 'STATUS=%s\n' "$(cat "$OUT/STATUS.txt")"

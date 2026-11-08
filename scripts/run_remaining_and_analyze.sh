#!/usr/bin/env bash
# Runs D2-D4 in order, reusing completed immutable folds, then performs
# independent external LOCO validation. Does not touch/retrain completed D1.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs runs/tentacles_stage9_loco runs/tentacles_stage10_post
MASTER="$ROOT/logs/tentacles_overnight_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$MASTER") 2>&1
echo "MASTER_LOG=$MASTER"
echo "PIPELINE_STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec 8>"$ROOT/runs/tentacles_stage9_loco/.overnight_queue.lock"
flock -n 8 || { echo 'STOP: Another overnight queue is running'; exit 2; }
RUN_LOCK="$ROOT/runs/tentacles_stage9_loco/.single_runner.lock"
[[ -x "$ROOT/opt/R-4.6.1/bin/Rscript" ]] || { echo 'STOP: Private R is missing'; exit 2; }
[[ -x "$ROOT/.venv/bin/python" ]] || { echo 'STOP: Local Python venv is missing'; exit 2; }
[[ -f "$ROOT/r/tentacles_stage9_consensus.R" ]] || { echo 'STOP: Official consensus runner missing'; exit 2; }
[[ -f "$ROOT/src/cmps355/tentacles_stage10_post.py" ]] || { echo 'STOP: Stage 10 evaluator missing'; exit 2; }
if grep -q 'fit\[\["model.features"\]\]' "$ROOT/r/tentacles_stage9_loco_classifiers.R" "$ROOT/r/tentacles_stage9_consensus.R"; then
  echo 'STOP: S4 access bug still present; apply the D1 recovery patch first'; exit 2
fi
if ! grep -q 'library(TENTACLES)' "$ROOT/r/tentacles_stage9_loco_classifiers.R"; then
  echo 'STOP: annotated.genes namespace fix is missing'; exit 2
fi

latest_run() {
  local fold="$1" base="$ROOT/runs/tentacles_stage9_loco/$1"
  [[ -d "$base" ]] || return 0
  find "$base" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -n 1 | while IFS= read -r stamp; do
    [[ -n "$stamp" ]] && printf '%s/%s\n' "$base" "$stamp"
  done
}

status_is_pass() {
  local dir="$1"
  [[ -d "$dir" && -s "$dir/STATUS.txt" && -s "$dir/runclassifiers_finished.rds" &&
     -s "$dir/consensus_native_finished.rds" && -s "$dir/consensus_genes_native.csv" ]] &&
    [[ "$(cat "$dir/STATUS.txt")" == 'PASS_NATIVE_CLASSIFIERS_AND_CONSENSUS' ]]
}

# Lock synchronization matters: D2 might ALREADY be running in another terminal.
wait_for_active_fit() {
  echo 'Waiting for any running TENTACLES fold to finish (no parallel local fits)...'
  (
    exec 9>"$RUN_LOCK"
    flock 9
  )
  echo "Current fold lock released at $(date -u +%H:%M:%SZ)"
}

recover_consensus_if_needed() {
  local dir="$1" fold="$2" r="$ROOT/opt/R-4.6.1/bin/Rscript"
  [[ -s "$dir/runclassifiers_finished.rds" ]] || return 1
  echo "Saved official classifier object exists for $fold. Attempting ONLY consensus recovery."
  export R_LIBS="$ROOT/r_lib" R_LIBS_USER="$ROOT/r_lib" CMPS355_R_LIB="$ROOT/r_lib"
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  # Require an intact 10-model S4 object before any consensus computation.
  "$r" --vanilla - "$dir/runclassifiers_finished.rds" <<'RS'
args<-commandArgs(trailingOnly=TRUE)
x<-readRDS(args[[1]])
stopifnot(isS4(x),inherits(x,'runClassifiers.obj'),
          'model.features'%in%methods::slotNames(x))
f<-methods::slot(x,'model.features')
stopifnot(is.list(f),length(f)==10L)
cat('RECOVERABLE_TEN_MODEL_OBJECT=TRUE\n')
RS
  if [[ ! -s "$dir/consensus_native_finished.rds" ]]; then
    "$r" --vanilla "$ROOT/r/tentacles_stage9_consensus.R" \
      "$dir/runclassifiers_finished.rds" "$dir" 2>&1 | tee "$dir/consensus_recovery.log"
  fi
  # The already-saved D1 native consensus may require gene-vector inspection;
  # do not invent an absent CSV for any fold.
  if [[ -s "$dir/consensus_native_finished.rds" && -s "$dir/consensus_genes_native.csv" ]]; then
    printf 'PASS_NATIVE_CLASSIFIERS_AND_CONSENSUS\n' > "$dir/STATUS.txt"
    sha256sum "$dir"/*.rds > "$dir/output_rds_sha256.txt"
    echo "CONSENSUS_RECOVERED=$fold"
    return 0
  fi
  return 1
}

# D1 has already been trained and recovered. Never run it again.
D1="$(latest_run d1)"
if [[ -z "$D1" ]] || ! status_is_pass "$D1"; then
  echo 'STOP: Completed D1 classifier+consensus evidence not verified.'
  echo 'Do not start over; inspect D1 STATUS.txt and saved files.'
  exit 2
fi
echo "PRESERVED_D1=$D1"

for fold in d2 d3 d4; do
  echo "================= $fold ================="
  wait_for_active_fit
  dir="$(latest_run "$fold")"
  if [[ -n "$dir" ]] && status_is_pass "$dir"; then
    echo "REUSING_COMPLETE_$fold=$dir"
    continue
  fi
  if [[ -n "$dir" && -s "$dir/runclassifiers_finished.rds" ]]; then
    recover_consensus_if_needed "$dir" "$fold" || {
      echo "STOP: $fold native fit saved but consensus not recovered. Training was NOT rerun."
      exit 3
    }
    status_is_pass "$dir" || { echo "STOP: $fold recovery incomplete"; exit 3; }
    continue
  fi
  if [[ -n "$dir" ]]; then
    echo "STOP: $fold has an incomplete run without a completed native RDS: $dir"
    echo 'Manual inspection required; automatically rerunning expensive classifiers is unsafe.'
    exit 3
  fi
  echo "STARTING_NATIVE_$fold=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ! bash "$ROOT/scripts/run_fold.sh" "$fold"; then
    echo "STOP: $fold failed. Saved evidence retained. See newest STATUS.txt and logs."
    exit 4
  fi
  dir="$(latest_run "$fold")"
  if ! status_is_pass "$dir"; then
    recover_consensus_if_needed "$dir" "$fold" || {
      echo "STOP: $fold has incomplete consensus. Preserving native fit."
      exit 4
    }
  fi
  status_is_pass "$dir" || { echo "STOP: $fold final evidence incomplete"; exit 4; }
  echo "COMPLETE_$fold=$dir"
done

echo 'ALL_FOUR_NATIVE_CONSENSUS_FOLDS_PASS=TRUE'
echo 'STARTING_STAGE10_HELDOUT_EVALUATION'
"$ROOT/.venv/bin/python" "$ROOT/src/cmps355/tentacles_stage10_post.py" "$ROOT"
echo 'FULL_QUEUE_AND_HELDOUT_ANALYSIS_PASS=TRUE'
echo "PIPELINE_FINISHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

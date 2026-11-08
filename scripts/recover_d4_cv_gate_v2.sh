#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
R="$ROOT/opt/R-4.6.1/bin/Rscript"
[[ -x "$R" ]] || { echo 'STOP: pinned R 4.6.1 binary not found' >&2; exit 2; }
SRC="$ROOT/runs/tentacles_cd_positive_v2/d4/20261108T174423655704315Z"
[[ -d "$SRC" ]] || { echo "STOP: Expected exact D4 source missing: $SRC" >&2; exit 2; }
[[ -s "$SRC/runclassifiers_finished.rds" ]] || { echo 'STOP: saved native fit missing. Do NOT rerun automatically.' >&2; exit 2; }
[[ -s "$SRC/STATUS.txt" ]] || { echo 'STOP: missing original D4 STATUS' >&2; exit 2; }
[[ "$(cat "$SRC/STATUS.txt")" != 'PASS_CD_POSITIVE_NATIVE_NMIN6_AND_NMIN5' ]] || { echo 'STOP: D4 already complete' >&2; exit 2; }
mkdir -p "$ROOT/runs/tentacles_cd_positive_v2_recovery/d4"
exec 9>"$ROOT/runs/tentacles_cd_positive_v2/.v2_single_process.lock"
flock -n 9 || { echo 'STOP: native V2 runner still active; do not run in parallel' >&2; exit 2; }
OUT="$ROOT/runs/tentacles_cd_positive_v2_recovery/d4/$(date -u +%Y%m%dT%H%M%S%N)Z"
mkdir "$OUT"
exec > >(tee "$OUT/recovery.log") 2>&1
printf 'ORIGINAL_D4=%s\nRECOVERY_OUTPUT=%s\n' "$SRC" "$OUT"
sha256sum "$SRC/runclassifiers_finished.rds" > "$OUT/source_rds_sha256_BEFORE.txt"
printf 'RECOVERY_IN_PROGRESS\n' > "$OUT/RECOVERY_STATUS.txt"
export R_LIBS="$ROOT/r_lib" R_LIBS_USER="$ROOT/r_lib" CMPS355_R_LIB="$ROOT/r_lib"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
set +e
"$R" --vanilla "$ROOT/r/recover_d4_cv_gate_v2.R" "$SRC" "$OUT"
rc=$?
set -e
sha256sum "$SRC/runclassifiers_finished.rds" > "$OUT/source_rds_sha256_AFTER.txt"
if ! cmp -s "$OUT/source_rds_sha256_BEFORE.txt" "$OUT/source_rds_sha256_AFTER.txt"; then
  printf 'FAIL_ORIGINAL_SOURCE_HASH_CHANGED\n' > "$OUT/RECOVERY_STATUS.txt"
  echo 'STOP: original trained RDS changed unexpectedly; inspect immediately' >&2
  exit 3
fi
if ((rc != 0)); then
 echo "D4_AUDIT_NEEDS_REVIEW=TRUE;R_EXIT=$rc"
 echo "D4_AUDIT_OUTPUT=$OUT"
 [[ -f "$OUT/native_cv_f1_by_model.csv" ]] && cat "$OUT/native_cv_f1_by_model.csv"
 exit "$rc"
fi
[[ -s "$OUT/consensus_native_finished.rds" && -s "$OUT/consensus_nmin5_finished.rds" ]] || { echo 'STOP: no verified consensus outputs' >&2; exit 3; }
grep -Fxq 'RECOVERY_PASS_OFFICIAL_NINE_VOTER_CONSENSUS' "$OUT/RECOVERY_STATUS.txt" || { echo 'STOP: recovery did not pass' >&2; exit 3; }
echo 'D4_RECOVERY_PASS_OFFICIAL_NINE_VOTER_CONSENSUS=TRUE'
echo "D4_RECOVERY_PASS_OUTPUT=$OUT"
echo 'Do not mark D4 COMPLETE or run evaluate yet; send recovery report for provenance review.'

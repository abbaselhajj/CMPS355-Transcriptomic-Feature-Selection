#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/CMPS355_TENTACLES_Local"
R_BIN="$PWD/opt/R-4.6.1/bin/Rscript"
if [[ ! -x "$R_BIN" ]]; then R_BIN="$(command -v Rscript)"; fi
export R_LIBS_USER="$PWD/r_lib" R_LIBS="$PWD/r_lib"
output="runs/native_quality_scores/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$(dirname "$output")"
"$R_BIN" --vanilla r/tentacles_readonly_classifier_scores.R "$PWD" "$output"
echo "RESULT_DIRECTORY=$PWD/$output"

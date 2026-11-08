#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/CMPS355_TENTACLES_Local"
export R_LIBS_USER="$PWD/r_lib" R_LIBS="$PWD/r_lib"
R_BIN="$PWD/opt/R-4.6.1/bin/Rscript"
if [[ ! -x "$R_BIN" ]]; then R_BIN="$(command -v Rscript)"; fi
for fold in d1 d2 d3 d4; do
  found=()
  while IFS= read -r -d '' candidate; do found+=("$candidate"); done < <(find "runs/tentacles_stage9_loco/$fold" -mindepth 1 -maxdepth 1 -type d -print0)
  for dir in "${found[@]}"; do
    if [[ -f "$dir/STATUS.txt" ]] && grep -qx 'PASS_NATIVE_CLASSIFIERS_AND_CONSENSUS' "$dir/STATUS.txt"; then
      target="runs/native_quality_readonly/$fold/$(basename "$dir")"
      mkdir -p "$(dirname "$target")"
      "$R_BIN" --vanilla r/tentacles_readonly_native_quality_audit.R "$dir" "$target"
    fi
  done
done

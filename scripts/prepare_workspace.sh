#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$HOME/CMPS355_StableMate_Local/ibd_dataset_sample_aligned.zip}"
DEST="$ROOT/data/validated/ibd_dataset_sample_aligned.zip"
[[ -f "$SRC" || -f "$DEST" ]] || { echo "Dataset missing. Supply its existing WSL path: bash scripts/prepare_workspace.sh /path/to/ibd_dataset_sample_aligned.zip"; exit 2; }
if [[ ! -f "$DEST" ]]; then cp "$SRC" "$DEST"; fi
printf '%s  %s\n' '8a83f20035fcac5ae5d8bfe496e4d7491e47400b61ca1080b828c8ad54e91a74' "$DEST" | sha256sum -c -
python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/requirements-local.txt"
"$ROOT/.venv/bin/python" "$ROOT/src/cmps355/tentacles_stage9_prepare_loco.py" "$ROOT"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for fold in d1 d2 d3 d4; do
  echo "=== OFFICIAL TENTACLES LOCO $fold ==="
  bash "$ROOT/scripts/run_fold.sh" "$fold"
done

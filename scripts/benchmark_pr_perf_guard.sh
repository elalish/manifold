#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <clipper2-dir> <repeats>"
  exit 2
fi

CLIPPER2_DIR="$1"
REPEATS="$2"

for variant in base head; do
  bash ./scripts/run_pr_perf_guard.sh "./wt-${variant}" "$CLIPPER2_DIR" "./bench/${variant}" "$REPEATS"
done

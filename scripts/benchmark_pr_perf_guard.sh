#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <repeats>"
  exit 2
fi

REPEATS="$1"

for variant in base head; do
  bash ./scripts/run_pr_perf_guard.sh "./wt-${variant}" "./bench/${variant}" "$REPEATS"
done

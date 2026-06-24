#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <base-dir> <head-dir> <warn-pct> <warn-abs-ms>"
  exit 2
fi

BASE_DIR="$1"
HEAD_DIR="$2"
WARN_PCT="$3"
WARN_ABS_MS="$4"

mkdir -p ./bench

python3 ./scripts/compare_pr_perf_guard.py \
  --base-dir "$BASE_DIR" \
  --head-dir "$HEAD_DIR" \
  --warn-pct "$WARN_PCT" \
  --warn-abs-ms "$WARN_ABS_MS" \
  --markdown-out ./bench/summary.md \
  --json-out ./bench/result.json

echo "::group::PR benchmark summary"
cat ./bench/summary.md
echo "::endgroup::"

echo "::group::PR benchmark result.json"
cat ./bench/result.json
echo "::endgroup::"

echo "::group::PR benchmark raw outputs"
for variant in base head; do
  for run_file in "./bench/${variant}"/run*.txt; do
    [ -f "${run_file}" ] || continue
    echo "--- ${run_file} ---"
    cat "${run_file}"
  done
done
echo "::endgroup::"

cat ./bench/summary.md >> "$GITHUB_STEP_SUMMARY"
echo "" >> "$GITHUB_STEP_SUMMARY"
echo "Raw logs: open this step and expand \`PR benchmark result.json\` and \`PR benchmark raw outputs\` groups." >> "$GITHUB_STEP_SUMMARY"

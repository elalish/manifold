#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <suite-dir> <repeats>"
  exit 2
fi

SUITE_DIR="$1"
REPEATS="$2"

python3 ./scripts/parse_weekly_benchmarks.py \
  --suite-dir "$SUITE_DIR" \
  --repeats "$REPEATS" \
  --markdown-out "${SUITE_DIR}/summary.md" \
  --json-out "${SUITE_DIR}/result.json"

echo "::group::Weekly benchmark summary"
cat "${SUITE_DIR}/summary.md"
echo "::endgroup::"

echo "::group::Weekly benchmark result.json"
cat "${SUITE_DIR}/result.json"
echo "::endgroup::"

echo "::group::Weekly benchmark raw outputs"
for run_file in "${SUITE_DIR}"/run*.txt; do
  [ -f "${run_file}" ] || continue
  echo "--- ${run_file} ---"
  cat "${run_file}"
done
echo "::endgroup::"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "${SUITE_DIR}/summary.md" >> "$GITHUB_STEP_SUMMARY"
  echo "" >> "$GITHUB_STEP_SUMMARY"
  echo "Raw logs: open this step and expand \`Weekly benchmark result.json\` and \`Weekly benchmark raw outputs\` groups." >> "$GITHUB_STEP_SUMMARY"
fi

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <source-dir> <suite-dir> <repeats>"
  exit 2
fi

SOURCE_DIR="$1"
SUITE_DIR="$2"
REPEATS="$3"

python3 ./scripts/parse_weekly_benchmarks.py \
  --source-dir "$SOURCE_DIR" \
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
find "$SUITE_DIR" -path "${SUITE_DIR}/build" -prune -o -name 'run*.txt' -type f -print | sort | while read -r run_file; do
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

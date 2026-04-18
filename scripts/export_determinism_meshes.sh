#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <run-dir> <test-binary> <output-dir>"
  exit 2
fi

RUN_DIR="$1"
TEST_BIN="$2"
OUT_DIR="$3"

source "$(dirname "$0")/determinism_cases.sh"

repo_root="$(pwd)"
if [[ "$OUT_DIR" = /* ]]; then
  out_abs="$OUT_DIR"
else
  out_abs="$repo_root/$OUT_DIR"
fi

cd "$RUN_DIR"
"$TEST_BIN" -e --gtest_filter="$DETERMINISM_GTEST_FILTER"

mkdir -p "$out_abs"
cp "${DETERMINISM_OBJ_FILES[@]}" "$out_abs/"

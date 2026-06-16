#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <source-dir> <clipper2-dir> <output-dir> <repeats>"
  exit 2
fi

SRC_DIR="$1"
CLIPPER2_DIR="$2"
OUT_DIR="$3"
REPEATS="$4"
BUILD_DIR="${OUT_DIR}/build"
SPEC_FILE="${SRC_DIR}/extras/ember_tests/testfiles/ember-benchmark-cases.json"
MESH_DIR="${OUT_DIR}/raw_meshes"

WEEKLY_BENCHMARK_CASES="${WEEKLY_BENCHMARK_CASES:-16 84 667 695 260 406 551 582}"
WEEKLY_BENCHMARK_THREADS="${WEEKLY_BENCHMARK_THREADS:-1}"
WEEKLY_BENCHMARK_VERBOSE="${WEEKLY_BENCHMARK_VERBOSE:-2}"

cmake \
  -S "$SRC_DIR" \
  -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMANIFOLD_STRICT=ON \
  -DMANIFOLD_PYBIND=OFF \
  -DMANIFOLD_TEST=ON \
  -DMANIFOLD_PAR=ON \
  -DMANIFOLD_TIMING=ON \
  -DASSIMP_ENABLE=ON \
  -DFETCHCONTENT_SOURCE_DIR_CLIPPER2="$CLIPPER2_DIR"

cmake --build "$BUILD_DIR" --target man_bench

BIN=""
for candidate in "${BUILD_DIR}/extras/man_bench" "${BUILD_DIR}/bin/man_bench"; do
  if [ -x "$candidate" ]; then
    BIN="$candidate"
    break
  fi
done
if [ -z "$BIN" ]; then
  echo "man_bench binary not found in expected paths."
  exit 1
fi

mkdir -p "$OUT_DIR"
python3 "${SRC_DIR}/scripts/run_weekly_benchmark_cases.py" \
  --binary "$BIN" \
  --spec "$SPEC_FILE" \
  --mesh-dir "$MESH_DIR" \
  --out-dir "$OUT_DIR" \
  --cases "$WEEKLY_BENCHMARK_CASES" \
  --repeats "$REPEATS" \
  --threads "$WEEKLY_BENCHMARK_THREADS" \
  --verbose "$WEEKLY_BENCHMARK_VERBOSE"

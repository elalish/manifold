#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <source-dir> <output-dir> <repeats>"
  exit 2
fi

SRC_DIR="$1"
OUT_DIR="$2"
REPEATS="$3"
BUILD_DIR="${OUT_DIR}/build"

cmake \
  -S "$SRC_DIR" \
  -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMANIFOLD_STRICT=ON \
  -DMANIFOLD_PYBIND=OFF \
  -DMANIFOLD_TEST=ON \
  -DMANIFOLD_PAR=OFF

cmake --build "$BUILD_DIR" --target perfTest

BIN=""
for candidate in "${BUILD_DIR}/extras/perfTest" "${BUILD_DIR}/bin/perfTest"; do
  if [ -x "$candidate" ]; then
    BIN="$candidate"
    break
  fi
done
if [ -z "$BIN" ]; then
  echo "perfTest binary not found in expected paths."
  exit 1
fi

mkdir -p "$OUT_DIR"
for i in $(seq 1 "$REPEATS"); do
  "$BIN" > "${OUT_DIR}/run${i}.txt"
done

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
OS_TYPE="$(uname -s)"

cmake_args=(
  -S "$SRC_DIR"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DMANIFOLD_STRICT=ON
  -DMANIFOLD_DOWNLOADS=OFF
  -DMANIFOLD_PYBIND=OFF
  -DMANIFOLD_TEST=ON
  -DMANIFOLD_PAR=OFF
)

if [ "$OS_TYPE" = "Darwin" ]; then
  cmake_args+=(-DCMAKE_OSX_ARCHITECTURES="${CMAKE_OSX_ARCHITECTURES:-arm64}")
fi

cmake "${cmake_args[@]}"

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

run_measured_perf_size() {
  local size_index="$1"
  local output status ntri peak_rss_mb peak_rss_bytes peak_rss_kb

  set +e
  if [ "$OS_TYPE" = "Darwin" ]; then
    output="$(/usr/bin/time -l "$BIN" --size-index "$size_index" 2>&1)"
    status="$?"
    peak_rss_bytes="$(printf "%s\n" "$output" | awk '/maximum resident set size/ {print $1; exit}')"
    peak_rss_mb="$(awk -v bytes="${peak_rss_bytes:-0}" 'BEGIN {printf "%.2f", bytes / 1024 / 1024}')"
  else
    output="$(/usr/bin/time -v "$BIN" --size-index "$size_index" 2>&1)"
    status="$?"
    peak_rss_kb="$(printf "%s\n" "$output" | awk -F: '/Maximum resident set size/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
    peak_rss_bytes="$(awk -v kb="${peak_rss_kb:-0}" 'BEGIN {printf "%.0f", kb * 1024}')"
    peak_rss_mb="$(awk -v kb="${peak_rss_kb:-0}" 'BEGIN {printf "%.2f", kb / 1024}')"
  fi
  set -e

  printf "%s\n" "$output"
  ntri="$(printf "%s\n" "$output" | sed -nE 's/^nTri = ([0-9]+),.*/\1/p' | head -n 1)"
  echo "PEAK_RSS nTri=${ntri:-unknown} size_index=${size_index} peak_rss_mb=${peak_rss_mb} peak_rss_bytes=${peak_rss_bytes:-0}"
  return "$status"
}

mkdir -p "$OUT_DIR"
for i in $(seq 1 "$REPEATS"); do
  run_file="${OUT_DIR}/run${i}.txt"
  : > "$run_file"
  for size_index in $(seq 0 7); do
    {
      echo "### perfTest size_index=${size_index}"
      run_measured_perf_size "$size_index"
      echo
    } >> "$run_file"
  done
done

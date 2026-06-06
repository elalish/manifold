#!/usr/bin/env bash
set -euo pipefail

source ./scripts/sanitizer_cases.sh

case "${SANITIZER_SUBSET}" in
  core)
    SANITIZER_GTEST_FILTER="${SANITIZER_GTEST_FILTER_CORE}"
    ;;
  extended)
    SANITIZER_GTEST_FILTER="${SANITIZER_GTEST_FILTER_EXTENDED}"
    ;;
  *)
    echo "::warning::Unknown SANITIZER_SUBSET=${SANITIZER_SUBSET}, falling back to core."
    SANITIZER_GTEST_FILTER="${SANITIZER_GTEST_FILTER_CORE}"
    ;;
esac

SANITIZER_TEST_BIN=""
for CANDIDATE in \
  ./build/test/manifold_test \
  ./test/manifold_test \
  ./sanitizer-build/test/manifold_test \
  ./sanitizer-build/build/test/manifold_test; do
  if [ -f "${CANDIDATE}" ]; then
    SANITIZER_TEST_BIN="${CANDIDATE}"
    break
  fi
done

if [ -z "${SANITIZER_TEST_BIN}" ]; then
  SANITIZER_TEST_BIN="$(find . -type f -name manifold_test | head -n 1)"
fi
if [ -z "${SANITIZER_TEST_BIN}" ]; then
  echo "::error::Could not find manifold_test after downloading sanitizer artifacts."
  echo "::group::Artifact layout"
  find . -maxdepth 4 -type d | sort
  echo "::endgroup::"
  exit 127
fi
chmod +x "${SANITIZER_TEST_BIN}" || true

BIN_DIR="$(dirname "${SANITIZER_TEST_BIN}")"
LIB_CANDIDATES=(
  "${BIN_DIR}/../src"
  "${BIN_DIR}/../lib"
  "${BIN_DIR}/../bindings/c"
  ./src
  ./lib
  ./bindings/c
  ./build/src
  ./build/lib
  ./build/bindings/c
)

LIB_PATHS=()
for LIB_DIR in "${LIB_CANDIDATES[@]}"; do
  if [ -d "${LIB_DIR}" ]; then
    LIB_PATHS+=("${LIB_DIR}")
  fi
done

if [ "${#LIB_PATHS[@]}" -gt 0 ]; then
  mapfile -t LIB_PATHS < <(printf '%s\n' "${LIB_PATHS[@]}" | awk '!seen[$0]++')
  SANITIZER_LD_PATH="$(IFS=:; echo "${LIB_PATHS[*]}")"
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${SANITIZER_LD_PATH}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${SANITIZER_LD_PATH}"
  fi
fi

set +e
timeout "${SANITIZER_TEST_TIMEOUT_SEC}" \
  "${SANITIZER_TEST_BIN}" --gtest_filter="${SANITIZER_GTEST_FILTER}"
TEST_RC="$?"
set -e
if [ "${TEST_RC}" -eq 124 ]; then
  echo "::warning::Sanitizer test timed out after ${SANITIZER_TEST_TIMEOUT_SEC}s."
fi

exit "${TEST_RC}"

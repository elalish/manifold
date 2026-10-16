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
    echo "::error::Unknown SANITIZER_SUBSET=${SANITIZER_SUBSET}"
    exit 2
    ;;
esac

# The sanitizer build artifact is always downloaded into build/.
SANITIZER_TEST_BIN=./build/test/manifold_test
if [ ! -f "${SANITIZER_TEST_BIN}" ]; then
  echo "::error::Could not find ${SANITIZER_TEST_BIN} after downloading sanitizer artifacts."
  echo "::group::Artifact layout"
  find . -maxdepth 4 -type d | sort
  echo "::endgroup::"
  exit 127
fi
# Artifact downloads do not preserve the executable bit.
chmod +x "${SANITIZER_TEST_BIN}"

set +e
timeout "${SANITIZER_TEST_TIMEOUT_SEC}" "${SANITIZER_TEST_BIN}" --gtest_filter="${SANITIZER_GTEST_FILTER}"
TEST_RC="$?"
set -e
if [ "${TEST_RC}" -eq 124 ]; then
  echo "::warning::Sanitizer test timed out after ${SANITIZER_TEST_TIMEOUT_SEC}s."
fi

exit "${TEST_RC}"

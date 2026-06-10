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

set +e
timeout "${SANITIZER_TEST_TIMEOUT_SEC}" \
  "${SANITIZER_TEST_BIN}" --gtest_filter="${SANITIZER_GTEST_FILTER}"
TEST_RC="$?"
set -e
if [ "${TEST_RC}" -eq 124 ]; then
  echo "::warning::Sanitizer test timed out after ${SANITIZER_TEST_TIMEOUT_SEC}s."
fi

exit "${TEST_RC}"

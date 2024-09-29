#!/usr/bin/env bash

# Modified from https://github.com/google/fuzztest/blob/main/tools/minimizer.sh
# If you want to use a specific reproducer, pass it in FUZZTEST_MINIMIZE_REPRODUCER
# environment variable.
# Otherwise, just run it as
# $ ./minimizer.sh ./build/test/polygon_fuzz --fuzz=PolygonFuzz.TriangulationNoCrash
# If you want to kill it, try SIGKILL, SIGTERM is not very useful
# If you want to end the loop early, rename $(pwd)/reproducers into something
# else...

if [ -z "${FUZZTEST_MINIMIZE_REPRODUCER}"]; then
  mkdir -p $(pwd)/reproducers
  FUZZTEST_REPRODUCERS_OUT_DIR=$(pwd)/reproducers "$@"
  wildcard="$(pwd)/reproducers/*"
  reproducers=($wildcard)
  FUZZTEST_MINIMIZE_REPRODUCER="${reproducers[0]}"
fi
readonly ORIGINAL_REPRODUCER="${FUZZTEST_MINIMIZE_REPRODUCER}"

for i in {0001..9999}; do
  echo
  echo "╔════════════════════════════════════════════════╗"
  echo "║ Minimization round: ${i}                       ║"
  echo "╚════════════════════════════════════════════════╝"
  echo "Note that to terminate early, simply move $ORIGINAL_REPRODUCER to somewhere else..."
  echo

  if [ ! -f $ORIGINAL_REPRODUCER ]; then
    echo "Terminated by the user"
    break
  fi

  TEMP_DIR=$(mktemp -d)
  FUZZTEST_REPRODUCERS_OUT_DIR="${TEMP_DIR}" \
  FUZZTEST_MINIMIZE_REPRODUCER="${FUZZTEST_MINIMIZE_REPRODUCER}" \
  "$@"

  if [ $? -eq 130 ]; then
    echo
    echo "╔═══════════════════════════════════════════════╗"
    echo "║ Minimization terminated.                      ║"
    echo "╚═══════════════════════════════════════════════╝"
    echo
    echo "Find the smallest reproducer at:"
    echo
    echo "${FUZZTEST_MINIMIZE_REPRODUCER}"

    rm -rf "${TEMP_DIR}"
    break
  fi

  SMALLER_REPRODUCER=$(find "${TEMP_DIR}" -type f)
  NEW_NAME="${ORIGINAL_REPRODUCER}-min-${i}"
  mv "${SMALLER_REPRODUCER}" "${NEW_NAME}"
  FUZZTEST_MINIMIZE_REPRODUCER="${NEW_NAME}"

  rm -rf "${TEMP_DIR}"
done

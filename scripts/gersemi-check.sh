#!/usr/bin/env bash

FAILED=0

function check() {
  if [[ $1 != *build* ]]; then
    gersemi -c $1 2> /dev/null
    if [ $? -ne 0 ]; then
      gersemi --diff $1 2> /dev/null
      FAILED=1
    fi
  fi
}

for f in $(find -name CMakeLists.txt); do
  check $f
done

for f in $(find -name '*.cmake.in'); do
  check $f
done

if [[ $FAILED -ne 0 ]]; then
  exit 1
fi

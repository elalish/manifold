#!/usr/bin/env bash

shopt -s extglob
if [ -z "$CLANG_FORMAT" ]; then
CLANG_FORMAT=clang-format
fi

$CLANG_FORMAT -i extras/*.cpp
$CLANG_FORMAT -i samples/*/*.{h,cpp}
$CLANG_FORMAT -i test/*.{h,cpp}
$CLANG_FORMAT -i bindings/*/*.cpp
$CLANG_FORMAT -i bindings/c/include/manifold/*.h
$CLANG_FORMAT -i bindings/wasm/*.{js,ts}
$CLANG_FORMAT -i bindings/wasm/examples/*.{js,ts,html}
$CLANG_FORMAT -i bindings/wasm/examples/public/*.{js,ts}
$CLANG_FORMAT -i src/*.{h,cpp}
$CLANG_FORMAT -i src/*/*.cpp
$CLANG_FORMAT -i include/manifold/*.h

black bindings/python/examples/*.py

for pattern in 'CMakeLists.txt' '*.cmake*'; do
  for f in $(find -name ${pattern}); do
    # skip build directories
    if [[ $f != *build* && $f != *node_modules* ]]; then
      gersemi -i $f
    fi
  done
done

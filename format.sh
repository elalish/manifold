#!/usr/bin/env bash

shopt -s extglob
if [ -z "$CLANG_FORMAT" ]; then
CLANG_FORMAT=$(dirname $(which clang-11))/clang-format
fi

$CLANG_FORMAT -i extras/*.cpp
$CLANG_FORMAT -i meshIO/**/*.{h,cpp}
$CLANG_FORMAT -i samples/**/*.{h,cpp}
$CLANG_FORMAT -i test/*.{h,cpp}
$CLANG_FORMAT -i bindings/*/*.cpp
$CLANG_FORMAT -i bindings/c/include/*.h
$CLANG_FORMAT -i bindings/wasm/**/*.{js,ts,html}
$CLANG_FORMAT -i src/!(third_party)/*/*.{h,cpp}
black bindings/python/examples/*.py

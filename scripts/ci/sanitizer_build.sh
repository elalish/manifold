#!/usr/bin/env bash
set -euo pipefail

SANITIZER_C_COMPILER="${SANITIZER_C_COMPILER:-clang-18}"
SANITIZER_CXX_COMPILER="${SANITIZER_CXX_COMPILER:-clang++-18}"

mkdir -p build
cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_SHARED_LIBS=ON \
  -DMANIFOLD_STRICT=ON \
  -DMANIFOLD_PYBIND=OFF \
  -DMANIFOLD_DEBUG=ON \
  -DMANIFOLD_ASSERT=ON \
  -DMANIFOLD_PAR=OFF \
  -DCMAKE_C_COMPILER="${SANITIZER_C_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${SANITIZER_CXX_COMPILER}" \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  . -B build | tee build/cmake_configure.log

cmake --build build

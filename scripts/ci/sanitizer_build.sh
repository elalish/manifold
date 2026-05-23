#!/usr/bin/env bash
set -euo pipefail

cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_SHARED_LIBS=ON \
  -DMANIFOLD_STRICT=ON \
  -DMANIFOLD_PYBIND=OFF \
  -DMANIFOLD_DEBUG=ON \
  -DMANIFOLD_ASSERT=ON \
  -DMANIFOLD_CROSS_SECTION=ON \
  -DMANIFOLD_PAR=OFF \
  -DCMAKE_C_COMPILER=clang-18 \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DFETCHCONTENT_SOURCE_DIR_CLIPPER2=clipper2 \
  . -B build

cmake --build build

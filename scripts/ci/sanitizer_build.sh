#!/usr/bin/env -S bash -euo pipefail

# Only CXX flags are set: manifold is declared `project(manifold LANGUAGES CXX)`,
# so cmake never reads CMAKE_C_FLAGS.
#
# MANIFOLD_STRICT is deliberately OFF here - it adds -Werror, and this lane is
# for finding runtime memory/UB bugs, not for enforcing warnings.
cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_SHARED_LIBS=ON \
  -DMANIFOLD_STRICT=OFF \
  -DMANIFOLD_PYBIND=OFF \
  -DMANIFOLD_DEBUG=ON \
  -DMANIFOLD_ASSERT=ON \
  -DMANIFOLD_PAR=OFF \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  . -B build

cmake --build build

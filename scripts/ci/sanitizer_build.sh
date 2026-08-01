#!/usr/bin/env -S bash -euo pipefail

# Only CXX flags are set: manifold is declared `project(manifold LANGUAGES CXX)`,
# so cmake never reads CMAKE_C_FLAGS.
cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_SHARED_LIBS=ON \
  -DMANIFOLD_STRICT=ON \
  -DMANIFOLD_PYBIND=OFF \
  -DMANIFOLD_DEBUG=ON \
  -DMANIFOLD_ASSERT=ON \
  -DMANIFOLD_PAR=OFF \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -Wno-maybe-uninitialized" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  . -B build

cmake --build build

#!/usr/bin/env -S bash -euo pipefail

# No compiler forced by default - GCC (the runner's default) supports
# -fsanitize=address,undefined fine. Set SANITIZER_C_COMPILER/
# SANITIZER_CXX_COMPILER to override if a specific compiler is needed.
cmake_args=(
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DBUILD_SHARED_LIBS=ON
  -DMANIFOLD_STRICT=ON
  -DMANIFOLD_PYBIND=OFF
  -DMANIFOLD_DEBUG=ON
  -DMANIFOLD_ASSERT=ON
  -DMANIFOLD_PAR=OFF
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined"
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined"
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
)
if [ -n "${SANITIZER_C_COMPILER:-}" ]; then
  cmake_args+=(-DCMAKE_C_COMPILER="${SANITIZER_C_COMPILER}")
fi
if [ -n "${SANITIZER_CXX_COMPILER:-}" ]; then
  cmake_args+=(-DCMAKE_CXX_COMPILER="${SANITIZER_CXX_COMPILER}")
fi

mkdir -p build
cmake "${cmake_args[@]}" . -B build | tee build/cmake_configure.log

cmake --build build

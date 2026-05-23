#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF' >> "${GITHUB_STEP_SUMMARY}"
### Sanitizer Reproduction

Reproduce locally from repo root:

```bash
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
  . -B build
cmake --build build
source ./scripts/sanitizer_cases.sh
ASAN_OPTIONS=detect_container_overflow=0:strict_init_order=1 \
UBSAN_OPTIONS=print_stacktrace=1 \
./build/test/manifold_test --gtest_filter="${SANITIZER_GTEST_FILTER_CORE}"
```
EOF

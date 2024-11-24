#!/usr/bin/env bash
mkdir make-consumer
cd make-consumer

cat <<'EOT' > Makefile
override CXXFLAGS += $(shell pkg-config --cflags manifold)
override LDFLAGS += $(shell pkg-config --libs manifold)

testing : testing.cpp
EOT

cat <<EOT > testing.cpp
#include <manifold/manifold.h>
#include <manifold/version.h>

#if MANIFOLD_VERSION < MANIFOLD_VERSION_NUMBER(2, 5, 1)
# error "Unexpected: minimum version number not available"
#endif

int main() { manifold::Manifold foo; return 0; }
EOT

make
./testing

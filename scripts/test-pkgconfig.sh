#!/usr/bin/env bash
mkdir make-consumer
cd make-consumer

cat <<'EOT' > Makefile
CXXFLAGS=$(shell pkg-config --cflags manifold)
LDFLAGS=$(shell pkg-config --libs manifold)

testing : testing.cpp
EOT

cat <<EOT > testing.cpp
#include <manifold/manifold.h>
#include <manifold/parallel.h>
int main() { manifold::Manifold foo; return 0; }
EOT

make
./testing

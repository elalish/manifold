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
int main() { manifold::Manifold foo; return 0; }
EOT

make
./testing

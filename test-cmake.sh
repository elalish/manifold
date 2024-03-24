#!/usr/bin/env bash
mkdir cmake-consumer
cd cmake-consumer

cat <<EOT >> CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(testing LANGUAGES CXX)
find_package(manifold "2.4.5" REQUIRED)
add_executable(testing test.cpp)
target_link_libraries(testing PRIVATE manifold)
EOT

cat <<EOT >> test.cpp
#include "manifold.h"
int main() { manifold::Manifold foo; return 0; }
EOT

mkdir build
cd build
cmake ..
make
./testing


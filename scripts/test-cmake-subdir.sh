#!/usr/bin/env bash
mkdir cmake-consumer
cd cmake-consumer

cat <<EOT > CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(testing LANGUAGES CXX)
set(MANIFOLD_PAR ON)
add_subdirectory(manifold EXCLUDE_FROM_ALL)
add_executable(testing test.cpp)
target_link_libraries(testing PRIVATE manifold::manifold)
EOT

cat <<EOT > test.cpp
#include <manifold/manifold.h>
#include <manifold/version.h>
int main() { manifold::Manifold foo; return 0; }
EOT

cp -r ../manifold ./
mkdir build
cd build
cmake ..
make
./testing


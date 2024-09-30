// Copyright 2020 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <iostream>

#include "manifold/manifold.h"

using namespace manifold;

/*
  Build & execute with the following command:

  ( mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DMANIFOLD_PAR=ON .. && \
    make -j && \
    time ./extras/largeSceneTest 50 )
*/
int main(int argc, char **argv) {
  int n = 20;
  if (argc == 2) n = atoi(argv[1]);

  std::cout << "n = " << n << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  Manifold scene;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        if (i == 0 && j == 0 && k == 0) continue;

        Manifold sphere = Manifold::Sphere(1).Translate(vec3(i, j, k));
        scene = scene.Boolean(sphere, OpType::Add);
      }
    }
  }
  scene.NumTri();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "nTri = " << scene.NumTri() << ", time = " << elapsed.count()
            << " sec" << std::endl;
}

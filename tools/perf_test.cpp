// Copyright 2020 Emmett Lalish
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

#include "manifold.h"

using namespace manifold;

int main(int argc, char **argv) {
  for (int i = 0; i < 8; ++i) {
    Manifold sphere = Manifold::Sphere(1, (8 << i) * 4);
    Manifold sphere2 = sphere;
    sphere2.Translate(glm::vec3(0.5));
    auto start = std::chrono::high_resolution_clock::now();
    Manifold diff = sphere - sphere2;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "nTri = " << sphere.NumTri() << ", time = " << elapsed.count()
              << " sec" << std::endl;
  }
}

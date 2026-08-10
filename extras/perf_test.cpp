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
#include <cstdlib>
#include <iostream>
#include <string>

#include "manifold/manifold.h"

using namespace manifold;

namespace {

constexpr int kNumSizes = 8;

void RunSizeIndex(int i) {
  Manifold sphere = Manifold::Sphere(1, (8 << i) * 4);
  Manifold sphere2 = sphere.Translate(vec3(0.5));
  auto start = std::chrono::high_resolution_clock::now();
  Manifold diff = sphere - sphere2;
  diff.NumTri();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "nTri = " << sphere.NumTri() << ", time = " << elapsed.count()
            << " sec" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--size-index") {
    int index = std::atoi(argv[2]);
    if (index < 0 || index >= kNumSizes) {
      std::cerr << "size index must be in range 0.." << (kNumSizes - 1)
                << std::endl;
      return 2;
    }
    RunSizeIndex(index);
    return 0;
  }

  if (argc != 1) {
    std::cerr << "Usage: " << argv[0] << " [--size-index 0.." << (kNumSizes - 1)
              << "]" << std::endl;
    return 2;
  }

  for (int i = 0; i < kNumSizes; ++i) {
    RunSizeIndex(i);
  }
}

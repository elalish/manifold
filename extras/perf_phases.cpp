// Copyright 2026 The Manifold Authors.
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
//
// Prints per-phase Boolean3 wall-clock timings on representative workloads.
// Requires MANIFOLD_TIMING=ON (or MANIFOLD_DEBUG=ON) so Timer::Print fires.
// Used for cancel-latency analysis and for identifying dominant phases.

#include <chrono>
#include <iostream>
#include <string>

#include "manifold/manifold.h"
#include "samples.h"

using namespace manifold;

namespace {
// Runs one boolean op of each kind (Add, Subtract, Intersect) against the
// same input pair. Each is a single boolean so Timer output is clean.
void BenchAll(const std::string& workload, const Manifold& a,
              const Manifold& b) {
  struct Op {
    const char* name;
    Manifold (*apply)(const Manifold&, const Manifold&);
  };
  const Op ops[] = {
      {"add", [](const Manifold& x, const Manifold& y) { return x + y; }},
      {"subtract", [](const Manifold& x, const Manifold& y) { return x - y; }},
      {"intersect", [](const Manifold& x, const Manifold& y) { return x ^ y; }},
  };
  for (const auto& op : ops) {
    std::cout << "=== " << workload << ": " << op.name
              << ", nTri(a) = " << a.NumTri() << " ===" << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    Manifold result = op.apply(a, b);
    result.NumTri();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "total = " << std::chrono::duration<double>(t1 - t0).count()
              << " sec\n"
              << std::endl;
  }
}
}  // namespace

int main(int argc, char** argv) {
  ManifoldParams().verbose = 2;

  // Sphere at escalating scales — uniform geometry, single boolean per op
  // so Timer output is readable.
  for (int i = 0; i < 8; ++i) {
    const int segments = (8 << i) * 4;
    Manifold sphere = Manifold::Sphere(1, segments);
    Manifold sphere2 = sphere.Translate(vec3(0.5));
    BenchAll("Sphere(" + std::to_string(segments) + ")", sphere, sphere2);
  }

  // Triangulator-heavy: Menger sponge. Forcing sponge.NumTri() first
  // separates the sponge's internal CSG construction from the measured
  // ops, so `total` reflects just each boolean.
  for (int level : {3, 4}) {
    Manifold sponge = MengerSponge(level);
    Manifold sponge2 = sponge.Translate(vec3(0.3));
    sponge.NumTri();  // force construction before timing the ops
    BenchAll("MengerSponge(" + std::to_string(level) + ")", sponge, sponge2);
  }
}

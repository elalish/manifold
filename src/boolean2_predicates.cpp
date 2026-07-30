// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Leaf primitives for the Boolean2 pipeline.

#include <algorithm>
#include <cmath>
#include <vector>

#include "boolean2.h"
#include "shared.h"
#include "utils.h"

namespace manifold {

// Centered-shoelace signed area of a closed polygon loop. Same FP trick
// as the per-face area computation in winding_filter.h: subtract a
// reference vert before multiplying so products stay at edge-length scale.
double SignedArea(const SimplePolygon& loop) {
  if (loop.size() < 3) return 0.0;
  const auto& r = loop[0];
  double sum = 0.0;
  for (size_t i = 0; i < loop.size(); ++i) {
    const auto& a = loop[i];
    const auto& b = loop[(i + 1) % loop.size()];
    sum += la::cross(a - r, b - r);
  }
  return 0.5 * sum;
}

double TotalSignedArea(const Polygons& polys) {
  double total = 0.0;
  for (const auto& loop : polys) total += SignedArea(loop);
  return total;
}

// Choose epsilon from the operation scale using Smith's rounded power-of-two
// length bound and the caller's adjustment budget.
double EpsilonFromScale(double L, int k_budget) {
  if (L <= 0) return 0;
  int expBits;
  std::frexp(L, &expBits);
  return std::ldexp((k_budget + 1) * kAlphaCoeff * kU, expBits);
}

double Coord(vec2 p, int axis) { return axis == 0 ? p.x : p.y; }

}  // namespace manifold

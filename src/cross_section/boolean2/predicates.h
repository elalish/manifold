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

#pragma once

#include <vector>

#include "manifold/common.h"

namespace manifold {
namespace boolean2 {

constexpr double kU = 1.110223024625156540423631668e-16;
constexpr double kAlphaCoeff = 12.37;

struct EdgeM {
  int v0, v1;
  int mult = 1;
};
using OutEdge = EdgeM;

double SignedArea(const SimplePolygon& loop);
double TotalSignedArea(const Polygons& polys);
double EpsilonFromScale(double L, int k_budget = 1000);
double Coord(vec2 p, int axis);
bool IntersectSegments(vec2 a0, vec2 a1, vec2 b0, vec2 b1, double eps,
                       vec2* out);

}  // namespace boolean2
}  // namespace manifold

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

enum class GraphOrderKind {
  ALessOrtho,
  AGreaterOrtho,
  EndpointTouch,
  NoProjectionOverlap
};

struct GraphSegment2D {
  vec2 p0;
  vec2 p1;
  // Stable fallback for geometrically identical ties (e.g. two input loops
  // sharing an edge). Must come from a deterministic source, not BVH pair
  // order.
  int stableEdgeId = -1;
};

struct GraphOrder2D {
  GraphOrderKind atMinProjection = GraphOrderKind::NoProjectionOverlap;
  GraphOrderKind atMaxProjection = GraphOrderKind::NoProjectionOverlap;
  bool coincidentOverlap = false;
  bool properCrossing = false;
};

double SignedArea(const SimplePolygon& loop);
double TotalSignedArea(const Polygons& polys);
double EpsilonFromScale(double L, int k_budget = 1000);
double Coord(vec2 p, int axis);
// Projection-frame graph order over a positive-width shared projection
// interval. `ALessOrtho`/`AGreaterOrtho` compare the coordinate orthogonal to
// `axis`, so for axis==1 they compare x over a y interval.
GraphOrder2D CompareProjectedOrder(const GraphSegment2D& a,
                                   const GraphSegment2D& b, int axis,
                                   double overlapL, double overlapR,
                                   double eps = 0.0);
bool IntersectSegments(const GraphSegment2D& a, const GraphSegment2D& b,
                       double eps, vec2* out);

}  // namespace boolean2
}  // namespace manifold

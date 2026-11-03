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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "manifold/common.h"
#include "manifold/optional_assert.h"
#include "parallel.h"

namespace manifold {

// Shared grain size for boolean2 parallel loops whose per-item work is small
// but non-trivial: incidence queries and merge broad-phase sweeps.
inline constexpr int kFineParallelGrainSize = 512;

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

struct Box2 {
  vec2 min = vec2(std::numeric_limits<double>::infinity());
  vec2 max = vec2(-std::numeric_limits<double>::infinity());

  Box2() = default;
  Box2(vec2 a, vec2 b)
      : min(std::min(a.x, b.x), std::min(a.y, b.y)),
        max(std::max(a.x, b.x), std::max(a.y, b.y)) {}

  Box2 Union(const Box2& b) const {
    return {vec2(std::min(min.x, b.min.x), std::min(min.y, b.min.y)),
            vec2(std::max(max.x, b.max.x), std::max(max.y, b.max.y))};
  }

  bool DoesOverlap(const Box2& b) const {
    return min.x <= b.max.x && max.x >= b.min.x && min.y <= b.max.y &&
           max.y >= b.min.y;
  }
};

struct VertexMerge {
  std::vector<int> inputVert2Merged;
  std::vector<vec2> verts;
};

VertexMerge MergeVerts(const std::vector<vec2>& in, double eps);
std::vector<EdgeM> RemapAndCollapse(const std::vector<EdgeM>& edges,
                                    const std::vector<int>& inputVert2Merged);

// For each edge, the sorted list of other edges' endpoints lying within eps of
// its interior. Splitting each edge at these vertices yields the two-vertex
// sub-edges the sweep operates on.
std::vector<std::vector<int>> BuildIncidenceLists(
    const std::vector<EdgeM>& edges, const std::vector<vec2>& verts,
    double eps);

struct Trace;

enum class WindRule {
  Add,
  Intersect,
  EvenOdd,
  NonZero,
};

// Smith sweep-line arrangement + winding over the collapsed input edges: a
// maintained-status Bentley-Ottmann sweep discovers all crossings and
// vertex-on-edge incidences and applies the 7.6.2 block rule (a dense
// near-concurrence collapses to one shared vertex), then a second,
// forced-through winding sweep emits the retained boundary under `rule`.
// `verts` is extended with the constructed crossing vertices the emitted edges
// reference.
std::vector<OutEdge> SweepWinding(const std::vector<EdgeM>& edges,
                                  std::vector<vec2>& verts, WindRule rule);

struct OverlapResult {
  std::vector<vec2> verts;
  std::vector<OutEdge> edges;
  std::vector<int> inputVert2Merged;
  int numMergedVerts;
};

// `eps` is the per-op FP-noise bound (3D: Impl::epsilon_). The arrangement is
// eps-only; tolerance-scale decimation is Simplify's job, as in boolean3.
OverlapResult RemoveOverlaps2D(const std::vector<vec2>& vertsIn,
                               const std::vector<EdgeM>& edgesIn, double eps,
                               bool debug = false,
                               WindRule pred = WindRule::Add,
                               Trace* trace = nullptr);

double InferEps(const Polygons& a, const Polygons& b);

std::pair<std::vector<vec2>, std::vector<EdgeM>> PolygonsToInput(
    const Polygons& polys);
Polygons OutEdgesToPolygons(const std::vector<vec2>& verts,
                            const std::vector<OutEdge>& edges);

// Regularize one polygon set at machine-scale eps under `rule`, which decides
// how the input's own winding is read: Positive (Add), EvenOdd, or NonZero.
// The output is normal-oriented and positive, so this is the only place the
// rule is visible. No default: every caller states which rule it wants, since
// only CrossSection construction has a reason to pick anything but Add.
// Fill-rule application, not tolerance decimation.
Polygons ApplyFillRule(const Polygons& polys, double eps, WindRule rule);
Polygons Boolean2D(const Polygons& a, const Polygons& b, OpType op,
                   double eps = 0.0);

// Polygon offset backing CrossSection::Offset.
Polygons Offset(const Polygons& in, double delta, JoinType jt,
                double miterLimit = 2.0, int circularSegments = 0);

// Group regularized simple loops into outer-ring components with their
// directly contained holes, backing CrossSection::Decompose.
std::vector<Polygons> DecomposeByContainment(const Polygons& polys);

}  // namespace manifold

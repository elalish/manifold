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

#include "collider.h"
#include "manifold/common.h"
#include "manifold/optional_assert.h"
#include "parallel.h"

namespace manifold {

// Shared grain size for boolean2 parallel loops whose per-item work is small
// but non-trivial: BVH traversal, pair narrow tests, and merge broad-phase
// sweeps. Radix-tree construction uses a coarser build-specific grain because
// its per-index work is much smaller.
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

inline constexpr int kEdgePairBvhThreshold = 1024;
// The radix-tree BVH is binary and has at most 32 Morton-code bits plus
// 32 index tie-breaker bits, so a depth-first traversal stack of 64 is enough.
inline constexpr int kBvhTraversalStackCapacity = 64;

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

  vec2 Center() const { return (min + max) * 0.5; }

  bool DoesOverlap(const Box2& b) const {
    return min.x <= b.max.x && max.x >= b.min.x && min.y <= b.max.y &&
           max.y >= b.min.y;
  }
};

Box2 BoxOf2DEdge(vec2 p0, vec2 p1, double eps);
uint32_t MortonCode2(vec2 position, Box2 bBox);

struct BVH {
  std::vector<Box2> nodeBBox;
  std::vector<std::pair<int, int>> internalChildren;
  std::vector<int> leafToOrig;

  bool Empty() const { return internalChildren.empty(); }
};

BVH BVHBuildFromBoxes(const std::vector<Box2>& boxes);

template <typename Recorder, typename F>
inline void BVHCollisions(const BVH& bvh, Recorder& recorder, F&& queryBox,
                          int n, bool parallel) {
  using namespace collider_internal;
  if (bvh.Empty()) return;
  auto collideOne = [&](int queryIdx) {
    const Box2 query = queryBox(queryIdx);
    int stack[kBvhTraversalStackCapacity];
    int top = -1;
    int node = kRoot;
    auto& local = recorder.local();
    while (true) {
      const int internal = Node2Internal(node);
      const int child1 = bvh.internalChildren[internal].first;
      const int child2 = bvh.internalChildren[internal].second;
      auto recordOrTraverse = [&](int child) {
        const bool overlaps = bvh.nodeBBox[child].DoesOverlap(query);
        if (overlaps && IsLeaf(child)) {
          const int leafIdx = Node2Leaf(child);
          recorder.record(queryIdx, leafIdx, local);
        }
        return overlaps && IsInternal(child);
      };
      const bool traverse1 = recordOrTraverse(child1);
      const bool traverse2 = recordOrTraverse(child2);
      if (!traverse1 && !traverse2) {
        if (top < 0) break;
        node = stack[top--];
      } else {
        node = traverse1 ? child1 : child2;
        if (traverse1 && traverse2) {
          DEBUG_ASSERT(top + 1 < kBvhTraversalStackCapacity, logicErr,
                       "Boolean2 BVH traversal stack overflow");
          stack[++top] = child2;
        }
      }
    }
  };
  manifold::for_each_n(
      parallel ? autoPolicy(n, kFineParallelGrainSize) : ExecutionPolicy::Seq,
      countAt(0), n, collideOne);
}

template <typename F>
inline void CollidePairs(const BVH& bvh, const std::vector<Box2>& queries,
                         F&& f) {
  if (bvh.Empty() || queries.empty()) return;
  auto adapter = [&](int qi, int leafIdx) { f(qi, bvh.leafToOrig[leafIdx]); };
  auto recorder = MakeSimpleRecorder(adapter);
  auto qf = [&](int i) { return queries[i]; };
  BVHCollisions(bvh, recorder, qf, static_cast<int>(queries.size()),
                /*parallel=*/false);
}

struct VertexMerge {
  std::vector<int> inputVert2Merged;
  std::vector<vec2> verts;
};

VertexMerge MergeVerts(const std::vector<vec2>& in, double eps);
std::vector<EdgeM> RemapAndCollapse(const std::vector<EdgeM>& edges,
                                    const std::vector<int>& inputVert2Merged);

// Incidence narrow phase over broad-phase edge pairs: for each edge, the sorted
// list of other-edge endpoints lying within eps of its interior. Splitting each
// edge at these vertices yields the two-vertex sub-edges the sweep operates on.
std::vector<std::vector<int>> BuildIncidenceLists(
    const std::vector<EdgeM>& edges, const std::vector<vec2>& verts, double eps,
    const std::vector<std::pair<int, int>>& pairs);

void CollectIntersectionPairs(const std::vector<EdgeM>& edges,
                              const std::vector<vec2>& verts, double eps,
                              const std::vector<Box2>& edgeBoxes,
                              const BVH& bvh,
                              std::vector<std::pair<int, int>>& pairs);

struct Trace;

enum class WindRule {
  Add,
  Intersect,
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

// Regularize one polygon set under the Positive (Add) winding rule at
// machine-scale eps. Fill-rule application, not tolerance decimation.
Polygons ApplyFillRule(const Polygons& polys, double eps);
Polygons Boolean2D(const Polygons& a, const Polygons& b, OpType op,
                   double eps = 0.0);

// Polygon offset backing CrossSection::Offset.
Polygons Offset(const Polygons& in, double delta, JoinType jt,
                double miterLimit = 2.0, int circularSegments = 0);

// Group regularized simple loops into outer-ring components with their
// directly contained holes, backing CrossSection::Decompose.
std::vector<Polygons> DecomposeByContainment(const Polygons& polys);

}  // namespace manifold

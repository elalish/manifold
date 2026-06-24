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
                       double eps, vec2& out);

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

Box2 BoxOf2DPoint(vec2 p, double eps);
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

struct CanonEdge {
  int vMin, vMax;
  int mult;
};

struct CanonicalSubEdges {
  std::vector<CanonEdge> edges;

  inline void Add(int v0, int v1, int mult) {
    if (v0 == v1) return;
    int vMin = std::min(v0, v1);
    int vMax = std::max(v0, v1);
    int signedMult = (v0 < v1) ? mult : -mult;
    edges.push_back({vMin, vMax, signedMult});
  }

  inline void Finalize() {
    manifold::stable_sort(edges.begin(), edges.end(),
                          [](const CanonEdge& a, const CanonEdge& b) {
                            if (a.vMin != b.vMin) return a.vMin < b.vMin;
                            return a.vMax < b.vMax;
                          });
    size_t w = 0;
    for (size_t r = 0; r < edges.size();) {
      size_t k = r;
      int sumMult = 0;
      while (k < edges.size() && edges[k].vMin == edges[r].vMin &&
             edges[k].vMax == edges[r].vMax) {
        sumMult += edges[k].mult;
        ++k;
      }
      if (sumMult != 0) {
        edges[w] = {edges[r].vMin, edges[r].vMax, sumMult};
        ++w;
      }
      r = k;
    }
    edges.resize(w);
  }
};

// Split each directed input edge at the vertices in `lists[e]`, then merge
// matching undirected sub-edges by summing signed multiplicities.
//
// `edges` are the collapsed input edges. `lists[e]` is the sorted list of
// interior vertices that split edge `e`. The returned `edges` vector is sorted
// by `(vMin, vMax)` and omits sub-edges whose summed multiplicity is zero.
CanonicalSubEdges Canonicalize(const std::vector<EdgeM>& edges,
                               const std::vector<std::vector<int>>& lists);

struct IntersectionPoint {
  int i;
  int j;
  vec2 p;
};

struct VertexMerge {
  std::vector<int> inputVert2Merged;
  std::vector<vec2> verts;
};

VertexMerge MergeVerts(const std::vector<vec2>& in, double eps);
bool VESetContains(const std::vector<int>& vec, int x);
void VESetInsert(std::vector<int>& vec, int x);
std::vector<EdgeM> RemapAndCollapse(const std::vector<EdgeM>& edges,
                                    const std::vector<int>& inputVert2Merged);

struct NarrowPhaseResult {
  std::vector<std::vector<int>> lists;
  std::vector<IntersectionPoint> intersections;
};

// Combined narrow phase over broad-phase edge pairs. Produces sorted
// edge-vertex split lists and independent proper edge-edge intersection
// candidates without mutating `verts` or `edges`; serial vs TBB execution is
// an internal thresholded implementation detail. With `findCrossings` false,
// only the split lists are produced and `intersections` is left empty.
NarrowPhaseResult BuildListsAndFindIntersections(
    const std::vector<EdgeM>& edges, const std::vector<vec2>& verts, double eps,
    const std::vector<std::pair<int, int>>& pairs, bool findCrossings = true);

void CollectIntersectionPairs(const std::vector<EdgeM>& edges,
                              const std::vector<vec2>& verts, double eps,
                              const std::vector<Box2>& edgeBoxes,
                              const BVH& bvh,
                              std::vector<std::pair<int, int>>& pairs);

// Serially materialize precomputed proper intersections into caller-owned
// containers. `verts` and `lists` are taken by value so callers can move in the
// post-narrow-phase state; the returned fields are those same containers after
// appending/snapping intersection vertices and updating edge split lists.
struct IntersectionInsertion {
  std::vector<vec2> verts;
  std::vector<std::vector<int>> lists;
};

IntersectionInsertion FindAndInsertIntersections(
    const std::vector<EdgeM>& edges, std::vector<vec2> verts,
    std::vector<std::vector<int>> lists, double eps,
    const std::vector<Box2>& edgeBoxes, const BVH& bvh,
    const std::vector<IntersectionPoint>& precomputedIntersections);

struct Trace;

enum class WindRule {
  Add,
  Intersect,
};

// Per-edge winding filter: for each canonical sub-edge, evaluate the winding of
// the face just left of vMin->vMax at the start vertex (a +x ray-cast under a
// symbolic perturbation into that face), take the right winding as leftW-mult,
// and keep the edge iff the rule disagrees across it. Correct only on a true
// arrangement. The ray-cast reuses boolean2's BVH over the canonical sub-edges,
// so the pass is ~O(E log E) amortized rather than O(E^2).
std::vector<OutEdge> FilterByWinding(const CanonicalSubEdges& canon,
                                     const std::vector<vec2>& verts,
                                     WindRule rule = WindRule::Add);

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

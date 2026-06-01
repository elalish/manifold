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
// Edge-edge intersection discovery: broad phase collects candidate edge
// pairs; the edge-vertex narrow phase precomputes proper intersections;
// FindAndInsertIntersections then snaps/inserts them into both edges'
// parameter lists. Eager-propagation re-sweeps any vert that snapped onto
// its k-th edge to detect k+1 incidences in one pass.

#include "intersections.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "../../parallel.h"
#include "bvh.h"
#include "predicates.h"
#include "vertex_merge.h"

namespace manifold {
namespace boolean2 {

namespace {

// Encode (first, second) as uint64 and run manifold::stable_sort, which
// dispatches to the LSB-radix specialization for integral types - much
// faster than the comparator path for pair<int,int>. Casts through
// uint32_t preserve non-negative edge indices and put `first` in the
// high bits so int64 order matches lexicographic pair order.
void RadixSortPairs(std::vector<std::pair<int, int>>* pairs) {
  const size_t n = pairs->size();
  if (n < 2) return;
  thread_local static std::vector<uint64_t> encoded;
  encoded.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const auto& pr = (*pairs)[i];
    encoded[i] =
        (static_cast<uint64_t>(static_cast<uint32_t>(pr.first)) << 32) |
        static_cast<uint32_t>(pr.second);
  }
  manifold::stable_sort(encoded.begin(), encoded.end());
  for (size_t i = 0; i < n; ++i) {
    (*pairs)[i] = {static_cast<int>(encoded[i] >> 32),
                   static_cast<int>(encoded[i] & 0xFFFFFFFFu)};
  }
}

void SortSmallInts(std::vector<int>* values) {
  if (values->size() < 32) {
    for (size_t i = 1; i < values->size(); ++i) {
      const int x = (*values)[i];
      size_t j = i;
      while (j > 0 && x < (*values)[j - 1]) {
        (*values)[j] = (*values)[j - 1];
        --j;
      }
      (*values)[j] = x;
    }
  } else {
    manifold::stable_sort(values->begin(), values->end());
  }
}

// Drop shared-endpoint pairs when both non-shared endpoints are more than eps
// from the opposite edge line; then no T-junction is possible.
bool SharedEndpointSafelySkippable(const EdgeM& a, const EdgeM& b,
                                   const std::vector<vec2>& verts, double eps) {
  int vS, wA, wB;
  if (a.v0 == b.v0) {
    vS = a.v0;
    wA = a.v1;
    wB = b.v1;
  } else if (a.v0 == b.v1) {
    vS = a.v0;
    wA = a.v1;
    wB = b.v0;
  } else if (a.v1 == b.v0) {
    vS = a.v1;
    wA = a.v0;
    wB = b.v1;
  } else if (a.v1 == b.v1) {
    vS = a.v1;
    wA = a.v0;
    wB = b.v0;
  } else {
    return false;  // not shared
  }
  const vec2& pS = verts[vS];
  const vec2& pA = verts[wA];
  const vec2& pB = verts[wB];
  const double dAx = pA.x - pS.x;
  const double dAy = pA.y - pS.y;
  const double dBx = pB.x - pS.x;
  const double dBy = pB.y - pS.y;
  const double cross = dAx * dBy - dAy * dBx;
  const double cross2 = cross * cross;
  const double lenA2 = dAx * dAx + dAy * dAy;
  const double lenB2 = dBx * dBx + dBy * dBy;
  const double eps2 = eps * eps;
  // Drop iff w_A is > eps from line B AND w_B is > eps from line A.
  // The check is symmetric in the |cross| numerator, so we only have
  // to compare cross^2 against eps^2 * max(lenA2, lenB2).
  return cross2 > eps2 * std::max(lenA2, lenB2);
}

// Broad phase: find overlapping edge AABBs, drop safe shared-endpoint pairs,
// and return lex-sorted pairs for deterministic insertion.
#if (MANIFOLD_PAR == 1)
struct PairsRecorder {
  using Local = std::vector<std::pair<int, int>>;
  const std::vector<int>& leafToOrig;
  const std::vector<EdgeM>& edges;
  const std::vector<vec2>& verts;
  double eps;
  tbb::combinable<Local> tls;
  inline void record(int queryIdx, int leafIdx, Local& local) const {
    const int li = leafToOrig[leafIdx];
    if (queryIdx >= li) return;
    if (SharedEndpointSafelySkippable(edges[queryIdx], edges[li], verts, eps))
      return;
    local.emplace_back(queryIdx, li);
  }
  Local& local() { return tls.local(); }
};
#endif

}  // namespace

void CollectIntersectionPairs(const std::vector<EdgeM>& edges,
                              const std::vector<vec2>& verts, double eps,
                              const std::vector<Box2>& edgeBoxes,
                              const BVH& bvh,
                              std::vector<std::pair<int, int>>* pairs) {
  const int nE = static_cast<int>(edges.size());
  pairs->clear();
  if (bvh.leafToOrig.empty()) {
    thread_local static std::vector<int> order;
    thread_local static std::vector<std::vector<int>> byFirst;
    order.resize(nE);
    for (int i = 0; i < nE; ++i) order[i] = i;
    manifold::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      if (edgeBoxes[a].min.x != edgeBoxes[b].min.x)
        return edgeBoxes[a].min.x < edgeBoxes[b].min.x;
      return a < b;
    });
    if ((int)byFirst.size() < nE) byFirst.resize(nE);
    for (int i = 0; i < nE; ++i) byFirst[i].clear();
    int numPairs = 0;
    for (int oi = 0; oi < nE; ++oi) {
      const int i = order[oi];
      const auto& bi = edgeBoxes[i];
      for (int oj = oi + 1; oj < nE; ++oj) {
        const int j = order[oj];
        const auto& bj = edgeBoxes[j];
        if (bj.min.x > bi.max.x) break;
        if (bi.min.y <= bj.max.y && bi.max.y >= bj.min.y) {
          if (SharedEndpointSafelySkippable(edges[i], edges[j], verts, eps))
            continue;
          const int first = std::min(i, j);
          const int second = std::max(i, j);
          byFirst[first].push_back(second);
          ++numPairs;
        }
      }
    }
    pairs->reserve(numPairs);
    for (int first = 0; first < nE; ++first) {
      auto& seconds = byFirst[first];
      if (seconds.empty()) continue;
      SortSmallInts(&seconds);
      for (int second : seconds) pairs->emplace_back(first, second);
    }
    return;
  }
#if (MANIFOLD_PAR == 1)
  PairsRecorder rec{bvh.leafToOrig, edges, verts, eps, {}};
  auto qf = [&](int i) { return edgeBoxes[i]; };
  BVHCollisions(bvh, rec, qf, nE, /*parallel=*/true);
  rec.tls.combine_each([&](const auto& localPairs) {
    pairs->insert(pairs->end(), localPairs.begin(), localPairs.end());
  });
  RadixSortPairs(pairs);
#else
  CollidePairs(bvh, edgeBoxes, [&](int qi, int li) {
    if (qi >= li) return;
    if (SharedEndpointSafelySkippable(edges[qi], edges[li], verts, eps)) return;
    pairs->emplace_back(qi, li);
  });
  RadixSortPairs(pairs);
#endif
}

void FindAndInsertIntersections(
    const std::vector<EdgeM>& edges, std::vector<vec2>* verts,
    std::vector<std::vector<int>>* lists,
    std::vector<std::vector<int>>* vertEdges, double eps,
    const std::vector<Box2>& edgeBoxes, const BVH& bvh,
    const std::vector<IntersectionPoint>& precomputedIntersections) {
  const int nE = static_cast<int>(edges.size());
  const double eps2 = eps * eps;
  vertEdges->resize(verts->size());
  const int origNumVerts = static_cast<int>(verts->size());

  // Serial snap+insert pass. Each iteration may push to `verts` and
  // mutate `lists[*]`, so subsequent iterations see the latest state.
  // Avoid structured-binding capture (C++20-only) by naming locals.
  for (const IntersectionPoint& ix : precomputedIntersections) {
    const int i = ix.i;
    const int j = ix.j;
    const vec2 p = ix.p;
    // Snap: is p within eps of any existing vert? Search the union of
    // (i,j)'s endpoints and existing list members of i and j.
    auto nearVert = [&](int candidate) -> bool {
      vec2 d = p - (*verts)[candidate];
      return dot(d, d) <= eps2;
    };
    int snapTo = -1;
    for (int v : {edges[i].v0, edges[i].v1, edges[j].v0, edges[j].v1}) {
      if (nearVert(v)) {
        snapTo = v;
        break;
      }
    }
    if (snapTo < 0) {
      for (int v : (*lists)[i]) {
        if (nearVert(v)) {
          snapTo = v;
          break;
        }
      }
    }
    if (snapTo < 0) {
      for (int v : (*lists)[j]) {
        if (nearVert(v)) {
          snapTo = v;
          break;
        }
      }
    }
    int vNew;
    if (snapTo >= 0) {
      vNew = snapTo;
    } else {
      vNew = static_cast<int>(verts->size());
      verts->push_back(p);
      vertEdges->emplace_back();
    }
    VESetInsert(&(*vertEdges)[vNew], i);
    VESetInsert(&(*vertEdges)[vNew], j);
    auto insertSorted = [&](int eIdx) {
      if (vNew == edges[eIdx].v0 || vNew == edges[eIdx].v1) return;
      auto& lst = (*lists)[eIdx];
      if (VESetContains(lst, vNew)) return;
      vec2 a = (*verts)[edges[eIdx].v0];
      vec2 b = (*verts)[edges[eIdx].v1];
      vec2 ab = b - a;
      double abLen2 = dot(ab, ab);
      if (abLen2 == 0) return;
      double tNew = dot(p - a, ab) / abLen2;
      auto pos =
          std::lower_bound(lst.begin(), lst.end(), tNew, [&](int v, double t) {
            double tv = dot((*verts)[v] - a, ab) / abLen2;
            return tv < t;
          });
      if (pos == lst.end() || *pos != vNew) lst.insert(pos, vNew);
    };
    insertSorted(i);
    insertSorted(j);
  }

  // Eager propagation: after all independent edge-pair intersections are
  // inserted, add each new intersection vertex to every other edge it
  // geometrically splits. Otherwise a later canonical sub-edge can pass through
  // a new vertex without being split there, leaving the halfedge arrangement
  // dependent on tiny angular-sort differences.
  if ((int)verts->size() == origNumVerts) return;
  const int numNewVerts = static_cast<int>(verts->size()) - origNumVerts;

  // Per-(qi, eIdx) propagation step. Same logic for BVH and brute-force
  // broad phases.
  auto propagateNarrow = [&](int qi, int eIdx) {
    const int v = origNumVerts + qi;
    if (v == edges[eIdx].v0 || v == edges[eIdx].v1) return;
    if (VESetContains((*vertEdges)[v], eIdx)) return;  // already incident
    const vec2 a = (*verts)[edges[eIdx].v0];
    const vec2 b = (*verts)[edges[eIdx].v1];
    const vec2 ab = b - a;
    const double abLen2 = dot(ab, ab);
    if (abLen2 == 0) return;
    const vec2 p = (*verts)[v];
    const double t = dot(p - a, ab) / abLen2;
    if (t <= 0 || t >= 1) return;
    const vec2 closest = a + ab * t;
    const vec2 d = p - closest;
    if (dot(d, d) > eps2) return;
    auto& lst = (*lists)[eIdx];
    auto pos =
        std::lower_bound(lst.begin(), lst.end(), t, [&](int vv, double tQ) {
          double tv = dot((*verts)[vv] - a, ab) / abLen2;
          return tv < tQ;
        });
    if (pos == lst.end() || *pos != v) lst.insert(pos, v);
    VESetInsert(&(*vertEdges)[v], eIdx);
  };
  if (bvh.leafToOrig.empty()) {
    for (int qi = 0; qi < numNewVerts; ++qi) {
      const Box2 queryBox = BoxOf2DPoint((*verts)[origNumVerts + qi], eps);
      for (int e = 0; e < nE; ++e) {
        if (queryBox.DoesOverlap(edgeBoxes[e])) propagateNarrow(qi, e);
      }
    }
  } else {
    auto adapter = [&](int qi, int leafIdx) {
      propagateNarrow(qi, bvh.leafToOrig[leafIdx]);
    };
    auto recorder = MakeSimpleRecorder(adapter);
    auto qf = [&](int qi) {
      return BoxOf2DPoint((*verts)[origNumVerts + qi], eps);
    };
    BVHCollisions(bvh, recorder, qf, numNewVerts, /*parallel=*/false);
  }
}

}  // namespace boolean2
}  // namespace manifold

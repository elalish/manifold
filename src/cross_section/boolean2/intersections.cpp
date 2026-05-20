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
// pairs; FindAndInsertIntersections runs the per-pair narrow phase
// (IntersectSegments + snap-to-existing-vert + sorted insertion into both
// edges' parameter lists). Eager-propagation re-sweeps any vert that
// snapped onto its k-th edge to detect k+1 incidences in one pass.

#include "intersections.h"

#include <algorithm>
#include <chrono>
#include <utility>
#include <vector>

#include "../../parallel.h"
#include "bvh.h"
#include "diagnostics.h"
#include "edge_vert_lists.h"
#include "parallel_policy.h"
#include "predicates.h"
#include "vertex_merge.h"

namespace manifold {
namespace boolean2 {

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

// Broad-phase filter for shared-endpoint pairs: when two edges share a vertex
// and are not near-collinear there, no T-junction is possible. Returns true
// when the pair can be safely dropped.
//
// Distance from the non-shared endpoint w_A to the line through
// (v_shared, w_B) is |cross(w_A - v, w_B - v)| / |w_B - v|; T-junction
// requires distance <= eps, i.e. cross^2 <= eps^2 * |w_B - v|^2. The
// symmetric condition tests w_B against A's line; both must fail for
// the pair to be safely dropped, which collapses to a single compare
// against eps^2 * max(|d_A|^2, |d_B|^2).
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

// Broad phase: find candidate (i, j) edge pairs whose AABBs overlap.
// Output is sorted lex-ascending so downstream insertion is
// deterministic. Shared-endpoint pairs that fail the collinearity
// gate (above) are dropped here.
//
// For the BVH path, uses boolean2's parallel-capable 2D collision walk
// with a thread-local accumulator (tbb::combinable). Each thread emits
// pairs into its own vector; combine_each merges them, then a single
// sort re-establishes lex order. This is used only when the candidate
// count justifies the parallel overhead.
#if (MANIFOLD_PAR == 1)
namespace intersection_pair_recorder {
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
}  // namespace intersection_pair_recorder
#endif

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
  intersection_pair_recorder::PairsRecorder rec{
      bvh.leafToOrig, edges, verts, eps, {}};
  auto qf = [&](int i) { return edgeBoxes[i]; };
  BVHCollisions<false>(bvh, rec, qf, nE, /*parallel=*/true);
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

// Internal: takes either `pairs` (and computes intersections inline) or
// `precomputedIntersections` (when the caller has already run the
// parallel compute, e.g. fused into the per-pair narrow pass). Exactly
// one of the two should be supplied.
void FindAndInsertIntersectionsImpl(
    const std::vector<EdgeM>& edges, std::vector<vec2>* verts,
    std::vector<std::vector<int>>* lists,
    std::vector<std::vector<int>>* vertEdges, double eps,
    const std::vector<Box2>& edgeBoxes, const BVH& bvh,
    const std::vector<std::pair<int, int>>* pairsIn,
    const std::vector<IntersectionPoint>* precomputedIntersections) {
  using Clock = std::chrono::steady_clock;
  auto Ns = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
  };
  const bool timing = TimingEnabled();
  auto tNarrowStart = timing ? Clock::now() : Clock::time_point{};
  const int nE = static_cast<int>(edges.size());
  const double eps2 = eps * eps;
  vertEdges->resize(verts->size());
  const int origNumVerts = static_cast<int>(verts->size());

  // Compute intersections independently, then snap/insert in a deterministic
  // serial pass because insertion mutates shared vertex and edge lists.
  using PairIx = IntersectionPoint;
  thread_local static std::vector<PairIx> intersections;
  // If caller already computed intersections via the fused parallel
  // pass, use them directly; otherwise compute below.
  if (precomputedIntersections != nullptr) {
    // Just point at the supplied data; the serial pass below reads it.
    // We can't quite assign-from-ref since the loop reads `intersections`
    // by name; copy is cheap relative to recomputing.
    intersections = *precomputedIntersections;
  } else {
    intersections.clear();
    const auto& pairs = *pairsIn;
    // Per-pair work: compute the segment-segment intersection point, gated
    // on the pair not sharing an endpoint. Shared-endpoint pairs would
    // produce a bogus intersection at the shared vertex itself; their
    // vertex-on-edge interactions are handled separately by the narrow
    // phase in BuildEdgeVertListsFromEdgePairs.
    auto pairIx = [&](size_t k, std::vector<PairIx>& out) {
      const int i = pairs[k].first;
      const int j = pairs[k].second;
      const auto& ei = edges[i];
      const auto& ej = edges[j];
      if (ei.v0 == ej.v0 || ei.v0 == ej.v1 || ei.v1 == ej.v0 || ei.v1 == ej.v1)
        return;
      vec2 p;
      if (!IntersectSegments((*verts)[ei.v0], (*verts)[ei.v1], (*verts)[ej.v0],
                             (*verts)[ej.v1], eps, &p))
        return;
      out.push_back({i, j, p});
    };
    // Match the fused narrow-pass threshold used by the in-tree driver. Direct
    // callers that bypass the fused path still use the same parallel policy.
    const size_t kParallelIxMin = kFusedNarrowParallelMin;
#if (MANIFOLD_PAR == 1)
    if (pairs.size() >= kParallelIxMin) {
      tbb::combinable<std::vector<PairIx>> tls;
      manifold::for_each_n(autoPolicy(pairs.size(), kFineParallelGrainSize),
                           countAt(size_t{0}), pairs.size(),
                           [&](size_t k) { pairIx(k, tls.local()); });
      tls.combine_each([&](const std::vector<PairIx>& l) {
        intersections.insert(intersections.end(), l.begin(), l.end());
      });
      // Re-sort by (i, j) so the serial pass below sees a deterministic
      // order (matches the order the sequential path would have produced).
      manifold::stable_sort(intersections.begin(), intersections.end(),
                            [](const PairIx& a, const PairIx& b) {
                              if (a.i != b.i) return a.i < b.i;
                              return a.j < b.j;
                            });
    } else
#endif
    {
      for (size_t k = 0; k < pairs.size(); ++k) pairIx(k, intersections);
    }
  }  // end else (precomputedIntersections == nullptr)

  // Serial snap+insert pass. Each iteration may push to `verts` and
  // mutate `lists[*]`, so subsequent iterations see the latest state.
  // Avoid structured-binding capture (C++20-only) by naming locals.
  for (const PairIx& ix : intersections) {
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
      vec2 a = (*verts)[edges[eIdx].v0];
      vec2 b = (*verts)[edges[eIdx].v1];
      vec2 ab = b - a;
      double abLen2 = dot(ab, ab);
      if (abLen2 == 0) return;
      double tNew = dot(p - a, ab) / abLen2;
      auto& lst = (*lists)[eIdx];
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

  // Eager propagation pass (2D analog of boolean_result.cpp::AddNewEdgeVerts).
  // The pair-by-pair loop above only adds each new intersection vert to the
  // two edges in its source pair. But k>=3 input edges can be concurrent at
  // one true point: e.g., edges A, B, C all crossing at P generates three
  // pair-intersections (A,B), (A,C), (B,C); the snap-to-existing logic
  // catches duplicates within eps when they share an edge in the iteration
  // order, but misses the "no shared edge" case (A,B vs E,F both at P).
  // Without propagation, edge C ends up with no on-edge vert at P even though
  // P geometrically lies on C; canonicalization then emits C as one un-split
  // sub-edge and the resulting DCEL has the wrong incidence at P.
  //
  // When newly-created intersection verts contain near-duplicates, query the
  // edge BVH for all edges within eps and add those verts to their on-edge
  // lists. Non-duplicate new intersections already got their two incident
  // edges in the pair loop above; the expensive global propagation is for the
  // rare case where independent edge pairs discover the same geometric point.
  // Use the same generous same-point cutoff as the structural+geometric
  // re-merge in driver.h when deciding whether propagation is needed. The
  // propagation itself still uses eps-padded edge queries and the same exact
  // projection-distance gate as BuildEdgeVertLists.
  // Fast-path: if no new intersection verts were created in the main
  // loop, the propagation pass has nothing to do.
  auto tNarrowEnd = timing ? Clock::now() : Clock::time_point{};
  if (timing)
    GlobalPhases().intersectionNarrowNs.fetch_add(Ns(tNarrowStart, tNarrowEnd),
                                                  std::memory_order_relaxed);
  if ((int)verts->size() == origNumVerts) return;
  auto tPropStart = timing ? Clock::now() : Clock::time_point{};
  const int numNewVerts = static_cast<int>(verts->size()) - origNumVerts;
  if (timing)
    GlobalPhases().propagationCalls.fetch_add(1, std::memory_order_relaxed);
  const double propagationDupThresh = kIntersectionMergeEpsFactor * eps;
  const double propagationDupThresh2 =
      propagationDupThresh * propagationDupThresh;

  auto hasNearDuplicateNewVert = [&]() {
    if (numNewVerts < 2) return false;
    if (numNewVerts < 16) {
      for (int i = 0; i < numNewVerts; ++i) {
        const vec2 pi = (*verts)[origNumVerts + i];
        for (int j = i + 1; j < numNewVerts; ++j) {
          const vec2 d = pi - (*verts)[origNumVerts + j];
          if (dot(d, d) <= propagationDupThresh2) return true;
        }
      }
      return false;
    }
    thread_local static std::vector<int> order;
    order.resize(numNewVerts);
    for (int i = 0; i < numNewVerts; ++i) order[i] = i;
    manifold::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      const vec2 pa = (*verts)[origNumVerts + a];
      const vec2 pb = (*verts)[origNumVerts + b];
      if (pa.x != pb.x) return pa.x < pb.x;
      if (pa.y != pb.y) return pa.y < pb.y;
      return a < b;
    });
    for (int oi = 0; oi < numNewVerts; ++oi) {
      const int i = order[oi];
      const vec2 pi = (*verts)[origNumVerts + i];
      for (int oj = oi + 1; oj < numNewVerts; ++oj) {
        const int j = order[oj];
        const vec2 pj = (*verts)[origNumVerts + j];
        if (pj.x - pi.x > propagationDupThresh) break;
        if (std::abs(pj.y - pi.y) > propagationDupThresh) continue;
        const vec2 d = pi - pj;
        if (dot(d, d) <= propagationDupThresh2) return true;
      }
    }
    return false;
  };

  if (!hasNearDuplicateNewVert()) {
    if (timing) {
      auto& P = GlobalPhases();
      P.propagationSkippedNoNearDup.fetch_add(1, std::memory_order_relaxed);
      P.intersectionPropagationNs.fetch_add(Ns(tPropStart, Clock::now()),
                                            std::memory_order_relaxed);
    }
    return;
  }

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
    BVHCollisions<false>(bvh, recorder, qf, numNewVerts, /*parallel=*/false);
  }
  auto tPropEnd = timing ? Clock::now() : Clock::time_point{};
  if (timing)
    GlobalPhases().intersectionPropagationNs.fetch_add(
        Ns(tPropStart, tPropEnd), std::memory_order_relaxed);
}

// Public wrapper: compute intersections inline from `pairs`.
void FindAndInsertIntersections(const std::vector<EdgeM>& edges,
                                std::vector<vec2>* verts,
                                std::vector<std::vector<int>>* lists,
                                std::vector<std::vector<int>>* vertEdges,
                                double eps, const std::vector<Box2>& edgeBoxes,
                                const BVH& bvh,
                                const std::vector<std::pair<int, int>>& pairs) {
  FindAndInsertIntersectionsImpl(edges, verts, lists, vertEdges, eps, edgeBoxes,
                                 bvh, &pairs, nullptr);
}

#if (MANIFOLD_PAR == 1)
void FindAndInsertIntersectionsFromPrecomputed(
    const std::vector<EdgeM>& edges, std::vector<vec2>* verts,
    std::vector<std::vector<int>>* lists,
    std::vector<std::vector<int>>* vertEdges, double eps,
    const std::vector<Box2>& edgeBoxes, const BVH& bvh,
    const std::vector<IntersectionPoint>& precomputed) {
  FindAndInsertIntersectionsImpl(edges, verts, lists, vertEdges, eps, edgeBoxes,
                                 bvh, nullptr, &precomputed);
}
#endif

}  // namespace boolean2
}  // namespace manifold

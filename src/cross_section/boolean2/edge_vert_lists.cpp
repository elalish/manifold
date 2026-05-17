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
// BuildEdgeVertLists: for each input edge, the sorted-by-parameter list of
// vertices that lie within eps of the edge interior. The production driver
// derives candidates from the same edge-pair broad phase used for
// intersections; the standalone helper keeps direct vertex-vs-edge queries.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "bvh.h"
#include "diagnostics.h"
#include "predicates.h"
#include "vertex_merge.h"

namespace manifold {
namespace boolean2 {

// Per-edge ordered list of vertices within eps of the edge interior.
// Returns vertList[edgeIdx] = sorted list of vert indices along the edge.
//
// `edgeBoxes` and `bvh` are the eps-padded segment AABBs and the BVH
// built over them; they are passed in from the caller so this pass and
// the intersections pass can share a single build (the edges array
// doesn't change between them, so the boxes don't either).
std::vector<std::vector<int>> BuildEdgeVertLists(
    const std::vector<EdgeM>& edges, const std::vector<vec2>& verts, double eps,
    const std::vector<Box2>& edgeBoxes, const BVH& bvh) {
  const int nE = static_cast<int>(edges.size());
  const int nV = static_cast<int>(verts.size());
  const double eps2 = eps * eps;
  std::vector<std::vector<int>> lists(nE);
  const bool timing = TimingEnabled();
  int64_t candCount = 0;
  int64_t endpointRejects = 0;
  int64_t degenerateRejects = 0;
  int64_t tRangeRejects = 0;
  int64_t distanceRejects = 0;
  int64_t apexRejects = 0;
  int64_t hitCount = 0;
  const bool useBvh = !bvh.leafToOrig.empty();
  const bool useVertexBvh =
      !useBvh && nE >= 64 && static_cast<int64_t>(nE) * nV >= 100000;
  // BVH broad phase: edges as eps-padded segment AABBs, queried by vert
  // points (eps-padded boxes). Each candidate (edge, vert) pair runs the
  // exact projection test below. Per-edge `hits` are sorted by parameter
  // at the end so the result is independent of broad-phase visit order.
  // Per-calling-thread scratch; safe because this function owns the buffers for
  // the duration of the call.
  // `vertBoxes` is only filled for the brute-force path.
  thread_local static std::vector<Box2> vertBoxes;
  thread_local static std::vector<std::vector<int>> adj;
  // Single flat (edge, t, vert) buffer instead of vector-of-vectors.
  struct Hit {
    int e;
    double t;
    int v;
  };
  std::vector<Hit> flatHits;
  // Pre-compute per-edge geometry used by the narrow phase.
  struct EdgeG {
    vec2 a, ab;
    double abLen2;
  };
  std::vector<EdgeG> edgeG(nE);
  for (int e = 0; e < nE; ++e) {
    edgeG[e].a = verts[edges[e].v0];
    edgeG[e].ab = verts[edges[e].v1] - edgeG[e].a;
    edgeG[e].abLen2 = dot(edgeG[e].ab, edgeG[e].ab);
  }
  // Build vert -> neighbors adjacency only when the apex skip needs it. Most
  // calls have no non-collinear near-edge candidates, so eager construction is
  // avoidable work.
  bool adjBuilt = false;
  auto ensureAdj = [&]() {
    if (adjBuilt) return;
    if ((int)adj.size() < nV) adj.resize(nV);
    for (int i = 0; i < nV; ++i) adj[i].clear();
    for (const auto& e : edges) {
      VESetInsert(&adj[e.v0], e.v1);
      VESetInsert(&adj[e.v1], e.v0);
    }
    adjBuilt = true;
  };
  // Per-(v, e) narrow-phase test. Captures the same logic whether
  // candidates come from BVH or brute-force broad phase.
  auto narrow = [&](int v, int e) {
    if (timing) ++candCount;
    if (v == edges[e].v0 || v == edges[e].v1) {
      if (timing) ++endpointRejects;
      return;
    }
    const auto& g = edgeG[e];
    if (g.abLen2 == 0) {
      if (timing) ++degenerateRejects;
      return;
    }
    const vec2 ap = verts[v] - g.a;
    // Avoid the divide for the t-range gate: `t = dot/abLen2 in (0,1)` iff
    // `0 < dot < abLen2`. Reject before computing closest-point distance.
    const double dotAB = ap.x * g.ab.x + ap.y * g.ab.y;
    if (dotAB <= 0 || dotAB >= g.abLen2) {
      if (timing) ++tRangeRejects;
      return;
    }
    // Perpendicular squared distance from p to line ab is
    // (cross(ap, ab))^2 / abLen2. Compare to eps^2 by rearranging:
    // cross^2 <= eps^2 * abLen2. Avoids the closest-point computation.
    const double cross = ap.x * g.ab.y - ap.y * g.ab.x;
    const double cross2 = cross * cross;
    const double eps2_abLen2 = eps2 * g.abLen2;
    if (cross2 > eps2_abLen2) {
      if (timing) ++distanceRejects;
      return;
    }
    // Thin-triangle-apex skip: when V is connected to BOTH edge
    // endpoints by other edges, V is the apex of a triangle (V, e.v0,
    // e.v1) whose base is this edge. With non-tiny eps (large
    // displacement), the apex can fall within eps of its base;
    // without the skip, canonicalization cancels the apex-
    // split sub-edges against the triangle's other two sides,
    // producing empty output. Only-one-endpoint adjacency is normal
    // polygon-neighbor configuration, so we require BOTH to be
    // conservative.
    //
    // **Collinear escape**: when V is essentially ON the line through
    // (v0, v1) - i.e., the "triangle" is degenerate (zero area) - the
    // skip's premise doesn't apply, and dropping V here breaks
    // canonicalization on multi-polygon inputs with collinear
    // shared edges (the Manifold.Project test case). We define
    // "essentially collinear" as `cross^2 < eps^2 * abLen2 * 1e-4`,
    // i.e., perpendicular distance well below 1% of eps. Vertices
    // with meaningful perpendicular distance (apex-y) still get the
    // skip; vertices on the line get inserted.
    if (cross2 < eps2_abLen2 * 1e-4) {
      // Genuine on-edge case; bypass the apex skip.
    } else {
      ensureAdj();
      if (VESetContains(adj[v], edges[e].v0) &&
          VESetContains(adj[v], edges[e].v1)) {
        if (timing) ++apexRejects;
        return;
      }
    }
    // Only compute t (for sort key) on the ~rare survivor.
    if (timing) ++hitCount;
    flatHits.push_back({e, dotAB / g.abLen2, v});
  };
  if (useVertexBvh) {
    vertBoxes.resize(nV);
    for (int v = 0; v < nV; ++v) vertBoxes[v] = BoxOf2DPoint(verts[v], eps);
    const BVH vertBvh = BVHBuildFromBoxes(vertBoxes);
    auto adapter = [&](int e, int leafIdx) {
      narrow(vertBvh.leafToOrig[leafIdx], e);
    };
    auto recorder = MakeSimpleRecorder(adapter);
    auto qf = [&](int e) { return edgeBoxes[e]; };
    BVHCollisions<false>(vertBvh, recorder, qf, nE, /*parallel=*/false);
  } else if (!useBvh) {
    // Brute-force broad phase: O(V * E). Faster than building+querying
    // a BVH for small E (caller decided E < threshold).
    vertBoxes.resize(nV);
    for (int v = 0; v < nV; ++v) vertBoxes[v] = BoxOf2DPoint(verts[v], eps);
    for (int v = 0; v < nV; ++v) {
      for (int e = 0; e < nE; ++e) {
        if (vertBoxes[v].DoesOverlap(edgeBoxes[e])) narrow(v, e);
      }
    }
  } else {
    auto adapter = [&](int v, int leafIdx) {
      narrow(v, bvh.leafToOrig[leafIdx]);
    };
    auto recorder = MakeSimpleRecorder(adapter);
    auto qf = [&](int v) { return BoxOf2DPoint(verts[v], eps); };
    BVHCollisions<false>(bvh, recorder, qf, nV, /*parallel=*/false);
  }
  // Sort flat hits by (edge, t, v) and emit per-edge subsequences. The
  // tertiary `v` key makes output deterministic regardless of parallel
  // insertion order from the BVH walk above.
  manifold::stable_sort(flatHits.begin(), flatHits.end(),
                        [](const Hit& a, const Hit& b) {
                          if (a.e != b.e) return a.e < b.e;
                          if (a.t != b.t) return a.t < b.t;
                          return a.v < b.v;
                        });
  for (size_t i = 0; i < flatHits.size();) {
    const int e = flatHits[i].e;
    size_t j = i;
    while (j < flatHits.size() && flatHits[j].e == e) ++j;
    lists[e].reserve(j - i);
    for (size_t k = i; k < j; ++k) lists[e].push_back(flatHits[k].v);
    i = j;
  }
  if (timing) {
    auto& P = GlobalPhases();
    P.edgeVertCalls.fetch_add(1, std::memory_order_relaxed);
    if (useBvh) {
      P.edgeVertBvhCalls.fetch_add(1, std::memory_order_relaxed);
    } else if (useVertexBvh) {
      P.edgeVertVertexBvhCalls.fetch_add(1, std::memory_order_relaxed);
    } else {
      P.edgeVertBruteCalls.fetch_add(1, std::memory_order_relaxed);
    }
    P.edgeVertTotalEdges.fetch_add(nE, std::memory_order_relaxed);
    P.edgeVertTotalVerts.fetch_add(nV, std::memory_order_relaxed);
    P.edgeVertHitsFlat.fetch_add(static_cast<int64_t>(flatHits.size()),
                                 std::memory_order_relaxed);
    if (nE < 64) {
      P.edgeVertBucketLt64.fetch_add(1, std::memory_order_relaxed);
    } else if (nE < 256) {
      P.edgeVertBucketLt256.fetch_add(1, std::memory_order_relaxed);
    } else if (nE < 1024) {
      P.edgeVertBucketLt1024.fetch_add(1, std::memory_order_relaxed);
    } else {
      P.edgeVertBucketGe1024.fetch_add(1, std::memory_order_relaxed);
    }
    P.edgeVertCandidates.fetch_add(candCount, std::memory_order_relaxed);
    P.edgeVertEndpointRejects.fetch_add(endpointRejects,
                                        std::memory_order_relaxed);
    P.edgeVertDegenerateRejects.fetch_add(degenerateRejects,
                                          std::memory_order_relaxed);
    P.edgeVertTRangeRejects.fetch_add(tRangeRejects, std::memory_order_relaxed);
    P.edgeVertDistanceRejects.fetch_add(distanceRejects,
                                        std::memory_order_relaxed);
    P.edgeVertApexRejects.fetch_add(apexRejects, std::memory_order_relaxed);
    P.edgeVertHits.fetch_add(hitCount, std::memory_order_relaxed);
  }
  return lists;
}

std::vector<std::vector<int>> BuildEdgeVertListsFromEdgePairs(
    const std::vector<EdgeM>& edges, const std::vector<vec2>& verts, double eps,
    const std::vector<std::pair<int, int>>& pairs) {
  const int nE = static_cast<int>(edges.size());
  const int nV = static_cast<int>(verts.size());
  const double eps2 = eps * eps;
  std::vector<std::vector<int>> lists(nE);
  const bool timing = TimingEnabled();
  int64_t candCount = 0;
  int64_t endpointRejects = 0;
  int64_t degenerateRejects = 0;
  int64_t tRangeRejects = 0;
  int64_t distanceRejects = 0;
  int64_t apexRejects = 0;
  int64_t hitCount = 0;

  thread_local static std::vector<std::vector<int>> adj;

  struct Hit {
    int e;
    double t;
    int v;
  };
  std::vector<Hit> flatHits;

  struct EdgeG {
    vec2 a, ab;
    double abLen2;
  };
  std::vector<EdgeG> edgeG(nE);
  for (int e = 0; e < nE; ++e) {
    edgeG[e].a = verts[edges[e].v0];
    edgeG[e].ab = verts[edges[e].v1] - edgeG[e].a;
    edgeG[e].abLen2 = dot(edgeG[e].ab, edgeG[e].ab);
  }

  bool adjBuilt = false;
  auto ensureAdj = [&]() {
    if (adjBuilt) return;
    if ((int)adj.size() < nV) adj.resize(nV);
    for (int i = 0; i < nV; ++i) adj[i].clear();
    for (const auto& e : edges) {
      VESetInsert(&adj[e.v0], e.v1);
      VESetInsert(&adj[e.v1], e.v0);
    }
    adjBuilt = true;
  };

  auto narrow = [&](int v, int e) {
    if (timing) ++candCount;
    if (v == edges[e].v0 || v == edges[e].v1) {
      if (timing) ++endpointRejects;
      return;
    }
    const auto& g = edgeG[e];
    if (g.abLen2 == 0) {
      if (timing) ++degenerateRejects;
      return;
    }
    const vec2 ap = verts[v] - g.a;
    const double dotAB = ap.x * g.ab.x + ap.y * g.ab.y;
    if (dotAB <= 0 || dotAB >= g.abLen2) {
      if (timing) ++tRangeRejects;
      return;
    }
    const double cross = ap.x * g.ab.y - ap.y * g.ab.x;
    const double cross2 = cross * cross;
    const double eps2_abLen2 = eps2 * g.abLen2;
    if (cross2 > eps2_abLen2) {
      if (timing) ++distanceRejects;
      return;
    }
    if (cross2 >= eps2_abLen2 * 1e-4) {
      ensureAdj();
      if (VESetContains(adj[v], edges[e].v0) &&
          VESetContains(adj[v], edges[e].v1)) {
        if (timing) ++apexRejects;
        return;
      }
    }
    if (timing) ++hitCount;
    flatHits.push_back({e, dotAB / g.abLen2, v});
  };

  auto narrowIfNonEndpoint = [&](int v, int e) {
    if (v == edges[e].v0 || v == edges[e].v1) {
      if (timing) ++candCount;
      if (timing) ++endpointRejects;
      return;
    }
    narrow(v, e);
  };

#if (MANIFOLD_PAR == 1)
  // Gate the parallel pair-walk on a candidate-count threshold. Each pair
  // emits 4 (v, e) narrow tests; at 4096 pairs that's 16k candidates - enough
  // to amortize TBB task-spawn overhead. Below it, stay serial.
  constexpr size_t kPairsParallelMin = 4096;
  if (pairs.size() >= kPairsParallelMin) {
    // Bind `adjRef` below before launching workers; direct `adj` access inside
    // the lambda would resolve to each worker's thread-local copy.
    if ((int)adj.size() < nV) adj.resize(nV);
    for (int i = 0; i < nV; ++i) adj[i].clear();
    for (const auto& edge : edges) {
      VESetInsert(&adj[edge.v0], edge.v1);
      VESetInsert(&adj[edge.v1], edge.v0);
    }
    auto& adjRef = adj;
    struct Local {
      std::vector<Hit> hits;
      int64_t candCount = 0;
      int64_t endpointRejects = 0;
      int64_t degenerateRejects = 0;
      int64_t tRangeRejects = 0;
      int64_t distanceRejects = 0;
      int64_t apexRejects = 0;
      int64_t hitCount = 0;
    };
    tbb::combinable<Local> tls;
    auto narrowLocal = [&](int v, int e, Local& l) {
      if (timing) ++l.candCount;
      if (v == edges[e].v0 || v == edges[e].v1) {
        if (timing) ++l.endpointRejects;
        return;
      }
      const auto& g = edgeG[e];
      if (g.abLen2 == 0) {
        if (timing) ++l.degenerateRejects;
        return;
      }
      const vec2 ap = verts[v] - g.a;
      const double dotAB = ap.x * g.ab.x + ap.y * g.ab.y;
      if (dotAB <= 0 || dotAB >= g.abLen2) {
        if (timing) ++l.tRangeRejects;
        return;
      }
      const double cross = ap.x * g.ab.y - ap.y * g.ab.x;
      const double cross2 = cross * cross;
      const double eps2_abLen2 = eps2 * g.abLen2;
      if (cross2 > eps2_abLen2) {
        if (timing) ++l.distanceRejects;
        return;
      }
      if (cross2 >= eps2_abLen2 * 1e-4) {
        if (VESetContains(adjRef[v], edges[e].v0) &&
            VESetContains(adjRef[v], edges[e].v1)) {
          if (timing) ++l.apexRejects;
          return;
        }
      }
      if (timing) ++l.hitCount;
      l.hits.push_back({e, dotAB / g.abLen2, v});
    };
    manifold::for_each_n(autoPolicy(pairs.size(), 512), countAt(size_t{0}),
                         pairs.size(), [&](size_t idx) {
                           auto& l = tls.local();
                           const auto [i, j] = pairs[idx];
                           narrowLocal(edges[i].v0, j, l);
                           narrowLocal(edges[i].v1, j, l);
                           narrowLocal(edges[j].v0, i, l);
                           narrowLocal(edges[j].v1, i, l);
                         });
    tls.combine_each([&](const Local& l) {
      flatHits.insert(flatHits.end(), l.hits.begin(), l.hits.end());
      if (timing) {
        candCount += l.candCount;
        endpointRejects += l.endpointRejects;
        degenerateRejects += l.degenerateRejects;
        tRangeRejects += l.tRangeRejects;
        distanceRejects += l.distanceRejects;
        apexRejects += l.apexRejects;
        hitCount += l.hitCount;
      }
    });
  } else
#endif
  {
    for (const auto& [i, j] : pairs) {
      narrowIfNonEndpoint(edges[i].v0, j);
      narrowIfNonEndpoint(edges[i].v1, j);
      narrowIfNonEndpoint(edges[j].v0, i);
      narrowIfNonEndpoint(edges[j].v1, i);
    }
  }

  manifold::stable_sort(flatHits.begin(), flatHits.end(),
                        [](const Hit& a, const Hit& b) {
                          if (a.e != b.e) return a.e < b.e;
                          if (a.t != b.t) return a.t < b.t;
                          return a.v < b.v;
                        });
  for (size_t i = 0; i < flatHits.size();) {
    const int e = flatHits[i].e;
    size_t j = i;
    while (j < flatHits.size() && flatHits[j].e == e) ++j;
    lists[e].reserve(j - i);
    int lastV = -1;
    for (size_t k = i; k < j; ++k) {
      if (flatHits[k].v == lastV) continue;
      lists[e].push_back(flatHits[k].v);
      lastV = flatHits[k].v;
    }
    i = j;
  }
  if (timing) {
    auto& P = GlobalPhases();
    P.edgeVertCalls.fetch_add(1, std::memory_order_relaxed);
    P.edgeVertPairDerivedCalls.fetch_add(1, std::memory_order_relaxed);
    P.edgeVertTotalEdges.fetch_add(nE, std::memory_order_relaxed);
    P.edgeVertTotalVerts.fetch_add(nV, std::memory_order_relaxed);
    P.edgeVertHitsFlat.fetch_add(static_cast<int64_t>(flatHits.size()),
                                 std::memory_order_relaxed);
    if (nE < 64) {
      P.edgeVertBucketLt64.fetch_add(1, std::memory_order_relaxed);
    } else if (nE < 256) {
      P.edgeVertBucketLt256.fetch_add(1, std::memory_order_relaxed);
    } else if (nE < 1024) {
      P.edgeVertBucketLt1024.fetch_add(1, std::memory_order_relaxed);
    } else {
      P.edgeVertBucketGe1024.fetch_add(1, std::memory_order_relaxed);
    }
    P.edgeVertCandidates.fetch_add(candCount, std::memory_order_relaxed);
    P.edgeVertEndpointRejects.fetch_add(endpointRejects,
                                        std::memory_order_relaxed);
    P.edgeVertDegenerateRejects.fetch_add(degenerateRejects,
                                          std::memory_order_relaxed);
    P.edgeVertTRangeRejects.fetch_add(tRangeRejects, std::memory_order_relaxed);
    P.edgeVertDistanceRejects.fetch_add(distanceRejects,
                                        std::memory_order_relaxed);
    P.edgeVertApexRejects.fetch_add(apexRejects, std::memory_order_relaxed);
    P.edgeVertHits.fetch_add(hitCount, std::memory_order_relaxed);
  }
  return lists;
}

// Pair-count threshold above which the driver dispatches to the
// fused parallel pass below. Lower than the standalone narrow-only
// threshold (`kPairsParallelMin` = 4096) because the fused pass does
// roughly 2x the per-pair work (narrow + intersect), so it amortizes
// TBB setup at smaller pair counts.
extern const size_t kFusedNarrowParallelMin = 1024;

// Combined parallel narrow + IntersectSegments. Each pair gets its 4
// vert-on-edge narrow tests AND a segment-segment intersection test
// in a single parallel pass over `pairs`. Thread-local accumulators
// hold both outputs.
//
// Used when the input is large enough to justify parallel (the
// caller's pair list is >= kFusedNarrowParallelMin). Eliminates one
// parallel section's TBB setup vs running BuildEdgeVertListsFromEdgePairs
// and FindAndInsertIntersections's parallel-compute stage separately.
//
// Output:
//   - `*lists` = per-edge sorted-by-parameter vert list (deduped)
//   - `*intersections` = (i, j, p) intersection points, sorted by (i, j)
void BuildListsAndFindIntersectionsParallel(
    const std::vector<EdgeM>& edges, const std::vector<vec2>& verts, double eps,
    const std::vector<std::pair<int, int>>& pairs,
    std::vector<std::vector<int>>* lists,
    std::vector<IntersectionPoint>* intersections) {
  const int nE = static_cast<int>(edges.size());
  const int nV = static_cast<int>(verts.size());
  const double eps2 = eps * eps;
  lists->assign(nE, {});
  intersections->clear();

  // Per-calling-thread scratch; workers read it only through `edgeGRef`.
  struct EdgeG {
    vec2 a, ab;
    double abLen2;
  };
  thread_local static std::vector<EdgeG> edgeG;
  edgeG.resize(nE);
  for (int e = 0; e < nE; ++e) {
    edgeG[e].a = verts[edges[e].v0];
    edgeG[e].ab = verts[edges[e].v1] - edgeG[e].a;
    edgeG[e].abLen2 = dot(edgeG[e].ab, edgeG[e].ab);
  }

  // Build adj eagerly for the apex check.
  thread_local static std::vector<std::vector<int>> adj;
  if ((int)adj.size() < nV) adj.resize(nV);
  for (int i = 0; i < nV; ++i) adj[i].clear();
  for (const auto& e : edges) {
    VESetInsert(&adj[e.v0], e.v1);
    VESetInsert(&adj[e.v1], e.v0);
  }
  // References bind the calling thread's scratch before worker dispatch.
  auto& adjRef = adj;
  auto& edgeGRef = edgeG;

  struct Hit {
    int e;
    double t;
    int v;
  };

  auto narrowLocal = [&](int v, int e, std::vector<Hit>& out) {
    if (v == edges[e].v0 || v == edges[e].v1) return;
    const auto& g = edgeGRef[e];
    if (g.abLen2 == 0) return;
    const vec2 ap = verts[v] - g.a;
    const double dotAB = ap.x * g.ab.x + ap.y * g.ab.y;
    if (dotAB <= 0 || dotAB >= g.abLen2) return;
    const double cross = ap.x * g.ab.y - ap.y * g.ab.x;
    const double cross2 = cross * cross;
    const double eps2_abLen2 = eps2 * g.abLen2;
    if (cross2 > eps2_abLen2) return;
    if (cross2 >= eps2_abLen2 * 1e-4) {
      if (VESetContains(adjRef[v], edges[e].v0) &&
          VESetContains(adjRef[v], edges[e].v1))
        return;
    }
    out.push_back({e, dotAB / g.abLen2, v});
  };

  std::vector<Hit> flatHits;

  // Per-pair work: four vert-on-edge narrow tests plus an
  // IntersectSegments call gated on the pair not sharing an endpoint.
  // The narrow tests always run because T-junctions and butterfly
  // polygons with cancel-pair retraces can put the non-shared endpoint
  // of one edge on the interior of the other, even when the pair
  // shares a vertex. IntersectSegments is skipped for shared-endpoint
  // pairs: it would produce a bogus intersection at the shared vertex
  // itself. CollectIntersectionPairs has already pruned shared-
  // endpoint pairs that aren't near-collinear at the shared vertex
  // (no possible T-junction), so by the time we get here, shared
  // pairs are a small minority.
  auto processPair = [&](size_t idx, std::vector<Hit>& hitsOut,
                         std::vector<IntersectionPoint>& ixOut) {
    const auto& pr = pairs[idx];
    const int i = pr.first;
    const int j = pr.second;
    const auto& ei = edges[i];
    const auto& ej = edges[j];
    narrowLocal(ei.v0, j, hitsOut);
    narrowLocal(ei.v1, j, hitsOut);
    narrowLocal(ej.v0, i, hitsOut);
    narrowLocal(ej.v1, i, hitsOut);
    if (ei.v0 == ej.v0 || ei.v0 == ej.v1 || ei.v1 == ej.v0 || ei.v1 == ej.v1)
      return;
    vec2 p;
    if (IntersectSegments(verts[ei.v0], verts[ei.v1], verts[ej.v0],
                          verts[ej.v1], eps, &p)) {
      ixOut.push_back({i, j, p});
    }
  };

#if (MANIFOLD_PAR == 1)
  struct Local {
    std::vector<Hit> hits;
    std::vector<IntersectionPoint> ix;
  };
  tbb::combinable<Local> tls;
  manifold::for_each_n(autoPolicy(pairs.size(), 512), countAt(size_t{0}),
                       pairs.size(), [&](size_t idx) {
                         auto& l = tls.local();
                         processPair(idx, l.hits, l.ix);
                       });
  tls.combine_each([&](const Local& l) {
    flatHits.insert(flatHits.end(), l.hits.begin(), l.hits.end());
    intersections->insert(intersections->end(), l.ix.begin(), l.ix.end());
  });
#else
  // Serial fallback: same per-pair work, no TBB. Keeps the function
  // callable from any caller regardless of MANIFOLD_PAR.
  for (size_t idx = 0; idx < pairs.size(); ++idx) {
    processPair(idx, flatHits, *intersections);
  }
#endif

  // Sort hits by (e, t, v), build per-edge lists, dedupe.
  manifold::stable_sort(flatHits.begin(), flatHits.end(),
                        [](const Hit& a, const Hit& b) {
                          if (a.e != b.e) return a.e < b.e;
                          if (a.t != b.t) return a.t < b.t;
                          return a.v < b.v;
                        });
  for (size_t i = 0; i < flatHits.size();) {
    const int e = flatHits[i].e;
    size_t j = i;
    while (j < flatHits.size() && flatHits[j].e == e) ++j;
    auto& lst = (*lists)[e];
    lst.reserve(j - i);
    int lastV = -1;
    for (size_t k = i; k < j; ++k) {
      if (flatHits[k].v == lastV) continue;
      lst.push_back(flatHits[k].v);
      lastV = flatHits[k].v;
    }
    i = j;
  }
  // Sort intersections by (i, j) for deterministic serial snap+insert.
  manifold::stable_sort(
      intersections->begin(), intersections->end(),
      [](const IntersectionPoint& a, const IntersectionPoint& b) {
        if (a.i != b.i) return a.i < b.i;
        return a.j < b.j;
      });
}

}  // namespace boolean2
}  // namespace manifold

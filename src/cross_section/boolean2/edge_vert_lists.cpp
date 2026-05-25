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
// Edge-vertex list construction: for each input edge, the sorted-by-parameter
// list of vertices that lie within eps of the edge interior. Candidates are
// derived from the same edge-pair broad phase used for intersections.

#include "edge_vert_lists.h"

#include <algorithm>
#include <vector>

#include "../../parallel.h"
#include "parallel_policy.h"
#include "predicates.h"
#include "vertex_merge.h"

namespace manifold {
namespace boolean2 {

namespace {

// Only reject an adjacent-edge apex once the tested vertex is clearly away
// from the candidate edge's line. Extremely near-line apexes are valid
// T-junction candidates and must stay in the edge-vert list. The cutoff is
// 1% of the eps distance, squared for the cross^2 compare: lower values admit
// too many ordinary adjacent-edge apexes as split candidates, while larger
// values drop the near-line apex regression exercised by ApexSkipNearLine.
constexpr double kApexRejectMinDistance2Frac = 1e-4;

bool FarEnoughFromLineForApexReject(double cross2, double eps2_abLen2) {
  return cross2 >= eps2_abLen2 * kApexRejectMinDistance2Frac;
}

}  // namespace

std::vector<std::vector<int>> BuildEdgeVertListsFromEdgePairs(
    const std::vector<EdgeM>& edges, const std::vector<vec2>& verts, double eps,
    const std::vector<std::pair<int, int>>& pairs) {
  const int nE = static_cast<int>(edges.size());
  const int nV = static_cast<int>(verts.size());
  const double eps2 = eps * eps;
  std::vector<std::vector<int>> lists(nE);

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
    if (v == edges[e].v0 || v == edges[e].v1) {
      return;
    }
    const auto& g = edgeG[e];
    if (g.abLen2 == 0) {
      return;
    }
    const vec2 ap = verts[v] - g.a;
    const double dotAB = ap.x * g.ab.x + ap.y * g.ab.y;
    if (dotAB <= 0 || dotAB >= g.abLen2) {
      return;
    }
    const double cross = ap.x * g.ab.y - ap.y * g.ab.x;
    const double cross2 = cross * cross;
    const double eps2_abLen2 = eps2 * g.abLen2;
    if (cross2 > eps2_abLen2) {
      return;
    }
    if (FarEnoughFromLineForApexReject(cross2, eps2_abLen2)) {
      ensureAdj();
      if (VESetContains(adj[v], edges[e].v0) &&
          VESetContains(adj[v], edges[e].v1)) {
        return;
      }
    }
    flatHits.push_back({e, dotAB / g.abLen2, v});
  };

  auto narrowIfNonEndpoint = [&](int v, int e) {
    if (v == edges[e].v0 || v == edges[e].v1) {
      return;
    }
    narrow(v, e);
  };

  for (const auto& [i, j] : pairs) {
    narrowIfNonEndpoint(edges[i].v0, j);
    narrowIfNonEndpoint(edges[i].v1, j);
    narrowIfNonEndpoint(edges[j].v0, i);
    narrowIfNonEndpoint(edges[j].v1, i);
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
  return lists;
}

// Pair-count threshold above which the driver dispatches to the fused
// parallel pass below. Smaller inputs stay on the serial two-pass path to
// avoid TBB setup overhead.
extern const size_t kFusedNarrowParallelMin = 1024;

#if (MANIFOLD_PAR == 1)
// Combined parallel narrow + IntersectSegments pass for large pair lists.
// Outputs per-edge split lists and sorted (i, j, p) intersection points.
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
    if (FarEnoughFromLineForApexReject(cross2, eps2_abLen2)) {
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
    if (IntersectSegments({verts[ei.v0], verts[ei.v1], i},
                          {verts[ej.v0], verts[ej.v1], j}, eps, &p)) {
      ixOut.push_back({i, j, p});
    }
  };

  struct Local {
    std::vector<Hit> hits;
    std::vector<IntersectionPoint> ix;
  };
  tbb::combinable<Local> tls;
  manifold::for_each_n(autoPolicy(pairs.size(), kFineParallelGrainSize),
                       countAt(size_t{0}), pairs.size(), [&](size_t idx) {
                         auto& l = tls.local();
                         processPair(idx, l.hits, l.ix);
                       });
  tls.combine_each([&](const Local& l) {
    flatHits.insert(flatHits.end(), l.hits.begin(), l.hits.end());
    intersections->insert(intersections->end(), l.ix.begin(), l.ix.end());
  });

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
#endif

}  // namespace boolean2
}  // namespace manifold

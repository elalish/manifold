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

// Pair-count threshold above which the combined helper uses TBB. Smaller
// inputs stay serial to avoid setup overhead.
constexpr size_t kFusedNarrowParallelMin = 1024;

bool FarEnoughFromLineForApexReject(double cross2, double eps2_abLen2) {
  return cross2 >= eps2_abLen2 * kApexRejectMinDistance2Frac;
}

struct EdgeGeom {
  vec2 a, ab;
  double abLen2;
};

struct EdgeVertHit {
  int e;
  double t;
  int v;
};

void BuildEdgeGeometry(const std::vector<EdgeM>& edges,
                       const std::vector<vec2>& verts,
                       std::vector<EdgeGeom>* edgeG) {
  const int nE = static_cast<int>(edges.size());
  edgeG->resize(nE);
  for (int e = 0; e < nE; ++e) {
    (*edgeG)[e].a = verts[edges[e].v0];
    (*edgeG)[e].ab = verts[edges[e].v1] - (*edgeG)[e].a;
    (*edgeG)[e].abLen2 = dot((*edgeG)[e].ab, (*edgeG)[e].ab);
  }
}

void BuildAdjacency(const std::vector<EdgeM>& edges, int nV,
                    std::vector<std::vector<int>>* adj) {
  if ((int)adj->size() < nV) adj->resize(nV);
  for (int i = 0; i < nV; ++i) (*adj)[i].clear();
  for (const auto& e : edges) {
    VESetInsert(&(*adj)[e.v0], e.v1);
    VESetInsert(&(*adj)[e.v1], e.v0);
  }
}

void RecordEdgeVertHit(const std::vector<EdgeM>& edges,
                       const std::vector<vec2>& verts,
                       const std::vector<EdgeGeom>& edgeG,
                       const std::vector<std::vector<int>>& adj, double eps2,
                       int v, int e, std::vector<EdgeVertHit>* hits) {
  if (v == edges[e].v0 || v == edges[e].v1) return;
  const auto& g = edgeG[e];
  if (g.abLen2 == 0) return;
  const vec2 ap = verts[v] - g.a;
  const double dotAB = ap.x * g.ab.x + ap.y * g.ab.y;
  if (dotAB <= 0 || dotAB >= g.abLen2) return;
  const double cross = ap.x * g.ab.y - ap.y * g.ab.x;
  const double cross2 = cross * cross;
  const double eps2_abLen2 = eps2 * g.abLen2;
  if (cross2 > eps2_abLen2) return;
  if (FarEnoughFromLineForApexReject(cross2, eps2_abLen2) &&
      VESetContains(adj[v], edges[e].v0) &&
      VESetContains(adj[v], edges[e].v1)) {
    return;
  }
  hits->push_back({e, dotAB / g.abLen2, v});
}

bool SharesEndpoint(const EdgeM& a, const EdgeM& b) {
  return a.v0 == b.v0 || a.v0 == b.v1 || a.v1 == b.v0 || a.v1 == b.v1;
}

void ProcessEdgePair(const std::vector<EdgeM>& edges,
                     const std::vector<vec2>& verts,
                     const std::vector<EdgeGeom>& edgeG,
                     const std::vector<std::vector<int>>& adj, double eps,
                     const std::pair<int, int>& pr,
                     std::vector<EdgeVertHit>* hits,
                     std::vector<IntersectionPoint>* intersections) {
  const int i = pr.first;
  const int j = pr.second;
  const auto& ei = edges[i];
  const auto& ej = edges[j];
  const double eps2 = eps * eps;
  RecordEdgeVertHit(edges, verts, edgeG, adj, eps2, ei.v0, j, hits);
  RecordEdgeVertHit(edges, verts, edgeG, adj, eps2, ei.v1, j, hits);
  RecordEdgeVertHit(edges, verts, edgeG, adj, eps2, ej.v0, i, hits);
  RecordEdgeVertHit(edges, verts, edgeG, adj, eps2, ej.v1, i, hits);
  if (SharesEndpoint(ei, ej)) return;
  vec2 p;
  if (IntersectSegments({verts[ei.v0], verts[ei.v1], i},
                        {verts[ej.v0], verts[ej.v1], j}, eps, &p)) {
    intersections->push_back({i, j, p});
  }
}

void MaterializeEdgeVertLists(int nE, std::vector<EdgeVertHit>* flatHits,
                              std::vector<std::vector<int>>* lists) {
  lists->assign(nE, {});
  manifold::stable_sort(flatHits->begin(), flatHits->end(),
                        [](const EdgeVertHit& a, const EdgeVertHit& b) {
                          if (a.e != b.e) return a.e < b.e;
                          if (a.t != b.t) return a.t < b.t;
                          return a.v < b.v;
                        });
  for (size_t i = 0; i < flatHits->size();) {
    const int e = (*flatHits)[i].e;
    size_t j = i;
    while (j < flatHits->size() && (*flatHits)[j].e == e) ++j;
    auto& lst = (*lists)[e];
    lst.reserve(j - i);
    int lastV = -1;
    for (size_t k = i; k < j; ++k) {
      if ((*flatHits)[k].v == lastV) continue;
      lst.push_back((*flatHits)[k].v);
      lastV = (*flatHits)[k].v;
    }
    i = j;
  }
}

void SortIntersections(std::vector<IntersectionPoint>* intersections) {
  manifold::stable_sort(
      intersections->begin(), intersections->end(),
      [](const IntersectionPoint& a, const IntersectionPoint& b) {
        if (a.i != b.i) return a.i < b.i;
        return a.j < b.j;
      });
}

}  // namespace

void BuildListsAndFindIntersections(
    const std::vector<EdgeM>& edges, const std::vector<vec2>& verts, double eps,
    const std::vector<std::pair<int, int>>& pairs,
    std::vector<std::vector<int>>* lists,
    std::vector<IntersectionPoint>* intersections) {
  const int nE = static_cast<int>(edges.size());
  const int nV = static_cast<int>(verts.size());
  intersections->clear();

  // Per-calling-thread scratch; workers read these only through const refs.
  thread_local static std::vector<EdgeGeom> edgeG;
  BuildEdgeGeometry(edges, verts, &edgeG);
  thread_local static std::vector<std::vector<int>> adj;
  BuildAdjacency(edges, nV, &adj);
  const auto& edgeGRef = edgeG;
  const auto& adjRef = adj;

  std::vector<EdgeVertHit> flatHits;

#if (MANIFOLD_PAR == 1)
  if (pairs.size() >= kFusedNarrowParallelMin) {
    struct Local {
      std::vector<EdgeVertHit> hits;
      std::vector<IntersectionPoint> ix;
    };
    tbb::combinable<Local> tls;
    manifold::for_each_n(autoPolicy(pairs.size(), kFineParallelGrainSize),
                         countAt(size_t{0}), pairs.size(), [&](size_t idx) {
                           auto& l = tls.local();
                           ProcessEdgePair(edges, verts, edgeGRef, adjRef, eps,
                                           pairs[idx], &l.hits, &l.ix);
                         });
    tls.combine_each([&](const Local& l) {
      flatHits.insert(flatHits.end(), l.hits.begin(), l.hits.end());
      intersections->insert(intersections->end(), l.ix.begin(), l.ix.end());
    });
  } else
#endif
  {
    for (const auto& pr : pairs) {
      ProcessEdgePair(edges, verts, edgeGRef, adjRef, eps, pr, &flatHits,
                      intersections);
    }
  }
  MaterializeEdgeVertLists(nE, &flatHits, lists);
  SortIntersections(intersections);
}

}  // namespace boolean2
}  // namespace manifold

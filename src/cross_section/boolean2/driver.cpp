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
// End-to-end Boolean2 driver. Stitches together vertex merging,
// edge collapse, near-vertex indexing, proper edge-edge crossing insertion,
// nearby-intersection merging, sub-edge canonicalization,
// and winding-rule filtering.
// Returns an OverlapResult holding the merged-vert list, the retained
// directed sub-edges, the inputVert2Merged remap, and the merged-vert count.

#include "driver.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "../../boolean2_diagnostics.h"
#include "../../disjoint_sets.h"
#include "../../parallel.h"
#include "bvh.h"
#include "canonicalize.h"
#include "edge_vert_lists.h"
#include "intersections.h"
#include "predicates.h"
#include "vertex_merge.h"
#include "winding_filter.h"

namespace manifold {

namespace {

// Unlike MergeVerts, this pass only considers vertices with intersection
// incidence: newly inserted intersection points and old vertices that an
// intersection snapped onto. The close-pair threshold is intersection-specific,
// and endpoint tolerance snaps are limited to truly-new vertices, so this does
// not re-run broad old-vertex clustering across the input.
void MergeNearbyIntersectionVerts(
    std::vector<vec2>& verts, std::vector<EdgeM>& edges,
    std::vector<std::vector<int>>& lists,
    const std::vector<std::vector<int>>& vertEdges, int oldVertEnd, double eps,
    double tolerance, std::vector<int>& inputVert2Merged) {
  DisjointSets uf(static_cast<int>(verts.size()));
  const double newToNewThresh = kIntersectionMergeEpsFactor * eps;
  const double newToNewThresh2 = newToNewThresh * newToNewThresh;
  // Fresh intersection vs old endpoint: prior drift plus current-op error.
  const double newToOldThresh = tolerance + eps;
  const double newToOldThresh2 = newToOldThresh * newToOldThresh;
  std::vector<std::pair<int, int>> pairs;
  std::vector<std::pair<double, int>> tlist;
  for (size_t e = 0; e < edges.size(); ++e) {
    const auto& list = lists[e];
    const int v0 = edges[e].v0;
    const int v1 = edges[e].v1;
    const vec2 a = verts[v0];
    const vec2 b = verts[v1];
    const vec2 ab = b - a;
    const double abLen2 = dot(ab, ab);
    if (abLen2 == 0) continue;
    const double abLen = std::sqrt(abLen2);
    const double tThresh = newToNewThresh / abLen;
    tlist.clear();
    tlist.reserve(list.size());
    for (int v : list) {
      // Skip pure-old verts; keep snapped-onto-old for new-to-new pairing.
      if (v >= static_cast<int>(vertEdges.size()) || vertEdges[v].empty())
        continue;
      const vec2 p = verts[v];
      const double t = dot(p - a, ab) / abLen2;
      tlist.emplace_back(t, v);
      // Only truly-new verts: widening old-old would stack error across ops.
      if (v >= oldVertEnd && newToOldThresh > 0.0) {
        const vec2 dA = p - a;
        if (dot(dA, dA) <= newToOldThresh2) {
          const int p0 = std::min(v, v0);
          const int q0 = std::max(v, v0);
          if (p0 != q0) pairs.emplace_back(p0, q0);
        }
        const vec2 dB = p - b;
        if (dot(dB, dB) <= newToOldThresh2) {
          const int p0 = std::min(v, v1);
          const int q0 = std::max(v, v1);
          if (p0 != q0) pairs.emplace_back(p0, q0);
        }
      }
    }
    if (tlist.size() < 2) continue;
    for (size_t i = 0; i < tlist.size(); ++i) {
      for (size_t j = i + 1;
           j < tlist.size() && tlist[j].first - tlist[i].first <= tThresh;
           ++j) {
        const int va = tlist[i].second;
        const int vb = tlist[j].second;
        const vec2 d = verts[vb] - verts[va];
        if (dot(d, d) > newToNewThresh2) continue;
        const int p = std::min(va, vb);
        const int q = std::max(va, vb);
        pairs.emplace_back(p, q);
      }
    }
  }
  manifold::stable_sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  for (const auto& pr : pairs) uf.unite(pr.first, pr.second);

  const int nV = static_cast<int>(verts.size());
  std::vector<vec2> sumPos(nV, vec2{0, 0});
  std::vector<int> sumCnt(nV, 0);
  int distinctClusters = 0;
  for (int i = 0; i < nV; ++i) {
    int r = uf.find(i);
    if (sumCnt[r] == 0) ++distinctClusters;
    sumPos[r] = sumPos[r] + verts[i];
    sumCnt[r] += 1;
  }
  if (distinctClusters == nV) return;

  std::vector<int> rootToNew(nV, -1);
  std::vector<vec2> newVerts;
  newVerts.reserve(distinctClusters);
  for (int r = 0; r < nV; ++r) {
    if (sumCnt[r] == 0) continue;
    rootToNew[r] = static_cast<int>(newVerts.size());
    newVerts.push_back(sumPos[r] * (1.0 / sumCnt[r]));
  }
  std::vector<int> preMergeToPost(nV);
  for (int i = 0; i < nV; ++i) preMergeToPost[i] = rootToNew[uf.find(i)];
  for (auto& e : edges) {
    e.v0 = preMergeToPost[e.v0];
    e.v1 = preMergeToPost[e.v1];
  }
  for (auto& list : lists) {
    for (auto& v : list) v = preMergeToPost[v];
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }
  for (auto& r : inputVert2Merged) r = preMergeToPost[r];
  verts = std::move(newVerts);
}

}  // namespace

OverlapResult RemoveOverlaps2D(const std::vector<vec2>& vertsIn,
                               const std::vector<EdgeM>& edgesIn, double eps,
                               double tolerance, bool debug, WindRule pred,
                               Trace* trace) {
  if (tolerance < eps) tolerance = eps;
  auto& P = GlobalPhases();
  ScopedTiming totalTiming(P.totalNs);
  TraceRecorder traceRecorder(trace, eps, pred);
  traceRecorder.RecordInput(vertsIn, edgesIn);

  // Vertex merge.
  VertexMerge merge;
  {
    ScopedTiming timing(P.mergeNs);
    merge = MergeVerts(vertsIn, eps);
  }
  const int numMerged = static_cast<int>(merge.verts.size());
  traceRecorder.RecordMergedVertices(merge.verts, merge.inputVert2Merged);
  // Edge collapse.
  std::vector<EdgeM> edges;
  {
    ScopedTiming timing(P.remapNs);
    edges = RemapAndCollapse(edgesIn, merge.inputVert2Merged);
  }
  traceRecorder.RecordCollapsedEdges(merge.verts, edges);
  // Build a shared edge-box array for edge-edge broad phase and near-vertex
  // derivation. Medium cases use a sweep over these boxes; very large cases
  // build a BVH when tree construction amortizes over enough queries.
  thread_local static std::vector<Box2> edgeBoxes;
  edgeBoxes.resize(edges.size());
  for (size_t e = 0; e < edges.size(); ++e) {
    edgeBoxes[e] =
        BoxOf2DEdge(merge.verts[edges[e].v0], merge.verts[edges[e].v1], eps);
  }
  BVH bvh;
  {
    ScopedTiming timing(P.bvhBuildNs);
    if (edges.size() >= kEdgePairBvhThreshold)
      bvh = BVHBuildFromBoxes(edgeBoxes);
  }
  thread_local static std::vector<std::pair<int, int>> intersectionPairs;
  // Collect edge pairs once, then derive both intersection candidates and
  // near-vertex lists from that pair set. In polygon arrangements, a vertex
  // that can split a non-incident edge must have an eps-padded vertex box
  // overlapping that edge's eps-padded box, so it appears as one endpoint of
  // an overlapping edge pair.
  {
    ScopedTiming timing(P.broadPairWorkNs);
    CollectIntersectionPairs(edges, merge.verts, eps, edgeBoxes, bvh,
                             &intersectionPairs);
  }
  traceRecorder.RecordBroadPhasePairs(merge.verts, edges, intersectionPairs);
  // Combined narrow phase: derive edge-vertex split lists and independent
  // edge-edge intersections from the same broad-phase pair set. The helper
  // decides internally whether the pair loop is serial or TBB.
  NarrowPhaseResult narrow;
  {
    ScopedTiming timing(P.narrowPhaseNs);
    narrow = BuildListsAndFindIntersections(edges, merge.verts, eps,
                                            intersectionPairs);
  }
  traceRecorder.RecordEdgeVertLists(merge.verts, edges, narrow.lists);
  IntersectionInsertion inserted;
  {
    ScopedTiming timing(P.findIxNs);
    inserted = FindAndInsertIntersections(edges, std::move(merge.verts),
                                          std::move(narrow.lists), eps,
                                          edgeBoxes, bvh, narrow.intersections);
  }
  merge.verts = std::move(inserted.verts);
  std::vector<std::vector<int>> lists = std::move(inserted.lists);
  std::vector<std::vector<int>> vertEdges = std::move(inserted.vertEdges);
  traceRecorder.RecordInsertedIntersections(merge.verts, edges, lists);

  {
    ScopedTiming timing(P.nearbyIxMergeNs);
    MergeNearbyIntersectionVerts(merge.verts, edges, lists, vertEdges,
                                 numMerged, eps, tolerance,
                                 merge.inputVert2Merged);
  }
  traceRecorder.RecordNearbyIntersectionMerge(merge.verts, edges, lists);
  // Sub-edge canonicalization.
  CanonicalSubEdges canon;
  {
    ScopedTiming timing(P.canonNs);
    canon = Canonicalize(edges, lists);
  }
  traceRecorder.RecordCanonicalSubedges(merge.verts, canon);
  // Halfedge face-traversal winding filter.
  std::vector<OutEdge> out;
  {
    ScopedTiming timing(P.filterHalfedgeNs);
    out = FilterByWindingHalfedges(canon, merge.verts, debug, pred, trace);
  }
  traceRecorder.RecordFilteredOutput(merge.verts, out);
  CountTimingCase();
  return {std::move(merge.verts), std::move(out),
          std::move(merge.inputVert2Merged), numMerged};
}

}  // namespace manifold

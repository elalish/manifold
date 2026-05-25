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
// structural re-merge of crossing verts, sub-edge canonicalization,
// and winding-rule filtering.
// Returns an OverlapResult holding the merged-vert list, the retained
// directed sub-edges, the inputVert2Merged remap, and the merged-vert count.

#include "driver.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "../../disjoint_sets.h"
#include "../../parallel.h"
#include "bvh.h"
#include "canonicalize.h"
#include "diagnostics.h"
#include "edge_vert_lists.h"
#include "intersections.h"
#include "predicates.h"
#include "vertex_merge.h"
#include "winding_filter.h"

namespace manifold {
namespace boolean2 {

OverlapResult RemoveOverlaps2D(const std::vector<vec2>& vertsIn,
                               const std::vector<EdgeM>& edgesIn, double eps,
                               bool debug, WindRule pred, Trace* trace) {
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
  std::vector<std::vector<int>> lists;
  thread_local static std::vector<std::pair<int, int>> intersectionPairs;
  thread_local static std::vector<IntersectionPoint> fusedIntersections;
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
  // Fused parallel pass: narrow vert-on-edge tests AND IntersectSegments
  // run in one parallel for over `pairs`, producing `lists` and the
  // precomputed intersection points in a single TBB section. Gated at
  // >= 1024 pairs to skip TBB overhead for small workloads; below it,
  // fall back to the two-pass form (BuildEdgeVertListsFromEdgePairs then
  // FindAndInsertIntersections, the latter handling intersect compute
  // inline).
  fusedIntersections.clear();
  bool useFused = false;
  {
    ScopedTiming timing(P.edgeVertListsNs);
#if (MANIFOLD_PAR == 1)
    if (intersectionPairs.size() >= kFusedNarrowParallelMin) {
      BuildListsAndFindIntersectionsParallel(edges, merge.verts, eps,
                                             intersectionPairs, &lists,
                                             &fusedIntersections);
      useFused = true;
    } else
#endif
    {
      lists = BuildEdgeVertListsFromEdgePairs(edges, merge.verts, eps,
                                              intersectionPairs);
    }
  }
  traceRecorder.RecordEdgeVertLists(merge.verts, edges, lists);
  std::vector<std::vector<int>> vertEdges;
  {
    ScopedTiming timing(P.findIxNs);
#if (MANIFOLD_PAR == 1)
    if (useFused) {
      FindAndInsertIntersectionsFromPrecomputed(edges, &merge.verts, &lists,
                                                &vertEdges, eps, edgeBoxes, bvh,
                                                fusedIntersections);
    } else {
#endif
      FindAndInsertIntersections(edges, &merge.verts, &lists, &vertEdges, eps,
                                 edgeBoxes, bvh, intersectionPairs);
#if (MANIFOLD_PAR == 1)
    }
#endif
  }
  traceRecorder.RecordInsertedIntersections(merge.verts, edges, lists);

  // Structural re-merge: after insertion, duplicate intersection verts that
  // share an incident edge and still land FP-close represent one true point.
  {
    ScopedTiming timing(P.restructNs);
    DisjointSets uf(static_cast<int>(merge.verts.size()));
    // Shallow crossings can put duplicates up to eps/sin(theta) apart; the
    // shared-edge gate prevents that wider threshold from merging unrelated
    // intersections.
    const double mergeThresh = kIntersectionMergeEpsFactor * eps;
    const double mergeThresh2 = mergeThresh * mergeThresh;
    // Duplicate intersection verts that should merge share an incident edge.
    // Sweep each edge's sorted list and distance-check nearby candidates.
    std::vector<std::pair<int, int>> pairs;
    std::vector<std::pair<double, int>> tlist;  // reused per edge
    for (size_t e = 0; e < edges.size(); ++e) {
      const auto& list = lists[e];
      if (list.size() < 2) continue;
      const vec2 a = merge.verts[edges[e].v0];
      const vec2 b = merge.verts[edges[e].v1];
      const vec2 ab = b - a;
      const double abLen2 = dot(ab, ab);
      if (abLen2 == 0) continue;
      const double abLen = std::sqrt(abLen2);
      const double tThresh = mergeThresh / abLen;
      tlist.clear();
      tlist.reserve(list.size());
      // Lists are sorted by t; filter to intersection verts only.
      for (int v : list) {
        if (v >= (int)vertEdges.size() || vertEdges[v].empty()) continue;
        const vec2 p = merge.verts[v];
        const double t = dot(p - a, ab) / abLen2;
        tlist.emplace_back(t, v);
      }
      if (tlist.size() < 2) continue;
      // Sweep window: for each i, advance j while along-edge gap ≤ tThresh.
      for (size_t i = 0; i < tlist.size(); ++i) {
        for (size_t j = i + 1;
             j < tlist.size() && tlist[j].first - tlist[i].first <= tThresh;
             ++j) {
          const int va = tlist[i].second;
          const int vb = tlist[j].second;
          const vec2 d = merge.verts[vb] - merge.verts[va];
          if (dot(d, d) > mergeThresh2) continue;
          const int p = std::min(va, vb);
          const int q = std::max(va, vb);
          pairs.emplace_back(p, q);
        }
      }
    }
    manifold::stable_sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    for (const auto& pr : pairs) uf.unite(pr.first, pr.second);
    // Build preRestruct2Post from union-find clusters; cluster position is
    // centroid. Vector-by-root-id replaces std::map (same fix as MergeVerts).
    const int nV = static_cast<int>(merge.verts.size());
    std::vector<vec2> sumPos(nV, vec2{0, 0});
    std::vector<int> sumCnt(nV, 0);
    int distinctClusters = 0;
    for (int i = 0; i < nV; ++i) {
      int r = uf.find(i);
      if (sumCnt[r] == 0) ++distinctClusters;
      sumPos[r] = sumPos[r] + merge.verts[i];
      sumCnt[r] += 1;
    }
    if (distinctClusters < nV) {
      std::vector<int> rootToNew(nV, -1);
      std::vector<vec2> newVerts;
      newVerts.reserve(distinctClusters);
      for (int r = 0; r < nV; ++r) {
        if (sumCnt[r] == 0) continue;
        rootToNew[r] = static_cast<int>(newVerts.size());
        newVerts.push_back(sumPos[r] * (1.0 / sumCnt[r]));
      }
      std::vector<int> preRestruct2Post(nV);
      for (int i = 0; i < nV; ++i) preRestruct2Post[i] = rootToNew[uf.find(i)];
      // Apply preRestruct2Post to edges + lists + composed input remap.
      for (auto& e : edges) {
        e.v0 = preRestruct2Post[e.v0];
        e.v1 = preRestruct2Post[e.v1];
      }
      for (auto& list : lists) {
        for (auto& v : list) v = preRestruct2Post[v];
        list.erase(std::unique(list.begin(), list.end()), list.end());
      }
      for (auto& r : merge.inputVert2Merged) r = preRestruct2Post[r];
      merge.verts = std::move(newVerts);
    }
  }
  traceRecorder.RecordStructuralRemerge(merge.verts, edges, lists);
  // Sub-edge canonicalization.
  thread_local static CanonicalSubEdges canon;
  {
    ScopedTiming timing(P.canonNs);
    Canonicalize(edges, lists, &canon);
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

}  // namespace boolean2
}  // namespace manifold

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
// edge collapse, near-vertex indexing, edge-edge intersection insertion,
// structural re-merge of intersection verts, sub-edge canonicalization,
// and winding-rule filtering.
// Returns an OverlapResult holding the merged-vert list, the retained
// directed sub-edges, the input->output vert remap, and the merged-vert
// count.

#include "driver.h"

#include <algorithm>
#include <chrono>
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
                               bool debug, WindRule pred) {
  using timing_detail::Clock;
  using timing_detail::Ns;
  const bool timing = TimingEnabled();
  auto& P = GlobalPhases();
  const auto tStart = timing ? Clock::now() : Clock::time_point{};
  // Vertex merge.
  auto t0 = timing ? Clock::now() : Clock::time_point{};
  auto merge = MergeVerts(vertsIn, eps);
  auto t1 = timing ? Clock::now() : Clock::time_point{};
  if (timing) P.mergeNs.fetch_add(Ns(t0, t1), std::memory_order_relaxed);
  const int numMerged = static_cast<int>(merge.verts.size());
  // Edge collapse.
  auto edges = RemapAndCollapse(edgesIn, merge.remap);
  auto t2 = timing ? Clock::now() : Clock::time_point{};
  if (timing) P.remapNs.fetch_add(Ns(t1, t2), std::memory_order_relaxed);
  // Build a shared edge-box array for edge-edge broad phase and near-vertex
  // derivation. Medium cases use a sweep over these boxes; very large cases
  // build a BVH when tree construction amortizes over enough queries.
  thread_local static std::vector<Box2> edgeBoxes;
  edgeBoxes.resize(edges.size());
  for (size_t e = 0; e < edges.size(); ++e) {
    edgeBoxes[e] =
        BoxOf2DEdge(merge.verts[edges[e].v0], merge.verts[edges[e].v1], eps);
  }
  auto tBvhStart = timing ? Clock::now() : Clock::time_point{};
  BVH bvh;
  if (edges.size() >= kEdgePairBvhThreshold) bvh = BVHBuildFromBoxes(edgeBoxes);
  auto tBvhEnd = timing ? Clock::now() : Clock::time_point{};
  if (timing)
    P.bvhBuildNs.fetch_add(Ns(tBvhStart, tBvhEnd), std::memory_order_relaxed);
  std::vector<std::vector<int>> lists;
  thread_local static std::vector<std::pair<int, int>> intersectionPairs;
  thread_local static std::vector<IntersectionPoint> fusedIntersections;
  // Collect edge pairs once, then derive both intersection candidates and
  // near-vertex lists from that pair set. In polygon arrangements, a vertex
  // that can split a non-incident edge must have an eps-padded vertex box
  // overlapping that edge's eps-padded box, so it appears as one endpoint of
  // an overlapping edge pair.
  auto tBroadPairWorkStart = timing ? Clock::now() : Clock::time_point{};
  {
    const auto tWorkStart = timing ? Clock::now() : Clock::time_point{};
    CollectIntersectionPairs(edges, merge.verts, eps, edgeBoxes, bvh,
                             &intersectionPairs);
    const auto tWorkEnd = timing ? Clock::now() : Clock::time_point{};
    if (timing)
      P.intersectionBroadNs.fetch_add(Ns(tWorkStart, tWorkEnd),
                                      std::memory_order_relaxed);
  }
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
    const auto tWorkStart = timing ? Clock::now() : Clock::time_point{};
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
    const auto tWorkEnd = timing ? Clock::now() : Clock::time_point{};
    if (timing)
      P.edgeVertListsNs.fetch_add(Ns(tWorkStart, tWorkEnd),
                                  std::memory_order_relaxed);
  }
  auto tBroadPairWorkEnd = timing ? Clock::now() : Clock::time_point{};
  if (timing)
    P.broadPairWorkNs.fetch_add(Ns(tBroadPairWorkStart, tBroadPairWorkEnd),
                                std::memory_order_relaxed);
  auto t3 = timing ? Clock::now() : Clock::time_point{};
  if (timing) P.buildListsNs.fetch_add(Ns(t2, t3), std::memory_order_relaxed);
  std::vector<std::vector<int>> vertEdges;
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
  auto t4 = timing ? Clock::now() : Clock::time_point{};
  if (timing) P.findIxNs.fetch_add(Ns(t3, t4), std::memory_order_relaxed);

  // Structural re-merge of intersection verts. FindAndInsertIntersections
  // above inserts each intersection at the time its parent edge pair is
  // processed; if pairs
  // (A, B) and (A, C) both produce intersections at the same true point P*
  // (i.e. three edges meet there) they may land FP-close but not snap to
  // each other because neither saw the other yet at insertion time.
  //
  // Two intersections that should be the same true point share at least one
  // common edge in their incidence list (any two of {AB, AC, BC} share an
  // edge). Two intersections from disjoint edge sets cannot be the same
  // true point, regardless of geometric distance. So we union-find verts
  // that share an edge AND fall within eps; this avoids the angle-dependent
  // threshold of a pure-geometric merge (a fixed factor like 1.5*eps fails
  // for shallow crossings; large factors over-merge legitimately-distinct
  // intersections from unrelated edge pairs).
  //
  // Eager propagation in FindAndInsertIntersections handles k-way
  // incidence by inserting each new intersection vert into every edge it
  // lies on. This structural re-merge has a narrower job: merge FP-close
  // duplicate intersection verts that share an edge. Independent edge
  // pairs that geometrically meet at the same point are split by
  // propagation, but they are not unioned here unless a shared-edge pair
  // links them.
  {
    DisjointSets uf(static_cast<int>(merge.verts.size()));
    // The geometric upper bound for "same true point" is eps/sin(theta)
    // where theta is the crossing angle. For shallow crossings this can
    // be large; kIntersectionMergeEpsFactor covers theta down
    // to ~6 degrees. The structural gate prevents over-merging
    // legitimately-distinct intersections (e.g. edge A crosses B at one
    // point and C at a different point along A: vAB and vAC share edge A
    // but are at different true points and shouldn't merge unless they
    // ALSO geometrically coincide). A sweep across the displacement fuzz
    // showed kIntersectionMergeEpsFactor gives the best iteration count
    // (1:448 2:2) without
    // over-merging; tightening below 3*eps causes single-pass failures,
    // loosening to 100*eps causes new over-merge failures.
    //
    // Note: when union-find creates a multi-vert cluster, the centroid
    // computed below is offset from the original positions by up to ~eps;
    // that's intentional (we WANT the merged point to land at the average)
    // and is the source of the residual iter=2 cases. Smith's bound
    // proves convergence in ≤2 iterations under his α-budget framework.
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
      // The list is sorted by t (invariant maintained by
      // BuildEdgeVertListsFromEdgePairs and FindAndInsertIntersections's
      // lower_bound
      // insertion); filter to intersection verts only.
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
    // Build remap from union-find clusters; cluster position is centroid.
    // Vector-by-root-id replaces std::map (same fix as MergeVerts).
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
      std::vector<int> remap(nV);
      for (int i = 0; i < nV; ++i) remap[i] = rootToNew[uf.find(i)];
      // Apply remap to edges + lists + composed input remap.
      for (auto& e : edges) {
        e.v0 = remap[e.v0];
        e.v1 = remap[e.v1];
      }
      for (auto& list : lists) {
        for (auto& v : list) v = remap[v];
        list.erase(std::unique(list.begin(), list.end()), list.end());
      }
      for (auto& r : merge.remap) r = remap[r];
      merge.verts = std::move(newVerts);
    }
  }

  auto t4b = timing ? Clock::now() : Clock::time_point{};
  if (timing) P.restructNs.fetch_add(Ns(t4, t4b), std::memory_order_relaxed);
  // Sub-edge canonicalization.
  thread_local static CanonicalSubEdges canon;
  Canonicalize(edges, lists, &canon);
  auto t5 = timing ? Clock::now() : Clock::time_point{};
  if (timing) P.canonNs.fetch_add(Ns(t4b, t5), std::memory_order_relaxed);
  // DCEL face-traversal winding filter.
  auto out = FilterByWindingDCEL(canon, merge.verts, debug, pred);
  auto t6 = timing ? Clock::now() : Clock::time_point{};
  if (timing) {
    P.filterDcelNs.fetch_add(Ns(t5, t6), std::memory_order_relaxed);
    P.totalNs.fetch_add(Ns(tStart, t6), std::memory_order_relaxed);
    P.cases.fetch_add(1, std::memory_order_relaxed);
  }
  return {std::move(merge.verts), std::move(out), std::move(merge.remap),
          numMerged};
}

}  // namespace boolean2
}  // namespace manifold

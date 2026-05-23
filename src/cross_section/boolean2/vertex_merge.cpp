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
// Vertex-merge and edge-collapse passes. MergeVerts buckets verts within eps of
// each other onto an existing representative vertex; CollapseDegenerateEdges
// drops any input edge whose endpoints merged to the same vert. Both run before
// the BVH-broad intersection discovery.
//
// Also hosts the tiny "sorted-vector as set" helpers (VESetContains,
// VESetInsert) used by the per-edge / per-vert adjacency tracking in
// edge_vert_lists.h and intersections.h.

#include "vertex_merge.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "../../disjoint_sets.h"
#include "../../parallel.h"
#include "bvh.h"
#include "diagnostics.h"
#include "parallel_policy.h"
#include "predicates.h"

namespace manifold {
namespace boolean2 {

using manifold::la::dot;

namespace {

void AtomicMax(std::atomic<int64_t>* target, int64_t value) {
  int64_t observed = target->load(std::memory_order_relaxed);
  while (observed < value && !target->compare_exchange_weak(
                                 observed, value, std::memory_order_relaxed,
                                 std::memory_order_relaxed)) {
  }
}

}  // namespace

VertexMerge MergeVerts(const std::vector<vec2>& in, double eps) {
  using timing_detail::Clock;
  using timing_detail::Ns;
  const bool timing = TimingEnabled();
  auto t0 = timing ? Clock::now() : Clock::time_point{};
  const int n = static_cast<int>(in.size());
  DisjointSets uf(n);
  const double eps2 = eps * eps;
  // Broad-phase: collect candidate (i, j) pairs whose padded boxes overlap.
  // Small inputs use brute force; larger inputs use an x-sorted sweep. The
  // pairs are sorted so the unite step below is deterministic.
  //
  // Shared with worker threads below, so this must not be thread-local.
  std::vector<std::pair<int, int>> pairs;
  if (n < 32) {
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        // AABB overlap test (the same the BVH would do): both axes must
        // overlap on [v[i] - eps, v[i] + eps] intersect [v[j] - eps, v[j] +
        // eps].
        const vec2 d = in[i] - in[j];
        if (std::fabs(d.x) <= 2 * eps && std::fabs(d.y) <= 2 * eps)
          pairs.emplace_back(i, j);
      }
    }
  } else {
    // Sort by x and scan forward until x-distance exceeds 2*eps. Pairs are
    // sorted at the end so the unite step below stays deterministic.
    auto tBvhStart = timing ? Clock::now() : Clock::time_point{};
    thread_local static std::vector<int> idx;
    idx.resize(n);
    std::iota(idx.begin(), idx.end(), 0);
    manifold::stable_sort(idx.begin(), idx.end(),
                          [&](int a, int b) { return in[a].x < in[b].x; });
    auto tBvhEnd = timing ? Clock::now() : Clock::time_point{};
    if (timing)
      GlobalPhases().mergeBvhBuildNs.fetch_add(Ns(tBvhStart, tBvhEnd),
                                               std::memory_order_relaxed);
    auto tCollideStart = timing ? Clock::now() : Clock::time_point{};
    const double thresh = 2 * eps;
    // Each i iteration writes only to its local pair buffer, so large inputs
    // can collect candidates in parallel.
    constexpr int kParallelMergeMin = 1024;
#if (MANIFOLD_PAR == 1)
    if (n >= kParallelMergeMin) {
      // Bind before worker dispatch; direct `idx` access would resolve to each
      // worker's thread-local copy.
      auto& idxRef = idx;
      tbb::combinable<std::vector<std::pair<int, int>>> tls;
      manifold::for_each_n(autoPolicy(n, kFineParallelGrainSize), countAt(0), n,
                           [&](int i) {
                             const int ai = idxRef[i];
                             const double ax = in[ai].x;
                             const double ay = in[ai].y;
                             auto& local = tls.local();
                             for (int j = i + 1; j < n; ++j) {
                               const int bi = idxRef[j];
                               const double dx = in[bi].x - ax;
                               if (dx > thresh) break;
                               if (std::fabs(in[bi].y - ay) > thresh) continue;
                               if (ai < bi)
                                 local.emplace_back(ai, bi);
                               else
                                 local.emplace_back(bi, ai);
                             }
                           });
      tls.combine_each([&](const std::vector<std::pair<int, int>>& l) {
        pairs.insert(pairs.end(), l.begin(), l.end());
      });
    } else
#endif
    {
      for (int i = 0; i < n; ++i) {
        const int ai = idx[i];
        const double ax = in[ai].x;
        const double ay = in[ai].y;
        for (int j = i + 1; j < n; ++j) {
          const int bi = idx[j];
          const double dx = in[bi].x - ax;
          if (dx > thresh) break;
          if (std::fabs(in[bi].y - ay) > thresh) continue;
          if (ai < bi)
            pairs.emplace_back(ai, bi);
          else
            pairs.emplace_back(bi, ai);
        }
      }
    }
    manifold::stable_sort(pairs.begin(), pairs.end());
    auto tCollideEnd = timing ? Clock::now() : Clock::time_point{};
    if (timing)
      GlobalPhases().mergeCollideNs.fetch_add(Ns(tCollideStart, tCollideEnd),
                                              std::memory_order_relaxed);
  }
  // Fast path: no candidates means no merges, identity remap.
  if (pairs.empty()) {
    auto tRest = timing ? Clock::now() : Clock::time_point{};
    if (timing)
      GlobalPhases().mergeRestNs.fetch_add(Ns(t0, tRest),
                                           std::memory_order_relaxed);
    std::vector<int> remap(n);
    std::iota(remap.begin(), remap.end(), 0);
    return {std::move(remap), in};
  }
  // Parallelize the geometric distance gate (read-only on `in`); unite
  // serially in sorted pair order so cluster roots are deterministic
  // regardless of thread scheduling. Same pattern as the structural
  // re-merge in driver.h (after intersections).
  //
  // Written by worker threads, so this must be shared storage.
  std::vector<uint8_t> doUnite;
  doUnite.assign(pairs.size(), 0);
  manifold::for_each_n(manifold::autoPolicy(pairs.size()), manifold::countAt(0),
                       pairs.size(), [&](size_t k) {
                         const auto [i, j] = pairs[k];
                         vec2 d = in[i] - in[j];
                         if (dot(d, d) <= eps2) doUnite[k] = 1;
                       });
  bool anyMerge = false;
  for (size_t k = 0; k < pairs.size(); ++k) {
    if (doUnite[k]) {
      uf.unite(pairs[k].first, pairs[k].second);
      anyMerge = true;
    }
  }
  if (!anyMerge) {
    std::vector<int> remap(n);
    std::iota(remap.begin(), remap.end(), 0);
    return {std::move(remap), in};
  }
  // Compute representative per cluster.
  //
  // Contract note: this is a transitive proximity merge. If A is within eps of
  // B and B is within eps of C, all three collapse even when A and C are more
  // than eps apart. Representatives must be chosen from existing component
  // vertices rather than from a centroid: centroids can create a new within-eps
  // pair between two already-merged components on a second MergeVerts pass even
  // when no such pair existed in the original input.
  std::vector<vec2> sumPos(n, vec2{0, 0});
  std::vector<int> sumCnt(n, 0);
  for (int i = 0; i < n; ++i) {
    int r = uf.find(i);
    sumPos[r] = sumPos[r] + in[i];
    sumCnt[r] += 1;
  }
  // Pick the source vertex closest to the component centroid. This keeps the
  // representative near the center while preserving idempotence: any output
  // representative was an input vertex, so two representatives can be within
  // eps only if their original components should already have been connected.
  std::vector<int> representative(n, -1);
  std::vector<double> representativeDist2(
      n, std::numeric_limits<double>::infinity());
  for (int i = 0; i < n; ++i) {
    const int r = uf.find(i);
    const vec2 centroid = sumPos[r] * (1.0 / sumCnt[r]);
    const vec2 d = in[i] - centroid;
    const double dist2 = dot(d, d);
    if (dist2 < representativeDist2[r] ||
        (dist2 == representativeDist2[r] && i < representative[r])) {
      representative[r] = i;
      representativeDist2[r] = dist2;
    }
  }

  // Assign new indices in ascending root-id order so output ordering is
  // deterministic and matches what the old std::map iteration produced.
  std::vector<int> rootToNew(n, -1);
  std::vector<vec2> verts;
  verts.reserve(n);
  for (int r = 0; r < n; ++r) {
    if (sumCnt[r] == 0) continue;
    rootToNew[r] = static_cast<int>(verts.size());
    verts.push_back(in[representative[r]]);
  }
  if (timing) {
    auto& P = GlobalPhases();
    const double invEps = eps > 0.0 ? 1.0 / eps : 0.0;
    std::vector<vec2> bmin(n, vec2{0, 0});
    std::vector<vec2> bmax(n, vec2{0, 0});
    std::vector<double> maxDrift2(n, 0.0);
    std::vector<uint8_t> sawRoot(n, 0);
    for (int i = 0; i < n; ++i) {
      const int r = uf.find(i);
      const vec2 rep = in[representative[r]];
      if (!sawRoot[r]) {
        bmin[r] = in[i];
        bmax[r] = in[i];
        sawRoot[r] = 1;
      } else {
        bmin[r].x = std::min(bmin[r].x, in[i].x);
        bmin[r].y = std::min(bmin[r].y, in[i].y);
        bmax[r].x = std::max(bmax[r].x, in[i].x);
        bmax[r].y = std::max(bmax[r].y, in[i].y);
      }
      const vec2 d = in[i] - rep;
      maxDrift2[r] = std::max(maxDrift2[r], dot(d, d));
    }
    for (int r = 0; r < n; ++r) {
      if (sumCnt[r] <= 1) continue;
      P.mergeMergedComponents.fetch_add(1, std::memory_order_relaxed);
      AtomicMax(&P.mergeMaxMergedComponentVerts, sumCnt[r]);
      const double maxDrift = std::sqrt(maxDrift2[r]);
      if (maxDrift > eps)
        P.mergeMergedComponentsDriftGtEps.fetch_add(1,
                                                    std::memory_order_relaxed);
      const vec2 diag = bmax[r] - bmin[r];
      AtomicMax(&P.mergeMaxMergedRepresentativeDriftMilliEps,
                static_cast<int64_t>(std::ceil(1000.0 * maxDrift * invEps)));
      AtomicMax(&P.mergeMaxMergedBboxDiagMilliEps,
                static_cast<int64_t>(
                    std::ceil(1000.0 * std::sqrt(dot(diag, diag)) * invEps)));
    }
  }
  std::vector<int> remap(n);
  for (int i = 0; i < n; ++i) remap[i] = rootToNew[uf.find(i)];
  return {std::move(remap), std::move(verts)};
}

// vertEdges[v] (filled by FindAndInsertIntersections) and adj[v] (filled
// by BuildEdgeVertLists) both hold a small set of int ids per vertex.
// Almost always 2-4 elements; occasionally larger at concurrent
// intersection points. A sorted std::vector<int> beats a std::set<int>
// by 5-10x on per-op cost for sets this small (no node allocation, no
// tree rebalancing, contiguous memory). Helpers keep the "set"
// semantics: idempotent insert, fast contains, ordered iteration.
bool VESetContains(const std::vector<int>& vec, int x) {
  return std::binary_search(vec.begin(), vec.end(), x);
}
void VESetInsert(std::vector<int>* vec, int x) {
  auto it = std::lower_bound(vec->begin(), vec->end(), x);
  if (it == vec->end() || *it != x) vec->insert(it, x);
}

// Drop edges whose endpoints map to the same vertex after MergeVerts.
std::vector<EdgeM> RemapAndCollapse(const std::vector<EdgeM>& edges,
                                    const std::vector<int>& remap) {
  std::vector<EdgeM> out;
  out.reserve(edges.size());
  for (const auto& e : edges) {
    int a = remap[e.v0];
    int b = remap[e.v1];
    if (a != b) out.push_back({a, b, e.mult});
  }
  return out;
}

}  // namespace boolean2
}  // namespace manifold

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
// Polygons-typed API over the Boolean2 arrangement pipeline.

#include "boolean2.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "boolean2_diagnostics.h"
#include "disjoint_sets.h"
#include "manifold/optional_assert.h"
#include "parallel.h"

namespace manifold {

namespace {

bool AllFinite(const Polygons& polys) {
  for (const auto& loop : polys) {
    for (const vec2& v : loop) {
      if (!la::all(la::isfinite(v))) return false;
    }
  }
  return true;
}

void AccumulateBounds(const Polygons& polys, Rect& box) {
  for (const auto& loop : polys) {
    for (const vec2& v : loop) {
      box.Union(v);
    }
  }
}

// Classify the CCW turn from `ref` to `dir` for deterministic polar ordering.
// Group 0 is (0, pi], group 1 is (pi, 2pi), and group 2 is the zero turn.
// A zero turn is ordered last so loop tracing does not immediately reverse
// over the incoming edge when any positive turn is available.
int CcwTurnGroup(vec2 ref, vec2 dir) {
  const double cross = la::cross(ref, dir);
  if (cross > 0) return 0;
  if (cross < 0) return 1;
  return la::dot(ref, dir) < 0 ? 0 : 2;
}

// Order outgoing candidates for loop extraction by the smallest positive CCW
// turn from the reverse incoming direction. Collinear ties prefer the nearer
// endpoint, then stable edge id, so extraction is deterministic.
bool CcwTurnLess(vec2 ref, vec2 a, int edgeA, vec2 b, int edgeB) {
  const int groupA = CcwTurnGroup(ref, a);
  const int groupB = CcwTurnGroup(ref, b);
  if (groupA != groupB) return groupA < groupB;

  const double cross = la::cross(a, b);
  if (cross != 0) return cross > 0;

  const double dist2A = la::dot(a, a);
  const double dist2B = la::dot(b, b);
  if (dist2A != dist2B) return dist2A < dist2B;
  return edgeA < edgeB;
}

void PushLoopIfNondegenerate(const std::vector<vec2>& verts,
                             const std::vector<int>& loopVerts,
                             Polygons& polys) {
  if (loopVerts.size() >= 3) {
    SimplePolygon loop;
    loop.reserve(loopVerts.size());
    for (int v : loopVerts) loop.push_back(verts[v]);
    polys.push_back(std::move(loop));
  }
}

void PushSimpleLoops(const std::vector<vec2>& verts, std::vector<int> loopVerts,
                     Polygons& polys) {
  for (;;) {
    bool split = false;
    for (size_t i = 1; i < loopVerts.size() && !split; ++i) {
      for (size_t j = 0; j < i; ++j) {
        if (loopVerts[i] != loopVerts[j]) continue;
        std::vector<int> simple(loopVerts.begin() + j, loopVerts.begin() + i);
        PushLoopIfNondegenerate(verts, simple, polys);
        loopVerts.erase(loopVerts.begin() + j + 1, loopVerts.begin() + i + 1);
        split = true;
        break;
      }
    }
    if (!split) break;
  }
  PushLoopIfNondegenerate(verts, loopVerts, polys);
}

void AppendInput(const Polygons& polys, int mult, std::vector<vec2>& verts,
                 std::vector<EdgeM>& edges) {
  for (const auto& loop : polys) {
    if (loop.size() < 3) continue;
    const int base = static_cast<int>(verts.size());
    const int n = static_cast<int>(loop.size());
    for (const auto& v : loop) verts.push_back(v);
    for (int i = 0; i < n; ++i) {
      edges.push_back({base + i, base + ((i + 1) % n), mult});
    }
  }
}

// `bSign` is +1 for Add/Intersect-style accumulation and -1 for Subtract.
Polygons ApplyFillRule(const Polygons& a, const Polygons& b, int bSign,
                       WindRule rule, double eps) {
  DEBUG_ASSERT(bSign == 1 || bSign == -1, logicErr,
               "Boolean2 input multiplicity must be +/-1");
  if (!AllFinite(a) || !AllFinite(b)) return {};
  // No local-origin recenter: the inferred eps is position-inclusive
  // (bBox.Scale()), matching Boolean3's MaxEpsilon and the polygon
  // triangulator, so the arrangement stays resolvable far from origin.
  if (eps <= 0.0) eps = InferEps(a, b);

  std::vector<vec2> verts;
  std::vector<EdgeM> edges;
  AppendInput(a, 1, verts, edges);
  AppendInput(b, bSign, verts, edges);
  if (verts.empty()) return {};

  OverlapResult r = RemoveOverlaps2D(verts, edges, eps, /*debug=*/false, rule);
  return OutEdgesToPolygons(r.verts, r.edges);
}

}  // namespace

// Flatten manifold::Polygons into the lower-level (verts, edges) input.
// Each loop becomes a sequence of edges with mult=+1.
std::pair<std::vector<vec2>, std::vector<EdgeM>> PolygonsToInput(
    const Polygons& polys) {
  std::vector<vec2> verts;
  std::vector<EdgeM> edges;
  AppendInput(polys, 1, verts, edges);
  return {std::move(verts), std::move(edges)};
}

// Walk retained directed sub-edges into regularized polygon loops.
Polygons OutEdgesToPolygons(const std::vector<vec2>& verts,
                            const std::vector<OutEdge>& edges) {
  const int nE = static_cast<int>(edges.size());
  // Per-vertex outgoing edges; the next-pointer loop scans each list with
  // deterministic cross/dot comparisons.
  std::vector<std::vector<int>> outgoing(verts.size());
  for (int i = 0; i < nE; ++i) outgoing[edges[i].v0].push_back(i);
  std::vector<bool> visited(nE, false);
  Polygons polys;
  for (int start = 0; start < nE; ++start) {
    if (visited[start]) continue;
    const int startV = edges[start].v0;
    std::vector<int> loopVerts;
    int cur = start;
    bool closed = false;
    while (cur >= 0 && !visited[cur]) {
      visited[cur] = true;
      loopVerts.push_back(edges[cur].v0);
      const int destV = edges[cur].v1;
      // The next-pointer scan skips visited edges (including `start`), so
      // detect closure by the walk reaching startV rather than by re-selecting
      // the start edge as `next`.
      if (destV == startV) {
        closed = true;
        break;
      }
      if (destV < 0 || destV >= static_cast<int>(outgoing.size()) ||
          outgoing[destV].empty()) {
        cur = -1;
        break;
      }
      // Continue with the smallest positive CCW turn from the reverse incoming
      // edge, the next edge along the boundary loop.
      const vec2 vp = verts[destV];
      const vec2 ref = verts[edges[cur].v0] - vp;
      const auto& lst = outgoing[destV];
      int next = -1;
      vec2 bestDir(0, 0);
      for (int e : lst) {
        if (visited[e]) continue;
        const vec2 d = verts[edges[e].v1] - vp;
        if (next < 0 || CcwTurnLess(ref, d, e, bestDir, next)) {
          next = e;
          bestDir = d;
        }
      }
      cur = next;
    }
    if (!closed) {
      DEBUG_ASSERT(false, logicErr,
                   "retained directed edges must form closed walks");
      continue;
    }
    if (loopVerts.size() >= 3) {
      PushSimpleLoops(verts, std::move(loopVerts), polys);
    }
  }
  return polys;
}

// Apply the Positive (Add) winding rule to one polygon set, regularizing it at
// machine-scale eps with no extra tolerance. This is fill-rule application -
// what construction and Offset use to resolve self-intersections - as opposed
// to CrossSection::Simplify, which decimates at a user-designated tolerance.
Polygons ApplyFillRule(const Polygons& polys, double eps) {
  return ApplyFillRule(polys, {}, 1, WindRule::Add, eps);
}

// Infer eps from a polygon set's absolute coordinate scale via Smith's
// alpha-budget formula. Position-inclusive (bBox.Scale() = max abs coordinate),
// matching Boolean3's MaxEpsilon and the polygon triangulator: eps tracks
// absolute magnitude so the arrangement stays resolvable far from origin
// without a recenter. Not translation invariant by design.
double InferEps(const Polygons& a, const Polygons& b) {
  Rect box;
  AccumulateBounds(a, box);
  AccumulateBounds(b, box);
  if (!box.IsFinite()) return 0.0;
  return EpsilonFromScale(box.Scale());
}

// Binary boolean over one combined edge set; Subtract flips B's multiplicity.
Polygons Boolean2D(const Polygons& a, const Polygons& b, OpType op,
                   double eps) {
  const int bSign = op == OpType::Subtract ? -1 : 1;
  const WindRule rule =
      op == OpType::Intersect ? WindRule::Intersect : WindRule::Add;
  return ApplyFillRule(a, b, bSign, rule, eps);
}

// ===== BVH =====
// BVH helpers for eps-padded 2D box queries.

namespace {

constexpr int kRadixTreeBuildGrainSize = 10000;

}  // namespace

Box2 BoxOf2DEdge(vec2 p0, vec2 p1, double eps) {
  const vec2 pad(eps, eps);
  Box2 b(p0, p1);
  return Box2(b.min - pad, b.max + pad);
}

uint32_t MortonCode2(vec2 position, Box2 bBox) {
  const vec2 size = bBox.max - bBox.min;
  const double xNorm = size.x > 0 ? (position.x - bBox.min.x) / size.x : 0.5;
  const double yNorm = size.y > 0 ? (position.y - bBox.min.y) / size.y : 0.5;
  const double xClamped = std::min(1023.0, std::max(0.0, 1024.0 * xNorm));
  const double yClamped = std::min(1023.0, std::max(0.0, 1024.0 * yNorm));
  const uint32_t x =
      collider_internal::SpreadBits3(static_cast<uint32_t>(xClamped));
  const uint32_t y =
      collider_internal::SpreadBits3(static_cast<uint32_t>(yClamped));
  return x * 2 + y;
}

BVH BVHBuildFromBoxes(const std::vector<Box2>& boxes) {
  const int n = static_cast<int>(boxes.size());
  BVH out;
  out.leafToOrig.resize(n);
  for (int i = 0; i < n; ++i) out.leafToOrig[i] = i;
  if (n == 0) return out;
  Box2 bbox = boxes[0];
  for (const auto& b : boxes) bbox = bbox.Union(b);
  std::vector<uint32_t> morton(n);
  for (int i = 0; i < n; ++i) morton[i] = MortonCode2(boxes[i].Center(), bbox);
  manifold::stable_sort(out.leafToOrig.begin(), out.leafToOrig.end(),
                        [&](int a, int b) { return morton[a] < morton[b]; });
  std::vector<uint32_t> sortedMorton(n);
  for (int i = 0; i < n; ++i) {
    sortedMorton[i] = morton[out.leafToOrig[i]];
  }
  const int numNodes = 2 * n - 1;
  out.nodeBBox.resize(numNodes);
  std::vector<int> nodeParent(numNodes, -1);
  out.internalChildren.resize(n - 1, std::make_pair(-1, -1));
  // Radix-tree node creation does little work per index, so use a coarser
  // grain than BVH query traversal/narrow predicates to avoid scheduling
  // overhead dominating construction.
  manifold::for_each_n(
      autoPolicy(n - 1, kRadixTreeBuildGrainSize), countAt(0), n - 1,
      collider_internal::CreateRadixTree(
          {VecView<int>(nodeParent.data(), nodeParent.size()),
           VecView<std::pair<int, int>>(out.internalChildren.data(),
                                        out.internalChildren.size()),
           VecView<const uint32_t>(sortedMorton)}));
  for (int i = 0; i < n; ++i)
    out.nodeBBox[collider_internal::Leaf2Node(i)] = boxes[out.leafToOrig[i]];
  auto buildNode = [&](auto&& self, int node) -> Box2 {
    if (collider_internal::IsLeaf(node)) return out.nodeBBox[node];
    const auto [left, right] =
        out.internalChildren[collider_internal::Node2Internal(node)];
    out.nodeBBox[node] = self(self, left).Union(self(self, right));
    return out.nodeBBox[node];
  };
  if (n > 1) buildNode(buildNode, collider_internal::kRoot);
  return out;
}

// ===== Vertex merge =====
// Vertex-merge and edge-collapse passes. MergeVerts buckets verts within eps
// of each other onto an existing representative vertex; RemapAndCollapse drops
// any input edge whose endpoints merged to the same vert. Both run before the
// edge-pair broad phase.

VertexMerge MergeVerts(const std::vector<vec2>& in, double eps) {
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
        if (la::all(la::lequal(la::abs(d), vec2(2 * eps))))
          pairs.emplace_back(i, j);
      }
    }
  } else {
    // Sort by x and scan forward until x-distance exceeds 2*eps. Pairs are
    // sorted at the end so the unite step below stays deterministic.
    thread_local static std::vector<int> idx;
    idx.resize(n);
    std::iota(idx.begin(), idx.end(), 0);
    manifold::stable_sort(idx.begin(), idx.end(),
                          [&](int a, int b) { return in[a].x < in[b].x; });
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
  }
  // Fast path: no candidates means no merges, identity inputVert2Merged.
  if (pairs.empty()) {
    std::vector<int> inputVert2Merged(n);
    std::iota(inputVert2Merged.begin(), inputVert2Merged.end(), 0);
    return {std::move(inputVert2Merged), in};
  }
  // Parallelize the geometric distance gate (read-only on `in`); unite
  // serially in sorted pair order so cluster roots are deterministic
  // regardless of thread scheduling.
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
    std::vector<int> inputVert2Merged(n);
    std::iota(inputVert2Merged.begin(), inputVert2Merged.end(), 0);
    return {std::move(inputVert2Merged), in};
  }
  // Transitive proximity merge: pick an existing component vertex so a second
  // MergeVerts pass cannot create a new within-eps pair via centroid drift.
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
  std::vector<int> inputVert2Merged(n);
  for (int i = 0; i < n; ++i) inputVert2Merged[i] = rootToNew[uf.find(i)];
  return {std::move(inputVert2Merged), std::move(verts)};
}

// Drop edges whose endpoints map to the same vertex after MergeVerts.
std::vector<EdgeM> RemapAndCollapse(const std::vector<EdgeM>& edges,
                                    const std::vector<int>& inputVert2Merged) {
  std::vector<EdgeM> out;
  out.reserve(edges.size());
  for (const auto& e : edges) {
    int a = inputVert2Merged[e.v0];
    int b = inputVert2Merged[e.v1];
    if (a != b) out.push_back({a, b, e.mult});
  }
  return out;
}

// ===== Edge-vertex lists =====
// Edge-vertex list construction: for each input edge, the sorted-by-parameter
// list of vertices that lie within eps of the edge interior. Candidates are
// derived from the same edge-pair broad phase used for intersections.

namespace {

// Pair-count threshold above which the combined helper uses TBB. Smaller
// inputs stay serial to avoid setup overhead.
constexpr size_t kFusedNarrowParallelMin = 1024;

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
                       std::vector<EdgeGeom>& edgeG) {
  const int nE = static_cast<int>(edges.size());
  edgeG.resize(nE);
  for (int e = 0; e < nE; ++e) {
    edgeG[e].a = verts[edges[e].v0];
    edgeG[e].ab = verts[edges[e].v1] - edgeG[e].a;
    edgeG[e].abLen2 = dot(edgeG[e].ab, edgeG[e].ab);
  }
}

void RecordEdgeVertHit(const std::vector<EdgeM>& edges,
                       const std::vector<vec2>& verts,
                       const std::vector<EdgeGeom>& edgeG, double eps2, int v,
                       int e, std::vector<EdgeVertHit>& hits) {
  if (v == edges[e].v0 || v == edges[e].v1) return;
  const auto& g = edgeG[e];
  if (g.abLen2 == 0) return;

  const vec2 ap = verts[v] - g.a;
  const double dotAB = dot(ap, g.ab);
  if (dotAB <= 0 || dotAB >= g.abLen2) return;
  const double cross = la::cross(ap, g.ab);
  const double cross2 = cross * cross;
  const double eps2_abLen2 = eps2 * g.abLen2;
  if (cross2 > eps2_abLen2) return;
  hits.push_back({e, dotAB / g.abLen2, v});
}

void ProcessEdgePair(const std::vector<EdgeM>& edges,
                     const std::vector<vec2>& verts,
                     const std::vector<EdgeGeom>& edgeG, double eps,
                     const std::pair<int, int>& pr,
                     std::vector<EdgeVertHit>& hits) {
  const int i = pr.first;
  const int j = pr.second;
  const auto& ei = edges[i];
  const auto& ej = edges[j];
  const double eps2 = eps * eps;
  RecordEdgeVertHit(edges, verts, edgeG, eps2, ei.v0, j, hits);
  RecordEdgeVertHit(edges, verts, edgeG, eps2, ei.v1, j, hits);
  RecordEdgeVertHit(edges, verts, edgeG, eps2, ej.v0, i, hits);
  RecordEdgeVertHit(edges, verts, edgeG, eps2, ej.v1, i, hits);
}

void MaterializeEdgeVertLists(int nE, std::vector<EdgeVertHit>& flatHits,
                              std::vector<std::vector<int>>& lists) {
  lists.assign(nE, {});
  manifold::stable_sort(flatHits.begin(), flatHits.end(),
                        [](const EdgeVertHit& a, const EdgeVertHit& b) {
                          if (a.e != b.e) return a.e < b.e;
                          if (a.t != b.t) return a.t < b.t;
                          return a.v < b.v;
                        });
  for (size_t i = 0; i < flatHits.size();) {
    const int e = flatHits[i].e;
    size_t j = i;
    while (j < flatHits.size() && flatHits[j].e == e) ++j;
    auto& lst = lists[e];
    lst.reserve(j - i);
    int lastV = -1;
    for (size_t k = i; k < j; ++k) {
      if (flatHits[k].v == lastV) continue;
      lst.push_back(flatHits[k].v);
      lastV = flatHits[k].v;
    }
    i = j;
  }
}

}  // namespace

std::vector<std::vector<int>> BuildIncidenceLists(
    const std::vector<EdgeM>& edges, const std::vector<vec2>& verts, double eps,
    const std::vector<std::pair<int, int>>& pairs) {
  const int nE = static_cast<int>(edges.size());
  std::vector<std::vector<int>> lists;

  // Per-calling-thread scratch; workers read these only through const refs.
  thread_local static std::vector<EdgeGeom> edgeG;
  BuildEdgeGeometry(edges, verts, edgeG);
  const auto& edgeGRef = edgeG;

  std::vector<EdgeVertHit> flatHits;

#if (MANIFOLD_PAR == 1)
  if (pairs.size() >= kFusedNarrowParallelMin) {
    tbb::combinable<std::vector<EdgeVertHit>> tls;
    manifold::for_each_n(autoPolicy(pairs.size(), kFineParallelGrainSize),
                         countAt(size_t{0}), pairs.size(), [&](size_t idx) {
                           ProcessEdgePair(edges, verts, edgeGRef, eps,
                                           pairs[idx], tls.local());
                         });
    tls.combine_each([&](const std::vector<EdgeVertHit>& l) {
      flatHits.insert(flatHits.end(), l.begin(), l.end());
    });
  } else
#endif
  {
    for (const auto& pr : pairs) {
      ProcessEdgePair(edges, verts, edgeGRef, eps, pr, flatHits);
    }
  }
  MaterializeEdgeVertLists(nE, flatHits, lists);
  return lists;
}

// ===== Broad phase =====
// Edge-pair broad phase: collect the candidate edge pairs whose eps-padded
// boxes overlap. The pairs drive the incidence narrow phase (vertex-on-edge
// split lists); the sweep does its own crossing discovery.

namespace {

// Encode (first, second) as uint64 and run manifold::stable_sort, which
// dispatches to the LSB-radix specialization for integral types - much
// faster than the comparator path for pair<int,int>. Casts through
// uint32_t preserve non-negative edge indices and put `first` in the
// high bits so int64 order matches lexicographic pair order.
void RadixSortPairs(std::vector<std::pair<int, int>>& pairs) {
  const size_t n = pairs.size();
  if (n < 2) return;
  thread_local static std::vector<uint64_t> encoded;
  encoded.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const auto& pr = pairs[i];
    encoded[i] =
        (static_cast<uint64_t>(static_cast<uint32_t>(pr.first)) << 32) |
        static_cast<uint32_t>(pr.second);
  }
  manifold::stable_sort(encoded.begin(), encoded.end());
  for (size_t i = 0; i < n; ++i) {
    pairs[i] = {static_cast<int>(encoded[i] >> 32),
                static_cast<int>(encoded[i] & 0xFFFFFFFFu)};
  }
}

void SortSmallInts(std::vector<int>& values) {
  if (values.size() < 32) {
    for (size_t i = 1; i < values.size(); ++i) {
      const int x = values[i];
      size_t j = i;
      while (j > 0 && x < values[j - 1]) {
        values[j] = values[j - 1];
        --j;
      }
      values[j] = x;
    }
  } else {
    manifold::stable_sort(values.begin(), values.end());
  }
}

// Drop shared-endpoint pairs when both non-shared endpoints are more than eps
// from the opposite edge line; then no near-line sliver split is possible.
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
  const vec2 dA = pA - pS;
  const vec2 dB = pB - pS;
  const double cross = la::cross(dA, dB);
  const double cross2 = cross * cross;
  const double lenA2 = dot(dA, dA);
  const double lenB2 = dot(dB, dB);
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
                              std::vector<std::pair<int, int>>& pairs) {
  const int nE = static_cast<int>(edges.size());
  pairs.clear();
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
    if (static_cast<int>(byFirst.size()) < nE) byFirst.resize(nE);
    for (int i = 0; i < nE; ++i) byFirst[i].clear();
    int numPairs = 0;
    for (int oi = 0; oi < nE; ++oi) {
      const int i = order[oi];
      const Box2& bi = edgeBoxes[i];
      for (int oj = oi + 1; oj < nE; ++oj) {
        const int j = order[oj];
        const Box2& bj = edgeBoxes[j];
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
    pairs.reserve(numPairs);
    for (int first = 0; first < nE; ++first) {
      auto& seconds = byFirst[first];
      if (seconds.empty()) continue;
      SortSmallInts(seconds);
      for (int second : seconds) pairs.emplace_back(first, second);
    }
    return;
  }
#if (MANIFOLD_PAR == 1)
  PairsRecorder rec{bvh.leafToOrig, edges, verts, eps, {}};
  auto qf = [&](int i) { return edgeBoxes[i]; };
  BVHCollisions(bvh, rec, qf, nE, /*parallel=*/true);
  rec.tls.combine_each([&](const auto& localPairs) {
    pairs.insert(pairs.end(), localPairs.begin(), localPairs.end());
  });
  RadixSortPairs(pairs);
#else
  CollidePairs(bvh, edgeBoxes, [&](int qi, int li) {
    if (qi >= li) return;
    if (SharedEndpointSafelySkippable(edges[qi], edges[li], verts, eps)) return;
    pairs.emplace_back(qi, li);
  });
  RadixSortPairs(pairs);
#endif
}

// ===== Driver =====
// End-to-end Boolean2 driver: vertex merge, edge collapse, incidence pre-split,
// then the Smith sweep-line (arrangement + winding). Returns an OverlapResult
// holding the merged-vert list, the retained directed sub-edges, the
// inputVert2Merged remap, and the merged-vert count.

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
  // Broad phase: candidate edge pairs whose eps-padded boxes overlap.
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
  {
    ScopedTiming timing(P.broadPairWorkNs);
    CollectIntersectionPairs(edges, merge.verts, eps, edgeBoxes, bvh,
                             intersectionPairs);
  }
  // Incidence pre-split: split each edge at the input vertices lying within eps
  // of its interior. The sweep's own crossing tests are exact, so this
  // eps-scale vertex-on-edge quantization is what the sweep cannot reproduce.
  std::vector<std::vector<int>> incidenceLists;
  {
    ScopedTiming timing(P.narrowPhaseNs);
    incidenceLists =
        BuildIncidenceLists(edges, merge.verts, eps, intersectionPairs);
  }
  {
    std::vector<EdgeM> subEdges;
    subEdges.reserve(edges.size());
    for (size_t e = 0; e < edges.size(); ++e) {
      int prev = edges[e].v0;
      for (int v : incidenceLists[e]) {
        if (v != prev) subEdges.push_back({prev, v, edges[e].mult});
        prev = v;
      }
      if (edges[e].v1 != prev)
        subEdges.push_back({prev, edges[e].v1, edges[e].mult});
    }
    edges = std::move(subEdges);
  }
  // Smith sweep-line arrangement + winding over the incidence-split sub-edges.
  std::vector<OutEdge> out;
  {
    ScopedTiming timing(P.filterWindingNs);
    out = SweepWinding(edges, merge.verts, pred);
  }
  traceRecorder.RecordFilteredOutput(merge.verts, out);
  CountTimingCase();
  return {std::move(merge.verts), std::move(out),
          std::move(merge.inputVert2Merged), numMerged};
}

}  // namespace manifold

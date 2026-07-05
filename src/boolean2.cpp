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
#include <map>
#include <numeric>
#include <queue>
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

Box2 BoxOf2DPoint(vec2 p, double eps) {
  const vec2 pad(eps, eps);
  return Box2(p - pad, p + pad);
}

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

// ===== Canonicalize =====

CanonicalSubEdges Canonicalize(const std::vector<EdgeM>& edges,
                               const std::vector<std::vector<int>>& lists) {
  CanonicalSubEdges out;
  // Pre-reserve. Each input edge contributes (1 + lists[e].size()) sub-edges.
  size_t total = edges.size();
  for (const auto& l : lists) total += l.size();
  out.edges.reserve(total);
  for (size_t e = 0; e < edges.size(); ++e) {
    int prev = edges[e].v0;
    for (int v : lists[e]) {
      out.Add(prev, v, edges[e].mult);
      prev = v;
    }
    out.Add(prev, edges[e].v1, edges[e].mult);
  }
  out.Finalize();
  return out;
}

// ===== Vertex merge =====
// Vertex-merge and edge-collapse passes. MergeVerts buckets verts within eps
// of each other onto an existing representative vertex;
// CollapseDegenerateEdges drops any input edge whose endpoints merged to the
// same vert. Both run before the BVH-broad intersection discovery. Also hosts
// the tiny "sorted-vector as set" helpers (VESetContains, VESetInsert) used by
// the per-edge / per-vert adjacency tracking below.

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

// Per-edge split lists and per-vertex edge-incidence lists both hold small sets
// of int ids. Almost always 2-4 elements; occasionally larger at concurrent
// intersection points. A sorted std::vector<int> beats a std::set<int> by 5-10x
// on per-op cost for sets this small (no node allocation, no tree rebalancing,
// contiguous memory). Helpers keep the "set" semantics: idempotent insert, fast
// contains, ordered iteration.
bool VESetContains(const std::vector<int>& vec, int x) {
  return std::binary_search(vec.begin(), vec.end(), x);
}
void VESetInsert(std::vector<int>& vec, int x) {
  auto it = std::lower_bound(vec.begin(), vec.end(), x);
  if (it == vec.end() || *it != x) vec.insert(it, x);
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

// Visit every crossing among the piece combos of edges i and j, on the
// incidence-split PIECES (the chains Canonicalize emits): two eps-bent chains
// can cross where their straight parents do not, so detecting on the parents
// would miss it. Same-vertex combos are skipped (the whole-edge shared-endpoint
// skip for straight pairs); the unpadded-box pre-filter only drops combos the
// kernel would reject. A hit is the crossing point and its piece positions
// (insertion indices into lists[i]/lists[j]).
struct PieceHit {
  vec2 q;
  int si, sj;
};

void ForEachPieceCrossing(const std::vector<EdgeM>& edges,
                          const std::vector<vec2>& verts,
                          const std::vector<std::vector<int>>& lists,
                          double eps, int i, int j,
                          std::vector<PieceHit>& hits) {
  hits.clear();
  struct Piece {
    Box2 box;
    int v0, v1;
  };
  // Per-calling-thread scratch: edge j's pieces are boxed once, not once per
  // i-piece.
  thread_local static std::vector<Piece> jPieces;
  jPieces.clear();
  const auto& lj = lists[j];
  int prevJ = edges[j].v0;
  for (size_t sj = 0; sj <= lj.size(); ++sj) {
    const int endJ = sj < lj.size() ? lj[sj] : edges[j].v1;
    jPieces.push_back({Box2(verts[prevJ], verts[endJ]), prevJ, endJ});
    prevJ = endJ;
  }
  const auto& li = lists[i];
  int prevI = edges[i].v0;
  for (size_t si = 0; si <= li.size(); ++si) {
    const int endI = si < li.size() ? li[si] : edges[i].v1;
    if (endI != prevI) {
      const Box2 boxI(verts[prevI], verts[endI]);
      for (size_t sj = 0; sj < jPieces.size(); ++sj) {
        const Piece& pc = jPieces[sj];
        if (pc.v1 != pc.v0 && prevI != pc.v0 && prevI != pc.v1 &&
            endI != pc.v0 && endI != pc.v1 && boxI.DoesOverlap(pc.box)) {
          vec2 q;
          if (IntersectSegments({verts[prevI], verts[endI], i},
                                {verts[pc.v0], verts[pc.v1], j}, eps, q)) {
            hits.push_back({q, static_cast<int>(si), static_cast<int>(sj)});
          }
        }
      }
    }
    prevI = endI;
  }
}

// Crossing candidates for one pair over its current pieces (see
// ForEachPieceCrossing).
void PieceCrossings(const std::vector<EdgeM>& edges,
                    const std::vector<vec2>& verts,
                    const std::vector<std::vector<int>>& lists, double eps,
                    const std::pair<int, int>& pr,
                    std::vector<IntersectionPoint>& intersections) {
  thread_local static std::vector<PieceHit> hits;
  ForEachPieceCrossing(edges, verts, lists, eps, pr.first, pr.second, hits);
  for (const PieceHit& h : hits)
    intersections.push_back({pr.first, pr.second, h.q});
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

NarrowPhaseResult BuildListsAndFindIntersections(
    const std::vector<EdgeM>& edges, const std::vector<vec2>& verts, double eps,
    const std::vector<std::pair<int, int>>& pairs, bool findCrossings) {
  const int nE = static_cast<int>(edges.size());
  NarrowPhaseResult result;

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
  MaterializeEdgeVertLists(nE, flatHits, result.lists);
  if (!findCrossings) return result;

  // Crossing detection needs the finished incidence lists (see
  // PieceCrossings), so it runs as a second read-only pass over the pairs.
#if (MANIFOLD_PAR == 1)
  if (pairs.size() >= kFusedNarrowParallelMin) {
    tbb::combinable<std::vector<IntersectionPoint>> tls;
    manifold::for_each_n(autoPolicy(pairs.size(), kFineParallelGrainSize),
                         countAt(size_t{0}), pairs.size(), [&](size_t idx) {
                           PieceCrossings(edges, verts, result.lists, eps,
                                          pairs[idx], tls.local());
                         });
    tls.combine_each([&](const std::vector<IntersectionPoint>& l) {
      result.intersections.insert(result.intersections.end(), l.begin(),
                                  l.end());
    });
  } else
#endif
  {
    for (const auto& pr : pairs) {
      PieceCrossings(edges, verts, result.lists, eps, pr, result.intersections);
    }
  }
  return result;
}

// ===== Intersections =====
// Edge-edge intersection discovery: broad phase collects candidate edge
// pairs; the edge-vertex narrow phase precomputes proper intersections;
// FindAndInsertIntersections then inserts them in lexicographic point order,
// recomputing each against the current pieces and sharing constructed
// vertices by exact coordinate equality.

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

IntersectionInsertion FindAndInsertIntersections(
    const std::vector<EdgeM>& edges, std::vector<vec2> verts,
    std::vector<std::vector<int>> lists, double eps,
    const std::vector<Box2>& edgeBoxes, const BVH& bvh,
    const std::vector<IntersectionPoint>& precomputedIntersections) {
  const int nE = static_cast<int>(edges.size());
  std::vector<std::vector<int>> vertEdges;
  vertEdges.resize(verts.size());

  // Serial point-order insertion: the precomputed crossings seed a queue
  // ordered by lexicographic point, and each pop recomputes against the
  // current pieces of both edges so every decision sees the earlier splits.
  // Constructed vertices are shared only on exact bit-equal coordinates, never
  // aliased onto a nearby one - aliasing constructed points at eps is what made
  // clustered crossings collapse inconsistently.
  if (precomputedIntersections.empty())
    return {std::move(verts), std::move(lists)};

  auto lexLess = [](vec2 a, vec2 b) {
    return a.x != b.x ? a.x < b.x : a.y < b.y;
  };

  // Exact-coordinate vertex identity, smallest id wins. Only ever used for
  // point-equality lookups, so iteration order does not matter.
  // Exact-coordinate vertex identity. Keys are finite (inputs pass AllFinite,
  // constructed points pass the finite assert below), so value equality on the
  // coordinate pair is exact; -0.0 and +0.0 compare equal and share a slot.
  // Must be GLOBAL: a crossing can bit-equal a vertex lying on neither of its
  // two edges (a chain can bend into an old vertex after that vertex's attach
  // pass ran, constructibly on lattice inputs), so identity is not resolvable
  // from the two chains alone. A hash map on the same key would do if this
  // shows up in profiles.
  std::map<std::pair<double, double>, int> exactVert;
  for (int v = 0; v < static_cast<int>(verts.size()); ++v) {
    exactVert.emplace(std::make_pair(verts[v].x, verts[v].y), v);
  }

  // Squared distance from p to edge eIdx's parent segment (clamped
  // projection).
  auto parentSegDist2 = [&](int eIdx, vec2 p) {
    const vec2 e0 = verts[edges[eIdx].v0];
    const vec2 dE = verts[edges[eIdx].v1] - e0;
    const double lenE2 = dot(dE, dE);
    const double tE =
        lenE2 > 0 ? la::clamp(dot(p - e0, dE) / lenE2, 0.0, 1.0) : 0.0;
    return la::length2(p - (e0 + dE * tE));
  };

  // Max distance of any chain vert from its parent segment (eps floor for
  // incidence verts; constructed verts can compound ~alpha of drift per attach
  // generation). Measured, not assumed, so the whole-edge attach rejects below
  // stay exact under arbitrary compounding.
  std::vector<double> chainSlack(edges.size(), eps);

  // Insert v into edge e's chain before lists[e][pos]. Order is POSITIONAL:
  // a bent chain's straight-chord parameter does not order its verts (a
  // crossing can project outside its own piece's span), so the piece it was
  // found on picks the slot, never a parameter sort.
  auto insertAtPiece = [&](int eIdx, size_t pos, int v) {
    if (v == edges[eIdx].v0 || v == edges[eIdx].v1) return;
    auto& lst = lists[eIdx];
    if (std::find(lst.begin(), lst.end(), v) != lst.end()) return;
    lst.insert(lst.begin() + pos, v);
    VESetInsert(vertEdges[v], eIdx);
    chainSlack[eIdx] =
        std::max(chainSlack[eIdx], std::sqrt(parentSegDist2(eIdx, verts[v])));
  };

  // A constructed vertex splits another edge only where a PIECE of it passes
  // within the intersection construction's error bound (Smith's alpha =
  // EpsilonFromScale at budget 0, vs the feature-scale eps at budget 1000) of
  // the vertex. The scale spans the edge endpoints, not just the point, since
  // the on-piece test reconstructs a + ab*t. Attaching at the feature eps
  // instead bends pieces after their pair's crossing decision - re-creating
  // eps-deep crossings no later decision sees, and fanning out on dense
  // clusters.
  int64_t nAttached = 0;
  auto attachNarrow = [&](int v, int eIdx) {
    if (v == edges[eIdx].v0 || v == edges[eIdx].v1) return;
    if (VESetContains(vertEdges[v], eIdx)) return;  // already incident
    const vec2 p = verts[v];
    // Conservative pre-filter bound: chain verts lie within chainSlack of the
    // parent, so no piece's alpha bound exceeds the bound at the padded-box
    // corners inflated by the slack. Pieces farther than this from p cannot
    // pass the exact per-piece test below.
    const Box2& eBox = edgeBoxes[eIdx];
    const double slack = chainSlack[eIdx];
    const double preBound = EpsilonFromScale(
        la::maxelem(la::max(la::abs(p),
                            la::max(la::abs(eBox.min), la::abs(eBox.max)))) +
            slack,
        0);
    // Whole-edge rejects: every piece point stays within chainSlack of the
    // parent, so a point beyond preBound + slack of it reaches no piece. The
    // multiplicative pad keeps the reject conservative against the ulp rounding
    // of the two measured distances.
    const double margin = (preBound + slack) * (1 + 4 * kU);
    if (p.x < eBox.min.x - margin || p.x > eBox.max.x + margin ||
        p.y < eBox.min.y - margin || p.y > eBox.max.y + margin)
      return;
    if (parentSegDist2(eIdx, p) > margin * margin) return;
    bool found = false;
    size_t bestPos = 0;
    double bestDist2 = 0;
    int prev = edges[eIdx].v0;
    for (size_t s = 0; s <= lists[eIdx].size(); ++s) {
      const int end = s < lists[eIdx].size() ? lists[eIdx][s] : edges[eIdx].v1;
      if (end != prev && v != prev && v != end) {
        const vec2 a = verts[prev];
        const vec2 b = verts[end];
        if (p.x < std::min(a.x, b.x) - preBound ||
            p.x > std::max(a.x, b.x) + preBound ||
            p.y < std::min(a.y, b.y) - preBound ||
            p.y > std::max(a.y, b.y) + preBound) {
          prev = end;
          continue;
        }
        const vec2 ab = b - a;
        const double abLen2 = dot(ab, ab);
        if (abLen2 > 0) {
          const double t = dot(p - a, ab) / abLen2;
          if (t > 0 && t < 1) {
            const vec2 d = p - (a + ab * t);
            const double dist2 = dot(d, d);
            const double scale = la::maxelem(
                la::max(la::abs(p), la::max(la::abs(a), la::abs(b))));
            const double bound = EpsilonFromScale(scale, 0);
            if (dist2 <= bound * bound && (!found || dist2 < bestDist2)) {
              found = true;
              bestPos = s;
              bestDist2 = dist2;
            }
          }
        }
      }
      prev = end;
    }
    if (found) {
      insertAtPiece(eIdx, bestPos, v);
      ++nAttached;
    }
  };
  // Query at eps (the candidate set the padded edge boxes already encode); the
  // alpha gate above makes the decision. Reach is bounded by these fixed eps
  // pads, so a chain drifted past ~2*eps of its parent could be acceptable yet
  // undiscovered - shared with the propagation this replaced.
  auto attachNewVert = [&](int v) {
    const Box2 queryBox = BoxOf2DPoint(verts[v], eps);
    if (bvh.leafToOrig.empty()) {
      for (int e = 0; e < nE; ++e) {
        if (queryBox.DoesOverlap(edgeBoxes[e])) attachNarrow(v, e);
      }
    } else {
      auto adapter = [&](int, int leafIdx) {
        attachNarrow(v, bvh.leafToOrig[leafIdx]);
      };
      auto recorder = MakeSimpleRecorder(adapter);
      auto qf = [&](int) { return queryBox; };
      BVHCollisions(bvh, recorder, qf, 1, /*parallel=*/false);
    }
  };

  // Recompute pair (i, j)'s crossing against the current pieces and return the
  // lex-smallest one with its piece positions. All combos matter: earlier
  // splits move the point by ulps, enough to cross into an adjacent piece in a
  // dense tangle. O(|pieces_i| * |pieces_j|) per pop is known-slow on soups
  // whose crossing count is quadratic in the edges - worth an interval sweep
  // over the piece spans if those show up outside fuzzing.
  auto currentCrossing = [&](int i, int j, vec2& best, size_t& posI,
                             size_t& posJ) {
    thread_local static std::vector<PieceHit> hits;
    ForEachPieceCrossing(edges, verts, lists, eps, i, j, hits);
    bool found = false;
    for (const PieceHit& h : hits) {
      if (!found || lexLess(h.q, best)) {
        best = h.q;
        posI = h.si;
        posJ = h.sj;
        found = true;
      }
    }
    return found;
  };

  // Min-heap over (point lex, edge pair). Every queued point came out of
  // IntersectSegments, which guarantees finite coordinates, so the comparator
  // is a strict weak order; the edge-pair tie-break makes the pop order a
  // function of the event multiset alone, independent of the order the
  // parallel crossing pass combined its per-thread results.
  struct Event {
    vec2 p;
    int i, j;
  };
  auto later = [](const Event& a, const Event& b) {
    if (a.p.x != b.p.x) return a.p.x > b.p.x;
    if (a.p.y != b.p.y) return a.p.y > b.p.y;
    if (a.i != b.i) return a.i > b.i;
    return a.j > b.j;
  };
  std::priority_queue<Event, std::vector<Event>, decltype(later)> queue(later);
  for (const IntersectionPoint& ip : precomputedIntersections) {
    DEBUG_ASSERT(la::all(la::isfinite(ip.p)), logicErr,
                 "Boolean2 crossing event has non-finite coordinates");
    queue.push({ip.p, ip.i, ip.j});
  }

  int64_t nGone = 0, nRequeued = 0, nApplied = 0, nReused = 0;
  while (!queue.empty()) {
    const Event ev = queue.top();
    queue.pop();
    const int i = ev.i;
    const int j = ev.j;
    vec2 q;
    size_t posI = 0, posJ = 0;
    if (!currentCrossing(i, j, q, posI, posJ)) {
      ++nGone;  // resolved by earlier splits
      continue;
    }
    // If the recomputed crossing moved past the next event, let the
    // intervening events decide first (a backward or in-place move cannot sort
    // after the heap minimum, so this fires only on forward moves). Backward
    // moves apply immediately - decisions are never committed by position.
    // This cannot cycle: a requeue only re-pops after strictly earlier events
    // apply, splits only accumulate, and applies are bounded by the finite
    // crossing count, so the queue drains.
    if (!queue.empty() && later({q, i, j}, queue.top())) {
      queue.push({q, i, j});
      ++nRequeued;
      continue;
    }

    const std::pair<double, double> key(q.x, q.y);
    const auto it = exactVert.find(key);
    const bool isNew = it == exactVert.end();
    int vNew;
    if (isNew) {
      vNew = static_cast<int>(verts.size());
      verts.push_back(q);
      vertEdges.emplace_back();
      exactVert.emplace(key, vNew);
      ++nApplied;
    } else {
      vNew = it->second;
      ++nReused;
    }
    insertAtPiece(i, posI, vNew);
    insertAtPiece(j, posJ, vNew);
    VESetInsert(vertEdges[vNew], i);
    VESetInsert(vertEdges[vNew], j);
    if (isNew) attachNewVert(vNew);
  }
  if (TimingEnabled()) {
    auto& P = GlobalPhases();
    P.insertSeeds += static_cast<int64_t>(precomputedIntersections.size());
    P.insertGone += nGone;
    P.insertRequeued += nRequeued;
    P.insertApplied += nApplied;
    P.insertReused += nReused;
    P.insertAttached += nAttached;
  }
  return {std::move(verts), std::move(lists)};
}

// ===== Driver =====
// End-to-end Boolean2 driver. Stitches together vertex merging, edge
// collapse, near-vertex indexing, proper edge-edge crossing insertion,
// sub-edge canonicalization, and winding-rule filtering. Returns an
// OverlapResult holding the merged-vert list, the retained directed sub-edges,
// the inputVert2Merged remap, and the merged-vert count.
//
// Crossings are found among two-vertex sub-edges (edges pre-split at their
// incident vertices); FindAndInsertIntersections inserts them in point order
// with exact-coordinate vertex identity, so coincident crossings share a
// vertex and distinct ones stay distinct.

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
  thread_local static std::vector<std::pair<int, int>> intersectionPairs;
  // Collect edge pairs once, then derive both intersection candidates and
  // near-vertex lists from that pair set. In polygon arrangements, a vertex
  // that can split a non-incident edge must have an eps-padded vertex box
  // overlapping that edge's eps-padded box, so it appears as one endpoint of
  // an overlapping edge pair.
  {
    ScopedTiming timing(P.broadPairWorkNs);
    CollectIntersectionPairs(edges, merge.verts, eps, edgeBoxes, bvh,
                             intersectionPairs);
  }
  traceRecorder.RecordBroadPhasePairs(merge.verts, edges, intersectionPairs);
  // Build the arrangement on two-vertex sub-edges: split each edge at its
  // incident vertices, then find crossings among the sub-edges. Splitting first
  // keeps a crossing that lies near an incidence distinct from it.

  // Incidence-only narrow over the whole-edge pairs.
  NarrowPhaseResult incidence;
  {
    ScopedTiming timing(P.narrowPhaseNs);
    incidence = BuildListsAndFindIntersections(
        edges, merge.verts, eps, intersectionPairs, /*findCrossings=*/false);
  }
  // Split each edge at its incident vertices; the sub-edges become the edge
  // set. Lists are sorted and exclude the endpoints, so the guards only skip a
  // zero-length sub-edge.
  {
    std::vector<EdgeM> subEdges;
    subEdges.reserve(edges.size());
    for (size_t e = 0; e < edges.size(); ++e) {
      int prev = edges[e].v0;
      for (int v : incidence.lists[e]) {
        if (v != prev) subEdges.push_back({prev, v, edges[e].mult});
        prev = v;
      }
      if (edges[e].v1 != prev)
        subEdges.push_back({prev, edges[e].v1, edges[e].mult});
    }
    edges = std::move(subEdges);
  }
  edgeBoxes.resize(edges.size());
  for (size_t e = 0; e < edges.size(); ++e)
    edgeBoxes[e] =
        BoxOf2DEdge(merge.verts[edges[e].v0], merge.verts[edges[e].v1], eps);
  // Broad phase and crossing narrow over the sub-edges. Cost scales with
  // sub-edge box overlaps, so a dense near-collinear bundle (one edge split
  // into many pieces) is super-quadratic.
  BVH subBvh;
  {
    ScopedTiming timing(P.bvhBuildNs);
    if (edges.size() >= kEdgePairBvhThreshold)
      subBvh = BVHBuildFromBoxes(edgeBoxes);
  }
  thread_local static std::vector<std::pair<int, int>> subPairs;
  {
    ScopedTiming timing(P.broadPairWorkNs);
    CollectIntersectionPairs(edges, merge.verts, eps, edgeBoxes, subBvh,
                             subPairs);
  }
  NarrowPhaseResult narrow;
  {
    ScopedTiming timing(P.narrowPhaseNs);
    narrow = BuildListsAndFindIntersections(edges, merge.verts, eps, subPairs);
  }
  traceRecorder.RecordEdgeVertLists(merge.verts, edges, narrow.lists);
  // Insert the crossings; the sub-edge BVH accelerates the attach queries for
  // newly constructed vertices.
  IntersectionInsertion inserted;
  {
    ScopedTiming timing(P.findIxNs);
    inserted = FindAndInsertIntersections(
        edges, std::move(merge.verts), std::move(narrow.lists), eps, edgeBoxes,
        subBvh, narrow.intersections);
  }
  merge.verts = std::move(inserted.verts);
  std::vector<std::vector<int>> lists = std::move(inserted.lists);
  traceRecorder.RecordInsertedIntersections(merge.verts, edges, lists);

  // Sub-edge canonicalization.
  CanonicalSubEdges canon;
  {
    ScopedTiming timing(P.canonNs);
    canon = Canonicalize(edges, lists);
  }
  traceRecorder.RecordCanonicalSubedges(merge.verts, canon);
  // Per-edge ray-cast winding filter.
  std::vector<OutEdge> out;
  {
    ScopedTiming timing(P.filterWindingNs);
    out = FilterByWinding(canon, merge.verts, pred);
  }
  traceRecorder.RecordFilteredOutput(merge.verts, out);
  CountTimingCase();
  return {std::move(merge.verts), std::move(out),
          std::move(merge.inputVert2Merged), numMerged};
}

}  // namespace manifold

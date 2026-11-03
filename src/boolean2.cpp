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

// Items grouped by bucket in flat storage, keeping the order they were emitted
// in within each bucket. The constructor calls forEach(emit) twice, first to
// count and then to fill; forEach must call emit(bucket, item) for every entry
// the same way both times.
class BucketIndex {
 public:
  struct Range {
    const int* first;
    const int* last;
    const int* begin() const { return first; }
    const int* end() const { return last; }
    bool empty() const { return first == last; }
  };

  BucketIndex() = default;
  template <typename F>
  BucketIndex(int numBuckets, F&& forEach) : start_(numBuckets + 1, 0) {
    forEach([&](int bucket, int) { ++start_[bucket + 1]; });
    for (int b = 0; b < numBuckets; ++b) start_[b + 1] += start_[b];
    items_.resize(start_[numBuckets]);
    std::vector<int> next(start_.begin(), start_.end() - 1);
    forEach([&](int bucket, int item) { items_[next[bucket]++] = item; });
  }

  Range Bucket(int b) const {
    return {items_.data() + start_[b], items_.data() + start_[b + 1]};
  }

 private:
  std::vector<int> start_;
  std::vector<int> items_;
};

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

void PushLoopIfNondegenerate(const std::vector<vec2>& verts, const int* begin,
                             const int* end, Polygons& polys) {
  if (end - begin >= 3) {
    SimplePolygon loop;
    loop.reserve(end - begin);
    for (const int* v = begin; v != end; ++v) loop.push_back(verts[*v]);
    polys.push_back(std::move(loop));
  }
}

// loopVerts is a scratch buffer, consumed.
void PushSimpleLoops(const std::vector<vec2>& verts,
                     std::vector<int>& loopVerts, Polygons& polys,
                     std::vector<int>& positions) {
  // positions is a shared scratch buffer mapping vertex IDs to active-path
  // indices, or -1 if absent.
  // Each input occurrence enters and leaves the active path at most once.
  size_t size = 0;
  for (size_t read = 0; read < loopVerts.size(); ++read) {
    const int vertex = loopVerts[read];
    const int previous = positions[vertex];
    if (previous < 0) {
      positions[vertex] = static_cast<int>(size);
      loopVerts[size++] = vertex;
      continue;
    }
    PushLoopIfNondegenerate(verts, loopVerts.data() + previous,
                            loopVerts.data() + size, polys);
    for (size_t i = previous + 1; i < size; ++i) positions[loopVerts[i]] = -1;
    size = previous + 1;
  }
  loopVerts.resize(size);
  // Reset remaining entries to empty the shared table for the next contour.
  for (int vertex : loopVerts) positions[vertex] = -1;
  PushLoopIfNondegenerate(verts, loopVerts.data(),
                          loopVerts.data() + loopVerts.size(), polys);
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
  const int nV = static_cast<int>(verts.size());
  // Per-vertex outgoing edges in increasing edge index; the next-pointer loop
  // scans each list with deterministic cross/dot comparisons.
  const BucketIndex outgoing(nV, [&](auto&& emit) {
    for (int i = 0; i < nE; ++i) emit(edges[i].v0, i);
  });
  std::vector<bool> visited(nE, false);
  std::vector<int> positions(verts.size(), -1);
  Polygons polys;
  std::vector<int> loopVerts;
  for (int start = 0; start < nE; ++start) {
    if (visited[start]) continue;
    const int startV = edges[start].v0;
    loopVerts.clear();
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
      if (destV < 0 || destV >= nV || outgoing.Bucket(destV).empty()) {
        cur = -1;
        break;
      }
      // Continue with the smallest positive CCW turn from the reverse incoming
      // edge, the next edge along the boundary loop.
      const vec2 vp = verts[destV];
      const vec2 ref = verts[edges[cur].v0] - vp;
      int next = -1;
      vec2 bestDir(0, 0);
      for (const int e : outgoing.Bucket(destV)) {
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
      PushSimpleLoops(verts, loopVerts, polys, positions);
    }
  }
  return polys;
}

// Apply a winding rule to one polygon set, regularizing it at machine-scale eps
// with no extra tolerance. This is fill-rule application - what construction
// and Offset use to resolve self-intersections - as opposed to
// CrossSection::Simplify, which decimates at a user-designated tolerance.
Polygons ApplyFillRule(const Polygons& polys, double eps, WindRule rule) {
  return ApplyFillRule(polys, {}, 1, rule, eps);
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

// ===== Vertex merge =====
// Vertex-merge and edge-collapse passes. MergeVerts buckets verts within eps
// of each other onto an existing representative vertex; RemapAndCollapse drops
// any input edge whose endpoints merged to the same vert. Both run before the
// incidence lists are built.

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
    // Sort by x and cut into slabs: each slab holds the verts within 2*eps in
    // x of its first vert, sorted by y. Any candidate pair lies in one slab or
    // two adjacent ones, so each vert scans a y-window of its own slab and the
    // next. Pairs are sorted at the end so the unite step stays deterministic.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    manifold::stable_sort(order.begin(), order.end(),
                          [&](int a, int b) { return in[a].x < in[b].x; });
    // in[order[i]], gathered so the scans read linearly.
    std::vector<vec2> pos(n);
    for (int i = 0; i < n; ++i) pos[i] = in[order[i]];
    const double thresh = 2 * eps;
    std::vector<int> start;
    for (int i = 0; i < n; ++i) {
      if (start.empty() || !(pos[i].x - pos[start.back()].x <= thresh))
        start.push_back(i);
    }
    start.push_back(n);
    const int numSlabs = static_cast<int>(start.size()) - 1;
    for (int s = 0; s < numSlabs; ++s) {
      const int lo = start[s], hi = start[s + 1];
      if (hi - lo < 2) continue;
      // Mirror-symmetric outlines make many two-vert slabs.
      if (hi - lo == 2) {
        if (pos[lo + 1].y < pos[lo].y) {
          std::swap(order[lo], order[lo + 1]);
          std::swap(pos[lo], pos[lo + 1]);
        }
        continue;
      }
      manifold::stable_sort(order.begin() + lo, order.begin() + hi,
                            [&](int a, int b) { return in[a].y < in[b].y; });
      for (int i = lo; i < hi; ++i) pos[i] = in[order[i]];
    }
    // Slabs only read shared state, so large inputs can collect candidates in
    // parallel.
    auto visitSlab = [&](int s, std::vector<std::pair<int, int>>& out) {
      const int end = start[s + 1];
      const int nextEnd = s + 1 < numSlabs ? start[s + 2] : end;
      int next = end;
      for (int i = start[s]; i < end; ++i) {
        const vec2 a = pos[i];
        auto test = [&](int j) {
          if (std::fabs(pos[j].x - a.x) > thresh) return;
          if (std::fabs(pos[j].y - a.y) > thresh) return;
          out.emplace_back(std::min(order[i], order[j]),
                           std::max(order[i], order[j]));
        };
        for (int j = i + 1; j < end && pos[j].y - a.y <= thresh; ++j) test(j);
        // a.y only grows with i, so the next slab's window start only moves
        // forward.
        while (next < nextEnd && a.y - pos[next].y > thresh) ++next;
        for (int j = next; j < nextEnd && pos[j].y - a.y <= thresh; ++j)
          test(j);
      }
    };
    constexpr int kParallelMergeMin = 1024;
#if (MANIFOLD_PAR == 1)
    if (n >= kParallelMergeMin) {
      tbb::combinable<std::vector<std::pair<int, int>>> tls;
      manifold::for_each_n(autoPolicy(numSlabs, kFineParallelGrainSize),
                           countAt(0), numSlabs,
                           [&](int s) { visitSlab(s, tls.local()); });
      tls.combine_each([&](const std::vector<std::pair<int, int>>& l) {
        pairs.insert(pairs.end(), l.begin(), l.end());
      });
    } else
#endif
    {
      for (int s = 0; s < numSlabs; ++s) visitSlab(s, pairs);
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

// ===== Incidence lists =====
// For each edge, the sorted-by-parameter list of vertices lying within eps of
// its interior. Each vertex queries a uniform grid of eps-padded edge boxes.

namespace {

// Vertex-count threshold above which the incidence queries use TBB. Smaller
// inputs stay serial to avoid setup overhead.
constexpr int kIncidenceParallelMin = 1024;

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

// Edge pairs sharing an endpoint whose non-shared endpoints are both more than
// eps from the other edge's line; no near-line sliver split is possible.
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

Box2 BoxOf2DEdge(vec2 p0, vec2 p1, double eps) {
  const vec2 pad(eps, eps);
  Box2 b(p0, p1);
  return Box2(b.min - pad, b.max + pad);
}

// Uniform grid over a set of boxes, sized to about one box per cell. Each box
// is binned into every cell it touches.
class BoxGrid {
 public:
  explicit BoxGrid(const std::vector<Box2>& boxes) : boxes_(boxes) {
    const int n = static_cast<int>(boxes.size());
    if (n == 0) return;
    for (const auto& b : boxes) bounds_ = bounds_.Union(b);
    const vec2 size = bounds_.max - bounds_.min;
    // The second term keeps a long, thin layout from getting more than O(n)
    // cells.
    const double cell = std::max(std::sqrt(size.x * size.y / n),
                                 std::max(size.x, size.y) / (2.0 * n + 1));
    nx_ = std::max(1, std::min(n, static_cast<int>(size.x / cell) + 1));
    ny_ = std::max(1, std::min(n, static_cast<int>(size.y / cell) + 1));
    scale_ = vec2(nx_ / std::max(size.x, kTiny), ny_ / std::max(size.y, kTiny));
    cells_ = BucketIndex(nx_ * ny_, [&](auto&& emit) {
      for (int i = 0; i < n; ++i) {
        const Box2& b = boxes[i];
        for (int y = CellY(b.min.y); y <= CellY(b.max.y); ++y)
          for (int x = CellX(b.min.x); x <= CellX(b.max.x); ++x)
            emit(y * nx_ + x, i);
      }
    });
  }

  // Calls f(i) once for each box i overlapping `query`. A box spanning several
  // of the query's cells is reported only from the cell holding the lower-left
  // corner of the two boxes' intersection.
  template <typename F>
  void ForEachOverlap(const Box2& query, F&& f) const {
    if (boxes_.empty() || !query.DoesOverlap(bounds_)) return;
    for (int y = CellY(query.min.y); y <= CellY(query.max.y); ++y)
      for (int x = CellX(query.min.x); x <= CellX(query.max.x); ++x)
        for (const int i : cells_.Bucket(y * nx_ + x)) {
          const Box2& b = boxes_[i];
          if (query.DoesOverlap(b) &&
              CellX(std::max(query.min.x, b.min.x)) == x &&
              CellY(std::max(query.min.y, b.min.y)) == y)
            f(i);
        }
  }

 private:
  static constexpr double kTiny = 1e-300;
  int CellX(double x) const {
    return std::clamp(static_cast<int>((x - bounds_.min.x) * scale_.x), 0,
                      nx_ - 1);
  }
  int CellY(double y) const {
    return std::clamp(static_cast<int>((y - bounds_.min.y) * scale_.y), 0,
                      ny_ - 1);
  }

  const std::vector<Box2>& boxes_;
  Box2 bounds_;
  int nx_ = 1, ny_ = 1;
  vec2 scale_;
  BucketIndex cells_;
};

}  // namespace

// Vertex v goes on edge e's list iff RecordEdgeVertHit accepts (v, e) and some
// other edge i at v makes a qualifying pair with e: their padded boxes overlap
// and SharedEndpointSafelySkippable does not exempt them. For a hit, the pair
// condition can fail only through rounding; applying it settles those
// borderline cases the same way as testing the endpoints of every overlapping
// edge pair. A hit puts v within about eps of e, so a query box around v padded
// by eps plus a rounding margin finds every candidate e.
std::vector<std::vector<int>> BuildIncidenceLists(
    const std::vector<EdgeM>& edges, const std::vector<vec2>& verts,
    double eps) {
  const int nE = static_cast<int>(edges.size());
  const int nV = static_cast<int>(verts.size());
  std::vector<Box2> edgeBoxes(nE);
  for (int e = 0; e < nE; ++e)
    edgeBoxes[e] = BoxOf2DEdge(verts[edges[e].v0], verts[edges[e].v1], eps);
  const BoxGrid grid(edgeBoxes);
  const BucketIndex incident(nV, [&](auto&& emit) {
    for (int e = 0; e < nE; ++e) {
      emit(edges[e].v0, e);
      emit(edges[e].v1, e);
    }
  });
  std::vector<EdgeGeom> edgeG;
  BuildEdgeGeometry(edges, verts, edgeG);
  const double eps2 = eps * eps;

  auto queryVert = [&](int v, std::vector<EdgeVertHit>& hits) {
    if (incident.Bucket(v).empty()) return;
    const double pad =
        eps + 1e-9 * (std::abs(verts[v].x) + std::abs(verts[v].y) + eps);
    const Box2 query(verts[v] - vec2(pad), verts[v] + vec2(pad));
    grid.ForEachOverlap(query, [&](int e) {
      const size_t before = hits.size();
      RecordEdgeVertHit(edges, verts, edgeG, eps2, v, e, hits);
      if (hits.size() == before) return;
      for (const int i : incident.Bucket(v)) {
        if (i != e && edgeBoxes[i].DoesOverlap(edgeBoxes[e]) &&
            !SharedEndpointSafelySkippable(edges[i], edges[e], verts, eps))
          return;
      }
      hits.pop_back();
    });
  };

  std::vector<EdgeVertHit> hits;
#if (MANIFOLD_PAR == 1)
  if (nV >= kIncidenceParallelMin) {
    tbb::combinable<std::vector<EdgeVertHit>> tls;
    manifold::for_each_n(autoPolicy(nV, kFineParallelGrainSize), countAt(0), nV,
                         [&](int v) { queryVert(v, tls.local()); });
    tls.combine_each([&](const std::vector<EdgeVertHit>& l) {
      hits.insert(hits.end(), l.begin(), l.end());
    });
  } else
#endif
  {
    for (int v = 0; v < nV; ++v) queryVert(v, hits);
  }
  // Sorting makes the result independent of the order hits were found in.
  std::vector<std::vector<int>> lists;
  MaterializeEdgeVertLists(nE, hits, lists);
  return lists;
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
  // Incidence pre-split: split each edge at the input vertices lying within eps
  // of its interior. The sweep's own crossing tests are exact, so this
  // eps-scale vertex-on-edge quantization is what the sweep cannot reproduce.
  std::vector<std::vector<int>> incidenceLists;
  {
    ScopedTiming timing(P.incidenceNs);
    incidenceLists = BuildIncidenceLists(edges, merge.verts, eps);
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

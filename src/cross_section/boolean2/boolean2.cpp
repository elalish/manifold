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
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

#include "driver.h"
#include "predicates.h"
#include "winding_filter.h"

namespace manifold {
namespace boolean2 {

namespace {

double Cross(vec2 a, vec2 b) { return a.x * b.y - a.y * b.x; }

double Dot(vec2 a, vec2 b) { return a.x * b.x + a.y * b.y; }

bool AllFinite(const Polygons& polys) {
  for (const auto& loop : polys) {
    for (const vec2& v : loop) {
      if (!std::isfinite(v.x) || !std::isfinite(v.y)) return false;
    }
  }
  return true;
}

struct LocalFrame {
  vec2 origin = vec2(0.0);
  bool hasVerts = false;
};

void AccumulateBounds(const Polygons& polys, Rect* box) {
  for (const auto& loop : polys) {
    for (const vec2& v : loop) {
      box->Union(v);
    }
  }
}

LocalFrame MakeLocalFrame(const Polygons& a, const Polygons& b = {}) {
  Rect box;
  AccumulateBounds(a, &box);
  AccumulateBounds(b, &box);
  LocalFrame frame;
  frame.hasVerts = box.IsFinite();
  if (frame.hasVerts) frame.origin = box.Center();
  return frame;
}

Polygons TranslatePolygons(const Polygons& polys, vec2 delta) {
  Polygons out = polys;
  for (auto& loop : out) {
    for (vec2& v : loop) v = v + delta;
  }
  return out;
}

// Classify the CCW turn from `ref` to `dir` for deterministic polar ordering.
// Group 0 is (0, pi], group 1 is (pi, 2pi), and group 2 is the zero turn.
// A zero turn is ordered last so loop tracing does not immediately reverse
// over the incoming edge when any positive turn is available.
int CcwTurnGroup(vec2 ref, vec2 dir) {
  const double cross = Cross(ref, dir);
  if (cross > 0) return 0;
  if (cross < 0) return 1;
  return Dot(ref, dir) < 0 ? 0 : 2;
}

bool CcwTurnLess(vec2 ref, vec2 a, int edgeA, vec2 b, int edgeB) {
  const int groupA = CcwTurnGroup(ref, a);
  const int groupB = CcwTurnGroup(ref, b);
  if (groupA != groupB) return groupA < groupB;

  const double cross = Cross(a, b);
  if (cross != 0) return cross > 0;

  const double lenA = Dot(a, a);
  const double lenB = Dot(b, b);
  if (lenA != lenB) return lenA < lenB;
  return edgeA < edgeB;
}

void PushLoopIfNondegenerate(const std::vector<vec2>& verts,
                             const std::vector<int>& loopVerts,
                             Polygons* polys) {
  if (loopVerts.size() >= 3) {
    SimplePolygon loop;
    loop.reserve(loopVerts.size());
    for (int v : loopVerts) loop.push_back(verts[v]);
    polys->push_back(std::move(loop));
  }
}

void PushSimpleLoops(const std::vector<vec2>& verts, std::vector<int> loopVerts,
                     Polygons* polys) {
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

void AppendInput(const Polygons& polys, int mult, std::vector<vec2>* verts,
                 std::vector<EdgeM>* edges) {
  for (const auto& loop : polys) {
    if (loop.size() < 3) continue;
    const int base = static_cast<int>(verts->size());
    const int n = static_cast<int>(loop.size());
    for (const auto& v : loop) verts->push_back(v);
    for (int i = 0; i < n; ++i) {
      edges->push_back({base + i, base + ((i + 1) % n), mult});
    }
  }
}

Polygons BinaryOpByRule(const Polygons& a, const Polygons& b, int bMult,
                        WindRule rule, double eps, double tolerance) {
  if (!AllFinite(a) || !AllFinite(b)) return {};
  const auto frame = MakeLocalFrame(a, b);
  Polygons localA = TranslatePolygons(a, -frame.origin);
  Polygons localB = TranslatePolygons(b, -frame.origin);
  if (eps <= 0.0) eps = InferEps(localA, localB);
  if (tolerance < eps) tolerance = eps;

  std::vector<vec2> verts;
  std::vector<EdgeM> edges;
  AppendInput(localA, 1, &verts, &edges);
  AppendInput(localB, bMult, &verts, &edges);
  if (verts.empty()) return {};

  auto r =
      RemoveOverlaps2D(verts, edges, eps, tolerance, /*debug=*/false, rule);
  return TranslatePolygons(OutEdgesToPolygons(r.verts, r.edges), frame.origin);
}

Polygons RegularizeByRule(const Polygons& in, WindRule rule, double eps,
                          double tolerance) {
  if (!AllFinite(in)) return {};
  const auto frame = MakeLocalFrame(in);
  Polygons local = TranslatePolygons(in, -frame.origin);
  if (eps <= 0.0) eps = InferEps(local, {});
  if (tolerance < eps) tolerance = eps;
  auto [verts, edges] = PolygonsToInput(local);
  if (verts.empty()) return {};
  auto r =
      RemoveOverlaps2D(verts, edges, eps, tolerance, /*debug=*/false, rule);
  return TranslatePolygons(OutEdgesToPolygons(r.verts, r.edges), frame.origin);
}
}  // namespace

// Flatten manifold::Polygons into the lower-level (verts, edges) input.
// Each loop becomes a sequence of edges with mult=+1.
std::pair<std::vector<vec2>, std::vector<EdgeM>> PolygonsToInput(
    const Polygons& polys) {
  std::vector<vec2> verts;
  std::vector<EdgeM> edges;
  AppendInput(polys, 1, &verts, &edges);
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
      if (destV < 0 || destV >= (int)outgoing.size() ||
          outgoing[destV].empty()) {
        cur = -1;
        break;
      }
      // Continue with the smallest positive CCW turn from the reverse incoming
      // edge, matching the halfedge winding filter.
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
      assert(false && "retained directed edges must form closed walks");
      continue;
    }
    if (loopVerts.size() >= 3) {
      PushSimpleLoops(verts, std::move(loopVerts), &polys);
    }
  }
  return polys;
}

// Single-input regularization used by `CrossSection::Simplify(eps)`.
Polygons Simplify(const Polygons& in, double eps, double tolerance) {
  return RegularizeByRule(in, WindRule::Add, eps, tolerance);
}

// Infer eps from a polygon set's coordinate half-extent via Smith's
// alpha-budget formula. Callers translate to a local frame first.
double InferEps(const Polygons& a, const Polygons& b) {
  Rect box;
  AccumulateBounds(a, &box);
  AccumulateBounds(b, &box);
  if (!box.IsFinite()) return 0.0;
  const vec2 size = box.Size();
  const double L = 0.5 * std::max(size.x, size.y);
  return EpsilonFromScale(L);
}

// Regularize one Polygons input under an explicit CrossSection fill rule.
Polygons FillByRule(const Polygons& in, WindRule rule, double eps) {
  return RegularizeByRule(in, rule, eps, eps);
}

// Binary boolean over one combined edge set; Subtract flips B's multiplicity.
Polygons Boolean2D(const Polygons& a, const Polygons& b, OpType op, double eps,
                   double tolerance) {
  const int bMult = op == OpType::Subtract ? -1 : 1;
  const WindRule rule = op == OpType::Intersect  ? WindRule::Intersect
                        : op == OpType::Subtract ? WindRule::Add
                                                 : WindRule::Add;
  return BinaryOpByRule(a, b, bMult, rule, eps, tolerance);
}

}  // namespace boolean2
}  // namespace manifold

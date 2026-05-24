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
#include <limits>
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

void AccumulateBounds(const Polygons& polys, vec2* min, vec2* max, bool* any) {
  for (const auto& loop : polys) {
    for (const vec2& v : loop) {
      if (!*any) {
        *min = v;
        *max = v;
        *any = true;
      } else {
        min->x = std::min(min->x, v.x);
        min->y = std::min(min->y, v.y);
        max->x = std::max(max->x, v.x);
        max->y = std::max(max->y, v.y);
      }
    }
  }
}

LocalFrame MakeLocalFrame(const Polygons& a, const Polygons& b = {}) {
  vec2 min(0.0), max(0.0);
  bool any = false;
  AccumulateBounds(a, &min, &max, &any);
  AccumulateBounds(b, &min, &max, &any);
  LocalFrame frame;
  frame.hasVerts = any;
  if (any) frame.origin = min * 0.5 + max * 0.5;
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
                     double nearRepeatedVertexTol, Polygons* polys) {
  for (;;) {
    const double mergeTol2 = nearRepeatedVertexTol * nearRepeatedVertexTol;
    bool split = false;
    for (size_t i = 1; i < loopVerts.size() && !split; ++i) {
      for (size_t j = 0; j < i; ++j) {
        const bool adjacent =
            i == j + 1 || (j == 0 && i + 1 == loopVerts.size());
        const vec2 delta = verts[loopVerts[i]] - verts[loopVerts[j]];
        if (loopVerts[i] != loopVerts[j] &&
            (nearRepeatedVertexTol <= 0.0 || adjacent ||
             Dot(delta, delta) > mergeTol2)) {
          continue;
        }
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
                        WindRule rule, double eps) {
  if (!AllFinite(a) || !AllFinite(b)) return {};
  const auto frame = MakeLocalFrame(a, b);
  Polygons localA = TranslatePolygons(a, -frame.origin);
  Polygons localB = TranslatePolygons(b, -frame.origin);
  if (eps <= 0.0) eps = InferEps(localA, localB);

  std::vector<vec2> verts;
  std::vector<EdgeM> edges;
  AppendInput(localA, 1, &verts, &edges);
  AppendInput(localB, bMult, &verts, &edges);
  if (verts.empty()) return {};

  auto r = RemoveOverlaps2D(verts, edges, eps, /*debug=*/false, rule);
  // Binary operations can consume earlier binary results. If two output
  // endpoints should meet at the same true point, their separation can include
  // one eps of drift per endpoint from the producer op plus one eps per
  // endpoint from this op. Split those near-repeated vertices during polygon
  // extraction; this does not merge vertices in the arrangement itself.
  constexpr double kComparedEndpoints = 2.0;
  constexpr double kPriorOpDriftPerEndpoint = 1.0;
  constexpr double kThisOpDriftPerEndpoint = 1.0;
  const double nearRepeatedVertexTol =
      kComparedEndpoints *
      (kPriorOpDriftPerEndpoint + kThisOpDriftPerEndpoint) * eps;
  return TranslatePolygons(
      OutEdgesToPolygons(r.verts, r.edges, nearRepeatedVertexTol),
      frame.origin);
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
                            const std::vector<OutEdge>& edges,
                            double nearRepeatedVertexTol) {
  const int nE = static_cast<int>(edges.size());
  // Per-vertex outgoing edges; the next-pointer loop scans each list with
  // deterministic cross/dot comparisons.
  std::vector<std::vector<int>> outgoing(verts.size());
  for (int i = 0; i < nE; ++i) outgoing[edges[i].v0].push_back(i);
  std::vector<bool> visited(nE, false);
  Polygons polys;
  for (int start = 0; start < nE; ++start) {
    if (visited[start]) continue;
    std::vector<int> loopVerts;
    int cur = start;
    while (cur >= 0 && !visited[cur]) {
      visited[cur] = true;
      loopVerts.push_back(edges[cur].v0);
      const int destV = edges[cur].v1;
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
    if (loopVerts.size() >= 3) {
      PushSimpleLoops(verts, std::move(loopVerts), nearRepeatedVertexTol,
                      &polys);
    } else {
      // Regularization drop: straight-line 1- or 2-vert loops enclose zero
      // area and cannot be represented in manifold::Polygons.
      assert(loopVerts.size() < 3 &&
             "regularized-drop loop should have zero area");
    }
  }
  return polys;
}

// Single-input regularization used by `CrossSection::Simplify(eps)`.
Polygons Simplify(const Polygons& in, double eps) {
  if (!AllFinite(in)) return {};
  const auto frame = MakeLocalFrame(in);
  Polygons local = TranslatePolygons(in, -frame.origin);
  if (eps <= 0.0) eps = InferEps(local, {});
  auto [verts, edges] = PolygonsToInput(local);
  if (verts.empty()) return {};
  auto r = RemoveOverlaps2D(verts, edges, eps);
  return TranslatePolygons(
      OutEdgesToPolygons(r.verts, r.edges, /*nearRepeatedVertexTol=*/0.0),
      frame.origin);
}

// Infer eps from a polygon set's coordinate half-extent via Smith's
// alpha-budget formula. Callers translate to a local frame first.
double InferEps(const Polygons& a, const Polygons& b) {
  double xMin = std::numeric_limits<double>::infinity();
  double yMin = xMin, xMax = -xMin, yMax = -xMin;
  auto bound = [&](const Polygons& polys) {
    for (const auto& loop : polys) {
      for (const auto& v : loop) {
        xMin = std::min(xMin, v.x);
        yMin = std::min(yMin, v.y);
        xMax = std::max(xMax, v.x);
        yMax = std::max(yMax, v.y);
      }
    }
  };
  bound(a);
  bound(b);
  if (!std::isfinite(xMin)) return 0.0;
  const double halfX = xMax * 0.5 - xMin * 0.5;
  const double halfY = yMax * 0.5 - yMin * 0.5;
  const double L = std::max(halfX, halfY);
  return EpsilonFromScale(L);
}

// Regularize one Polygons input under an explicit CrossSection fill rule.
Polygons FillByRule(const Polygons& in, WindRule rule, double eps) {
  if (!AllFinite(in)) return {};
  const auto frame = MakeLocalFrame(in);
  Polygons local = TranslatePolygons(in, -frame.origin);
  if (eps <= 0.0) eps = InferEps(local, {});
  auto [verts, edges] = PolygonsToInput(local);
  if (verts.empty()) return {};
  auto r = RemoveOverlaps2D(verts, edges, eps, /*debug=*/false, rule);
  return TranslatePolygons(
      OutEdgesToPolygons(r.verts, r.edges, /*nearRepeatedVertexTol=*/0.0),
      frame.origin);
}

// Binary boolean over one combined edge set; Subtract flips B's multiplicity.
Polygons Boolean2D(const Polygons& a, const Polygons& b, OpType op,
                   double eps) {
  const int bMult = op == OpType::Subtract ? -1 : 1;
  const WindRule rule = op == OpType::Intersect  ? WindRule::Intersect
                        : op == OpType::Subtract ? WindRule::Add
                                                 : WindRule::Add;
  return BinaryOpByRule(a, b, bMult, rule, eps);
}

// Symmetric difference (XOR): the region covered by A or B but not both.
// `manifold::OpType` only has three values (Add/Subtract/Intersect), so
// XOR is exposed as a separate core helper.
Polygons Xor(const Polygons& a, const Polygons& b, double eps) {
  return BinaryOpByRule(a, b, 1, WindRule::EvenOdd, eps);
}

}  // namespace boolean2
}  // namespace manifold

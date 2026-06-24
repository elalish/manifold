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
// Implements FilterByWinding; its contract lives on the declaration in
// boolean2.h. The helpers below are file-local.

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "boolean2.h"
#include "shared.h"

namespace manifold {

namespace {

bool IsInside(WindRule rule, int w) {
  switch (rule) {
    case WindRule::Add:
      return w > 0;
    case WindRule::Intersect:
      return w > 1;
  }
  return false;
}

// Winding of the face immediately to the LEFT of the directed sub-edge
// `start`->`end`, evaluated AT the start vertex.
//
// The query point is the start vertex P pushed an infinitesimal distance into
// the left face: Q = P + e*d + e^2*n, where e is a symbolic infinitesimal
// (e -> 0+) for tie-breaking, not the machine eps used elsewhere in boolean2;
// d = end - start, and n = (-d.y, d.x) is the left normal. The primary step is
// ALONG the sub-edge,
// the secondary a tiny push to its left: this hugs start->end so Q lands in its
// left face even when other edges incident to P fall between the bare left
// normal and the edge (the bare normal can overshoot past them into a different
// face). We then count signed crossings of a +x ray from Q over the edges the
// BVH reports (the slab [P.x, maxX] x {P.y} bounds every edge that can cross to
// the right). The two-level perturbation also resolves the ties a +x ray hits
// at a vertex - an axis can leave d.y == 0 (horizontal edge) or cross(d, w) ==
// 0 (an incident edge collinear with this one's direction).
//
// Edges incident to P (sharing it as a vertex, including this sub-edge itself)
// pass through the ray origin, so the standard "is the crossing to the right of
// P.x" test is degenerate. They are resolved instead by the Smith shadow of
// their OTHER endpoint O: post-insertion no edges cross and coincidences are
// one shared vertex, so an incident edge cannot cross the ray at P, and whether
// the perturbed ray crosses it is fixed by which side of the classified edge O
// sits on - an Interpolate+Shadows side test (O and the classified far vert
// share a y-side of P), with no explicit cross product (see the branch below).
int LeftWindingAtVertex(int start, int end, const BVH& bvh,
                        const CanonicalSubEdges& canon,
                        const std::vector<vec2>& verts, double globalMaxX) {
  const vec2 P = verts[start];
  const vec2 d = verts[end] - P;
  const vec2 n(-d.y, d.x);  // left normal of start->end

  // Effective sign of the y-perturbation Q.y - P.y under (primary d, secondary
  // n). d.y == 0 exactly iff start->end is horizontal, and then n.y == d.x !=
  // 0, so this is always well defined.
  const bool perturbUp = d.y != 0.0 ? d.y > 0.0 : n.y > 0.0;
  // dir argument for the on-ray (crossing-x == P.x) tie-break, with the same
  // (primary d, secondary n) priority: Shadows treats dir < 0 as "P.x < x".
  const double xTieDir = d.x != 0.0 ? d.x : n.x;

  int winding = 0;
  const Box2 slab(vec2(P.x, P.y), vec2(globalMaxX, P.y));
  auto accumulate = [&](int /*queryIdx*/, int leafIdx) {
    const int ei = bvh.leafToOrig[leafIdx];
    const CanonEdge& e = canon.edges[ei];
    const vec2 a = verts[e.vMin];
    const vec2 b = verts[e.vMax];
    if (a.y == b.y) return;  // horizontal edges never cross a +x ray
    const vec2& lo = a.y < b.y ? a : b;
    const vec2& hi = a.y < b.y ? b : a;

    // Half-open y-membership [lo.y, Q.y) under the perturbed ray height: an
    // endpoint exactly at P.y is decided by perturbUp.
    auto below = [&](double y) { return y == P.y ? perturbUp : y < P.y; };
    if (!(below(lo.y) && !below(hi.y))) return;

    bool rightOf;
    if (e.vMin == start || e.vMax == start) {
      // Incident edge through P: its crossing-x is P.x to leading order, so the
      // perturbation decides. O (its far vert) and E (the classified far vert)
      // share a y-side of P, so the side test is just Interpolate+Shadows:
      // interpolate whichever of O, E lies between P and the other (so the
      // query stays in Interpolate's domain) onto the other edge and
      // Shadows-compare x. The classified sub-edge itself crosses the ray iff
      // perturbed up.
      const int other = e.vMin == start ? e.vMax : e.vMin;
      if (other == end) {
        rightOf = perturbUp;
      } else {
        const vec2 O = verts[other];
        const vec2 E = verts[end];
        const bool oInterior = (O.y - P.y) * (O.y - E.y) <= 0.0;
        const vec2 farEnd = oInterior ? E : O;
        const vec3 pYX(P.y, P.x, 0.0);
        const vec3 farYX(farEnd.y, farEnd.x, 0.0);
        const double xq = Interpolate(pYX, farYX, oInterior ? O.y : E.y).x;
        rightOf =
            oInterior ? Shadows(xq, O.x, xTieDir) : !Shadows(xq, E.x, xTieDir);
      }
    } else {
      // Non-incident edge: crossing-x at height P.y via Smith's Interpolate
      // (axes swapped so the segment is interpolated at its y). The finite gap
      // crossX - P.x dominates; xTieDir only matters on an exact tie.
      const vec3 loYX(lo.y, lo.x, 0.0);
      const vec3 hiYX(hi.y, hi.x, 0.0);
      const double crossX = Interpolate(loYX, hiYX, P.y).x;
      rightOf = Shadows(P.x, crossX, xTieDir);
    }
    // CCW=+1: an upward sub-edge (vMin->vMax) crossing to the right of P raises
    // the winding by its multiplicity, a downward one lowers it.
    if (rightOf) winding += a.y < b.y ? e.mult : -e.mult;
  };
  auto recorder = MakeSimpleRecorder(accumulate);
  auto qf = [&](int) { return slab; };
  BVHCollisions(bvh, recorder, qf, /*n=*/1, /*parallel=*/false);
  return winding;
}

}  // namespace

std::vector<OutEdge> FilterByWinding(const CanonicalSubEdges& canon,
                                     const std::vector<vec2>& verts,
                                     WindRule rule) {
  // Reuse boolean2's BVH over the canonical sub-edges for the per-edge +x
  // ray-cast winding. One tree, built once, so the pass is ~O(E log E).
  const int nE = static_cast<int>(canon.edges.size());
  std::vector<Box2> boxes(nE);
  double globalMaxX = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < nE; ++i) {
    boxes[i] = BoxOf2DEdge(verts[canon.edges[i].vMin],
                           verts[canon.edges[i].vMax], 0.0);
  }
  for (const auto& v : verts) globalMaxX = std::max(globalMaxX, v.x);
  const BVH bvh = BVHBuildFromBoxes(boxes);

  std::vector<OutEdge> out;
  out.reserve(canon.edges.size());
  for (int ei = 0; ei < nE; ++ei) {
    const auto& edge = canon.edges[ei];
    if (edge.vMin == edge.vMax) continue;
    // Winding of the face just left of the directed sub-edge vMin->vMax,
    // evaluated at the start vertex vMin. Crossing that edge from right to left
    // raises the winding by its signed multiplicity, so the right face is
    // leftW - mult (one computation, not two). For a stable arrangement the
    // step across a retained boundary is +-1; coincident input can sum to a
    // larger magnitude, hence -mult rather than a literal -1.
    const int leftW = LeftWindingAtVertex(edge.vMin, edge.vMax, bvh, canon,
                                          verts, globalMaxX);
    const int rightW = leftW - edge.mult;
    const bool leftIn = IsInside(rule, leftW);
    const bool rightIn = IsInside(rule, rightW);
    if (leftIn == rightIn) continue;
    if (leftIn)
      out.push_back({edge.vMin, edge.vMax, 1});
    else
      out.push_back({edge.vMax, edge.vMin, 1});
  }
  return out;
}

}  // namespace manifold

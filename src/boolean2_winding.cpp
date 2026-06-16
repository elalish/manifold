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
#include <limits>
#include <utility>
#include <vector>

#include "boolean2.h"

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

// Winding number at `p`, which the caller keeps off every edge (offset
// perpendicular into a face). +x ray over the canonical-sub-edge BVH; the slab
// x in [p.x, globalMaxX] bounds every edge that can cross to the right of `p`.
int WindingNumberAt(vec2 p, const BVH& bvh, const CanonicalSubEdges& canon,
                    const std::vector<vec2>& verts, double globalMaxX) {
  int winding = 0;
  const Box2 slab(vec2(p.x, p.y), vec2(globalMaxX, p.y));
  auto accumulate = [&](int /*queryIdx*/, int leafIdx) {
    const int e = bvh.leafToOrig[leafIdx];
    const auto& edge = canon.edges[e];
    vec2 a = verts[edge.vMin];
    vec2 b = verts[edge.vMax];
    const bool upward = a.y < b.y;
    if (!upward) std::swap(a, b);
    if (a.y == b.y) return;  // horizontal edges contribute nothing
    if (!(a.y <= p.y && p.y < b.y)) return;
    const double t = (p.y - a.y) / (b.y - a.y);
    const double xi = a.x + t * (b.x - a.x);
    if (xi > p.x) winding += upward ? edge.mult : -edge.mult;
  };
  auto recorder = MakeSimpleRecorder(accumulate);
  auto qf = [&](int) { return slab; };
  BVHCollisions(bvh, recorder, qf, /*n=*/1, /*parallel=*/false);
  return winding;
}

double DistPointSeg(vec2 p, vec2 a, vec2 b) {
  const vec2 ab = b - a;
  const double L2 = la::dot(ab, ab);
  if (L2 == 0.0) return la::length(p - a);
  const double t = la::clamp(la::dot(p - a, ab) / L2, 0.0, 1.0);
  return la::length(p - (a + ab * t));
}

}  // namespace

std::vector<OutEdge> FilterByWinding(const CanonicalSubEdges& canon,
                                     const std::vector<vec2>& verts,
                                     WindRule rule) {
  // Accelerate both per-edge O(E) scans (nearest-other-edge clearance and the
  // +x ray-cast winding) by reusing boolean2's BVH over the canonical
  // sub-edges. One tree, built once.
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
    const vec2 p = verts[edge.vMin];
    const vec2 q = verts[edge.vMax];
    const vec2 mid = (p + q) * 0.5;
    const vec2 d = q - p;
    const double L = la::length(d);
    if (L == 0.0) continue;
    // Perpendicular sample offset, bounded by a quarter of the clearance to the
    // nearest other sub-edge so each side-point lands in the adjacent face.
    // Clearance is capped at L: winding is constant within a face, so a smaller
    // in-face offset gives the same winding, and any edge nearer than L to mid
    // has its tight AABB inside the half-extent-L query box, so it is found.
    double clr = L;
    {
      const Box2 query(mid - vec2(L, L), mid + vec2(L, L));
      auto nearest = [&](int /*queryIdx*/, int leafIdx) {
        const int f = bvh.leafToOrig[leafIdx];
        if (f == ei) return;
        clr = std::min(clr, DistPointSeg(mid, verts[canon.edges[f].vMin],
                                         verts[canon.edges[f].vMax]));
      };
      auto recorder = MakeSimpleRecorder(nearest);
      auto qf = [&](int) { return query; };
      BVHCollisions(bvh, recorder, qf, /*n=*/1, /*parallel=*/false);
    }
    const vec2 nrm = la::normalize(vec2(-d.y, d.x));  // left unit normal
    const double delta = 0.25 * clr;
    const vec2 leftP = mid + nrm * delta;
    const vec2 rightP = mid - nrm * delta;
    const int leftW = WindingNumberAt(leftP, bvh, canon, verts, globalMaxX);
    const int rightW = WindingNumberAt(rightP, bvh, canon, verts, globalMaxX);
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

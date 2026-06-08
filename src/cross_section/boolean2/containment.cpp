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

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../../utils.h"
#include "boolean2.h"
#include "manifold/optional_assert.h"
#include "predicates.h"

// Standalone polygon utilities used by `CrossSection::Decompose`
// (containment grouping). Not part of the boolean-overlap algorithm
// pipeline; operates on already-regularized Positive `Polygons`.
namespace manifold {
namespace boolean2 {

namespace {

bool PointOnSegment(vec2 p, vec2 a, vec2 b, double eps) {
  // Cheap bounding-box reject first; CCW (the colinearity test) is costlier.
  if (p.x < std::min(a.x, b.x) - eps || p.x > std::max(a.x, b.x) + eps ||
      p.y < std::min(a.y, b.y) - eps || p.y > std::max(a.y, b.y) + eps) {
    return false;
  }
  return CCW(a, b, p, eps) == 0;
}

// Standard ray-cast point-in-polygon: cast +x ray from `p`, count
// crossings of `ring`'s edges. Returns true when `p` is inside `ring`
// or on its boundary. Boundary inclusion matters for containment
// grouping: boolean output can contain a hole ring touching its outer
// ring at a vertex, and that hole still belongs to the outer component.
// `eps` is `ring`'s scale-derived tolerance, hoisted in by the caller so it
// is computed once per ring rather than once per query point.
bool PointInRing(vec2 p, const SimplePolygon& ring, double eps) {
  bool inside = false;
  const int n = static_cast<int>(ring.size());
  for (int i = 0, j = n - 1; i < n; j = i++) {
    const vec2 a = ring[i], b = ring[j];
    if (PointOnSegment(p, a, b, eps)) return true;
    if (((a.y > p.y) != (b.y > p.y)) &&
        (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)) {
      inside = !inside;
    }
  }
  return inside;
}

// Half the larger bbox extent: the length scale feeding EpsilonFromScale.
// This is a SIZE scale (extent), deliberately not Rect::Scale() (which is the
// max absolute coordinate); the two diverge for rings far from the origin.
double BoxScale(const Rect& box) {
  const vec2 size = box.Size();
  return 0.5 * std::max(size.x, size.y);
}

struct RingInfo {
  Rect box;     // bbox
  double area;  // signed; CCW > 0, CW (hole) < 0
  double eps;   // scale-derived tolerance, EpsilonFromScale(BoxScale(box))
};

RingInfo Summarize(const SimplePolygon& ring) {
  RingInfo r;
  for (const vec2& v : ring) r.box.Union(v);
  r.area = SignedArea(ring);
  r.eps = EpsilonFromScale(BoxScale(r.box));
  return r;
}

// Tolerant bbox containment prefilter: is a's box inside b's box, inflated by
// b's epsilon? Uses the same eps as the later ring test so it is never
// stricter than that test.
bool BoxInside(const RingInfo& a, const RingInfo& b) {
  const double eps = b.eps;
  return a.box.min.x >= b.box.min.x - eps && a.box.min.y >= b.box.min.y - eps &&
         a.box.max.x <= b.box.max.x + eps && a.box.max.y <= b.box.max.y + eps;
}

bool RingInside(const SimplePolygon& a, const SimplePolygon& b, double bEps) {
  return std::all_of(a.begin(), a.end(),
                     [&](const vec2& p) { return PointInRing(p, b, bEps); });
}

}  // namespace

// Decompose regularized simple loops into outer-ring components with their
// directly contained holes.
std::vector<Polygons> DecomposeByContainment(const Polygons& polys) {
  Polygons rings;
  std::vector<RingInfo> info;
  rings.reserve(polys.size());
  info.reserve(polys.size());
  for (const auto& r : polys) {
    if (r.size() < 3) continue;  // sub-3 rings bound no area
    RingInfo ri = Summarize(r);
    // Drop near-zero-area (collinear/sliver) rings before they leak as
    // degenerate hole contours. Area is length^2, so the threshold is
    // length * epsilon (scale-consistent), matching the CrossSection
    // area-drop idiom rather than comparing an area against a length eps.
    const vec2 size = ri.box.Size();
    if (std::fabs(ri.area) <= std::max(size.x, size.y) * ri.eps) continue;
    rings.push_back(r);
    info.push_back(ri);
  }
  const int n = static_cast<int>(rings.size());
  if (n == 0) return {};

  // For each ring, find its parent: the smallest-area ring (by |area|)
  // that contains it. O(n^2) bbox/ring-in-poly check; fine for the
  // hundreds of rings typical of CrossSection inputs.
  std::vector<int> parent(n, -1);
  for (int i = 0; i < n; ++i) {
    double bestParentArea = std::numeric_limits<double>::infinity();
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      if (!BoxInside(info[i], info[j])) continue;
      if (!RingInside(rings[i], rings[j], info[j].eps)) continue;
      const double aj = std::fabs(info[j].area);
      if (aj < bestParentArea) {
        bestParentArea = aj;
        parent[i] = j;
      }
    }
  }
  // Every positive ring seeds a component, regardless of its parent's sign.
  // For regularized Positive input a positive ring never nests directly in
  // another positive ring, so this is the ordinary even-odd nesting; for raw
  // input it makes positive-in-positive a deliberate double-region rather than
  // a silent area drop.
  std::vector<int> compOf(n, -1);
  std::vector<Polygons> components;
  for (int i = 0; i < n; ++i) {
    if (info[i].area <= 0) continue;
    compOf[i] = static_cast<int>(components.size());
    components.emplace_back();
    components.back().push_back(rings[i]);
  }
  // Hole rings attach to the component of their nearest positive ancestor.
  for (int i = 0; i < n; ++i) {
    if (info[i].area > 0) continue;  // skip outers
    int p = parent[i];
    // Walk up until we find a positive ring; that's the containing component.
    // Bounded by the ring count in case a malformed parent chain loops.
    for (int hops = 0; p >= 0 && info[p].area < 0 && hops <= n; ++hops) {
      p = parent[p];
    }
    if (p < 0 || compOf[p] < 0) continue;  // orphan hole; drop
    components[compOf[p]].push_back(rings[i]);
  }
  // Seeding every positive ring above leaves none without a component.
  for (int i = 0; i < n; ++i) {
    DEBUG_ASSERT(info[i].area <= 0 || compOf[i] >= 0, logicErr,
                 "positive ring left without a component");
  }
  return components;
}

}  // namespace boolean2
}  // namespace manifold

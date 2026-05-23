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
// Standalone polygon utilities intended for the later
// `CrossSection::Decompose` backend mapping. Not part of the boolean-overlap
// algorithm pipeline; operates on already-regularized `Polygons` produced by
// FillByRule.

#include "containment.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../../utils.h"
#include "predicates.h"

namespace manifold {
namespace boolean2 {

namespace polyutils_detail {

bool PointOnSegment(vec2 p, vec2 a, vec2 b, double eps) {
  if (CCW(a, b, p, eps) != 0) return false;
  return p.x >= std::min(a.x, b.x) - eps && p.x <= std::max(a.x, b.x) + eps &&
         p.y >= std::min(a.y, b.y) - eps && p.y <= std::max(a.y, b.y) + eps;
}

double RingScale(const SimplePolygon& ring) {
  if (ring.empty()) return 0.0;
  vec2 bmin(std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity());
  vec2 bmax(-bmin.x, -bmin.y);
  for (const vec2& v : ring) {
    bmin.x = std::min(bmin.x, v.x);
    bmin.y = std::min(bmin.y, v.y);
    bmax.x = std::max(bmax.x, v.x);
    bmax.y = std::max(bmax.y, v.y);
  }
  return std::max(bmax.x - bmin.x, bmax.y - bmin.y) * 0.5;
}

// Standard ray-cast point-in-polygon: cast +x ray from `p`, count
// crossings of `ring`'s edges. Returns true when `p` is inside `ring`
// or on its boundary. Boundary inclusion matters for containment
// grouping: boolean output can contain a hole ring touching its outer
// ring at a vertex, and that hole still belongs to the outer component.
bool PointInRing(vec2 p, const SimplePolygon& ring) {
  const double eps = EpsilonFromScale(RingScale(ring));
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

struct RingInfo {
  vec2 bmin, bmax;  // bbox
  double area;      // signed; CCW > 0, CW (hole) < 0
};

RingInfo Summarize(const SimplePolygon& ring) {
  RingInfo r;
  r.bmin = vec2(std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity());
  r.bmax = vec2(-r.bmin.x, -r.bmin.y);
  for (const auto& v : ring) {
    r.bmin.x = std::min(r.bmin.x, v.x);
    r.bmin.y = std::min(r.bmin.y, v.y);
    r.bmax.x = std::max(r.bmax.x, v.x);
    r.bmax.y = std::max(r.bmax.y, v.y);
  }
  r.area = SignedArea(ring);
  // Empty rings are dropped by FillByRule before reaching this code path
  // in production; the guard here is for direct callers of
  // DecomposeByContainment that may pass raw Polygons.
  return r;
}

bool BoxInside(const RingInfo& a, const RingInfo& b) {
  return a.bmin.x >= b.bmin.x && a.bmin.y >= b.bmin.y && a.bmax.x <= b.bmax.x &&
         a.bmax.y <= b.bmax.y;
}

bool RingInside(const SimplePolygon& a, const SimplePolygon& b) {
  return std::all_of(a.begin(), a.end(),
                     [&](const vec2& p) { return PointInRing(p, b); });
}

}  // namespace polyutils_detail

// Decompose a regularized `Polygons` set into containment components. Input is
// assumed to be simple-loop output from FillByRule or similar: CCW outer rings
// + CW hole rings, non-self-intersecting, non-overlapping except at containment
// boundaries. Each outermost CCW ring plus its directly-contained CW holes form
// one component; CCW rings nested inside holes start new components.
//
// Returns: one vector<SimplePolygon> per component, with the outer
// ring first and any holes following.
std::vector<Polygons> DecomposeByContainment(const Polygons& polys) {
  const int n = static_cast<int>(polys.size());
  if (n == 0) return {};
  std::vector<polyutils_detail::RingInfo> info;
  info.reserve(n);
  for (const auto& r : polys) info.push_back(polyutils_detail::Summarize(r));

  // For each ring, find its parent: the smallest-area ring (by |area|)
  // that contains it. O(n^2) bbox/ring-in-poly check; fine for the
  // hundreds of rings typical of CrossSection inputs.
  std::vector<int> parent(n, -1);
  for (int i = 0; i < n; ++i) {
    double bestParentArea = std::numeric_limits<double>::infinity();
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      if (!polyutils_detail::BoxInside(info[i], info[j])) continue;
      if (!polyutils_detail::RingInside(polys[i], polys[j])) continue;
      const double aj = std::fabs(info[j].area);
      if (aj < bestParentArea) {
        bestParentArea = aj;
        parent[i] = j;
      }
    }
  }
  // Each outermost positive ring (no parent, or parent is negative)
  // seeds a component. Hole rings join the component of their (positive)
  // parent. Positive rings nested inside negatives become new
  // components.
  std::vector<int> compOf(n, -1);
  std::vector<Polygons> components;
  // Pass 1: positive rings whose parent is negative-or-absent become
  // component seeds.
  for (int i = 0; i < n; ++i) {
    const bool positive = info[i].area > 0;
    if (!positive) continue;
    const int p = parent[i];
    if (p < 0 || info[p].area < 0) {
      compOf[i] = static_cast<int>(components.size());
      components.emplace_back();
      components.back().push_back(polys[i]);
    }
  }
  // Pass 2: holes attach to their positive parent's component.
  for (int i = 0; i < n; ++i) {
    if (info[i].area > 0) continue;  // skip outers
    int p = parent[i];
    // Walk up until we find a positive ring; that's the containing component.
    // Simple-loop output from FillByRule shouldn't produce hole-inside-hole,
    // but a malformed parent chain would loop here forever; bound by the ring
    // count.
    for (int hops = 0; p >= 0 && info[p].area < 0 && hops <= n; ++hops) {
      p = parent[p];
    }
    if (p < 0 || compOf[p] < 0) continue;  // orphan hole; drop
    components[compOf[p]].push_back(polys[i]);
  }
  return components;
}

}  // namespace boolean2
}  // namespace manifold

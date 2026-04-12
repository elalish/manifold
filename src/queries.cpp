// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cmath>
#include <limits>

#include "impl.h"

namespace manifold {

namespace {

// Closest point on triangle (v0, v1, v2) to point p.
// Uses the Voronoi region method from Real-Time Collision Detection.
inline vec3 ClosestPointOnTriangle(vec3 p, vec3 v0, vec3 v1, vec3 v2) {
  const vec3 ab = v1 - v0;
  const vec3 ac = v2 - v0;
  const vec3 ap = p - v0;

  const double d1 = la::dot(ab, ap);
  const double d2 = la::dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) return v0;  // Vertex region A

  const vec3 bp = p - v1;
  const double d3 = la::dot(ab, bp);
  const double d4 = la::dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) return v1;  // Vertex region B

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double v = d1 / (d1 - d3);
    return v0 + v * ab;  // Edge region AB
  }

  const vec3 cp = p - v2;
  const double d5 = la::dot(ab, cp);
  const double d6 = la::dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) return v2;  // Vertex region C

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double w = d2 / (d2 - d6);
    return v0 + w * ac;  // Edge region AC
  }

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return v1 + w * (v2 - v1);  // Edge region BC
  }

  // Inside triangle
  const double denom = 1.0 / (va + vb + vc);
  const double v = vb * denom;
  const double w = vc * denom;
  return v0 + ab * v + ac * w;
}

// Möller–Trumbore ray-triangle intersection.
// Ray: origin + t * dir, t in [0, tMax].
// Returns t >= 0 on hit, or -1 on miss.
inline double RayTriangleIntersect(vec3 origin, vec3 dir, double tMax, vec3 v0,
                                   vec3 v1, vec3 v2) {
  constexpr double kEps = 1e-15;
  const vec3 edge1 = v1 - v0;
  const vec3 edge2 = v2 - v0;
  const vec3 h = la::cross(dir, edge2);
  const double a = la::dot(edge1, h);

  if (std::fabs(a) < kEps) return -1;  // Parallel

  const double f = 1.0 / a;
  const vec3 s = origin - v0;
  const double u = f * la::dot(s, h);
  if (u < 0.0 || u > 1.0) return -1;

  const vec3 q = la::cross(s, edge1);
  const double v = f * la::dot(dir, q);
  if (v < 0.0 || u + v > 1.0) return -1;

  const double t = f * la::dot(edge2, q);
  if (t >= 0.0 && t <= tMax) return t;
  return -1;
}

// Intersect a ray (origin, direction) with an axis-aligned bounding box.
// Returns true if the ray hits the box, with tNear and tFar set to the
// entry and exit t values. tNear may be negative (origin inside box).
inline bool RayBoxIntersect(vec3 origin, vec3 dir, const Box& box,
                            double& tNear, double& tFar) {
  tNear = 0.0;
  tFar = std::numeric_limits<double>::infinity();

  for (int i = 0; i < 3; ++i) {
    if (std::fabs(dir[i]) < 1e-30) {
      // Ray parallel to slab
      if (origin[i] < box.min[i] || origin[i] > box.max[i]) return false;
    } else {
      double invD = 1.0 / dir[i];
      double t1 = (box.min[i] - origin[i]) * invD;
      double t2 = (box.max[i] - origin[i]) * invD;
      if (t1 > t2) std::swap(t1, t2);
      tNear = std::max(tNear, t1);
      tFar = std::min(tFar, t2);
      if (tNear > tFar) return false;
    }
  }
  return true;
}

}  // namespace

RayHit Manifold::Impl::RayCast(vec3 origin, vec3 endpoint) const {
  ZoneScoped;
  if (IsEmpty()) return {};

  const vec3 dir = endpoint - origin;
  const double rayLen = la::length(dir);
  if (rayLen < 1e-30) return {};

  // Build AABB around the ray segment for BVH query
  const Box rayBox(la::min(origin, endpoint), la::max(origin, endpoint));

  // Track the nearest hit
  double bestT = std::numeric_limits<double>::infinity();
  int bestTri = -1;

  auto recorderf = [&](int /*queryIdx*/, int tri) {
    const vec3 v0 = vertPos_[halfedge_[3 * tri].startVert];
    const vec3 v1 = vertPos_[halfedge_[3 * tri + 1].startVert];
    const vec3 v2 = vertPos_[halfedge_[3 * tri + 2].startVert];

    const double t = RayTriangleIntersect(origin, dir, 1.0, v0, v1, v2);
    if (t >= 0.0 && t < bestT) {
      bestT = t;
      bestTri = tri;
    }
  };
  auto recorder = MakeSimpleRecorder(recorderf);
  auto f = [&rayBox](int) { return rayBox; };
  collider_.Collisions<false>(recorder, f, 1, false);

  if (bestTri < 0) return {};

  RayHit hit;
  hit.distance = bestT;
  hit.position = origin + bestT * dir;
  hit.normal = faceNormal_[bestTri];
  hit.faceID = bestTri;
  return hit;
}

int Manifold::Impl::WindingNumber(vec3 point) const {
  ZoneScoped;
  if (IsEmpty()) return 0;

  // Cast a +Z ray from the point. The collider's DoesOverlap(vec3)
  // checks XY containment, which is exactly what we need for a Z-axis ray.
  int winding = 0;

  auto recorderf = [&](int /*queryIdx*/, int tri) {
    const vec3 v0 = vertPos_[halfedge_[3 * tri].startVert];
    const vec3 v1 = vertPos_[halfedge_[3 * tri + 1].startVert];
    const vec3 v2 = vertPos_[halfedge_[3 * tri + 2].startVert];

    // 2D point-in-triangle test via edge crossing (XY projection).
    // For each edge, test if point.x crosses the edge's X span and
    // if the edge is above the point in Y at that X.
    int crossings = 0;
    const vec3 verts[3] = {v0, v1, v2};
    for (int i = 0; i < 3; ++i) {
      const vec3& a = verts[i];
      const vec3& b = verts[(i + 1) % 3];

      // Check if edge spans point.x
      if ((a.x <= point.x) == (b.x <= point.x)) continue;

      // Interpolate Y at point.x
      const double t = (point.x - a.x) / (b.x - a.x);
      const double y_at_x = a.y + t * (b.y - a.y);

      if (point.y < y_at_x) {
        crossings += (b.x > a.x) ? 1 : -1;
      }
    }

    if (crossings == 0) return;

    // Interpolate Z at (point.x, point.y) using barycentric coordinates
    // on the XY projection of the triangle.
    const vec3 e0 = v1 - v0;
    const vec3 e1 = v2 - v0;
    const vec3 dp = point - v0;

    const double d00 = e0.x * e0.x + e0.y * e0.y;
    const double d01 = e0.x * e1.x + e0.y * e1.y;
    const double d11 = e1.x * e1.x + e1.y * e1.y;
    const double d20 = dp.x * e0.x + dp.y * e0.y;
    const double d21 = dp.x * e1.x + dp.y * e1.y;
    const double denom = d00 * d11 - d01 * d01;
    if (std::fabs(denom) < 1e-30) return;

    const double baryV = (d11 * d20 - d01 * d21) / denom;
    const double baryW = (d00 * d21 - d01 * d20) / denom;
    const double z_hit = v0.z + baryV * e0.z + baryW * e1.z;

    // Only count intersections above the query point (+Z direction)
    if (z_hit > point.z) {
      // Sign based on face normal Z component
      winding += (faceNormal_[tri].z > 0) ? crossings : -crossings;
    }
  };
  auto recorder = MakeSimpleRecorder(recorderf);
  // Use vec3 query: DoesOverlap(vec3) checks XY containment
  auto f = [&point](int) { return point; };
  collider_.Collisions<false>(recorder, f, 1, false);

  return winding;
}

NearestPointResult Manifold::Impl::NearestPoint(vec3 point) const {
  ZoneScoped;
  if (IsEmpty()) return {};

  NearestPointResult best;

  // Use BVH with expanding search radius for acceleration.
  // Start with a heuristic radius based on mesh density.
  const vec3 bboxSize = bBox_.Size();
  const double bboxDiag = la::length(bboxSize);
  double searchRadius = bboxDiag / std::max(1.0, std::pow(NumTri(), 1.0 / 3.0));

  // Distance from point to the nearest point on the AABB surface
  const vec3 clamped = la::clamp(point, bBox_.min, bBox_.max);
  const double distToBox = la::length(point - clamped);
  searchRadius = std::max(searchRadius, distToBox + searchRadius);

  for (int attempt = 0; attempt < 32; ++attempt) {
    const Box queryBox(point - vec3(searchRadius), point + vec3(searchRadius));

    auto recorderf = [&](int /*queryIdx*/, int tri) {
      const vec3 v0 = vertPos_[halfedge_[3 * tri].startVert];
      const vec3 v1 = vertPos_[halfedge_[3 * tri + 1].startVert];
      const vec3 v2 = vertPos_[halfedge_[3 * tri + 2].startVert];

      const vec3 closest = ClosestPointOnTriangle(point, v0, v1, v2);
      const double dist = la::length(closest - point);

      if (dist < best.distance) {
        best.distance = dist;
        best.position = closest;
        best.normal = faceNormal_[tri];
        best.faceID = tri;
      }
    };
    auto recorder = MakeSimpleRecorder(recorderf);
    auto f = [&queryBox](int) { return queryBox; };
    collider_.Collisions<false>(recorder, f, 1, false);

    if (best.faceID >= 0) {
      // We found a hit. Do a refinement pass: search with the exact
      // best distance to ensure no closer triangle was missed.
      if (best.distance < searchRadius) {
        const Box refineBox(point - vec3(best.distance),
                            point + vec3(best.distance));
        auto recorderf2 = [&](int /*queryIdx*/, int tri) {
          const vec3 v0 = vertPos_[halfedge_[3 * tri].startVert];
          const vec3 v1 = vertPos_[halfedge_[3 * tri + 1].startVert];
          const vec3 v2 = vertPos_[halfedge_[3 * tri + 2].startVert];

          const vec3 closest = ClosestPointOnTriangle(point, v0, v1, v2);
          const double dist = la::length(closest - point);

          if (dist < best.distance) {
            best.distance = dist;
            best.position = closest;
            best.normal = faceNormal_[tri];
            best.faceID = tri;
          }
        };
        auto recorder2 = MakeSimpleRecorder(recorderf2);
        auto f2 = [&refineBox](int) { return refineBox; };
        collider_.Collisions<false>(recorder2, f2, 1, false);
      }
      break;
    }

    searchRadius *= 4;
    if (searchRadius > bboxDiag * 4) {
      // Fallback: cover entire space around the mesh
      searchRadius = bboxDiag * 4 + distToBox;
    }
  }

  return best;
}

}  // namespace manifold

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

}  // namespace

NearestPointResult Manifold::Impl::NearestPoint(vec3 point) const {
  ZoneScoped;
  if (IsEmpty()) return {};

  NearestPointResult best;

  // Use BVH with expanding search radius for acceleration.
  const vec3 bboxSize = bBox_.Size();
  const double bboxDiag = la::length(bboxSize);
  double searchRadius = bboxDiag / std::max(1.0, std::pow(NumTri(), 1.0 / 3.0));

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
      searchRadius = bboxDiag * 4 + distToBox;
    }
  }

  return best;
}

}  // namespace manifold

// Copyright 2021 The Manifold Authors.
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

#include <iostream>
#include <optional>
#include <vector>

#include "../collider.h"
#include "../vec.h"
#include "clipper2/clipper.core.h"
#include "clipper2/clipper.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"

const double EPSILON = 1e-9;

namespace {
// Utility

using namespace manifold;

namespace C2 = Clipper2Lib;
vec2 v2_of_pd(const C2::PointD p) { return {p.x, p.y}; }

vec3 toVec3(vec2 in) { return vec3(in.x, in.y, 0); }

// Check if line segment intersect with another line segment
bool intersectSegmentSegment(const vec2& p1, const vec2& p2, const vec2& p3,
                             const vec2& p4, vec2& intersectionPoint) {
  double det = la::cross(p2 - p1, p4 - p3);

  if (std::abs(det) < EPSILON) {
    // Parallel
    return false;
  }

  double num_t = la::cross(p3 - p1, p4 - p3);
  double num_u = la::cross(p3 - p1, p2 - p1);

  double t = num_t / det;
  double u = num_u / det;

  // Check if the intersection point inside line segment
  if ((t >= 0.0 - EPSILON && t <= 1.0 + EPSILON) &&
      (u >= 0.0 - EPSILON && u <= 1.0 + EPSILON)) {
    // Inside
    double intersect_x = p1.x + t * (p2.x - p1.x);
    double intersect_y = p1.y + t * (p2.y - p1.y);

    intersectionPoint = {intersect_x, intersect_y};
    return true;
  } else {
    // Outside -> No intersection

    intersectionPoint = {};
    return false;
  }
};

// Check if line segment intersect with a circle
bool intersectCircleSegment(const vec2& p1, const vec2& p2, const vec2& center,
                            double radius) {
  vec2 d = p2 - p1;

  if (la::length(d) < EPSILON)
    return la::dot(p1 - center, p1 - center) <= radius * radius;

  // Project vec p1 -> circle to line segment
  double t = la::dot(center - p1, d) / la::dot(d, d);

  vec2 closestPoint;

  if (t < 0) {
    closestPoint = p1;
  } else if (t > 1) {
    closestPoint = p2;
  } else {
    closestPoint = p1 + t * d;
  }

  // Calculate the distance from the closest point to the circle's center
  double distanceSquared =
      la::dot(closestPoint - center, closestPoint - center);

  return distanceSquared <= radius * radius;
};

// Projection point to line and check if it's on the line segment
bool isProjectionOnSegment(const vec2& c, const vec2& p1, const vec2& p2,
                           double& t) {
  t = la::dot(c - p1, p2 - p1) / la::length2(p2 - p1);

  return t >= 0 && t <= 1;
};

// Calculate Circle center that tangent to one segment, and cross a point
bool calculatePointSegmentCircleCenter(const vec2& p1, const vec2& p2,
                                       const vec2& endpoint, double radius,
                                       double& e1T, vec2& circleCenter) {
  vec2 e = p2 - p1, dir = la::normalize(e), normal = {-e.y, e.x};

  if (la::length(e) < EPSILON) {
    // FIXME: Degenerate
    std::cout << "Degenerate" << std::endl;

    return false;
  }

  vec2 projectedEndpoint = p1 + dir * la::dot(endpoint - p1, dir);
  double dist = la::length(projectedEndpoint - endpoint);
  if (dist > radius * 2.0) {
    // Bad math, something wrong before
    throw std::exception();
  }

  dist = dist > radius ? dist - radius : radius - dist;

  double len = la::sqrt(radius * radius - dist * dist);
  vec2 tangentPoint1 = projectedEndpoint + dir * len,
       tangentPoint2 = projectedEndpoint - dir * len;

  //  FIXME: degenerate case
  circleCenter = (la::length2(tangentPoint1 - circleCenter) <
                  la::length2(tangentPoint2 - circleCenter))
                     ? tangentPoint1
                     : tangentPoint2;

  return true;
}

// Calculate Circle center that tangent to both line segment
// If not, degenerate to only tangent to one segment, and cross another's
// endpoint
bool calculateSegmentSegmentCircleCenter(const vec2& p1, const vec2& p2,
                                         const vec2& p3, const vec2& p4,
                                         double radius, double& e1T,
                                         double& e2T, vec2& circleCenter) {
  vec2 e1 = p2 - p1;
  vec2 normal1 = {-e1.y, e1.x};
  double c1 = -la::dot(normal1, p1);
  vec2 e2 = p4 - p3;
  vec2 normal2 = {-e2.y, e2.x};
  double c2 = -la::dot(normal2, p3);
  if (la::length(e1) < EPSILON || la::length(e2) < EPSILON) {
    // FIXME: Degenerate
    std::cout << "Degenerate" << std::endl;

    return false;
  }
  mat2 A = {{normal1.x, normal2.x}, {normal1.y, normal2.y}};
  vec2 b = {radius * la::length(normal1) - c1,
            radius * la::length(normal2) - c2};
  if (std::abs(la::determinant(A)) < EPSILON) {
    // FIXME: Parallel line
    std::cout << "Parallel" << std::endl;
    return false;
  }
  circleCenter = la::mul(la::inverse(A), b);

  bool onE1 = isProjectionOnSegment(circleCenter, p1, p2, e1T),
       onE2 = isProjectionOnSegment(circleCenter, p3, p4, e2T);

  // Check Circle center projection for tangent point status
  if (onE1 && onE2) {
    // Tangent point on both line segment
  } else if ((!onE1) && (!onE2)) {
    // Both not on line segment, invalid result
    return false;
  } else if (onE1) {
    // Won't tangent to both line, degenerated to Point Segment Intersection,
    // recalculate new circle position

    // Only on e1, tangent point might be e2 endpoint
    vec2 endpoint =
        (la::length2(circleCenter - p3) < la::length2(circleCenter - p4)) ? p3
                                                                          : p4;

    if (!calculatePointSegmentCircleCenter(p1, p2, endpoint, radius, e1T,
                                           circleCenter))
      return false;

    e2T = (e2T < 0 ? 0 : 1);

    // Calc new tangent point's parameter value, and check is valid
    isProjectionOnSegment(circleCenter, p1, p2, e1T);
    if (e1T < EPSILON || e1T > (1 + EPSILON)) return false;

    std::cout << "Center (line-vertex on e2): ";

  } else if (onE2) {
    // Won't tangent to both line, degenerated to Point Segment Intersection,
    // recalculate new circle position

    // tangent point might be e1 endpoint

    vec2 endpoint =
        (la::length2(circleCenter - p1) < la::length2(circleCenter - p2)) ? p1
                                                                          : p2;

    if (!calculatePointSegmentCircleCenter(p3, p4, endpoint, radius, e1T,
                                           circleCenter))
      return false;

    e1T = (e1T < 0 ? 0 : 1);

    isProjectionOnSegment(circleCenter, p3, p4, e2T);

    if (e2T < EPSILON || e2T > (1 + EPSILON)) return false;
    std::cout << "Center (line-vertex on e1): ";
  }

  return true;
}

struct ArcConnectionInfo {
  vec2 center;

  double t1, t2;  // Parameter value of arc tangent points of edges
  size_t e1, e2;  // Edge Idx of tangent points lie on
  double startRad, endRad;
};

std::vector<vec2> discreteArcToPoint(ArcConnectionInfo arc, double radius,
                                     int circularSegments) {
  std::vector<vec2> pts;

  double totalRad = arc.endRad - arc.startRad;

  if (totalRad < 0) {
    totalRad += 2.0 * M_PI;
  }

  double dPhi = 2 * M_PI / circularSegments;
  int seg = int(totalRad / dPhi) + 1;
  for (int i = 0; i < seg; ++i) {
    double current = arc.startRad + dPhi * i;

    vec2 pnt = {arc.center.x + radius * cos(current),
                arc.center.y + radius * sin(current)};

    pts.push_back(pnt);
  }

  return pts;
}

}  // namespace

// Sub process
namespace {
using namespace manifold;

// For build and query Collider result
struct edgeOld2New {
  manifold::Box box;
  uint32_t morton;
  size_t p1Ref;
  size_t p2Ref;
};

manifold::Collider BuildCollider(const manifold::SimplePolygon& loop,
                                 std::vector<edgeOld2New>& edgeOld2NewVec) {
  Vec<manifold::Box> boxVec;
  Vec<uint32_t> mortonVec;

  for (size_t i = 0; i != loop.size(); i++) {
    const vec2 p1 = loop[i], p2 = loop[(i + 1) % loop.size()];

    vec3 center = toVec3(p1) + toVec3(p2);
    center /= 2;

    manifold::Box bbox(toVec3(p1), toVec3(p2));

    edgeOld2NewVec.push_back({bbox,
                              manifold::Collider::MortonCode(center, bbox), i,
                              (i + 1) % loop.size()});
  }

  std::stable_sort(edgeOld2NewVec.begin(), edgeOld2NewVec.end(),
                   [](const edgeOld2New& lhs, const edgeOld2New& rhs) -> bool {
                     return rhs.morton > lhs.morton;
                   });

  for (auto it = edgeOld2NewVec.begin(); it != edgeOld2NewVec.end(); it++) {
    boxVec.push_back(it->box);
    mortonVec.push_back(it->morton);
  }

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose) {
    boxVec.Dump();
    mortonVec.Dump();
  }
#endif

  return Collider(boxVec, mortonVec);
}

struct ColliderInfo {
  Collider outerCollider;
  std::vector<edgeOld2New> outerEdgeOld2NewVec;

  // Multi inner loops
  std::vector<Collider> innerCollider;
  std::vector<std::vector<edgeOld2New>> innerVec;
};

std::vector<std::vector<ArcConnectionInfo>> CalculateFilletArc(
    const Polygons& input, Polygons& output, const ColliderInfo& collider,
    double radius) {
  auto& outerLoop = input[0];

  std::vector<bool> markEE(outerLoop.size() * outerLoop.size(), false);
  std::vector<bool> markEV(outerLoop.size() * outerLoop.size(), false);
  std::vector<bool> markVV(outerLoop.size() * outerLoop.size(), false);

  std::vector<std::vector<ArcConnectionInfo>> arcConnection(
      outerLoop.size(), std::vector<ArcConnectionInfo>());

  std::cout << "Collider BBox Testing" << std::endl;

  // create BBox for every line to find Collision
  for (size_t e1i = 0; e1i != outerLoop.size(); e1i++) {
    // Outer loop is CCW, p1 p2 -> current edge start end
    const size_t p1i = e1i, p2i = (e1i + 1) % outerLoop.size();
    const vec2 p1 = outerLoop[p1i], p2 = outerLoop[p2i];
    vec2 e1 = p2 - p1;
    const bool p2IsConvex =
        la::cross(e1, outerLoop[(p2i + 1) % outerLoop.size()] - p2) >= EPSILON;

    vec2 normal = la::normalize(vec2(-e1.y, e1.x));

    /* Only check e1 and p2 to avoid duplicate process
    normalOffsetP1 --- normalOffsetP2  --- circleOffsetP2
           |                 |      \--        |
           |                 |          \--    |
           |                 | Circle Arc-> \--|
           |                 | <- 2 * radius ->\
          p1 --------------- p2 ----------------|
                             |                 /
                             | Circle Arc-> /--  |
                             |           /--     |
                             |        /--        |
                             |-----/--    circleOffsetP2N
    */

    vec2 normalOffsetP1 = p1 + normal * 2.0 * radius,
         normalOffsetP2 = p2 + normal * 2.0 * radius;

    manifold::Box box(toVec3(p1), toVec3(p2));
    box.Union(toVec3(normalOffsetP1));
    box.Union(toVec3(normalOffsetP2));

    if (!p2IsConvex) {
      // Handle concave endpoint
      vec2 circleOffsetP2 = p2 + e1 * 2.0 * radius + normal * 2.0 * radius,
           circleOffsetP2N = p2 + e1 * 2.0 * radius - normal * 2.0 * radius;

      box.Union(toVec3(circleOffsetP2));
      box.Union(toVec3(circleOffsetP2N));
    }

    auto r = collider.outerCollider.Collisions(
        manifold::Vec<manifold::Box>({box}).cview());
    // r.Dump();
    r.Sort();

    std::cout << "Now " << p1i << "->" << (e1i + 1) % outerLoop.size()
              << std::endl;

    // In Out Classify
    for (size_t j = 0; j != r.size(); j++) {
      auto ele = collider.outerEdgeOld2NewVec[r.Get(j, true)];

      size_t e2i = ele.p1Ref;

      // Skip self and last one
      if ((e1i == e2i) ||
          e2i == (e1i + outerLoop.size() - 1) % outerLoop.size())
        continue;

      // CCW, p3 p4 -> bbox hit edge start end
      size_t p3i = e2i, p4i = ele.p2Ref;
      vec2 p3 = outerLoop[p3i], p4 = outerLoop[p4i];
      vec2 t;

      bool segmentIntersected = intersectSegmentSegment(
               normalOffsetP1, normalOffsetP2, p3, p4, t),
           circleIntersected =
               p2IsConvex ? intersectCircleSegment(p3, p4, p2, radius) : false;
      // FIXME: intersectCircleSegment start rad and end rad

      std::cout << "Testing " << p3i << "->" << p4i << "\t";

      std::cout << (segmentIntersected ? "Segment Intersected" : "Segment - ")
                << " "
                << (circleIntersected ? "Circle Intersected" : "Circle -");

      double e1T = 0, e2T = 0;
      vec2 circleCenter(0, 0);

      if (!segmentIntersected && !circleIntersected) {
        std::cout << std::endl;
        continue;
      } else if (circleIntersected) {
        // Concave p2 endpoint intersected, degenerated natively

        if (!calculatePointSegmentCircleCenter(p3, p4, p2, radius, e1T,
                                               circleCenter)) {
          continue;
        }
      } else if (segmentIntersected) {
        if (!calculateSegmentSegmentCircleCenter(p1, p2, p3, p4, radius, e1T,
                                                 e2T, circleCenter)) {
          continue;
        } else {
          // Check if is endpoint, and add duplicate mark
          if (e1T == 0) {
            // p1
            if (markEV[e2i * outerLoop.size() + p1i]) continue;
            markEV[e2i * outerLoop.size() + p1i] = 1;
          } else if (e1T == 1) {
            // p2
            if (markEV[e2i * outerLoop.size() + p2i]) continue;
            markEV[e2i * outerLoop.size() + p2i] = 1;
          }

          if (e2T == 0) {
            if (markEV[e1i * outerLoop.size() + p3i]) continue;
            markEV[e1i * outerLoop.size() + p3i] = 1;
          } else if (e2T == 1) {
            if (markEV[e1i * outerLoop.size() + p4i]) continue;
            markEV[e1i * outerLoop.size() + p4i] = 1;
          }
        }
      }

      {
        vec2 e1 = p2 - p1, e2 = p4 - p3;

        vec2 tangent1 = p1 + e1T * e1;
        vec2 tangent2 = p3 + e2T * e2;

        vec2 v_start = tangent1 - circleCenter;
        vec2 v_end = tangent2 - circleCenter;

        double start_rad = atan2(v_start.y, v_start.x);
        double end_rad = atan2(v_end.y, v_end.x);

        // Normalize to [0, 2π]
        if (start_rad < 0) start_rad += 2 * M_PI;
        if (end_rad < 0) end_rad += 2 * M_PI;

        // Sort result by CW
        double arcAngle = end_rad - start_rad;
        if (arcAngle < 0) arcAngle += 2 * M_PI;

        if (arcAngle <= M_PI) {
          arcConnection[e1i].emplace_back(ArcConnectionInfo{
              circleCenter, e1T, e2T, e1i, e2i, start_rad, end_rad});
        } else {
          arcConnection[e2i].emplace_back(ArcConnectionInfo{
              circleCenter, e2T, e1T, e2i, e1i, end_rad, start_rad});
        }
      }

#ifdef MANIFOLD_DEBUG
      if (ManifoldParams().verbose) {
        std::cout << "Circle center " << circleCenter << " " << e1i << " "
                  << ele.p1Ref << " Vertex index " << output[0].size() << "~"
                  << output[0].size() + 20 << std::endl;
      }
#endif

      // NOTE: inter result shown in upper figure
      // const uint32_t seg = 20;
      // for (size_t k = 0; k != seg; k++) {
      //   newLoop.push_back(circleCenter +
      //                     vec2{radius * cos(M_PI * 2 / seg * k),
      //                          radius * sin(M_PI * 2 / seg * k)});
      // }
    }
  }

  // Construct Result

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose) {
    for (size_t i = 0; i != arcConnection.size(); i++) {
      std::cout << i << " " << arcConnection[i].size();
      for (size_t j = 0; j != arcConnection[i].size(); j++) {
        std::cout << "\t" << arcConnection[i][j].e1 << " "
                  << arcConnection[i][j].e2 << " " << arcConnection[i][j].t1
                  << " " << arcConnection[i][j].t2 << " "
                  << arcConnection[i][j].startRad << " "
                  << arcConnection[i][j].endRad << std::endl;
      }

      std::cout << std::endl;
    }
  }
#endif

  return arcConnection;
}

manifold::Polygons Tracing(
    const manifold::SimplePolygon& loop,
    std::vector<std::vector<ArcConnectionInfo>> arcConnection,
    int circularSegments, double radius) {
  const double EPSILON = 1e-9;

  manifold::Polygons newPoly;

  while (true) {
    SimplePolygon rLoop{};

    std::vector<size_t> tracingEList;
    std::vector<size_t> mapVV;

    // Tracing to construct result
    size_t currentEdgeIndex = 0, endEdgeIndex = 0;

    double currentEdgeT = 0;

    // Find first fillet arc to start
    auto it = arcConnection.begin();
    for (; it != arcConnection.end(); it++) {
      if (!it->empty()) {
        ArcConnectionInfo& arc = *it->begin();

        const auto pts = discreteArcToPoint(arc, radius, circularSegments);
        rLoop.insert(rLoop.end(), pts.begin(), pts.end());

        currentEdgeIndex = arc.e2;
        endEdgeIndex = arc.e1;
        currentEdgeT = arc.t2;

        it->erase(it->begin());
        break;
      }
    }

    if (it == arcConnection.end()) break;

    // For detecting inner loop
    tracingEList.push_back(currentEdgeIndex);
    mapVV.push_back(rLoop.size());

    while (currentEdgeIndex != endEdgeIndex) {
      // Trace to find next arc on current edge
      auto it = std::find_if(
          arcConnection[currentEdgeIndex].begin(),
          arcConnection[currentEdgeIndex].end(),
          [currentEdgeT, EPSILON](const ArcConnectionInfo& ele) -> bool {
            return ele.t1 + EPSILON > currentEdgeT;
          });

      if (it == arcConnection[currentEdgeIndex].end()) {
        // Not found, just add vertex
        // FIXME: shouldn't add vertex directly, should search for next edge
        // with fillet arc
        rLoop.push_back(loop[(currentEdgeIndex + 1) % loop.size()]);
        currentEdgeIndex = (currentEdgeIndex + 1) % loop.size();
        currentEdgeT = 0;

        tracingEList.push_back(currentEdgeIndex);
        mapVV.push_back(rLoop.size());
      } else {
        // Found next circle fillet

        ArcConnectionInfo arc = *it;
        arcConnection[currentEdgeIndex].erase(it);

        const auto pts = discreteArcToPoint(arc, radius, circularSegments);
        rLoop.insert(rLoop.end(), pts.begin(), pts.end());

        // Check if current result contain inner loop
        auto itt =
            std::find(tracingEList.rbegin(), tracingEList.rend(), arc.e2);

        if (itt != tracingEList.rend()) {
          size_t pos = tracingEList.size() -
                       std::distance(tracingEList.rbegin(), itt) - 1;

          SimplePolygon innerLoop{};
          innerLoop.insert(innerLoop.end(), rLoop.begin() + mapVV[pos],
                           rLoop.end());

          newPoly.push_back(innerLoop);

          rLoop.erase(rLoop.begin() + mapVV[pos], rLoop.end());

          currentEdgeIndex = (currentEdgeIndex + 1) % loop.size();
          currentEdgeT = 0;

          continue;
        }

        currentEdgeIndex = arc.e2;
        currentEdgeT = arc.t2;

        tracingEList.push_back(currentEdgeIndex);
        mapVV.push_back(rLoop.size());
      }
    }
    newPoly.push_back(rLoop);
  }

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose) {
    std::cout << "Result loop count:" << newPoly.size() << std::endl;
  }
#endif

  return newPoly;
}

Vec<CrossSection> FilletImpl(const Polygons& polygons, double radius,
                             int circularSegments) {
  using namespace manifold;

  auto& loop = polygons[0];
  manifold::Polygons newPoly = polygons;
  auto& newLoop = newPoly[0];
  newLoop.push_back(loop[0]);

  ColliderInfo info{};
  info.outerCollider = BuildCollider(polygons[0], info.outerEdgeOld2NewVec);

  // Process inner loops
  info.innerCollider = std::vector<Collider>(polygons.size() - 1, Collider());
  info.innerVec = std::vector<std::vector<edgeOld2New>>(
      polygons.size() - 1, std::vector<edgeOld2New>());
  for (size_t i = 1; i != polygons.size(); i++) {
    info.innerCollider[i] = BuildCollider(polygons[i], info.innerVec[i]);
  }

  // Calc all arc that bridge 2 edge
  auto arcConnection = CalculateFilletArc(polygons, newPoly, info, radius);

  // Tracing along the arc
  int n = circularSegments > 2 ? circularSegments
                               : Quality::GetCircularSegments(radius);

  auto result = Tracing(loop, arcConnection, n, radius);

  newPoly.insert(newPoly.end(), result.begin(), result.end());

  return Vec<CrossSection>();
}

}  // namespace

namespace manifold {

struct PathImpl {
  PathImpl(const C2::PathsD paths_) : paths_(paths_) {}
  operator const C2::PathsD&() const { return paths_; }
  const C2::PathsD paths_;
};

Vec<CrossSection> CrossSection::Fillet(double radius,
                                       int circularSegments) const {
  auto paths = this->GetPaths()->paths_;
  Polygons polygons(paths.size(), SimplePolygon());

  for (size_t i = 0; i != paths.size(); i++) {
    auto& pts = polygons[i];

    for (auto p : paths[i]) {
      pts.push_back(v2_of_pd(p));
    }
  }

  return FilletImpl(polygons, radius, circularSegments);
}

Vec<CrossSection> CrossSection::Fillet(const SimplePolygon pts, double radius,
                                       int circularSegments) {
  return Fillet(Polygons{pts}, radius, circularSegments);
}

Vec<CrossSection> CrossSection::Fillet(const Polygons& polygons, double radius,
                                       int circularSegments) {
  return FilletImpl(polygons, radius, circularSegments);
}

}  // namespace manifold
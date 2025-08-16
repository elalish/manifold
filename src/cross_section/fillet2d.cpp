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

#include <fstream>
#include <iostream>
#include <optional>
#include <vector>

#include "../vec.h"

#define _USE_MATH_DEFINES
#include <cmath>

#include "../collider.h"
#include "clipper2/clipper.core.h"
#include "clipper2/clipper.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"

const double EPSILON = 1e-9;

using namespace manifold;
namespace C2 = Clipper2Lib;

namespace {

vec2 v2_of_pd(const C2::PointD p) { return {p.x, p.y}; }

C2::PointD v2_to_pd(const vec2 v) { return C2::PointD(v.x, v.y); }

C2::PathD pathd_of_contour(const SimplePolygon& ctr) {
  auto p = C2::PathD();
  p.reserve(ctr.size());
  for (auto v : ctr) {
    p.push_back(v2_to_pd(v));
  }
  return p;
}

}  // namespace

namespace {
struct SimpleLoop {
  SimplePolygon Loop;
  bool isCCW;
};

using Loops = std::vector<SimpleLoop>;
}  // namespace

namespace {
// Utility

vec3 toVec3(vec2 in) { return vec3(in.x, in.y, 0); }

double distancePointSegment(const vec2& p, const vec2& p1, const vec2& p2,
                            double& t) {
  vec2 d = p2 - p1;

  t = -1;

  t = la::dot(p - p1, d) / la::dot(d, d);

  vec2 closestPoint;

  if (t < 0) {
    t = 0;
    closestPoint = p1;
  } else if (t > 1) {
    t = 1;
    closestPoint = p2;
  } else {
    closestPoint = p1 + t * d;
  }

  return la::length(closestPoint - p);
}

// Check if line segment intersect with a circle
bool intersectCircleSegment(const vec2& p1, const vec2& p2, const vec2& center,
                            double radius) {
  vec2 d = p2 - p1;

  if (la::length(d) < EPSILON)
    return la::length2(p1 - center) <= radius * radius;

  double t;
  return distancePointSegment(center, p1, p2, t) < radius;
};

// Check if line segment intersect with another line segment
bool intersectSegmentSegment(const vec2& p1, const vec2& p2, const vec2& p3,
                             const vec2& p4, double& t, double& u) {
  double det = la::cross(p2 - p1, p4 - p3);

  if (std::abs(det) < EPSILON) {
    // Parallel
    return false;
  }

  double num_t = la::cross(p3 - p1, p4 - p3);
  double num_u = la::cross(p3 - p1, p2 - p1);

  t = num_t / det;
  u = num_u / det;

  // Check if the intersection point inside line segment
  if ((t >= 0.0 - EPSILON && t <= 1.0 + EPSILON) &&
      (u >= 0.0 - EPSILON && u <= 1.0 + EPSILON)) {
    // Inside
    return true;
  } else {
    // Outside -> No intersection
    t = u = 0;

    return false;
  }
};

enum class IntersectStadiumResult {
  EdgeEdgeIntersect,  // Surely a result
  P2Degenerate,       // Degenerate to P2
  P1Degenerate,       // Only use to assert
  P1P2Degenerate,     // Both degenerate, and e1 should be removed
  E2Degenerate,       // NOTE: e2 full inside and parallel with e1, this can be
                      // skipped by now, later process will handle it
  Outside,            // No result
};

IntersectStadiumResult intersectStadiumCollider(
    const vec2& p1, const vec2& p2,
    const bool e1CCW,  // p1 -> p2 CCW ?
    const vec2& p3, const vec2& p4,
    const bool e2CCW,  // p3 -> p4 CCW ?
    const double radius) {
  // Two boundary shouldn't touch, this should be handle previous
  // if (e1CCW && e2CCW) throw std::exception();

  const vec2 e1 = p2 - p1, e2 = p4 - p3,
             normal1 =
                 la::normalize(e1CCW ? vec2(-e1.y, e1.x) : vec2(e1.y, -e1.x));

  const double det = la::cross(e1, e2);
  const bool e2Parallel = std::abs(det) < EPSILON;

  if (e2Parallel) {
    double distance = la::dot(p3 - p1, normal1);

    if (distance >= radius) return IntersectStadiumResult::Outside;
  }

  // Check is intersect at endpoint, result in degenerate
  {
    const bool p1Intersect =
        intersectCircleSegment(p3, p4, p1 + normal1 * radius, radius);

    const bool p2Intersect =
        intersectCircleSegment(p3, p4, p2 + normal1 * radius, radius);

    if (p1Intersect && p2Intersect)
      return IntersectStadiumResult::P1P2Degenerate;

    if (p1Intersect) {
      bool sign = la::cross(e1, e2) < 0;

      if (sign == (e1CCW ^ e2CCW)) {
        // TODO: Ensure degenerate processed
        return IntersectStadiumResult::P1Degenerate;
      } else {
        return IntersectStadiumResult::EdgeEdgeIntersect;
      }
    }

    if (p2Intersect) {
      bool sign = la::cross(e1, e2) > 0;

      if (sign == (e1CCW ^ e2CCW)) {
        // Degenerate process
        return IntersectStadiumResult::P2Degenerate;
      } else {
        return IntersectStadiumResult::EdgeEdgeIntersect;
      }
    }
  }

  // Check edge intersect
  {
    double t, u;
    if (intersectSegmentSegment(p1, p2, p3, p4, t, u))
      return IntersectStadiumResult::EdgeEdgeIntersect;
  }

  // Now e2 is either both in or both out, check this
  {
    const vec2 p1Offset = p1 + normal1 * 2.0 * radius,
               p2Offset = p2 + normal1 * 2.0 * radius;

    // Determine full inside or full outside
    auto isInsideRect = [&](const vec2 p) -> bool {
      int sign = e1CCW ? 1 : -1;
      if ((sign * la::cross(e1, p - p1) >= 0) &&
          (sign * la::cross(p2Offset - p2, p - p2) >= 0) &&
          (sign * la::cross(p1Offset - p2Offset, p - p2Offset) >= 0) &&
          (sign * la::cross(p1 - p1Offset, p - p1Offset) >= 0))
        return true;

      return false;
    };

    bool p3InsideRect = isInsideRect(p3), p4InsideRect = isInsideRect(p4);
    if (p3InsideRect && p4InsideRect) {
      // Full inside, no intersect with boundary
      // FIXME: both inside, which must lead to EdgeEdgeIntersect, but
      // rethink about it.
      if (e2Parallel) return IntersectStadiumResult::E2Degenerate;

      return IntersectStadiumResult::EdgeEdgeIntersect;
    } else if (!p3InsideRect && !p4InsideRect) {
      // Full outside, no intersect with boundary
      return IntersectStadiumResult::Outside;
    } else {
      // Should be both in or both out
      throw std::exception();
    }
  }

  return IntersectStadiumResult::Outside;
}

// Normalize angle to [0, 2*PI)
float normalizeAngle(float angle) {
  while (angle < 0) angle += M_2_PI;
  while (angle >= M_2_PI) angle -= M_2_PI;
  return angle;
}

bool intersectSectorCollider(const vec2& ppre, const vec2& p, const vec2& pnext,
                             const bool pCCW, const vec2& p1, const vec2& p2,
                             const bool eCCW, const double radius) {
  const bool pIntersect = intersectCircleSegment(p1, p2, p, 2.0 * radius);
  if (!pIntersect) return false;

  const vec2 epre = p - ppre, enext = pnext - p, e = p1 - p2,
             normalpre = la::normalize(pCCW ? vec2(-epre.y, epre.x)
                                            : vec2(epre.y, -epre.x)),
             normalnext = la::normalize(pCCW ? vec2(-enext.y, enext.x)
                                             : vec2(enext.y, -enext.x));

  // ppreIntersect have been checked in intersectStadiumCollider

  const bool pnextIntersect =
      intersectCircleSegment(p1, p2, p + normalnext * radius, radius);
  if (pnextIntersect) {
    bool sign = la::cross(enext, e) > 0;

    if (sign == (pCCW ^ eCCW)) {
      // Degenerate
      return true;
    } else {
      // EdgeEdge Intersect will be processed next edge
      return false;
    }
  }

  auto getLineCircleIntersection =
      [](const vec2& p1, const vec2& p2, const vec2& center, float radius,
         vec2& intersection1, vec2& intersection2) -> int {
    vec2 d = p2 - p1;
    vec2 f = p1 - center;

    float a = dot(d, d);
    float b = 2 * dot(f, d);
    float c = dot(f, f) - radius * radius;

    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return 0;  // No intersection

    float sqrtDisc = sqrt(discriminant);
    double t1 = (-b - sqrtDisc) / (2 * a);
    double t2 = (-b + sqrtDisc) / (2 * a);

    int count = 0;
    if (t1 >= -EPSILON && t1 <= 1 + EPSILON) {
      intersection1 = p1 + d * t1;
      count++;
    }
    if (t2 >= -EPSILON && t2 <= 1 + EPSILON && abs(t2 - t1) > EPSILON) {
      if (count == 0) {
        intersection1 = p1 + d * t2;
      } else {
        intersection2 = p1 + d * t2;
      }
      count++;
    }

    return count;
  };

  auto isAngleInSector = [](float angle, float startRad, float endRad) -> bool {
    angle = normalizeAngle(angle);
    startRad = normalizeAngle(startRad);
    endRad = normalizeAngle(endRad);

    if (startRad <= endRad) {
      return angle >= startRad && angle <= endRad;
    } else {
      // Sector crosses 0 degrees
      return angle >= startRad || angle <= endRad;
    }
  };

  auto isPointInPie = [&isAngleInSector](const vec2& p, const vec2& center,
                                         float radius, float startRad,
                                         float endRad) -> bool {
    vec2 diff = p - center;
    float distSq = length2(diff);

    if (distSq > radius * radius + EPSILON) return false;

    float angle = atan2(diff.y, diff.x);
    return isAngleInSector(angle, startRad, endRad);
  };

  vec2 intersections[2];
  int numIntersections = getLineCircleIntersection(
      p1, p2, p, 2.0 * radius, intersections[0], intersections[1]);

  double startRad = atan2(epre.y, epre.x), endRad = atan2(enext.y, enext.x);

  for (int i = 0; i < numIntersections; i++) {
    vec2 diff = intersections[i] - p;
    float angle = atan2(diff.y, diff.x);
    if (isAngleInSector(angle, startRad, endRad)) {
      return true;
    }
  }

  return false;
}

// Projection point to line and check if it's on the line segment
bool isProjectionOnSegment(const vec2& c, const vec2& p1, const vec2& p2,
                           double& t) {
  t = la::dot(c - p1, p2 - p1) / la::length2(p2 - p1);

  return t >= 0 && t <= 1;
};

enum class PointSegmentResult {
  PreviousEdgeIntersect,
  NextEdgeIntersect,
  BothIntersect,
  Ignore
};

struct PointSegmentIntersectResult {
  double eT;
  vec2 circleCenter;
  double endPointRad, edgeTangentRad;
};

// Calculate Circle center that tangent to one segment, and cross a point
PointSegmentResult calculatePointSegmentCircleCenter(
    const vec2& ppre, const vec2& p, const vec2& pnext, const bool pCCW,
    const vec2& p1, const vec2& p2, const bool eCCW, double radius,
    PointSegmentIntersectResult& result1,
    PointSegmentIntersectResult& result2) {
  double t, dis = distancePointSegment(p, p1, p2, t);
  if (dis > radius)
    throw std::exception();
  else if (dis == radius) {
    // FIXME: EPSILON
    // Ignore, mark as outside
    return PointSegmentResult::Ignore;
  }

  vec2 e = p2 - p1, dir = la::normalize(e),
       normal = la::normalize(vec2(-e.y, e.x));

  vec2 projectedP = p1 + dir * la::dot(p - p1, dir);
  double dist = la::length(projectedP - p);
  dist = dist > radius ? dist - radius : radius - dist;

  double len = la::sqrt(radius * radius - dist * dist);

  std::array<vec2, 2> tangentPoint{projectedP - dir * len,
                                   projectedP + dir * len};

  auto isOnSegment = [](const vec2& p1, const vec2& p2, const vec2& p) -> bool {
    double v = la::dot(p - p1, p2 - p1);
    return 0 <= v && v <= la::length2(p2 - p1);
  };

  std::array<bool, 2> edgeIntersect = {false, false};

  for (int i = 0; i != 2; i++) {
    vec2 tangent = tangentPoint[i];
    vec2 center(0, 0);

    if (!isOnSegment(p1, p2, tangent)) {
      // Tangent valid -> check circle valid
      vec2 center = tangent + normal * radius;

    } else {
      tangent = (i == 0) ? p1 : p2;

      vec2 d = p1 - p, ddir = la::normalize(d);
      vec2 dnormal = vec2(-ddir.y, ddir.x);

      double halfLength = la::length(d) / 2;
      double projectDistance =
          la::sqrt(radius * radius - halfLength * halfLength);

      vec2 center1 = p + halfLength * ddir + projectDistance * dnormal,
           center2 = p + halfLength * ddir - projectDistance * dnormal;

      center = center1;
      double t = 0;
      if (distancePointSegment(center1, p1, p2, t) < radius || t != (i == 0)
              ? 0
              : 1) {
        center = center2;
      }
    }

    vec2 pointToCenter = center - p;
    vec2 tangentToCenter = center - tangent;
    PointSegmentIntersectResult result{
        la::dot(tangent - p1, e) / la::length2(e), tangent + normal * radius,
        atan2(pointToCenter.y, pointToCenter.x),
        atan2(tangentToCenter.y, tangentToCenter.x)};

    vec2 tangentDir = vec2(pointToCenter.y, -pointToCenter.x);

    const vec2 epre = p - ppre, enext = pnext - p;

    bool sign = la::cross(e, pointToCenter) > 0;
    if (sign & eCCW) {
      // Pre Edge

      // Check is collision with next edge
      if (la::cross(enext, tangentDir) > 0) {
        edgeIntersect[0] = true;
        result1 = result;
      }
    } else {
      // Next Edge

      // Check is collision with pre edge
      if (la::cross(epre, tangentDir) > 0) {
        edgeIntersect[1] = true;
        result2 = result;
      }
    }
  }

  if (edgeIntersect[0] && edgeIntersect[1]) {
    return PointSegmentResult::BothIntersect;

  } else if (edgeIntersect[0]) {
    return PointSegmentResult::PreviousEdgeIntersect;

  } else if (edgeIntersect[1]) {
    return PointSegmentResult::NextEdgeIntersect;
  }

  return PointSegmentResult::Ignore;
}

bool calculateSegmentSegmentCircleCenter(const vec2& p1, const vec2& p2,
                                         const vec2& p3, const vec2& p4,
                                         const double radius, double& e1T,
                                         double& e2T, vec2& circleCenter,
                                         double& startRad, double& endRad) {
  e1T = e2T = 0;
  circleCenter = vec2(0, 0);

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
    // NOTE: degenerate case, should be handled in PointSegment
    throw std::exception();
  }

  vec2 center = la::mul(la::inverse(A), b);
  double r1T = 0, r2T = 0;

  bool onE1 = isProjectionOnSegment(center, p1, p2, r1T),
       onE2 = isProjectionOnSegment(center, p3, p4, r2T);

  // Check Circle center projection for tangent point status
  if (onE1 && onE2) {
    // Tangent point on both line segment
    circleCenter = center;

    e1T = r1T;
    e2T = r2T;

    vec2 tangent1 = p1 + e1T * e1;
    vec2 tangent2 = p3 + e2T * e2;

    vec2 dirStart = tangent1 - circleCenter;
    vec2 dirEnd = tangent2 - circleCenter;

    startRad = atan2(dirStart.y, dirStart.x);
    endRad = atan2(dirEnd.y, dirEnd.x);

    startRad = normalizeAngle(startRad);
    endRad = normalizeAngle(endRad);

    return true;
  }

  return false;
}

enum class ArcEdgeState {
  E1CurrentEdge,
  E1NextEdge,
  E2PreviousEdge,
  E2CurrentEdge,
  E2NextEdge
};

struct ArcBridgeInfo {
  std::array<ArcEdgeState, 2> States;
  std::array<double, 2> ParameterValues;
  vec2 CircleCenter;
  std::array<double, 2> RadValues;
};

int calculatePointSegmentCircles(const vec2& p, const vec2& p1, const vec2& p2,
                                 const bool eCCW, double radius,
                                 std::array<vec2, 2>& circleCenters) {
  vec2 e = p2 - p1, dir = la::normalize(e),
       normal = la::normalize(CCW ? vec2(-e.y, e.x) : vec2(e.y, -e.x));

  vec2 projectedP = p1 + dir * la::dot(p - p1, dir);
  double dist = la::length(projectedP - p);
  dist = dist > radius ? dist - radius : radius - dist;

  double len = la::sqrt(radius * radius - dist * dist);

  std::array<vec2, 2> tangentPoint{projectedP - dir * len,
                                   projectedP + dir * len};

  auto isOnSegment = [](const vec2& p1, const vec2& p2, const vec2& p) -> bool {
    double v = la::dot(p - p1, p2 - p1);
    return 0 <= v && v <= la::length2(p2 - p1);
  };

  int count = 0;
  for (int i = 0; i != 2; i++) {
    vec2 tangent = tangentPoint[i];
    if (isOnSegment(p1, p2, tangent)) {
      // Tangent valid -> check circle valid
      circleCenters[i] = tangent + normal * radius;
      count++;
    }
  }

  // No result check
  if (!count) throw std::exception();

  return count;
}

int calculatePointPointCircles(const vec2& p1, const vec2& p2, double radius,
                               std::array<vec2, 2>& circleCenters) {
  double dx = p2.x - p1.x;
  double dy = p2.y - p1.y;
  double distance = sqrt(dx * dx + dy * dy);

  vec2 v = p2 - p1;

  // Find midpoint between the two points
  vec2 midpoint((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);

  // If distance equals 2*radius, there's exactly one solution (midpoint)
  if (abs(la::length(v) - 2 * radius) < 1e-9) {
    circleCenters[0] = midpoint;
    return 1;
  }

  // Calculate distance from midpoint to circle centers
  double h = sqrt(radius * radius - (distance / 2.0) * (distance / 2.0));

  // Calculate unit vector perpendicular to the line connecting p1 and p2
  vec2 d(-dy, dx);
  d /= distance;

  circleCenters[0] = midpoint + h * d;
  circleCenters[1] = midpoint - h * d;

  return 2;
}

std::vector<ArcBridgeInfo> calculateStadiumIntersect(
    const std::array<vec2, 3>& e1Points, const bool e1CCW,
    const std::array<vec2, 4>& e2Points, const bool e2CCW,
    const double radius) {
  const vec2 e1Cur = e1Points[1] - e1Points[0],
             e1Next = e1Points[2] - e1Points[1],
             e2Pre = e2Points[1] - e2Points[0],
             e2Cur = e2Points[2] - e2Points[1],
             e2Next = e2Points[3] - e2Points[2];
  auto getNormal = [](bool CCW, vec2 e) -> vec2 {
    return la::normalize(CCW ? vec2(-e.y, e.x) : vec2(e.y, -e.x));
  };

  std::vector<ArcBridgeInfo> arcBridgeInfoVec;

  const vec2 e1CurNormal = getNormal(e1CCW, e1Cur),
             e2CurNormal = getNormal(e2CCW, e2Cur);
  {
    // Edge - Edge

    const std::array<vec2, 2> offsetE1{
        e1Points[0] + e1CurNormal * radius,
        e1Points[1] + e1CurNormal * radius,
    },
        offsetE2{
            e2Points[1] + e2CurNormal * radius,
            e2Points[2] + e2CurNormal * radius,
        };

    double t, u;
    if (intersectSegmentSegment(offsetE1[0], offsetE1[1], offsetE2[0],
                                offsetE2[1], t, u)) {
      arcBridgeInfoVec.emplace_back(ArcBridgeInfo{
          std::array<ArcEdgeState, 2>{ArcEdgeState::E1CurrentEdge,
                                      ArcEdgeState::E2CurrentEdge},
          std::array<double, 2>{t, u}, e1Points[0] + e1Cur * t,
          std::array<double, 2>{0, 0}});
    }
  }

  {
    const vec2 p1 = e1Points[0], p2 = e1Points[1];
    const vec2 p1Offset = p1 + e1CurNormal * 2.0 * radius,
               p2Offset = p2 + e1CurNormal * 2.0 * radius;

    // Determine full inside or full outside
    auto isInsideRect = [&](const vec2 p) -> bool {
      int sign = e1CCW ? 1 : -1;
      if ((sign * la::cross(e1Cur, p - p1) >= 0) &&
          (sign * la::cross(p2Offset - p2, p - p2) >= 0) &&
          (sign * la::cross(p1Offset - p2Offset, p - p2Offset) >= 0) &&
          (sign * la::cross(p1 - p1Offset, p - p1Offset) >= 0))
        return true;

      return false;
    };

    auto isInsideEndpointCircles = [&](const vec2 p) -> bool {
      if (la::length2(p - e1Points[0] + e1CurNormal * radius) <=
          radius * radius)
        return true;

      if (la::length2(p - e2Points[0] + e2CurNormal * radius) <=
          radius * radius)
        return true;

      return false;
    };

    // Edge - Point

    std::array<vec2, 2> points{e2Points[1], e2Points[2]};
    for (int i = 0; i != 2; i++) {
      const vec2 point = points[i];
      if (isInsideRect(point) || isInsideEndpointCircles(point)) {
        std::array<vec2, 2> centers;
        int count =
            calculatePointSegmentCircles(point, p1, p2, e1CCW, radius, centers);
        for (int j = 0; j != count; j++) {
          int sign = e2CCW ? 1 : -1;
          if (sign * la::cross(e2Points[i] - point, centers[j] - point) < 0 &&
              sign * la::cross(e2Points[i + 2] - point, centers[j] - point) >
                  0) {
            arcBridgeInfoVec.emplace_back(ArcBridgeInfo{
                std::array<ArcEdgeState, 2>{ArcEdgeState::E1CurrentEdge,
                                            j == 0
                                                ? ArcEdgeState::E2PreviousEdge
                                                : ArcEdgeState::E2NextEdge},
                std::array<double, 2>{0, 1}, centers[j],
                std::array<double, 2>{0, 0}});
          }
        }
      }
    }
  }

  return arcBridgeInfoVec;
}

std::vector<ArcBridgeInfo> calculateSectorIntersect(
    const std::array<vec2, 3>& e1Points, const bool e1CCW,
    const std::array<vec2, 4>& e2Points, const bool e2CCW,
    const double radius) {
  if (!intersectCircleSegment(e2Points[1], e2Points[2], e1Points[1],
                              2.0 * radius))
    return {};

  std::vector<ArcBridgeInfo> arcBridgeInfoVec;

  const vec2 e1Cur = e1Points[1] - e1Points[0],
             e1Next = e1Points[2] - e1Points[1],
             e2Pre = e2Points[1] - e2Points[0],
             e2Cur = e2Points[2] - e2Points[1],
             e2Next = e2Points[3] - e2Points[2];
  auto getNormal = [](bool CCW, vec2 e) -> vec2 {
    return la::normalize(CCW ? vec2(-e.y, e.x) : vec2(e.y, -e.x));
  };

  const vec2 e1CurNormal = getNormal(e1CCW, e1Cur),
             e1NextNormal = getNormal(e1CCW, e1Next);

  double startRad = atan2(e1CurNormal.y, e1CurNormal.x),
         endRad = atan2(e1NextNormal.y, e1NextNormal.x);

  auto getLineCircleIntersection =
      [](const vec2& p1, const vec2& p2, const vec2& center, float radius,
         std::array<vec2, 2> intersections) -> int {
    vec2 d = p2 - p1;
    vec2 f = p1 - center;

    float a = dot(d, d);
    float b = 2 * dot(f, d);
    float c = dot(f, f) - radius * radius;

    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return 0;  // No intersection

    float sqrtDisc = sqrt(discriminant);
    double t1 = (-b - sqrtDisc) / (2 * a);
    double t2 = (-b + sqrtDisc) / (2 * a);

    int count = 0;
    if (t1 >= -EPSILON && t1 <= 1 + EPSILON) {
      intersections[count] = p1 + d * t1;
      count++;
    }
    if (t2 >= -EPSILON && t2 <= 1 + EPSILON && abs(t2 - t1) > EPSILON) {
      intersections[count] = p1 + d * t2;
      count++;
    }

    return count;
  };

  auto isAngleInSector = [](float angle, float startRad, float endRad) -> bool {
    angle = normalizeAngle(angle);
    startRad = normalizeAngle(startRad);
    endRad = normalizeAngle(endRad);

    if (startRad <= endRad) {
      return angle >= startRad && angle <= endRad;
    } else {
      // Sector crosses 0 degrees
      return angle >= startRad || angle <= endRad;
    }
  };

  auto isPointInPie = [&isAngleInSector](const vec2& p, const vec2& center,
                                         float radius, float startRad,
                                         float endRad) -> bool {
    vec2 diff = p - center;
    float distSq = length2(diff);

    if (distSq > radius * radius + EPSILON) return false;

    float angle = atan2(diff.y, diff.x);
    return isAngleInSector(angle, startRad, endRad);
  };

  // Point - Edge
  std::array<vec2, 2> intersections;
  int count = getLineCircleIntersection(e2Points[1], e2Points[2], e1Points[1],
                                        radius, intersections);
  for (int i = 0; i != count; i++) {
    vec2 diff = intersections[i] - e1Points[1];
    float angle = atan2(diff.y, diff.x);
    if (isAngleInSector(angle, startRad, endRad)) {
      arcBridgeInfoVec.emplace_back(ArcBridgeInfo{
          std::array<ArcEdgeState, 2>{ArcEdgeState::E1CurrentEdge,
                                      ArcEdgeState::E2CurrentEdge},
          std::array<double, 2>{0, 1}, intersections[i],
          std::array<double, 2>{0, 0}});
    }
  }

  //  Point - Point
  std::array<vec2, 2> points{e2Points[1], e2Points[2]};
  for (int i = 0; i != 2; i++) {
    const vec2 point = points[i];
    if (isPointInPie(point, e1Points[1], radius, startRad, endRad)) {
      std::array<vec2, 2> centers;
      int count =
          calculatePointPointCircles(point, e1Points[1], radius, centers);
      for (int j = 0; j != count; j++) {
        int sign = e2CCW ? 1 : -1;
        if (sign * la::cross(e2Points[i] - point, centers[j] - point) < 0 &&
            sign * la::cross(e2Points[i + 2] - point, centers[j] - point) > 0) {
          arcBridgeInfoVec.emplace_back(ArcBridgeInfo{
              std::array<ArcEdgeState, 2>{ArcEdgeState::E1CurrentEdge,
                                          j == 0 ? ArcEdgeState::E2PreviousEdge
                                                 : ArcEdgeState::E2NextEdge},
              std::array<double, 2>{0, 1}, centers[j],
              std::array<double, 2>{0, 0}});
        }
      }
    }
  }

  return arcBridgeInfoVec;
}

struct ArcConnectionInfo {
  vec2 center;

  double t1, t2;  // Parameter value of arc tangent points of edges
  size_t e1, e2;  // Edge Idx of tangent points lie on
  size_t e1Loopi, e2Loopi;

  double startRad, endRad;
};

std::vector<vec2> discreteArcToPoint(ArcConnectionInfo arc, double radius,
                                     int circularSegments) {
  std::vector<vec2> pts;

  double totalRad = normalizeAngle(arc.endRad - arc.startRad);

  double dPhi = 2 * M_PI / circularSegments;
  int seg = int(totalRad / dPhi) + 1;
  for (int i = 0; i != seg + 1; ++i) {
    double current = arc.startRad + dPhi * i;
    if (i == seg) current = arc.endRad;

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
  size_t loopRef;
};

manifold::Collider BuildCollider(const manifold::Polygons& polygons,
                                 std::vector<edgeOld2New>& edgeOld2NewVec) {
  Vec<manifold::Box> boxVec;
  Vec<uint32_t> mortonVec;

  for (size_t j = 0; j != polygons.size(); j++) {
    auto& loop = polygons[j];
    for (size_t i = 0; i != loop.size(); i++) {
      const vec2 p1 = loop[i], p2 = loop[(i + 1) % loop.size()];

      vec3 center = toVec3(p1) + toVec3(p2);
      center /= 2;

      manifold::Box bbox(toVec3(p1), toVec3(p2));

      edgeOld2NewVec.push_back({bbox,
                                manifold::Collider::MortonCode(center, bbox), i,
                                (i + 1) % loop.size(), j});
    }
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
    const Loops& loops, const ColliderInfo& collider, double radius) {
  std::vector<size_t> loopOffset(loops.size());

  bool invert = false;
  if (radius < EPSILON) invert = true;

  size_t count = 0;
  for (size_t i = 0; i != loops.size(); i++) {
    loopOffset[i] = count;
    count += loops[i].Loop.size();
  }

  const size_t loopElementCount = count;

  std::vector<bool> markEE(loopElementCount * loopElementCount, false);
  std::vector<bool> markEV(loopElementCount * loopElementCount, false);
  std::vector<bool> markVV(loopElementCount * loopElementCount, false);

  auto getMarkPosition = [&loopElementCount, &loopOffset](
                             const size_t loop1i, const size_t ele1i,
                             const size_t loop2i,
                             const size_t ele2i) -> size_t {
    return (loopOffset[loop1i] + ele1i) * loopElementCount +
           loopOffset[loop2i] + ele2i;
  };

  std::vector<std::vector<ArcConnectionInfo>> arcConnection(
      loopElementCount, std::vector<ArcConnectionInfo>());

  std::cout << "Collider BBox Testing" << std::endl;

  for (size_t e1Loopi = 0; e1Loopi != loops.size(); e1Loopi++) {
    const auto& e1Loop = loops[e1Loopi].Loop;
    const bool isE1LoopCCW = loops[e1Loopi].isCCW ^ invert;

    // create BBox for every line to find Collision
    for (size_t e1i = 0; e1i != e1Loop.size(); e1i++) {
      // Outer loop is CCW, p1 p2 -> current edge start end
      const size_t p1i = e1i, p2i = (e1i + 1) % e1Loop.size();
      const vec2 p1 = e1Loop[p1i], p2 = e1Loop[p2i];
      const vec2 e1 = p2 - p1;
      const bool p2IsConvex =
          la::cross(e1, e1Loop[(p2i + 1) % e1Loop.size()] - p2) >= EPSILON;

      const vec2 normal = la::normalize(vec2(-e1.y, e1.x));

      // Create BBox
      manifold::Box box(toVec3(p1), toVec3(p2));
      {
        // See
        // https://docs.google.com/presentation/d/1P-3oxmjmEw_Av0rq7q7symoL5VB5DSWpRvqoa3LK7Pg/edit?usp=sharing

        vec2 normalOffsetP1 = p1 + normal * 2.0 * radius,
             normalOffsetP2 = p2 + normal * 2.0 * radius;

        box.Union(toVec3(normalOffsetP1));
        box.Union(toVec3(normalOffsetP2));

        const vec2 e1n = la::normalize(e1);
        box.Union(toVec3(p1 - e1n * radius + normal * radius));
        box.Union(toVec3(p2 + e1n * radius + normal * radius));

        if (!p2IsConvex) {
          const vec2 pnext = e1Loop[(p2i + 1) % e1Loop.size()],
                     enext = pnext - p2, enextn = la::normalize(enext),
                     normalnext = la::normalize(vec2(-enext.y, enext.x));

          box.Union(toVec3(p2 + normalnext * 2.0 * radius));
          box.Union(toVec3(p2 + enextn * radius + normalnext * radius));
        }
      }

      auto r = collider.outerCollider.Collisions(
          manifold::Vec<manifold::Box>({box}).cview());
      // r.Dump();
      r.Sort();

      std::cout << std::endl
                << "Now " << p1i << "->" << (e1i + 1) % e1Loop.size() << " "
                << r.size() << std::endl;

      // In Out Classify
      for (size_t j = 0; j != r.size(); j++) {
        const auto& ele = collider.outerEdgeOld2NewVec[r.Get(j, true)];

        // e2i is the index of detected possible bridge edge

        const size_t e2i = ele.p1Ref, e2Loopi = ele.loopRef;
        const auto& e2Loop = loops[e2Loopi].Loop;
        const bool isE2LoopCCW = loops[e2Loopi].isCCW ^ invert;

        // Skip self and pre one, only process forward
        if (e1Loopi == e2Loopi &&
            ((e1i == e2i) ||
             (e1i == (e1i + e1Loop.size() - 1) % e1Loop.size())))
          continue;

        // Check if processed, and add duplicate mark
        markEE[getMarkPosition(e1Loopi, e1i, e2Loopi, e2i)] = 1;
        if (markEE[getMarkPosition(e2Loopi, e2i, e1Loopi, e1i)] != 0) {
          std::cout << "Skipped" << std::endl;
          continue;
        }

        // CCW, p3 p4 -> bbox hit edge start end
        size_t p3i = e2i, p4i = ele.p2Ref;
        vec2 p3 = e2Loop[p3i], p4 = e2Loop[p4i];

        bool continueFlag = false;
        bool degenerateFlag = false;

        switch (intersectStadiumCollider(p1, p2, isE1LoopCCW, p3, p4,
                                         isE2LoopCCW, radius)) {
          case IntersectStadiumResult::EdgeEdgeIntersect: {
            double startRad = 0, endRad = 0;
            double e1T = 0, e2T = 0;
            vec2 circleCenter(0, 0);

            if (calculateSegmentSegmentCircleCenter(p1, p2, p3, p4, radius, e1T,
                                                    e2T, circleCenter, startRad,
                                                    endRad)) {
              // Sort result by CCW
              double arcAngle = endRad - startRad;
              arcAngle = normalizeAngle(arcAngle);

              if (arcAngle <= M_PI) {
                arcConnection[loopOffset[e1Loopi] + e1i].emplace_back(
                    ArcConnectionInfo{circleCenter, e1T, e2T, e1i, e2i, e1Loopi,
                                      e2Loopi, startRad, endRad});
              } else {
                arcConnection[loopOffset[e2Loopi] + e2i].emplace_back(
                    ArcConnectionInfo{circleCenter, e2T, e1T, e2i, e1i, e2Loopi,
                                      e1Loopi, endRad, startRad});
              }

              std::cout << "Segment Center " << circleCenter << std::endl;

              continueFlag = true;
            } else {
              throw std::exception();
            }

            break;
          }
          case IntersectStadiumResult::E2Degenerate: {
            continueFlag = true;
            break;
          }
          case IntersectStadiumResult::P1Degenerate: {
            // Check previous flag
            continueFlag = true;
            break;
          }
          case IntersectStadiumResult::P2Degenerate:
          case IntersectStadiumResult::P1P2Degenerate: {
            // Degenerate
            degenerateFlag = true;
            break;
          }
          case IntersectStadiumResult::Outside: {
            // Keep checking, no early exit
            break;
          }
        }

        if (continueFlag) continue;

        if (p2IsConvex || (!degenerateFlag &&
                           !intersectSectorCollider(p1, p2, vec2(), true, p3,
                                                    p4, false, radius))) {
          continue;
        }

        // Handle concave vertex degenerate case

        std::array<PointSegmentIntersectResult, 2> r{};

        const vec2 pnext = e1Loop[(p2i + 1) % e1Loop.size()];

        const size_t enexti = p2i, eprei = e1i;

        auto addArcConnection =
            [&arcConnection, loopOffset, e1Loopi, e2Loopi](
                const double t, const size_t edgeIndex,
                const size_t resultEdgeIndex,
                PointSegmentIntersectResult& result) -> void {
          double arcAngle =
              normalizeAngle(result.edgeTangentRad - result.endPointRad);

          if (arcAngle <= M_PI) {
            arcConnection[loopOffset[e1Loopi] + edgeIndex].emplace_back(
                ArcConnectionInfo{result.circleCenter, t, result.eT, edgeIndex,
                                  resultEdgeIndex, e1Loopi, e2Loopi,
                                  result.endPointRad, result.edgeTangentRad});
          } else {
            arcConnection[loopOffset[e2Loopi] + resultEdgeIndex].emplace_back(
                ArcConnectionInfo{result.circleCenter, result.eT, t,
                                  resultEdgeIndex, edgeIndex, e2Loopi, e1Loopi,
                                  result.edgeTangentRad, result.endPointRad});
          }
        };

        switch (calculatePointSegmentCircleCenter(p1, p2, pnext, isE1LoopCCW,
                                                  p3, p4, isE2LoopCCW, radius,
                                                  r[0], r[1])) {
          case PointSegmentResult::PreviousEdgeIntersect:
            // Add to previous edge
            addArcConnection(1, eprei, e2i, r[0]);
            break;
          case PointSegmentResult::NextEdgeIntersect:
            // Add to next edge
            addArcConnection(0, enexti, e2i, r[1]);
            break;
          case PointSegmentResult::BothIntersect:
            addArcConnection(1, eprei, e2i, r[0]);
            addArcConnection(0, enexti, e2i, r[1]);
            break;
          case PointSegmentResult::Ignore:
          default:
            break;
        }
      }
    }
  }

  // Detect arc self-intersection, and remove
  for (size_t i = 0; i != arcConnection.size(); i++) {
    for (auto it = arcConnection[i].begin(); it != arcConnection[i].end();
         it++) {
      const auto& arc = *it;

      manifold::Box box(toVec3(arc.center - vec2(1, 0) * radius),
                        toVec3(arc.center + vec2(1, 0) * radius));

      box.Union(toVec3(arc.center - vec2(0, 1) * radius));
      box.Union(toVec3(arc.center + vec2(0, 1) * radius));

      auto r = collider.outerCollider.Collisions(
          manifold::Vec<manifold::Box>({box}).cview());
      // r.Dump();
      r.Sort();

      for (size_t k = 0; k != r.size(); k++) {
        const auto& edge = collider.outerEdgeOld2NewVec[r.Get(k, true)];

        if (edge.loopRef == arc.e1Loopi && edge.p1Ref == arc.e1)
          continue;
        else if (edge.loopRef == arc.e2Loopi && edge.p1Ref == arc.e2)
          continue;

        const auto& eLoop = loops[edge.loopRef].Loop;
        const size_t p1i = edge.p1Ref, p2i = edge.p2Ref;
        const vec2 p1 = eLoop[p1i], p2 = eLoop[p2i];
        if (intersectCircleSegment(p1, p2, arc.center, radius)) {
          it = arcConnection[i].erase(it);
          std::cout << "Remove" << arc.center << std::endl;
        }
      }
    }
  }

  std::ofstream f("circle.txt");

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose) {
    for (size_t i = 0; i != arcConnection.size(); i++) {
      std::cout << i << " " << arcConnection[i].size();
      for (size_t j = 0; j != arcConnection[i].size(); j++) {
        std::cout << "\t" << arcConnection[i][j].e1 << " "
                  << arcConnection[i][j].e2 << " " << arcConnection[i][j].t1
                  << " " << arcConnection[i][j].t2 << " <"
                  << arcConnection[i][j].center << "> "
                  << arcConnection[i][j].startRad << " "
                  << arcConnection[i][j].endRad << std::endl;

        f << arcConnection[i][j].center.x << " " << arcConnection[i][j].center.y
          << std::endl;
      }

      std::cout << std::endl;
    }
  }
#endif

  f.close();

  return arcConnection;
}

manifold::Polygons Tracing(
    const Loops& loops,
    std::vector<std::vector<ArcConnectionInfo>> arcConnection,
    int circularSegments, double radius) {
  const double EPSILON = 1e-9;

  const manifold::SimplePolygon& loop = loops[0].Loop;

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

Polygons FilletImpl(const Polygons& polygons, double radius,
                    int circularSegments) {
  ColliderInfo info{};
  info.outerCollider = BuildCollider(polygons, info.outerEdgeOld2NewVec);

  Loops loops;
  loops.reserve(polygons.size());

  for (const auto& loop : polygons) {
    C2::PathD path = pathd_of_contour(loop);

    loops.push_back(SimpleLoop{loop, C2::Area(path) > EPSILON});
  }

  for (size_t i = 0; i != polygons.size(); i++) {
    std::cout << "------ Loop " << i << " START" << std::endl;
    for (size_t j = 0; j != polygons[i].size(); j++) {
      std::cout << polygons[i][j] << std::endl;
    }
    std::cout << "------ Loop " << i << " END" << std::endl;
  }

  // Calc all arc that bridge 2 edge
  auto arcConnection = CalculateFilletArc(loops, info, radius);

  // Tracing along the arc
  int n = circularSegments > 2 ? circularSegments
                               : Quality::GetCircularSegments(radius);

  auto result = Tracing(loops, arcConnection, n, radius);

  return result;
}

}  // namespace

namespace manifold {

std::vector<CrossSection> CrossSection::Fillet(double radius,
                                               int circularSegments) const {
  auto r = FilletImpl(ToPolygons(), radius, circularSegments);

  std::vector<CrossSection> crossSections;
  for (const auto& ele : r) {
    crossSections.push_back(CrossSection(r));
  }
  return crossSections;
}

std::vector<CrossSection> CrossSection::Fillet(const SimplePolygon pts,
                                               double radius,
                                               int circularSegments) {
  return Fillet(Polygons{pts}, radius, circularSegments);
}

std::vector<CrossSection> CrossSection::Fillet(const Polygons& polygons,
                                               double radius,
                                               int circularSegments) {
  return CrossSection(polygons).Fillet(radius, circularSegments);
}

std::vector<CrossSection> Fillet(const std::vector<CrossSection>& crossSections,
                                 double radius, int circularSegments) {
  return CrossSection::Compose(crossSections).Fillet(radius, circularSegments);
}

}  // namespace manifold
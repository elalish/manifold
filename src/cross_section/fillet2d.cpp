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
#include <tuple>
#include <vector>

#define _USE_MATH_DEFINES
#include <cmath>

#include "../collider.h"
#include "../vec.h"
#include "clipper2/clipper.core.h"
#include "clipper2/clipper.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"

const double EPSILON = 1e-6;

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

enum class EdgeTangentState {
  E1CurrentEdge,
  E1NextEdge,
  E2PreviousEdge,
  E2CurrentEdge,
  E2NextEdge
};

std::string EdgeTangentStateToString(EdgeTangentState e) {
  switch (e) {
    case EdgeTangentState::E1CurrentEdge:
      return "E1CurrentEdge";
    case EdgeTangentState::E1NextEdge:
      return "E1NextEdge";
    case EdgeTangentState::E2PreviousEdge:
      return "E2PreviousEdge";
    case EdgeTangentState::E2CurrentEdge:
      return "E2CurrentEdge";
    case EdgeTangentState::E2NextEdge:
      return "E2NextEdge";
    default:
      return "ERROR";
  }
}

// Two edge tangent by same circle
struct GeomTangentPair {
  std::array<EdgeTangentState, 2> States;
  std::array<double, 2> ParameterValues;
  vec2 CircleCenter;

  // CCW or CW is determined by loop's direction
  std::array<double, 2> RadValues;
};

struct TopoConnectionPair {
  TopoConnectionPair(const GeomTangentPair& geomPair, size_t Edge1Index,
                     size_t Edge1LoopIndex, size_t Edge2Index,
                     size_t Edge2LoopIndex, size_t index)
      : Index(index) {
    CircleCenter = geomPair.CircleCenter;
    ParameterValues = geomPair.ParameterValues;
    RadValues = geomPair.RadValues;

    EdgeIndex = std::array<size_t, 2>{Edge1Index, Edge2Index};
    LoopIndex = std::array<size_t, 2>{Edge1LoopIndex, Edge2LoopIndex};
  }

  size_t Index;

  vec2 CircleCenter;
  std::array<double, 2> ParameterValues;
  std::array<double, 2> RadValues;

  // CCW or CW is determined by loop's direction
  std::array<size_t, 2> EdgeIndex;
  std::array<size_t, 2> LoopIndex;

  TopoConnectionPair Swap() const {
    TopoConnectionPair pair = *this;
    std::swap(pair.ParameterValues[0], pair.ParameterValues[1]);
    std::swap(pair.RadValues[0], pair.RadValues[1]);
    std::swap(pair.EdgeIndex[0], pair.EdgeIndex[1]);
    std::swap(pair.LoopIndex[0], pair.LoopIndex[1]);
    return pair;
  };
};

// For build and query Collider result
struct EdgeOld2New {
  manifold::Box box;
  uint32_t morton;
  size_t p1Ref;
  size_t p2Ref;
  size_t loopRef;
};

struct ColliderInfo {
  Collider MyCollider;
  std::vector<EdgeOld2New> EdgeOld2NewVec;
};

}  // namespace

namespace {
// Utility

vec3 toVec3(vec2 in) { return vec3(in.x, in.y, 0); }

double toRad(const vec2& v) { return atan2(v.y, v.x); }

// Normalize angle to [0, 2*PI)
double normalizeAngle(double angle) {
  while (angle < 0) angle += 2 * M_PI;
  while (angle >= 2 * M_PI) angle -= 2 * M_PI;
  return angle;
}

// Get line normal direction
vec2 getEdgeNormal(bool CCW, const vec2& e) {
  return normalize(CCW ? vec2(-e.y, e.x) : vec2(e.y, -e.x));
};

// Get 2 rad vec by 3 point
std::array<double, 2> getRadPair(const vec2& p1, const vec2& p2,
                                 const vec2& center) {
  return std::array<double, 2>{normalizeAngle(toRad(p1 - center)),
                               normalizeAngle(toRad(p2 - center))};
};

// Get Point by parameter value
vec2 getPointOnEdgeByParameter(const vec2& p1, const vec2& p2, double t) {
  return p1 + t * (p2 - p1);
};

/// @brief Is angle inside CCW sector range
/// @param angle target angle
/// @param startRad CCW sector start rad
/// @param endRad CCW sector end rad
/// @return Is inside
bool isAngleInSector(double angle, double startRad, double endRad) {
  angle = normalizeAngle(angle);
  startRad = normalizeAngle(startRad);
  endRad = normalizeAngle(endRad);

  if (startRad <= endRad + EPSILON) {
    return (angle >= startRad + EPSILON) && (angle <= endRad + EPSILON);
  } else {
    // Sector crosses 0 degrees
    return (angle >= startRad + EPSILON) || (angle <= endRad + EPSILON);
  }
};

// For P-* situation to check local validation
// If circle only touch endpoint of edge, meaning this endpoint should be convex
bool isCircleLocalValid(const std::array<vec2, 3>& points, bool isCCW) {
  return (cross(points[1] - points[0], points[2] - points[1]) >= EPSILON) ^
         isCCW;
}

// For P-* situation to check local validation
// The circle should be the correct side determine by isCCW and negative radius
// FIXME: negative radius situation
bool isCircleLocalValid(const std::array<vec2, 3>& points, bool isCCW,
                        vec2 circleCenter) {
  double rad = toRad(circleCenter - points[1]);

  vec2 pre = getEdgeNormal(!isCCW, points[0] - points[1]),
       next = getEdgeNormal(isCCW, points[2] - points[1]);

  double startRad = normalizeAngle(toRad(pre)),
         endRad = normalizeAngle(toRad(next));

  return isCCW ^ isAngleInSector(rad, startRad, endRad);
};

// Projection point to line and check if it's on the line segment
bool isPointProjectionOnSegment(const vec2& p, const vec2& p1, const vec2& p2,
                                double& t) {
  if (length(p2 - p1) < EPSILON) {
    t = 0;
    return false;
  }

  t = dot(p - p1, p2 - p1) / length2(p2 - p1);

  return (t >= 0.0 - EPSILON) && (t <= 1.0 + EPSILON);
};

/// @brief Distance from a point to a segment
/// @param p Point
/// @param p1 Edge start point
/// @param p2 Edge end point
/// @return length, parameter value of closest point
double distancePointSegment(const vec2& p, const vec2& p1, const vec2& p2) {
  if (length(p2 - p1) < EPSILON) {
    return length(p1 - p);
  }

  vec2 d = p2 - p1;

  double t = dot(p - p1, d) / dot(d, d);

  vec2 closestPoint;

  if (t < 0) {
    closestPoint = p1;
  } else if (t > 1) {
    closestPoint = p2;
  } else {
    closestPoint = p1 + t * d;
  }

  return length(closestPoint - p);
}

// Check if line segment intersect with a circle
bool isIntersectCircleSegment(const vec2& p1, const vec2& p2,
                              const vec2& center, double radius) {
  if (length(p2 - p1) < EPSILON)
    return length2(p1 - center) < (radius + EPSILON) * (radius + EPSILON);

  return distancePointSegment(center, p1, p2) < (radius + EPSILON);
};

std::vector<vec2> discreteArcToPoint(TopoConnectionPair arc, double radius,
                                     int circularSegments) {
  std::vector<vec2> pts;

  double totalRad = normalizeAngle(arc.RadValues[1] - arc.RadValues[0]);

  double dPhi = 2 * M_PI / circularSegments;
  int seg = int(totalRad / dPhi) + 1;
  for (int i = 0; i != seg + 1; ++i) {
    double current = arc.RadValues[0] + dPhi * i;
    if (i == seg) current = arc.RadValues[1];

    vec2 pnt = {arc.CircleCenter.x + radius * cos(current),
                arc.CircleCenter.y + radius * sin(current)};

    pts.push_back(pnt);
  }

  return pts;
}

}  // namespace

namespace {

// Calculate line segment intersections with a circle
std::pair<int, std::array<vec2, 2>> intersectCircleSegment(const vec2& p1,
                                                           const vec2& p2,
                                                           const vec2& center,
                                                           double radius) {
  vec2 d = p2 - p1;
  vec2 f = p1 - center;

  double a = dot(d, d);
  double b = 2 * dot(f, d);
  double c = dot(f, f) - radius * radius;

  double discriminant = b * b - 4 * a * c;
  std::array<vec2, 2> intersections{};

  if (discriminant < 0)
    return std::make_pair(0, intersections);  // No intersection

  double sqrtDisc = sqrt(discriminant);
  double t1 = (-b - sqrtDisc) / (2 * a);
  double t2 = (-b + sqrtDisc) / (2 * a);

  int count = 0;
  if (t1 >= -EPSILON && t1 <= 1 + EPSILON) {
    intersections[count] = p1 + d * t1;
    count++;
  }
  if (t2 >= -EPSILON && t2 <= 1 + EPSILON && std::abs(t2 - t1) > EPSILON) {
    intersections[count] = p1 + d * t2;
    count++;
  }

  return std::make_pair(count, intersections);
};

// Check if line segment intersect with another line segment
std::tuple<bool, double, double> intersectSegmentSegment(const vec2& p1,
                                                         const vec2& p2,
                                                         const vec2& p3,
                                                         const vec2& p4) {
  double t = -1, u = -1;

  double det = cross(p2 - p1, p4 - p3);

  if (std::abs(det) < EPSILON) {
    // Parallel
    return std::make_tuple(false, -1.0, -1.0);
  }

  double num_t = cross(p3 - p1, p4 - p3);
  double num_u = cross(p3 - p1, p2 - p1);

  t = num_t / det;
  u = num_u / det;

  // Check if the intersection point inside line segment
  if ((t >= 0.0 - EPSILON && t <= 1.0 + EPSILON) &&
      (u >= 0.0 - EPSILON && u <= 1.0 + EPSILON)) {
    // Inside
    return std::make_tuple(true, t, u);
  } else {
    return std::make_tuple(false, -1.0, -1.0);
  }
};

std::pair<int, std::array<vec2, 2>> calculatePointPointTangentCircles(
    const vec2& p1, const vec2& p2, double radius) {
  std::array<vec2, 2> circleCenters{};

  double dx = p2.x - p1.x;
  double dy = p2.y - p1.y;

  double distance = sqrt(dx * dx + dy * dy);

  vec2 v = p2 - p1;

  // Find midpoint between the two points
  vec2 midpoint((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);

  // If distance equals 2*radius, there's exactly one solution (midpoint)
  if (std::abs(length(v) - 2 * radius) < 1e-9) {
    circleCenters[0] = midpoint;

    return std::make_pair(1, circleCenters);
  }

  // Calculate distance from midpoint to circle centers
  double h = sqrt(radius * radius - (distance / 2.0) * (distance / 2.0));

  // Calculate unit vector perpendicular to the line connecting p1 and p2
  vec2 d(-dy, dx);
  d /= distance;

  circleCenters[0] = midpoint + h * d;
  circleCenters[1] = midpoint - h * d;

  return std::make_pair(2, circleCenters);
}

// E - * Intersect
std::vector<GeomTangentPair> processPillShapeIntersect(
    const std::array<vec2, 3>& e1Points, const bool e1CCW,
    const std::array<vec2, 4>& e2Points, const bool e2CCW,
    const double radius) {
  const vec2 e1Cur = e1Points[1] - e1Points[0],
             e1Next = e1Points[2] - e1Points[1],
             e2Pre = e2Points[1] - e2Points[0],
             e2Cur = e2Points[2] - e2Points[1],
             e2Next = e2Points[3] - e2Points[2];

  std::vector<GeomTangentPair> GeomTangentPairVec;

  const vec2 e1CurNormal = getEdgeNormal(e1CCW, e1Cur),
             e2CurNormal = getEdgeNormal(e2CCW, e2Cur);
  {
    // Edge - Edge
    // Offset two edge by radius and calculate intersection to get tangent
    // circle center

    const std::array<vec2, 2> offsetE1{
        e1Points[0] + e1CurNormal * radius,
        e1Points[1] + e1CurNormal * radius,
    },
        offsetE2{
            e2Points[1] + e2CurNormal * radius,
            e2Points[2] + e2CurNormal * radius,
        };

    const auto& [isIntersected, t, u] = intersectSegmentSegment(
        offsetE1[0], offsetE1[1], offsetE2[0], offsetE2[1]);

    if (isIntersected) {
      vec2 center = offsetE1[0] + e1Cur * t,
           tangent1 = getPointOnEdgeByParameter(e1Points[0], e1Points[1], t),
           tangent2 = getPointOnEdgeByParameter(e2Points[1], e2Points[2], u);

      GeomTangentPairVec.emplace_back(GeomTangentPair{
          std::array<EdgeTangentState, 2>{EdgeTangentState::E1CurrentEdge,
                                          EdgeTangentState::E2CurrentEdge},
          std::array<double, 2>{t, u}, center,
          getRadPair(tangent1, tangent2, center)});

#ifdef MANIFOLD_DEBUG
      if (ManifoldParams().verbose) {
        std::cout << "E-E" << offsetE1[0] + e1Cur * t << std::endl;
      }
#endif
    }
  }

  {
    const vec2 p1 = e1Points[0], p2 = e1Points[1];
    const vec2 p1Offset = p1 + e1CurNormal * 2.0 * radius,
               p2Offset = p2 + e1CurNormal * 2.0 * radius;

    // Determine full inside or full outside
    auto isInsideRect = [&](const vec2 p) -> bool {
      int sign = e1CCW ? 1 : -1;
      if ((sign * cross(e1Cur, p - p1) >= 0) &&
          (sign * cross(p2Offset - p2, p - p2) >= 0) &&
          (sign * cross(p1Offset - p2Offset, p - p2Offset) >= 0) &&
          (sign * cross(p1 - p1Offset, p - p1Offset) >= 0))
        return true;

      return false;
    };

    auto isInsideEndpointCircles = [&](const vec2 p) -> bool {
      if (length2(p - e1Points[0] + e1CurNormal * radius) <= radius * radius)
        return true;

      if (length2(p - e2Points[0] + e2CurNormal * radius) <= radius * radius)
        return true;

      return false;
    };

    // Edge - Point
    // Offset e1 and test if intersect with circle center at e2's endpoints, and
    // get tangent circle center
    const vec2 e1CurNormal = getEdgeNormal(e1CCW, e1Cur);

    const std::array<vec2, 2> offsetE1{
        e1Points[0] + e1CurNormal * radius,
        e1Points[1] + e1CurNormal * radius,
    };

    for (int i = 0; i != 2; i++) {
      // FIXME: Check if E2's point convex
      if (!isCircleLocalValid(std::array<vec2, 3>{e2Points[i], e2Points[i + 1],
                                                  e2Points[i + 2]},
                              e2CCW))
        continue;

      const vec2 point = e2Points[i + 1];
      if (isInsideRect(point) || isInsideEndpointCircles(point)) {
        if (distancePointSegment(point, p1, p2) < EPSILON) continue;

        const auto& [count, centers] =
            intersectCircleSegment(offsetE1[0], offsetE1[1], point, radius);

        for (int j = 0; j != count; j++) {
          if (isCircleLocalValid(
                  std::array<vec2, 3>{e2Points[i], e2Points[i + 1],
                                      e2Points[i + 2]},
                  e2CCW, centers[j])) {
            double t;
            if (!isPointProjectionOnSegment(centers[j], e1Points[0],
                                            e1Points[1], t)) {
              DEBUG_ASSERT(false, logicErr, "Tangent not on segment");
            }

            auto radVec = getRadPair(
                getPointOnEdgeByParameter(e1Points[0], e1Points[1], t), point,
                centers[j]);

            std::array<double, 2> paramVal{t, 1};

            EdgeTangentState e2EdgeTangentState =
                (i == 0 ? EdgeTangentState::E2PreviousEdge
                        : EdgeTangentState::E2CurrentEdge);

            // CCW Loop must end with CW, or start with CCW
            if (e2CCW ^ (normalizeAngle(radVec[1] - radVec[0]) > M_PI)) {
              e2EdgeTangentState = (i == 0 ? EdgeTangentState::E2CurrentEdge
                                           : EdgeTangentState::E2NextEdge);
              paramVal[1] = 0;
            }

            GeomTangentPairVec.emplace_back(GeomTangentPair{
                std::array<EdgeTangentState, 2>{EdgeTangentState::E1CurrentEdge,
                                                e2EdgeTangentState},
                paramVal, centers[j], radVec});

#ifdef MANIFOLD_DEBUG
            if (ManifoldParams().verbose) {
              std::cout << "E-P" << centers[j] << p1 << p2 << "->" << point
                        << std::endl;

              std::cout << EdgeTangentStateToString(e2EdgeTangentState)
                        << std::endl;
            }
#endif
          }
        }
      }
    }
  }

  return GeomTangentPairVec;
}

// P - * Intersect
std::vector<GeomTangentPair> processPieShapeIntersect(
    const std::array<vec2, 3>& e1Points, const bool e1CCW,
    const std::array<vec2, 4>& e2Points, const bool e2CCW,
    const double radius) {
  if (!isCircleLocalValid(e1Points, e1CCW)) return {};

  if (!isIntersectCircleSegment(e2Points[1], e2Points[2], e1Points[1],
                                2.0 * radius))
    return {};

  std::vector<GeomTangentPair> GeomTangentPairVec;

  const vec2 e1Cur = e1Points[1] - e1Points[0],
             e1Next = e1Points[2] - e1Points[1],
             e2Pre = e2Points[1] - e2Points[0],
             e2Cur = e2Points[2] - e2Points[1],
             e2Next = e2Points[3] - e2Points[2];

  const vec2 e1CurNormal = getEdgeNormal(e1CCW, e1Cur),
             e1NextNormal = getEdgeNormal(e1CCW, e1Next);

  auto isPointInPieArea = [](const vec2& p, const vec2& center, double radius,
                             bool isCCW, double startRad,
                             double endRad) -> bool {
    vec2 diff = p - center;
    double distSq = length2(diff);

    if (distSq > (radius * radius + EPSILON)) return false;

    double angle = toRad(diff);
    return isCCW ^ isAngleInSector(angle, startRad, endRad);
  };

  {
    // Point - Edge
    // Offset e2 and test if intersect with circle center at e1's endpoints, and
    // get tangent circle center

    const vec2 e2CurNormal = getEdgeNormal(e2CCW, e2Cur);

    const std::array<vec2, 2> offsetE2{
        e2Points[1] + e2CurNormal * radius,
        e2Points[2] + e2CurNormal * radius,
    };

    if (length(e2Points[1] - vec2(5, -1)) < EPSILON &&
        length(e2Points[2] - vec2(2.6, 0)) < EPSILON) {
      int i = 0;
    }

    const auto& [count, centers] =
        intersectCircleSegment(offsetE2[0], offsetE2[1], e1Points[1], radius);

    for (int i = 0; i != count; i++) {
      if (isCircleLocalValid(e1Points, e1CCW, centers[i])) {
        double t;
        if (!isPointProjectionOnSegment(centers[i], e2Points[1], e2Points[2],
                                        t)) {
          DEBUG_ASSERT(false, logicErr, "Tangent not on segment");
        }

        auto radVec = getRadPair(
            e1Points[1], getPointOnEdgeByParameter(e2Points[1], e2Points[2], t),
            centers[i]);

        EdgeTangentState e1EdgeTangentState = EdgeTangentState::E1CurrentEdge;
        std::array<double, 2> paramVal{1, t};

        // CCW Loop must end with CW, or start with CCW
        if (e1CCW ^ !(normalizeAngle(radVec[1] - radVec[0]) > M_PI)) {
          e1EdgeTangentState = EdgeTangentState::E1NextEdge;
          paramVal[0] = 0;
        }

        GeomTangentPairVec.emplace_back(GeomTangentPair{
            std::array<EdgeTangentState, 2>{e1EdgeTangentState,
                                            EdgeTangentState::E2CurrentEdge},
            paramVal, centers[i], radVec});

#ifdef MANIFOLD_DEBUG
        if (ManifoldParams().verbose) {
          std::cout << "P-E" << centers[i] << "\t" << e1Points[1] << "->"
                    << offsetE2[0] << offsetE2[1] << std::endl;

          std::cout << EdgeTangentStateToString(e1EdgeTangentState)
                    << std::endl;
        }
#endif
      }
    }
  }

  // Point - Point
  // Get two circles which passed both point, and check if they valid.

  double startRad = toRad(e1CurNormal), endRad = toRad(e1NextNormal);

  for (int i = 0; i != 2; i++) {
    if (!isCircleLocalValid(
            std::array<vec2, 3>{e2Points[i], e2Points[i + 1], e2Points[i + 2]},
            e2CCW))
      continue;

    const vec2 point = e2Points[i + 1];
    if (length(point - e1Points[1]) < EPSILON) continue;

    if (isPointInPieArea(point, e1Points[1], 2.0 * radius, e1CCW, startRad,
                         endRad)) {
      const auto& [count, centers] =
          calculatePointPointTangentCircles(e1Points[1], point, radius);

      for (int j = 0; j != count; j++) {
        if (isCircleLocalValid(std::array<vec2, 3>{e2Points[i], e2Points[i + 1],
                                                   e2Points[i + 2]},
                               e2CCW, centers[j])) {
          std::array<double, 2> paramVal{1, 1};
          auto radVec = getRadPair(e1Points[1], point, centers[j]);

          EdgeTangentState e1EdgeTangentState = EdgeTangentState::E1CurrentEdge;

          // CCW Loop must end with CW, or start with CCW

          bool arcCCW = normalizeAngle(radVec[1] - radVec[0]) > M_PI;
          if (e1CCW ^ !arcCCW) {
            e1EdgeTangentState = EdgeTangentState::E1NextEdge;
            paramVal[0] = 0;
          }

          EdgeTangentState e2EdgeTangentState =
              (i == 0 ? EdgeTangentState::E2PreviousEdge
                      : EdgeTangentState::E2CurrentEdge);

          // CCW Loop must end with CW, or start with CCW
          if (e2CCW ^ arcCCW) {
            e2EdgeTangentState = (i == 0 ? EdgeTangentState::E2CurrentEdge
                                         : EdgeTangentState::E2NextEdge);
            paramVal[1] = 0;
          }

          GeomTangentPairVec.emplace_back(
              GeomTangentPair{std::array<EdgeTangentState, 2>{
                                  e1EdgeTangentState, e2EdgeTangentState},
                              paramVal, centers[j], radVec});
#ifdef MANIFOLD_DEBUG
          if (ManifoldParams().verbose) {
            std::cout << "P-P" << centers[j] << "\t" << point << "->"
                      << e1Points[1] << std::endl;
            std::cout << EdgeTangentStateToString(e1EdgeTangentState) << "->"
                      << EdgeTangentStateToString(e2EdgeTangentState)
                      << std::endl;
          }
#endif
        }
      }
    }
  }

  return GeomTangentPairVec;
}

}  // namespace

// Sub process
namespace {
using namespace manifold;

ColliderInfo BuildCollider(const manifold::Polygons& polygons) {
  Vec<manifold::Box> boxVec;
  Vec<uint32_t> mortonVec;

  std::vector<EdgeOld2New> edgeOld2NewVec;

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
                   [](const EdgeOld2New& lhs, const EdgeOld2New& rhs) -> bool {
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

  return ColliderInfo{Collider(boxVec, mortonVec), edgeOld2NewVec};
}

std::vector<std::vector<TopoConnectionPair>> CalculateFilletArc(
    const Loops& loops, const ColliderInfo& colliderInfo, double radius) {
  bool invert = false;
  if (radius < EPSILON) invert = true;

  size_t circleIndex = 0;

  std::vector<size_t> loopOffset(loops.size());

  const size_t loopElementCount = ([&loopOffset, &loops]() -> size_t {
    size_t count = 0;
    for (size_t i = 0; i != loops.size(); i++) {
      loopOffset[i] = count;
      count += loops[i].Loop.size();
    }

    return count;
  })();

  // E-E 1<<0
  // P-E 1<<1
  // P-P 1<<2
  std::vector<uint8_t> processedMark(loopElementCount * loopElementCount, 0);

  auto getMarkPosition = [&loopElementCount, &loopOffset](
                             const size_t loop1i, const size_t ele1i,
                             const size_t loop2i,
                             const size_t ele2i) -> size_t {
    return (loopOffset[loop1i] + ele1i) * loopElementCount +
           loopOffset[loop2i] + ele2i;
  };

  std::vector<std::vector<TopoConnectionPair>> arcConnection(
      loopElementCount, std::vector<TopoConnectionPair>());

  std::vector<std::vector<GeomTangentPair>> arcInfoVec(
      loopElementCount, std::vector<GeomTangentPair>());

  std::ofstream f("circle.txt");
  f << radius << std::endl;

  for (size_t e1Loopi = 0; e1Loopi != loops.size(); e1Loopi++) {
    const auto& e1Loop = loops[e1Loopi].Loop;
    const bool isE1LoopCCW = loops[e1Loopi].isCCW ^ invert;

    // create BBox for every line to find Collision
    for (size_t e1i = 0; e1i != e1Loop.size(); e1i++) {
      // Outer loop is CCW, p1 p2 -> current edge start end
      const size_t p1i = e1i, p2i = (e1i + 1) % e1Loop.size();
      const std::array<vec2, 3> e1Points{e1Loop[e1i],
                                         e1Loop[(e1i + 1) % e1Loop.size()],
                                         e1Loop[(e1i + 2) % e1Loop.size()]};
      const vec2 e1 = e1Points[1] - e1Points[0];
      const bool p2IsConvex = cross(e1, e1Points[2] - e1Points[1]) >= EPSILON;

      const vec2 normal = normalize(vec2(-e1.y, e1.x));

      // Create BBox
      manifold::Box box(toVec3(e1Points[0]), toVec3(e1Points[1]));
      {
        // See
        // https://docs.google.com/presentation/d/1P-3oxmjmEw_Av0rq7q7symoL5VB5DSWpRvqoa3LK7Pg/edit?usp=sharing

        vec2 normalOffsetP1 = e1Points[0] + normal * 2.0 * radius,
             normalOffsetP2 = e1Points[1] + normal * 2.0 * radius;

        box.Union(toVec3(normalOffsetP1));
        box.Union(toVec3(normalOffsetP2));

        const vec2 e1n = normalize(e1);
        box.Union(toVec3(e1Points[0] - e1n * radius + normal * radius));
        box.Union(toVec3(e1Points[1] + e1n * radius + normal * radius));

        if (!p2IsConvex) {
          const vec2 pnext = e1Loop[(p2i + 1) % e1Loop.size()],
                     enext = pnext - e1Points[1], enextn = normalize(enext),
                     normalnext = normalize(vec2(-enext.y, enext.x));

          box.Union(toVec3(e1Points[1] + normalnext * 2.0 * radius));
          box.Union(
              toVec3(e1Points[1] + enextn * radius + normalnext * radius));
        }
      }

      std::vector<EdgeOld2New> r;
      auto recordCollision = [&](int, int edge) {
        r.push_back(colliderInfo.EdgeOld2NewVec[edge]);
      };
      auto recorder = MakeSimpleRecorder(recordCollision);

      colliderInfo.MyCollider.Collisions(
          manifold::Vec<manifold::Box>({box}).cview(), recorder);

#ifdef MANIFOLD_DEBUG
      if (ManifoldParams().verbose) {
        std::cout << std::endl
                  << "Now " << p1i << "->" << (e1i + 1) % e1Loop.size() << " "
                  << r.size() << std::endl;
      }
#endif

      // In Out Classify
      for (auto it = r.begin(); it != r.end(); it++) {
        const auto& ele = *it;

        // e2i is the index of detected possible bridge edge

        const size_t e2i = ele.p1Ref, e2Loopi = ele.loopRef;

        const auto& e2Loop = loops[e2Loopi].Loop;
        const size_t p3i = e2i, p4i = (e2i + 1) % e2Loop.size();
        const bool isE2LoopCCW = loops[e2Loopi].isCCW ^ invert;

        const std::array<vec2, 4> e2Points{
            e2Loop[(e2i + e2Loop.size() - 1) % e2Loop.size()], e2Loop[e2i],
            e2Loop[(e2i + 1) % e2Loop.size()],
            e2Loop[(e2i + 2) % e2Loop.size()]};

        // Skip self and pre one, only process forward
        if (e1Loopi == e2Loopi &&
            ((e2i == e1i) ||
             (e2i == (e1i + e1Loop.size() - 1) % e1Loop.size())))
          continue;

#ifdef MANIFOLD_DEBUG
        if (ManifoldParams().verbose) {
          std::cout << "-----------" << std::endl
                    << e2Points[1] << " " << e2Points[2] << std::endl;
        }
#endif

        if (e1i == 0 && e2i == 5) {
          int i = 0;
        }

        const uint8_t EEMASK = 1, PEMASK = 1 << 1, PPMASK = 1 << 2;

        std::vector<GeomTangentPair> r1;
        // Check if processed, and add duplicate mark
        processedMark[getMarkPosition(e1Loopi, e1i, e2Loopi, e2i)] |= EEMASK;
        processedMark[getMarkPosition(e1Loopi, e1i, e2Loopi, p3i)] |= PEMASK;
        processedMark[getMarkPosition(e1Loopi, e1i, e2Loopi, p4i)] |= PEMASK;

        if (((processedMark[getMarkPosition(e2Loopi, e2i, e1Loopi, e1i)] &
              EEMASK) == 0) ||
            ((processedMark[getMarkPosition(e2Loopi, p3i, e1Loopi, e1i)] &
              PEMASK) == 0) ||
            ((processedMark[getMarkPosition(e2Loopi, p4i, e1Loopi, e1i)] &
              PEMASK) == 0)) {
          r1 = processPillShapeIntersect(e1Points, isE1LoopCCW, e2Points,
                                         isE2LoopCCW, radius);

          processedMark[getMarkPosition(e2Loopi, e2i, e1Loopi, e1i)] |= EEMASK;
          processedMark[getMarkPosition(e2Loopi, p3i, e1Loopi, e1i)] |= PEMASK;
          processedMark[getMarkPosition(e2Loopi, p4i, e1Loopi, e1i)] |= PEMASK;
        }

        processedMark[getMarkPosition(e1Loopi, p2i, e2Loopi, e2i)] |= PEMASK;
        processedMark[getMarkPosition(e1Loopi, p2i, e2Loopi, p3i)] |= PPMASK;
        processedMark[getMarkPosition(e1Loopi, p2i, e2Loopi, p4i)] |= PPMASK;

        if (((processedMark[getMarkPosition(e2Loopi, e2i, e1Loopi, p2i)] &
              PEMASK) == 0) ||
            ((processedMark[getMarkPosition(e2Loopi, p3i, e1Loopi, p2i)] &
              PPMASK) == 0) ||
            ((processedMark[getMarkPosition(e2Loopi, p4i, e1Loopi, p2i)] &
              PPMASK) == 0)) {
          if (e1Loopi != e2Loopi ||
              (e2i != (e1i + e1Loop.size() - 1) % e1Loop.size())) {
            // P-* won't happened if e1 e2 is neighbour
            // Only check forward neighbour because pre neighbour have been
            // skipped

            auto r2 = processPieShapeIntersect(e1Points, isE1LoopCCW, e2Points,
                                               isE2LoopCCW, radius);

            r1.insert(r1.end(), r2.begin(), r2.end());

            processedMark[getMarkPosition(e2Loopi, e2i, e1Loopi, p2i)] |=
                PEMASK;
            processedMark[getMarkPosition(e2Loopi, p3i, e1Loopi, p2i)] |=
                PPMASK;
            processedMark[getMarkPosition(e2Loopi, p4i, e1Loopi, p2i)] |=
                PPMASK;
          }
        }

        std::vector<size_t> intersectEdgeIndex;

        // Check each potential circle center
        for (auto it = r1.begin(); it != r1.end();) {
          const auto& arc = *it;

          manifold::Box box(toVec3(arc.CircleCenter - vec2(1, 0) * radius),
                            toVec3(arc.CircleCenter + vec2(1, 0) * radius));

          box.Union(toVec3(arc.CircleCenter - vec2(0, 1) * radius));
          box.Union(toVec3(arc.CircleCenter + vec2(0, 1) * radius));

          std::vector<EdgeOld2New> rr;
          auto recordCollision = [&](int, int edge) {
            rr.push_back(colliderInfo.EdgeOld2NewVec[edge]);
          };
          auto recorder = MakeSimpleRecorder(recordCollision);

          colliderInfo.MyCollider.Collisions(
              manifold::Vec<manifold::Box>({box}).cview(), recorder);

          bool eraseFlag = false;

          for (auto it = rr.begin(); it != rr.end(); it++) {
            const auto& edge = *it;

            if (edge.loopRef == e1Loopi && edge.p1Ref == e1i)
              continue;
            else if (edge.loopRef == e2Loopi && edge.p1Ref == e2i)
              continue;

            const auto& eLoop = loops[edge.loopRef].Loop;
            const size_t p1i = edge.p1Ref, p2i = edge.p2Ref;
            const vec2 p1 = eLoop[p1i], p2 = eLoop[p2i];

            double distance = distancePointSegment(arc.CircleCenter, p1, p2);

            if (std::abs(distance - radius) < EPSILON) {
              // TODO: Intersected, this can be used to optimize speed for
              // pre-detect all intersected point and avoid double processed
            } else if (distance < radius) {
#ifdef MANIFOLD_DEBUG
              if (ManifoldParams().verbose) {
                std::cout << "Remove" << arc.CircleCenter << std::endl;
              }
#endif
              eraseFlag = true;
              break;
            }
          }

          if (eraseFlag)
            it = r1.erase(it);
          else
            it++;
        }

        size_t e1Nexti = (e1i + 1) % e1Loop.size(),
               e2Prei = (e2i + e2Loop.size() - 1) % e2Loop.size(),
               e2Nexti = (e2i + 1) % e2Loop.size();

        for (auto it = r1.begin(); it != r1.end(); it++) {
          size_t i, j;

          for (auto k = 0; k != 2; k++) {
            switch (it->States[k]) {
              case EdgeTangentState::E1CurrentEdge:
                i = e1i;
                break;
              case EdgeTangentState::E1NextEdge:
                i = e1Nexti;
                break;
              case EdgeTangentState::E2PreviousEdge:
                j = e2Prei;
                break;
              case EdgeTangentState::E2CurrentEdge:
                j = e2i;
                break;
              case EdgeTangentState::E2NextEdge:
                j = e2Nexti;
                break;
            }
          }

          auto connectPair =
              TopoConnectionPair(*it, i, e1Loopi, j, e2Loopi, circleIndex++);

          bool arcCCW =
              normalizeAngle(it->RadValues[1] - it->RadValues[0]) < M_PI;

          size_t index = 0;
          TopoConnectionPair pair = connectPair;

          if (arcCCW == isE1LoopCCW) {
            index = loopOffset[e1Loopi] + i;
          } else {
            index = loopOffset[e2Loopi] + j;
            pair = pair.Swap();
          }
          auto compare = [](const std::array<double, 2>& arr1,
                            const std::array<double, 2>& arr2) -> bool {
            return std::abs(arr1[0] - arr2[0]) < EPSILON &&
                   std::abs(arr1[1] - arr2[1]) < EPSILON;
          };
          auto itt = std::find_if(
              arcConnection[index].begin(), arcConnection[index].end(),
              [pair, compare](const TopoConnectionPair& ele) -> bool {
                return ele.EdgeIndex == pair.EdgeIndex &&
                       ele.LoopIndex == pair.LoopIndex &&
                       ele.CircleCenter == pair.CircleCenter &&
                       compare(ele.RadValues, pair.RadValues) &&
                       compare(ele.ParameterValues, pair.ParameterValues);
              });

          if (itt == arcConnection[index].end()) {
            arcConnection[index].push_back(pair);
          }
        }

        for (const auto& e : r1) {
          f << e.CircleCenter.x << " " << e.CircleCenter.y << std::endl;
        }
      }
    }
  }

  f.close();

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose) {
    for (size_t i = 0; i != arcConnection.size(); i++) {
      std::cout << i << " " << arcConnection[i].size();
      for (size_t j = 0; j != arcConnection[i].size(); j++) {
        std::cout << "\t Index:" << arcConnection[i][j].Index << " ["
                  << arcConnection[i][j].LoopIndex[0] << ", "
                  << arcConnection[i][j].EdgeIndex[0] << "] ["
                  << arcConnection[i][j].LoopIndex[1] << ", "
                  << arcConnection[i][j].EdgeIndex[1] << "] \t Param # "
                  << arcConnection[i][j].ParameterValues[0] << ", "
                  << arcConnection[i][j].ParameterValues[1]
                  << " # \t { Center: " << arcConnection[i][j].CircleCenter
                  << " } \t Rad < " << arcConnection[i][j].RadValues[0] << ", "
                  << arcConnection[i][j].RadValues[1] << " >" << std::endl;
      }

      std::cout << std::endl;
    }
  }
#endif

  return arcConnection;
}

std::vector<CrossSection> Tracing(
    const Loops& loops,
    std::vector<std::vector<TopoConnectionPair>> arcConnection,
    int circularSegments, double radius) {
  std::vector<size_t> loopOffset(loops.size());

  const size_t loopElementCount = ([&loopOffset, &loops]() -> size_t {
    size_t count = 0;
    for (size_t i = 0; i != loops.size(); i++) {
      loopOffset[i] = count;
      count += loops[i].Loop.size();
    }

    return count;
  })();

  struct EdgeLoopPair {
    size_t EdgeIndex, LoopIndex;
    double ParameterValue;

    bool operator==(const EdgeLoopPair& o) {
      return (EdgeIndex == o.EdgeIndex) && (LoopIndex == o.LoopIndex);
    }

    bool operator!=(const EdgeLoopPair& o) { return !(*this == o); }
  };

  auto getEdgePosition = [&loopOffset](const EdgeLoopPair& edge) -> size_t {
    return loopOffset[edge.LoopIndex] + edge.EdgeIndex;
  };

  std::vector<uint8_t> loopFlag(loops.size(), 0);

  manifold::Polygons resultLoops;

  while (true) {
    SimplePolygon tracingLoop{};

    // Tracing to construct result
    EdgeLoopPair current, end;

    // Find first fillet arc to start
    auto it = arcConnection.begin();
    for (; it != arcConnection.end(); it++) {
      if (!it->empty()) {
        TopoConnectionPair arc = *it->begin();

        const auto pts = discreteArcToPoint(arc, radius, circularSegments);
        tracingLoop.insert(tracingLoop.end(), pts.begin(), pts.end());

        current = EdgeLoopPair{arc.EdgeIndex[1], arc.LoopIndex[1],
                               arc.ParameterValues[1]};
        end = EdgeLoopPair{arc.EdgeIndex[0], arc.LoopIndex[0],
                           arc.ParameterValues[0]};

        loopFlag[arc.LoopIndex[0]] = 1;
        loopFlag[arc.LoopIndex[1]] = 1;

        it->erase(it->begin());

        break;
      }
    }

    if (it == arcConnection.end()) break;

    while (true) {
      // Trace to find next arc on current edge

      const double EPSILON = 1e-6;

      auto it = std::find_if(
          arcConnection[getEdgePosition(current)].begin(),
          arcConnection[getEdgePosition(current)].end(),
          [current, EPSILON](const TopoConnectionPair& ele) -> bool {
            return ele.ParameterValues[0] + EPSILON > current.ParameterValue;
          });

      if (it == arcConnection[getEdgePosition(current)].end()) {
        // Not found, just add vertex
        // FIXME: shouldn't add vertex directly, should search for next edge
        // with fillet arc

        if (current == end) {
          break;
        }

        const auto& loop = loops[current.LoopIndex].Loop;
        tracingLoop.push_back(loop[(current.EdgeIndex + 1) % loop.size()]);

        current.EdgeIndex = (current.EdgeIndex + 1) % loop.size();
        current.ParameterValue = 0;

      } else {
        // Found next circle fillet

        if (current == end && it->ParameterValues[0] > end.ParameterValue) {
          break;
        }

        TopoConnectionPair arc = *it;
        arcConnection[getEdgePosition(current)].erase(it);

        const auto pts = discreteArcToPoint(arc, radius, circularSegments);
        tracingLoop.insert(tracingLoop.end(), pts.begin(), pts.end());

        current.EdgeIndex = arc.EdgeIndex[1];
        current.LoopIndex = arc.LoopIndex[1];
        current.ParameterValue = arc.ParameterValues[1];

        loopFlag[arc.LoopIndex[1]] = 1;
      }
    }
    resultLoops.push_back(tracingLoop);
  }

  CrossSection hole;
  bool invert = false;
  if (radius < EPSILON) invert = true;

  for (size_t i = 0; i != loops.size(); i++) {
    if (loopFlag[i] == 0 && (invert ^ (!loops[i].isCCW)))
      hole = hole.Boolean(CrossSection(Polygons{loops[i].Loop}),
                          manifold::OpType::Add);
  }

  std::vector<CrossSection> result;
  for (auto it = resultLoops.begin(); it != resultLoops.end(); it++) {
    result.push_back(
        CrossSection(Polygons{*it}).Boolean(hole, manifold::OpType::Subtract));
  }
#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose) {
    std::cout << "Result loop count:" << resultLoops.size() << std::endl;
  }
#endif

  return result;
}

void SavePolygons(const std::string& filename, const Polygons& polygons) {
  // Open a file stream for writing.
  std::ofstream outFile(filename);

  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }

  outFile << filename << " " << 1 << "\n";
  outFile << polygons.size() << "\n";

  for (const auto& loop : polygons) {
    outFile << loop.size() << "\n";
    for (const auto& point : loop) {
      outFile << point.x << " " << point.y << "\n";
    }
  }

  outFile.close();

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose) {
    std::cout << "Successfully saved " << polygons.size() << " tests to "
              << filename << std::endl;
  }
#endif
}

std::vector<CrossSection> FilletImpl(const Polygons& polygons, double radius,
                                     int circularSegments) {
  if (polygons.empty()) return {};

  ColliderInfo colliderInfo = BuildCollider(polygons);

  SavePolygons("input.txt", polygons);

  Loops loops;
  loops.reserve(polygons.size());

  for (const auto& loop : polygons) {
    C2::PathD path = pathd_of_contour(loop);

    loops.push_back(SimpleLoop{loop, C2::Area(path) > EPSILON});
  }

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose) {
    for (size_t i = 0; i != polygons.size(); i++) {
      std::cout << "------ Loop " << i << " START" << std::endl;
      for (size_t j = 0; j != polygons[i].size(); j++) {
        std::cout << polygons[i][j] << std::endl;
      }
      std::cout << "------ Loop " << i << " END" << std::endl;
    }
  }
#endif

  // Calc all arc that bridge 2 edge
  auto arcConnection = CalculateFilletArc(loops, colliderInfo, radius);

  int n = circularSegments > 2 ? circularSegments
                               : Quality::GetCircularSegments(radius);

  // Tracing along the arc
  auto result = Tracing(loops, arcConnection, n, radius);

  return result;
}

}  // namespace

namespace manifold {

std::vector<CrossSection> CrossSection::Fillet(double radius,
                                               int circularSegments) const {
  return FilletImpl(ToPolygons(), radius, circularSegments);
}

}  // namespace manifold
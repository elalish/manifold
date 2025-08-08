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

#define _USE_MATH_DEFINES
#include <cmath>

#include "../collider.h"
#include "clipper2/clipper.core.h"
#include "clipper2/clipper.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"

const double EPSILON = 1e-9;
namespace C2 = Clipper2Lib;

namespace {
// Utility

using namespace manifold;

namespace C2 = Clipper2Lib;
vec2 v2_of_pd(const C2::PointD p) { return {p.x, p.y}; }

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
  return distancePointSegment(center, p1, p2, t) <= radius;
};

// Check if line segment intersect with another line segment
bool intersectSegmentSegment(const vec2& p1, const vec2& p2, const vec2& p3,
                             const vec2& p4) {
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
    return true;
  } else {
    // Outside -> No intersection
    return false;
  }
};

double distanceSegmentSegment(const vec2& p1, const vec2& p2, const vec2& p3,
                              const vec2& p4, bool& isEndpoint) {}

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
  if (e1CCW && e2CCW) throw std::exception();

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
      bool sign = la::cross(e1, e2) > 0;

      if (sign == (e1CCW ^ e2CCW)) {
        // TODO: Ensure degenerate processed
        return IntersectStadiumResult::P1Degenerate;
      } else {
        return IntersectStadiumResult::EdgeEdgeIntersect;
      }
    }

    if (p2Intersect) {
      bool sign = la::cross(e1, e2) < 0;

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
    if (intersectSegmentSegment(p1, p2, p3, p4))
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
    float t1 = (-b - sqrtDisc) / (2 * a);
    float t2 = (-b + sqrtDisc) / (2 * a);

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
    return 0 <= la::dot(p - p1, p2 - p1) <= la::length2(p2 - p1);
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

    if (startRad < 0) startRad += 2 * M_PI;
    if (endRad < 0) endRad += 2 * M_PI;

    return true;
  }

  return false;
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
    const SimplePolygon& outerLoop, const ColliderInfo& collider,
    double radius) {
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
    const vec2 e1 = p2 - p1;
    const bool p2IsConvex =
        la::cross(e1, outerLoop[(p2i + 1) % outerLoop.size()] - p2) >= EPSILON;

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
        const vec2 pnext = outerLoop[(p2i + 1) % outerLoop.size()],
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
              << "Now " << p1i << "->" << (e1i + 1) % outerLoop.size()
              << std::endl;

    // In Out Classify
    for (size_t j = 0; j != r.size(); j++) {
      auto ele = collider.outerEdgeOld2NewVec[r.Get(j, true)];

      // e2i is the index of detected possible bridge edge
      size_t e2i = ele.p1Ref;

      // Skip self and pre one, only process forward
      if ((e1i == e2i) ||
          e2i == (e1i + outerLoop.size() - 1) % outerLoop.size())
        continue;

      // Check if processed, and add duplicate mark
      markEE[e1i * outerLoop.size() + e2i] = 1;
      if (markEE[e2i * outerLoop.size() + e1i] != 0) {
        std::cout << "Skipped" << std::endl;
        continue;
      }

      // CCW, p3 p4 -> bbox hit edge start end
      size_t p3i = e2i, p4i = ele.p2Ref;
      vec2 p3 = outerLoop[p3i], p4 = outerLoop[p4i];

      bool continueFlag = false;
      bool degenerateFlag = false;

      switch (intersectStadiumCollider(p1, p2, true, p3, p4, false, radius)) {
        case IntersectStadiumResult::EdgeEdgeIntersect: {
          double startRad = 0, endRad = 0;
          double e1T = 0, e2T = 0;
          vec2 circleCenter(0, 0);

          if (calculateSegmentSegmentCircleCenter(p1, p2, p3, p4, radius, e1T,
                                                  e2T, circleCenter, startRad,
                                                  endRad)) {
            // Sort result by CCW
            double arcAngle = endRad - startRad;
            if (arcAngle < 0) arcAngle += 2 * M_PI;

            if (arcAngle <= M_PI) {
              arcConnection[e1i].emplace_back(ArcConnectionInfo{
                  circleCenter, e1T, e2T, e1i, e2i, startRad, endRad});
            } else {
              arcConnection[e2i].emplace_back(ArcConnectionInfo{
                  circleCenter, e2T, e1T, e2i, e1i, endRad, startRad});
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

      if (p2IsConvex ||
          (!degenerateFlag && !intersectSectorCollider(p1, p2, vec2(), true, p3,
                                                       p4, false, radius))) {
        continue;
      }

      // Handle concave vertex degenerate case

      std::vector<PointSegmentIntersectResult> r(2,
                                                 PointSegmentIntersectResult{});

      const vec2 pnext = outerLoop[(p2i + 1) % outerLoop.size()],
                 enext = pnext - p2, enextn = la::normalize(enext),
                 normalnext = la::normalize(vec2(-enext.y, enext.x));
      switch (calculatePointSegmentCircleCenter(p1, p2, pnext, true, p3, p4,
                                                false, radius, r[0], r[1])) {
        case PointSegmentResult::PreviousEdgeIntersect:
          // Add to previous edge
          break;
        case PointSegmentResult::NextEdgeIntersect:
          // Add to next edge
          break;
        case PointSegmentResult::BothIntersect:
          break;
        case PointSegmentResult::Ignore:
        default:
          break;
      }
    }
  }

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

Polygons FilletImpl(const SimplePolygon& loop, double radius,
                    int circularSegments) {
  using namespace manifold;

  ColliderInfo info{};
  info.outerCollider = BuildCollider(loop, info.outerEdgeOld2NewVec);

  // // Process inner loops
  // info.innerCollider = std::vector<Collider>(polygons.size() - 1,
  // Collider()); info.innerVec = std::vector<std::vector<edgeOld2New>>(
  //     polygons.size() - 1, std::vector<edgeOld2New>());
  // for (size_t i = 1; i != polygons.size(); i++) {
  //   info.innerCollider[i] = BuildCollider(polygons[i], info.innerVec[i]);
  // }

  // Calc all arc that bridge 2 edge
  auto arcConnection = CalculateFilletArc(loop, info, radius);

  // Tracing along the arc
  int n = circularSegments > 2 ? circularSegments
                               : Quality::GetCircularSegments(radius);

  auto result = Tracing(loop, arcConnection, n, radius);

  return result;
}

}  // namespace

namespace manifold {

struct PathImpl {
  PathImpl(const C2::PathsD paths_) : paths_(paths_) {}
  operator const C2::PathsD&() const { return paths_; }
  const C2::PathsD paths_;
};

std::vector<CrossSection> CrossSection::Fillet(double radius,
                                               int circularSegments) const {
  auto paths = this->GetPaths()->paths_;

  Polygons outer, inner;
  for (const auto& loop : paths) {
    SimplePolygon polygon;
    polygon.reserve(loop.size());

    for (auto p : loop) {
      polygon.push_back(v2_of_pd(p));
    }

    if (C2::Area(loop) > EPSILON)
      outer.push_back(polygon);
    else
      inner.push_back(polygon);
  }

  std::vector<CrossSection> result;

  for (const auto& loop : outer) {
    auto r = FilletImpl(loop, radius, circularSegments);

    for (const auto& ele : r) {
      Polygons poly{ele};
      poly.insert(poly.end(), inner.begin(), inner.end());
      result.push_back(poly);
    }
  }

  return result;
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
  std::vector<CrossSection> result;

  for (const auto& crossSection : crossSections) {
    auto r = crossSection.Fillet(radius, circularSegments);
    result.insert(result.end(), r.begin(), r.end());
  }

  return result;
}

}  // namespace manifold
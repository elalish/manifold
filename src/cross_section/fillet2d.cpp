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

const double EPSILON = 1e-12;

size_t caseIndex = 0;
std::ofstream resultOutputFile;

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
  while (angle < 0) angle += 2.0 * M_PI;
  while (angle >= 2.0 * M_PI) angle -= 2.0 * M_PI;
  return angle;
}

// Get line normal direction
vec2 getEdgeNormal(const vec2& e, bool invert) {
  return normalize(!invert ? vec2(-e.y, e.x) : vec2(e.y, -e.x));
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

std::vector<vec2> discreteArcToPoint(TopoConnectionPair arc, double radius,
                                     int circularSegments) {
  std::vector<vec2> pts;

  double totalRad = normalizeAngle(arc.RadValues[1] - arc.RadValues[0]);

  double dPhi = 2.0 * M_PI / circularSegments;
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
const double MIN_LEN_SQ = 1e-12;

bool isConvex(const std::array<vec2, 3>& e) {
  float val = la::cross(e[1] - e[0], e[2] - e[1]);

  return val > EPSILON;
}

bool isVectorInSector(vec2 v, vec2 start, vec2 end, bool CCW) {
  double crossStartEnd = cross(start, end);
  double crossStartV = cross(start, v);
  double crossVEnd = cross(v, end);

  if (length2(v - start) < EPSILON || length2(v - end) < EPSILON) return true;

  if (CCW) {
    if (crossStartEnd >= -EPSILON) {
      return crossStartV >= -EPSILON && crossVEnd >= -EPSILON;
    } else {
      return crossStartV >= -EPSILON || crossVEnd >= -EPSILON;
    }
  } else {
    return isVectorInSector(v, end, start, true);
  }
}

struct IntersectResult {
  int Count = 0;
  std::array<vec2, 2> Points{vec2(), vec2()};
};

IntersectResult intersectSegments(vec2 p1, vec2 p2, vec2 p3, vec2 p4) {
  vec2 r = p2 - p1;
  vec2 s = p4 - p3;
  double rxs = cross(r, s);

  if (std::abs(rxs) < EPSILON) return IntersectResult{0, {}};

  vec2 qp = p3 - p1;
  double t = cross(qp, s) / rxs;
  double u = cross(qp, r) / rxs;

  if (t >= -EPSILON && t <= 1.0 + EPSILON && u >= -EPSILON &&
      u <= 1.0 + EPSILON) {
    return IntersectResult{1, {p1 + r * t, vec2()}};
  }

  return IntersectResult{0, {}};
}

/// @brief Calculate intersection point of segment and arc
/// @param p1 Start Point of Segment
/// @param p2 End Point of Segment
/// @param c Center of Arc
/// @param r Radius of Arc
/// @param arcStart Start Direction of Arc
/// @param arcEnd End Direction of Arc
/// @param isArcCCW Is Start to End CCW?
/// @return
IntersectResult intersectSegmentArc(vec2 p1, vec2 p2, vec2 c, double r,
                                    vec2 arcStart, vec2 arcEnd, bool isArcCCW) {
  vec2 v = p2 - p1;
  double lenSq = dot(v, v);

  if (lenSq < MIN_LEN_SQ) return IntersectResult{0, {}};

  double t = dot(c - p1, v) / lenSq;

  vec2 p = p1 + v * t;

  vec2 distVec = c - p;
  double distSq = dot(distVec, distVec);

  IntersectResult result;
  result.Count = 0;

  if (distSq > r * r + EPSILON) {
    return result;
  }

  double offset = std::sqrt(std::max(0.0, r * r - distSq));
  double tOffset = offset / std::sqrt(lenSq);

  double t1 = t - tOffset;
  double t2 = t + tOffset;

  auto checkAdd = [&](double t) {
    if (t >= -EPSILON && t <= 1.0 + EPSILON) {
      vec2 p = p1 + v * t;
      vec2 radiusVec = p - c;
      if (isVectorInSector(radiusVec, arcStart, arcEnd, isArcCCW)) {
        result.Points[result.Count++] = p;
      }
    }
  };

  checkAdd(t1);

  if (std::abs(t1 - t2) > EPSILON) {
    checkAdd(t2);
  }

  return result;
}

IntersectResult intersectArcArc(vec2 c1, vec2 arc1Start, vec2 arc1End,
                                bool isArc1CCW, vec2 c2, vec2 arc2Start,
                                vec2 arc2End, bool isArc2CCW, double r) {
  double dSquare = length2(c1 - c2);
  double d = std::sqrt(dSquare);

  if (d < EPSILON || d > 2 * r + EPSILON) return IntersectResult{0, {}};

  double a = d / 2.0;
  double arg = r * r - a * a;
  if (arg < -EPSILON) arg = 0;
  if (arg < 0) arg = 0;

  double h = std::sqrt(arg);

  vec2 p2 = c1 + (c2 - c1) * 0.5;
  vec2 offset = vec2{c2.y - c1.y, c1.x - c2.x} * (h / d);

  IntersectResult result;

  auto checkAdd = [&](vec2 p) {
    vec2 v1 = p - c1;
    vec2 v2 = p - c2;
    if (isVectorInSector(v1, arc1Start, arc1End, isArc1CCW) &&
        isVectorInSector(v2, arc2Start, arc2End, isArc2CCW)) {
      result.Points[result.Count++] = p;
    }
  };

  checkAdd(p2 + offset);
  if (h > EPSILON) checkAdd(p2 - offset);

  return result;
}

std::vector<GeomTangentPair> Intersect(const std::array<vec2, 3>& e1Points,
                                       const std::array<vec2, 3>& e2Points,
                                       const double radius, const bool invert) {
  bool e1Convex = isConvex(e1Points) ^ invert,
       e2Convex = isConvex(e2Points) ^ invert;

  vec2 e1Normal = getEdgeNormal(e1Points[1] - e1Points[0], invert),
       e1NextNormal = getEdgeNormal(e1Points[2] - e1Points[1], invert),
       e2Normal = getEdgeNormal(e2Points[1] - e2Points[0], invert),
       e2NextNormal = getEdgeNormal(e2Points[2] - e2Points[1], invert);

  std::array<vec2, 2> offsetE1{e1Points[0] + e1Normal * radius,
                               e1Points[1] + e1Normal * radius},
      offsetE2{e2Points[0] + e2Normal * radius,
               e2Points[1] + e2Normal * radius};

  std::vector<GeomTangentPair> result;

  IntersectResult EE =
      intersectSegments(offsetE1[0], offsetE1[1], offsetE2[0], offsetE2[1]);

  if (EE.Count) {
    double e1T = 0, e2T = 0;
    isPointProjectionOnSegment(EE.Points[0], e1Points[0], e1Points[1], e1T);
    isPointProjectionOnSegment(EE.Points[0], e2Points[0], e2Points[1], e2T);

    if (e1T > EPSILON && e2T > EPSILON) {
      result.emplace_back(GeomTangentPair{{}, {e1T, e2T}, EE.Points[0], {}});
    }
  }

  if (!e1Convex) {
    IntersectResult VE =
        intersectSegmentArc(offsetE2[0], offsetE2[1], e1Points[1], radius,
                            e1Normal, e1NextNormal, invert);

    for (int i = 0; i != VE.Count; i++) {
      double e2T = 0;
      isPointProjectionOnSegment(VE.Points[i], e2Points[0], e2Points[1], e2T);

      if (e2T > EPSILON) {
        result.emplace_back(GeomTangentPair{{}, {1, e2T}, VE.Points[i], {}});
      }
    }
  }

  if (!e2Convex) {
    IntersectResult EV =
        intersectSegmentArc(offsetE1[0], offsetE1[1], e2Points[1], radius,
                            e2Normal, e2NextNormal, invert);

    for (int i = 0; i != EV.Count; i++) {
      double e1T = 0;
      isPointProjectionOnSegment(EV.Points[i], e1Points[0], e1Points[1], e1T);

      if (e1T > EPSILON) {
        result.emplace_back(GeomTangentPair{{}, {e1T, 1}, EV.Points[i], {}});
      }
    }
  }

  if (!e1Convex && !e2Convex) {
    IntersectResult VV =
        intersectArcArc(e1Points[1], e1Normal, e1NextNormal, invert,
                        e2Points[1], e2Normal, e2NextNormal, invert, radius);

    for (int i = 0; i != VV.Count; i++) {
      result.emplace_back(GeomTangentPair{{}, {1, 1}, VV.Points[i], {}});
    }
  }

  return result;
}

}  // namespace

// Sub process
namespace {
using namespace manifold;

ColliderInfo BuildCollider(const Polygons& polygons) {
  Vec<Box> boxVec;
  Vec<uint32_t> mortonVec;

  std::vector<EdgeOld2New> edgeOld2NewVec;

  for (size_t j = 0; j != polygons.size(); j++) {
    auto& loop = polygons[j];
    for (size_t i = 0; i != loop.size(); i++) {
      const vec2 p1 = loop[i], p2 = loop[(i + 1) % loop.size()];

      vec3 center = toVec3(p1) + toVec3(p2);
      center /= 2;

      Box bbox(toVec3(p1), toVec3(p2));

      edgeOld2NewVec.push_back({bbox, Collider::MortonCode(center, bbox), i,
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
    const Polygons& loops, const ColliderInfo& colliderInfo, double radius) {
  bool invert = false;
  if (radius < EPSILON) {
    invert = true;
    radius = std::abs(radius);
  }

  size_t circleIndex = 0;

  std::vector<size_t> loopOffset(loops.size());

  const size_t loopElementCount = ([&loopOffset, &loops]() -> size_t {
    size_t count = 0;
    for (size_t i = 0; i != loops.size(); i++) {
      loopOffset[i] = count;
      count += loops[i].size();
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

  resultOutputFile << std::setprecision(
                          std::numeric_limits<double>::max_digits10)
                   << radius << std::endl;
  std::vector<vec2> removedCircleCenter;
  std::vector<vec2> resultCircleCenter;

  // Multi loops in single polygon
  for (size_t e1Loopi = 0; e1Loopi != loops.size(); e1Loopi++) {
    const auto& e1Loop = loops[e1Loopi];

    // Create BBox for e1 to find potential fillet pair
    for (size_t e1i = 0; e1i != e1Loop.size(); e1i++) {
      const size_t p1i = e1i, p2i = (e1i + 1) % e1Loop.size();
      const std::array<vec2, 3> e1Points{e1Loop[e1i],
                                         e1Loop[(e1i + 1) % e1Loop.size()],
                                         e1Loop[(e1i + 2) % e1Loop.size()]};
      const vec2 e1 = e1Points[1] - e1Points[0];
      const bool p2IsConvex = cross(e1, e1Points[2] - e1Points[1]) >= EPSILON;

      const vec2 normal = getEdgeNormal(e1, invert);

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

      // TODO:  Add flag to avoid double process
      // Potential fillet edge pair with e1
      std::vector<EdgeOld2New> potentialEdges;
      auto recordCollision = [&](int, int edge) {
        potentialEdges.push_back(colliderInfo.EdgeOld2NewVec[edge]);
      };
      auto recorder = MakeSimpleRecorder(recordCollision);

      colliderInfo.MyCollider.Collisions(
          manifold::Vec<manifold::Box>({box}).cview(), recorder);

#ifdef MANIFOLD_DEBUG
      if (ManifoldParams().verbose) {
        std::cout << std::endl
                  << "Now " << e1Points[0] << " -> " << e1Points[1]
                  << " Collider Size:" << potentialEdges.size() << std::endl;
      }
#endif

      // NOTE: Calculate fillet circle center
      for (auto it = potentialEdges.begin(); it != potentialEdges.end(); it++) {
        const auto& ele = *it;

        // e2i is the index of detected possible bridge edge

        const size_t e2i = ele.p1Ref, e2Loopi = ele.loopRef;

        const auto& e2Loop = loops[e2Loopi];
        const size_t p3i = e2i, p4i = (e2i + 1) % e2Loop.size();

        const std::array<vec2, 3> e2Points{e2Loop[e2i],
                                           e2Loop[(e2i + 1) % e2Loop.size()],
                                           e2Loop[(e2i + 2) % e2Loop.size()]};

        // Skip self
        if (e1Loopi == e2Loopi && e2i == e1i) continue;

        std::array<size_t, 4> vBreakPoint{0, 1, 1, 1};

        if (e1Loopi == vBreakPoint[0] && e1i == vBreakPoint[1] &&
            e2Loopi == vBreakPoint[2] && e2i == vBreakPoint[3]) {
          int i = 0;
        }

#ifdef MANIFOLD_DEBUG
        if (ManifoldParams().verbose) {
          std::cout << "-----------" << std::endl
                    << e2Points[0] << " -> " << e2Points[1] << std::endl;

          std::cout << "std::array<size_t, 4> vBreakPoint{" << e1Loopi << ", "
                    << e1i << ", " << e2Loopi << ", " << e2i << "}; "
                    << std::endl;
        }
#endif

        const uint8_t EEMASK = 1, PEMASK = 1 << 1, PPMASK = 1 << 2;

        // Check if processed, and add duplicate mark
        processedMark[getMarkPosition(e1Loopi, e1i, e2Loopi, e2i)] = EEMASK;
        if (processedMark[getMarkPosition(e2Loopi, e2i, e1Loopi, e1i)] ==
            EEMASK) {
          continue;
        }

        std::vector<GeomTangentPair> filletCircles;

        // NOTE: Calculate fillet intersection center
        filletCircles = Intersect(e1Points, e2Points, radius, invert);

        // NOTE: Remove current <e1,e2> pair fillet circle's global overlapping.
        // Local collision has been removed.
        for (auto it = filletCircles.begin(); it != filletCircles.end();) {
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

            const auto& eLoop = loops[edge.loopRef];
            const size_t p1i = edge.p1Ref, p2i = edge.p2Ref;
            const vec2 p1 = eLoop[p1i], p2 = eLoop[p2i];

            double distance = distancePointSegment(arc.CircleCenter, p1, p2);

            if (std::abs(distance - radius) < EPSILON) {
              // TODO: Intersected, this can be used to optimize speed for
              // pre-detect all intersected point and avoid double processed
            } else if (distance < radius) {
              eraseFlag = true;
              break;
            }
          }

#ifdef MANIFOLD_DEBUG
          if (ManifoldParams().verbose) {
            if (eraseFlag) {
              std::cout << "vRemove " << it->CircleCenter << std::endl;
              std::cout << "std::array<size_t, 4> vBreakPoint{" << e1Loopi
                        << ", " << e1i << ", " << e2Loopi << ", " << e2i
                        << "}; " << std::endl;
            } else {
              std::cout << "vAdd " << it->CircleCenter << std::endl;
              std::cout << "std::array<size_t, 4> vBreakPoint{" << e1Loopi
                        << ", " << e1i << ", " << e2Loopi << ", " << e2i
                        << "}; " << std::endl;
            }
          }
#endif

          if (eraseFlag) {
            removedCircleCenter.push_back(it->CircleCenter);
            it = filletCircles.erase(it);
          } else {
            it++;
          }
        }

        // NOTE: Map GeomTangentPair to TopoConnectionPair for Topo Building
        {
          for (auto it = filletCircles.begin(); it != filletCircles.end();
               it++) {
            vec2 p1 = getPointOnEdgeByParameter(e1Points[0], e1Points[1],
                                                it->ParameterValues[0]),
                 p2 = getPointOnEdgeByParameter(e2Points[0], e2Points[1],
                                                it->ParameterValues[1]);

            it->RadValues = getRadPair(p1, p2, it->CircleCenter);

            auto pair = TopoConnectionPair(*it, e1i, e1Loopi, e2i, e2Loopi,
                                           circleIndex++);

            // Ensure Arc start and end direction fit the Loop direction.
            auto check = [&](const TopoConnectionPair& pair) -> bool {
              vec2 p1 = getPointOnEdgeByParameter(e1Points[0], e1Points[1],
                                                  pair.ParameterValues[0]),
                   p2 = getPointOnEdgeByParameter(e2Points[0], e2Points[1],
                                                  pair.ParameterValues[1]);

              vec2 n1 = p1 - pair.CircleCenter, n2 = p2 - pair.CircleCenter;
              if (!invert) {
                return la::cross(n1, n2) < EPSILON;
              } else {
                return la::cross(n1, n2) > EPSILON;
              }

              return false;
            };

            if (check(pair)) pair = pair.Swap();

            size_t index = loopOffset[pair.LoopIndex[0]] + pair.EdgeIndex[0];

            auto order = [&](const TopoConnectionPair& a,
                             const TopoConnectionPair& b) {
              if (a.ParameterValues[0] < b.ParameterValues[0] - EPSILON)
                return true;
              if (a.ParameterValues[0] > b.ParameterValues[0] + EPSILON)
                return false;

              bool aEnd = std::abs(a.ParameterValues[0] - 1.0) < EPSILON;
              bool bEnd = std::abs(b.ParameterValues[0] - 1.0) < EPSILON;

              if (aEnd && bEnd) {
                auto n1 = a.CircleCenter - e1Points[1];
                auto n2 = b.CircleCenter - e1Points[1];
                double det = la::cross(n1, n2);

                return !invert ? (det < EPSILON) : (det > EPSILON);
              }

              return false;
            };

            auto itt = std::find_if(arcConnection[index].begin(),
                                    arcConnection[index].end(),
                                    [&](const TopoConnectionPair& ele) -> bool {
                                      return order(pair, ele);
                                    });

            arcConnection[index].insert(itt, pair);
          }
        }

#ifdef MANIFOLD_DEBUG
        if (ManifoldParams().verbose) {
          for (const auto& e : filletCircles) {
            resultCircleCenter.push_back(e.CircleCenter);
          }
#endif
        }
      }
    }
  }

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose) {
    resultOutputFile << removedCircleCenter.size() << std::endl;
    for (const auto& e : removedCircleCenter) {
      resultOutputFile << e.x << " " << e.y << std::endl;
    }

    resultOutputFile << resultCircleCenter.size() << std::endl;

    for (const auto& e : resultCircleCenter) {
      resultOutputFile << e.x << " " << e.y << std::endl;
    }

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
    const Polygons& loops,
    std::vector<std::vector<TopoConnectionPair>> arcConnection,
    int circularSegments, double radius) {
  std::vector<size_t> loopOffset(loops.size());

  const size_t loopElementCount = ([&loopOffset, &loops]() -> size_t {
    size_t count = 0;
    for (size_t i = 0; i != loops.size(); i++) {
      loopOffset[i] = count;
      count += loops[i].size();
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

  Polygons resultLoops;

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

      const double EPS = EPSILON;

      auto it = std::find_if(
          arcConnection[getEdgePosition(current)].begin(),
          arcConnection[getEdgePosition(current)].end(),
          [current, EPS](const TopoConnectionPair& ele) -> bool {
            return ele.ParameterValues[0] + EPSILON > current.ParameterValue;
          });

      if (it == arcConnection[getEdgePosition(current)].end()) {
        // Not found, just add vertex
        // FIXME: shouldn't add vertex directly, should search for next edge
        // with fillet arc

        if (current == end) {
          break;
        }

        const auto& loop = loops[current.LoopIndex];
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

  for (size_t i = 0; i != loops.size(); i++) {
    if (loopFlag[i] == 0 &&
        (CrossSection(Polygons{loops[i]}).Area() > EPSILON)) {
      SimplePolygon loop = loops[i];
      std::reverse(loop.begin(), loop.end());
      hole = hole.Boolean(CrossSection(Polygons{loop}), manifold::OpType::Add);
    }
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

  outFile << filename << " "
          << "\n";
  outFile << polygons.size() << "\n";
  outFile << std::setprecision(std::numeric_limits<double>::max_digits10);

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

void SaveCrossSection(std::ofstream& outFile,
                      const std::vector<CrossSection>& result) {
  outFile << result.size() << "\n";

  // Write each CrossSection within the test.
  for (const auto& crossSection : result) {
    const auto polygons = crossSection.ToPolygons();

    outFile << polygons.size() << "\n";
    for (const auto& loop : polygons) {
      outFile << loop.size() << "\n";
      for (const auto& point : loop) {
        outFile << point.x << " " << point.y << "\n";
      }
    }
  }

  std::cout << "Successfully saved " << result.size() << " CrossSections. "
            << std::endl;
}

std::vector<CrossSection> FilletImpl(const Polygons& polygons, double radius,
                                     int circularSegments) {
  if (polygons.empty()) return {};

  ColliderInfo colliderInfo = BuildCollider(polygons);

  if (caseIndex == 0) SavePolygons("Testing/Fillet/input.txt", polygons);

  resultOutputFile.open("Testing/Fillet/" + std::to_string(caseIndex) + ".txt");
  if (!resultOutputFile.is_open()) {
    std::cerr << "Error: Could not open file "
              << std::to_string(caseIndex) + ".txt" << " for writing."
              << std::endl;
    throw std::exception();
  }
  caseIndex++;

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
  auto arcConnection = CalculateFilletArc(polygons, colliderInfo, radius);

  int n = circularSegments > 2 ? circularSegments
                               : Quality::GetCircularSegments(radius);

  // Tracing along the arc
  auto result = Tracing(polygons, arcConnection, n, radius);

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose) {
    SaveCrossSection(resultOutputFile, result);
  }
#endif
  resultOutputFile.close();

  return result;
}

}  // namespace

namespace manifold {

std::vector<CrossSection> CrossSection::Fillet(double radius,
                                               int circularSegments) const {
  return FilletImpl(ToPolygons(), radius, circularSegments);
}

}  // namespace manifold
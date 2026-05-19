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

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../collider.h"
#include "../disjoint_sets.h"
#include "../parallel.h"
#include "../vec.h"
#include "clipper2/clipper.core.h"
#include "clipper2/clipper.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"
#include "oneapi/tbb/enumerable_thread_specific.h"

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

struct Edge {
  size_t LoopIndex;
  size_t EdgeIndex;
};

struct EdgePair {
  std::array<size_t, 2> LoopIndex;
  std::array<size_t, 2> EdgeIndex;
};

struct ColliderContext {
  manifold::Collider Collider;
  std::vector<Edge> EdgeOld2NewVec;
};

enum class FilletSubCase : uint8_t { EE, VE, EV, VV };

// Two edge tangent by same circle
struct GeomTangentPair {
  std::array<double, 2> ParameterValues;
  vec2 CircleCenter;

  // CCW or CW is determined by loop's direction
  std::array<double, 2> RadValues;

  FilletSubCase SubCase;
};

struct TopoConnectionPair {
  TopoConnectionPair(const GeomTangentPair& geomPair, size_t Edge1Index,
                     size_t Edge1LoopIndex, size_t Edge2Index,
                     size_t Edge2LoopIndex) {
    CircleCenter = geomPair.CircleCenter;
    ParameterValues = geomPair.ParameterValues;
    RadValues = geomPair.RadValues;

    EdgeIndex = std::array<size_t, 2>{Edge1Index, Edge2Index};
    LoopIndex = std::array<size_t, 2>{Edge1LoopIndex, Edge2LoopIndex};
  }

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

double filletNumericalTolerance(double radius) {
  return std::max(1e-12, 1e-10 * std::max(1.0, radius));
}

vec2 chebyshevCenter(const std::vector<vec2>& points) {
  if (points.empty()) return {};
  if (points.size() == 1) return points.front();

  double bestRadiusSq = std::numeric_limits<double>::infinity();
  vec2 bestCenter = points.front();

  auto maxRadiusSq = [&](const vec2& center) {
    double maxDistSq = 0.0;
    for (const vec2& p : points) {
      maxDistSq = std::max(maxDistSq, length2(p - center));
    }
    return maxDistSq;
  };

  auto consider = [&](const vec2& center) {
    const double radiusSq = maxRadiusSq(center);
    if (radiusSq < bestRadiusSq) {
      bestRadiusSq = radiusSq;
      bestCenter = center;
    }
  };

  for (const vec2& p : points) {
    consider(p);
  }

  for (size_t i = 0; i < points.size(); ++i) {
    for (size_t j = i + 1; j < points.size(); ++j) {
      consider((points[i] + points[j]) * 0.5);
    }
  }

  for (size_t i = 0; i < points.size(); ++i) {
    for (size_t j = i + 1; j < points.size(); ++j) {
      for (size_t k = j + 1; k < points.size(); ++k) {
        const vec2 a = points[i], b = points[j], c = points[k];
        const double d =
            2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));

        if (std::abs(d) < EPSILON) continue;

        const double aSq = dot(a, a), bSq = dot(b, b), cSq = dot(c, c);
        const vec2 center{
            (aSq * (b.y - c.y) + bSq * (c.y - a.y) + cSq * (a.y - b.y)) / d,
            (aSq * (c.x - b.x) + bSq * (a.x - c.x) + cSq * (b.x - a.x)) / d};

        consider(center);
      }
    }
  }

  return bestCenter;
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
  return la::cross(e[1] - e[0], e[2] - e[1]) > EPSILON;
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

// Calculate the fillet circle center
// The distance from edge or vertex to circle center is offset radius.
// So this can be calculated by intersect of Edge and Arc.
//
// Step 3.1: Accepts pre-computed convexity instead of recomputing.
// Step 3.2: Sub-cases are mutually exclusive by strict parameter bounds.
// Step 2.2: RadValues computed internally — single source of truth.
std::vector<GeomTangentPair> Intersect(const std::array<vec2, 3>& e1Points,
                                       const std::array<vec2, 3>& e2Points,
                                       const double radius, const bool invert,
                                       bool e1Convex, bool e2Convex) {
  vec2 e1Normal = getEdgeNormal(e1Points[1] - e1Points[0], invert),
       e1NextNormal = getEdgeNormal(e1Points[2] - e1Points[1], invert),
       e2Normal = getEdgeNormal(e2Points[1] - e2Points[0], invert),
       e2NextNormal = getEdgeNormal(e2Points[2] - e2Points[1], invert);

  std::array<vec2, 2> offsetE1{e1Points[0] + e1Normal * radius,
                               e1Points[1] + e1Normal * radius},
      offsetE2{e2Points[0] + e2Normal * radius,
               e2Points[1] + e2Normal * radius};

  std::vector<GeomTangentPair> result;

  // Helper: compute RadValues for a result (Step 2.2)
  auto computeRadValues = [&](double e1T, double e2T,
                              const vec2& center) -> std::array<double, 2> {
    vec2 p1 = getPointOnEdgeByParameter(e1Points[0], e1Points[1], e1T);
    vec2 p2 = getPointOnEdgeByParameter(e2Points[0], e2Points[1], e2T);
    return getRadPair(p1, p2, center);
  };

  // EE: both parameters strictly interior (Step 3.2)
  IntersectResult EE =
      intersectSegments(offsetE1[0], offsetE1[1], offsetE2[0], offsetE2[1]);

  if (EE.Count) {
    double e1T = 0, e2T = 0;
    isPointProjectionOnSegment(EE.Points[0], e1Points[0], e1Points[1], e1T);
    isPointProjectionOnSegment(EE.Points[0], e2Points[0], e2Points[1], e2T);

    if (e1T > 0 && e1T < 1 && e2T > 0 && e2T < 1) {
      auto radVals = computeRadValues(e1T, e2T, EE.Points[0]);
      result.emplace_back(GeomTangentPair{
          {e1T, e2T}, EE.Points[0], radVals, FilletSubCase::EE});
    }
  }

  // VE: e1 vertex × e2 strictly interior (Step 3.2)
  if (!e1Convex) {
    IntersectResult VE =
        intersectSegmentArc(offsetE2[0], offsetE2[1], e1Points[1], radius,
                            e1Normal, e1NextNormal, invert);

    for (int i = 0; i != VE.Count; i++) {
      double e2T = 0;
      isPointProjectionOnSegment(VE.Points[i], e2Points[0], e2Points[1], e2T);

      if (e2T > 0 && e2T < 1) {
        auto radVals = computeRadValues(1.0, e2T, VE.Points[i]);
        result.emplace_back(GeomTangentPair{
            {1, e2T}, VE.Points[i], radVals, FilletSubCase::VE});
      }
    }
  }

  // EV: e1 strictly interior × e2 vertex (Step 3.2)
  if (!e2Convex) {
    IntersectResult EV =
        intersectSegmentArc(offsetE1[0], offsetE1[1], e2Points[1], radius,
                            e2Normal, e2NextNormal, invert);

    for (int i = 0; i != EV.Count; i++) {
      double e1T = 0;
      isPointProjectionOnSegment(EV.Points[i], e1Points[0], e1Points[1], e1T);

      if (e1T > 0 && e1T < 1) {
        auto radVals = computeRadValues(e1T, 1.0, EV.Points[i]);
        result.emplace_back(GeomTangentPair{
            {e1T, 1}, EV.Points[i], radVals, FilletSubCase::EV});
      }
    }
  }

  // VV: both vertices (Step 3.2)
  if (!e1Convex && !e2Convex) {
    IntersectResult VV =
        intersectArcArc(e1Points[1], e1Normal, e1NextNormal, invert,
                        e2Points[1], e2Normal, e2NextNormal, invert, radius);

    for (int i = 0; i != VV.Count; i++) {
      auto radVals = computeRadValues(1.0, 1.0, VV.Points[i]);
      result.emplace_back(
          GeomTangentPair{{1, 1}, VV.Points[i], radVals, FilletSubCase::VV});
    }
  }

  return result;
}

}  // namespace

// Sub process
namespace {
using namespace manifold;

ColliderContext BuildCollider(const Polygons& polygons,
                              const std::vector<size_t>& loopOffsetVec,
                              std::vector<EdgePair>& edgePairVec, double radius,
                              bool invert) {
  struct EdgeOld2New {
    manifold::Box Box;
    uint32_t Morton;
    size_t LoopIndex;
    size_t EdgeIndex;
  };

  Vec<EdgeOld2New> edgeOld2NewVec;

  Vec<Box> filletBoxVec;
  // Loop index and Edge index
  Vec<std::array<size_t, 2>> filletBoxMapVec;

  for (size_t j = 0; j != polygons.size(); j++) {
    const auto& loop = polygons[j];

    for (size_t i = 0; i != loop.size(); i++) {
      const std::array<vec2, 3> points{loop[i], loop[(i + 1) % loop.size()],
                                       loop[(i + 2) % loop.size()]};

      Box filletBox, edgeBox;
      filletBox = edgeBox = Box(toVec3(points[0]), toVec3(points[1]));

      edgeOld2NewVec.push_back(
          {edgeBox,
           Collider::MortonCode(toVec3((points[0] + points[1]) / 2), edgeBox),
           j, i});

      // See
      // https://docs.google.com/presentation/d/1P-3oxmjmEw_Av0rq7q7symoL5VB5DSWpRvqoa3LK7Pg/edit?usp=sharing

      const vec2 edgeDir = normalize(points[1] - points[0]);

      const vec2 normal = getEdgeNormal(edgeDir, invert),
                 offsetP1 = points[0] + normal * 2.0 * radius,
                 offsetP2 = points[1] + normal * 2.0 * radius;

      filletBox.Union(toVec3(offsetP1));
      filletBox.Union(toVec3(offsetP2));

      filletBox.Union(toVec3(points[0] - edgeDir * radius + normal * radius));
      filletBox.Union(toVec3(points[1] + edgeDir * radius + normal * radius));

      if (!isConvex(points)) {
        const vec2 nextEdgeDir = normalize(points[2] - points[1]),
                   nextNormal = getEdgeNormal(nextEdgeDir, invert);

        filletBox.Union(toVec3(points[1] + nextNormal * 2.0 * radius));
        filletBox.Union(
            toVec3(points[1] + nextEdgeDir * radius + nextNormal * radius));
      }

      filletBoxVec.push_back(filletBox);
      filletBoxMapVec.push_back({j, i});
    }
  }

  std::stable_sort(edgeOld2NewVec.begin(), edgeOld2NewVec.end(),
                   [](const EdgeOld2New& lhs, const EdgeOld2New& rhs) -> bool {
                     return rhs.Morton > lhs.Morton;
                   });

  Vec<Box> boxVec;
  Vec<uint32_t> mortonVec;
  boxVec.resize(edgeOld2NewVec.size());
  mortonVec.resize(edgeOld2NewVec.size());

  manifold::transform(edgeOld2NewVec.begin(), edgeOld2NewVec.end(),
                      boxVec.begin(),
                      [](const EdgeOld2New& edge) { return edge.Box; });

  manifold::transform(edgeOld2NewVec.begin(), edgeOld2NewVec.end(),
                      mortonVec.begin(),
                      [](const EdgeOld2New& edge) { return edge.Morton; });

  ColliderContext info{Collider(boxVec, mortonVec), {}};
  info.EdgeOld2NewVec.resize(edgeOld2NewVec.size());

  manifold::transform(edgeOld2NewVec.begin(), edgeOld2NewVec.end(),
                      info.EdgeOld2NewVec.begin(), [](const EdgeOld2New& edge) {
                        return Edge{edge.LoopIndex, edge.EdgeIndex};
                      });

  // FIXME: this parallel pattern is not very good, search project for preferred
  // pattern
  tbb::enumerable_thread_specific<std::vector<EdgePair>> localPairs;
  auto recordCollision = [&](int i, int j) {
    size_t ii = loopOffsetVec[filletBoxMapVec[i][0]] + filletBoxMapVec[i][1],
           jj = loopOffsetVec[edgeOld2NewVec[j].LoopIndex] +
                edgeOld2NewVec[j].EdgeIndex;

    if (ii <= jj) {
      localPairs.local().push_back(
          EdgePair{{filletBoxMapVec[i][0], edgeOld2NewVec[j].LoopIndex},
                   {filletBoxMapVec[i][1], edgeOld2NewVec[j].EdgeIndex}});
    }
  };

  auto recorder = MakeSimpleRecorder(recordCollision);

  info.Collider.Collisions(recorder, filletBoxVec.cview());

  // merge
  std::vector<EdgePair> edgePair;
  size_t total = 0;
  for (auto& v : localPairs) total += v.size();
  edgePair.reserve(total);
  for (auto& v : localPairs) {
    edgePair.insert(edgePair.end(), v.begin(), v.end());
  }

  edgePairVec.clear();
  edgePairVec = edgePair;

  return info;
}

struct CircleCluster {
  vec2 Center;
  std::vector<size_t> Members;
};

void UpdateClusteredArcCenter(TopoConnectionPair& pair, const Polygons& loops,
                              const vec2& center) {
  pair.CircleCenter = center;

  const auto& edge1Loop = loops[pair.LoopIndex[0]];
  const auto& edge2Loop = loops[pair.LoopIndex[1]];

  const vec2 p1 = getPointOnEdgeByParameter(
      edge1Loop[pair.EdgeIndex[0]],
      edge1Loop[(pair.EdgeIndex[0] + 1) % edge1Loop.size()],
      pair.ParameterValues[0]);
  const vec2 p2 = getPointOnEdgeByParameter(
      edge2Loop[pair.EdgeIndex[1]],
      edge2Loop[(pair.EdgeIndex[1] + 1) % edge2Loop.size()],
      pair.ParameterValues[1]);

  pair.RadValues = getRadPair(p1, p2, pair.CircleCenter);
}

std::vector<CircleCluster> ClusterFilletCircles(
    std::vector<TopoConnectionPair>& topoConnectionVec, const Polygons& loops,
    double radius) {
  const size_t n = topoConnectionVec.size();
  std::vector<CircleCluster> clusters;
  if (n == 0) return clusters;

  const double clusterTol = filletNumericalTolerance(radius);
  const double clusterTolSq = clusterTol * clusterTol;

  DisjointSets circleSets(static_cast<uint32_t>(n));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (length2(topoConnectionVec[i].CircleCenter -
                  topoConnectionVec[j].CircleCenter) <= clusterTolSq) {
        circleSets.unite(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
      }
    }
  }

  std::vector<int> components;
  const int clusterCount = circleSets.connectedComponents(components);
  clusters.resize(clusterCount);

  for (size_t i = 0; i < n; ++i) {
    const size_t clusterIndex = static_cast<size_t>(components[i]);
    clusters[clusterIndex].Members.push_back(i);
  }

  for (CircleCluster& cluster : clusters) {
    std::vector<vec2> centers;
    centers.reserve(cluster.Members.size());

    for (const size_t member : cluster.Members) {
      centers.push_back(topoConnectionVec[member].CircleCenter);
    }

    cluster.Center = chebyshevCenter(centers);

    for (const size_t member : cluster.Members) {
      UpdateClusteredArcCenter(topoConnectionVec[member], loops,
                               cluster.Center);
    }
  }

  return clusters;
}

std::vector<std::vector<TopoConnectionPair>> CalculateFilletArc(
    const Polygons& loops, const std::vector<size_t>& loopOffsetVec,
    const size_t edgeCount, const std::vector<EdgePair>& intersectEdgePair,
    const ColliderContext& collider, double radius, bool invert) {
#ifdef MANIFOLD_DEBUG
  resultOutputFile << std::setprecision(
                          std::numeric_limits<double>::max_digits10)
                   << radius << std::endl;
#endif

  std::vector<vec2> removedCircleCenter;
  std::vector<vec2> resultCircleCenter;

#ifdef MANIFOLD_DEBUG
  auto saveCircleCenters = [&]() {
    resultOutputFile << removedCircleCenter.size() << std::endl;
    for (const auto& e : removedCircleCenter) {
      resultOutputFile << e.x << " " << e.y << std::endl;
    }

    resultOutputFile << resultCircleCenter.size() << std::endl;
    for (const auto& e : resultCircleCenter) {
      resultOutputFile << e.x << " " << e.y << std::endl;
    }
  };
#endif

  std::vector<TopoConnectionPair> topoConnetionVec;

  // NOTE: Calculate fillet circle center, output connect information to
  // topoConnectionVec
  for (auto it = intersectEdgePair.begin(); it != intersectEdgePair.end();
       it++) {
    const size_t edge1InputLoopIdx = it->LoopIndex[0],
                 edge1InputIdx = it->EdgeIndex[0],
                 edge2InputLoopIdx = it->LoopIndex[1],
                 edge2InputIdx = it->EdgeIndex[1];

    // Step 1.2: Canonical ordering — smaller global edge ID is always e1
    const size_t globalId1 = loopOffsetVec[edge1InputLoopIdx] + edge1InputIdx;
    const size_t globalId2 = loopOffsetVec[edge2InputLoopIdx] + edge2InputIdx;
    const bool swapped = globalId1 > globalId2;

    const size_t edge1LoopIdx = swapped ? edge2InputLoopIdx : edge1InputLoopIdx,
                 edge1Idx = swapped ? edge2InputIdx : edge1InputIdx,
                 edge2LoopIdx = swapped ? edge1InputLoopIdx : edge2InputLoopIdx,
                 edge2Idx = swapped ? edge1InputIdx : edge2InputIdx;

    const auto &edge1Loop = loops[edge1LoopIdx],
               &edge2Loop = loops[edge2LoopIdx];

    const std::array<vec2, 3> edge1Points{
        edge1Loop[edge1Idx], edge1Loop[(edge1Idx + 1) % edge1Loop.size()],
        edge1Loop[(edge1Idx + 2) % edge1Loop.size()]},
        edge2Points{edge2Loop[edge2Idx],
                    edge2Loop[(edge2Idx + 1) % edge2Loop.size()],
                    edge2Loop[(edge2Idx + 2) % edge2Loop.size()]};

    // FIXME: why still need this????
    // Skip self
    if (edge1LoopIdx == edge2LoopIdx && edge2Idx == edge1Idx) continue;

#ifdef MANIFOLD_DEBUG
    if (ManifoldParams().verbose) {
      std::cout << std::endl
                << "Now " << edge1Points[0] << " -> " << edge1Points[1]
                << std::endl;

      std::cout << "-----------" << std::endl
                << edge2Points[0] << " -> " << edge2Points[1] << std::endl;

      std::cout << "std::array<size_t, 4> vBreakPoint{" << edge1LoopIdx << ", "
                << edge1Idx << ", " << edge2LoopIdx << ", " << edge2Idx << "}; "
                << std::endl;
    }

    std::array<size_t, 4> vBreakPoint{0, 1, 1, 1};

    if (edge1LoopIdx == vBreakPoint[0] && edge1Idx == vBreakPoint[1] &&
        edge2LoopIdx == vBreakPoint[2] && edge2Idx == vBreakPoint[3]) {
      int i = 0;
    }

#endif

    std::vector<GeomTangentPair> filletCircles;

    // Step 3.1: Pre-compute convexity from stored geometry
    bool e1Convex = isConvex(edge1Points) ^ invert;
    bool e2Convex = isConvex(edge2Points) ^ invert;

    // Calculate fillet intersection center
    // Step 2.2: RadValues now computed inside Intersect()
    filletCircles =
        Intersect(edge1Points, edge2Points, radius, invert, e1Convex, e2Convex);

    for (auto it = filletCircles.begin(); it != filletCircles.end(); it++) {
      // If we swapped the edges for canonical ordering, swap the results back
      auto paramValues = it->ParameterValues;
      auto radValues = it->RadValues;
      size_t outEdge1Idx = edge1Idx, outEdge1LoopIdx = edge1LoopIdx,
             outEdge2Idx = edge2Idx, outEdge2LoopIdx = edge2LoopIdx;
      if (swapped) {
        std::swap(paramValues[0], paramValues[1]);
        std::swap(radValues[0], radValues[1]);
        std::swap(outEdge1Idx, outEdge2Idx);
        std::swap(outEdge1LoopIdx, outEdge2LoopIdx);
      }

      topoConnetionVec.emplace_back(TopoConnectionPair(
          GeomTangentPair{paramValues, it->CircleCenter, radValues,
                          it->SubCase},
          outEdge1Idx, outEdge1LoopIdx, outEdge2Idx, outEdge2LoopIdx));
    }
  }

  if (topoConnetionVec.empty()) {
#ifdef MANIFOLD_DEBUG
    saveCircleCenters();
#endif
    return std::vector<std::vector<TopoConnectionPair>>();
  }

  const bool disableCircleCluster = true;
  std::vector<CircleCluster> circleClusters;

  if (disableCircleCluster) {
    circleClusters.reserve(topoConnetionVec.size());
    for (size_t i = 0; i < topoConnetionVec.size(); ++i) {
      circleClusters.push_back(
          CircleCluster{topoConnetionVec[i].CircleCenter, {i}});
    }
  } else {
    // NOTE: Cluster nearly identical candidate circles before validity testing.
    // This prevents tiny center differences from giving duplicate candidates
    // different keep/remove decisions against neighboring edges.
    circleClusters = ClusterFilletCircles(topoConnetionVec, loops, radius);
  }

  // NOTE: Filter invalid fillet circles per cluster.
  std::vector<uint8_t> mark(topoConnetionVec.size(), 0);
  {
    Vec<Box> circleBoxVec(circleClusters.size());

    for (auto it = circleClusters.begin(); it != circleClusters.end(); it++) {
      vec2 center = it->Center;
      Box box(toVec3(center - vec2(1, 0) * radius),
              toVec3(center + vec2(1, 0) * radius));

      box.Union(toVec3(center - vec2(0, 1) * radius));
      box.Union(toVec3(center + vec2(0, 1) * radius));

      size_t index = std::distance(circleClusters.begin(), it);
      circleBoxVec[index] = box;
    }

    // If a circle cluster is invalid, every candidate in the cluster is marked.
    {
      std::vector<uint8_t> markCluster(circleClusters.size(), 0);
      auto markInvalidCircle = [&](int i, int j) {
        if (markCluster[i]) return;

        const size_t ei = collider.EdgeOld2NewVec[j].EdgeIndex,
                     eLoopi = collider.EdgeOld2NewVec[j].LoopIndex;
        const auto& eLoop = loops[eLoopi];
        const CircleCluster& cluster = circleClusters[i];

        bool edgeCanInvalidateCluster = false;
        for (const size_t member : cluster.Members) {
          const TopoConnectionPair& pair = topoConnetionVec[member];
          const bool isSource =
              (eLoopi == pair.LoopIndex[0] && ei == pair.EdgeIndex[0]) ||
              (eLoopi == pair.LoopIndex[1] && ei == pair.EdgeIndex[1]);

          if (!isSource) {
            edgeCanInvalidateCluster = true;
            break;
          }
        }

        if (!edgeCanInvalidateCluster) {
          return;
        }

        const std::array<vec2, 3> ePoints{eLoop[ei],
                                          eLoop[(ei + 1) % eLoop.size()],
                                          eLoop[(ei + 2) % eLoop.size()]};

        double distance =
            distancePointSegment(cluster.Center, ePoints[0], ePoints[1]);

        const double gap = distance - radius;
        const double tol = filletNumericalTolerance(radius);

        if (gap <= tol) {
#ifdef MANIFOLD_DEBUG
          if (ManifoldParams().verbose) {
            const TopoConnectionPair& pair =
                topoConnetionVec[cluster.Members.front()];
            std::cout << "Removed by edge [" << eLoopi << ", " << ei << "] "
                      << "arc [" << pair.LoopIndex[0] << ", "
                      << pair.EdgeIndex[0] << "] -> [" << pair.LoopIndex[1]
                      << ", " << pair.EdgeIndex[1] << "] "
                      << "cluster=" << i << " center=" << cluster.Center
                      << " distance=" << distance << " radius=" << radius
                      << " gap=" << gap << " tol=" << tol << std::endl;
          }
#endif
          markCluster[i] = 1;
        }
      };

      auto recorder = MakeSimpleRecorder(markInvalidCircle);

      collider.Collider.Collisions(recorder, circleBoxVec.cview());

      for (size_t clusterIndex = 0; clusterIndex < circleClusters.size();
           ++clusterIndex) {
        if (!markCluster[clusterIndex]) continue;

        for (const size_t member : circleClusters[clusterIndex].Members) {
          mark[member] = 1;
        }
      }
    }

#ifdef MANIFOLD_DEBUG
    if (ManifoldParams().verbose) {
      for (size_t i = 0; i != topoConnetionVec.size(); i++) {
        if (mark[i]) {
          std::cout << "Removed " << topoConnetionVec[i].CircleCenter
                    << std::endl;

          removedCircleCenter.push_back(topoConnetionVec[i].CircleCenter);
        } else {
          std::cout << "Added " << topoConnetionVec[i].CircleCenter
                    << std::endl;

          resultCircleCenter.push_back(topoConnetionVec[i].CircleCenter);
        }
      }
    }
#endif
  }

  // NOTE: Map GeomTangentPair to TopoConnectionPair for Topo Building

  std::vector<std::vector<TopoConnectionPair>> arcConnection(
      edgeCount, std::vector<TopoConnectionPair>());

  for (auto it = topoConnetionVec.begin(); it != topoConnetionVec.end(); it++) {
    // Invalid candidate removed
    if (mark[size_t(std::distance(topoConnetionVec.begin(), it))]) continue;

    auto pair = *it;

    // Step 2.1: Direction is fixed by canonical ordering (Step 1.2).
    // Ensure arc goes from e1's tangent point to e2's tangent point
    // in the polygon's winding direction.
    {
      const size_t edge1LoopIdx = pair.LoopIndex[0],
                   edge1Idx = pair.EdgeIndex[0],
                   edge2LoopIdx = pair.LoopIndex[1],
                   edge2Idx = pair.EdgeIndex[1];
      const auto &edge1Loop = loops[edge1LoopIdx],
                 &edge2Loop = loops[edge2LoopIdx];

      vec2 p1 = getPointOnEdgeByParameter(
               edge1Loop[edge1Idx],
               edge1Loop[(edge1Idx + 1) % edge1Loop.size()],
               pair.ParameterValues[0]),
           p2 = getPointOnEdgeByParameter(
               edge2Loop[edge2Idx],
               edge2Loop[(edge2Idx + 1) % edge2Loop.size()],
               pair.ParameterValues[1]);

      vec2 n1 = p1 - pair.CircleCenter, n2 = p2 - pair.CircleCenter;
      bool needsSwap =
          !invert ? (la::cross(n1, n2) < 0) : (la::cross(n1, n2) > 0);
      if (needsSwap) pair = pair.Swap();
    }

    size_t index = loopOffsetVec[pair.LoopIndex[0]] + pair.EdgeIndex[0];

    // Step 1.3: Push_back instead of find_if+insert. Sort happens below.
    arcConnection[index].push_back(pair);
  }

  // Step 1.3 + Step 4: Stable sort each edge's arcs with composite key
  for (size_t idx = 0; idx < edgeCount; idx++) {
    auto& arcs = arcConnection[idx];
    if (arcs.size() <= 1) continue;

    // Find which loop/edge this index belongs to (for vertex angle tiebreak)
    size_t loopIdx = 0, edgeIdx = 0;
    for (size_t li = 0; li < loops.size(); li++) {
      if (idx < loopOffsetVec[li] + loops[li].size()) {
        loopIdx = li;
        edgeIdx = idx - loopOffsetVec[li];
        break;
      }
    }
    const auto& loop = loops[loopIdx];
    const vec2 vertex = loop[(edgeIdx + 1) % loop.size()];

    std::stable_sort(
        arcs.begin(), arcs.end(),
        [&](const TopoConnectionPair& a, const TopoConnectionPair& b) {
          // Primary: parameter value
          if (a.ParameterValues[0] != b.ParameterValues[0])
            return a.ParameterValues[0] < b.ParameterValues[0];

          // Secondary: at endpoint (t==1), sort by vertex angle
          if (a.ParameterValues[0] == 1.0 && b.ParameterValues[0] == 1.0) {
            auto n1 = a.CircleCenter - vertex;
            auto n2 = b.CircleCenter - vertex;
            double det = la::cross(n1, n2);
            if (det != 0.0) return !invert ? (det < 0) : (det > 0);
          }

          // Tertiary: other edge global ID for full determinism
          size_t aOther = loopOffsetVec[a.LoopIndex[1]] + a.EdgeIndex[1];
          size_t bOther = loopOffsetVec[b.LoopIndex[1]] + b.EdgeIndex[1];
          return aOther < bOther;
        });
  }

#ifdef MANIFOLD_DEBUG
  saveCircleCenters();

  if (ManifoldParams().verbose) {
    for (size_t i = 0; i != arcConnection.size(); i++) {
      std::cout << i << " " << arcConnection[i].size();
      for (size_t j = 0; j != arcConnection[i].size(); j++) {
        std::cout << "\t  [" << arcConnection[i][j].LoopIndex[0] << ", "
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
    const Polygons& loops, const std::vector<size_t>& loopOffsetVec,
    std::vector<std::vector<TopoConnectionPair>> arcConnection,
    int circularSegments, double radius) {
  struct EdgeLoopPair {
    size_t EdgeIndex = 0;
    size_t LoopIndex = 0;
    double ParameterValue = 0.0;
    vec2 CircleCenter{};
  };

  auto fail = [](const std::string& msg) -> void {
    throw std::runtime_error("Tracing(): " + msg);
  };

  auto failState = [](const std::string& msg,
                      const EdgeLoopPair& current) -> void {
    std::ostringstream oss;
    oss << "Tracing(): " << msg << " current=[" << current.LoopIndex << ", "
        << current.EdgeIndex << "], t=" << current.ParameterValue
        << ", center={" << current.CircleCenter.x << ", "
        << current.CircleCenter.y << "}";
    throw std::runtime_error(oss.str());
  };

  auto near = [](double a, double b) -> bool {
    return std::abs(a - b) <= EPSILON;
  };

  auto validFinite = [](double v) -> bool { return std::isfinite(v); };

  auto validVec = [&](const vec2& v) -> bool {
    return validFinite(v.x) && validFinite(v.y);
  };

  auto validParam = [&](double t) -> bool {
    return validFinite(t) && t >= -EPSILON && t <= 1.0 + EPSILON;
  };

  auto samePoint = [&](const vec2& a, const vec2& b) -> bool {
    return length2(a - b) <= EPSILON * EPSILON;
  };

  size_t inputArcCount = 0;
  for (const auto& bucket : arcConnection) {
    inputArcCount += bucket.size();
  }

  if (loops.empty()) {
    if (!loopOffsetVec.empty() || inputArcCount != 0) {
      fail(
          "empty loops but non-empty loopOffsetVec or non-empty arcConnection");
    }
    return {};
  }

  if (!std::isfinite(radius) || radius <= 0.0) {
    fail("radius must be finite and positive");
  }

  if (circularSegments < 3) {
    fail("circularSegments must be >= 3");
  }

  if (loopOffsetVec.size() != loops.size()) {
    fail("loopOffsetVec size must equal loops size");
  }

  // Count local edges and validate loop vertices.
  // Do NOT require loopOffsetVec to be a contiguous prefix sum here.
  // For clipped/sparse subparts, loopOffsetVec may contain global offsets
  // with gaps.
  size_t localEdgeCount = 0;
  std::set<size_t> globalEdgePositions;

  for (size_t li = 0; li < loops.size(); ++li) {
    if (loops[li].size() < 3) {
      std::ostringstream oss;
      oss << "loop " << li << " has fewer than 3 vertices";
      fail(oss.str());
    }

    for (size_t vi = 0; vi < loops[li].size(); ++vi) {
      if (!validVec(loops[li][vi])) {
        std::ostringstream oss;
        oss << "loop " << li << " vertex " << vi
            << " contains non-finite coordinate";
        fail(oss.str());
      }
    }

    for (size_t ei = 0; ei < loops[li].size(); ++ei) {
      const size_t globalEdgePos = loopOffsetVec[li] + ei;
      if (!globalEdgePositions.insert(globalEdgePos).second) {
        std::ostringstream oss;
        oss << "duplicate global edge position " << globalEdgePos
            << " from loop " << li << ", edge " << ei;
        fail(oss.str());
      }
    }

    localEdgeCount += loops[li].size();
  }

  auto edgePositionOf = [&](size_t loopIdx, size_t edgeIdx) -> size_t {
    if (loopIdx >= loops.size()) {
      std::ostringstream oss;
      oss << "invalid loop index " << loopIdx;
      fail(oss.str());
    }

    if (edgeIdx >= loops[loopIdx].size()) {
      std::ostringstream oss;
      oss << "invalid edge index " << edgeIdx << " for loop " << loopIdx
          << ", loop size = " << loops[loopIdx].size();
      fail(oss.str());
    }

    return loopOffsetVec[loopIdx] + edgeIdx;
  };

  auto getEdgePosition = [&](const EdgeLoopPair& edge) -> size_t {
    return edgePositionOf(edge.LoopIndex, edge.EdgeIndex);
  };

  auto sameArc = [&](const TopoConnectionPair& a,
                     const TopoConnectionPair& b) -> bool {
    return a.LoopIndex == b.LoopIndex && a.EdgeIndex == b.EdgeIndex &&
           near(a.ParameterValues[0], b.ParameterValues[0]) &&
           near(a.ParameterValues[1], b.ParameterValues[1]) &&
           samePoint(a.CircleCenter, b.CircleCenter);
  };

  auto validateArcGeometryOnly = [&](const TopoConnectionPair& arc) -> void {
    for (int k = 0; k < 2; ++k) {
      if (arc.LoopIndex[k] >= loops.size()) {
        std::ostringstream oss;
        oss << "arc has invalid LoopIndex[" << k << "] = " << arc.LoopIndex[k];
        fail(oss.str());
      }

      if (arc.EdgeIndex[k] >= loops[arc.LoopIndex[k]].size()) {
        std::ostringstream oss;
        oss << "arc has invalid EdgeIndex[" << k << "] = " << arc.EdgeIndex[k]
            << " for loop " << arc.LoopIndex[k];
        fail(oss.str());
      }

      if (!validParam(arc.ParameterValues[k])) {
        std::ostringstream oss;
        oss << "arc has invalid ParameterValues[" << k
            << "] = " << arc.ParameterValues[k];
        fail(oss.str());
      }

      if (!validFinite(arc.RadValues[k])) {
        std::ostringstream oss;
        oss << "arc has invalid RadValues[" << k << "] = " << arc.RadValues[k];
        fail(oss.str());
      }
    }

    if (!validVec(arc.CircleCenter)) {
      fail("arc has non-finite CircleCenter");
    }
  };

  // Sparse/global lookup table.
  // Important: ignore the original bucket index. In clipped cases,
  // arcConnection[i] may not correspond to global edge i.
  std::unordered_map<size_t, std::vector<TopoConnectionPair>> arcsByEdge;
  size_t totalArcCount = 0;

  for (const auto& bucket : arcConnection) {
    for (const auto& arc : bucket) {
      validateArcGeometryOnly(arc);

      const size_t startEdgePos =
          edgePositionOf(arc.LoopIndex[0], arc.EdgeIndex[0]);

      arcsByEdge[startEdgePos].push_back(arc);
      ++totalArcCount;
    }
  }

  if (totalArcCount == 0) {
    return {};
  }

  // Sort only by start parameter. Keep equal-parameter order stable so that
  // endpoint ordering from CalculateFilletArc() is not destroyed.
  for (auto& kv : arcsByEdge) {
    auto& arcs = kv.second;

    std::stable_sort(
        arcs.begin(), arcs.end(),
        [&](const TopoConnectionPair& a, const TopoConnectionPair& b) {
          if (!near(a.ParameterValues[0], b.ParameterValues[0])) {
            return a.ParameterValues[0] < b.ParameterValues[0];
          }

          return false;
        });
  }

  auto appendArcPoints = [&](SimplePolygon& dst,
                             const TopoConnectionPair& arc) -> void {
    auto pts = discreteArcToPoint(arc, radius, circularSegments);

    if (pts.empty()) {
      fail("discreteArcToPoint returned empty point list");
    }

    for (const auto& p : pts) {
      if (!validVec(p)) {
        fail("discreteArcToPoint returned non-finite point");
      }
    }

    size_t begin = 0;
    if (!dst.empty() && samePoint(dst.back(), pts.front())) {
      begin = 1;
    }

    dst.insert(dst.end(), pts.begin() + begin, pts.end());
  };

  auto isWrongEndpointCandidate = [&](const TopoConnectionPair& candidate,
                                      const EdgeLoopPair& current) -> bool {
    const bool bothStart = near(current.ParameterValue, 0.0) &&
                           near(candidate.ParameterValues[0], 0.0);

    const bool bothEnd = near(current.ParameterValue, 1.0) &&
                         near(candidate.ParameterValues[0], 1.0);

    if (!bothStart && !bothEnd) {
      return false;
    }

    const auto& loop = loops[current.LoopIndex];

    // t == 1 means the endpoint is the edge end vertex, not the edge start.
    const vec2 vertex = bothEnd ? loop[(current.EdgeIndex + 1) % loop.size()]
                                : loop[current.EdgeIndex];

    const double sideDot =
        la::dot(candidate.CircleCenter - vertex, current.CircleCenter - vertex);

    // Preserve old intent: if the candidate and previous circle are on the
    // same side of the vertex, skip this candidate.
    return sideDot > EPSILON;
  };

  auto findNextArc = [&](std::vector<TopoConnectionPair>& currentEdge,
                         const EdgeLoopPair& current) {
    for (auto it = currentEdge.begin(); it != currentEdge.end(); ++it) {
      if (it->ParameterValues[0] + EPSILON < current.ParameterValue) {
        continue;
      }

      if (isWrongEndpointCandidate(*it, current)) {
        continue;
      }

      return it;
    }

    return currentEdge.end();
  };

  auto findFirstNonEmptyEdge = [&]() -> std::optional<size_t> {
    std::optional<size_t> best;

    for (const auto& kv : arcsByEdge) {
      const size_t edgePos = kv.first;
      const auto& arcs = kv.second;

      if (arcs.empty()) {
        continue;
      }

      if (!best || edgePos < *best) {
        best = edgePos;
      }
    }

    return best;
  };

  std::vector<uint8_t> loopFlag(loops.size(), 0);
  Polygons resultLoops;

  const size_t maxGlobalSteps =
      totalArcCount * (localEdgeCount + 1) + localEdgeCount + 16;

  while (true) {
    const std::optional<size_t> startEdgePosOpt = findFirstNonEmptyEdge();

    if (!startEdgePosOpt) {
      break;
    }

    const size_t startEdgePos = *startEdgePosOpt;
    auto startBucketIt = arcsByEdge.find(startEdgePos);

    if (startBucketIt == arcsByEdge.end() || startBucketIt->second.empty()) {
      fail("internal error: selected empty start edge");
    }

    const TopoConnectionPair startArc = startBucketIt->second.front();

    SimplePolygon tracingLoop;
    appendArcPoints(tracingLoop, startArc);

    loopFlag[startArc.LoopIndex[0]] = 1;
    loopFlag[startArc.LoopIndex[1]] = 1;

    EdgeLoopPair current{startArc.EdgeIndex[1], startArc.LoopIndex[1],
                         startArc.ParameterValues[1], startArc.CircleCenter};

    bool closed = false;
    size_t noArcWalkCount = 0;
    size_t stepCount = 0;

    while (true) {
      if (++stepCount > maxGlobalSteps) {
        failState("step limit exceeded; possible non-terminating trace",
                  current);
      }

      const size_t currentEdgePos = getEdgePosition(current);
      auto edgeBucketIt = arcsByEdge.find(currentEdgePos);

      // No fillet arc bucket on this edge: walk along the original polygon.
      if (edgeBucketIt == arcsByEdge.end() || edgeBucketIt->second.empty()) {
        const auto& loop = loops[current.LoopIndex];

        tracingLoop.push_back(loop[(current.EdgeIndex + 1) % loop.size()]);

        current.EdgeIndex = (current.EdgeIndex + 1) % loop.size();
        current.ParameterValue = 0.0;

        ++noArcWalkCount;

        if (noArcWalkCount > loop.size()) {
          failState(
              "open/orphan trace: walked one complete loop without finding "
              "the next arc",
              current);
        }

        continue;
      }

      auto& currentEdge = edgeBucketIt->second;
      auto it = findNextArc(currentEdge, current);

      // Bucket exists, but no usable arc ahead on this edge.
      if (it == currentEdge.end()) {
        const auto& loop = loops[current.LoopIndex];

        tracingLoop.push_back(loop[(current.EdgeIndex + 1) % loop.size()]);

        current.EdgeIndex = (current.EdgeIndex + 1) % loop.size();
        current.ParameterValue = 0.0;

        ++noArcWalkCount;

        if (noArcWalkCount > loop.size()) {
          failState(
              "open/orphan trace: walked one complete loop without finding "
              "the next arc",
              current);
        }

        continue;
      }

      noArcWalkCount = 0;

      const TopoConnectionPair arc = *it;
      const bool returnedToStart = sameArc(arc, startArc);

      currentEdge.erase(it);

      if (returnedToStart) {
        closed = true;
        break;
      }

      appendArcPoints(tracingLoop, arc);

      current = {arc.EdgeIndex[1], arc.LoopIndex[1], arc.ParameterValues[1],
                 arc.CircleCenter};

      loopFlag[arc.LoopIndex[1]] = 1;
    }

    if (!closed) {
      fail("internal error: trace exited without closure");
    }

    if (tracingLoop.size() < 3) {
      fail("closed trace has fewer than 3 points");
    }

    resultLoops.push_back(tracingLoop);
  }

  CrossSection hole;

  for (size_t i = 0; i < loops.size(); ++i) {
    SimplePolygon loop = loops[i];
    std::reverse(loop.begin(), loop.end());

    CrossSection cs = loop;
    const double area = cs.Area();

    if (!std::isfinite(area)) {
      std::ostringstream oss;
      oss << "non-finite area for untouched loop " << i;
      fail(oss.str());
    }

    if (loopFlag[i] == 0 && area > 0) {
      hole = hole.Boolean(cs, manifold::OpType::Add);
    }
  }

  std::vector<CrossSection> result;
  result.reserve(resultLoops.size());

  for (const auto& loop : resultLoops) {
    CrossSection cs = CrossSection(Polygons{loop});
    result.push_back(cs.Boolean(hole, manifold::OpType::Subtract));
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

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose) {
    std::cout << "Successfully saved " << result.size() << " CrossSections. "
              << std::endl;
  }
#endif
}

std::vector<CrossSection> FilletImpl(const Polygons& polygons, double radius,
                                     int circularSegments) {
  if (polygons.empty()) return {};

  bool invert = false;
  if (radius < 0) {
    invert = true;
    radius = std::abs(radius);
  }

  std::vector<size_t> loopOffsetVec(polygons.size());

  auto numVert = [](const SimplePolygon& loop) { return loop.size(); };

  manifold::exclusive_scan(TransformIterator(polygons.begin(), numVert),
                           TransformIterator(polygons.end(), numVert),
                           loopOffsetVec.begin(), (size_t)0);

  const size_t edgeCount = loopOffsetVec.back() + polygons.back().size();

  std::vector<EdgePair> edgePairVec;
  const ColliderContext colliderContext =
      BuildCollider(polygons, loopOffsetVec, edgePairVec, radius, invert);

  // Step 1.1: Sort edgePairVec deterministically, independent of BVH traversal
  std::stable_sort(edgePairVec.begin(), edgePairVec.end(),
                   [](const EdgePair& a, const EdgePair& b) {
                     if (a.LoopIndex[0] != b.LoopIndex[0])
                       return a.LoopIndex[0] < b.LoopIndex[0];
                     if (a.EdgeIndex[0] != b.EdgeIndex[0])
                       return a.EdgeIndex[0] < b.EdgeIndex[0];
                     if (a.LoopIndex[1] != b.LoopIndex[1])
                       return a.LoopIndex[1] < b.LoopIndex[1];
                     return a.EdgeIndex[1] < b.EdgeIndex[1];
                   });

#ifdef MANIFOLD_DEBUG
  if (caseIndex == 0) SavePolygons("Testing/Fillet/input.txt", polygons);

  resultOutputFile.open("Testing/Fillet/" + std::to_string(caseIndex) + ".txt");
  if (!resultOutputFile.is_open()) {
    std::cerr << "Error: Could not open file "
              << std::to_string(caseIndex) + ".txt"
              << " for writing." << std::endl;
    throw std::exception();
  }
  caseIndex++;

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
  auto arcConnection =
      CalculateFilletArc(polygons, loopOffsetVec, edgeCount, edgePairVec,
                         colliderContext, radius, invert);

  int n = circularSegments > 2 ? circularSegments
                               : Quality::GetCircularSegments(radius);

  // Tracing along the arc
  auto result = Tracing(polygons, loopOffsetVec, arcConnection, n, radius);
  // auto result = std::vector<CrossSection>();
#ifdef MANIFOLD_DEBUG
  SaveCrossSection(resultOutputFile, result);
  resultOutputFile.close();
#endif

  return result;
}

}  // namespace

namespace manifold {

std::vector<CrossSection> CrossSection::Fillet(double radius,
                                               int circularSegments) const {
  return FilletImpl(ToPolygons(), radius, circularSegments);
}

}  // namespace manifold

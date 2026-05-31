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
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../collider.h"
#include "../parallel.h"
#include "../vec.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"
#include "oneapi/tbb/enumerable_thread_specific.h"

using namespace manifold;

namespace {

#ifdef MANIFOLD_DEBUG
size_t caseIndex = 0;
std::ofstream resultOutputFile;
#endif

struct EdgeRef {
  size_t loop;
  size_t edge;
};

struct FilletInput {
  const Polygons& loops;
  const std::vector<size_t>& loopOffsets;
  const std::vector<EdgeRef>& edges;
  bool invert;
  double minArcAngle;
};

struct FilletGeometry {
  int fromEdge;
  int toEdge;
  double fromT;
  double toT;
  vec2 fromPoint;
  vec2 toPoint;
  vec2 center;
  double startRad;
  double endRad;
};

struct IntersectResult {
  int count = 0;
  std::array<vec2, 2> points{vec2(), vec2()};
};

struct Tangent {
  double t;
  size_t fillet;
  vec2 point;
};

struct Segment {
  int edge;
  double startT;
  double endT;
  vec2 startPoint;
  vec2 endPoint;
  size_t endFillet;
  bool used = false;
};

struct EdgeBox {
  manifold::Box box;
  uint32_t morton;
  int edge;
};

struct EdgeBoxMortonLess {
  bool operator()(const EdgeBox& a, const EdgeBox& b) const {
    return a.morton < b.morton;
  }
};

struct TangentParameterLess {
  bool operator()(const Tangent& a, const Tangent& b) const { return a.t < b.t; }
};

struct CollisionRecorder {
  const Vec<int>& filletBoxEdges;
  const Vec<EdgeBox>& edgeBoxes;
  tbb::enumerable_thread_specific<std::vector<std::pair<int, int>>>& localPairs;

  void operator()(int filletBox, int edgeBox) const {
    int a = filletBoxEdges[filletBox];
    int b = edgeBoxes[edgeBox].edge;
    if (a == b) return;
    if (b < a) std::swap(a, b);
    localPairs.local().push_back({a, b});
  }
};

size_t NumVert(const SimplePolygon& loop) { return loop.size(); }

vec3 ToVec3(vec2 v) { return vec3(v.x, v.y, 0); }

double ToRad(const vec2& v) { return atan2(v.y, v.x); }

double NormalizeAngle(double angle) {
  while (angle < 0) angle += 2.0 * M_PI;
  while (angle >= 2.0 * M_PI) angle -= 2.0 * M_PI;
  return angle;
}

bool IsFinite(const vec2& v) { return std::isfinite(v.x) && std::isfinite(v.y); }

bool IsInFilletRange(const vec2& v) {
  constexpr double kMaxCoordinate = 1e30;
  return IsFinite(v) && std::abs(v.x) < kMaxCoordinate &&
         std::abs(v.y) < kMaxCoordinate;
}

bool SamePoint(const vec2& a, const vec2& b) {
  return a.x == b.x && a.y == b.y;
}

bool SamePointWithin(const vec2& a, const vec2& b, double tolerance) {
  return tolerance > 0 && length2(a - b) <= tolerance * tolerance;
}

void AppendPoint(SimplePolygon& loop, const vec2& point, double tolerance = 0) {
  if (loop.empty() ||
      (!SamePoint(loop.back(), point) &&
       !SamePointWithin(loop.back(), point, tolerance))) {
    loop.push_back(point);
  }
}

vec2 EdgeStart(const FilletInput& input, int edge) {
  const EdgeRef ref = input.edges[edge];
  return input.loops[ref.loop][ref.edge];
}

vec2 EdgeEnd(const FilletInput& input, int edge) {
  const EdgeRef ref = input.edges[edge];
  const SimplePolygon& loop = input.loops[ref.loop];
  return loop[(ref.edge + 1) % loop.size()];
}

int NextEdge(const FilletInput& input, int edge) {
  const EdgeRef ref = input.edges[edge];
  const SimplePolygon& loop = input.loops[ref.loop];
  return static_cast<int>(input.loopOffsets[ref.loop] +
                          (ref.edge + 1) % loop.size());
}

vec2 EdgeNormal(const vec2& edge, bool invert) {
  return normalize(!invert ? vec2(-edge.y, edge.x) : vec2(edge.y, -edge.x));
}

bool IsConvexTurn(const vec2& a, const vec2& b, const vec2& c) {
  return la::cross(b - a, c - b) > 0;
}

bool IntersectLines(vec2 p, vec2 d, vec2 q, vec2 e, vec2& point, double& t,
                    double& u) {
  const double den = la::cross(d, e);
  if (den == 0) return false;

  const vec2 qp = q - p;
  t = la::cross(qp, e) / den;
  u = la::cross(qp, d) / den;
  point = p + t * d;
  return IsFinite(point) && std::isfinite(t) && std::isfinite(u);
}

bool IsVectorInSector(vec2 v, vec2 start, vec2 end, bool ccw) {
  const double crossStartEnd = la::cross(start, end);
  const double crossStartV = la::cross(start, v);
  const double crossVEnd = la::cross(v, end);

  if (length2(v - start) == 0 || length2(v - end) == 0) return true;

  if (!ccw) return IsVectorInSector(v, end, start, true);

  if (crossStartEnd >= 0) return crossStartV >= 0 && crossVEnd >= 0;
  return crossStartV >= 0 || crossVEnd >= 0;
}

IntersectResult IntersectSegmentArc(vec2 p, vec2 q, vec2 center, double radius,
                                    vec2 arcStart, vec2 arcEnd, bool arcCCW) {
  const vec2 edge = q - p;
  const double lenSq = dot(edge, edge);
  if (lenSq == 0) return {};

  const double projectionT = dot(center - p, edge) / lenSq;
  const vec2 projection = p + edge * projectionT;
  const double distSq = length2(center - projection);
  const double radiusSq = radius * radius;
  if (distSq > radiusSq) return {};

  const double offset = std::sqrt(radiusSq - distSq);
  const double tOffset = offset / std::sqrt(lenSq);
  const std::array<double, 2> values{projectionT - tOffset,
                                     projectionT + tOffset};

  IntersectResult result;
  for (const double t : values) {
    if (result.count == 1 && t == values[0]) continue;
    if (!(t >= 0 && t <= 1)) continue;

    const vec2 point = p + edge * t;
    if (IsVectorInSector(point - center, arcStart, arcEnd, arcCCW)) {
      result.points[result.count++] = point;
    }
  }

  return result;
}

IntersectResult IntersectArcArc(vec2 c1, vec2 arc1Start, vec2 arc1End,
                                bool arc1CCW, vec2 c2, vec2 arc2Start,
                                vec2 arc2End, bool arc2CCW, double radius) {
  const double dSq = length2(c1 - c2);
  const double d = std::sqrt(dSq);
  if (d == 0 || d > 2 * radius) return {};

  const double halfDistance = d * 0.5;
  const double heightSq = radius * radius - halfDistance * halfDistance;
  if (heightSq < 0) return {};

  const double height = std::sqrt(heightSq);
  const vec2 midpoint = (c1 + c2) * 0.5;
  const vec2 offset = vec2{c2.y - c1.y, c1.x - c2.x} * (height / d);

  IntersectResult result;
  const std::array<vec2, 2> points{midpoint + offset, midpoint - offset};
  for (const vec2 point : points) {
    if (result.count == 1 && SamePoint(point, result.points[0])) continue;
    if (IsVectorInSector(point - c1, arc1Start, arc1End, arc1CCW) &&
        IsVectorInSector(point - c2, arc2Start, arc2End, arc2CCW)) {
      result.points[result.count++] = point;
    }
  }

  return result;
}

std::vector<EdgeRef> BuildEdges(const Polygons& polygons,
                                const std::vector<size_t>& loopOffsets) {
  const size_t edgeCount = loopOffsets.back() + polygons.back().size();
  std::vector<EdgeRef> edges(edgeCount);

  for (size_t loop = 0; loop < polygons.size(); ++loop) {
    for (size_t edge = 0; edge < polygons[loop].size(); ++edge) {
      edges[loopOffsets[loop] + edge] = EdgeRef{loop, edge};
    }
  }

  return edges;
}

std::vector<size_t> BuildLoopOffsets(const Polygons& polygons) {
  std::vector<size_t> loopOffsets(polygons.size());

  manifold::exclusive_scan(TransformIterator(polygons.begin(), NumVert),
                           TransformIterator(polygons.end(), NumVert),
                           loopOffsets.begin(), (size_t)0);

  return loopOffsets;
}

std::vector<std::pair<int, int>> BuildCollider(
    const Polygons& polygons, const std::vector<size_t>& loopOffsets,
    double radius, bool invert) {
  Vec<EdgeBox> edgeBoxes;
  Vec<Box> filletBoxes;
  Vec<int> filletBoxEdges;

  for (size_t loopIdx = 0; loopIdx < polygons.size(); ++loopIdx) {
    const SimplePolygon& loop = polygons[loopIdx];

    for (size_t edgeIdx = 0; edgeIdx < loop.size(); ++edgeIdx) {
      const int globalEdge = static_cast<int>(loopOffsets[loopIdx] + edgeIdx);
      const vec2 p0 = loop[edgeIdx];
      const vec2 p1 = loop[(edgeIdx + 1) % loop.size()];
      const vec2 p2 = loop[(edgeIdx + 2) % loop.size()];
      const vec2 edge = p1 - p0;

      Box edgeBox(ToVec3(p0), ToVec3(p1));
      edgeBoxes.push_back(
          EdgeBox{edgeBox, Collider::MortonCode(ToVec3((p0 + p1) / 2), edgeBox),
                  globalEdge});

      const vec2 edgeDir = normalize(edge);
      const vec2 normal = EdgeNormal(edgeDir, invert);

      Box filletBox = edgeBox;
      filletBox.Union(ToVec3(p0 + normal * 2.0 * radius));
      filletBox.Union(ToVec3(p1 + normal * 2.0 * radius));
      filletBox.Union(ToVec3(p0 - edgeDir * radius + normal * radius));
      filletBox.Union(ToVec3(p1 + edgeDir * radius + normal * radius));

      if (!IsConvexTurn(p0, p1, p2)) {
        const vec2 nextEdgeDir = normalize(p2 - p1);
        const vec2 nextNormal = EdgeNormal(nextEdgeDir, invert);
        filletBox.Union(ToVec3(p1 + nextNormal * 2.0 * radius));
        filletBox.Union(ToVec3(p1 + nextEdgeDir * radius + nextNormal * radius));
      }

      filletBoxes.push_back(filletBox);
      filletBoxEdges.push_back(globalEdge);
    }
  }

  std::stable_sort(edgeBoxes.begin(), edgeBoxes.end(), EdgeBoxMortonLess());

  Vec<Box> boxes(edgeBoxes.size());
  Vec<uint32_t> morton(edgeBoxes.size());
  for (size_t i = 0; i < edgeBoxes.size(); ++i) {
    boxes[i] = edgeBoxes[i].box;
    morton[i] = edgeBoxes[i].morton;
  }

  Collider collider(boxes, morton);
  tbb::enumerable_thread_specific<std::vector<std::pair<int, int>>> localPairs;
  CollisionRecorder recordCollision{filletBoxEdges, edgeBoxes, localPairs};
  auto recorder = MakeSimpleRecorder(recordCollision);
  collider.Collisions(recorder, filletBoxes.cview());

  std::vector<std::pair<int, int>> pairs;
  size_t total = 0;
  for (auto& local : localPairs) total += local.size();
  pairs.reserve(total);
  for (auto& local : localPairs) {
    pairs.insert(pairs.end(), local.begin(), local.end());
  }

  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  return pairs;
}

bool FinishFilletGeometry(const FilletInput& input, int firstEdge, int secondEdge,
                          double firstT, double secondT, vec2 center,
                          FilletGeometry& geometry) {
  int fromEdge = firstEdge;
  int toEdge = secondEdge;
  double fromT = firstT;
  double toT = secondT;
  vec2 fromPoint = EdgeStart(input, firstEdge) +
                   firstT * (EdgeEnd(input, firstEdge) - EdgeStart(input, firstEdge));
  vec2 toPoint = EdgeStart(input, secondEdge) +
                 secondT * (EdgeEnd(input, secondEdge) - EdgeStart(input, secondEdge));
  vec2 fromRadius = fromPoint - center;
  vec2 toRadius = toPoint - center;

  const double radiusCross = la::cross(fromRadius, toRadius);
  if ((!input.invert && radiusCross < 0) ||
      (input.invert && radiusCross > 0)) {
    std::swap(fromEdge, toEdge);
    std::swap(fromT, toT);
    std::swap(fromPoint, toPoint);
    std::swap(fromRadius, toRadius);
  }

  const double startRad = NormalizeAngle(ToRad(fromRadius));
  const double endRad = NormalizeAngle(ToRad(toRadius));
  if (!(NormalizeAngle(endRad - startRad) > input.minArcAngle)) return false;

  if (toT == 1) {
    toEdge = NextEdge(input, toEdge);
    toT = 0;
  }

  geometry = FilletGeometry{fromEdge,
                            toEdge,
                            fromT,
                            toT,
                            fromPoint,
                            toPoint,
                            center,
                            startRad,
                            endRad};
  return true;
}

std::vector<FilletGeometry> BuildFilletGeometries(const FilletInput& input,
                                                  int firstEdge, int secondEdge,
                                                  double radius) {
  std::vector<FilletGeometry> geometries;
  if (firstEdge == secondEdge) return geometries;

  const vec2 firstStart = EdgeStart(input, firstEdge);
  const vec2 firstEnd = EdgeEnd(input, firstEdge);
  const vec2 firstNextEnd = EdgeEnd(input, NextEdge(input, firstEdge));
  const vec2 secondStart = EdgeStart(input, secondEdge);
  const vec2 secondEnd = EdgeEnd(input, secondEdge);
  const vec2 secondNextEnd = EdgeEnd(input, NextEdge(input, secondEdge));
  const vec2 firstVec = firstEnd - firstStart;
  const vec2 secondVec = secondEnd - secondStart;

  if (length2(firstVec) == 0 || length2(secondVec) == 0) return geometries;

  const vec2 firstNormal = EdgeNormal(firstVec, input.invert);
  const vec2 secondNormal = EdgeNormal(secondVec, input.invert);
  const vec2 firstNextNormal = EdgeNormal(firstNextEnd - firstEnd, input.invert);
  const vec2 secondNextNormal =
      EdgeNormal(secondNextEnd - secondEnd, input.invert);
  const vec2 firstOffset = firstStart + firstNormal * radius;
  const vec2 secondOffset = secondStart + secondNormal * radius;
  const bool firstConvex =
      IsConvexTurn(firstStart, firstEnd, firstNextEnd) != input.invert;
  const bool secondConvex =
      IsConvexTurn(secondStart, secondEnd, secondNextEnd) != input.invert;

  FilletGeometry geometry;
  vec2 center;
  double firstT = 0;
  double secondT = 0;
  if (IntersectLines(firstOffset, firstVec, secondOffset, secondVec, center,
                     firstT, secondT) &&
      firstT > 0 && firstT < 1 && secondT > 0 && secondT < 1 &&
      FinishFilletGeometry(input, firstEdge, secondEdge, firstT, secondT,
                           center, geometry)) {
    geometries.push_back(geometry);
  }

  if (!secondConvex) {
    const IntersectResult ev = IntersectSegmentArc(
        firstOffset, firstOffset + firstVec, secondEnd, radius, secondNormal,
        secondNextNormal, input.invert);

    for (int i = 0; i < ev.count; ++i) {
      const vec2 candidate = ev.points[i];
      const double t = dot(candidate - firstOffset, firstVec) / dot(firstVec, firstVec);
      if (t > 0 && t < 1 &&
          FinishFilletGeometry(input, firstEdge, secondEdge, t, 1, candidate,
                               geometry)) {
        geometries.push_back(geometry);
      }
    }
  }

  if (!firstConvex) {
    const IntersectResult ve = IntersectSegmentArc(
        secondOffset, secondOffset + secondVec, firstEnd, radius, firstNormal,
        firstNextNormal, input.invert);

    for (int i = 0; i < ve.count; ++i) {
      const vec2 candidate = ve.points[i];
      const double t =
          dot(candidate - secondOffset, secondVec) / dot(secondVec, secondVec);
      if (t > 0 && t < 1 &&
          FinishFilletGeometry(input, firstEdge, secondEdge, 1, t, candidate,
                               geometry)) {
        geometries.push_back(geometry);
      }
    }
  }

  if (!firstConvex && !secondConvex) {
    const IntersectResult vv =
        IntersectArcArc(firstEnd, firstNormal, firstNextNormal, input.invert,
                        secondEnd, secondNormal, secondNextNormal, input.invert,
                        radius);

    for (int i = 0; i < vv.count; ++i) {
      if (FinishFilletGeometry(input, firstEdge, secondEdge, 1, 1, vv.points[i],
                               geometry)) {
        geometries.push_back(geometry);
      }
    }
  }

  return geometries;
}

bool RadiusFits(const FilletInput& input, ivec3 edgeIndices, double radius) {
  std::array<int, 3> edges{edgeIndices[0], edgeIndices[1], edgeIndices[2]};
  std::sort(edges.begin(), edges.end());

  std::array<vec2, 3> starts;
  std::array<vec2, 3> dirs;
  for (int i = 0; i < 3; ++i) {
    starts[i] = EdgeStart(input, edges[i]);
    dirs[i] = EdgeEnd(input, edges[i]) - starts[i];
    if (length2(dirs[i]) == 0) return false;
  }

  vec2 p01;
  vec2 p12;
  vec2 p20;
  double t = 0;
  double u = 0;
  if (!IntersectLines(starts[0], dirs[0], starts[1], dirs[1], p01, t, u)) {
    return false;
  }
  if (!IntersectLines(starts[1], dirs[1], starts[2], dirs[2], p12, t, u)) {
    return false;
  }
  if (!IntersectLines(starts[2], dirs[2], starts[0], dirs[0], p20, t, u)) {
    return false;
  }

  const double doubleArea = la::cross(p12 - p01, p20 - p01);
  if (!(doubleArea > 0)) return false;

  const double a = length(p12 - p01);
  const double b = length(p20 - p12);
  const double c = length(p01 - p20);
  const double perimeter = a + b + c;
  if (!(perimeter > 0)) return false;

  const double area = doubleArea * 0.5;
  const double inradius = area / (perimeter * 0.5);
  return inradius > radius;
}

std::vector<FilletGeometry> FindFilletPairs(
    const FilletInput& input, const std::vector<std::pair<int, int>>& broadPairs,
    double radius) {
  struct Candidate {
    std::pair<int, int> pair;
    FilletGeometry geometry;
  };

  std::vector<Candidate> candidates;
  candidates.reserve(broadPairs.size());

  for (const auto& pair : broadPairs) {
    for (const FilletGeometry& geometry :
         BuildFilletGeometries(input, pair.first, pair.second, radius)) {
      candidates.push_back(Candidate{{geometry.fromEdge, geometry.toEdge},
                                     geometry});
    }
  }

  std::vector<std::vector<size_t>> pairsByFirst(input.edges.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    pairsByFirst[candidates[i].geometry.fromEdge].push_back(i);
  }

  std::vector<FilletGeometry> fillets;
  fillets.reserve(candidates.size());

  for (const Candidate& candidate : candidates) {
    const FilletGeometry& geometry = candidate.geometry;

    bool fits = true;
    for (const size_t otherIndex : pairsByFirst[geometry.fromEdge]) {
      const FilletGeometry& other = candidates[otherIndex].geometry;
      if (other.toEdge == geometry.toEdge) continue;
      if (!RadiusFits(input, ivec3(geometry.fromEdge, geometry.toEdge,
                                   other.toEdge),
                      radius)) {
        fits = false;
        break;
      }
    }

    if (fits) fillets.push_back(candidate.geometry);
  }

  return fillets;
}

std::vector<vec2> discreteArcToPoint(const FilletGeometry& arc, double radius,
                                     int circularSegments) {
  std::vector<vec2> points;
  points.push_back(arc.fromPoint);

  const double totalRad = NormalizeAngle(arc.endRad - arc.startRad);
  const double dPhi = 2.0 * M_PI / circularSegments;
  const int segments = int(totalRad / dPhi) + 1;

  for (int i = 1; i < segments; ++i) {
    const double current = arc.startRad + dPhi * i;
    points.push_back({arc.center.x + radius * cos(current),
                      arc.center.y + radius * sin(current)});
  }

  points.push_back(arc.toPoint);
  return points;
}

void AppendArcPoints(SimplePolygon& loop, const FilletGeometry& arc,
                     double radius, int circularSegments) {
  const std::vector<vec2> points = discreteArcToPoint(arc, radius, circularSegments);
  const double pointTolerance = std::max(radius * 1e-8, 1e-12);
  for (size_t i = 1; i < points.size(); ++i) {
    AppendPoint(loop, points[i], pointTolerance);
  }
}

bool BuildSegments(const FilletInput& input,
                   const std::vector<FilletGeometry>& geometries,
                   std::vector<Segment>& segments,
                   std::vector<int>& segmentFromStartFillet) {
  std::vector<std::vector<Tangent>> starts(input.edges.size());
  std::vector<std::vector<Tangent>> ends(input.edges.size());

  for (size_t i = 0; i < geometries.size(); ++i) {
    const FilletGeometry& geometry = geometries[i];
    ends[geometry.fromEdge].push_back(
        Tangent{geometry.fromT, i, geometry.fromPoint});
    starts[geometry.toEdge].push_back(Tangent{geometry.toT, i, geometry.toPoint});
  }

  segmentFromStartFillet.assign(geometries.size(), -1);

  for (size_t edge = 0; edge < input.edges.size(); ++edge) {
    auto& edgeStarts = starts[edge];
    auto& edgeEnds = ends[edge];

    if (edgeStarts.size() != edgeEnds.size()) return false;
    if (edgeStarts.empty()) continue;

    std::sort(edgeStarts.begin(), edgeStarts.end(), TangentParameterLess());
    std::sort(edgeEnds.begin(), edgeEnds.end(), TangentParameterLess());

    for (size_t i = 0; i < edgeStarts.size(); ++i) {
      const Tangent& start = edgeStarts[i];
      const Tangent& end = edgeEnds[i];
      if (!(end.t > start.t)) return false;

      const int segmentIndex = static_cast<int>(segments.size());
      segmentFromStartFillet[start.fillet] = segmentIndex;
      segments.push_back(Segment{static_cast<int>(edge),
                                 start.t,
                                 end.t,
                                 start.point,
                                 end.point,
                                 end.fillet,
                                 false});
    }
  }

  return true;
}

std::vector<CrossSection> TraceFillets(const FilletInput& input,
                                       const std::vector<FilletGeometry>& geometries,
                                       double radius, int circularSegments) {
  if (geometries.empty()) return {};

  std::vector<Segment> segments;
  std::vector<int> segmentFromStartFillet;
  if (!BuildSegments(input, geometries, segments, segmentFromStartFillet)) {
    return {};
  }

  if (segments.empty()) return {};

  Polygons resultLoops;
  std::vector<uint8_t> touchedLoop(input.loops.size(), 0);
  const size_t maxSteps = segments.size() + geometries.size() + input.edges.size();
  const double pointTolerance = std::max(radius * 1e-8, 1e-12);

  for (size_t startSegment = 0; startSegment < segments.size(); ++startSegment) {
    if (segments[startSegment].used) continue;

    SimplePolygon loop;
    size_t current = startSegment;
    bool closed = false;

    for (size_t step = 0; step <= maxSteps; ++step) {
      Segment& segment = segments[current];
      if (segment.used) {
        if (current != startSegment) return {};
        closed = true;
        break;
      }

      segment.used = true;
      touchedLoop[input.edges[segment.edge].loop] = 1;

      AppendPoint(loop, segment.startPoint, pointTolerance);
      AppendPoint(loop, segment.endPoint, pointTolerance);

      const FilletGeometry& fillet = geometries[segment.endFillet];
      AppendArcPoints(loop, fillet, radius, circularSegments);
      touchedLoop[input.edges[fillet.toEdge].loop] = 1;

      const int next = segmentFromStartFillet[segment.endFillet];
      if (next < 0) return {};

      current = static_cast<size_t>(next);
    }

    if (!closed || loop.size() < 3) return {};
    if (loop.size() > 1 &&
        (SamePoint(loop.front(), loop.back()) ||
         SamePointWithin(loop.front(), loop.back(), pointTolerance))) {
      loop.pop_back();
    }
    resultLoops.push_back(loop);
  }

  for (const Segment& segment : segments) {
    if (!segment.used) return {};
  }

  for (size_t loopIdx = 0; loopIdx < input.loops.size(); ++loopIdx) {
    if (!touchedLoop[loopIdx]) resultLoops.push_back(input.loops[loopIdx]);
  }

  if (resultLoops.empty()) return {};

  return CrossSection(resultLoops).Decompose();
}

bool HasFilletAngles(const CrossSection& crossSection, int circularSegments) {
  const double dPhi = 2.0 * M_PI / circularSegments;
  for (const SimplePolygon& loop : crossSection.ToPolygons()) {
    if (loop.size() < 3) return false;
    for (size_t i = 0; i < loop.size(); ++i) {
      const vec2 p1 = loop[i];
      const vec2 p2 = loop[(i + 1) % loop.size()];
      const vec2 p3 = loop[(i + 2) % loop.size()];
      const vec2 e1 = p2 - p1;
      const vec2 e2 = p3 - p2;
      if (length2(e1) == 0 || length2(e2) == 0) return false;

      const double angle = NormalizeAngle(ToRad(e2) - ToRad(e1));
      if (!(angle > M_PI || angle < dPhi + dPhi * 0.1)) return false;
    }
  }
  return true;
}

bool HasValidFilletGeometry(const CrossSection& crossSection) {
  for (const SimplePolygon& loop : crossSection.ToPolygons()) {
    if (loop.size() < 3) return false;
    for (size_t i = 0; i < loop.size(); ++i) {
      const vec2 p1 = loop[i];
      const vec2 p2 = loop[(i + 1) % loop.size()];
      if (!IsInFilletRange(p1) || length2(p2 - p1) == 0) return false;
    }
  }
  return true;
}

bool IsValidFilletResult(const std::vector<CrossSection>& result,
                         const CrossSection& original, int circularSegments,
                         bool invert) {
  if (result.empty()) return false;
  for (const CrossSection& crossSection : result) {
    if (!HasValidFilletGeometry(crossSection)) return false;
    if (!invert && !HasFilletAngles(crossSection, circularSegments)) return false;
  }

  const CrossSection composed = CrossSection::Compose(result);
  if (composed.IsEmpty()) return false;
  if (!HasValidFilletGeometry(composed)) return false;
  if (invert) {
    if (!(composed.Area() > original.Area())) return false;
  } else {
    if (!(composed.Area() < original.Area())) return false;
  }
  return invert || HasFilletAngles(composed, circularSegments);
}

#ifdef MANIFOLD_DEBUG
void SavePolygons(const std::string& filename, const Polygons& polygons) {
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

  if (ManifoldParams().verbose) {
    std::cout << "Successfully saved " << polygons.size() << " tests to "
              << filename << std::endl;
  }
}

void SaveCrossSection(std::ofstream& outFile,
                      const std::vector<CrossSection>& result) {
  outFile << result.size() << "\n";

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

  if (ManifoldParams().verbose) {
    std::cout << "Successfully saved " << result.size() << " CrossSections. "
              << std::endl;
  }
}
#endif

std::vector<CrossSection> FilletImpl(const Polygons& polygons, double radius,
                                     int circularSegments) {
  if (polygons.empty() || radius == 0) return {};

  bool invert = false;
  if (radius < 0) {
    invert = true;
    radius = std::abs(radius);
  }

  for (const SimplePolygon& loop : polygons) {
    if (loop.size() < 3) return {};
    for (size_t i = 0; i < loop.size(); ++i) {
      const vec2& point = loop[i];
      if (!IsInFilletRange(point)) return {};
      if (length2(loop[(i + 1) % loop.size()] - point) == 0) return {};
    }
  }

  const int segments =
      circularSegments > 2 ? circularSegments : Quality::GetCircularSegments(radius);

  const std::vector<size_t> loopOffsets = BuildLoopOffsets(polygons);
  const std::vector<EdgeRef> edges = BuildEdges(polygons, loopOffsets);
  const FilletInput input{polygons, loopOffsets, edges, invert,
                          2.0 * M_PI / segments};

#ifdef MANIFOLD_DEBUG
  if (caseIndex == 0) SavePolygons("Testing/Fillet/input.txt", polygons);

  resultOutputFile.open("Testing/Fillet/" + std::to_string(caseIndex) + ".txt");
  if (!resultOutputFile.is_open()) {
    std::cerr << "Error: Could not open file "
              << "Testing/Fillet/" + std::to_string(caseIndex) + ".txt"
              << " for writing." << std::endl;
    throw std::exception();
  }
  resultOutputFile << std::setprecision(std::numeric_limits<double>::max_digits10)
                   << radius << "\n";
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

  const std::vector<std::pair<int, int>> broadPairs =
      BuildCollider(polygons, loopOffsets, radius, invert);
  const std::vector<FilletGeometry> geometries =
      FindFilletPairs(input, broadPairs, radius);

  const CrossSection original(polygons);
  auto result = TraceFillets(input, geometries, radius, segments);
  if (!IsValidFilletResult(result, original, segments, invert)) result.clear();

  if (result.empty() && segments > 3) {
    const double firstOffset = invert ? radius : -radius;
    const CrossSection opened =
        original.Offset(firstOffset, CrossSection::JoinType::Round, 2.0, segments)
            .Offset(-firstOffset, CrossSection::JoinType::Round, 2.0, segments);
    const std::vector<CrossSection> fallback = opened.Decompose();
    if (IsValidFilletResult(fallback, original, segments, invert)) result = fallback;
  }

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
  try {
    return FilletImpl(ToPolygons(), radius, circularSegments);
  } catch (const std::exception&) {
    return {};
  }
}

}  // namespace manifold

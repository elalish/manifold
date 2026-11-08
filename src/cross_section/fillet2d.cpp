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
#include <tuple>
#include <type_traits>
#include <vector>

#include "../collider.h"
#include "../parallel.h"
#include "../vec.h"
#include "manifold/cross_section.h"
#include "manifold/manifold.h"

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
  const std::vector<int>& nextEdges;
  const std::vector<int>& prevEdges;
  bool invert;
  double minArcAngle;
};

// What a fillet ball touches: the interior of an edge (its line), or a reflex
// vertex, referenced as the start of the edge that follows it.
struct Contact {
  int edge;
  bool isVertex;
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
  // Order of tangent points at the start vertex of their edge (t == 0); see
  // PivotKey.
  double fromKey;
  double toKey;
  Contact fromContact;
  Contact toContact;
};

struct IntersectResult {
  int count = 0;
  std::array<vec2, 2> points{vec2(), vec2()};
  // For IntersectSegmentArc, the parameter of each point along the segment.
  std::array<double, 2> params{0, 0};
  // The two solutions coincide: the ball fits exactly through a neck and
  // carries an arc on each side of it.
  bool tangent = false;
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

struct PairRecorder {
  const Vec<int>& leafEdge;
  std::vector<std::pair<int, int>>& pairs;

  void operator()(int query, int leaf) const {
    pairs.push_back({query, leafEdge[leaf]});
  }
};

struct EdgeIndex {
  Collider collider;
  Vec<int> leafEdge;

  // Returns (query index, global edge) for every query box overlapping an edge
  // box.
  std::vector<std::pair<int, int>> Query(const Vec<Box>& queries) const {
    std::vector<std::pair<int, int>> pairs;
    PairRecorder record{leafEdge, pairs};
    auto recorder = MakeSimpleRecorder(record);
    collider.Collisions(recorder, queries.cview(), false);
    return pairs;
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

// Angle swept by a fillet arc from startRad to endRad. Arcs run CCW around the
// ball for a positive radius and CW for a negative one (invert).
double ArcAngle(double startRad, double endRad, bool invert) {
  return NormalizeAngle(invert ? startRad - endRad : endRad - startRad);
}

bool IsFinite(const vec2& v) {
  return std::isfinite(v.x) && std::isfinite(v.y);
}

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
  if (loop.empty() || (!SamePoint(loop.back(), point) &&
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
  return input.nextEdges[edge];
}

vec2 EdgeNormal(const vec2& edge, bool invert) {
  return normalize(!invert ? vec2(-edge.y, edge.x) : vec2(edge.y, -edge.x));
}

double SignedArea(const SimplePolygon& loop) {
  double area = 0;
  for (size_t i = 0; i < loop.size(); ++i)
    area += la::cross(loop[i], loop[(i + 1) % loop.size()]);
  return area * 0.5;
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

// Strictly inside the sector: a center on the boundary is tangent to the
// adjacent edge at the vertex, which is the edge-edge case, not a vertex pivot.
bool IsVectorInSector(vec2 v, vec2 start, vec2 end, bool ccw) {
  if (!ccw) return IsVectorInSector(v, end, start, true);

  const double crossStartEnd = la::cross(start, end);
  const double crossStartV = la::cross(start, v);
  const double crossVEnd = la::cross(v, end);

  if (crossStartEnd >= 0) return crossStartV > 0 && crossVEnd > 0;
  return crossStartV > 0 || crossVEnd > 0;
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
  // Coincident when they round to the same parameter, even if tOffset is not
  // exactly zero.
  const int numValues = values[0] == values[1] ? 1 : 2;

  IntersectResult result;
  for (int i = 0; i < numValues; ++i) {
    const double t = values[i];
    if (!(t >= 0 && t <= 1)) continue;

    const vec2 point = p + edge * t;
    if (IsVectorInSector(point - center, arcStart, arcEnd, arcCCW)) {
      result.params[result.count] = t;
      result.points[result.count++] = point;
    }
  }
  result.tangent = numValues == 1 && result.count == 1;

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
  const int numPoints = SamePoint(points[0], points[1]) ? 1 : 2;
  for (int i = 0; i < numPoints; ++i) {
    const vec2 point = points[i];
    if (IsVectorInSector(point - c1, arc1Start, arc1End, arc1CCW) &&
        IsVectorInSector(point - c2, arc2Start, arc2End, arc2CCW)) {
      result.points[result.count++] = point;
    }
  }
  result.tangent = numPoints == 1 && result.count == 1;

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

// The edge the boundary continues on past the end of each edge. Normally the
// next edge of the same loop, but where several edges start at the same point
// (a pinched loop, or loops touching at a vertex) it is chosen geometrically:
// the first outgoing edge clockwise from the reversed incoming edge, so the
// material (on the left) stays on the left. With invert the traced region is
// the complement, so the first counter-clockwise edge is taken instead.
std::vector<int> BuildNextEdges(const Polygons& polygons,
                                const std::vector<size_t>& loopOffsets,
                                const std::vector<EdgeRef>& edges,
                                bool invert) {
  auto start = [&](int e) { return polygons[edges[e].loop][edges[e].edge]; };
  auto end = [&](int e) {
    const SimplePolygon& loop = polygons[edges[e].loop];
    return loop[(edges[e].edge + 1) % loop.size()];
  };

  std::vector<int> byStart(edges.size());
  for (size_t i = 0; i < edges.size(); ++i) byStart[i] = static_cast<int>(i);
  auto pointLess = [](const vec2& a, const vec2& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  };
  std::sort(byStart.begin(), byStart.end(),
            [&](int a, int b) { return pointLess(start(a), start(b)); });

  std::vector<int> nextEdges(edges.size());
  for (size_t e = 0; e < edges.size(); ++e) {
    const EdgeRef ref = edges[e];
    nextEdges[e] = static_cast<int>(loopOffsets[ref.loop] +
                                    (ref.edge + 1) % polygons[ref.loop].size());

    const vec2 vertex = end(e);
    auto range = std::equal_range(
        byStart.begin(), byStart.end(), vertex,
        [&](const auto& a, const auto& b) {
          if constexpr (std::is_same_v<std::decay_t<decltype(a)>, vec2>) {
            return pointLess(a, start(b));
          } else {
            return pointLess(start(a), b);
          }
        });
    if (range.second - range.first < 2) continue;

    const double back = ToRad(start(e) - vertex);
    double best = std::numeric_limits<double>::infinity();
    for (auto it = range.first; it != range.second; ++it) {
      const double out = ToRad(end(*it) - vertex);
      double turn = NormalizeAngle(invert ? out - back : back - out);
      if (turn == 0) turn = 2 * M_PI;
      if (turn < best) {
        best = turn;
        nextEdges[e] = *it;
      }
    }
  }
  return nextEdges;
}

std::vector<size_t> BuildLoopOffsets(const Polygons& polygons) {
  std::vector<size_t> loopOffsets(polygons.size());

  manifold::exclusive_scan(TransformIterator(polygons.begin(), NumVert),
                           TransformIterator(polygons.end(), NumVert),
                           loopOffsets.begin(), (size_t)0);

  return loopOffsets;
}

EdgeIndex BuildEdgeIndex(const Polygons& polygons) {
  Box bounds;
  for (const SimplePolygon& loop : polygons)
    for (const vec2& v : loop) bounds.Union(ToVec3(v));

  std::vector<EdgeBox> edgeBoxes;
  int globalEdge = 0;
  for (const SimplePolygon& loop : polygons) {
    for (size_t edgeIdx = 0; edgeIdx < loop.size(); ++edgeIdx, ++globalEdge) {
      const vec2 p0 = loop[edgeIdx];
      const vec2 p1 = loop[(edgeIdx + 1) % loop.size()];
      const Box edgeBox(ToVec3(p0), ToVec3(p1));
      edgeBoxes.push_back(
          EdgeBox{edgeBox, Collider::MortonCode(ToVec3((p0 + p1) / 2), bounds),
                  globalEdge});
    }
  }

  std::stable_sort(edgeBoxes.begin(), edgeBoxes.end(), EdgeBoxMortonLess());

  Vec<Box> boxes(edgeBoxes.size());
  Vec<uint32_t> morton(edgeBoxes.size());
  EdgeIndex index;
  index.leafEdge.resize(edgeBoxes.size());
  for (size_t i = 0; i < edgeBoxes.size(); ++i) {
    boxes[i] = edgeBoxes[i].box;
    morton[i] = edgeBoxes[i].morton;
    index.leafEdge[i] = edgeBoxes[i].edge;
  }
  index.collider = Collider(boxes, morton);
  return index;
}

std::vector<std::pair<int, int>> BuildCollider(const FilletInput& input,
                                               const EdgeIndex& index,
                                               double radius) {
  const Polygons& polygons = input.loops;
  const bool invert = input.invert;
  Vec<Box> filletBoxes;
  Vec<int> filletBoxEdges;

  int globalEdge = 0;
  for (const SimplePolygon& loop : polygons) {
    for (size_t edgeIdx = 0; edgeIdx < loop.size(); ++edgeIdx, ++globalEdge) {
      const vec2 p0 = loop[edgeIdx];
      const vec2 p1 = loop[(edgeIdx + 1) % loop.size()];
      const vec2 p2 = EdgeEnd(input, NextEdge(input, globalEdge));
      const vec2 edge = p1 - p0;

      const vec2 normal = EdgeNormal(edge, invert);

      // Every ball tangent to this edge has its center on the edge offset by
      // radius, and touches other edges within radius of that center.
      const vec2 r(radius, radius);
      Box filletBox(ToVec3(p0 + normal * radius), ToVec3(p1 + normal * radius));
      filletBox.min -= ToVec3(r);
      filletBox.max += ToVec3(r);

      // At a reflex end vertex balls also pivot around it, centers within
      // radius of the vertex.
      if (IsConvexTurn(p0, p1, p2) == invert) {
        filletBox.Union(ToVec3(p1 - 2.0 * r));
        filletBox.Union(ToVec3(p1 + 2.0 * r));
      }

      filletBoxes.push_back(filletBox);
      filletBoxEdges.push_back(globalEdge);
    }
  }

  // A ball touching edges a and b lies in both of their fillet boxes, so each
  // unordered pair is found from both sides; keep only one direction.
  std::vector<std::pair<int, int>> pairs;
  for (auto [query, b] : index.Query(filletBoxes)) {
    const int a = filletBoxEdges[query];
    if (a < b) pairs.push_back({a, b});
  }
  return pairs;
}

// A ball touching the start vertex V of an edge pivots around V, its center
// sweeping from the previous edge's normal towards this edge's normal. Several
// balls touching V are met in that order while tracing, so a tangent point at
// t == 0 is ordered by how far its center has swept (0 = just arrived from the
// previous edge). Tangent points with t > 0 do not need a key.
double PivotKey(const FilletInput& input, int edge, double t, vec2 center) {
  if (t != 0) return 0;
  const int prev = input.prevEdges[edge];
  const vec2 prevNormal =
      EdgeNormal(EdgeEnd(input, prev) - EdgeStart(input, prev), input.invert);
  const double from = ToRad(prevNormal);
  const double to = ToRad(center - EdgeStart(input, edge));
  // Without invert the boundary turns right around a reflex vertex, so the
  // center sweeps clockwise.
  return NormalizeAngle(input.invert ? to - from : from - to);
}

// The point at parameter t along edge e, exactly its start or end vertex at
// t = 0 or 1.
vec2 PointOnEdge(const FilletInput& input, int e, double t) {
  if (t == 0) return EdgeStart(input, e);
  if (t == 1) return EdgeEnd(input, e);
  return EdgeStart(input, e) + t * (EdgeEnd(input, e) - EdgeStart(input, e));
}

bool FinishFilletGeometry(const FilletInput& input, int firstEdge,
                          int secondEdge, double firstT, double secondT,
                          bool firstIsVertex, bool secondIsVertex, vec2 center,
                          FilletGeometry& geometry, bool reverse = false) {
  int fromEdge = firstEdge;
  int toEdge = secondEdge;
  // A vertex contact is always the end vertex of the given edge.
  Contact fromContact = firstIsVertex
                            ? Contact{NextEdge(input, firstEdge), true}
                            : Contact{firstEdge, false};
  Contact toContact = secondIsVertex
                          ? Contact{NextEdge(input, secondEdge), true}
                          : Contact{secondEdge, false};
  double fromT = firstT;
  double toT = secondT;
  vec2 fromPoint = PointOnEdge(input, firstEdge, firstT);
  vec2 toPoint = PointOnEdge(input, secondEdge, secondT);
  vec2 fromRadius = fromPoint - center;
  vec2 toRadius = toPoint - center;

  // Arcs run CCW around the ball (CW with invert). `reverse` asks for the
  // other arc of a ball through a neck (see IntersectResult::tangent).
  const double radiusCross = la::cross(fromRadius, toRadius);
  const bool backwards = input.invert ? radiusCross > 0 : radiusCross < 0;
  if (backwards != reverse) {
    std::swap(fromEdge, toEdge);
    std::swap(fromContact, toContact);
    std::swap(fromT, toT);
    std::swap(fromPoint, toPoint);
    std::swap(fromRadius, toRadius);
  }

  const double startRad = NormalizeAngle(ToRad(fromRadius));
  const double endRad = NormalizeAngle(ToRad(toRadius));
  // A tangent point on a vertex is always referenced as the start of the
  // following edge, so arrivals and departures there match during tracing.
  if (fromT == 1) {
    fromEdge = NextEdge(input, fromEdge);
    fromT = 0;
  }
  if (toT == 1) {
    toEdge = NextEdge(input, toEdge);
    toT = 0;
  }

  // An arc too flat to discretize may be skipped only at a plain corner: both
  // tangent points inside two adjacent edges, so the arc angle equals the
  // corner's turn and tracing still reaches the next edge via the vertex. Any
  // other arc spans more of the boundary and must be kept.
  const bool corner = fromT > 0 && toEdge == NextEdge(input, fromEdge);
  const double minArc = corner ? input.minArcAngle : 0;
  if (!(ArcAngle(startRad, endRad, input.invert) > minArc)) return false;

  geometry = FilletGeometry{fromEdge,
                            toEdge,
                            fromT,
                            toT,
                            fromPoint,
                            toPoint,
                            center,
                            startRad,
                            endRad,
                            PivotKey(input, fromEdge, fromT, center),
                            PivotKey(input, toEdge, toT, center),
                            fromContact,
                            toContact};
  return true;
}

std::vector<FilletGeometry> BuildFilletGeometries(const FilletInput& input,
                                                  int firstEdge, int secondEdge,
                                                  double radius) {
  std::vector<FilletGeometry> geometries;

  const vec2 firstStart = EdgeStart(input, firstEdge);
  const vec2 firstEnd = EdgeEnd(input, firstEdge);
  const vec2 firstNextEnd = EdgeEnd(input, NextEdge(input, firstEdge));
  const vec2 secondStart = EdgeStart(input, secondEdge);
  const vec2 secondEnd = EdgeEnd(input, secondEdge);
  const vec2 secondNextEnd = EdgeEnd(input, NextEdge(input, secondEdge));
  const vec2 firstVec = firstEnd - firstStart;
  const vec2 secondVec = secondEnd - secondStart;

  const vec2 firstNormal = EdgeNormal(firstVec, input.invert);
  const vec2 secondNormal = EdgeNormal(secondVec, input.invert);
  const vec2 firstNextNormal =
      EdgeNormal(firstNextEnd - firstEnd, input.invert);
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
      firstT >= 0 && firstT <= 1 && secondT >= 0 && secondT <= 1 &&
      FinishFilletGeometry(input, firstEdge, secondEdge, firstT, secondT, false,
                           false, center, geometry)) {
    geometries.push_back(geometry);
  }

  // A vertex on the line of the other edge can only be touched at the tangent
  // point itself, a zero-length arc; skip vertex-line pairs where the vertex
  // is an endpoint of that edge.
  auto onEdge = [&](vec2 v, vec2 a, vec2 b) {
    return SamePoint(v, a) || SamePoint(v, b);
  };

  if (!secondConvex && !onEdge(secondEnd, firstStart, firstEnd)) {
    const IntersectResult ev = IntersectSegmentArc(
        firstOffset, firstOffset + firstVec, secondEnd, radius, secondNormal,
        secondNextNormal, input.invert);

    for (int i = 0; i < ev.count; ++i) {
      for (const bool reverse : {false, true}) {
        if (reverse && !ev.tangent) break;
        if (FinishFilletGeometry(input, firstEdge, secondEdge, ev.params[i], 1,
                                 false, true, ev.points[i], geometry, reverse))
          geometries.push_back(geometry);
      }
    }
  }

  if (!firstConvex && !onEdge(firstEnd, secondStart, secondEnd)) {
    const IntersectResult ve =
        IntersectSegmentArc(secondOffset, secondOffset + secondVec, firstEnd,
                            radius, firstNormal, firstNextNormal, input.invert);

    for (int i = 0; i < ve.count; ++i) {
      for (const bool reverse : {false, true}) {
        if (reverse && !ve.tangent) break;
        if (FinishFilletGeometry(input, firstEdge, secondEdge, 1, ve.params[i],
                                 true, false, ve.points[i], geometry, reverse))
          geometries.push_back(geometry);
      }
    }
  }

  if (!firstConvex && !secondConvex) {
    const IntersectResult vv = IntersectArcArc(
        firstEnd, firstNormal, firstNextNormal, input.invert, secondEnd,
        secondNormal, secondNextNormal, input.invert, radius);

    for (int i = 0; i < vv.count; ++i) {
      for (const bool reverse : {false, true}) {
        if (reverse && !vv.tangent) break;
        if (FinishFilletGeometry(input, firstEdge, secondEdge, 1, 1, true, true,
                                 vv.points[i], geometry, reverse))
          geometries.push_back(geometry);
      }
    }
  }

  return geometries;
}

vec2 LineNormal(const FilletInput& input, int e) {
  return EdgeNormal(EdgeEnd(input, e) - EdgeStart(input, e), input.invert);
}

bool SameContact(const Contact& a, const Contact& b) {
  return a.edge == b.edge && a.isVertex == b.isVertex;
}

bool ContactLess(const Contact& a, const Contact& b) {
  return std::make_pair(a.isVertex, a.edge) <
         std::make_pair(b.isVertex, b.edge);
}

// Solves r0 . x = rhs.x, r1 . x = rhs.y.
bool Solve2(vec2 r0, vec2 r1, vec2 rhs, vec2& x) {
  const double det = la::cross(r0, r1);
  if (det == 0) return false;
  x = vec2((rhs.x * r1.y - r0.y * rhs.y) / det,
           (r0.x * rhs.y - rhs.x * r1.x) / det);
  return IsFinite(x);
}

// Circles centered on the line m + s dir, with radius r = a + b s, passing
// through point p: |m + s dir - p|^2 = r^2 is a quadratic in s. dir is a unit
// vector. Keeps the roots with positive radius.
int BisectorCircles(vec2 m, vec2 dir, double a, double b, vec2 p,
                    std::array<vec2, 2>& centers,
                    std::array<double, 2>& radii) {
  const vec2 d = m - p;
  const double qa = 1 - b * b;
  const double qb = 2 * (dot(d, dir) - a * b);
  const double qc = dot(d, d) - a * a;
  std::array<double, 2> roots;
  int n = 0;
  if (qa == 0) {
    if (qb == 0) return 0;
    roots[n++] = -qc / qb;
  } else {
    const double disc = qb * qb - 4 * qa * qc;
    if (disc < 0) return 0;
    const double q = -0.5 * (qb + std::copysign(std::sqrt(disc), qb));
    roots[n++] = q / qa;
    if (q != 0) roots[n++] = qc / q;
  }
  int count = 0;
  for (int i = 0; i < n; ++i) {
    const double r = a + b * roots[i];
    if (!(r > 0) || !std::isfinite(r)) continue;
    radii[count] = r;
    centers[count] = m + roots[i] * dir;
    ++count;
  }
  return count;
}

// Circles touching three contacts: tangent to each line on the ball side
// (n . x - c = r) and passing through each vertex (|x - P| = r). The contacts
// are sorted first and each case uses a fixed sequence of operations, so every
// fillet of the triple gets bit-identical circles. Returns how many (<= 2).
int TouchingCircles(const FilletInput& input, std::array<Contact, 3> f,
                    std::array<vec2, 2>& centers,
                    std::array<double, 2>& radii) {
  std::sort(f.begin(), f.end(), ContactLess);
  const int lines = !f[0].isVertex + !f[1].isVertex + !f[2].isVertex;
  std::array<vec2, 3> n, p;
  std::array<double, 3> c;
  for (int i = 0; i < 3; ++i) {
    p[i] = EdgeStart(input, f[i].edge);
    if (!f[i].isVertex) {
      n[i] = LineNormal(input, f[i].edge);
      c[i] = dot(n[i], p[i]);
    }
  }

  if (lines == 3) {
    // Solve n_i . x - r = c_i by Cramer's rule.
    auto det3 = [](vec3 a, vec3 b, vec3 d) { return dot(a, la::cross(b, d)); };
    const vec3 cx(n[0].x, n[1].x, n[2].x), cy(n[0].y, n[1].y, n[2].y);
    const vec3 cr(-1, -1, -1), rhs(c[0], c[1], c[2]);
    const double det = det3(cx, cy, cr);
    if (det == 0) return 0;
    radii[0] = det3(cx, cy, rhs) / det;
    centers[0] = vec2(det3(rhs, cy, cr) / det, det3(cx, rhs, cr) / det);
    return radii[0] > 0 && std::isfinite(radii[0]) && IsFinite(centers[0]);
  }
  if (lines == 2) {
    // Centers equidistant from both lines, (n0 - n1) . x = c0 - c1, with
    // radius n0 . x - c0; then through the vertex. Parametrizing along this
    // bisector stays well conditioned for nearly parallel lines.
    const vec2 dn = n[0] - n[1];
    const double len = length(dn);
    if (len == 0) return 0;
    const vec2 normal = dn / len;
    const vec2 m = normal * ((c[0] - c[1]) / len);
    const vec2 dir(-normal.y, normal.x);
    return BisectorCircles(m, dir, dot(n[0], m) - c[0], dot(n[0], dir), p[2],
                           centers, radii);
  }
  if (lines == 1) {
    // Centers on the perpendicular bisector of the two vertices, with radius
    // n0 . x - c0; then through a vertex.
    const vec2 chord = p[2] - p[1];
    const double len = length(chord);
    if (len == 0) return 0;
    const vec2 m = 0.5 * (p[1] + p[2]);
    const vec2 dir(-chord.y / len, chord.x / len);
    return BisectorCircles(m, dir, dot(n[0], m) - c[0], dot(n[0], dir), p[1],
                           centers, radii);
  }
  // Circumcircle of three vertices.
  if (!Solve2(2.0 * (p[1] - p[0]), 2.0 * (p[2] - p[0]),
              vec2(dot(p[1], p[1]) - dot(p[0], p[0]),
                   dot(p[2], p[2]) - dot(p[0], p[0])),
              centers[0]))
    return 0;
  radii[0] = length(centers[0] - p[0]);
  return 1;
}

// Velocity of the center of the ball touching contacts g and h as its radius
// grows, at center x and radius r: a line gives n . v = 1, a vertex P gives
// (x - P) . v = r.
bool CenterVelocity(const FilletInput& input, const Contact& g,
                    const Contact& h, vec2 x, double r, vec2& v) {
  auto row = [&](const Contact& f, vec2& n, double& rhs) {
    if (f.isVertex) {
      n = x - EdgeStart(input, f.edge);
      rhs = r;
    } else {
      n = LineNormal(input, f.edge);
      rhs = 1;
    }
  };
  vec2 n0, n1;
  double r0, r1;
  row(g, n0, r0);
  row(h, n1, r1);
  return Solve2(n0, n1, vec2(r0, r1), v);
}

// Balls touching a vertex and a line, or two vertices, come in two branches at
// any radius. Whether centers a and b are on the same one.
bool SameBranch(const FilletInput& input, const Contact& g, const Contact& h,
                vec2 a, vec2 b) {
  if (!g.isVertex && !h.isVertex) return true;
  double sa, sb;
  if (g.isVertex && h.isVertex) {
    const vec2 p = EdgeStart(input, g.edge);
    const vec2 q = EdgeStart(input, h.edge);
    sa = la::cross(q - p, a - p);
    sb = la::cross(q - p, b - p);
  } else {
    const Contact& line = g.isVertex ? h : g;
    const vec2 p = EdgeStart(input, (g.isVertex ? g : h).edge);
    const vec2 dir = EdgeEnd(input, line.edge) - EdgeStart(input, line.edge);
    sa = dot(a - p, dir);
    sb = dot(b - p, dir);
  }
  return (sa > 0 && sb > 0) || (sa < 0 && sb < 0);
}

// The circle of the triple {g, h, f} relevant to the ball of fillet (g, h):
// among touching circles on the fillet's branch, the one nearest in radius.
bool TripleCircle(const FilletInput& input, const Contact& g, const Contact& h,
                  const Contact& f, vec2 ballCenter, double radius,
                  vec2& center, double& rho) {
  if (SameContact(f, g) || SameContact(f, h) || SameContact(g, h)) return false;
  std::array<vec2, 2> centers;
  std::array<double, 2> radii;
  const int count = TouchingCircles(input, {g, h, f}, centers, radii);
  bool found = false;
  for (int i = 0; i < count; ++i) {
    if (!SameBranch(input, g, h, ballCenter, centers[i])) continue;
    if (found && !(std::abs(radii[i] - radius) < std::abs(rho - radius)))
      continue;
    center = centers[i];
    rho = radii[i];
    found = true;
  }
  return found;
}

// Relative size below which a plain floating-point comparison of two nearly
// equal lengths is not trusted and the decision is made from the contacts'
// touching circle instead. It only picks between two methods that agree
// whenever the difference is well above rounding error, so it never changes a
// result by more than that.
constexpr double kNearTie = 1e-9;

// Whether contact f cuts the ball of `fillet`, where `distance` is the plain
// distance from the ball's center to f. When distance and radius are nearly
// equal, the three contacts nearly share one touching circle, and the fillets
// around it must agree: all fit or none do. Then decide by that circle's
// radius rho and the slope there, which depend only on the set of contacts.
bool ContactBlocks(const FilletInput& input, const FilletGeometry& fillet,
                   const Contact& f, double distance, double radius) {
  const double scale =
      radius + std::max(std::abs(fillet.center.x), std::abs(fillet.center.y));
  if (std::abs(distance - radius) > kNearTie * scale) return distance < radius;
  vec2 x, v;
  double rho;
  if (!TripleCircle(input, fillet.fromContact, fillet.toContact, f,
                    fillet.center, radius, x, rho) ||
      !CenterVelocity(input, fillet.fromContact, fillet.toContact, x, rho, v))
    return distance < radius;
  const vec2 toward = f.isVertex ? (x - EdgeStart(input, f.edge)) / rho
                                 : LineNormal(input, f.edge);
  const double k = dot(toward, v) - 1;
  if (k == 0) return distance < radius;
  return k * (radius - rho) < 0;
}

// Whether edge c cuts the fillet ball. Its nearest point to the center is
// either inside the edge (its line) or one of its endpoints (a vertex).
// Endpoints matter even when convex: across a narrow gap a spike tip can be
// the nearest point with both of its edges nearest there too.
bool Blocks(const FilletInput& input, const FilletGeometry& fillet, int c,
            double radius) {
  const vec2 p0 = EdgeStart(input, c);
  const vec2 p1 = EdgeEnd(input, c);
  const vec2 d = p1 - p0;
  const double t = dot(fillet.center - p0, d) / dot(d, d);

  if (t > 0 && t < 1) {
    const double s = dot(EdgeNormal(d, input.invert), fillet.center - p0);
    // A ball behind the edge cannot touch it from the ball side.
    if (!(s > 0)) return -s < radius;
    return ContactBlocks(input, fillet, Contact{c, false}, s, radius);
  }
  const Contact vertex =
      t <= 0 ? Contact{c, true} : Contact{input.nextEdges[c], true};
  // A vertex on a line the ball is tangent to is never closer than radius.
  const vec2 point = t <= 0 ? p0 : p1;
  for (const Contact& contact : {fillet.fromContact, fillet.toContact}) {
    if (!contact.isVertex &&
        (SamePoint(point, EdgeStart(input, contact.edge)) ||
         SamePoint(point, EdgeEnd(input, contact.edge))))
      return false;
  }
  return ContactBlocks(input, fillet, vertex,
                       length(fillet.center - (t <= 0 ? p0 : p1)), radius);
}

// Edges through a point the ball touches are at distance exactly radius from
// its center, so they must not count as blocking it: the edge whose line it
// touches, and every edge ending at a vertex it touches.
bool TouchesContact(const FilletInput& input, const FilletGeometry& fillet,
                    int c) {
  for (const Contact& contact : {fillet.fromContact, fillet.toContact}) {
    if (!contact.isVertex && contact.edge == c) return true;
  }
  for (const auto& [edge, t] : {std::make_pair(fillet.fromEdge, fillet.fromT),
                                std::make_pair(fillet.toEdge, fillet.toT)}) {
    if (t != 0) continue;
    const vec2 vertex = EdgeStart(input, edge);
    if (SamePoint(EdgeStart(input, c), vertex) ||
        SamePoint(EdgeEnd(input, c), vertex))
      return true;
  }
  return false;
}

// A fillet circle is part of the result only if the rolling ball fits there:
// no edge other than the ones it touches may cut it.
std::vector<FilletGeometry> FindFilletPairs(
    const FilletInput& input, const EdgeIndex& index,
    const std::vector<std::pair<int, int>>& broadPairs, double radius) {
  std::vector<FilletGeometry> candidates;
  candidates.reserve(broadPairs.size());
  for (const auto& pair : broadPairs) {
    for (const FilletGeometry& geometry :
         BuildFilletGeometries(input, pair.first, pair.second, radius)) {
      candidates.push_back(geometry);
    }
  }

  Vec<Box> circleBoxes;
  circleBoxes.reserve(candidates.size());
  for (const FilletGeometry& geometry : candidates) {
    // Twice the radius: a blocker exactly tangent to the ball must still be
    // offered, or one fillet of a near-tie can miss it while the others don't.
    // A larger box only adds edges to test; it never changes a decision.
    const vec2 r(2 * radius, 2 * radius);
    circleBoxes.push_back(
        Box(ToVec3(geometry.center - r), ToVec3(geometry.center + r)));
  }

  std::vector<uint8_t> fits(candidates.size(), 1);
  for (auto [query, edge] : index.Query(circleBoxes)) {
    if (!fits[query]) continue;
    const FilletGeometry& geometry = candidates[query];

    // Skip topology neighbour geometry because it must touch the arc.
    if (TouchesContact(input, geometry, edge)) continue;

    if (Blocks(input, geometry, edge, radius)) fits[query] = 0;
  }

  std::vector<FilletGeometry> fillets;
  fillets.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (fits[i]) fillets.push_back(candidates[i]);
  }
  return fillets;
}

std::vector<vec2> discreteArcToPoint(const FilletGeometry& arc, double radius,
                                     int circularSegments, bool invert) {
  std::vector<vec2> points;
  points.push_back(arc.fromPoint);

  const double totalRad = ArcAngle(arc.startRad, arc.endRad, invert);
  // Spread the points evenly: fixed dPhi steps leave a sliver segment at the
  // end when totalRad is just over a multiple of dPhi, whose direction is
  // meaningless after rounding.
  const int segments = std::max(
      1,
      static_cast<int>(std::ceil(totalRad * circularSegments / (2.0 * M_PI))));
  const double step = (invert ? -totalRad : totalRad) / segments;

  for (int i = 1; i < segments; ++i) {
    const double current = arc.startRad + step * i;
    points.push_back({arc.center.x + radius * cos(current),
                      arc.center.y + radius * sin(current)});
  }

  points.push_back(arc.toPoint);
  return points;
}

void AppendArcPoints(SimplePolygon& loop, const FilletGeometry& arc,
                     double radius, int circularSegments, bool invert) {
  const std::vector<vec2> points =
      discreteArcToPoint(arc, radius, circularSegments, invert);
  const double pointTolerance = std::max(radius * 1e-8, 1e-12);
  for (size_t i = 1; i < points.size(); ++i) {
    AppendPoint(loop, points[i], pointTolerance);
  }
}

// Marks fillets that no other fillet's arrival leads to when walking forward
// along the boundary to the next departure, repeatedly: such a fillet cannot be
// part of a closed result loop. Only used to repair degenerate cases.
void RemoveUnreachedFillets(const FilletInput& input,
                            const std::vector<FilletGeometry>& geometries,
                            std::vector<uint8_t>& removed) {
  std::vector<std::vector<size_t>> starts(input.edges.size());
  for (size_t i = 0; i < geometries.size(); ++i) {
    if (!removed[i]) starts[geometries[i].fromEdge].push_back(i);
  }
  for (auto& edgeStarts : starts) {
    std::sort(edgeStarts.begin(), edgeStarts.end(), [&](size_t a, size_t b) {
      return std::make_pair(geometries[a].fromT, geometries[a].fromKey) <
             std::make_pair(geometries[b].fromT, geometries[b].fromKey);
    });
  }
  auto findNext = [&](size_t i) -> int {
    int edge = geometries[i].toEdge;
    double t = geometries[i].toT;
    double key = geometries[i].toKey;
    for (size_t walk = 0; walk <= input.edges.size(); ++walk) {
      for (size_t f : starts[edge]) {
        const FilletGeometry& g = geometries[f];
        if (removed[f] || g.fromT < t || (g.fromT == t && g.fromKey < key))
          continue;
        return static_cast<int>(f);
      }
      edge = NextEdge(input, edge);
      t = 0;
      key = 0;
    }
    return -1;
  };

  std::vector<int> next(geometries.size(), -1);
  std::vector<int> inDegree(geometries.size(), 0);
  for (size_t i = 0; i < geometries.size(); ++i) {
    if (removed[i]) continue;
    next[i] = findNext(i);
    if (next[i] >= 0) ++inDegree[next[i]];
  }
  std::vector<size_t> unreached;
  for (size_t i = 0; i < geometries.size(); ++i) {
    if (!removed[i] && inDegree[i] == 0) unreached.push_back(i);
  }
  while (!unreached.empty()) {
    const size_t i = unreached.back();
    unreached.pop_back();
    removed[i] = 1;
    if (next[i] >= 0 && --inDegree[next[i]] == 0) unreached.push_back(next[i]);
  }
}

// Where a fillet arrives on or departs from the input boundary, as a position
// along its boundary cycle: the edge's index in the cycle, then the parameter
// along the edge, then the pivot key for points at its start vertex.
struct BoundaryEvent {
  size_t fillet;
  int pos;
  double t;
  double key;
};

// Strictly before along the cycle; events at the same position are tied.
bool PositionLess(const BoundaryEvent& a, const BoundaryEvent& b) {
  return std::make_tuple(a.pos, a.t, a.key) <
         std::make_tuple(b.pos, b.t, b.key);
}

// Total order for sorting: the fillet index only makes ties deterministic.
bool EventLess(const BoundaryEvent& a, const BoundaryEvent& b) {
  return std::make_tuple(a.pos, a.t, a.key, a.fillet) <
         std::make_tuple(b.pos, b.t, b.key, b.fillet);
}

std::vector<CrossSection> TraceFillets(
    const FilletInput& input, const std::vector<FilletGeometry>& geometries,
    double radius, int circularSegments) {
  // No early exit without fillets: loops can still vanish (see below), e.g.
  // small holes filled by a negative radius on a convex outline.

  // Boundary cycles: the input boundary as walked past each edge's end.
  const size_t numEdges = input.edges.size();
  std::vector<int> cycleOf(numEdges, -1);
  std::vector<int> posOf(numEdges, -1);
  int numCycles = 0;
  for (size_t start = 0; start < numEdges; ++start) {
    if (cycleOf[start] >= 0) continue;
    int edge = static_cast<int>(start);
    int pos = 0;
    while (cycleOf[edge] < 0) {
      cycleOf[edge] = numCycles;
      posOf[edge] = pos++;
      edge = NextEdge(input, edge);
    }
    if (edge != static_cast<int>(start)) return {};
    ++numCycles;
  }

  // Along a boundary cycle, the result boundary runs from each arrival to the
  // next departure, so arrivals and departures alternate. Pair them by rank:
  // sort each list on its own and match the k-th arrival with the k-th
  // departure. An arrival is never compared with its own departure, so when
  // the stretch between them shrinks below rounding error (consecutive arcs of
  // a near-complete circle) their order cannot flip the pairing.
  std::vector<uint8_t> removed(geometries.size(), 0);
  std::vector<std::vector<BoundaryEvent>> arrivals(numCycles);
  std::vector<std::vector<BoundaryEvent>> departures(numCycles);
  auto collectEvents = [&]() {
    for (int c = 0; c < numCycles; ++c) {
      arrivals[c].clear();
      departures[c].clear();
    }
    for (size_t i = 0; i < geometries.size(); ++i) {
      if (removed[i]) continue;
      const FilletGeometry& g = geometries[i];
      arrivals[cycleOf[g.toEdge]].push_back(
          BoundaryEvent{i, posOf[g.toEdge], g.toT, g.toKey});
      departures[cycleOf[g.fromEdge]].push_back(
          BoundaryEvent{i, posOf[g.fromEdge], g.fromT, g.fromKey});
    }
  };
  auto balanced = [&]() {
    for (int c = 0; c < numCycles; ++c)
      if (arrivals[c].size() != departures[c].size()) return false;
    return true;
  };
  collectEvents();

  // Degenerate radii (e.g. a channel exactly as wide as the ball) can leave a
  // fillet without its counterpart, so a cycle has more arrivals than
  // departures or vice versa. Then drop fillets that walking the boundary
  // never reaches, as these cannot be part of a closed result loop, until
  // every cycle balances. This keeps the result valid; it is not used
  // otherwise, so it cannot hide pairing decisions in regular cases.
  if (!balanced()) {
    RemoveUnreachedFillets(input, geometries, removed);
    collectEvents();
    if (!balanced()) return {};
#ifdef MANIFOLD_DEBUG
    if (ManifoldParams().verbose)
      std::cout << "Fillet: unbalanced boundary cycle at radius " << radius
                << ", dropped unreached fillets" << std::endl;
#endif
  }

  std::vector<int> next(geometries.size(), -1);
  // Whether the stretch from a fillet's arrival to the next departure passes
  // the start of its cycle.
  std::vector<uint8_t> wraps(geometries.size(), 0);
  for (int c = 0; c < numCycles; ++c) {
    std::vector<BoundaryEvent>& arr = arrivals[c];
    std::vector<BoundaryEvent>& dep = departures[c];

    const size_t m = arr.size();
    if (m == 0) continue;
    std::sort(arr.begin(), arr.end(), EventLess);
    std::sort(dep.begin(), dep.end(), EventLess);

    // Alternation leaves at most one departure before the first arrival, so
    // arrival k pairs with departure k + shift, shift 0 or 1. Pick the shift
    // whose order violations are fewest: noise only affects near ties, a wrong
    // shift breaks nearly every pair. Events at the same position are
    // consistent with either order, so ties never count; when both shifts fit
    // equally, a tie is read as an empty stretch (arrival, then departure).
    auto violations = [&](size_t shift) {
      size_t count = 0;
      for (size_t k = 0; k < m; ++k) {
        const bool wrap = k + shift >= m;
        const BoundaryEvent& d = dep[(k + shift) % m];
        if (wrap) {
          if (PositionLess(arr[0], d)) ++count;
        } else {
          if (PositionLess(d, arr[k])) ++count;
          if (k + 1 < m && PositionLess(arr[k + 1], d)) ++count;
        }
      }
      return count;
    };
    const size_t shift = violations(1) < violations(0) ? 1 : 0;
    for (size_t k = 0; k < m; ++k) {
      next[arr[k].fillet] = static_cast<int>(dep[(k + shift) % m].fillet);
      wraps[arr[k].fillet] = k + shift >= m;
    }
  }

  Polygons resultLoops;
  std::vector<uint8_t> touchedLoop(input.loops.size(), 0);
  const double pointTolerance = std::max(radius * 1e-8, 1e-12);
  std::vector<uint8_t> used(geometries.size(), 0);

  for (size_t startFillet = 0; startFillet < geometries.size(); ++startFillet) {
    if (used[startFillet] || removed[startFillet]) continue;

    SimplePolygon loop;
    size_t current = startFillet;
    do {
      if (used[current]) return {};
      used[current] = 1;

      const FilletGeometry& fillet = geometries[current];
      AppendPoint(loop, fillet.fromPoint, pointTolerance);
      AppendArcPoints(loop, fillet, radius, circularSegments, input.invert);
      touchedLoop[input.edges[fillet.fromEdge].loop] = 1;
      touchedLoop[input.edges[fillet.toEdge].loop] = 1;

      if (next[current] < 0) return {};
      const FilletGeometry& nextFillet = geometries[next[current]];
      // Walk the input boundary to the paired departure. Without wrapping, a
      // departure not after the arrival is a tie: the stretch is empty.
      const BoundaryEvent from{current, posOf[fillet.toEdge], fillet.toT,
                               fillet.toKey};
      const BoundaryEvent to{static_cast<size_t>(next[current]),
                             posOf[nextFillet.fromEdge], nextFillet.fromT,
                             nextFillet.fromKey};
      const bool wrap = wraps[current];
      if (wrap || PositionLess(from, to)) {
        int edge = fillet.toEdge;
        bool first = true;
        while (edge != nextFillet.fromEdge || (first && wrap)) {
          AppendPoint(loop, EdgeEnd(input, edge), pointTolerance);
          touchedLoop[input.edges[edge].loop] = 1;
          edge = NextEdge(input, edge);
          first = false;
        }
      }
      AppendPoint(loop, nextFillet.fromPoint, pointTolerance);
      current = static_cast<size_t>(next[current]);
    } while (current != startFillet);

    if (loop.size() > 1 &&
        (SamePoint(loop.front(), loop.back()) ||
         SamePointWithin(loop.front(), loop.back(), pointTolerance))) {
      loop.pop_back();
    }
    if (loop.size() < 3) return {};
    resultLoops.push_back(loop);
  }

  // A loop no fillet touches is either untouched by the ball (a hole for
  // positive radius, an outer loop for negative) or entirely too small for it,
  // in which case it vanishes. Holes left without an enclosing outer loop are
  // removed by the Positive fill rule below.
  for (size_t loopIdx = 0; loopIdx < input.loops.size(); ++loopIdx) {
    if (touchedLoop[loopIdx]) continue;
    const bool isOuter = SignedArea(input.loops[loopIdx]) > 0;
    if (isOuter == input.invert) resultLoops.push_back(input.loops[loopIdx]);
  }

  if (resultLoops.empty()) return {};

  return CrossSection(resultLoops).Decompose();
}

// Turns towards the material side must come from discretized arcs, so be no
// sharper than one segment: left turns for a positive radius, right turns for
// a negative one. Turns the other way are untouched corners.
bool HasFilletAngles(const CrossSection& crossSection, int circularSegments,
                     bool invert) {
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

      const double turn = ToRad(e2) - ToRad(e1);
      const double angle = NormalizeAngle(invert ? -turn : turn);
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
    if (!HasFilletAngles(crossSection, circularSegments, invert)) return false;
  }

  // Decomposed pieces are disjoint, so their areas add up.
  double area = 0;
  for (const CrossSection& crossSection : result) area += crossSection.Area();
  return invert ? area > original.Area() : area < original.Area();
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

  const int segments = circularSegments > 2
                           ? circularSegments
                           : Quality::GetCircularSegments(radius);

  const std::vector<size_t> loopOffsets = BuildLoopOffsets(polygons);
  const std::vector<EdgeRef> edges = BuildEdges(polygons, loopOffsets);
  const std::vector<int> nextEdges =
      BuildNextEdges(polygons, loopOffsets, edges, invert);
  std::vector<int> prevEdges(edges.size(), -1);
  for (size_t e = 0; e < edges.size(); ++e) prevEdges[nextEdges[e]] = e;
  for (size_t e = 0; e < edges.size(); ++e) {
    if (prevEdges[e] >= 0) continue;
    const EdgeRef ref = edges[e];
    const size_t n = polygons[ref.loop].size();
    prevEdges[e] =
        static_cast<int>(loopOffsets[ref.loop] + (ref.edge + n - 1) % n);
  }
  const FilletInput input{
      polygons, loopOffsets,          edges, nextEdges, prevEdges,
      invert,   2.0 * M_PI / segments};

#ifdef MANIFOLD_DEBUG
  if (caseIndex == 0) SavePolygons("Testing/Fillet/input.txt", polygons);

  // Debug dump only; skipped when Testing/Fillet does not exist, so it never
  // changes the result.
  resultOutputFile.open("Testing/Fillet/" + std::to_string(caseIndex) + ".txt");
  resultOutputFile << std::setprecision(
                          std::numeric_limits<double>::max_digits10)
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

  const EdgeIndex index = BuildEdgeIndex(polygons);
  const std::vector<std::pair<int, int>> broadPairs =
      BuildCollider(input, index, radius);
  const std::vector<FilletGeometry> geometries =
      FindFilletPairs(input, index, broadPairs, radius);

  const CrossSection original(polygons);
  auto result = TraceFillets(input, geometries, radius, segments);
  if (!IsValidFilletResult(result, original, segments, invert)) result.clear();

#ifdef MANIFOLD_DEBUG
  if (resultOutputFile.is_open()) {
    SaveCrossSection(resultOutputFile, result);
    resultOutputFile.close();
  }
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

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
  bool operator()(const Tangent& a, const Tangent& b) const {
    return a.t < b.t;
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

bool FinishFilletGeometry(const FilletInput& input, int firstEdge,
                          int secondEdge, double firstT, double secondT,
                          bool firstIsVertex, bool secondIsVertex, vec2 center,
                          FilletGeometry& geometry) {
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
  vec2 fromPoint =
      EdgeStart(input, firstEdge) +
      firstT * (EdgeEnd(input, firstEdge) - EdgeStart(input, firstEdge));
  vec2 toPoint =
      EdgeStart(input, secondEdge) +
      secondT * (EdgeEnd(input, secondEdge) - EdgeStart(input, secondEdge));
  vec2 fromRadius = fromPoint - center;
  vec2 toRadius = toPoint - center;

  const double radiusCross = la::cross(fromRadius, toRadius);
  if ((!input.invert && radiusCross < 0) || (input.invert && radiusCross > 0)) {
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

  if (!secondConvex) {
    const IntersectResult ev = IntersectSegmentArc(
        firstOffset, firstOffset + firstVec, secondEnd, radius, secondNormal,
        secondNextNormal, input.invert);

    for (int i = 0; i < ev.count; ++i) {
      const vec2 candidate = ev.points[i];
      const double t =
          dot(candidate - firstOffset, firstVec) / dot(firstVec, firstVec);
      if (t >= 0 && t <= 1 &&
          FinishFilletGeometry(input, firstEdge, secondEdge, t, 1, false, true,
                               candidate, geometry)) {
        geometries.push_back(geometry);
      }
    }
  }

  if (!firstConvex) {
    const IntersectResult ve =
        IntersectSegmentArc(secondOffset, secondOffset + secondVec, firstEnd,
                            radius, firstNormal, firstNextNormal, input.invert);

    for (int i = 0; i < ve.count; ++i) {
      const vec2 candidate = ve.points[i];
      const double t =
          dot(candidate - secondOffset, secondVec) / dot(secondVec, secondVec);
      if (t >= 0 && t <= 1 &&
          FinishFilletGeometry(input, firstEdge, secondEdge, 1, t, true, false,
                               candidate, geometry)) {
        geometries.push_back(geometry);
      }
    }
  }

  if (!firstConvex && !secondConvex) {
    const IntersectResult vv = IntersectArcArc(
        firstEnd, firstNormal, firstNextNormal, input.invert, secondEnd,
        secondNormal, secondNextNormal, input.invert, radius);

    for (int i = 0; i < vv.count; ++i) {
      if (FinishFilletGeometry(input, firstEdge, secondEdge, 1, 1, true, true,
                               vv.points[i], geometry)) {
        geometries.push_back(geometry);
      }
    }
  }

  return geometries;
}

double DistanceToSegment(vec2 p, vec2 a, vec2 b) {
  const vec2 ab = b - a;
  const double lenSq = dot(ab, ab);
  const double t =
      lenSq > 0 ? std::clamp(dot(p - a, ab) / lenSq, 0.0, 1.0) : 0.0;
  return length(p - (a + t * ab));
}

vec2 LineNormal(const FilletInput& input, int e) {
  return EdgeNormal(EdgeEnd(input, e) - EdgeStart(input, e), input.invert);
}

// Velocity of the center of the ball tangent to the lines of edges a and b as
// its radius grows: na . v = nb . v = 1.
bool CenterVelocity(const FilletInput& input, int a, int b, vec2& v) {
  const vec2 na = LineNormal(input, a);
  const vec2 nb = LineNormal(input, b);
  const double den = la::cross(na, nb);
  if (den == 0) return false;
  v = vec2((nb.y - na.y) / den, (na.x - nb.x) / den);
  return true;
}

// Radius of the circle tangent to the lines of three edges, on the ball side of
// each (signed distance rho from all three). The edges are sorted first, so
// every fillet of the triple gets bit-identical results.
bool TangentRadius(const FilletInput& input, std::array<int, 3> edges,
                   double& rho) {
  std::sort(edges.begin(), edges.end());
  std::array<vec2, 3> n;
  std::array<double, 3> c;
  for (int i = 0; i < 3; ++i) {
    n[i] = LineNormal(input, edges[i]);
    c[i] = dot(n[i], EdgeStart(input, edges[i]));
  }
  // Solve n_i . x - rho = c_i by Cramer's rule.
  auto det3 = [](const std::array<vec3, 3>& m) {
    return dot(m[0], la::cross(m[1], m[2]));
  };
  const std::array<vec3, 3> cols{vec3(n[0].x, n[1].x, n[2].x),
                                 vec3(n[0].y, n[1].y, n[2].y),
                                 vec3(-1, -1, -1)};
  const double det = det3(cols);
  if (det == 0) return false;
  rho = det3({cols[0], cols[1], vec3(c[0], c[1], c[2])}) / det;
  return std::isfinite(rho);
}

// Whether the line of edge c cuts the ball tangent to the lines of edges a and
// b, with c's nearest point inside c on the ball side at signed distance s.
// The distance to c changes linearly with the radius, s(r) - r = k (r - rho),
// where rho is where the ball touches all three lines. Deciding by rho, which
// depends only on the set {a, b, c}, keeps the three fillets of a near-complete
// circle consistent: all fit or none do.
bool LineBlocks(const FilletInput& input, int a, int b, int c, double radius,
                double s) {
  vec2 v;
  double rho;
  if (!CenterVelocity(input, a, b, v) || !TangentRadius(input, {a, b, c}, rho))
    return s < radius;
  const double k = dot(LineNormal(input, c), v) - 1;
  if (k == 0) return s < radius;
  return k * (radius - rho) < 0;
}

// Order of fillet x's arrival and fillet y's departure on the edge they share,
// when both touch its line: true if y departs at or after x arrives. Both
// tangent points move linearly with the radius and meet where one circle
// touches all three lines, so the order follows from the same rho as
// LineBlocks instead of from nearly equal parameters. Returns false in
// `decided` when not applicable.
bool DepartsAfterArrival(const FilletInput& input, const FilletGeometry& x,
                         const FilletGeometry& y, double radius,
                         bool& decided) {
  decided = false;
  const int e = x.toEdge;
  if (y.fromEdge != e || x.fromContact.isVertex || x.toContact.isVertex ||
      y.fromContact.isVertex || y.toContact.isVertex || x.toContact.edge != e ||
      y.fromContact.edge != e)
    return false;
  const int a = x.fromContact.edge;
  const int c = y.toContact.edge;
  vec2 vx, vy;
  double rho;
  if (!CenterVelocity(input, a, e, vx) || !CenterVelocity(input, e, c, vy) ||
      !TangentRadius(input, {a, e, c}, rho))
    return false;
  const double m =
      dot(vy - vx, normalize(EdgeEnd(input, e) - EdgeStart(input, e)));
  if (m == 0) return false;
  decided = true;
  return m * (radius - rho) >= 0;
}

// Whether edge c cuts the fillet ball. Its nearest point to the center is
// either inside the edge (a line) or one of its endpoints (a point). Endpoints
// matter even when convex: across a narrow gap a spike tip can be the nearest
// point with both of its edges nearest there too.
bool Blocks(const FilletInput& input, const FilletGeometry& fillet, int c,
            double radius) {
  const vec2 p0 = EdgeStart(input, c);
  const vec2 p1 = EdgeEnd(input, c);
  const vec2 d = p1 - p0;
  const double t = dot(fillet.center - p0, d) / dot(d, d);

  if (t > 0 && t < 1) {
    const double s = dot(EdgeNormal(d, input.invert), fillet.center - p0);
    if (s > 0 && !fillet.fromContact.isVertex && !fillet.toContact.isVertex)
      return LineBlocks(input, fillet.fromContact.edge, fillet.toContact.edge,
                        c, radius, s);
    // TODO: a ball touching a vertex has no linear center path, so it still
    // uses a plain distance test, not consistent across a near-full circle.
    return std::abs(s) < radius;
  }

  // TODO: vertex blockers still use a plain distance test.
  return length(fillet.center - (t <= 0 ? p0 : p1)) < radius;
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
    const vec2 r(radius, radius);
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
    starts[geometry.toEdge].push_back(
        Tangent{geometry.toT, i, geometry.toPoint});
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
      segments.push_back(Segment{static_cast<int>(edge), start.t, end.t,
                                 start.point, end.point, end.fillet, false});
    }
  }

  return true;
}

std::vector<CrossSection> TraceFillets(
    const FilletInput& input, const std::vector<FilletGeometry>& geometries,
    double radius, int circularSegments) {
  // No early exit without fillets: loops can still vanish (see below), e.g.
  // small holes filled by a negative radius on a convex outline.
  std::vector<std::vector<size_t>> starts(input.edges.size());
  for (size_t i = 0; i < geometries.size(); ++i) {
    starts[geometries[i].fromEdge].push_back(i);
  }
  for (auto& edgeStarts : starts) {
    std::sort(edgeStarts.begin(), edgeStarts.end(), [&](size_t a, size_t b) {
      return std::make_pair(geometries[a].fromT, geometries[a].fromKey) <
             std::make_pair(geometries[b].fromT, geometries[b].fromKey);
    });
  }

  // Whether departure `fillet` comes at or after position (t, key) on its edge.
  auto departsAfter = [&](size_t fillet, double t, double key) {
    const FilletGeometry& g = geometries[fillet];
    return g.fromT > t || (g.fromT == t && g.fromKey >= key);
  };
  // Whether walking on from the end of fillet i first reaches departure
  // `fillet` on edge i ends on.
  auto departsAfterEnd = [&](size_t i, size_t fillet) {
    bool decided;
    const bool after = DepartsAfterArrival(input, geometries[i],
                                           geometries[fillet], radius, decided);
    if (decided) return after;
    return departsAfter(fillet, geometries[i].toT, geometries[i].toKey);
  };

  // The fillet reached by walking forward along the boundary from the end of
  // fillet i, skipping removed fillets; -1 if none is found.
  std::vector<uint8_t> removed(geometries.size(), 0);
  auto findNext = [&](size_t i) -> int {
    int edge = geometries[i].toEdge;
    for (size_t walk = 0; walk <= input.edges.size(); ++walk) {
      for (size_t fillet : starts[edge]) {
        if (removed[fillet]) continue;
        if (walk == 0 && !departsAfterEnd(i, fillet)) continue;
        return static_cast<int>(fillet);
      }
      edge = NextEdge(input, edge);
    }
    return -1;
  };

  std::vector<int> next(geometries.size());
  std::vector<int> inDegree(geometries.size(), 0);
  for (size_t i = 0; i < geometries.size(); ++i) {
    next[i] = findNext(i);
    if (next[i] >= 0) ++inDegree[next[i]];
  }

  // On the true result boundary every fillet is reached from exactly one
  // other. A fillet nothing reaches is spurious (e.g. a near-duplicate pivot at
  // a pinched vertex); removing it does not change any other fillet's
  // successor, so prune those repeatedly.
  std::vector<size_t> orphans;
  for (size_t i = 0; i < geometries.size(); ++i) {
    if (inDegree[i] == 0) orphans.push_back(i);
  }
  while (!orphans.empty()) {
    const size_t i = orphans.back();
    orphans.pop_back();
    removed[i] = 1;
    if (next[i] >= 0 && --inDegree[next[i]] == 0) orphans.push_back(next[i]);
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
      int edge = fillet.toEdge;
      bool firstEdge = true;
      while (edge != nextFillet.fromEdge ||
             (firstEdge && !departsAfterEnd(current, next[current]))) {
        AppendPoint(loop, EdgeEnd(input, edge), pointTolerance);
        touchedLoop[input.edges[edge].loop] = 1;
        edge = NextEdge(input, edge);
        firstEdge = false;
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

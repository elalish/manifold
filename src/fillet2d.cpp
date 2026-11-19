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
#include <limits>
#include <tuple>
#include <vector>

#include "collider.h"
#include "manifold/cross_section.h"
#include "parallel.h"
#include "vec.h"

// Debug output only; MANIFOLD_DEBUG is incompatible with MANIFOLD_NO_IOSTREAM.
#ifdef MANIFOLD_DEBUG
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "manifold/manifold.h"
#endif

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

// The direction of a line as an angle that makes "parallel" exact: the
// direction is first turned into the upper half-plane, by a half-turn if
// need be, and it is the angle of that - in [0, pi) - that is kept, with
// whether it was turned. Two lines are parallel exactly when those angles
// are the same number, and then all lines parallel to one are parallel to
// each other. From one angle in (-pi, pi] per line they were not: a
// direction and its opposite differ by a half-turn only as nearly as
// rounding puts it, and "differs by pi" could hold between one line and
// each of two others that differ from each other.
struct LineAngle {
  double folded = 0;
  bool turned = false;
};

LineAngle AngleOf(vec2 d) {
  const bool turned = d.y < 0 || (d.y == 0 && d.x < 0);
  const vec2 up = turned ? -d : d;
  return {std::atan2(up.y, up.x), turned};
}

// Sign of the turn from direction `from` to `to`: of the cross product of
// the two directions. Zero for the same and for opposite directions.
int TurnSign(LineAngle from, LineAngle to) {
  if (from.folded == to.folded) return 0;
  // Within the same half-turn the folded angles are in order; across, the
  // one further round is a half-turn back.
  return (from.folded < to.folded) == (from.turned == to.turned) ? 1 : -1;
}

struct FilletInput {
  const Polygons& loops;
  const std::vector<size_t>& loopOffsets;
  const std::vector<EdgeRef>& edges;
  const std::vector<int>& nextEdges;
  const std::vector<int>& prevEdges;
  const std::vector<int>& startPoints;
  bool invert;
  double minArcAngle;
  // For the vertex at the end of each edge, how the boundary turns there
  // (see EndKind), decided once.
  std::vector<int8_t> endKinds;
  // Set once endKinds is complete.
  bool kindsReady = false;
  // The unit normal of each edge, towards the ball. Edges of one straight
  // run share the normal of the run's line.
  std::vector<vec2> normals;
  // Edges that continue each other in a straight line - an edge split into
  // pieces - are one line: for each edge, the first and last edge of its run.
  // The direction of each edge's line, as one angle: every sign of a cross
  // product of two lines' directions is read from the order of their angles
  // (LineCross), so those signs cannot contradict each other the way
  // separately rounded cross products of nearly parallel lines can.
  std::vector<LineAngle> angles;
  std::vector<int> runFirst;
  std::vector<int> runLast;
  // For each edge, a box around the centers of the balls that touch its line
  // or pivot around its end vertex (see BuildCollider).
  std::vector<Box> pieces;
  // The endpoints of each edge, copied out of loops for fast access.
  std::vector<vec2> starts;
  std::vector<vec2> ends;
};

// At a vertex the boundary either turns away from the ball, which can then
// pivot around it; runs straight on; or turns towards the ball. In a corner
// turning towards it, the ball tangent to both edges is real if it touches
// each within the edge - then no ball rolls further into the corner than it
// does - and otherwise the corner is too short on one side to hold a ball.
enum EndKind : int8_t { kTouchable = -1, kStraight = 0, kShortCorner, kCorner };

// A feature of the input boundary a fillet ball can touch, as one id: 2e is
// the interior (line) of edge e, 2e + 1 the vertex at the start of edge e.
int LineFeature(int e) { return 2 * e; }
int VertexFeature(int e) { return 2 * e + 1; }
bool IsVertex(int feature) { return feature & 1; }
int FeatureEdge(int feature) { return feature >> 1; }

// One end of a fillet arc: the feature the ball touches, and the parameter
// along FeatureEdge(feature) - where a line is touched, 0 for a vertex.
struct FilletEnd {
  int feature;
  double t;
};

struct FilletGeometry {
  FilletEnd from;
  FilletEnd to;
  vec2 center;
  // Order of the ends at the start vertex of their edge (see Position and
  // PivotKey), cached for sorting.
  double fromKey;
  double toKey;
  // The center as first computed, on the circle of radius r around each
  // vertex the ball touches: what the order of balls around a vertex is read
  // from (ComesFirst).
  vec2 pivotCenter;
};

struct IntersectResult {
  int count = 0;
  std::array<vec2, 2> points{vec2(), vec2()};
  // For a line and a vertex, the parameter of each point along the edge.
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

struct EdgeIndex {
  Collider collider;
  Vec<int> leafEdge;

  // Calls f(query index, global edge) for every query box overlapping an edge
  // box.
  template <typename F>
  void ForEach(const Vec<Box>& queries, F f) const {
    auto record = [&](int query, int leaf) { f(query, leafEdge[leaf]); };
    auto recorder = MakeSimpleRecorder(record);
    collider.Collisions(recorder, queries.cview(), false);
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

void AppendPoint(SimplePolygon& loop, const vec2& point) {
  if (loop.empty() || !SamePoint(loop.back(), point)) loop.push_back(point);
}

vec2 EdgeStart(const FilletInput& input, int edge) {
  return input.starts[edge];
}

vec2 EdgeEnd(const FilletInput& input, int edge) { return input.ends[edge]; }

int NextEdge(const FilletInput& input, int edge) {
  return input.nextEdges[edge];
}

// Point ids: vertices with exactly the same coordinates share one, so a pinch
// or two loops touching at a vertex is a single point.
int StartPoint(const FilletInput& input, int edge) {
  return input.startPoints[edge];
}

int EndPoint(const FilletInput& input, int edge) {
  return input.startPoints[NextEdge(input, edge)];
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

// Sign of cross(a - b, c - d). The signs below are the decisions the fillets
// are classified by. Each is evaluated in one place, by one expression, and
// everything that follows from it is derived from that result rather than
// computed again: rounding can make a sign differ from the exact one, but it
// cannot make two uses of it disagree.
int CrossSign(vec2 a, vec2 b, vec2 c, vec2 d) {
  const vec2 u = a - b, w = c - d;
  const double det = u.x * w.y - u.y * w.x;
  return det > 0 ? 1 : det < 0 ? -1 : 0;
}

// Sign of dot(a - b, c - d).
int DotSign(vec2 a, vec2 b, vec2 c, vec2 d) {
  const vec2 u = a - b, w = c - d;
  const double sum = u.x * w.x + u.y * w.y;
  return sum > 0 ? 1 : sum < 0 ? -1 : 0;
}

// The line an edge lies on is that of its whole straight run: decisions
// about lines are made from the run's two ends, so that all pieces of one
// split edge are the same line in every one of them.
int Run(const FilletInput& input, int e) { return input.runFirst[e]; }

vec2 LineStart(const FilletInput& input, int e) {
  return input.starts[input.runFirst[e]];
}

vec2 LineEnd(const FilletInput& input, int e) {
  return input.ends[input.runLast[e]];
}

// Sign of the cross product of the directions of two edges' lines.
int LineCross(const FilletInput& input, int a, int b) {
  return TurnSign(input.angles[a], input.angles[b]);
}

// Which way along the line of an edge the point v lies from the point p: the
// sign of (v - p) . direction. One expression for every use: where the foot
// of a vertex falls relative to an end of the edge decides which of its two
// balls can reach that end, on the edge's side and on the vertex's alike.
int AlongLine(const FilletInput& input, int line, vec2 v, vec2 p) {
  return DotSign(v, p, LineEnd(input, line), LineStart(input, line));
}

// Whether the line of edge a ends at the vertex where that of edge b starts.
bool LinesMeet(const FilletInput& input, int a, int b) {
  return input.startPoints[input.nextEdges[input.runLast[a]]] ==
         input.startPoints[input.runFirst[b]];
}

// How many balls pass through two vertices: two closer than 2 r, a double one
// at exactly 2 r.
int CountArcArc(vec2 c1, vec2 c2, double radius) {
  if (SamePoint(c1, c2)) return 0;
  const double f = 4 * radius * radius - length2(c1 - c2);
  return f < 0 ? 0 : f == 0 ? 1 : 2;
}

// Centers of the `count` balls of CountArcArc, before any sector test: first
// the one to the right of c1 -> c2, then the one to the left.
IntersectResult IntersectArcArc(vec2 c1, vec2 c2, double radius, int count) {
  IntersectResult result;
  if (count == 0) return result;
  const double dSq = length2(c1 - c2);
  const double d = std::sqrt(dSq);
  const double halfDistance = d * 0.5;
  const double height =
      count == 1 ? 0
                 : std::sqrt(std::max(
                       0.0, radius * radius - halfDistance * halfDistance));
  const vec2 midpoint = (c1 + c2) * 0.5;
  const vec2 offset = vec2{c2.y - c1.y, c1.x - c2.x} * (height / d);
  const std::array<vec2, 2> points{midpoint + offset, midpoint - offset};
  for (int i = 0; i < count; ++i) result.points[result.count++] = points[i];
  result.tangent = count == 1;
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
// Also assigns each edge the point id of its start vertex.
std::vector<int> BuildNextEdges(const Polygons& polygons,
                                const std::vector<size_t>& loopOffsets,
                                const std::vector<EdgeRef>& edges, bool invert,
                                std::vector<int>& startPoints) {
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

  // pointBegin[p] is where the edges starting at point p begin in byStart.
  startPoints.assign(edges.size(), 0);
  std::vector<size_t> pointBegin;
  for (size_t i = 0; i < byStart.size(); ++i) {
    if (i == 0 || pointLess(start(byStart[i - 1]), start(byStart[i])))
      pointBegin.push_back(i);
    startPoints[byStart[i]] = static_cast<int>(pointBegin.size()) - 1;
  }
  pointBegin.push_back(byStart.size());

  std::vector<int> nextEdges(edges.size());
  for (size_t e = 0; e < edges.size(); ++e) {
    const EdgeRef ref = edges[e];
    nextEdges[e] = static_cast<int>(loopOffsets[ref.loop] +
                                    (ref.edge + 1) % polygons[ref.loop].size());

    const int point = startPoints[nextEdges[e]];
    const auto first = byStart.begin() + pointBegin[point];
    const auto last = byStart.begin() + pointBegin[point + 1];
    if (last - first < 2) continue;

    const vec2 vertex = end(e);
    const double back = ToRad(start(e) - vertex);
    double best = std::numeric_limits<double>::infinity();
    for (auto it = first; it != last; ++it) {
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

// Index over one box per edge, in edge order.
EdgeIndex BuildBoxIndex(const std::vector<Box>& edgeBoxesIn) {
  Box bounds;
  for (const Box& box : edgeBoxesIn) bounds = bounds.Union(box);

  std::vector<EdgeBox> edgeBoxes;
  edgeBoxes.reserve(edgeBoxesIn.size());
  for (size_t e = 0; e < edgeBoxesIn.size(); ++e) {
    const Box& box = edgeBoxesIn[e];
    edgeBoxes.push_back(
        EdgeBox{box, Collider::MortonCode((box.min + box.max) / 2.0, bounds),
                static_cast<int>(e)});
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

EdgeIndex BuildEdgeIndex(const FilletInput& input) {
  std::vector<Box> boxes(input.edges.size());
  for (size_t e = 0; e < boxes.size(); ++e)
    boxes[e] = Box(ToVec3(input.starts[e]), ToVec3(input.ends[e]));
  return BuildBoxIndex(boxes);
}

// Pairs of edges that may share a fillet ball. The centers of the balls
// touching an edge lie on its offset by the radius, and for a vertex the ball
// can pivot around, on the arc around it between the two edges' normals. Two
// edges share a ball only where these pieces meet, so only pairs whose pieces'
// bounding boxes overlap are kept. Far fewer than the edges within reach of
// each other's balls: for short edges and a large radius those are most of
// the neighbourhood.
std::vector<std::pair<int, int>> BuildCollider(FilletInput& input,
                                               double radius) {
  const size_t numEdges = input.edges.size();
  std::vector<Box>& pieces = input.pieces;
  pieces.resize(numEdges);
  Vec<Box> queries(numEdges);
  for (size_t e = 0; e < numEdges; ++e) {
    const vec2 normal = input.normals[e];
    const vec2 start = input.starts[e] + normal * radius;
    const vec2 end = input.ends[e] + normal * radius;
    Box box(ToVec3(start), ToVec3(end));

    if (input.endKinds[e] == kTouchable) {
      const vec2 vertex = input.ends[e];
      const vec2 nextNormal = input.normals[input.nextEdges[e]];
      const double cosine = dot(normal, nextNormal);
      if (cosine > 0) {
        // The arc bulges past the box of its ends by at most its sagitta.
        const vec2 arcEnd = vertex + nextNormal * radius;
        const double sagitta = radius * (1 - std::sqrt((1 + cosine) / 2));
        const vec2 bulge(sagitta, sagitta);
        Box arc(ToVec3(end), ToVec3(arcEnd));
        arc.min -= ToVec3(bulge);
        arc.max += ToVec3(bulge);
        box = box.Union(arc);
      } else {
        const vec2 r(radius, radius);
        box.Union(ToVec3(vertex - r));
        box.Union(ToVec3(vertex + r));
      }
    }

    // The corners are rounded sums; widen by more than that rounding so that
    // pieces meeting exactly at a shared end are never separated by it.
    const vec3 size = max(abs(box.min), abs(box.max));
    const double pad = 8 * std::numeric_limits<double>::epsilon() *
                       (std::max(size.x, size.y) + radius);
    box.min -= vec3(pad, pad, 0);
    box.max += vec3(pad, pad, 0);
    pieces[e] = box;
  }

  // Each unordered pair is found from both sides; keep one direction. The
  // boxes are queried in the index's own order, which keeps neighbouring
  // queries in neighbouring parts of it.
  const EdgeIndex index = BuildBoxIndex(pieces);
  for (size_t i = 0; i < numEdges; ++i) queries[i] = pieces[index.leafEdge[i]];
  std::vector<std::pair<int, int>> pairs;
  index.ForEach(queries, [&](int query, int b) {
    const int a = index.leafEdge[query];
    if (a < b) pairs.push_back({a, b});
  });
  return pairs;
}

vec2 LineNormal(const FilletInput& input, int e) { return input.normals[e]; }

// The point at parameter t along edge e, exactly its start or end vertex at
// t = 0 or 1.
vec2 PointOnEdge(const FilletInput& input, int e, double t) {
  if (t == 0) return EdgeStart(input, e);
  if (t == 1) return EdgeEnd(input, e);
  return EdgeStart(input, e) + t * (EdgeEnd(input, e) - EdgeStart(input, e));
}

// Where an end sits on the boundary for tracing: a point on a vertex is always
// referenced as the start of the following edge, so arrivals and departures
// there compare correctly.
struct BoundaryPos {
  int edge;
  double t;
};

BoundaryPos Position(const FilletInput& input, const FilletEnd& end) {
  const int edge = FeatureEdge(end.feature);
  if (end.t == 1) return {NextEdge(input, edge), 0};
  return {edge, end.t};
}

vec2 EndPoint(const FilletInput& input, const FilletEnd& end) {
  return PointOnEdge(input, FeatureEdge(end.feature), end.t);
}

double EndAngle(const FilletInput& input, const FilletEnd& end, vec2 center) {
  return NormalizeAngle(ToRad(EndPoint(input, end) - center));
}

// Where a fillet end sits among the ends at the same vertex V, in the order
// they are met while tracing. A ball touching V pivots around it, its center
// sweeping from the previous edge's normal towards the next edge's normal, so
// it is ordered by how far its center has swept. A ball tangent to the
// previous edge right at V comes before all of those, one tangent to the next
// edge at V after them. Tangent points inside an edge do not need a key.
double PivotKey(const FilletInput& input, const FilletEnd& end, vec2 center) {
  if (!IsVertex(end.feature)) return end.t == 1 ? -1 : end.t == 0 ? 7 : 0;
  const int edge = FeatureEdge(end.feature);
  const double from = ToRad(LineNormal(input, input.prevEdges[edge]));
  const double to = ToRad(center - EdgeStart(input, edge));
  // Without invert the boundary turns right around a reflex vertex, so the
  // center sweeps clockwise.
  auto swept = [&](double angle) {
    return NormalizeAngle(input.invert ? angle - from : from - angle);
  };
  const double key = swept(to);
  // A center decided to be within the sector but rounded just outside of it
  // belongs at the nearer end.
  const double width = swept(ToRad(LineNormal(input, edge)));
  if (key <= width) return key;
  return 2 * M_PI - key < key - width ? 0 : width;
}

bool FinishFilletGeometry(const FilletInput& input, int firstEdge,
                          int secondEdge, double firstT, double secondT,
                          bool firstIsVertex, bool secondIsVertex, vec2 center,
                          int turn, FilletGeometry& geometry,
                          bool reverse = false) {
  // A vertex contact is the end vertex of the given edge, i.e. the start of
  // the next one.
  auto end = [&](int edge, double t, bool isVertex) {
    return isVertex ? FilletEnd{VertexFeature(NextEdge(input, edge)), 0}
                    : FilletEnd{LineFeature(edge), t};
  };
  FilletEnd from = end(firstEdge, firstT, firstIsVertex);
  FilletEnd to = end(secondEdge, secondT, secondIsVertex);

  // Arcs run CCW around the ball (CW with invert). `turn` is the side of the
  // first contact the second is on, seen from the center (see
  // RawCandidate). `reverse` asks for the other arc of a ball through a neck
  // (see IntersectResult::tangent).
  const bool backwards = input.invert ? turn > 0 : turn < 0;
  if (backwards != reverse) std::swap(from, to);

  if (!(ArcAngle(EndAngle(input, from, center), EndAngle(input, to, center),
                 input.invert) > 0))
    return false;

  geometry = FilletGeometry{from, to, center, PivotKey(input, from, center),
                            PivotKey(input, to, center)};
  return true;
}

// Whether the ball can pivot around the vertex at the end of edge e.
bool TouchableEnd(const FilletInput& input, int e) {
  return input.endKinds[e] == kTouchable;
}

// A fillet ball between edges first < second, by what it touches on each:
// the edge's line, or the vertex at the edge's end. One ball near a vertex can
// be described several ways (on the line before it, on the vertex, on the
// line after it), each generated by a different pair or kind.
enum Kind { kLineLine, kLineVertex, kVertexLine, kVertexVertex };

bool FirstIsVertex(int kind) {
  return kind == kVertexLine || kind == kVertexVertex;
}
bool SecondIsVertex(int kind) {
  return kind == kLineVertex || kind == kVertexVertex;
}

// A ball of one kind before any test of where it touches: the line
// parameters (1 for a vertex) and whether it is a double root (a neck).
struct RawCandidate {
  vec2 center;
  double firstT;
  double secondT;
  bool tangent;
  // Sign of cross(first contact - center, second contact - center): which
  // way round the ball the arc runs. Known from how the ball was
  // found, where the computed points - antipodal for a ball stuck between
  // parallel edges - would only give noise.
  int turn;
};

// All balls of one kind between two edges, without testing that a tangent
// point lies on its edge or a center in its vertex's sector. Deterministic:
// every caller asking for the same (first, second, kind) gets the same bits.
// Whether the lines of two edges can share a ball at all. Parallel lines
// share no single one. Otherwise the two tangent points are r |n1 - n2|
// apart, so edges facing the same way that lie further apart than that along
// their direction - pieces of one straight line, typically - share none
// either. Decided once per pair of edges: the edge limits read it too
// (LineLimit), so a ball is never left to a pair that has none.
bool LinesShareBall(const FilletInput& input, int a, int b, double radius) {
  const int first = std::min(a, b), second = std::max(a, b);
  const vec2 firstStart = EdgeStart(input, first);
  const vec2 firstEnd = EdgeEnd(input, first);
  const vec2 secondStart = EdgeStart(input, second);
  const vec2 secondEnd = EdgeEnd(input, second);
  if (LineCross(input, first, second) == 0) return false;
  const vec2 firstNormal = LineNormal(input, first);
  const vec2 secondNormal = LineNormal(input, second);
  // Nor do lines whose normals are exactly parallel: every expression for
  // their ball then vanishes identically, and nothing could cut it.
  if (la::cross(firstNormal, secondNormal) == 0) return false;
  if (!(dot(firstNormal, secondNormal) > 0)) return true;
  const vec2 firstVec = firstEnd - firstStart;
  const double len = length(firstVec);
  const double s0 = dot(secondStart - firstStart, firstVec) / len;
  const double s1 = dot(secondEnd - firstStart, firstVec) / len;
  const double gap = std::max(std::min(s0, s1) - len, -std::max(s0, s1));
  return !(gap > radius * length(firstNormal - secondNormal));
}

// For two lines whose normals face each other, the gap between them against
// the ball's width, as seen from the start of the first: k_x + k_y for lines
// n . x = k tangent to the ball, with that start as origin. Negative where
// the ball fits. One number for the pair - every decision about a ball in the
// channel between two such lines starts from it and adds only how the gap
// changes along the channel.
double FacingGap(const FilletInput& input, int a, int b, double radius) {
  const int x = std::min(Run(input, a), Run(input, b));
  const int y = std::max(Run(input, a), Run(input, b));
  return dot(LineNormal(input, y), LineStart(input, y) - LineStart(input, x)) +
         2 * radius;
}

// dist(S, M) - r for S = v + n r, the center of the ball tangent to the line
// of edge `line` at its point v, and M the line of edge m facing it: from the
// gap between the two lines and how it changes up to S.
double FacingBeyond(const FilletInput& input, int line, int m, vec2 v,
                    double radius) {
  // Between lines parallel as far as their angles tell, the normals cancel
  // and the gap is the same all along: what rounding leaves of their sum
  // would tilt one line against the other, differently for every caller.
  // The one number for the pair is then all there is.
  if (LineCross(input, line, m) == 0) return -FacingGap(input, line, m, radius);
  const vec2 nL = LineNormal(input, line), nM = LineNormal(input, m);
  const vec2 origin =
      LineStart(input, std::min(Run(input, line), Run(input, m)));
  return dot(nL + nM, (v - origin) + nL * radius) -
         FacingGap(input, line, m, radius);
}

// The signed distance s of a point from the line of an edge, on its ball
// side. One expression for every use: how many balls the line shares with a
// vertex there, where they are, and which side of an end of the edge each is
// on all depend on it. Measured from the nearer end of the line: a vertex
// that all but coincides with that end - another loop touching there - is
// then placed as accurately as the two points are apart, not as the line is
// long.
double HeightOverLine(const FilletInput& input, int line, vec2 p) {
  const vec2 start = LineStart(input, line), end = LineEnd(input, line);
  const vec2 fromStart = p - start, fromEnd = p - end;
  return dot(
      dot(fromEnd, fromEnd) < dot(fromStart, fromStart) ? fromEnd : fromStart,
      LineNormal(input, line));
}

// How many balls the line of an edge shares with the vertex at the end of
// edge `vertexEdge`: none unless 0 <= s <= 2 r, and a double one (the ball
// fits exactly through the neck) at s == 2 r. A vertex computed to be on the
// line itself counts as on the ball side: it is then usually a distinct
// vertex a rounding error away from the edge's own end, which the ball does
// get to pivot around. Decided once per line and vertex: candidate
// generation and the vertex's sector both read it, so a ball is never left
// to an edge that has none.
// Where one of the vertex's own edges faces the line, the two can form a
// channel just as wide as the ball, and whether the vertex is within the
// ball's width of the line is then the same question as whether the ball
// tangent to that edge at the vertex is cut by the line (FacingBeyond):
// s - 2 r = that ball's distance beyond r, less r (1 + n . n_M). It is taken
// from there, so the two cannot disagree.
// `within`, if asked for, is 2 r - s as it was decided: the centers are
// placed from it (LineVertexCenters), so that two balls counted as distinct
// are distinct, on either side of the vertex's foot.
int LineVertexBalls(const FilletInput& input, int line, int vertexEdge,
                    double radius, double* within = nullptr) {
  const vec2 vertex = EdgeEnd(input, vertexEdge);
  const double s = HeightOverLine(input, line, vertex);
  if (s < 0) return 0;
  const vec2 nM = LineNormal(input, line);
  // Either of the vertex's edges that faces the line: if the ball tangent
  // to one of them at the vertex is found cut by the line, the vertex is
  // within the ball's width of it, whatever the other one says - that edge
  // then hands its ball over to the vertex, which must have one to take.
  double margin = 0;
  bool faces = false;
  for (const int j : {vertexEdge, NextEdge(input, vertexEdge)}) {
    const vec2 nJ = LineNormal(input, j);
    if (!(dot(nM, nJ) < 0)) continue;
    const vec2 sum = nM + nJ;
    const double tilt =
        LineCross(input, j, line) == 0 ? 0 : 0.5 * radius * dot(sum, sum);
    const double m = tilt - FacingBeyond(input, j, line, vertex, radius);
    if (!faces || m > margin) margin = m;
    faces = true;
    // An edge parallel to the line as far as their angles tell shares no
    // ball with it, so it is the same distance from it at both of its ends:
    // both of its vertices take the one value, measured at its start, so
    // that the edges next to it do not find the ball cut at one of them
    // and clear at the other.
    if (LineCross(input, j, line) == 0) {
      margin = -FacingBeyond(input, j, line, LineStart(input, j), radius);
      break;
    }
  }
  if (!faces) margin = 2 * radius - s;
  if (within) *within = margin;
  return margin < 0 ? 0 : margin == 0 ? 1 : 2;
}

// Those balls, with their parameters along the edge itself.
IntersectResult LineVertexCenters(const FilletInput& input, int line,
                                  int vertexEdge, double radius) {
  IntersectResult result;
  double within = 0;
  const int count = LineVertexBalls(input, line, vertexEdge, radius, &within);
  if (count == 0) return result;
  const vec2 vertex = EdgeEnd(input, vertexEdge);
  const vec2 start = LineStart(input, line);
  const vec2 edge = LineEnd(input, line) - start;
  const vec2 normal = LineNormal(input, line);
  // The centers are r^2 - (r - s)^2 = s (2 r - s) along the line from the
  // vertex's foot, squared: accurate also for a vertex nearly on the line,
  // where comparing r^2 with the squared distance to the offset line would
  // round s away.
  const double lenSq = dot(edge, edge);
  // s itself where it may be small; with a facing edge it is close to 2 r,
  // and what is left of 2 r is what was decided.
  const double s = HeightOverLine(input, line, vertex);
  const double discriminant = std::max(
      0.0, s <= radius ? s * (2 * radius - s) : (2 * radius - within) * within);
  const double projection = dot(vertex - start, edge) / lenSq;
  const double reach = count == 1 ? 0 : std::sqrt(discriminant / lenSq);
  const vec2 pieceStart = EdgeStart(input, line);
  const vec2 piece = EdgeEnd(input, line) - pieceStart;
  const vec2 offset = pieceStart + normal * radius;
  for (int i = 0; i < count; ++i) {
    const double t = i == 0 ? projection - reach : projection + reach;
    result.points[i] = start + normal * radius + edge * t;
    result.params[i] =
        dot(result.points[i] - offset, piece) / dot(piece, piece);
  }
  result.count = count;
  result.tangent = count == 1;
  return result;
}

struct RawCandidates {
  int count = 0;
  std::array<RawCandidate, 2> balls;
  void push_back(const RawCandidate& raw) { balls[count++] = raw; }
  size_t size() const { return count; }
  const RawCandidate& operator[](size_t i) const { return balls[i]; }
};

int Slope(const FilletInput& input, int m, int line, double radius);

// Whether every ball tangent to edge y within its ends touches the line of
// edge x within x's limits - its ends, or the corner balls there: the balls
// at the two ends of y do, and the rest lie between them.
bool Covers(const FilletInput& input, int x, int y, double radius) {
  const vec2 origin = LineStart(input, x);
  const vec2 d = LineEnd(input, x) - origin;
  const double dd = dot(d, d);
  auto limit = [&](bool atEnd) {
    const int before = atEnd ? x : input.prevEdges[x];
    const int other = atEnd ? NextEdge(input, x) : before;
    double at =
        dot((atEnd ? EdgeEnd(input, x) : EdgeStart(input, x)) - origin, d) / dd;
    if (input.endKinds[before] == kCorner) {
      const vec2 n = LineNormal(input, x), no = LineNormal(input, other);
      const double along = dot((n + no) * (radius / (1 + dot(n, no))), d) / dd;
      if (std::isfinite(along)) at += along;
    }
    return at;
  };
  const double lo = limit(false), hi = limit(true);
  const vec2 up = LineNormal(input, y) * radius;
  for (const vec2 v : {EdgeStart(input, y), EdgeEnd(input, y)}) {
    const double t = dot(v + up - origin, d) / dd;
    if (!(lo < t && t < hi)) return false;
  }
  return true;
}

// How many balls pass through the vertices at the ends of two edges: as
// CountArcArc, but where an edge at one of them is parallel to and faces an
// edge at the other - the two sides of a channel, the vertices at its mouth -
// from the number that already says whether the ball fits between those two
// lines. With `within` what the one vertex is short of the ball's width from
// the other's line, and `along` how far it is from the other vertex along
// that line, 4 r^2 - |P - Q|^2 = within (4 r - within) - along^2. Computed
// from the two points alone, a mouth exactly as wide as the channel could
// stop a ball that the channel lets through.
int VertexPairBalls(const FilletInput& input, int first, int second,
                    double radius) {
  const vec2 p = EdgeEnd(input, first), q = EdgeEnd(input, second);
  if (SamePoint(p, q)) return 0;
  for (const bool swap : {false, true}) {
    const int lineEdge = swap ? second : first;
    const int vertexEdge = swap ? first : second;
    const vec2 from = swap ? q : p, to = swap ? p : q;
    for (const int line : {lineEdge, NextEdge(input, lineEdge)}) {
      bool channel = false;
      for (const int j : {vertexEdge, NextEdge(input, vertexEdge)})
        channel =
            channel || (LineCross(input, j, line) == 0 &&
                        dot(LineNormal(input, j), LineNormal(input, line)) < 0);
      if (!channel) continue;
      double within = std::numeric_limits<double>::quiet_NaN();
      LineVertexBalls(input, line, vertexEdge, radius, &within);
      if (std::isnan(within)) continue;
      const vec2 d = LineEnd(input, line) - LineStart(input, line);
      const double along = dot(to - from, d) / length(d);
      const double f = within * (4 * radius - within) - along * along;
      return f < 0 ? 0 : f == 0 ? 1 : 2;
    }
  }
  return CountArcArc(p, q, radius);
}

RawCandidates FindRawCandidates(const FilletInput& input, int first, int second,
                                int kind, double radius) {
  const vec2 firstStart = EdgeStart(input, first);
  const vec2 firstEnd = EdgeEnd(input, first);
  const vec2 secondStart = EdgeStart(input, second);
  const vec2 secondEnd = EdgeEnd(input, second);
  const vec2 firstVec = firstEnd - firstStart;
  const vec2 secondVec = secondEnd - secondStart;
  const vec2 firstNormal = LineNormal(input, first);
  const vec2 secondNormal = LineNormal(input, second);

  // A vertex on the line of the other edge can only be touched at the tangent
  // point itself, a zero-length arc; skip vertex-line pairs where the vertex
  // is an endpoint of that edge.
  auto endOnEdge = [&](int vertexEdge, int edge) {
    const int v = EndPoint(input, vertexEdge);
    return v == StartPoint(input, edge) || v == EndPoint(input, edge);
  };

  RawCandidates raws;
  if (kind == kLineLine) {
    if (!LinesShareBall(input, first, second, radius)) return raws;
    vec2 center;
    double firstT, secondT;
    // Edges meeting at a point: their lines cross exactly there, and the
    // center lies along the bisector of the normals. Unlike intersecting the
    // offset lines, this stays accurate for nearly collinear edges.
    const bool firstThenSecond = LinesMeet(input, first, second);
    const bool secondThenFirst = LinesMeet(input, second, first);
    bool found = false;
    if (firstThenSecond != secondThenFirst) {
      // Normals exactly opposite have no bisector; the division then
      // leaves nothing finite and the lines are intersected instead.
      const double den = 1 + dot(firstNormal, secondNormal);
      center =
          (firstThenSecond ? LineEnd(input, first) : LineEnd(input, second)) +
          (firstNormal + secondNormal) * (radius / den);
      found = IsFinite(center);
    }
    if (!found)
      found = IntersectLines(firstStart + firstNormal * radius, firstVec,
                             secondStart + secondNormal * radius, secondVec,
                             center, firstT, secondT);
    const double firstLenSq = dot(firstVec, firstVec);
    if (!found) {
      // Too close to parallel to intersect in floating point. Where along
      // the lines the ball sits is then lost to rounding; if it is on both
      // edges (decided by Passes) it is where they run side by side.
      const double a = dot(secondStart - firstStart, firstVec) / firstLenSq;
      const double b = dot(secondEnd - firstStart, firstVec) / firstLenSq;
      const double t =
          0.5 * (std::max(0.0, std::min(a, b)) + std::min(1.0, std::max(a, b)));
      center = firstStart + firstVec * t + firstNormal * radius;
    }
    // The tangent points are the center's projections onto the edges, which
    // is well-conditioned where the solver's own parameters are not.
    firstT = dot(center - firstStart, firstVec) / firstLenSq;
    secondT = dot(center - secondStart, secondVec) / dot(secondVec, secondVec);
    // The contacts are at -r n, and the normals turn as the edges do.
    // Which way round the ball the arc runs follows the sign of the cross
    // product of the two directions. Between lines facing each other along
    // a channel that is also which way the channel closes, and there the
    // shorter one - the one whose balls all land within the other - has
    // said so already, by the values at its two ends (Slope). The longer
    // one could only repeat the question at its far ends, where the answer
    // is the rounding of a distance that hardly changes over its length.
    int turn = LineCross(input, first, second);
    if (input.kindsReady && dot(firstNormal, secondNormal) < 0) {
      const int sigma = input.invert ? -1 : 1;
      int closes = 0;
      if (Covers(input, first, second, radius))
        closes = sigma * Slope(input, first, second, radius);
      else if (Covers(input, second, first, radius))
        closes = -sigma * Slope(input, second, first, radius);
      if (closes != 0) turn = closes;
    }
    raws.push_back({center, firstT, secondT, false, turn});
  } else if (kind == kLineVertex) {
    if (!TouchableEnd(input, second) || endOnEdge(second, first)) return raws;
    const IntersectResult ev = LineVertexCenters(input, first, second, radius);
    for (int i = 0; i < ev.count; ++i)
      // The ball behind the vertex along the line has it on its left
      // (right with invert).
      raws.push_back({ev.points[i], ev.params[i], 1, ev.tangent,
                      (i == 0) != input.invert ? 1 : -1});
  } else if (kind == kVertexLine) {
    if (!TouchableEnd(input, first) || endOnEdge(first, second)) return raws;
    const IntersectResult ve = LineVertexCenters(input, second, first, radius);
    for (int i = 0; i < ve.count; ++i)
      raws.push_back({ve.points[i], 1, ve.params[i], ve.tangent,
                      (i == 0) != input.invert ? -1 : 1});
  } else {
    if (!TouchableEnd(input, first) || !TouchableEnd(input, second))
      return raws;
    const IntersectResult vv =
        IntersectArcArc(firstEnd, secondEnd, radius,
                        VertexPairBalls(input, first, second, radius));
    for (int i = 0; i < vv.count; ++i)
      // The first ball is right of the line from the first vertex to the
      // second.
      raws.push_back({vv.points[i], 1, 1, vv.tangent, i == 0 ? -1 : 1});
  }
  return raws;
}

// Where a ball's center lies relative to the sector of the vertex at the end
// of `edge`: strictly inside, or past the normal of the edge it ends (`in`
// fails) or of the edge it starts (`out` fails), where the ball would be
// tangent to that edge instead. Computed in floating point: InVertexSector
// replaces a side by the edge's own decision where the two could
// disagree.
struct SectorTest {
  bool inside;
  bool narrow;
  bool in;
  bool out;
};

SectorTest TestSector(const FilletInput& input, int edge, vec2 center) {
  const int next = NextEdge(input, edge);
  const vec2 v = center - EdgeEnd(input, edge);
  const vec2 normalIn = LineNormal(input, edge);
  const vec2 normalOut = LineNormal(input, next);
  const bool ccw = input.invert;
  SectorTest test{};
  // A sector of exactly a half-turn - the tip of a spike folded back on
  // itself - has no single side test per normal. The boundary never turns
  // by more than that, so a cross product of the wrong sign is rounding: of
  // a half-turn where the normals point apart, and of hardly any turn at
  // all where they point the same way. There the sector is as narrow as
  // can be, not all the way round.
  test.narrow = (ccw ? la::cross(normalIn, normalOut)
                     : la::cross(normalOut, normalIn)) >= 0 ||
                dot(normalIn, normalOut) > 0;
  if (!test.narrow) {
    test.inside = IsVectorInSector(v, normalIn, normalOut, ccw);
    return test;
  }
  test.in = (ccw ? la::cross(normalIn, v) : la::cross(v, normalIn)) > 0;
  test.out = (ccw ? la::cross(v, normalOut) : la::cross(normalOut, v)) > 0;
  test.inside = test.in && test.out;
  return test;
}

// The ball X tangent to the lines of edges e1 and e2. With W the point where
// the lines cross - the shared vertex itself for edges that meet -
// X = W + r (n1 + n2) / (1 + n1 . n2).
struct PairBall {
  const FilletInput& input;
  double radius;
  // No ball: the lines are parallel.
  bool exists;
  // X = base + offset.
  vec2 base, offset;

  PairBall(const FilletInput& in, int e1, int e2, double r)
      : input(in), radius(r) {
    const vec2 p0 = LineStart(in, e1), p1 = LineEnd(in, e1);
    const vec2 q0 = LineStart(in, e2), q1 = LineEnd(in, e2);
    const bool meetAtEnd1 = LinesMeet(in, e1, e2);
    const bool meetAtEnd2 = LinesMeet(in, e2, e1);
    const bool meet = meetAtEnd1 != meetAtEnd2;
    const vec2 n = LineNormal(in, e1), n2 = LineNormal(in, e2);
    const double den = 1 + dot(n, n2);
    const vec2 d = p1 - p0, d2 = q1 - q0;
    const double cross = la::cross(d, d2);
    exists = den > 0 && (meet || cross != 0);
    if (!exists) return;
    offset = (n + n2) * (radius / den);
    base = meet ? (meetAtEnd1 ? p1 : q1)
                : p0 + d * (la::cross(q0 - p0, d2) / cross);
  }

  // dist(X, P) - r has this sign; negative if P cuts the ball.
  double PointValue(vec2 p) const {
    const vec2 x = (base - p) + offset;
    return dot(x, x) - radius * radius;
  }

  // Signed distance of X to the line of edge m, on its ball side.
  double LineValue(int m) const {
    const vec2 n = LineNormal(input, m);
    return dot(n, base - LineStart(input, m)) + dot(n, offset);
  }

  // (X - P) . u: which way along a direction X lies from P.
  double Along(vec2 p, vec2 u) const {
    return dot(base - p, u) + dot(offset, u);
  }
};

// Whether the line of edge c cuts the ball tangent to the lines of edges a
// and b. With each line as n . x = k for a ball tangent to it (k including
// the radius), the ball's distance to c beyond r is -G / cross(n_a, n_b),
// where
//   G = cross(n_b, n_c) k_a + cross(n_c, n_a) k_b + cross(n_a, n_b) k_c
// is the same number for all three ways of picking the pair out of the
// triple. So the three answers follow from one sign, taken once per triple,
// and cannot disagree, whatever rounding does to it.
//
// G is evaluated through the pair (x, y) closest to parallel, as
//   cross(n_z, n_x) W + (small terms),  W = k_x + k_y  or  k_y - k_x
// for normals facing each other or the same way. W is the gap between the
// two lines against the ball's width, and it is one number for the pair:
// every triple that contains two opposite sides of a channel reads the same
// "does the ball fit between them", and can only differ from the others
// where the small terms - how the gap changes along the channel - matter.
// Signs of cross products are those already decided for the lines
// (LineCross). A ball only just touching c still fits.
bool TripleCut(const FilletInput& input, int c, int a, int b, double radius) {
  // By line: pieces of one split edge are the same line, which touches the
  // ball if it is one of the two and does not cut it.
  a = Run(input, a);
  b = Run(input, b);
  c = Run(input, c);
  if (c == a || c == b) return false;
  const std::array<int, 3> e{a, b, c};
  // Pair (e[i+1], e[i+2]) leaves out e[i].
  // A pair that meets at a vertex with its normals on the same side comes
  // first: its ball is known there however small the angle between the two
  // (see below), where the general expression loses it.
  auto meets = [&](int i) {
    const int p = e[(i + 1) % 3], q = e[(i + 2) % 3];
    return dot(LineNormal(input, p), LineNormal(input, q)) > 0 &&
           (LinesMeet(input, p, q) || LinesMeet(input, q, p));
  };
  int best = -1;
  double bestCross = 0;
  bool bestMeets = false;
  for (int i = 0; i < 3; ++i) {
    const double cross = std::abs(la::cross(LineNormal(input, e[(i + 1) % 3]),
                                            LineNormal(input, e[(i + 2) % 3])));
    const bool meet = meets(i);
    if (best < 0 || (meet && !bestMeets) ||
        (meet == bestMeets &&
         (cross < bestCross || (cross == bestCross && e[i] < e[best])))) {
      best = i;
      bestCross = cross;
      bestMeets = meet;
    }
  }
  int x = e[(best + 1) % 3], y = e[(best + 2) % 3];
  const int z = e[best];
  // (x, y, z) is (a, b, c) in cyclic order; putting the pair in a fixed order
  // may reverse it, and G with it.
  int parity = 1;
  if (x > y) {
    std::swap(x, y);
    parity = -1;
  }
  const vec2 nx = LineNormal(input, x), ny = LineNormal(input, y),
             nz = LineNormal(input, z);
  const vec2 origin = LineStart(input, x);
  const double kx = radius;
  const double ky = dot(ny, LineStart(input, y) - origin) + radius;
  const double kz = dot(nz, LineStart(input, z) - origin) + radius;
  const double dzx = LineCross(input, z, x) * std::abs(la::cross(nz, nx));
  const double dxy = LineCross(input, x, y) * std::abs(la::cross(nx, ny));
  double g;
  const bool meet = LinesMeet(input, x, y);
  if (bestMeets) {
    // Two lines that meet at a vertex and run nearly the same way: G and
    // cross(n_x, n_y) both vanish with the angle between them, and rounding
    // would decide their ratio. But the ball tangent to both is known
    // without either - at the vertex, along the bisector of the normals -
    // and G is cross(n_x, n_y) times how far short of r it is from z.
    const vec2 corner = meet ? LineEnd(input, x) : LineEnd(input, y);
    const vec2 center = corner + (nx + ny) * (radius / (1 + dot(nx, ny)));
    // Against a line facing one of the pair across a channel, measured as
    // every other ball in that channel is: from the gap between z and the
    // line of the pair that faces it most squarely, at the point where
    // this ball touches that line.
    const int w = dot(nz, nx) <= dot(nz, ny) ? x : y;
    const vec2 nw = w == x ? nx : ny;
    double beyond =
        dot(nz, nw) < 0
            ? FacingBeyond(input, w, z, center - nw * radius, radius)
            : dot(nz, center - LineStart(input, z)) - radius;
    // And from how far the vertex itself is within the ball's width of z,
    // where that is known: the number the two edges read for their own
    // balls at the vertex (EndValue), which this ball lies between.
    double within = std::numeric_limits<double>::quiet_NaN();
    LineVertexBalls(input, z, input.runLast[meet ? x : y], radius, &within);
    if (!std::isnan(within)) {
      // dist(ball, z) - r = r (1 + n_z . (n_x + n_y) / (1 + n_x . n_y)) -
      // within, and the bracket is (n_z + n_w) . (n_x + n_y) / (1 + n_x .
      // n_y) for w either of x and y. With w the one z faces, n_z + n_w is
      // the small vector that line's own ball at the vertex is measured by,
      // and a corner next to a channel reads the channel's width as the
      // edge along it does.
      // Split once more as |n_z + n_w|^2 + (n_z + n_w) . (n_o - n_w), o
      // the other of the two: the square is what w's own ball is measured
      // by and is never negative, and the rest is a product of two small
      // vectors where the edges hardly turn.
      const vec2 u = nz + nw;
      const vec2 turn = (w == x ? ny : nx) - nw;
      beyond = radius * (dot(u, u) + dot(u, turn)) / (1 + dot(nx, ny)) - within;
      // For z parallel to w as far as their angles tell, n_z + n_w is zero
      // and what is computed for it is what rounding left of two normals
      // that should cancel: its direction would decide a channel of exactly
      // the ball's width one way at one end of a strip and the other way
      // at the other. The gap itself is all there is.
      if (LineCross(input, z, w) == 0) beyond = -within;
    }
    g = -LineCross(input, x, y) * beyond;
  } else {
    g = dot(nx, ny) < 0
            ? dzx * FacingGap(input, x, y, radius) +
                  la::cross(nx + ny, nz) * kx + dxy * kz
            : dzx * (ky - kx) + la::cross(ny - nx, nz) * kx + dxy * kz;
    if (LineCross(input, x, y) == 0 && dot(nx, ny) < 0)
      g = dzx * FacingGap(input, x, y, radius);
  }
  const int signG = g > 0 ? parity : g < 0 ? -parity : 0;
  return signG * LineCross(input, a, b) > 0;
}

// At a vertex V where the boundary turns towards the ball, between edges
// `before` and the next, the ball X tangent to both is where the ball rolling
// along either edge stops. Whether feature f cuts X: one sign per (f, V)
// decides whether X is blocked by f and, for the balls touching f and either
// edge, whether they stop before X - so those decisions cannot disagree.
bool CornerCut(const FilletInput& input, int f, int before, double radius) {
  const int after = NextEdge(input, before);
  if (!IsVertex(f))
    return TripleCut(input, FeatureEdge(f), before, after, radius);
  const PairBall ball(input, before, after, radius);
  return ball.exists && ball.PointValue(EdgeStart(input, FeatureEdge(f))) < 0;
}

// The limit of edge `line` at an end in a corner (kCorner): whether the ball
// touching g and the line (root `root` of `roots`) stops before the corner
// ball there. Past it, the corner's other edge cuts the ball.
int Slope(const FilletInput& input, int m, int line, double radius);

bool CornerLimit(const FilletInput& input, int g, int line, bool atEnd,
                 int root, int roots, double radius) {
  const int before = atEnd ? line : input.prevEdges[line];
  const int other = atEnd ? NextEdge(input, line) : before;
  // The corner ball itself.
  if (!IsVertex(g) && Run(input, FeatureEdge(g)) == Run(input, other))
    return true;
  const bool cut = CornerCut(input, g, before, radius);
  if (!IsVertex(g)) {
    // dist(., M) - r grows along the line where it leans away from M.
    const int m = FeatureEdge(g);
    const int slope = Slope(input, m, line, radius);
    return atEnd ? (cut ? slope < 0 : slope > 0)
                 : (cut ? slope > 0 : slope < 0);
  }
  // X inside P's circle separates the two roots; else both lie on the side
  // of X that P's foot does.
  if (cut) return roots == 2 ? root == (atEnd ? 0 : 1) : atEnd;
  const PairBall ball(input, before, NextEdge(input, before), radius);
  if (!ball.exists) return false;
  const double foot =
      -ball.Along(EdgeStart(input, FeatureEdge(g)),
                  EdgeEnd(input, line) - EdgeStart(input, line));
  return atEnd ? foot <= 0 : foot >= 0;
}

// dist(S, M) - r, up to a positive factor, for S = v + n r, the center of the
// ball tangent to the line of edge `line` at its point v, and M the line of
// edge m. For lines facing each other, from the gap between them
// (FacingBeyond), so that it agrees with every other decision about the
// same channel; otherwise directly, as |dM| |dL| (dist(S, M) - r).
double LineBeyond(const FilletInput& input, int line, int m, vec2 v,
                  double radius) {
  if (dot(LineNormal(input, line), LineNormal(input, m)) < 0)
    return FacingBeyond(input, line, m, v, radius);
  const double sigma = input.invert ? -1 : 1;
  const vec2 dL = LineEnd(input, line) - LineStart(input, line);
  const vec2 m0 = LineStart(input, m);
  const vec2 dM = LineEnd(input, m) - m0;
  return length(dL) * (sigma * la::cross(dM, v - m0) - radius * length(dM)) +
         radius * dot(dM, dL);
}

// Whether the point p is strictly inside that ball: |S - P|^2 - r^2 =
// |v - p|^2 - 2 r s, as v is on the line.
bool PointInBall(const FilletInput& input, int line, vec2 v, vec2 p,
                 double radius) {
  const vec2 w = v - p;
  return dot(w, w) - 2 * radius * HeightOverLine(input, line, p) < 0;
}

// The edge-or-vertex decision at one end V of edge `line`, for the ball
// touching contact g and the line on root `root` of `roots`: whether its
// tangent point is within the edge at that end (t <= 1 at the end, t >= 0 at
// the start). S = V + n r is the center of the ball tangent to the line at V;
// where it lies relative to g decides, for the edge case and for the vertex
// case at V (which reads its negation), so the two never disagree. Unlike
// the computed tangent parameter, it stays meaningful for nearly parallel
// lines. With n = sigma perp(d) / |d|:
//  - g a line M: S on the edge's side of where M's offset crosses the line's,
//    sign(dist(S, M) - r) against the direction into the edge;
//  - g a vertex P: the roots are where P's circle crosses the line's offset,
//    in order along it; S strictly inside the circle separates them, else
//    both lie on the side of S that P's foot does.
// `strict` leaves a ball tangent exactly at V to the other side.
// Sign of dist(S, M) - r for the ball tangent to edge `line` at one of its
// ends, M the line of edge m: negative if M cuts it.
int EndValue(const FilletInput& input, int m, int line, bool atEnd,
             double radius) {
  const vec2 v = atEnd ? EdgeEnd(input, line) : EdgeStart(input, line);
  // An end of the other line found inside this ball - the decision its
  // own vertex case is made from - means the line cuts the ball: read
  // from there, not computed again.
  if (PointInBall(input, line, v, LineStart(input, m), radius) ||
      PointInBall(input, line, v, LineEnd(input, m), radius))
    return -1;
  // For a line facing the edge, from how far the vertex itself is within the
  // ball's width of that line (LineVertexBalls): the one number the vertex's
  // own balls are counted and placed from, and the one the other edge at
  // the vertex reads. Taken from each edge's own line, the two could say
  // the ball is cut on one side of the vertex and clear on the other, with
  // the vertex left out, and each edge would keep a ball of its own there.
  const vec2 nL = LineNormal(input, line), nM = LineNormal(input, m);
  if (dot(nL, nM) < 0) {
    double within = std::numeric_limits<double>::quiet_NaN();
    LineVertexBalls(input, m, atEnd ? line : input.prevEdges[line], radius,
                    &within);
    if (!std::isnan(within)) {
      const vec2 sum = nL + nM;
      const double f =
          (LineCross(input, line, m) == 0 ? 0 : 0.5 * radius * dot(sum, sum)) -
          within;
      return f > 0 ? 1 : f < 0 ? -1 : 0;
    }
  }
  const double f = LineBeyond(input, line, m, v, radius);
  return f > 0 ? 1 : f < 0 ? -1 : 0;
}

// Which way dist(S, M) - r changes as the ball tangent to edge `line` moves
// along it from start to end: it is linear, so where it has been found to
// have different signs at the two ends of the edge - at an end in a corner,
// for the corner ball there (CornerCut) - it grows from the negative end to
// the positive one, and the ball that just touches M lies between them. The
// direction is read from those signs, not from the cross product of the two
// directions: for lines that are parallel as far as rounding can tell, that
// product could say the distance falls where the two ends say it rises, and
// then neither this edge nor the vertices at its ends would keep the ball.
// Only where the ends agree does the cross product decide.
int Slope(const FilletInput& input, int m, int line, double radius) {
  // Over the whole run of the edge: its two ends are where the values are
  // furthest apart. `at` is where along the run the ball of each value is
  // tangent - the corner ball of a short run can lie before the one at its
  // start.
  const vec2 origin = LineStart(input, line);
  const vec2 d = LineEnd(input, line) - origin;
  auto value = [&](bool atEnd, double& at) {
    const int piece = atEnd ? input.runLast[line] : input.runFirst[line];
    const int before = atEnd ? piece : input.prevEdges[piece];
    const int other = atEnd ? NextEdge(input, piece) : before;
    at = atEnd ? 1 : 0;
    if (input.kindsReady && input.endKinds[before] == kCorner &&
        Run(input, m) != Run(input, other)) {
      const vec2 n = LineNormal(input, piece), no = LineNormal(input, other);
      const double den = 1 + dot(n, no);
      const double along = dot((n + no) * (radius / den), d) / dot(d, d);
      if (std::isfinite(along)) at += along;
      return CornerCut(input, LineFeature(m), before, radius) ? -1 : 1;
    }
    return EndValue(input, m, piece, atEnd, radius);
  };
  double startAt, endAt;
  const int start = value(false, startAt), end = value(true, endAt);
  if (start != end && startAt != endAt)
    return (end > start) == (endAt > startAt) ? 1 : -1;
  return (input.invert ? -1 : 1) * LineCross(input, m, line);
}

bool SegmentLimit(const FilletInput& input, int g, int line, bool atEnd,
                  int root, int roots, double radius, bool strict = false) {
  const vec2 v = atEnd ? EdgeEnd(input, line) : EdgeStart(input, line);
  if (!IsVertex(g)) {
    const int m = FeatureEdge(g);
    const int sf = EndValue(input, m, line, atEnd, radius);
    const int inward = (atEnd ? -1 : 1) * Slope(input, m, line, radius);
    return strict ? sf * inward < 0 : sf * inward <= 0;
  }
  const vec2 p = EdgeStart(input, FeatureEdge(g));
  if (PointInBall(input, line, v, p, radius) && roots == 2)
    return root == (atEnd ? 0 : 1);
  const int foot = -AlongLine(input, line, v, p);
  return atEnd ? foot <= 0 : foot >= 0;
}

// The limit of edge `line` at one end for the ball touching contact g and the
// line: the corner ball in a corner, and the end of the edge itself
// otherwise. Other edges, including a short corner's, are left to Blocks.
bool LineLimit(const FilletInput& input, int g, int line, bool atEnd, int root,
               int roots, double radius) {
  const int kind = input.endKinds[atEnd ? line : input.prevEdges[line]];
  // No ball between these lines at all, so none for this edge to keep.
  if (!IsVertex(g) && !LinesShareBall(input, FeatureEdge(g), line, radius))
    return false;
  // Where the ball between two lines is on each edge is one fact, but each
  // edge reads it from its own numbers: how its distance to the other line
  // changes out to its own ends. Between an edge and a longer one facing it
  // across a channel, the longer one's numbers are the rounding of a
  // distance that hardly changes over its length, and can put beyond its
  // far end a ball the shorter edge holds well within it - which is then
  // neither edge's, nor any vertex's. There the shorter edge alone says
  // where the ball is: every ball tangent to it within its ends touches
  // this line within the edge's limits (Covers), so the ball is on this
  // edge wherever it is on that one.
  // Only for lines nearer parallel than perpendicular, where that holds for
  // a whole stretch of this edge and not just at a point; only for an edge
  // on this line's ball side, where its balls do reach the line; and only
  // if this edge does not say the same of the other, so that one of the two
  // is still asked.
  if (!IsVertex(g) && input.kindsReady) {
    const int m = FeatureEdge(g);
    const vec2 n = LineNormal(input, line), nm = LineNormal(input, m);
    if (std::abs(dot(n, nm)) > std::abs(la::cross(n, nm)) &&
        HeightOverLine(input, line, EdgeStart(input, m)) > 0 &&
        HeightOverLine(input, line, EdgeEnd(input, m)) > 0 &&
        Covers(input, line, m, radius) && !Covers(input, m, line, radius))
      return true;
  }
  if (kind == kCorner)
    return CornerLimit(input, g, line, atEnd, root, roots, radius);
  // The other edge at this vertex: the ball tangent to both stops short of
  // the vertex exactly when the boundary turns towards it. Read from the
  // turn already decided, not computed again.
  if (!IsVertex(g) &&
      Run(input, FeatureEdge(g)) ==
          Run(input, atEnd ? NextEdge(input, line) : input.prevEdges[line]))
    return kind == kShortCorner;
  // Where the boundary runs straight on, a ball tangent exactly at the
  // vertex belongs to the edge ending there.
  return SegmentLimit(input, g, line, atEnd, root, roots, radius,
                      kind == kStraight && !atEnd);
}

// How the boundary turns at the end of each edge, and from that the straight
// runs and their normals. A vertex counts as straight where its three points
// are collinear to within the rounding of their own coordinates: moving the
// middle one by less than the spacing of representable numbers there would
// make them exactly so. That is all an edge split into pieces can be.
void BuildEndKinds(FilletInput& input, double radius) {
  const int numEdges = static_cast<int>(input.edges.size());
  std::vector<int> turns(numEdges);
  std::vector<LineAngle> pieceAngles(numEdges);
  for (int e = 0; e < numEdges; ++e) {
    const vec2 d = EdgeEnd(input, e) - EdgeStart(input, e);
    pieceAngles[e] = AngleOf(d);
  }
  for (int e = 0; e < numEdges; ++e)
    turns[e] = (input.invert ? -1 : 1) *
               TurnSign(pieceAngles[e], pieceAngles[NextEdge(input, e)]);

  input.runFirst.assign(numEdges, -1);
  input.runLast.assign(numEdges, -1);
  for (int e = 0; e < numEdges; ++e) {
    if (turns[input.prevEdges[e]] == 0) continue;
    int last = e;
    while (turns[last] == 0) last = NextEdge(input, last);
    for (int piece = e;; piece = NextEdge(input, piece)) {
      input.runFirst[piece] = e;
      input.runLast[piece] = last;
      if (piece == last) break;
    }
  }
  // All pieces of a run have the same angle.
  input.angles.resize(numEdges);
  for (int e = 0; e < numEdges; ++e) {
    if (input.runFirst[e] < 0) input.runFirst[e] = input.runLast[e] = e;
    input.angles[e] = pieceAngles[input.runFirst[e]];
  }
  input.normals.resize(numEdges);
  for (int e = 0; e < numEdges; ++e) {
    // A closed straight cycle has no start; leave its edges on their own.
    if (input.runFirst[e] < 0) input.runFirst[e] = input.runLast[e] = e;
    input.normals[e] =
        EdgeNormal(LineEnd(input, e) - LineStart(input, e), input.invert);
  }

  input.endKinds.resize(numEdges);
  for (int e = 0; e < numEdges; ++e) {
    if (turns[e] <= 0) {
      input.endKinds[e] = turns[e] < 0 ? kTouchable : kStraight;
      continue;
    }
    // The corner ball must touch each line within its run.
    const int next = NextEdge(input, e);
    const bool real = SegmentLimit(input, LineFeature(next), input.runFirst[e],
                                   false, 0, 1, radius) &&
                      SegmentLimit(input, LineFeature(e), input.runLast[next],
                                   true, 0, 1, radius);
    input.endKinds[e] = real ? kCorner : kShortCorner;
  }
  input.kindsReady = true;
}

// The same decision for the root that can reach V, as the vertex case at V
// needs it.
bool EdgeCaseAt(const FilletInput& input, int g, int line, bool atEnd,
                double radius) {
  if (!IsVertex(g)) return LineLimit(input, g, line, atEnd, 0, 1, radius);
  const vec2 v = atEnd ? EdgeEnd(input, line) : EdgeStart(input, line);
  const int ahead = AlongLine(input, line, v, EdgeStart(input, FeatureEdge(g)));
  return LineLimit(input, g, line, atEnd, ahead > 0 ? 1 : 0, 2, radius);
}

// Which of the two balls touching g and vertex v meets the balls touching g
// and `line` at v (where its center is v + n r), as the side of it the ball's
// center is on: of the line from g's vertex through v (left positive), or
// along g's line from v. Fixed by input coordinates alone.
int MeetingSide(const FilletInput& input, int g, vec2 v, int line) {
  const int sigma = input.invert ? -1 : 1;
  const int m = FeatureEdge(g);
  if (IsVertex(g))
    return sigma * AlongLine(input, line, v, EdgeStart(input, m));
  return sigma * LineCross(input, line, m);
}

// Whether the ball at `center` touching g touches the vertex at the end of
// `edge` strictly within its sector. `side` is the side of the vertex the
// center is on, as defined for MeetingSide (0 for a double ball): on that
// side of the sector the neighbouring edge's own decision is read, so exactly
// one of the edge case and the vertex case claims the ball.
bool InVertexSector(const FilletInput& input, int edge, int g, vec2 center,
                    double radius, int side) {
  // Against a line, the two edges at the vertex have each said whether the
  // ball tangent to them there is cut by it (EndValue). Those two balls are
  // the ends of the sector, and the line keeps the ball out of one arc
  // around the vertex: clear at one end and cut at the other, the ball
  // turning about the vertex meets the line exactly once in between - where
  // it leaves for the line if it started clear, where it arrives from it if
  // it ends clear. That ball is the vertex's, and the other one is not.
  if (!IsVertex(g) && side != 0 && input.kindsReady) {
    const int m = FeatureEdge(g);
    const int in = EndValue(input, m, edge, true, radius);
    const int out = EndValue(input, m, NextEdge(input, edge), false, radius);
    if (in > 0 && out < 0) return side > 0;
    if (in < 0 && out > 0) return side < 0;
  }
  const SectorTest test = TestSector(input, edge, center);
  if (!test.narrow) return test.inside;
  const vec2 v = EdgeEnd(input, edge);
  const int next = NextEdge(input, edge);
  bool in = test.in, out = test.out;
  // A contact that shares no ball with the neighbouring edge's line has no
  // edge case there to hand the ball over to; the sector alone decides.
  auto hasEdgeCase = [&](int line) {
    if (!IsVertex(g))
      return LinesShareBall(input, FeatureEdge(g), line, radius);
    return LineVertexBalls(input, line, input.prevEdges[FeatureEdge(g)],
                           radius) == 2;
  };
  if (side != 0 && hasEdgeCase(edge) && MeetingSide(input, g, v, edge) == side)
    in = !EdgeCaseAt(input, g, edge, true, radius);
  if (side != 0 && hasEdgeCase(next) && MeetingSide(input, g, v, next) == side)
    out = !EdgeCaseAt(input, g, next, false, radius);
  return in && out;
}

bool Passes(const FilletInput& input, int first, int second, int kind,
            const RawCandidates& raws, int root, double radius) {
  const RawCandidate& raw = raws[root];
  const int roots = static_cast<int>(raws.size());
  const int firstFeature = FirstIsVertex(kind)
                               ? VertexFeature(NextEdge(input, first))
                               : LineFeature(first);
  const int secondFeature = SecondIsVertex(kind)
                                ? VertexFeature(NextEdge(input, second))
                                : LineFeature(second);
  // Roots along a line come in order, so the first is behind the vertex and
  // the second ahead. Of two vertices, the first root is right of the line
  // from the first to the second.
  const int ahead = roots < 2 ? 0 : root == 0 ? -1 : 1;
  const int firstSide = kind == kVertexVertex ? -ahead : ahead;

  if (!FirstIsVertex(kind) &&
      !(LineLimit(input, secondFeature, first, false, root, roots, radius) &&
        LineLimit(input, secondFeature, first, true, root, roots, radius)))
    return false;
  if (!SecondIsVertex(kind) &&
      !(LineLimit(input, firstFeature, second, false, root, roots, radius) &&
        LineLimit(input, firstFeature, second, true, root, roots, radius)))
    return false;
  if (FirstIsVertex(kind) && !InVertexSector(input, first, secondFeature,
                                             raw.center, radius, firstSide))
    return false;
  return !SecondIsVertex(kind) ||
         InVertexSector(input, second, firstFeature, raw.center, radius, ahead);
}

// Makes a ball's computed geometry agree with the decisions that accepted
// it: a tangent point decided to be on its edge is moved onto it if rounding
// left it outside (by ulps, or by far for nearly parallel lines, whose
// intersection is ill-conditioned), and the center follows.
void Conform(const FilletInput& input, int first, int second, int kind,
             RawCandidate& raw, double radius) {
  // An edge holds the ball between its ends, and at an end in a corner only
  // as far as the corner ball there (CornerLimit): that is what the ball was
  // accepted within, and where it is put if its computed place is beyond.
  // In a strip just as wide as the ball, closed off at its end, the ball
  // between the two sides is then the one that also touches the end -
  // not one centered on the end itself, bulging out through it.
  auto limit = [&](int e, bool atEnd) {
    const double at = atEnd ? 1 : 0;
    const int before = atEnd ? e : input.prevEdges[e];
    if (!input.kindsReady || input.endKinds[before] != kCorner) return at;
    const int other = atEnd ? NextEdge(input, e) : before;
    const vec2 n = LineNormal(input, e), no = LineNormal(input, other);
    const vec2 d = EdgeEnd(input, e) - EdgeStart(input, e);
    const double along =
        dot((n + no) * (radius / (1 + dot(n, no))), d) / dot(d, d);
    return std::isfinite(along) ? at + along : at;
  };
  auto clampTo = [&](int e, double t) {
    double lo = limit(e, false), hi = limit(e, true);
    if (!(lo <= hi)) lo = 0, hi = 1;
    return !(t > lo) ? lo : t > hi ? hi : t;
  };
  auto offsetPoint = [&](int e, double t) {
    return PointOnEdge(input, e, t) + LineNormal(input, e) * radius;
  };
  auto project = [&](int e, vec2 c) {
    const vec2 start = EdgeStart(input, e);
    const vec2 d = EdgeEnd(input, e) - start;
    return dot(c - start, d) / dot(d, d);
  };
  if (!FirstIsVertex(kind)) {
    const double t = clampTo(first, raw.firstT);
    if (t != raw.firstT) {
      raw.firstT = t;
      raw.center = offsetPoint(first, t);
      if (!SecondIsVertex(kind)) raw.secondT = project(second, raw.center);
    }
  }
  if (!SecondIsVertex(kind)) {
    const double t = clampTo(second, raw.secondT);
    if (t != raw.secondT) {
      raw.secondT = t;
      raw.center = offsetPoint(second, t);
      if (!FirstIsVertex(kind))
        raw.firstT = clampTo(first, project(first, raw.center));
    }
  }
}

void AppendFillets(const FilletInput& input, int first, int second, int kind,
                   const RawCandidate& raw,
                   std::vector<FilletGeometry>& geometries) {
  FilletGeometry geometry;
  for (const bool reverse : {false, true}) {
    if (reverse && !raw.tangent) break;
    if (FinishFilletGeometry(input, first, second, raw.firstT, raw.secondT,
                             FirstIsVertex(kind), SecondIsVertex(kind),
                             raw.center, raw.turn, geometry, reverse))
      geometries.push_back(geometry);
  }
}

// A ball pivoting on a vertex stops at the first thing it meets. Around the
// vertex, a contact y keeps the ball's center out of one arc of the circle
// of radius r: from the ball that leaves the vertex for y, on in the
// direction the ball turns, to the ball that arrives from y. A ball touching
// the vertex is cut by y only if its center is within that arc. All such
// balls are placed around the vertex by one angle each, so two balls that
// leave it for different contacts cannot each be found inside the other's
// arc - which, decided separately on their own distances, they could, leaving
// the vertex with no ball at all.
// Returns whether `fillet`, touching the vertex `pivot`, is within the arc of
// feature y; true if y has no such pair of balls to compare with.
bool InArcAround(const FilletInput& input, const FilletGeometry& fillet,
                 int pivot, int y, double radius) {
  const int pivotEdge = input.prevEdges[FeatureEdge(pivot)];
  const int yEdge =
      IsVertex(y) ? input.prevEdges[FeatureEdge(y)] : FeatureEdge(y);
  if (pivotEdge == yEdge) return true;
  const bool pivotFirst = pivotEdge < yEdge;
  const int kind = IsVertex(y)  ? kVertexVertex
                   : pivotFirst ? kVertexLine
                                : kLineVertex;
  const int first = pivotFirst ? pivotEdge : yEdge;
  const int second = pivotFirst ? yEdge : pivotEdge;
  std::vector<FilletGeometry> rivals;
  const RawCandidates raws =
      FindRawCandidates(input, first, second, kind, radius);
  for (size_t i = 0; i < raws.size(); ++i) {
    AppendFillets(input, first, second, kind, raws[i], rivals);
  }
  if (rivals.size() != 2) return true;
  const FilletGeometry* leaves = nullptr;
  const FilletGeometry* arrives = nullptr;
  for (const FilletGeometry& g : rivals) {
    if (g.from.feature == pivot) leaves = &g;
    if (g.to.feature == pivot) arrives = &g;
  }
  if (!leaves || !arrives || leaves == arrives) return true;
  const vec2 v = EdgeStart(input, FeatureEdge(pivot));
  auto angle = [&](vec2 c) { return std::atan2(c.y - v.y, c.x - v.x); };
  // The ball turns about the vertex from the normal of the edge ending
  // there to that of the edge starting there: counterclockwise with invert.
  const double a = angle(input.invert ? leaves->center : arrives->center);
  const double b = angle(input.invert ? arrives->center : leaves->center);
  const double x = angle(fillet.pivotCenter);
  // Exactly at an end of the arc the two balls are one - where a row of
  // vertices lies along the ball's own tangent, say - and only one of them
  // may be kept: the one whose contact comes first.
  if (x == a || x == b)
    return y < (fillet.from.feature == pivot ? fillet.to : fillet.from).feature;
  return a < b ? a < x && x < b : x > a || x < b;
}

// A ball rolling along a line is cut by a vertex between the two balls that
// are tangent to the line and touch the vertex. Whether the ball tangent to
// the line at `end` is one of those is read from where the three touch the
// line, in order along it - the order their arcs' ends are traced in - so a
// ball between two lines and the balls that touch the vertex and either line
// are told the same thing. A line with no such pair of balls is out of the
// vertex's reach.
// 1 if it is strictly between them, 0 if it is exactly where one of them is,
// -1 otherwise.
int BetweenOnLine(const FilletInput& input, int vertexEdge,
                  const FilletEnd& end, double radius) {
  const IntersectResult balls =
      LineVertexCenters(input, FeatureEdge(end.feature), vertexEdge, radius);
  if (balls.count < 2) return -1;
  if (balls.params[0] < end.t && end.t < balls.params[1]) return 1;
  return end.t == balls.params[0] || end.t == balls.params[1] ? 0 : -1;
}

// An arc by the features it runs between, for looking up whether a given
// one is among the candidates.
uint64_t ArcKey(int from, int to) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(from)) << 32) |
         static_cast<uint32_t>(to);
}

// Whether edge c cuts the fillet ball. Its nearest point to the center is
// either inside the edge (its line) or one of its endpoints (a vertex).
// Endpoints matter even when convex: across a narrow gap a spike tip can be
// the nearest point with both of its edges nearest there too.
bool Blocks(const FilletInput& input, const FilletGeometry& fillet, int c,
            double radius, const std::vector<uint64_t>& arcs) {
  const vec2 p0 = EdgeStart(input, c);
  const vec2 p1 = EdgeEnd(input, c);
  // An edge on a line the ball is tangent to touches it at most.
  for (const int feature : {fillet.from.feature, fillet.to.feature}) {
    if (!IsVertex(feature) && Run(input, FeatureEdge(feature)) == Run(input, c))
      return false;
  }
  const vec2 d = p1 - p0;
  const double t = dot(fillet.center - p0, d) / dot(d, d);

  if (t > 0 && t < 1) {
    const double s = dot(LineNormal(input, c), fillet.center - p0);
    // A ball behind the edge cannot touch it from the ball side.
    if (!(s > 0)) return -s < radius;
    // For a ball tangent to two lines, the cut by a third is one decision
    // shared by the triple.
    const int g = fillet.from.feature, h = fillet.to.feature;
    if (!IsVertex(g) && !IsVertex(h)) {
      // Between two lines that face each other and are nearer parallel than
      // perpendicular - the two sides of a channel - where along them the
      // ball tangent to both sits is, for a channel just as wide as the
      // ball, anywhere: the sign shared by the three lines says nothing
      // about the place the ball was put (Conform). There the ball as
      // placed is asked.
      const vec2 ng = LineNormal(input, FeatureEdge(g));
      const vec2 nh = LineNormal(input, FeatureEdge(h));
      if (dot(ng, nh) < 0 &&
          std::abs(dot(ng, nh)) > std::abs(la::cross(ng, nh)))
        return s < radius;
      return TripleCut(input, c, FeatureEdge(g), FeatureEdge(h), radius);
    }
    if (!(s < radius)) return false;
    return (!IsVertex(g) ||
            InArcAround(input, fillet, g, LineFeature(c), radius)) &&
           (!IsVertex(h) ||
            InArcAround(input, fillet, h, LineFeature(c), radius));
  }
  // A vertex on a line the ball is tangent to is never closer than radius.
  const int point = t <= 0 ? StartPoint(input, c) : EndPoint(input, c);
  for (const int feature : {fillet.from.feature, fillet.to.feature}) {
    if (!IsVertex(feature) &&
        (point == StartPoint(input, FeatureEdge(feature)) ||
         point == EndPoint(input, FeatureEdge(feature))))
      return false;
  }
  // Nor is the far end of an edge at a vertex the ball touches: the ball's
  // center was found within the vertex's sector, on the far side of that
  // edge's normal, which is the same question.
  for (const int feature : {fillet.from.feature, fillet.to.feature}) {
    if (IsVertex(feature) &&
        (point == EndPoint(input, FeatureEdge(feature)) ||
         point == StartPoint(input, input.prevEdges[FeatureEdge(feature)])))
      return false;
  }
  const int g = fillet.from.feature, h = fillet.to.feature;
  // A vertex in a corner that opens towards the ball has no ball of its own
  // around the pivot to be compared with. A ball on the ball side of both
  // edges there that contains it is cut by one of those edges, and for a
  // ball turning about a vertex those edges are compared in order around
  // it: theirs is the decision.
  const int into = t <= 0 ? input.prevEdges[c] : c;
  const vec2 fromCorner = fillet.center - (t <= 0 ? p0 : p1);
  if ((IsVertex(g) || IsVertex(h)) && input.endKinds[into] > kStraight &&
      dot(LineNormal(input, into), fromCorner) > 0 &&
      dot(LineNormal(input, NextEdge(input, into)), fromCorner) > 0)
    return false;
  if (!IsVertex(g) && !IsVertex(h)) {
    // On both of its lines - but the two ask the same thing, and where on
    // one of them the ball is exactly where the vertex's own ball is, as
    // far as positions along that line can tell, the other one decides.
    const int from = BetweenOnLine(input, into, fillet.from, radius);
    const int to = BetweenOnLine(input, into, fillet.to, radius);
    return from >= 0 && to >= 0 && from + to > 0;
  }
  // A ball on one line and one vertex is on that line too, and its place
  // along it between the two balls that touch this vertex decides the same
  // way. Around the vertex it turns about, the ball and the one that
  // touches both vertices can be closer together than their centers tell
  // apart - a ball with no room to spare between the three - while along
  // the line the two balls of this vertex are still well apart.
  // Exactly where one of those is, the ball touches this vertex as well as
  // its own two contacts, and its arc is then two arcs if the vertex is on
  // it, and whole if it is not.
  auto onArc = [&]() {
    const double start = EndAngle(input, fillet.from, fillet.center);
    const double total = ArcAngle(
        start, EndAngle(input, fillet.to, fillet.center), input.invert);
    const double at = ArcAngle(
        start, NormalizeAngle(ToRad((t <= 0 ? p0 : p1) - fillet.center)),
        input.invert);
    return at > 0 && at < total;
  };
  if (!IsVertex(g) || !IsVertex(h)) {
    const int between = BetweenOnLine(
        input, into, IsVertex(g) ? fillet.to : fillet.from, radius);
    return between > 0 || (between == 0 && onArc());
  }
  const int y = VertexFeature(t <= 0 ? c : NextEdge(input, c));
  // Two balls that reach one vertex from different contacts compete for it,
  // and their order around that vertex alone says which of them the other's
  // contact cuts: exactly one, where the distance of each from the other's
  // contact could leave both.
  bool decided = false, within = true;
  // Only a vertex the ball can turn about has balls of its own.
  const bool turns = TouchableEnd(input, t <= 0 ? input.prevEdges[c] : c);
  for (const int pivot : {g, h}) {
    if (!turns || !IsVertex(pivot)) continue;
    // y's own ball at the same end of its arc: the one that competes with
    // this one for the vertex, if the two keep it.
    const uint64_t rival =
        fillet.from.feature == pivot ? ArcKey(pivot, y) : ArcKey(y, pivot);
    if (!std::binary_search(arcs.begin(), arcs.end(), rival)) continue;
    decided = true;
    within = within && InArcAround(input, fillet, pivot, y, radius);
  }
  if (decided) return within;
  if (!(length(fillet.center - (t <= 0 ? p0 : p1)) < radius)) return false;
  return (!IsVertex(g) || InArcAround(input, fillet, g, y, radius)) &&
         (!IsVertex(h) || InArcAround(input, fillet, h, y, radius));
}

// Edges through a point the ball touches are at distance exactly radius from
// its center, so they must not count as blocking it: the edge whose line it
// touches, and every edge ending at a vertex it touches.
bool TouchesContact(const FilletInput& input, const FilletGeometry& fillet,
                    int c) {
  for (const FilletEnd& end : {fillet.from, fillet.to}) {
    if (end.feature == LineFeature(c)) return true;
    // The edges before and after an edge the ball is tangent to were
    // already accounted for when the tangent point was found to be on the
    // edge (LineLimit): past a vertex turning away from the ball they stay
    // behind the tangent line, and in a corner the ball stops before the
    // corner ball. Only a short corner's other edge can still cut it.
    if (IsVertex(end.feature)) continue;
    const int e = FeatureEdge(end.feature);
    if ((c == NextEdge(input, e) && input.endKinds[e] != kShortCorner) ||
        (e == NextEdge(input, c) && input.endKinds[c] != kShortCorner))
      return true;
  }
  for (const FilletEnd& end : {fillet.from, fillet.to}) {
    if (!IsVertex(end.feature)) continue;
    const int after = FeatureEdge(end.feature);
    const int vertex = StartPoint(input, after);
    if (StartPoint(input, c) == vertex || EndPoint(input, c) == vertex)
      return true;
    // Nor can the further pieces of the two lines that meet at the vertex:
    // the sector the ball pivots in ends at their normals, so it stays on
    // its own side of both lines.
    if (Run(input, c) == Run(input, after) ||
        Run(input, c) == Run(input, input.prevEdges[after]))
      return true;
  }
  return false;
}

// An arc too flat to discretize may be skipped only at a plain corner: the
// ball tangent to two adjacent edges, running from the first to the second,
// so the arc angle equals the corner's turn and tracing still reaches the
// next edge via the vertex. Any other arc spans more of the boundary and must
// be kept. Decided from the edges' normals, before anything else is computed
// for the pair: most pairs of short edges along a curve are such corners.
bool FlatCorner(const FilletInput& input, int first, int second,
                double cosMinArc) {
  const int turn = LineCross(input, first, second);
  // The arc runs from `first` unless the edges turn the other way (see
  // FinishFilletGeometry).
  const bool backwards = input.invert ? turn > 0 : turn < 0;
  const int from = backwards ? second : first;
  const int to = backwards ? first : second;
  if (NextEdge(input, from) != to) return false;
  return dot(LineNormal(input, from), LineNormal(input, to)) >= cosMinArc;
}

// A fillet circle is part of the result only if the rolling ball fits there:
// no edge other than the ones it touches may cut it.
std::vector<FilletGeometry> FindFilletPairs(
    const FilletInput& input, const EdgeIndex& index,
    const std::vector<std::pair<int, int>>& broadPairs, double radius) {
  // No reserve: at large radii there can be ~1e9 broad pairs, of which only a
  // small fraction become candidates.
  // Each pair classifies its balls: edge or end vertex on each side. Where a
  // ball could be the neighbouring case, EdgeCaseAt decides for both.
  std::vector<FilletGeometry> candidates;
  const double cosMinArc = std::cos(input.minArcAngle);
  for (const auto& [first, second] : broadPairs) {
    for (const int kind :
         {kLineLine, kLineVertex, kVertexLine, kVertexVertex}) {
      if (kind == kLineLine && FlatCorner(input, first, second, cosMinArc))
        continue;
      const RawCandidates raws =
          FindRawCandidates(input, first, second, kind, radius);
      for (size_t i = 0; i < raws.size(); ++i) {
        if (!Passes(input, first, second, kind, raws, static_cast<int>(i),
                    radius))
          continue;
        RawCandidate raw = raws[i];
        Conform(input, first, second, kind, raw, radius);
        const size_t before = candidates.size();
        AppendFillets(input, first, second, kind, raw, candidates);
        for (size_t k = before; k < candidates.size(); ++k)
          candidates[k].pivotCenter = raws[i].center;
      }
    }
  }

  Vec<Box> circleBoxes;
  circleBoxes.reserve(candidates.size());
  for (const FilletGeometry& geometry : candidates) {
    // The ball's center lies where the pieces of its two contacts meet, so
    // only an edge reaching within the radius of both pieces' boxes can cut
    // it. The computed center is not used here: between nearly parallel
    // edges it can be far from where the decisions place the ball.
    auto piece = [&](const FilletEnd& end) {
      const int edge = FeatureEdge(end.feature);
      return input.pieces[IsVertex(end.feature) ? input.prevEdges[edge] : edge];
    };
    const Box a = piece(geometry.from), b = piece(geometry.to);
    const vec3 r(radius, radius, 0);
    circleBoxes.push_back(Box(max(a.min, b.min) - r, min(a.max, b.max) + r));
  }

  // Every candidate arc, sorted, so that a ball can find the one competing
  // with it for a vertex (Blocks).
  std::vector<uint64_t> arcs;
  arcs.reserve(candidates.size());
  for (const FilletGeometry& geometry : candidates)
    arcs.push_back(ArcKey(geometry.from.feature, geometry.to.feature));
  std::sort(arcs.begin(), arcs.end());

  std::vector<uint8_t> fits(candidates.size(), 1);
  index.ForEach(circleBoxes, [&](int query, int edge) {
    if (!fits[query]) return;
    const FilletGeometry& geometry = candidates[query];

    // Skip topology neighbour geometry because it must touch the arc.
    if (TouchesContact(input, geometry, edge)) return;

    if (Blocks(input, geometry, edge, radius, arcs)) fits[query] = 0;
  });

  std::vector<FilletGeometry> fillets;
  fillets.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (fits[i]) fillets.push_back(candidates[i]);
  }
  return fillets;
}

std::vector<vec2> discreteArcToPoint(const FilletInput& input,
                                     const FilletGeometry& arc, double radius,
                                     int circularSegments) {
  const bool invert = input.invert;
  std::vector<vec2> points;
  points.push_back(EndPoint(input, arc.from));

  const double startRad = EndAngle(input, arc.from, arc.center);
  const double totalRad =
      ArcAngle(startRad, EndAngle(input, arc.to, arc.center), invert);
  // Spread the points evenly: fixed dPhi steps leave a sliver segment at the
  // end when totalRad is just over a multiple of dPhi, whose direction is
  // meaningless after rounding.
  const int segments = std::max(
      1,
      static_cast<int>(std::ceil(totalRad * circularSegments / (2.0 * M_PI))));
  const double step = (invert ? -totalRad : totalRad) / segments;

  for (int i = 1; i < segments; ++i) {
    const double current = startRad + step * i;
    points.push_back({arc.center.x + radius * cos(current),
                      arc.center.y + radius * sin(current)});
  }

  points.push_back(EndPoint(input, arc.to));
  return points;
}

void AppendArcPoints(const FilletInput& input, SimplePolygon& loop,
                     const FilletGeometry& arc, double radius,
                     int circularSegments) {
  const std::vector<vec2> points =
      discreteArcToPoint(input, arc, radius, circularSegments);
  for (size_t i = 1; i < points.size(); ++i) {
    AppendPoint(loop, points[i]);
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
    if (!removed[i])
      starts[Position(input, geometries[i].from).edge].push_back(i);
  }
  for (auto& edgeStarts : starts) {
    std::sort(edgeStarts.begin(), edgeStarts.end(), [&](size_t a, size_t b) {
      return std::make_pair(Position(input, geometries[a].from).t,
                            geometries[a].fromKey) <
             std::make_pair(Position(input, geometries[b].from).t,
                            geometries[b].fromKey);
    });
  }
  auto findNext = [&](size_t i) -> int {
    const BoundaryPos arrival = Position(input, geometries[i].to);
    int edge = arrival.edge;
    double t = arrival.t;
    double key = geometries[i].toKey;
    for (size_t walk = 0; walk <= input.edges.size(); ++walk) {
      for (size_t f : starts[edge]) {
        const FilletGeometry& g = geometries[f];
        const double fromT = Position(input, g.from).t;
        if (removed[f] || fromT < t || (fromT == t && g.fromKey < key))
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
      const BoundaryPos to = Position(input, g.to);
      const BoundaryPos from = Position(input, g.from);
      arrivals[cycleOf[to.edge]].push_back(
          BoundaryEvent{i, posOf[to.edge], to.t, g.toKey});
      departures[cycleOf[from.edge]].push_back(
          BoundaryEvent{i, posOf[from.edge], from.t, g.fromKey});
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
    // Only arcs that amount to nothing may go: one too flat to discretize
    // (or just short of a full turn, which is one of those run the other
    // way), and one of the two half-turns of a ball squeezed through a
    // neck. Having to drop any other means a ball is missing somewhere, and
    // what is left would close up across the gap into a different shape.
    for (size_t i = 0; i < geometries.size(); ++i) {
      if (!removed[i]) continue;
      const FilletGeometry& g = geometries[i];
      const double arc =
          ArcAngle(EndAngle(input, g.from, g.center),
                   EndAngle(input, g.to, g.center), input.invert);
      const double flat = input.minArcAngle;
      if (!(arc < flat || arc > 2 * M_PI - flat || std::abs(arc - M_PI) < flat))
        return {};
    }
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

    // The ball of an arrival rolls on along the boundary until the next
    // event, which is therefore a departure: the events alternate, and
    // arrival k pairs with departure k + shift, shift 0 or 1. Which of the
    // two is not a new question either - a departure before any arrival
    // ends a stretch of the result boundary that the last arrival began, so
    // the start of the cycle is on it and the shift is 1.
    // At the same position the arrival comes first - a ball that arrives
    // and at once leaves for somewhere else - unless the two are the two
    // arcs of one ball squeezed through a neck: there the boundary on
    // either side of the neck crosses over to the other, so each contact
    // sees the departure first.
    auto neck = [&](const BoundaryEvent& d, const BoundaryEvent& a) {
      const FilletGeometry& out = geometries[d.fillet];
      const FilletGeometry& in = geometries[a.fillet];
      return out.from.feature == in.to.feature &&
             out.to.feature == in.from.feature;
    };
    const size_t shift =
        PositionLess(dep[0], arr[0]) ||
                (!PositionLess(arr[0], dep[0]) && neck(dep[0], arr[0]))
            ? 1
            : 0;
    for (size_t k = 0; k < m; ++k) {
      next[arr[k].fillet] = static_cast<int>(dep[(k + shift) % m].fillet);
      wraps[arr[k].fillet] = k + shift >= m;
    }
  }

  Polygons resultLoops;
  std::vector<uint8_t> touchedLoop(input.loops.size(), 0);
  std::vector<uint8_t> used(geometries.size(), 0);

  for (size_t startFillet = 0; startFillet < geometries.size(); ++startFillet) {
    if (used[startFillet] || removed[startFillet]) continue;

    SimplePolygon loop;
    size_t current = startFillet;
    do {
      if (used[current]) return {};
      used[current] = 1;

      const FilletGeometry& fillet = geometries[current];
      const BoundaryPos departure = Position(input, fillet.from);
      const BoundaryPos arrival = Position(input, fillet.to);
      AppendPoint(loop, EndPoint(input, fillet.from));
      AppendArcPoints(input, loop, fillet, radius, circularSegments);
      touchedLoop[input.edges[departure.edge].loop] = 1;
      touchedLoop[input.edges[arrival.edge].loop] = 1;

      if (next[current] < 0) return {};
      const FilletGeometry& nextFillet = geometries[next[current]];
      const BoundaryPos nextDeparture = Position(input, nextFillet.from);
      // Walk the input boundary to the paired departure. Without wrapping, a
      // departure not after the arrival is a tie: the stretch is empty.
      const BoundaryEvent from{current, posOf[arrival.edge], arrival.t,
                               fillet.toKey};
      const BoundaryEvent to{static_cast<size_t>(next[current]),
                             posOf[nextDeparture.edge], nextDeparture.t,
                             nextFillet.fromKey};
      const bool wrap = wraps[current];
      if (wrap || PositionLess(from, to)) {
        int edge = arrival.edge;
        bool first = true;
        while (edge != nextDeparture.edge || (first && wrap)) {
          AppendPoint(loop, EdgeEnd(input, edge));
          touchedLoop[input.edges[edge].loop] = 1;
          edge = NextEdge(input, edge);
          first = false;
        }
      }
      AppendPoint(loop, EndPoint(input, nextFillet.from));
      current = static_cast<size_t>(next[current]);
    } while (current != startFillet);

    if (loop.size() > 1 && SamePoint(loop.front(), loop.back()))
      loop.pop_back();
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

  // Decompose returns the section itself when it has fewer than two contours,
  // including when it is empty.
  const CrossSection result(resultLoops);
  if (result.IsEmpty()) return {};
  return result.Decompose();
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
                         double originalArea, int circularSegments,
                         bool invert) {
  if (result.empty()) return false;
  for (const CrossSection& crossSection : result) {
    if (!HasValidFilletGeometry(crossSection)) return false;
    if (!HasFilletAngles(crossSection, circularSegments, invert)) return false;
  }

  // Decomposed pieces are disjoint, so their areas add up.
  double area = 0;
  for (const CrossSection& crossSection : result) area += crossSection.Area();
  return invert ? area > originalArea : area < originalArea;
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

std::vector<CrossSection> FilletImpl(const Polygons& polygons,
                                     double originalArea, double radius,
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
  std::vector<int> startPoints;
  const std::vector<int> nextEdges =
      BuildNextEdges(polygons, loopOffsets, edges, invert, startPoints);
  std::vector<int> prevEdges(edges.size(), -1);
  for (size_t e = 0; e < edges.size(); ++e) prevEdges[nextEdges[e]] = e;
  for (size_t e = 0; e < edges.size(); ++e) {
    if (prevEdges[e] >= 0) continue;
    const EdgeRef ref = edges[e];
    const size_t n = polygons[ref.loop].size();
    prevEdges[e] =
        static_cast<int>(loopOffsets[ref.loop] + (ref.edge + n - 1) % n);
  }
  FilletInput input{polygons,
                    loopOffsets,
                    edges,
                    nextEdges,
                    prevEdges,
                    startPoints,

                    // FIXME: why minimal radius?
                    invert,
                    2.0 * M_PI / segments,
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    {}};
  input.starts.resize(edges.size());
  input.ends.resize(edges.size());
  for (size_t e = 0; e < edges.size(); ++e) {
    const SimplePolygon& loop = polygons[edges[e].loop];
    input.starts[e] = loop[edges[e].edge];
    input.ends[e] = loop[(edges[e].edge + 1) % loop.size()];
  }
  BuildEndKinds(input, radius);

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

  const EdgeIndex index = BuildEdgeIndex(input);
  const std::vector<std::pair<int, int>> broadPairs =
      BuildCollider(input, radius);
  const std::vector<FilletGeometry> geometries =
      FindFilletPairs(input, index, broadPairs, radius);

  auto result = TraceFillets(input, geometries, radius, segments);
  if (!IsValidFilletResult(result, originalArea, segments, invert))
    result.clear();

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
    return FilletImpl(ToPolygons(), Area(), radius, circularSegments);
  } catch (const std::exception&) {
    return {};
  }
}

}  // namespace manifold

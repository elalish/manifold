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
  // (see EndKind), found exactly.
  std::vector<int8_t> endKinds;
  // The unit normal of each edge, towards the ball.
  std::vector<vec2> normals;
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

// Exact signs of polynomials in the input coordinates and the radius, by
// expansion arithmetic: a value is an unevaluated sum of doubles that do not
// overlap, in increasing magnitude, so its sign is its largest component's.
namespace exact {

using Expansion = std::vector<double>;

void TwoSum(double a, double b, double& x, double& y) {
  x = a + b;
  const double bv = x - a;
  const double av = x - bv;
  y = (a - av) + (b - bv);
}

// Sum of two expansions, by merging their components in order of magnitude
// (Shewchuk's fast expansion sum, dropping zero components).
Expansion Add(const Expansion& e, const Expansion& f) {
  if (e.empty()) return f;
  if (f.empty()) return e;
  Expansion h;
  h.reserve(e.size() + f.size());
  size_t i = 0, j = 0;
  auto next = [&]() {
    if (j == f.size() || (i < e.size() && std::abs(e[i]) < std::abs(f[j])))
      return e[i++];
    return f[j++];
  };
  double q = next();
  bool first = true;
  while (i < e.size() || j < f.size()) {
    const double c = next();
    double x, y;
    if (first) {
      // |c| >= |q|.
      x = c + q;
      y = q - (x - c);
      first = false;
    } else {
      TwoSum(q, c, x, y);
    }
    if (y != 0) h.push_back(y);
    q = x;
  }
  if (q != 0) h.push_back(q);
  return h;
}

Expansion Neg(Expansion e) {
  for (double& c : e) c = -c;
  return e;
}

Expansion Sub(const Expansion& e, const Expansion& f) { return Add(e, Neg(f)); }

// Product of an expansion and a double (Shewchuk's scale expansion).
Expansion Scale(const Expansion& e, double b) {
  Expansion h;
  if (e.empty() || b == 0) return h;
  h.reserve(2 * e.size());
  double q = e[0] * b;
  double low = std::fma(e[0], b, -q);
  if (low != 0) h.push_back(low);
  for (size_t i = 1; i < e.size(); ++i) {
    const double p1 = e[i] * b;
    const double p0 = std::fma(e[i], b, -p1);
    double sum, y;
    TwoSum(q, p0, sum, y);
    if (y != 0) h.push_back(y);
    // |p1| >= |sum|.
    q = p1 + sum;
    y = sum - (q - p1);
    if (y != 0) h.push_back(y);
  }
  if (q != 0) h.push_back(q);
  return h;
}

Expansion Mul(const Expansion& e, const Expansion& f) {
  Expansion h;
  for (const double b : f) h = Add(h, Scale(e, b));
  return h;
}

int Sign(const Expansion& e) {
  if (e.empty()) return 0;
  return e.back() > 0 ? 1 : -1;
}

Expansion Diff(double a, double b) { return Add({a}, {-b}); }

struct Vec {
  Expansion x, y;
};

Vec Sub(vec2 a, vec2 b) { return {Diff(a.x, b.x), Diff(a.y, b.y)}; }

Expansion Cross(const Vec& a, const Vec& b) {
  return Sub(Mul(a.x, b.y), Mul(a.y, b.x));
}

Expansion Dot(const Vec& a, const Vec& b) {
  return Add(Mul(a.x, b.x), Mul(a.y, b.y));
}

// Sign of x + y sqrt(q), q >= 0.
int SignWithRoot(const Expansion& x, const Expansion& y, const Expansion& q) {
  const int sx = Sign(x), sy = Sign(y);
  if (sy == 0 || Sign(q) == 0) return sx;
  if (sx == 0 || sx == sy) return sy;
  return sx * Sign(Sub(Mul(x, x), Mul(Mul(y, y), q)));
}

// A number p + q a + s b + u ab with a = sqrt(A), b = sqrt(B), as its four
// coefficients.
using Rad2 = std::array<Expansion, 4>;

Rad2 Mul(const Rad2& x, const Rad2& y, const Expansion& A, const Expansion& B) {
  const Expansion AB = Mul(A, B);
  auto m = [](const Expansion& e, const Expansion& f) { return Mul(e, f); };
  return {Add(Add(m(x[0], y[0]), m(m(x[1], y[1]), A)),
              Add(m(m(x[2], y[2]), B), m(m(x[3], y[3]), AB))),
          Add(Add(m(x[0], y[1]), m(x[1], y[0])),
              m(Add(m(x[2], y[3]), m(x[3], y[2])), B)),
          Add(Add(m(x[0], y[2]), m(x[2], y[0])),
              m(Add(m(x[1], y[3]), m(x[3], y[1])), A)),
          Add(Add(m(x[0], y[3]), m(x[3], y[0])),
              Add(m(x[1], y[2]), m(x[2], y[1])))};
}

int Sign(const Rad2& x, const Expansion& A, const Expansion& B) {
  const int s1 = SignWithRoot(x[0], x[1], A);
  const int s2 = Sign(B) == 0 ? 0 : SignWithRoot(x[2], x[3], A);
  if (s2 == 0) return s1;
  if (s1 == 0 || s1 == s2) return s2;
  // sign((p + q a)^2 - B (s + u a)^2).
  const Expansion free =
      Sub(Add(Mul(x[0], x[0]), Mul(Mul(x[1], x[1]), A)),
          Mul(B, Add(Mul(x[2], x[2]), Mul(Mul(x[3], x[3]), A))));
  const Expansion root =
      Scale(Sub(Mul(x[0], x[1]), Mul(B, Mul(x[2], x[3]))), 2);
  return s1 * SignWithRoot(free, root, A);
}

}  // namespace exact

// A floating-point value is trusted for its sign when it exceeds this share
// of the sum of the magnitudes of its terms: far above the rounding error of
// the short expressions used here (a few ulps), so the filter never changes a
// sign, it only saves the exact evaluation.
constexpr double kFilter = 1e-12;

// Sign of cross(a - b, c - d), exactly.
int CrossSign(vec2 a, vec2 b, vec2 c, vec2 d) {
  const vec2 u = a - b, w = c - d;
  const double l = u.x * w.y, r = u.y * w.x;
  const double det = l - r;
  if (std::abs(det) > kFilter * (std::abs(l) + std::abs(r)))
    return det > 0 ? 1 : -1;
  return exact::Sign(exact::Cross(exact::Sub(a, b), exact::Sub(c, d)));
}

// Sign of dot(a - b, c - d), exactly.
int DotSign(vec2 a, vec2 b, vec2 c, vec2 d) {
  const vec2 u = a - b, w = c - d;
  const double l = u.x * w.x, r = u.y * w.y;
  const double sum = l + r;
  if (std::abs(sum) > kFilter * (std::abs(l) + std::abs(r)))
    return sum > 0 ? 1 : -1;
  return exact::Sign(exact::Dot(exact::Sub(a, b), exact::Sub(c, d)));
}

// How many balls tangent to the line of an edge (from start along `edge`), on
// its normal side, touch the vertex `center`, exactly: with s the vertex's
// signed distance to the line, none unless 0 < s <= 2 r, and a double one (the
// ball fits exactly through the neck) at s == 2 r.
int CountSegmentArc(vec2 start, vec2 edge, bool invert, vec2 center,
                    double radius) {
  const vec2 end = start + edge;
  const int sigma = invert ? -1 : 1;
  if (sigma * CrossSign(end, start, center, start) <= 0) return 0;
  // sign(2 r - s) = sign(2 r |d| - sigma cross(d, w)).
  const vec2 w = center - start;
  const double len = length(edge);
  const double l = edge.x * w.y, r = edge.y * w.x;
  const double f = 2 * radius * len - sigma * (l - r);
  int sign;
  if (std::abs(f) > kFilter * (2 * radius * len + std::abs(l) + std::abs(r))) {
    sign = f > 0 ? 1 : -1;
  } else {
    const exact::Vec d = exact::Sub(end, start);
    sign = exact::SignWithRoot(
        exact::Scale(exact::Cross(d, exact::Sub(center, start)), -sigma),
        {2 * radius}, exact::Dot(d, d));
  }
  return sign < 0 ? 0 : sign == 0 ? 1 : 2;
}

// Centers of the `count` balls of CountSegmentArc, in order along the edge.
// Measured from the vertex's signed distance s to the edge's own line:
// r^2 - (r - s)^2 = s (2r - s) stays accurate for a vertex lying nearly on the
// line, where comparing r^2 with the squared distance to the offset line
// would round s away below ulp(r).
// All roots, whether or not they lie on the edge or in the vertex's sector:
// those limits are decided by the caller (Passes).
IntersectResult IntersectSegmentArc(vec2 start, vec2 edge, vec2 normal,
                                    vec2 center, double radius, int count) {
  IntersectResult result;
  const double lenSq = dot(edge, edge);
  if (lenSq == 0 || count == 0) return result;

  const vec2 w = center - start;
  const double s = dot(w, normal);
  const double discriminant = std::max(0.0, s * (2 * radius - s));

  const vec2 p = start + normal * radius;
  const double projectionT = dot(w, edge) / lenSq;
  const double tOffset = count == 1 ? 0 : std::sqrt(discriminant / lenSq);
  const std::array<double, 2> values{projectionT - tOffset,
                                     projectionT + tOffset};
  for (int i = 0; i < count; ++i) {
    result.params[result.count] = values[i];
    result.points[result.count++] = p + edge * values[i];
  }
  result.tangent = count == 1;
  return result;
}

// How many balls pass through two vertices, exactly: two closer than 2 r, a
// double one at exactly 2 r.
int CountArcArc(vec2 c1, vec2 c2, double radius) {
  if (SamePoint(c1, c2)) return 0;
  const double dSq = length2(c1 - c2);
  const double f = 4 * radius * radius - dSq;
  int sign;
  if (std::abs(f) > kFilter * (4 * radius * radius + dSq)) {
    sign = f > 0 ? 1 : -1;
  } else {
    const exact::Vec d = exact::Sub(c1, c2);
    sign = exact::Sign(
        exact::Sub(exact::Scale({2 * radius}, 2 * radius), exact::Dot(d, d)));
  }
  return sign < 0 ? 0 : sign == 0 ? 1 : 2;
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

  // An arc too flat to discretize may be skipped only at a plain corner: both
  // tangent points inside two adjacent edges, so the arc angle equals the
  // corner's turn and tracing still reaches the next edge via the vertex. Any
  // other arc spans more of the boundary and must be kept.
  const bool corner =
      !IsVertex(from.feature) && !IsVertex(to.feature) &&
      FeatureEdge(to.feature) == NextEdge(input, FeatureEdge(from.feature));
  const double minArc = corner ? input.minArcAngle : 0;
  if (!(ArcAngle(EndAngle(input, from, center), EndAngle(input, to, center),
                 input.invert) > minArc))
    return false;

  geometry = FilletGeometry{from, to, center, PivotKey(input, from, center),
                            PivotKey(input, to, center)};
  return true;
}

// Lines before vertices, then by edge.
bool FeatureLess(int a, int b) {
  return std::make_pair(IsVertex(a), FeatureEdge(a)) <
         std::make_pair(IsVertex(b), FeatureEdge(b));
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
int TouchingCircles(const FilletInput& input, std::array<int, 3> f,
                    std::array<vec2, 2>& centers,
                    std::array<double, 2>& radii) {
  std::sort(f.begin(), f.end(), FeatureLess);
  const int lines = !IsVertex(f[0]) + !IsVertex(f[1]) + !IsVertex(f[2]);
  std::array<vec2, 3> n, p;
  std::array<double, 3> c;
  for (int i = 0; i < 3; ++i) {
    p[i] = EdgeStart(input, FeatureEdge(f[i]));
    if (!IsVertex(f[i])) {
      n[i] = LineNormal(input, FeatureEdge(f[i]));
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
bool CenterVelocity(const FilletInput& input, int g, int h, vec2 x, double r,
                    vec2& v) {
  auto row = [&](int f, vec2& n, double& rhs) {
    if (IsVertex(f)) {
      n = x - EdgeStart(input, FeatureEdge(f));
      rhs = r;
    } else {
      n = LineNormal(input, FeatureEdge(f));
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
bool SameBranch(const FilletInput& input, int g, int h, vec2 a, vec2 b) {
  if (!IsVertex(g) && !IsVertex(h)) return true;
  double sa, sb;
  if (IsVertex(g) && IsVertex(h)) {
    const vec2 p = EdgeStart(input, FeatureEdge(g));
    const vec2 q = EdgeStart(input, FeatureEdge(h));
    sa = la::cross(q - p, a - p);
    sb = la::cross(q - p, b - p);
  } else {
    const int line = FeatureEdge(IsVertex(g) ? h : g);
    const vec2 p = EdgeStart(input, FeatureEdge(IsVertex(g) ? g : h));
    const vec2 dir = EdgeEnd(input, line) - EdgeStart(input, line);
    sa = dot(a - p, dir);
    sb = dot(b - p, dir);
  }
  return (sa > 0 && sb > 0) || (sa < 0 && sb < 0);
}

// Relative size below which a plain floating-point comparison of two nearly
// equal lengths is not trusted and the decision is made from the contacts'
// touching circle instead. It only picks between two methods that agree
// whenever the difference is well above rounding error, so it never changes a
// result by more than that.
constexpr double kNearTie = 1e-9;

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
  // way round the ball the arc runs. Known exactly from how the ball was
  // found, where the computed points - antipodal for a ball stuck between
  // parallel edges - would only give noise.
  int turn;
};

// All balls of one kind between two edges, without testing that a tangent
// point lies on its edge or a center in its vertex's sector. Deterministic:
// every caller asking for the same (first, second, kind) gets the same bits.
struct RawCandidates {
  int count = 0;
  std::array<RawCandidate, 2> balls;
  void push_back(const RawCandidate& raw) { balls[count++] = raw; }
  size_t size() const { return count; }
  const RawCandidate& operator[](size_t i) const { return balls[i]; }
};

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
    // Parallel lines share no single ball.
    if (CrossSign(firstEnd, firstStart, secondEnd, secondStart) == 0)
      return raws;
    // The two tangent points are r |n1 - n2| apart, so edges facing the same
    // way that lie further apart than that along their direction share no
    // ball either. This spares the exact limits for pieces of one straight
    // line, which they could not tell apart in floating point; the margin
    // covers the rounding of this test itself, so it never rejects a ball.
    if (dot(firstNormal, secondNormal) > 0) {
      const double len = length(firstVec);
      const double a = dot(secondStart - firstStart, firstVec) / len;
      const double b = dot(secondEnd - firstStart, firstVec) / len;
      const double gap = std::max(std::min(a, b) - len, -std::max(a, b));
      const double scale = radius + std::abs(firstStart.x) +
                           std::abs(firstStart.y) + std::abs(secondStart.x) +
                           std::abs(secondStart.y) + len + std::abs(a) +
                           std::abs(b);
      if (gap > radius * length(firstNormal - secondNormal) + 1e-12 * scale)
        return raws;
    }
    vec2 center;
    double firstT, secondT;
    // Edges meeting at a point: their lines cross exactly there, and the
    // center lies along the bisector of the normals. Unlike intersecting the
    // offset lines, this stays accurate for nearly collinear edges.
    const bool firstThenSecond =
        EndPoint(input, first) == StartPoint(input, second);
    const bool secondThenFirst =
        EndPoint(input, second) == StartPoint(input, first);
    bool found = false;
    if (firstThenSecond != secondThenFirst) {
      const double den = 1 + dot(firstNormal, secondNormal);
      if (den > 0.5) {
        center = (firstThenSecond ? firstEnd : secondEnd) +
                 (firstNormal + secondNormal) * (radius / den);
        found = IsFinite(center);
      }
    }
    if (!found)
      found = IntersectLines(firstStart + firstNormal * radius, firstVec,
                             secondStart + secondNormal * radius, secondVec,
                             center, firstT, secondT);
    const double firstLenSq = dot(firstVec, firstVec);
    if (!found) {
      // Too close to parallel to intersect in floating point. Where along
      // the lines the ball sits is then lost to rounding; if it is on both
      // edges (decided exactly by Passes) it is where they run side by side.
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
    raws.push_back({center, firstT, secondT, false,
                    CrossSign(firstEnd, firstStart, secondEnd, secondStart)});
  } else if (kind == kLineVertex) {
    if (!TouchableEnd(input, second) || endOnEdge(second, first)) return raws;
    const IntersectResult ev = IntersectSegmentArc(
        firstStart, firstVec, firstNormal, secondEnd, radius,
        CountSegmentArc(firstStart, firstVec, input.invert, secondEnd, radius));
    for (int i = 0; i < ev.count; ++i)
      // The ball behind the vertex along the line has it on its left
      // (right with invert).
      raws.push_back({ev.points[i], ev.params[i], 1, ev.tangent,
                      (i == 0) != input.invert ? 1 : -1});
  } else if (kind == kVertexLine) {
    if (!TouchableEnd(input, first) || endOnEdge(first, second)) return raws;
    const IntersectResult ve = IntersectSegmentArc(
        secondStart, secondVec, secondNormal, firstEnd, radius,
        CountSegmentArc(secondStart, secondVec, input.invert, firstEnd,
                        radius));
    for (int i = 0; i < ve.count; ++i)
      raws.push_back({ve.points[i], 1, ve.params[i], ve.tangent,
                      (i == 0) != input.invert ? -1 : 1});
  } else {
    if (!TouchableEnd(input, first) || !TouchableEnd(input, second))
      return raws;
    const IntersectResult vv = IntersectArcArc(
        firstEnd, secondEnd, radius, CountArcArc(firstEnd, secondEnd, radius));
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
// replaces a side by the edge's own exact decision where the two could
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
  // A sector wider than a half-turn has no single side test per normal.
  test.narrow = (ccw ? la::cross(normalIn, normalOut)
                     : la::cross(normalOut, normalIn)) >= 0;
  if (!test.narrow) {
    test.inside = IsVectorInSector(v, normalIn, normalOut, ccw);
    return test;
  }
  test.in = (ccw ? la::cross(normalIn, v) : la::cross(v, normalIn)) > 0;
  test.out = (ccw ? la::cross(v, normalOut) : la::cross(normalOut, v)) > 0;
  test.inside = test.in && test.out;
  return test;
}

// Answers of a decision that may be asked for its floating-point filter
// alone: kUnknown when that cannot tell.
enum Tri : int8_t { kNo = 0, kYes = 1, kUnknown = -1 };

// The ball X tangent to the lines of edges e1 and e2, for exact signs of
// expressions in its center. With d, d' the edges' directions, a = |d|,
// b = |d'|, D = d . d' and W the point where the lines cross,
// X = W + r sigma (b perp(d) + a perp(d')) / (ab + D). W = WN / delta is
// rational in the coordinates (and the shared vertex itself for edges that
// meet), so expressions are scaled by delta (ab + D), a number of the form
// p + q a + s b + u ab.
struct PairBall {
  const FilletInput& input;
  double radius;
  double sigma;
  vec2 p0, p1, q0, q1;
  bool meet;
  vec2 meetPoint;
  // Floating-point filter: X = base + offset, base known to within `slack`.
  bool conditioned;
  vec2 fd, fd2, offset;
  double along = 0;
  double slack = 0;
  // Exact parts, built on demand.
  bool built = false;
  exact::Vec d, d2;
  exact::Expansion A, B, D, delta, rDelta;
  int signDelta = 1;

  PairBall(const FilletInput& in, int e1, int e2, double r)
      : input(in),
        radius(r),
        sigma(in.invert ? -1 : 1),
        p0(EdgeStart(in, e1)),
        p1(EdgeEnd(in, e1)),
        q0(EdgeStart(in, e2)),
        q1(EdgeEnd(in, e2)) {
    const bool meetAtEnd1 = EndPoint(in, e1) == StartPoint(in, e2);
    const bool meetAtEnd2 = EndPoint(in, e2) == StartPoint(in, e1);
    meet = meetAtEnd1 != meetAtEnd2;
    meetPoint = meetAtEnd1 ? p1 : q1;
    const vec2 n = LineNormal(in, e1), n2 = LineNormal(in, e2);
    const double den = 1 + dot(n, n2);
    fd = p1 - p0;
    fd2 = q1 - q0;
    // The crossing point of lines that do not meet at a vertex is only as
    // accurate as the lines are far from parallel, and the offset from it
    // only as the normals are far from opposite: the filter's error
    // allowance grows accordingly (see FloatRel).
    const double cross = la::cross(fd, fd2);
    const double sine = std::abs(cross) / (length(fd) * length(fd2));
    conditioned = den > 1e-12 && (meet || sine > 1e-12);
    if (conditioned) {
      offset = (n + n2) * (radius / den);
      slack = 1e-3 * length(offset) / den;
      if (!meet) {
        along = la::cross(q0 - p0, fd2) / cross;
        slack += 1e-3 / sine *
                 (std::abs(p0.x) + std::abs(p0.y) + std::abs(q0.x) +
                  std::abs(q0.y) + length(fd * along));
      }
    }
  }

  // W - g. With the offset to X it is uncertain by `uncertainty` times the
  // filter's relative threshold - hundreds of times the rounding error.
  vec2 FloatRel(vec2 g, double& uncertainty) const {
    uncertainty = slack;
    if (meet) return meetPoint - g;
    uncertainty += 1e-3 * (std::abs(g.x) + std::abs(g.y));
    return (p0 - g) + fd * along;
  }

  // False if the lines are parallel and there is no such ball.
  bool Build() {
    using namespace exact;
    if (!built) {
      built = true;
      d = Sub(p1, p0);
      d2 = Sub(q1, q0);
      A = Dot(d, d);
      B = Dot(d2, d2);
      D = Dot(d, d2);
      delta = meet ? Expansion{1} : Cross(d, d2);
      signDelta = Sign(delta);
      rDelta = Scale(delta, radius);
    }
    return signDelta != 0;
  }

  // delta (W - g).
  exact::Vec Rel(vec2 g) const {
    using namespace exact;
    if (meet) return Sub(meetPoint, g);
    const Expansion alongN = Cross(Sub(q0, p0), d2);
    const Vec base = Sub(p0, g);
    return {Add(Mul(delta, base.x), Mul(d.x, alongN)),
            Add(Mul(delta, base.y), Mul(d.y, alongN))};
  }

  // Sign of dist(X, P) - r for a point P.
  Tri CutByPoint(vec2 p, bool filterOnly) {
    using namespace exact;
    if (conditioned) {
      double slack;
      const vec2 w = FloatRel(p, slack);
      const vec2 x = w + offset;
      const double value = dot(x, x) - radius * radius;
      const double magnitude = dot(w, w) + dot(offset, offset) +
                               radius * radius + slack * (length(x) + slack);
      if (std::abs(value) > 1e-10 * magnitude) return value < 0 ? kYes : kNo;
    }
    if (filterOnly) return kUnknown;
    if (!Build()) return kNo;
    // delta^2 (ab + D) (|X - P|^2 - r^2) = (|w|^2 - r^2 delta^2) D
    //   + 2 r sigma delta C' a + 2 r sigma delta C b
    //   + (|w|^2 + r^2 delta^2) ab, C = cross(d, w), C' = cross(d', w).
    const Vec w = Rel(p);
    const Expansion ww = Dot(w, w);
    const Expansion rr = Mul(rDelta, rDelta);
    const Rad2 value{Mul(Sub(ww, rr), D),
                     Mul(Scale(Cross(d2, w), 2 * sigma), rDelta),
                     Mul(Scale(Cross(d, w), 2 * sigma), rDelta), Add(ww, rr)};
    return Sign(value, A, B) < 0 ? kYes : kNo;
  }

  // Sign of (signed distance of X to the line of edge m, on its ball side)
  // - k r, for k = 1, 0 or -1; 2 if the filter alone cannot tell.
  int LineSide(int m, int k, bool filterOnly) {
    using namespace exact;
    const vec2 g0 = EdgeStart(input, m), g1 = EdgeEnd(input, m);
    if (conditioned) {
      double slack;
      const vec2 w = FloatRel(g0, slack);
      const vec2 ng = LineNormal(input, m);
      const double value = dot(ng, w) + dot(ng, offset) - k * radius;
      const double magnitude = std::abs(ng.x * w.x) + std::abs(ng.y * w.y) +
                               std::abs(ng.x * offset.x) +
                               std::abs(ng.y * offset.y) + radius + slack;
      if (std::abs(value) > 1e-10 * magnitude) return value < 0 ? -1 : 1;
    }
    if (filterOnly) return 2;
    if (!Build()) return 1;
    // delta c (ab + D) (dist - k r) = T - k c R, c = |dM|,
    // R = r delta (ab + D), T = sigma A' (ab + D) + r delta (b K + a K'),
    // A' = cross(dM, w), K = dM . d, K' = dM . d'.
    const Vec dM = Sub(g1, g0);
    const Expansion a2 = Scale(Cross(dM, Rel(g0)), sigma);
    const Rad2 t{Mul(a2, D), Mul(Dot(dM, d2), rDelta), Mul(Dot(dM, d), rDelta),
                 a2};
    // Divided by delta, R is positive: T decides unless it has the sign of
    // k, and then their squares do.
    const int signT = Sign(t, A, B) * signDelta;
    if (k == 0 || signT != k) return signT != 0 ? signT : -k;
    const Rad2 r{Mul(D, rDelta), {}, {}, rDelta};
    const Expansion Q = Dot(dM, dM);
    Rad2 difference = Mul(t, t, A, B);
    const Rad2 rr = Mul(r, r, A, B);
    for (int i = 0; i < 4; ++i)
      difference[i] = Sub(difference[i], Mul(Q, rr[i]));
    return k * Sign(difference, A, B);
  }

  // Sign of (X - P) . (u1 - u0): which way along a direction X lies from P.
  int Along(vec2 p, vec2 u0, vec2 u1) {
    using namespace exact;
    if (conditioned) {
      double slack;
      const vec2 w = FloatRel(p, slack);
      const vec2 u = u1 - u0;
      const double value = dot(w, u) + dot(offset, u);
      const double magnitude = std::abs(w.x * u.x) + std::abs(w.y * u.y) +
                               std::abs(offset.x * u.x) +
                               std::abs(offset.y * u.y) + slack * length(u);
      if (std::abs(value) > 1e-10 * magnitude) return value < 0 ? -1 : 1;
    }
    if (!Build()) return 0;
    // delta (ab + D) (X - P) . u = (w . u)(ab + D)
    //   + r sigma delta (b cross(d, u) + a cross(d', u)).
    const Vec u = Sub(u1, u0);
    const Expansion wu = Dot(Rel(p), u);
    const Rad2 value{Mul(wu, D), Mul(Scale(Cross(d2, u), sigma), rDelta),
                     Mul(Scale(Cross(d, u), sigma), rDelta), wu};
    return Sign(value, A, B) * signDelta;
  }
};

// Whether feature f cuts the ball tangent to the lines of edges e1 and e2,
// exactly: a ball only just touching f still fits.
Tri PairCut(const FilletInput& input, int f, int e1, int e2, double radius,
            bool filterOnly = false) {
  PairBall ball(input, e1, e2, radius);
  const int m = FeatureEdge(f);
  if (IsVertex(f)) return ball.CutByPoint(EdgeStart(input, m), filterOnly);
  const int side = ball.LineSide(m, 1, filterOnly);
  return side == 2 ? kUnknown : side < 0 ? kYes : kNo;
}

// At a vertex V where the boundary turns towards the ball, between edges
// `before` and the next, the ball X tangent to both is where the ball rolling
// along either edge stops. Whether feature f cuts X: one sign per (f, V)
// decides whether X is blocked by f and, for the balls touching f and either
// edge, whether they stop before X - so those decisions cannot disagree.
Tri CornerCut(const FilletInput& input, int f, int before, double radius,
              bool filterOnly = false) {
  return PairCut(input, f, before, NextEdge(input, before), radius, filterOnly);
}

// Sign of (P - X) . d(line) for vertex P and the corner ball X of CornerCut:
// which way along `line` (one of the corner's two edges) P's foot lies from X.
int CornerFoot(const FilletInput& input, vec2 p, int before, bool alongBefore,
               double radius, bool filterOnly, bool& known) {
  using namespace exact;
  const int after = NextEdge(input, before);
  const vec2 v0 = EdgeStart(input, before), v = EdgeEnd(input, before),
             v1 = EdgeEnd(input, after);
  const double sigma = input.invert ? -1 : 1;
  {
    // Floating-point filter.
    const vec2 n = LineNormal(input, before), n2 = LineNormal(input, after);
    const double den = 1 + dot(n, n2);
    if (den > 1e-3) {
      const vec2 offset = (n + n2) * (radius / den);
      const vec2 w = p - v;
      const vec2 along = alongBefore ? v - v0 : v1 - v;
      const double value = dot(w, along) - dot(offset, along);
      const double magnitude =
          std::abs(w.x * along.x) + std::abs(w.y * along.y) +
          std::abs(offset.x * along.x) + std::abs(offset.y * along.y);
      if (std::abs(value) > 1e-10 * magnitude) return value > 0 ? 1 : -1;
    }
  }
  if (filterOnly) {
    known = false;
    return 0;
  }
  const Vec d = Sub(v, v0), d2 = Sub(v1, v);
  const Expansion A = Dot(d, d), B = Dot(d2, d2), D = Dot(d, d2);
  // (ab + D) (P - X) . d = -(w . d)(ab + D) + r sigma a cross(d, d'), and
  // (ab + D) (P - X) . d' = -(w . d')(ab + D) - r sigma b cross(d, d').
  const Expansion wd = Neg(Dot(Sub(v, p), alongBefore ? d : d2));
  const Expansion turn =
      Scale(Cross(d, d2), (alongBefore ? 1 : -1) * radius * sigma);
  const Rad2 value{Mul(wd, D), alongBefore ? turn : Expansion{},
                   alongBefore ? Expansion{} : turn, wd};
  return Sign(value, A, B);
}

// The limit of edge `line` at an end in a corner (kCorner): whether the ball
// touching g and the line (root `root` of `roots`) stops before the corner
// ball there. Past it, the corner's other edge cuts the ball.
Tri CornerLimit(const FilletInput& input, int g, int line, bool atEnd, int root,
                int roots, double radius, bool filterOnly) {
  const int before = atEnd ? line : input.prevEdges[line];
  const int other = atEnd ? NextEdge(input, line) : before;
  // The corner ball itself.
  if (g == LineFeature(other)) return kYes;
  const Tri cutTri = CornerCut(input, g, before, radius, filterOnly);
  if (cutTri == kUnknown) return kUnknown;
  const bool cut = cutTri == kYes;
  auto tri = [](bool b) { return b ? kYes : kNo; };
  if (!IsVertex(g)) {
    // dist(., M) - r grows along the line where it leans away from M.
    const int m = FeatureEdge(g);
    const int slope = (input.invert ? -1 : 1) *
                      CrossSign(EdgeEnd(input, m), EdgeStart(input, m),
                                EdgeEnd(input, line), EdgeStart(input, line));
    return tri(atEnd ? (cut ? slope < 0 : slope > 0)
                     : (cut ? slope > 0 : slope < 0));
  }
  // X inside P's circle separates the two roots; else both lie on the side
  // of X that P's foot does.
  if (cut) return tri(roots == 2 ? root == (atEnd ? 0 : 1) : atEnd);
  bool known = true;
  const int foot = CornerFoot(input, EdgeStart(input, FeatureEdge(g)), before,
                              atEnd, radius, filterOnly, known);
  if (!known) return kUnknown;
  return tri(atEnd ? foot <= 0 : foot >= 0);
}

// The edge-or-vertex decision at one end V of edge `line`, for the ball
// touching contact g and the line on root `root` of `roots`: whether its
// tangent point is within the edge at that end (t <= 1 at the end, t >= 0 at
// the start). S = V + n r is the center of the ball tangent to the line at V;
// where it lies relative to g decides, exactly, so the edge case and the
// vertex case at V (which reads its negation) never disagree, and nearly
// parallel lines - whose computed tangent parameters are noise - are decided
// as in exact geometry. With n = sigma perp(d) / |d|:
//  - g a line M: S on the edge's side of where M's offset crosses the line's,
//    sign(dist(S, M) - r) against the direction into the edge;
//  - g a vertex P: the roots are where P's circle crosses the line's offset,
//    in order along it; S strictly inside the circle separates them, else
//    both lie on the side of S that P's foot does.
Tri SegmentLimit(const FilletInput& input, int g, int line, bool atEnd,
                 int root, int roots, double radius, bool strict = false,
                 bool filterOnly = false) {
  using namespace exact;
  auto tri = [](bool b) { return b ? kYes : kNo; };
  const vec2 start = EdgeStart(input, line);
  const vec2 end = EdgeEnd(input, line);
  const vec2 v = atEnd ? end : start;
  const double sigma = input.invert ? -1 : 1;
  const vec2 fL = end - start;
  const double fLen = length(fL);
  if (!IsVertex(g)) {
    const int m = FeatureEdge(g);
    const vec2 m0 = EdgeStart(input, m);
    const vec2 m1 = EdgeEnd(input, m);
    {
      // Floating-point filter for the two signs below.
      const vec2 fM = m1 - m0, w = v - m0;
      const double fLenM = length(fM);
      const double al = fM.x * w.y, ar = fM.y * w.x;
      const double kl = fM.x * fL.x, kr = fM.y * fL.y;
      const double f =
          fLen * (sigma * (al - ar) - radius * fLenM) + radius * (kl + kr);
      const double mag = fLen * (std::abs(al) + std::abs(ar) + radius * fLenM) +
                         radius * (std::abs(kl) + std::abs(kr));
      const double cl = fM.x * fL.y, cr = fM.y * fL.x;
      const double c = cl - cr;
      if (std::abs(f) > kFilter * mag) {
        // The cross product alone is cheap to get exactly.
        const int cs = std::abs(c) > kFilter * (std::abs(cl) + std::abs(cr))
                           ? (c > 0 ? 1 : -1)
                           : CrossSign(m1, m0, end, start);
        const int inward = (atEnd ? -1 : 1) * cs * static_cast<int>(sigma);
        const int sf = f > 0 ? 1 : -1;
        return tri(strict ? sf * inward < 0 : sf * inward <= 0);
      }
    }
    if (filterOnly) return kUnknown;
    const Vec dL = Sub(end, start);
    const Expansion p2 = Dot(dL, dL);
    const Vec dM = Sub(EdgeEnd(input, m), m0);
    const Expansion a = Scale(Cross(dM, Sub(v, m0)), sigma);
    const Expansion q2 = Dot(dM, dM);
    const Expansion k = Scale(Dot(dM, dL), radius);
    // F = sqrt(P) u + k, u = sigma A - r sqrt(Q): |dM||dL| (dist(S, M) - r).
    const int su = SignWithRoot(a, {-radius}, q2);
    const int sk = Sign(k);
    int sf;
    if (su == 0 || sk == 0 || su == sk) {
      sf = su != 0 ? su : sk;
    } else {
      // sign(u^2 P - k^2) = sign(X - Y sqrt(Q)).
      const Expansion x = Sub(
          Mul(p2, Add(Mul(a, a), Scale(Scale(q2, radius), radius))), Mul(k, k));
      const Expansion y = Mul(Scale(a, 2 * radius), p2);
      sf = su * SignWithRoot(x, Neg(y), q2);
    }
    const int inward =
        (atEnd ? -1 : 1) * Sign(Cross(dM, dL)) * static_cast<int>(sigma);
    return tri(strict ? sf * inward < 0 : sf * inward <= 0);
  }
  const vec2 p = EdgeStart(input, FeatureEdge(g));
  // |dL| (|S - P|^2 - r^2) = |w|^2 sqrt(P) + 2 r sigma cross(dL, w).
  bool inside;
  {
    const vec2 w = v - p;
    const double cl = fL.x * w.y, cr = fL.y * w.x;
    const double ww = dot(w, w) * fLen;
    const double f = ww + 2 * radius * sigma * (cl - cr);
    const double mag = ww + 2 * radius * (std::abs(cl) + std::abs(cr));
    if (std::abs(f) > kFilter * mag) {
      inside = f < 0;
    } else if (filterOnly) {
      return kUnknown;
    } else {
      const Vec dL = Sub(end, start);
      const Vec ew = Sub(v, p);
      inside = SignWithRoot(Scale(Cross(dL, ew), 2 * radius * sigma),
                            Dot(ew, ew), Dot(dL, dL)) < 0;
    }
  }
  if (inside && roots == 2) return tri(root == (atEnd ? 0 : 1));
  const int foot = DotSign(p, v, end, start);
  return tri(atEnd ? foot <= 0 : foot >= 0);
}

// The limit of edge `line` at one end for the ball touching contact g and the
// line: the corner ball in a corner, and the end of the edge itself
// otherwise. Other edges, including a short corner's, are left to Blocks.
Tri LineLimit(const FilletInput& input, int g, int line, bool atEnd, int root,
              int roots, double radius, bool filterOnly = false) {
  const int kind = input.endKinds[atEnd ? line : input.prevEdges[line]];
  if (kind == kCorner)
    return CornerLimit(input, g, line, atEnd, root, roots, radius, filterOnly);
  // Where the boundary runs straight on, a ball tangent exactly at the
  // vertex belongs to the edge ending there.
  return SegmentLimit(input, g, line, atEnd, root, roots, radius,
                      kind == kStraight && !atEnd, filterOnly);
}

// How the boundary turns at the end of each edge.
std::vector<int8_t> BuildEndKinds(const FilletInput& input, double radius) {
  std::vector<int8_t> kinds(input.edges.size());
  for (size_t i = 0; i < kinds.size(); ++i) {
    const int e = static_cast<int>(i);
    const int next = NextEdge(input, e);
    const vec2 a = EdgeStart(input, e), b = EdgeEnd(input, e);
    const int turn =
        (input.invert ? -1 : 1) * CrossSign(b, a, EdgeEnd(input, next), b);
    if (turn <= 0) {
      kinds[i] = turn < 0 ? kTouchable : kStraight;
      continue;
    }
    const bool real =
        SegmentLimit(input, LineFeature(next), e, false, 0, 1, radius) ==
            kYes &&
        SegmentLimit(input, LineFeature(e), next, true, 0, 1, radius) == kYes;
    kinds[i] = real ? kCorner : kShortCorner;
  }
  return kinds;
}

// The same decision for the root that can reach V, as the vertex case at V
// needs it.
bool EdgeCaseAt(const FilletInput& input, int g, int line, bool atEnd,
                double radius) {
  if (!IsVertex(g))
    return LineLimit(input, g, line, atEnd, 0, 1, radius) == kYes;
  const vec2 v = atEnd ? EdgeEnd(input, line) : EdgeStart(input, line);
  const int ahead = DotSign(v, EdgeStart(input, FeatureEdge(g)),
                            EdgeEnd(input, line), EdgeStart(input, line));
  return LineLimit(input, g, line, atEnd, ahead > 0 ? 1 : 0, 2, radius) == kYes;
}

// Which of the two balls touching g and vertex v meets the balls touching g
// and `line` at v (where its center is v + n r), as the side of it the ball's
// center is on: of the line from g's vertex through v (left positive), or
// along g's line from v. Fixed by input coordinates alone.
int MeetingSide(const FilletInput& input, int g, vec2 v, int line) {
  const vec2 l0 = EdgeStart(input, line), l1 = EdgeEnd(input, line);
  const int sigma = input.invert ? -1 : 1;
  const int m = FeatureEdge(g);
  if (IsVertex(g)) return sigma * DotSign(v, EdgeStart(input, m), l1, l0);
  return sigma * CrossSign(l1, l0, EdgeEnd(input, m), EdgeStart(input, m));
}

// Whether the ball at `center` touching g touches the vertex at the end of
// `edge` strictly within its sector. `side` is the side of the vertex the
// center is on, as defined for MeetingSide (0 for a double ball): on that
// side of the sector the neighbouring edge's own decision is read, so exactly
// one of the edge case and the vertex case claims the ball.
bool InVertexSector(const FilletInput& input, int edge, int g, vec2 center,
                    double radius, int side) {
  const SectorTest test = TestSector(input, edge, center);
  if (!test.narrow) return test.inside;
  const vec2 v = EdgeEnd(input, edge);
  const int next = NextEdge(input, edge);
  bool in = test.in, out = test.out;
  if (side != 0 && MeetingSide(input, g, v, edge) == side)
    in = !EdgeCaseAt(input, g, edge, true, radius);
  if (side != 0 && MeetingSide(input, g, v, next) == side)
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

  // The limits of each edge touched on its line, at both ends. Most balls
  // fail one clearly, so first ask only for what floating point can tell,
  // and evaluate exactly only if none of them rules the ball out.
  struct Limit {
    int g, line;
    bool atEnd;
  };
  std::array<Limit, 4> limits;
  int numLimits = 0;
  if (!FirstIsVertex(kind)) {
    limits[numLimits++] = {secondFeature, first, false};
    limits[numLimits++] = {secondFeature, first, true};
  }
  if (!SecondIsVertex(kind)) {
    limits[numLimits++] = {firstFeature, second, false};
    limits[numLimits++] = {firstFeature, second, true};
  }
  std::array<Tri, 4> known;
  for (int i = 0; i < numLimits; ++i) {
    known[i] = LineLimit(input, limits[i].g, limits[i].line, limits[i].atEnd,
                         root, roots, radius, true);
    if (known[i] == kNo) return false;
  }
  for (int i = 0; i < numLimits; ++i) {
    if (known[i] == kUnknown &&
        LineLimit(input, limits[i].g, limits[i].line, limits[i].atEnd, root,
                  roots, radius) == kNo)
      return false;
  }
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
  auto clamp = [](double t) { return !(t > 0) ? 0.0 : t > 1 ? 1.0 : t; };
  auto offsetPoint = [&](int e, double t) {
    return PointOnEdge(input, e, t) + LineNormal(input, e) * radius;
  };
  auto project = [&](int e, vec2 c) {
    const vec2 start = EdgeStart(input, e);
    const vec2 d = EdgeEnd(input, e) - start;
    return dot(c - start, d) / dot(d, d);
  };
  if (!FirstIsVertex(kind)) {
    const double t = clamp(raw.firstT);
    if (t != raw.firstT) {
      raw.firstT = t;
      raw.center = offsetPoint(first, t);
      if (!SecondIsVertex(kind)) raw.secondT = project(second, raw.center);
    }
  }
  if (!SecondIsVertex(kind)) {
    const double t = clamp(raw.secondT);
    if (t != raw.secondT) {
      raw.secondT = t;
      raw.center = offsetPoint(second, t);
      if (!FirstIsVertex(kind)) raw.firstT = clamp(project(first, raw.center));
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

// The circle of the triple {g, h, f} relevant to the ball of fillet (g, h):
// among touching circles on the fillet's branch, the one nearest in radius.
bool TripleCircle(const FilletInput& input, int g, int h, int f,
                  vec2 ballCenter, double radius, vec2& center, double& rho) {
  if (f == g || f == h || g == h) return false;
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

// Whether contact f cuts the ball of `fillet`, where `distance` is the plain
// distance from the ball's center to f. When distance and radius are nearly
// equal, the three contacts nearly share one touching circle, and the fillets
// around it must agree: all fit or none do. Then decide by that circle's
// radius rho and the slope there, which depend only on the set of contacts.
bool ContactBlocks(const FilletInput& input, const FilletGeometry& fillet,
                   int f, double distance, double radius) {
  const int g = fillet.from.feature;
  const int h = fillet.to.feature;
  const double scale =
      radius + std::max(std::abs(fillet.center.x), std::abs(fillet.center.y));
  if (std::abs(distance - radius) > kNearTie * scale) return distance < radius;
  vec2 x, v;
  double rho;
  if (!TripleCircle(input, g, h, f, fillet.center, radius, x, rho) ||
      !CenterVelocity(input, g, h, x, rho, v))
    return distance < radius;
  const vec2 toward = IsVertex(f) ? (x - EdgeStart(input, FeatureEdge(f))) / rho
                                  : LineNormal(input, FeatureEdge(f));
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
  const int g = fillet.from.feature, h = fillet.to.feature;
  if (!IsVertex(g) && !IsVertex(h)) {
    // A ball tangent to two lines is decided exactly, from the lines alone:
    // its computed center may be far off where they are nearly parallel.
    PairBall ball(input, FeatureEdge(g), FeatureEdge(h), radius);
    const int fromStart = ball.Along(p0, p0, p1);
    if (fromStart > 0 && ball.Along(p1, p0, p1) < 0) {
      // Nearest inside the edge. A ball behind it cannot touch it from the
      // ball side, but must not reach through it either.
      if (ball.LineSide(c, 0, false) > 0) return ball.LineSide(c, 1, false) < 0;
      return ball.LineSide(c, -1, false) > 0;
    }
    const bool atStart = fromStart <= 0;
    // A vertex on a line the ball is tangent to is never closer than radius.
    const int point = atStart ? StartPoint(input, c) : EndPoint(input, c);
    for (const int feature : {g, h}) {
      if (point == StartPoint(input, FeatureEdge(feature)) ||
          point == EndPoint(input, FeatureEdge(feature)))
        return false;
    }
    return ball.CutByPoint(atStart ? p0 : p1, false) == kYes;
  }
  const vec2 d = p1 - p0;
  const double t = dot(fillet.center - p0, d) / dot(d, d);

  if (t > 0 && t < 1) {
    const double s = dot(EdgeNormal(d, input.invert), fillet.center - p0);
    // A ball behind the edge cannot touch it from the ball side.
    if (!(s > 0)) return -s < radius;
    return ContactBlocks(input, fillet, LineFeature(c), s, radius);
  }
  const int vertex =
      t <= 0 ? VertexFeature(c) : VertexFeature(input.nextEdges[c]);
  // A vertex on a line the ball is tangent to is never closer than radius.
  const int point = t <= 0 ? StartPoint(input, c) : EndPoint(input, c);
  for (const int feature : {fillet.from.feature, fillet.to.feature}) {
    if (!IsVertex(feature) &&
        (point == StartPoint(input, FeatureEdge(feature)) ||
         point == EndPoint(input, FeatureEdge(feature))))
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
    const int vertex = StartPoint(input, FeatureEdge(end.feature));
    if (StartPoint(input, c) == vertex || EndPoint(input, c) == vertex)
      return true;
  }
  return false;
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
  for (const auto& [first, second] : broadPairs) {
    for (const int kind :
         {kLineLine, kLineVertex, kVertexLine, kVertexVertex}) {
      const RawCandidates raws =
          FindRawCandidates(input, first, second, kind, radius);
      for (size_t i = 0; i < raws.size(); ++i) {
        if (!Passes(input, first, second, kind, raws, static_cast<int>(i),
                    radius))
          continue;
        RawCandidate raw = raws[i];
        Conform(input, first, second, kind, raw, radius);
        AppendFillets(input, first, second, kind, raw, candidates);
      }
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
  // An arc that is a whole number of steps up to rounding (a right angle at
  // 20 segments, say) must not get an extra one, so the count is taken just
  // short of the next whole number; the validity check on the result allows
  // steps far more oversized than this.
  const int segments = std::max(
      1, static_cast<int>(
             std::ceil(totalRad * circularSegments / (2.0 * M_PI) - 1e-9)));
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
      const BoundaryPos departure = Position(input, fillet.from);
      const BoundaryPos arrival = Position(input, fillet.to);
      AppendPoint(loop, EndPoint(input, fillet.from), pointTolerance);
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
          AppendPoint(loop, EdgeEnd(input, edge), pointTolerance);
          touchedLoop[input.edges[edge].loop] = 1;
          edge = NextEdge(input, edge);
          first = false;
        }
      }
      AppendPoint(loop, EndPoint(input, nextFillet.from), pointTolerance);
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
                    {}};
  input.starts.resize(edges.size());
  input.ends.resize(edges.size());
  input.normals.resize(edges.size());
  for (size_t e = 0; e < edges.size(); ++e) {
    const SimplePolygon& loop = polygons[edges[e].loop];
    input.starts[e] = loop[edges[e].edge];
    input.ends[e] = loop[(edges[e].edge + 1) % loop.size()];
    input.normals[e] = EdgeNormal(input.ends[e] - input.starts[e], invert);
  }
  input.endKinds = BuildEndKinds(input, radius);

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

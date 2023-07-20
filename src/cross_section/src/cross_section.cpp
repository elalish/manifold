// Copyright 2023 The Manifold Authors.
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

#include "cross_section.h"

using namespace manifold;

namespace {
const int precision_ = 8;

C2::ClipType cliptype_of_op(OpType op) {
  C2::ClipType ct = C2::ClipType::Union;
  switch (op) {
    case OpType::Add:
      break;
    case OpType::Subtract:
      ct = C2::ClipType::Difference;
      break;
    case OpType::Intersect:
      ct = C2::ClipType::Intersection;
      break;
  };
  return ct;
}

C2::FillRule fr(CrossSection::FillRule fillrule) {
  C2::FillRule fr = C2::FillRule::EvenOdd;
  switch (fillrule) {
    case CrossSection::FillRule::EvenOdd:
      break;
    case CrossSection::FillRule::NonZero:
      fr = C2::FillRule::NonZero;
      break;
    case CrossSection::FillRule::Positive:
      fr = C2::FillRule::Positive;
      break;
    case CrossSection::FillRule::Negative:
      fr = C2::FillRule::Negative;
      break;
  };
  return fr;
}

C2::JoinType jt(CrossSection::JoinType jointype) {
  C2::JoinType jt = C2::JoinType::Square;
  switch (jointype) {
    case CrossSection::JoinType::Square:
      break;
    case CrossSection::JoinType::Round:
      jt = C2::JoinType::Round;
      break;
    case CrossSection::JoinType::Miter:
      jt = C2::JoinType::Miter;
      break;
  };
  return jt;
}

glm::vec2 v2_of_pd(const C2::PointD p) { return {p.x, p.y}; }

C2::PointD v2_to_pd(const glm::vec2 v) { return C2::PointD(v.x, v.y); }

C2::PathD pathd_of_contour(const SimplePolygon& ctr) {
  auto p = C2::PathD();
  p.reserve(ctr.size());
  for (auto v : ctr) {
    p.push_back(v2_to_pd(v));
  }
  return p;
}

C2::PathsD transform(const C2::PathsD ps, const glm::mat3x2 m) {
  const bool invert = glm::determinant(glm::mat2(m)) < 0;
  auto transformed = C2::PathsD();
  transformed.reserve(ps.size());
  for (auto path : ps) {
    auto sz = path.size();
    auto s = C2::PathD(sz);
    for (int i = 0; i < sz; ++i) {
      auto idx = invert ? sz - 1 - i : i;
      s[idx] = v2_to_pd(m * glm::vec3(path[i].x, path[i].y, 1));
    }
    transformed.push_back(s);
  }
  return transformed;
}

std::shared_ptr<const C2::PathsD> shared_paths(const C2::PathsD& ps) {
  return std::make_shared<const C2::PathsD>(ps);
}

// forward declaration for mutual recursion
void decompose_hole(const C2::PolyTreeD* outline,
                    std::vector<C2::PathsD>& polys, C2::PathsD& poly,
                    int n_holes, int j);

void decompose_outline(const C2::PolyTreeD* tree,
                       std::vector<C2::PathsD>& polys, int i) {
  auto n_outlines = tree->Count();
  if (i < n_outlines) {
    auto outline = tree->Child(i);
    auto n_holes = outline->Count();
    auto poly = C2::PathsD(n_holes + 1);
    poly[0] = outline->Polygon();
    decompose_hole(outline, polys, poly, n_holes, 0);
    polys.push_back(poly);
    if (i < n_outlines - 1) {
      decompose_outline(tree, polys, i + 1);
    }
  }
}

void decompose_hole(const C2::PolyTreeD* outline,
                    std::vector<C2::PathsD>& polys, C2::PathsD& poly,
                    int n_holes, int j) {
  if (j < n_holes) {
    auto child = outline->Child(j);
    decompose_outline(child, polys, 0);
    poly[j + 1] = child->Polygon();
    decompose_hole(outline, polys, poly, n_holes, j + 1);
  }
}

bool V2Lesser(glm::vec2 a, glm::vec2 b) {
  if (a.x == b.x) return a.y < b.y;
  return a.x < b.x;
}

double Cross(glm::vec2 a, glm::vec2 b) { return a.x * b.y - a.y * b.x; }

double SquaredDistance(glm::vec2 a, glm::vec2 b) {
  auto d = a - b;
  return glm::dot(d, d);
}

bool IsCw(glm::vec2 a, glm::vec2 b, glm::vec2 c) {
  double lhs = Cross(a - c, b - c);
  double rhs = 1e-18 * SquaredDistance(a, c) * SquaredDistance(b, c);
  return lhs * std::abs(lhs) <= rhs;
}

void HullBacktrack(const SimplePolygon& pts, const int idx,
                   std::vector<int>& keep, const int hold) {
  const int stop = keep.size() - hold;
  int i = 0;
  while (i < stop && !IsCw(pts[idx], pts[keep[keep.size() - 1]],
                           pts[keep[keep.size() - 2]])) {
    keep.pop_back();
    i++;
  }
}

// Based on method described here:
// https://www.hackerearth.com/practice/math/geometry/line-sweep-technique/tutorial/
C2::PathD HullImpl(SimplePolygon& pts) {
  int len = pts.size();
  if (len < 3) return C2::PathD();  // not enough points to create a polygon
  std::sort(pts.begin(), pts.end(), V2Lesser);
  auto keep = std::vector<int>{0, 1};
  for (int i = 2; i < len; i++) {
    HullBacktrack(pts, i, keep, 1);
    keep.push_back(i);
  }
  int nLower = keep.size();
  for (int i = 0; i < len - 1; i++) {
    int idx = len - 2 - i;
    HullBacktrack(pts, idx, keep, nLower);
    if (idx > 0) keep.push_back(idx);
  }
  auto path = C2::PathD(keep.size());
  for (int i = 0; i < keep.size(); i++) {
    path[i] = v2_to_pd(pts[keep[i]]);
  }
  return path;
}
}  // namespace

namespace manifold {

/**
 * The default constructor is an empty cross-section (containing no contours).
 */
CrossSection::CrossSection() { paths_ = shared_paths(C2::PathsD()); }

CrossSection::~CrossSection() = default;
CrossSection::CrossSection(CrossSection&&) noexcept = default;
CrossSection& CrossSection::operator=(CrossSection&&) noexcept = default;

/**
 * The copy constructor avoids copying the underlying paths vector (sharing
 * with its parent via shared_ptr), however subsequent transformations, and
 * their application will not be shared. It is generally recommended to avoid
 * this, opting instead to simply create CrossSections with the available
 * const methods.
 */
CrossSection::CrossSection(const CrossSection& other) {
  paths_ = other.paths_;
  transform_ = other.transform_;
}

CrossSection& CrossSection::operator=(const CrossSection& other) {
  if (this != &other) {
    paths_ = other.paths_;
    transform_ = other.transform_;
  }
  return *this;
};

// Private, skips unioning.
CrossSection::CrossSection(C2::PathsD ps) { paths_ = shared_paths(ps); }

/**
 * Create a 2d cross-section from a single contour. A boolean union operation
 * (with Positive filling rule by default) is performed to ensure the
 * resulting CrossSection is free of self-intersections.
 *
 * @param contour A closed path outlining the desired cross-section.
 * @param fillrule The filling rule used to interpret polygon sub-regions
 * created by self-intersections in contour.
 */
CrossSection::CrossSection(const SimplePolygon& contour, FillRule fillrule) {
  auto ps = C2::PathsD{(pathd_of_contour(contour))};
  paths_ = shared_paths(C2::Union(ps, fr(fillrule), precision_));
}

/**
 * Create a 2d cross-section from a set of contours (complex polygons). A
 * boolean union operation (with Positive filling rule by default) is
 * performed to combine overlapping polygons and ensure the resulting
 * CrossSection is free of intersections.
 *
 * @param contours A set of closed paths describing zero or more complex
 * polygons.
 * @param fillrule The filling rule used to interpret polygon sub-regions in
 * contours.
 */
CrossSection::CrossSection(const Polygons& contours, FillRule fillrule) {
  auto ps = C2::PathsD();
  ps.reserve(contours.size());
  for (auto ctr : contours) {
    ps.push_back(pathd_of_contour(ctr));
  }
  paths_ = shared_paths(C2::Union(ps, fr(fillrule), precision_));
}

/**
 * Create a 2d cross-section from an axis-aligned rectangle (bounding box).
 *
 * @param rect An axis-aligned rectangular bounding box.
 */
CrossSection::CrossSection(const Rect& rect) {
  C2::PathD p(4);
  p[0] = C2::PointD(rect.min.x, rect.min.y);
  p[1] = C2::PointD(rect.max.x, rect.min.y);
  p[2] = C2::PointD(rect.max.x, rect.max.y);
  p[3] = C2::PointD(rect.min.x, rect.max.y);
  paths_ = shared_paths(C2::PathsD{p});
}

// Private
// All access to paths_ should be done through the GetPaths() method, which
// applies the accumulated transform_
C2::PathsD CrossSection::GetPaths() const {
  if (transform_ == glm::mat3x2(1.0f)) {
    return *paths_;
  }
  paths_ = shared_paths(transform(*paths_, transform_));
  transform_ = glm::mat3x2(1.0f);
  return *paths_;
}

/**
 * Constructs a square with the given XY dimensions. By default it is
 * positioned in the first quadrant, touching the origin. If any dimensions in
 * size are negative, or if all are zero, an empty Manifold will be returned.
 *
 * @param size The X, and Y dimensions of the square.
 * @param center Set to true to shift the center to the origin.
 */
CrossSection CrossSection::Square(const glm::vec2 size, bool center) {
  if (size.x < 0.0f || size.y < 0.0f || glm::length(size) == 0.0f) {
    return CrossSection();
  }

  auto p = C2::PathD(4);
  if (center) {
    const auto w = size.x / 2;
    const auto h = size.y / 2;
    p[0] = C2::PointD(w, h);
    p[1] = C2::PointD(-w, h);
    p[2] = C2::PointD(-w, -h);
    p[3] = C2::PointD(w, -h);
  } else {
    const double x = size.x;
    const double y = size.y;
    p[0] = C2::PointD(0.0, 0.0);
    p[1] = C2::PointD(x, 0.0);
    p[2] = C2::PointD(x, y);
    p[3] = C2::PointD(0.0, y);
  }
  return CrossSection(C2::PathsD{p});
}

/**
 * Constructs a circle of a given radius.
 *
 * @param radius Radius of the circle. Must be positive.
 * @param circularSegments Number of segments along its diameter. Default is
 * calculated by the static Quality defaults according to the radius.
 */
CrossSection CrossSection::Circle(float radius, int circularSegments) {
  if (radius <= 0.0f) {
    return CrossSection();
  }
  int n = circularSegments > 2 ? circularSegments
                               : Quality::GetCircularSegments(radius);
  float dPhi = 360.0f / n;
  auto circle = C2::PathD(n);
  for (int i = 0; i < n; ++i) {
    circle[i] = C2::PointD(radius * cosd(dPhi * i), radius * sind(dPhi * i));
  }
  return CrossSection(C2::PathsD{circle});
}

/**
 * Perform the given boolean operation between this and another CrossSection.
 */
CrossSection CrossSection::Boolean(const CrossSection& second,
                                   OpType op) const {
  auto ct = cliptype_of_op(op);
  auto res = C2::BooleanOp(ct, C2::FillRule::Positive, GetPaths(),
                           second.GetPaths(), precision_);
  return CrossSection(res);
}

/**
 * Perform the given boolean operation on a list of CrossSections. In case of
 * Subtract, all CrossSections in the tail are differenced from the head.
 */
CrossSection CrossSection::BatchBoolean(
    const std::vector<CrossSection>& crossSections, OpType op) {
  if (crossSections.size() == 0)
    return CrossSection();
  else if (crossSections.size() == 1)
    return crossSections[0];

  auto subjs = crossSections[0].GetPaths();
  int n_clips = 0;
  for (int i = 1; i < crossSections.size(); ++i) {
    n_clips += crossSections[i].GetPaths().size();
  }
  auto clips = C2::PathsD();
  clips.reserve(n_clips);
  for (int i = 1; i < crossSections.size(); ++i) {
    auto ps = crossSections[i].GetPaths();
    clips.insert(clips.end(), ps.begin(), ps.end());
  }

  auto ct = cliptype_of_op(op);
  auto res =
      C2::BooleanOp(ct, C2::FillRule::Positive, subjs, clips, precision_);
  return CrossSection(res);
}

/**
 * Compute the boolean union between two cross-sections.
 */
CrossSection CrossSection::operator+(const CrossSection& Q) const {
  return Boolean(Q, OpType::Add);
}

/**
 * Compute the boolean union between two cross-sections, assigning the result
 * to the first.
 */
CrossSection& CrossSection::operator+=(const CrossSection& Q) {
  *this = *this + Q;
  return *this;
}

/**
 * Compute the boolean difference of a (clip) cross-section from another
 * (subject).
 */
CrossSection CrossSection::operator-(const CrossSection& Q) const {
  return Boolean(Q, OpType::Subtract);
}

/**
 * Compute the boolean difference of a (clip) cross-section from a another
 * (subject), assigning the result to the subject.
 */
CrossSection& CrossSection::operator-=(const CrossSection& Q) {
  *this = *this - Q;
  return *this;
}

/**
 * Compute the boolean intersection between two cross-sections.
 */
CrossSection CrossSection::operator^(const CrossSection& Q) const {
  return Boolean(Q, OpType::Intersect);
}

/**
 * Compute the boolean intersection between two cross-sections, assigning the
 * result to the first.
 */
CrossSection& CrossSection::operator^=(const CrossSection& Q) {
  *this = *this ^ Q;
  return *this;
}

/**
 * Construct a CrossSection from a vector of other CrossSections (batch
 * boolean union).
 */
CrossSection CrossSection::Compose(std::vector<CrossSection>& crossSections) {
  return BatchBoolean(crossSections, OpType::Add);
}

/**
 * This operation returns a vector of CrossSections that are topologically
 * disconnected, each containing one outline contour with zero or more
 * holes.
 */
std::vector<CrossSection> CrossSection::Decompose() const {
  if (NumContour() < 2) {
    return std::vector<CrossSection>{CrossSection(*this)};
  }

  C2::PolyTreeD tree;
  C2::BooleanOp(C2::ClipType::Union, C2::FillRule::Positive, GetPaths(),
                C2::PathsD(), tree, precision_);

  auto polys = std::vector<C2::PathsD>();
  decompose_outline(&tree, polys, 0);

  auto n_polys = polys.size();
  auto comps = std::vector<CrossSection>(n_polys);
  // reverse the stack while wrapping
  for (int i = 0; i < n_polys; ++i) {
    comps[n_polys - i - 1] = CrossSection(polys[i]);
  }

  return comps;
}

/**
 * Move this CrossSection in space. This operation can be chained. Transforms
 * are combined and applied lazily.
 *
 * @param v The vector to add to every vertex.
 */
CrossSection CrossSection::Translate(const glm::vec2 v) const {
  glm::mat3x2 m(1.0f, 0.0f,  //
                0.0f, 1.0f,  //
                v.x, v.y);
  return Transform(m);
}

/**
 * Applies a (Z-axis) rotation to the CrossSection, in degrees. This operation
 * can be chained. Transforms are combined and applied lazily.
 *
 * @param degrees degrees about the Z-axis to rotate.
 */
CrossSection CrossSection::Rotate(float degrees) const {
  auto s = sind(degrees);
  auto c = cosd(degrees);
  glm::mat3x2 m(c, s,   //
                -s, c,  //
                0.0f, 0.0f);
  return Transform(m);
}

/**
 * Scale this CrossSection in space. This operation can be chained. Transforms
 * are combined and applied lazily.
 *
 * @param v The vector to multiply every vertex by per component.
 */
CrossSection CrossSection::Scale(const glm::vec2 scale) const {
  glm::mat3x2 m(scale.x, 0.0f,  //
                0.0f, scale.y,  //
                0.0f, 0.0f);
  return Transform(m);
}

/**
 * Mirror this CrossSection over the arbitrary axis described by the unit form
 * of the given vector. If the length of the vector is zero, an empty
 * CrossSection is returned. This operation can be chained. Transforms are
 * combined and applied lazily.
 *
 * @param ax the axis to be mirrored over
 */
CrossSection CrossSection::Mirror(const glm::vec2 ax) const {
  if (glm::length(ax) == 0.) {
    return CrossSection();
  }
  auto n = glm::normalize(glm::abs(ax));
  auto m = glm::mat3x2(glm::mat2(1.0f) - 2.0f * glm::outerProduct(n, n));
  return Transform(m);
}

/**
 * Transform this CrossSection in space. The first two columns form a 2x2
 * matrix transform and the last is a translation vector. This operation can
 * be chained. Transforms are combined and applied lazily.
 *
 * @param m The affine transform matrix to apply to all the vertices.
 */
CrossSection CrossSection::Transform(const glm::mat3x2& m) const {
  auto transformed = CrossSection();
  transformed.transform_ = m * glm::mat3(transform_);
  transformed.paths_ = paths_;
  return transformed;
}

/**
 * Move the vertices of this CrossSection (creating a new one) according to
 * any arbitrary input function, followed by a union operation (with a
 * Positive fill rule) that ensures any introduced intersections are not
 * included in the result.
 *
 * @param warpFunc A function that modifies a given vertex position.
 */
CrossSection CrossSection::Warp(
    std::function<void(glm::vec2&)> warpFunc) const {
  auto paths = GetPaths();
  auto warped = C2::PathsD();
  warped.reserve(paths.size());
  for (auto path : paths) {
    auto sz = path.size();
    auto s = C2::PathD(sz);
    for (int i = 0; i < sz; ++i) {
      auto v = v2_of_pd(path[i]);
      warpFunc(v);
      s[i] = v2_to_pd(v);
    }
    warped.push_back(s);
  }
  return CrossSection(C2::Union(warped, C2::FillRule::Positive, precision_));
}

/**
 * Remove vertices from the contours in this CrossSection that are less than
 * the specified distance epsilon from an imaginary line that passes through
 * its two adjacent vertices. Near duplicate vertices and collinear points
 * will be removed at lower epsilons, with elimination of line segments
 * becoming increasingly aggressive with larger epsilons.
 *
 * It is recommended to apply this function following Offset, in order to
 * clean up any spurious tiny line segments introduced that do not improve
 * quality in any meaningful way. This is particularly important if further
 * offseting operations are to be performed, which would compound the issue.
 */
CrossSection CrossSection::Simplify(double epsilon) const {
  auto ps = SimplifyPaths(GetPaths(), epsilon, false);
  return CrossSection(ps);
}

/**
 * Inflate the contours in CrossSection by the specified delta, handling
 * corners according to the given JoinType.
 *
 * @param delta Positive deltas will cause the expansion of outlining contours
 * to expand, and retraction of inner (hole) contours. Negative deltas will
 * have the opposite effect.
 * @param jt The join type specifying the treatment of contour joins
 * (corners).
 * @param miter_limit The maximum distance in multiples of delta that vertices
 * can be offset from their original positions with before squaring is
 * applied, <B>when the join type is Miter</B> (default is 2, which is the
 * minimum allowed). See the [Clipper2
 * MiterLimit](http://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/Properties/MiterLimit.htm)
 * page for a visual example.
 * @param circularSegments Number of segments per 360 degrees of
 * <B>JoinType::Round</B> corners (roughly, the number of vertices that
 * will be added to each contour). Default is calculated by the static Quality
 * defaults according to the radius.
 */
CrossSection CrossSection::Offset(double delta, JoinType jointype,
                                  double miter_limit,
                                  int circularSegments) const {
  double arc_tol = 0.;
  if (jointype == JoinType::Round) {
    int n = circularSegments > 2 ? circularSegments
                                 : Quality::GetCircularSegments(delta);
    // This calculates tolerance as a function of circular segments and delta
    // (radius) in order to get back the same number of segments in Clipper2:
    // steps_per_360 = PI / acos(1 - arc_tol / abs_delta)
    const double abs_delta = std::fabs(delta);
    const double scaled_delta = abs_delta * std::pow(10, precision_);
    arc_tol = (std::cos(Clipper2Lib::PI / n) - 1) * -scaled_delta;
  }
  auto ps =
      C2::InflatePaths(GetPaths(), delta, jt(jointype), C2::EndType::Polygon,
                       miter_limit, precision_, arc_tol);
  return CrossSection(ps);
}

/**
 * Compute the convex hull enveloping a set of cross-sections.
 *
 * @param crossSections A vector of cross-sections over which to compute a
 * convex hull.
 */
CrossSection CrossSection::Hull(
    const std::vector<CrossSection>& crossSections) {
  int n = 0;
  for (auto cs : crossSections) n += cs.NumVert();
  SimplePolygon pts;
  pts.reserve(n);
  for (auto cs : crossSections) {
    auto paths = cs.GetPaths();
    for (auto path : paths) {
      for (auto p : path) {
        pts.push_back(v2_of_pd(p));
      }
    }
  }
  return CrossSection(C2::PathsD{HullImpl(pts)});
}

/**
 * Compute the convex hull of this cross-section.
 */
CrossSection CrossSection::Hull() const {
  return Hull(std::vector<CrossSection>{*this});
}

/**
 * Compute the convex hull of a set of points. If the given points are fewer
 * than 3, an empty CrossSection will be returned.
 *
 * @param pts A vector of 2-dimensional points over which to compute a convex
 * hull.
 */
CrossSection CrossSection::Hull(SimplePolygon pts) {
  return CrossSection(C2::PathsD{HullImpl(pts)});
}

/**
 * Compute the convex hull of a set of points/polygons. If the given points are
 * fewer than 3, an empty CrossSection will be returned.
 *
 * @param pts A vector of vectors of 2-dimensional points over which to compute
 * a convex hull.
 */
CrossSection CrossSection::Hull(const Polygons polys) {
  SimplePolygon pts;
  for (auto poly : polys) {
    for (auto p : poly) {
      pts.push_back(p);
    }
  }
  return Hull(pts);
}

/**
 * Return the total area covered by complex polygons making up the
 * CrossSection.
 */
double CrossSection::Area() const { return C2::Area(GetPaths()); }

/**
 * Return the number of vertices in the CrossSection.
 */
int CrossSection::NumVert() const {
  int n = 0;
  auto paths = GetPaths();
  for (auto p : paths) {
    n += p.size();
  }
  return n;
}

/**
 * Return the number of contours (both outer and inner paths) in the
 * CrossSection.
 */
int CrossSection::NumContour() const { return GetPaths().size(); }

/**
 * Does the CrossSection contain any contours?
 */
bool CrossSection::IsEmpty() const { return GetPaths().empty(); }

/**
 * Returns the axis-aligned bounding rectangle of all the CrossSections'
 * vertices.
 */
Rect CrossSection::Bounds() const {
  auto r = C2::GetBounds(GetPaths());
  return Rect({r.left, r.bottom}, {r.right, r.top});
}

/**
 * Return the contours of this CrossSection as a Polygons.
 */
Polygons CrossSection::ToPolygons() const {
  auto polys = Polygons();
  auto paths = GetPaths();
  polys.reserve(paths.size());
  for (auto p : paths) {
    auto sp = SimplePolygon();
    sp.reserve(p.size());
    for (auto v : p) {
      sp.push_back({v.x, v.y});
    }
    polys.push_back(sp);
  }
  return polys;
}

// Rect

/**
 * Default constructor is an empty rectangle..
 */
Rect::Rect() {}

Rect::~Rect() = default;
Rect::Rect(Rect&&) noexcept = default;
Rect& Rect::operator=(Rect&&) noexcept = default;
Rect::Rect(const Rect& other) {
  min = glm::vec2(other.min);
  max = glm::vec2(other.max);
}

/**
 * Create a rectangle that contains the two given points.
 */
Rect::Rect(const glm::vec2 a, const glm::vec2 b) {
  min = glm::min(a, b);
  max = glm::max(a, b);
}

/**
 * Return the dimensions of the rectangle.
 */
glm::vec2 Rect::Size() const { return max - min; }

/**
 * Return the area of the rectangle.
 */
float Rect::Area() const {
  auto sz = Size();
  return sz.x * sz.y;
}

/**
 * Returns the absolute-largest coordinate value of any contained
 * point.
 */
float Rect::Scale() const {
  glm::vec2 absMax = glm::max(glm::abs(min), glm::abs(max));
  return glm::max(absMax.x, absMax.y);
}

/**
 * Returns the center point of the rectangle.
 */
glm::vec2 Rect::Center() const { return 0.5f * (max + min); }

/**
 * Does this rectangle contain (includes on border) the given point?
 */
bool Rect::Contains(const glm::vec2& p) const {
  return glm::all(glm::greaterThanEqual(p, min)) &&
         glm::all(glm::greaterThanEqual(max, p));
}

/**
 * Does this rectangle contain (includes equal) the given rectangle?
 */
bool Rect::Contains(const Rect& rect) const {
  return glm::all(glm::greaterThanEqual(rect.min, min)) &&
         glm::all(glm::greaterThanEqual(max, rect.max));
}

/**
 * Does this rectangle overlap the one given (including equality)?
 */
bool Rect::DoesOverlap(const Rect& rect) const {
  return min.x <= rect.max.x && min.y <= rect.max.y && max.x >= rect.min.x &&
         max.y >= rect.min.y;
}

/**
 * Is the rectangle empty (containing no space)?
 */
bool Rect::IsEmpty() const { return max.y <= min.y || max.x <= min.x; };

/**
 * Does this recangle have finite bounds?
 */
bool Rect::IsFinite() const {
  return glm::all(glm::isfinite(min)) && glm::all(glm::isfinite(max));
}

/**
 * Expand this rectangle (in place) to include the given point.
 */
void Rect::Union(const glm::vec2 p) {
  min = glm::min(min, p);
  max = glm::max(max, p);
}

/**
 * Expand this rectangle to include the given Rect.
 */
Rect Rect::Union(const Rect& rect) const {
  Rect out;
  out.min = glm::min(min, rect.min);
  out.max = glm::max(max, rect.max);
  return out;
}

/**
 * Shift this rectangle by the given vector.
 */
Rect Rect::operator+(const glm::vec2 shift) const {
  Rect out;
  out.min = min + shift;
  out.max = max + shift;
  return out;
}

/**
 * Shift this rectangle in-place by the given vector.
 */
Rect& Rect::operator+=(const glm::vec2 shift) {
  min += shift;
  max += shift;
  return *this;
}

/**
 * Scale this rectangle by the given vector.
 */
Rect Rect::operator*(const glm::vec2 scale) const {
  Rect out;
  out.min = min * scale;
  out.max = max * scale;
  return out;
}

/**
 * Scale this rectangle in-place by the given vector.
 */
Rect& Rect::operator*=(const glm::vec2 scale) {
  min *= scale;
  max *= scale;
  return *this;
}

/**
 * Transform the rectangle by the given axis-aligned affine transform.
 *
 * Ensure the transform passed in is axis-aligned (rotations are all
 * multiples of 90 degrees), or else the resulting rectangle will no longer
 * bound properly.
 */
Rect Rect::Transform(const glm::mat3x2& m) const {
  Rect rect;
  rect.min = m * glm::vec3(min, 1);
  rect.max = m * glm::vec3(max, 1);
  return rect;
}

/**
 * Return a CrossSection with an outline defined by this axis-aligned
 * rectangle.
 */
CrossSection Rect::AsCrossSection() const { return CrossSection(*this); }

}  // namespace manifold

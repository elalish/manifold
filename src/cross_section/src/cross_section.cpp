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

#include <clipper2/clipper.h>

#include <memory>
#include <vector>

#include "clipper2/clipper.core.h"
#include "clipper2/clipper.engine.h"
#include "clipper2/clipper.offset.h"
#include "glm/ext/matrix_float3x2.hpp"
#include "glm/ext/vector_float2.hpp"
#include "glm/geometric.hpp"
#include "glm/glm.hpp"
#include "public.h"

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
}  // namespace

namespace manifold {
CrossSection::CrossSection() { paths_ = shared_paths(C2::PathsD()); }
CrossSection::~CrossSection() = default;
CrossSection::CrossSection(CrossSection&&) noexcept = default;
CrossSection& CrossSection::operator=(CrossSection&&) noexcept = default;
CrossSection::CrossSection(const CrossSection& other) {
  paths_ = other.paths_;
  transform_ = other.transform_;
}
CrossSection::CrossSection(C2::PathsD ps) { paths_ = shared_paths(ps); }

CrossSection::CrossSection(const SimplePolygon& contour, FillRule fillrule) {
  auto ps = C2::PathsD{(pathd_of_contour(contour))};
  paths_ = shared_paths(C2::Union(ps, fr(fillrule), precision_));
}

CrossSection::CrossSection(const Polygons& contours, FillRule fillrule) {
  auto ps = C2::PathsD();
  ps.reserve(contours.size());
  for (auto ctr : contours) {
    ps.push_back(pathd_of_contour(ctr));
  }
  paths_ = shared_paths(C2::Union(ps, fr(fillrule), precision_));
}

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

CrossSection CrossSection::Square(const glm::vec2 dims, bool center) {
  auto p = C2::PathD(4);
  if (center) {
    auto w = dims.x / 2;
    auto h = dims.y / 2;
    p[0] = C2::PointD(w, h);
    p[1] = C2::PointD(-w, h);
    p[2] = C2::PointD(-w, -h);
    p[3] = C2::PointD(w, -h);
  } else {
    double x = dims.x;
    double y = dims.y;
    p[0] = C2::PointD(0.0, 0.0);
    p[1] = C2::PointD(x, 0.0);
    p[2] = C2::PointD(x, y);
    p[3] = C2::PointD(0.0, y);
  }
  return CrossSection(C2::PathsD{p});
}

CrossSection CrossSection::Circle(float radius, int circularSegments) {
  int n = circularSegments > 2 ? circularSegments
                               : Quality::GetCircularSegments(radius);
  float dPhi = 360.0f / n;
  auto circle = C2::PathD(n);
  for (int i = 0; i < n; ++i) {
    circle[i] = C2::PointD(radius * cosd(dPhi * i), radius * sind(dPhi * i));
  }
  return CrossSection(C2::PathsD{circle});
}

CrossSection CrossSection::Boolean(const CrossSection& second,
                                   OpType op) const {
  auto ct = cliptype_of_op(op);
  auto res = C2::BooleanOp(ct, C2::FillRule::Positive, GetPaths(),
                           second.GetPaths(), precision_);
  return CrossSection(res);
}

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
 * Shorthand for Boolean Union.
 */
CrossSection CrossSection::operator+(const CrossSection& Q) const {
  return Boolean(Q, OpType::Add);
}

/**
 * Shorthand for Boolean Union assignment.
 */
CrossSection& CrossSection::operator+=(const CrossSection& Q) {
  *this = *this + Q;
  return *this;
}

/**
 * Shorthand for Boolean Difference.
 */
CrossSection CrossSection::operator-(const CrossSection& Q) const {
  return Boolean(Q, OpType::Subtract);
}

/**
 * Shorthand for Boolean Difference assignment.
 */
CrossSection& CrossSection::operator-=(const CrossSection& Q) {
  *this = *this - Q;
  return *this;
}

/**
 * Shorthand for Boolean Intersection.
 */
CrossSection CrossSection::operator^(const CrossSection& Q) const {
  return Boolean(Q, OpType::Intersect);
}

/**
 * Shorthand for Boolean Intersection assignment.
 */
CrossSection& CrossSection::operator^=(const CrossSection& Q) {
  *this = *this ^ Q;
  return *this;
}

CrossSection CrossSection::RectClip(const Rect& rect) const {
  auto r = C2::RectD(rect.min.x, rect.min.y, rect.max.x, rect.max.y);
  auto ps = C2::RectClip(r, GetPaths(), false, precision_);
  return CrossSection(ps);
}

CrossSection CrossSection::Translate(const glm::vec2 v) const {
  glm::mat3x2 m(1.0f, 0.0f,  //
                0.0f, 1.0f,  //
                v.x, v.y);
  return Transform(m);
}

CrossSection CrossSection::Rotate(float degrees) const {
  auto s = sind(degrees);
  auto c = cosd(degrees);
  glm::mat3x2 m(c, s,   //
                -s, c,  //
                0.0f, 0.0f);
  return Transform(m);
}

CrossSection CrossSection::Scale(const glm::vec2 scale) const {
  glm::mat3x2 m(scale.x, 0.0f,  //
                0.0f, scale.y,  //
                0.0f, 0.0f);
  return Transform(m);
}

CrossSection CrossSection::Mirror(const glm::vec2 ax) const {
  if (glm::length(ax) == 0.) {
    return CrossSection();
  }
  auto n = glm::normalize(glm::abs(ax));
  auto m = glm::mat3x2(glm::mat2(1.0f) - 2.0f * glm::outerProduct(n, n));
  return Transform(m);
}

CrossSection CrossSection::Transform(const glm::mat3x2& m) const {
  auto transformed = CrossSection();
  transformed.transform_ = m * glm::mat3(transform_);
  transformed.paths_ = paths_;
  return transformed;
}

CrossSection CrossSection::Simplify(double epsilon) const {
  auto ps = SimplifyPaths(GetPaths(), epsilon, false);
  return CrossSection(ps);
}

CrossSection CrossSection::Offset(double delta, JoinType jointype,
                                  double miter_limit,
                                  double arc_tolerance) const {
  auto ps =
      C2::InflatePaths(GetPaths(), delta, jt(jointype), C2::EndType::Polygon,
                       miter_limit, precision_, arc_tolerance);
  return CrossSection(ps);
}

double CrossSection::Area() const { return C2::Area(GetPaths()); }
Rect CrossSection::Bounds() const {
  auto r = C2::GetBounds(GetPaths());
  return Rect({r.left, r.bottom}, {r.right, r.top});
}
bool CrossSection::IsEmpty() const { return GetPaths().empty(); }

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

Rect::Rect() {}
Rect::~Rect() = default;
Rect::Rect(Rect&&) noexcept = default;
Rect& Rect::operator=(Rect&&) noexcept = default;
Rect::Rect(const Rect& other) {
  min = glm::vec2(other.min);
  max = glm::vec2(other.max);
}

Rect::Rect(const glm::vec2 a, const glm::vec2 b) {
  min = glm::min(a, b);
  max = glm::max(a, b);
}
glm::vec2 Rect::Size() const { return max - min; }

float Rect::Scale() const {
  glm::vec2 absMax = glm::max(glm::abs(min), glm::abs(max));
  return glm::max(absMax.x, absMax.y);
}

glm::vec2 Rect::Center() const { return 0.5f * (max + min); }

bool Rect::Contains(const glm::vec2& p) const {
  return glm::all(glm::greaterThanEqual(p, min)) &&
         glm::all(glm::greaterThanEqual(max, p));
}

bool Rect::Contains(const Rect& rect) const {
  return glm::all(glm::greaterThanEqual(rect.min, min)) &&
         glm::all(glm::greaterThanEqual(max, rect.max));
}

void Rect::Union(const glm::vec2 p) {
  min = glm::min(min, p);
  max = glm::max(max, p);
}

Rect Rect::Union(const Rect& rect) const {
  Rect out;
  out.min = glm::min(min, rect.min);
  out.max = glm::max(max, rect.max);
  return out;
}

Rect Rect::operator+(const glm::vec2 shift) const {
  Rect out;
  out.min = min + shift;
  out.max = max + shift;
  return out;
}

Rect& Rect::operator+=(const glm::vec2 shift) {
  min += shift;
  max += shift;
  return *this;
}

Rect Rect::operator*(const glm::vec2 scale) const {
  Rect out;
  out.min = min * scale;
  out.max = max * scale;
  return out;
}

Rect& Rect::operator*=(const glm::vec2 scale) {
  min *= scale;
  max *= scale;
  return *this;
}

Rect Rect::Transform(const glm::mat3x2& m) const {
  Rect rect;
  rect.min = m * glm::vec3(min, 1);
  rect.max = m * glm::vec3(max, 1);
  return rect;
}

bool Rect::DoesOverlap(const Rect& rect) const {
  return min.x <= rect.max.x && min.y <= rect.max.y && max.x >= rect.min.x &&
         max.y >= rect.min.y;
}

bool Rect::IsEmpty() const { return max.y <= min.y || max.x <= min.x; };
bool Rect::IsFinite() const {
  return glm::all(glm::isfinite(min)) && glm::all(glm::isfinite(max));
}

CrossSection Rect::AsCrossSection() const {
  SimplePolygon p(4);
  p[0] = glm::vec2(min.x, max.y);
  p[1] = glm::vec2(max.x, max.y);
  p[2] = glm::vec2(max.x, min.y);
  p[3] = glm::vec2(min.x, min.y);
  return CrossSection(p);
}

}  // namespace manifold

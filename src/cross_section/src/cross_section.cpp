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

#include "cross_section.h"

#include <clipper2/clipper.h>

#include <vector>

#include "clipper2/clipper.core.h"
#include "clipper2/clipper.engine.h"
#include "clipper2/clipper.offset.h"
#include "glm/ext/vector_float2.hpp"
#include "public.h"

using namespace manifold;

namespace {
const int precision_ = 8;

C2::ClipType cliptype_of_op(CrossSection::OpType op) {
  C2::ClipType ct = C2::ClipType::Union;
  switch (op) {
    case CrossSection::OpType::Add:
      break;
    case CrossSection::OpType::Subtract:
      ct = C2::ClipType::Difference;
      break;
    case CrossSection::OpType::Intersect:
      ct = C2::ClipType::Intersection;
      break;
    case CrossSection::OpType::Xor:
      ct = C2::ClipType::Xor;
      break;
  };
  return ct;
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

C2::PathD pathd_of_contour(std::vector<glm::vec2> ctr) {
  auto p = C2::PathD();
  for (auto v : ctr) {
    p.push_back(C2::PointD(v.x, v.y));
  }
  return p;
}
}  // namespace

namespace manifold {
CrossSection::CrossSection() { paths_ = C2::PathsD(); }
CrossSection::~CrossSection() = default;
CrossSection::CrossSection(CrossSection&&) noexcept = default;
CrossSection& CrossSection::operator=(CrossSection&&) noexcept = default;
CrossSection::CrossSection(const CrossSection& other) {
  paths_ = C2::PathsD(other.paths_);
  clean_ = other.clean_;
}
CrossSection::CrossSection(C2::PathsD ps, bool clean) {
  paths_ = ps;
  clean_ = clean;
}

CrossSection::CrossSection(std::vector<glm::vec2> contour) {
  auto ps = C2::PathsD();
  ps.push_back(pathd_of_contour(contour));
  paths_ = ps;
}

CrossSection::CrossSection(std::vector<std::vector<glm::vec2>> contours) {
  auto ps = C2::PathsD();
  for (auto ctr : contours) {
    ps.push_back(pathd_of_contour(ctr));
  }
  paths_ = ps;
}

CrossSection CrossSection::Square(glm::vec2 dims, bool center) {
  auto p = C2::PathD();
  if (center) {
    auto w = dims.x / 2;
    auto h = dims.y / 2;
    p.push_back(C2::PointD(w, h));
    p.push_back(C2::PointD(-w, h));
    p.push_back(C2::PointD(-w, -h));
    p.push_back(C2::PointD(w, -h));
  } else {
    double x = dims.x;
    double y = dims.y;
    p.push_back(C2::PointD(0.0, 0.0));
    p.push_back(C2::PointD(x, 0.0));
    p.push_back(C2::PointD(x, y));
    p.push_back(C2::PointD(0.0, y));
  }
  auto ps = C2::PathsD();
  ps.push_back(p);
  return CrossSection(ps, true);
}

CrossSection CrossSection::Circle(float radius, int circularSegments) {
  // GetCircularSegments(radius) -- in Manifold, not available atm
  int n = circularSegments > 2 ? circularSegments : 3;
  float dPhi = 360.0f / n;
  auto circle = C2::PathsD(1);
  for (int i = 0; i < n; ++i) {
    circle[0].push_back(
        C2::PointD(radius * cosd(dPhi * i), radius * sind(dPhi * i)));
  }
  return CrossSection(circle, true);
}

CrossSection CrossSection::Boolean(const CrossSection& second,
                                   OpType op) const {
  auto ct = cliptype_of_op(op);
  auto res = C2::BooleanOp(ct, C2::FillRule::NonZero, paths_, second.paths_,
                           precision_);
  return CrossSection(res, true);
}

CrossSection CrossSection::BatchBoolean(
    const std::vector<CrossSection>& crossSections, OpType op) {
  if (crossSections.size() == 0)
    return CrossSection();
  else if (crossSections.size() == 1)
    return crossSections[0];

  auto subjs = crossSections[0].paths_;
  auto clips = C2::PathsD();
  for (int i = 1; i < crossSections.size(); ++i) {
    auto ps = crossSections[i].paths_;
    clips.insert(clips.end(), ps.begin(), ps.end());
  }

  auto ct = cliptype_of_op(op);
  auto res = C2::BooleanOp(ct, C2::FillRule::NonZero, subjs, clips, precision_);
  return CrossSection(res, true);
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

CrossSection CrossSection::Translate(glm::vec2 v) {
  auto ps = C2::TranslatePaths(paths_, v.x, v.y);
  return CrossSection(ps, true);
}

CrossSection CrossSection::Scale(glm::vec2 scale) {
  auto scaled = C2::PathsD();
  for (auto path : paths_) {
    auto s = C2::PathD();
    for (auto p : path) {
      s.push_back(C2::PointD(p.x * scale.x, p.y * scale.y));
    }
    scaled.push_back(s);
  }
  return CrossSection(scaled, false);
}

CrossSection CrossSection::Simplify(double epsilon) {
  auto ps = SimplifyPaths(paths_, epsilon, false);
  return CrossSection(ps, false);
}

CrossSection CrossSection::RamerDouglasPeucker(double epsilon) {
  auto ps = C2::RamerDouglasPeucker(paths_, epsilon);
  return CrossSection(ps, false);
}

CrossSection CrossSection::StripNearEqual(double epsilon) {
  auto ps = C2::StripNearEqual(paths_, epsilon, true);
  return CrossSection(ps, false);
}

CrossSection CrossSection::StripDuplicates() {
  auto ps = C2::StripDuplicates(paths_, true);
  return CrossSection(ps, false);
}

CrossSection CrossSection::TrimCollinear() {
  auto trimmed = C2::PathsD();
  for (auto p : paths_) {
    trimmed.push_back(C2::TrimCollinear(p, false));
  }
  return CrossSection(trimmed, false);
}

CrossSection CrossSection::Offset(double delta, JoinType jointype,
                                  double miter_limit, double arc_tolerance) {
  auto ps = C2::InflatePaths(paths_, delta, jt(jointype), C2::EndType::Polygon,
                             miter_limit, precision_, arc_tolerance);
  return CrossSection(ps, true);
}

CrossSection CrossSection::MinkowskiSum(const std::vector<glm::vec2> pattern) {
  auto pat = pathd_of_contour(pattern);
  auto summed = C2::PathsD();
  for (auto p : paths_) {
    auto ss = C2::MinkowskiSum(pat, p, true);
    summed.insert(summed.end(), ss.begin(), ss.end());
  }
  auto u = Union(summed, C2::FillRule::NonZero);
  return CrossSection(u, true);
}

CrossSection CrossSection::MinkowskiDiff(const std::vector<glm::vec2> pattern) {
  auto pat = pathd_of_contour(pattern);
  auto diffed = C2::PathsD();
  for (auto p : paths_) {
    auto ss = C2::MinkowskiDiff(pat, p, true);
    diffed.insert(diffed.end(), ss.begin(), ss.end());
  }
  auto u = Union(diffed, C2::FillRule::NonZero);
  return CrossSection(u, true);
}

Polygons CrossSection::ToPolygons() const {
  auto polys = Polygons();
  auto paths =
      clean_ ? paths_ : C2::Union(paths_, C2::FillRule::NonZero, precision_);
  for (auto p : paths) {
    auto sp = SimplePolygon();
    for (int i = 0; i < p.size(); ++i) {
      sp.push_back({{p[i].x, p[i].y}, i});
    }
    polys.push_back(sp);
  }
  return polys;
}

}  // namespace manifold

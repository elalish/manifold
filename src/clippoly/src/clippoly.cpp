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

#include "clippoly.h"

#include <clipper2/clipper.h>

#include <vector>

#include "clipper2/clipper.core.h"
#include "clipper2/clipper.engine.h"
#include "glm/ext/vector_float2.hpp"
#include "public.h"

using namespace manifold;

namespace {
const int precision_ = 8;

C2::ClipType cliptype_of_op(Clippoly::OpType op) {
  C2::ClipType ct = C2::ClipType::Union;
  switch (op) {
    case Clippoly::OpType::Add:
      break;
    case Clippoly::OpType::Subtract:
      ct = C2::ClipType::Difference;
      break;
    case Clippoly::OpType::Intersect:
      ct = C2::ClipType::Intersection;
      break;
    case Clippoly::OpType::Xor:
      ct = C2::ClipType::Xor;
      break;
  };
  return ct;
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
Clippoly::Clippoly() { paths_ = C2::PathsD(); }
Clippoly::~Clippoly() = default;
Clippoly::Clippoly(Clippoly&&) noexcept = default;
Clippoly& Clippoly::operator=(Clippoly&&) noexcept = default;
Clippoly::Clippoly(const Clippoly& other) {
  paths_ = C2::PathsD(other.paths_);
  clean_ = other.clean_;
}
Clippoly::Clippoly(C2::PathsD ps, bool clean) {
  paths_ = ps;
  clean_ = clean;
}

Clippoly::Clippoly(std::vector<glm::vec2> contour) {
  auto ps = C2::PathsD();
  ps.push_back(pathd_of_contour(contour));
  paths_ = ps;
}

Clippoly::Clippoly(std::vector<std::vector<glm::vec2>> contours) {
  auto ps = C2::PathsD();
  for (auto ctr : contours) {
    ps.push_back(pathd_of_contour(ctr));
  }
  paths_ = ps;
}

Clippoly Clippoly::Square(glm::vec2 dims, bool center) {
  auto p = C2::PathD();
  if (center) {
    auto w = dims.x / 2;
    auto h = dims.y / 2;
    p.push_back(C2::PointD(w, -h));
    p.push_back(C2::PointD(-w, -h));
    p.push_back(C2::PointD(-w, h));
    p.push_back(C2::PointD(w, h));
  } else {
    double x = dims.x;
    double y = dims.y;
    p.push_back(C2::PointD(0.0, y));
    p.push_back(C2::PointD(x, y));
    p.push_back(C2::PointD(x, 0.0));
    p.push_back(C2::PointD(0.0, 0.0));
  }
  auto ps = C2::PathsD();
  ps.push_back(p);
  return Clippoly(ps, true);
}

Clippoly Clippoly::Boolean(const Clippoly& second, OpType op) const {
  auto ct = cliptype_of_op(op);
  auto res = C2::BooleanOp(ct, C2::FillRule::NonZero, paths_, second.paths_,
                           precision_);
  return Clippoly(res, true);
}

Clippoly Clippoly::BatchBoolean(const std::vector<Clippoly>& clippolys,
                                OpType op) {
  if (clippolys.size() == 0)
    return Clippoly();
  else if (clippolys.size() == 1)
    return clippolys[0];

  auto subjs = clippolys[0].paths_;
  auto clips = C2::PathsD();
  for (int i = 1; i < clippolys.size(); ++i) {
    auto ps = clippolys[i].paths_;
    clips.insert(clips.end(), ps.begin(), ps.end());
  }

  auto ct = cliptype_of_op(op);
  auto res = C2::BooleanOp(ct, C2::FillRule::NonZero, subjs, clips, precision_);
  return Clippoly(res, true);
}

/**
 * Shorthand for Boolean Union.
 */
Clippoly Clippoly::operator+(const Clippoly& Q) const {
  return Boolean(Q, OpType::Add);
}

/**
 * Shorthand for Boolean Union assignment.
 */
Clippoly& Clippoly::operator+=(const Clippoly& Q) {
  *this = *this + Q;
  return *this;
}

/**
 * Shorthand for Boolean Difference.
 */
Clippoly Clippoly::operator-(const Clippoly& Q) const {
  return Boolean(Q, OpType::Subtract);
}

/**
 * Shorthand for Boolean Difference assignment.
 */
Clippoly& Clippoly::operator-=(const Clippoly& Q) {
  *this = *this - Q;
  return *this;
}

/**
 * Shorthand for Boolean Intersection.
 */
Clippoly Clippoly::operator^(const Clippoly& Q) const {
  return Boolean(Q, OpType::Intersect);
}

/**
 * Shorthand for Boolean Intersection assignment.
 */
Clippoly& Clippoly::operator^=(const Clippoly& Q) {
  *this = *this ^ Q;
  return *this;
}

Clippoly Clippoly::Translate(glm::vec2 v) {
  auto ps = C2::TranslatePaths(paths_, v.x, v.y);
  return Clippoly(ps, true);
}

Clippoly Clippoly::Scale(glm::vec2 scale) {
  auto scaled = C2::PathsD();
  for (auto path : paths_) {
    auto s = C2::PathD();
    for (auto p : path) {
      s.push_back(C2::PointD(p.x * scale.x, p.y * scale.y));
    }
    scaled.push_back(s);
  }
  return Clippoly(scaled, false);
}

Clippoly Clippoly::Simplify(double epsilon) {
  auto ps = SimplifyPaths(paths_, epsilon, false);
  return Clippoly(ps, false);
}

Clippoly Clippoly::RamerDouglasPeucker(double epsilon) {
  auto ps = C2::RamerDouglasPeucker(paths_, epsilon);
  return Clippoly(ps, false);
}

Clippoly Clippoly::StripNearEqual(double epsilon) {
  auto ps = C2::StripNearEqual(paths_, epsilon, true);
  return Clippoly(ps, false);
}

Clippoly Clippoly::StripDuplicates() {
  auto ps = C2::StripDuplicates(paths_, true);
  return Clippoly(ps, false);
}

Clippoly Clippoly::TrimCollinear() {
  auto trimmed = C2::PathsD();
  for (auto p : paths_) {
    trimmed.push_back(C2::TrimCollinear(p, false));
  }
  return Clippoly(trimmed, false);
}

Clippoly Clippoly::MinkowskiSum(const std::vector<glm::vec2> pattern) {
  auto pat = pathd_of_contour(pattern);
  auto summed = C2::PathsD();
  for (auto p : paths_) {
    auto ss = C2::MinkowskiSum(pat, p, true);
    summed.insert(summed.end(), ss.begin(), ss.end());
  }
  auto u = Union(summed, C2::FillRule::NonZero);
  return Clippoly(u, true);
}

Clippoly Clippoly::MinkowskiDiff(const std::vector<glm::vec2> pattern) {
  auto pat = pathd_of_contour(pattern);
  auto diffed = C2::PathsD();
  for (auto p : paths_) {
    auto ss = C2::MinkowskiDiff(pat, p, true);
    diffed.insert(diffed.end(), ss.begin(), ss.end());
  }
  auto u = Union(diffed, C2::FillRule::NonZero);
  return Clippoly(u, true);
}

Polygons Clippoly::ToPolygons() const {
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

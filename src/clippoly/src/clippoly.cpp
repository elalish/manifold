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
}  // namespace

namespace manifold {
Clippoly::Clippoly() { paths = C2::PathsD(); }
Clippoly::~Clippoly() = default;
Clippoly::Clippoly(Clippoly&&) noexcept = default;
Clippoly& Clippoly::operator=(Clippoly&&) noexcept = default;
Clippoly::Clippoly(const Clippoly& other) { paths = C2::PathsD(other.paths); }
Clippoly::Clippoly(C2::PathsD ps) { paths = ps; }

Clippoly::Clippoly(std::vector<glm::vec2> contour) {
  auto p = C2::PathD();
  for (auto v : contour) {
    p.push_back(C2::PointD(v.x, v.y));
  }
  auto ps = C2::PathsD();
  ps.push_back(p);
  paths = ps;
}

Clippoly::Clippoly(std::vector<std::vector<glm::vec2>> contours) {
  auto ps = C2::PathsD();
  for (auto ctr : contours) {
    auto p = C2::PathD();
    for (auto v : ctr) {
      p.push_back(C2::PointD(v.x, v.y));
    }
    ps.push_back(p);
  }
  paths = ps;
}

Clippoly Clippoly::Boolean(const Clippoly& second, OpType op) const {
  auto ct = cliptype_of_op(op);
  auto res =
      C2::BooleanOp(ct, C2::FillRule::NonZero, paths, second.paths, precision_);
  return Clippoly(res);
}

Clippoly Clippoly::BatchBoolean(const std::vector<Clippoly>& clippolys,
                                OpType op) {
  if (clippolys.size() == 0)
    return Clippoly();
  else if (clippolys.size() == 1)
    return clippolys[0];

  auto subjs = clippolys[0].paths;
  auto clips = C2::PathsD();
  for (int i = 1; i < clippolys.size(); ++i) {
    auto ps = clippolys[i].paths;
    clips.insert(clips.end(), ps.begin(), ps.end());
  }

  auto ct = cliptype_of_op(op);
  auto res = C2::BooleanOp(ct, C2::FillRule::NonZero, subjs, clips, precision_);
  return Clippoly(res);
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

}  // namespace manifold

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

#pragma once

#include <clipper2/clipper.h>

#include "clipper2/clipper.offset.h"
#include "glm/ext/vector_float2.hpp"
#include "public.h"

namespace C2 = Clipper2Lib;

namespace manifold {
class Clippoly {
 public:
  Clippoly();
  ~Clippoly();
  Clippoly(const Clippoly& other);
  Clippoly& operator=(const Clippoly& other);
  Clippoly(Clippoly&&) noexcept;
  Clippoly& operator=(Clippoly&&) noexcept;
  Clippoly(std::vector<glm::vec2> contour);
  Clippoly(std::vector<std::vector<glm::vec2>> contours);

  // Shapes
  static Clippoly Square(glm::vec2 dims, bool center = false);
  static Clippoly Circle(float radius, int circularSegments);

  // Booleans
  enum class OpType { Add, Subtract, Intersect, Xor };
  Clippoly Boolean(const Clippoly& second, OpType op) const;
  static Clippoly BatchBoolean(const std::vector<Clippoly>& clippolys,
                               OpType op);
  Clippoly operator+(const Clippoly&) const;  // Add (Union)
  Clippoly& operator+=(const Clippoly&);
  Clippoly operator-(const Clippoly&) const;  // Subtract (Difference)
  Clippoly& operator-=(const Clippoly&);
  Clippoly operator^(const Clippoly&) const;  // Intersect
  Clippoly& operator^=(const Clippoly&);

  // Transformations
  Clippoly Translate(glm::vec2 v);
  Clippoly Scale(glm::vec2 s);

  // Path Simplification
  Clippoly TrimCollinear();
  Clippoly Simplify(double epsilon = 1e-6);
  Clippoly RamerDouglasPeucker(double epsilon = 1e-6);
  Clippoly StripNearEqual(double epsilon = 1e-6);
  Clippoly StripDuplicates();

  enum class JoinType { Square, Round, Miter };
  Clippoly Offset(double delta, JoinType jt, double miter_limit = 2.0,
                  double arc_tolerance = 0.0);

  // Minkowski
  // NOTE: OpenSCAD does not use Minkowski as is (from Clipper1). Also, they
  // allow a list patterns (see applyMinkowski). Should read through that and
  // write these methods such that they can take Clippoly rather than a single
  // path pattern.
  // https://github.com/openscad/openscad/blob/master/src/geometry/ClipperUtils.cc#L190
  Clippoly MinkowskiSum(const std::vector<glm::vec2> pattern);
  Clippoly MinkowskiDiff(const std::vector<glm::vec2> pattern);

  // Output
  Polygons ToPolygons() const;

 private:
  C2::PathsD paths_;
  // if not clean, paths_ need to be unioned before extraction
  bool clean_ = false;
  Clippoly(C2::PathsD paths, bool clean);
};

}  // namespace manifold

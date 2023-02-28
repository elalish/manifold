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
class CrossSection {
 public:
  CrossSection();
  ~CrossSection();
  CrossSection(const CrossSection& other);
  CrossSection& operator=(const CrossSection& other);
  CrossSection(CrossSection&&) noexcept;
  CrossSection& operator=(CrossSection&&) noexcept;
  CrossSection(std::vector<glm::vec2> contour);
  CrossSection(std::vector<std::vector<glm::vec2>> contours);

  // Shapes
  static CrossSection Square(glm::vec2 dims, bool center = false);
  static CrossSection Circle(float radius, int circularSegments);

  // Booleans
  enum class OpType { Add, Subtract, Intersect, Xor };
  CrossSection Boolean(const CrossSection& second, OpType op) const;
  static CrossSection BatchBoolean(
      const std::vector<CrossSection>& crossSections, OpType op);
  CrossSection operator+(const CrossSection&) const;  // Add (Union)
  CrossSection& operator+=(const CrossSection&);
  CrossSection operator-(const CrossSection&) const;  // Subtract (Difference)
  CrossSection& operator-=(const CrossSection&);
  CrossSection operator^(const CrossSection&) const;  // Intersect
  CrossSection& operator^=(const CrossSection&);

  // Transformations
  CrossSection Translate(glm::vec2 v);
  CrossSection Scale(glm::vec2 s);

  // Path Simplification
  CrossSection TrimCollinear();
  CrossSection Simplify(double epsilon = 1e-6);
  CrossSection RamerDouglasPeucker(double epsilon = 1e-6);
  CrossSection StripNearEqual(double epsilon = 1e-6);
  CrossSection StripDuplicates();

  enum class JoinType { Square, Round, Miter };
  CrossSection Offset(double delta, JoinType jt, double miter_limit = 2.0,
                      double arc_tolerance = 0.0);

  // Minkowski
  // NOTE: OpenSCAD does not use Minkowski as is (from Clipper1). Also, they
  // allow a list patterns (see applyMinkowski). Should read through that and
  // write these methods such that they can take CrossSection rather than a
  // single path pattern.
  // https://github.com/openscad/openscad/blob/master/src/geometry/ClipperUtils.cc#L190
  CrossSection MinkowskiSum(const std::vector<glm::vec2> pattern);
  CrossSection MinkowskiDiff(const std::vector<glm::vec2> pattern);

  // Output
  Polygons ToPolygons() const;

 private:
  C2::PathsD paths_;
  // if not clean, paths_ need to be unioned before extraction
  bool clean_ = false;
  CrossSection(C2::PathsD paths, bool clean);
};

}  // namespace manifold

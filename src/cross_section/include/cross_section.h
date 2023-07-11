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

#pragma once

#include <clipper2/clipper.h>

#include <memory>
#include <vector>

#include "clipper2/clipper.core.h"
#include "clipper2/clipper.offset.h"
#include "glm/ext/matrix_float3x2.hpp"
#include "glm/ext/vector_float2.hpp"
#include "public.h"

namespace C2 = Clipper2Lib;

namespace manifold {

class Rect;

/** @addtogroup Core
 *  @{
 */

/**
 * Two-dimensional cross sections guaranteed to be without self-intersections,
 * or overlaps between polygons (from construction onwards). This class makes
 * use of the [Clipper2](http://www.angusj.com/clipper2/Docs/Overview.htm)
 * library for polygon clipping (boolean) and offsetting operations.
 */
class CrossSection {
 public:
  /** @name Creation
   *  Constructors
   */
  ///@{

  CrossSection();
  ~CrossSection();

  CrossSection(const CrossSection& other);
  CrossSection& operator=(const CrossSection& other);
  CrossSection(CrossSection&&) noexcept;
  CrossSection& operator=(CrossSection&&) noexcept;

  // Adapted from Clipper2 docs:
  // http://www.angusj.com/clipper2/Docs/Units/Clipper/Types/FillRule.htm
  // (Copyright © 2010-2023 Angus Johnson)
  /**
   * Filling rules defining which polygon sub-regions are considered to be
   * inside a given polygon, and which sub-regions will not (based on winding
   * numbers). See the [Clipper2
   * docs](http://www.angusj.com/clipper2/Docs/Units/Clipper/Types/FillRule.htm)
   * for a detailed explaination with illusrations.
   */
  enum class FillRule {
    EvenOdd,   ///< Only odd numbered sub-regions are filled.
    NonZero,   ///< Only non-zero sub-regions are filled.
    Positive,  ///< Only sub-regions with winding counts > 0 are filled.
    Negative   ///< Only sub-regions with winding counts < 0 are filled.
  };

  CrossSection(const SimplePolygon& contour,
               FillRule fillrule = FillRule::Positive);
  CrossSection(const Polygons& contours,
               FillRule fillrule = FillRule::Positive);
  CrossSection(const Rect& rect);
  static CrossSection Square(const glm::vec2 dims, bool center = false);
  static CrossSection Circle(float radius, int circularSegments = 0);
  ///@}

  /** @name Information
   *  Details of the cross-section
   */
  ///@{
  double Area() const;
  int NumVert() const;
  int NumContour() const;
  bool IsEmpty() const;
  Rect Bounds() const;
  ///@}

  /** @name Modification
   */
  ///@{
  CrossSection Translate(const glm::vec2 v) const;
  CrossSection Rotate(float degrees) const;
  CrossSection Scale(const glm::vec2 s) const;
  CrossSection Mirror(const glm::vec2 ax) const;
  CrossSection Transform(const glm::mat3x2& m) const;
  CrossSection Warp(std::function<void(glm::vec2&)> warpFunc) const;
  CrossSection Simplify(double epsilon = 1e-6) const;

  // Adapted from Clipper2 docs:
  // http://www.angusj.com/clipper2/Docs/Units/Clipper/Types/JoinType.htm
  // (Copyright © 2010-2023 Angus Johnson)
  /**
   * Specifies the treatment of path/contour joins (corners) when offseting
   * CrossSections. See the [Clipper2
   * doc](http://www.angusj.com/clipper2/Docs/Units/Clipper/Types/JoinType.htm)
   * for illustrations.
   */
  enum class JoinType {
    Square, /*!< Squaring is applied uniformly at all joins where the internal
              join angle is less that 90 degrees. The squared edge will be at
              exactly the offset distance from the join vertex. */
    Round,  /*!< Rounding is applied to all joins that have convex external
             angles, and it maintains the exact offset distance from the join
             vertex. */
    Miter   /*!< There's a necessary limit to mitered joins (to avoid narrow
             angled joins producing excessively long and narrow
             [spikes](http://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/Properties/MiterLimit.htm)).
             So where mitered joins would exceed a given maximum miter distance
             (relative to the offset distance), these are 'squared' instead. */
  };

  CrossSection Offset(double delta, JoinType jt, double miter_limit = 2.0,
                      int circularSegments = 0) const;
  ///@}

  /** @name Boolean
   *  Combine two manifolds
   */
  ///@{
  CrossSection Boolean(const CrossSection& second, OpType op) const;
  static CrossSection BatchBoolean(
      const std::vector<CrossSection>& crossSections, OpType op);
  CrossSection operator+(const CrossSection&) const;
  CrossSection& operator+=(const CrossSection&);
  CrossSection operator-(const CrossSection&) const;
  CrossSection& operator-=(const CrossSection&);
  CrossSection operator^(const CrossSection&) const;
  CrossSection& operator^=(const CrossSection&);
  ///@}

  /** @name Topological
   */
  ///@{
  static CrossSection Compose(std::vector<CrossSection>&);
  std::vector<CrossSection> Decompose() const;
  ///@}

  /** @name Convex Hulling
   */
  ///@{
  CrossSection Hull() const;
  static CrossSection Hull(const std::vector<CrossSection>& crossSections);
  static CrossSection Hull(const SimplePolygon poly);
  static CrossSection Hull(const Polygons polys);
  ///@}
  ///
  /** @name Conversion
   */
  ///@{
  Polygons ToPolygons() const;
  ///@}

 private:
  mutable std::shared_ptr<const C2::PathsD> paths_;
  mutable glm::mat3x2 transform_ = glm::mat3x2(1.0f);
  CrossSection(C2::PathsD paths);
  C2::PathsD GetPaths() const;
};
/** @} */

/** @addtogroup Connections
 *  @{
 */

/**
 * Axis-aligned rectangular bounds.
 */
class Rect {
 public:
  glm::vec2 min = glm::vec2(0);
  glm::vec2 max = glm::vec2(0);

  /** @name Creation
   *  Constructors
   */
  ///@{
  Rect();
  ~Rect();
  Rect(const Rect& other);
  Rect& operator=(const Rect& other);
  Rect(Rect&&) noexcept;
  Rect& operator=(Rect&&) noexcept;
  Rect(const glm::vec2 a, const glm::vec2 b);
  ///@}

  /** @name Information
   *  Details of the rectangle
   */
  ///@{
  glm::vec2 Size() const;
  float Area() const;
  float Scale() const;
  glm::vec2 Center() const;
  bool Contains(const glm::vec2& pt) const;
  bool Contains(const Rect& other) const;
  bool DoesOverlap(const Rect& other) const;
  bool IsEmpty() const;
  bool IsFinite() const;
  ///@}

  /** @name Modification
   */
  ///@{
  void Union(const glm::vec2 p);
  Rect Union(const Rect& other) const;
  Rect operator+(const glm::vec2 shift) const;
  Rect& operator+=(const glm::vec2 shift);
  Rect operator*(const glm::vec2 scale) const;
  Rect& operator*=(const glm::vec2 scale);
  Rect Transform(const glm::mat3x2& m) const;
  ///@}

  /** @name Conversion
   */
  ///@{
  CrossSection AsCrossSection() const;
  ///@}
};
/** @} */
}  // namespace manifold

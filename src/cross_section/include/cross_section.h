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

class CrossSection {
 public:
  /** @name Creation
   *  Constructors
   */
  ///@{

  /**
   * The default constructor is an empty cross-section (containing no contours).
   */
  CrossSection();
  ~CrossSection();
  CrossSection(const CrossSection& other);
  CrossSection& operator=(const CrossSection& other);
  CrossSection(CrossSection&&) noexcept;
  CrossSection& operator=(CrossSection&&) noexcept;

  enum class FillRule { EvenOdd, NonZero, Positive, Negative };
  CrossSection(const SimplePolygon& contour,
               FillRule fillrule = FillRule::Positive);
  CrossSection(const Polygons& contours,
               FillRule fillrule = FillRule::Positive);

  static CrossSection Square(const glm::vec2 dims, bool center = false);
  static CrossSection Circle(float radius, int circularSegments = 0);
  ///@}

  /** @name Information
   *  Details of the cross-section
   */
  ///@{
  // Output
  Polygons ToPolygons() const;
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
  enum class JoinType { Square, Round, Miter };
  CrossSection Offset(double delta, JoinType jt, double miter_limit = 2.0,
                      double arc_tolerance = 0.0) const;
  ///@}

  /** @name Boolean
   *  Combine two manifolds
   */
  ///@{
  CrossSection Boolean(const CrossSection& second, OpType op) const;
  static CrossSection BatchBoolean(
      const std::vector<CrossSection>& crossSections, OpType op);
  CrossSection operator+(const CrossSection&) const;  // Add (Union)
  CrossSection& operator+=(const CrossSection&);
  CrossSection operator-(const CrossSection&) const;  // Subtract (Difference)
  CrossSection& operator-=(const CrossSection&);
  CrossSection operator^(const CrossSection&) const;  // Intersect
  CrossSection& operator^=(const CrossSection&);
  CrossSection RectClip(const Rect& rect) const;
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

  /**
   * Default constructor is an empty rectangle..
   */
  Rect();
  ~Rect();
  Rect(const Rect& other);
  Rect& operator=(const Rect& other);
  Rect(Rect&&) noexcept;
  Rect& operator=(Rect&&) noexcept;

  /**
   * Creates a rectangle that contains the two given points.
   */
  Rect(const glm::vec2 a, const glm::vec2 b);

  /**
   * Returns the dimensions of the rectangle.
   */
  glm::vec2 Size() const;

  /**
   * Returns the absolute-largest coordinate value of any contained
   * point.
   */
  float Scale() const;

  /**
   * Returns the center point of the rectangle.
   */
  glm::vec2 Center() const;

  /**
   * Does this rectangle contain (includes equal) the given point?
   */
  bool Contains(const glm::vec2& pt) const;

  /**
   * Does this rectangle contain (includes equal) the given rectangle?
   */
  bool Contains(const Rect& other) const;

  /**
   * Does this rectangle overlap the one given (including equality)?
   */
  bool DoesOverlap(const Rect& other) const;

  /**
   * Is the rectangle empty (containing no space)?
   */
  bool IsEmpty() const;

  /**
   * Does this recangle have finite bounds?
   */
  bool IsFinite() const;

  /**
   * Expand this rectangle (in place) to include the given point.
   */
  void Union(const glm::vec2 p);

  /**
   * Expand this rectangle to include the given Rect.
   */
  Rect Union(const Rect& other) const;

  /**
   * Shift this rectangle by the given vector.
   */
  Rect operator+(const glm::vec2 shift) const;

  /**
   * Shift this rectangle in-place by the given vector.
   */
  Rect& operator+=(const glm::vec2 shift);

  /**
   * Scale this rectangle by the given vector.
   */
  Rect operator*(const glm::vec2 scale) const;

  /**
   * Scale this rectangle in-place by the given vector.
   */
  Rect& operator*=(const glm::vec2 scale);

  /**
   * Transform the rectangle by the given axis-aligned affine transform.
   *
   * Ensure the transform passed in is axis-aligned (rotations are all
   * multiples of 90 degrees), or else the resulting rectangle will no longer
   * bound properly.
   */
  Rect Transform(const glm::mat3x2& m) const;

  /**
   * Return a CrossSection with an outline defined by this axis-aligned
   * rectangle.
   */
  CrossSection AsCrossSection() const;
};
/** @} */
}  // namespace manifold

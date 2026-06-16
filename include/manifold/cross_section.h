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

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "manifold/common.h"
#include "manifold/vec_view.h"

namespace manifold {

/** @addtogroup Optional
 * @brief Optional features that can be enabled through build flags and may
 * require extra dependencies.
 *  @{
 */

struct PathImpl;

/**
 * @brief Two-dimensional cross sections guaranteed to be without
 * self-intersections, or overlaps between polygons (from construction onwards).
 * Polygon clipping (boolean) and offsetting use Manifold's own robust
 * floating-point predicates.
 */
class CrossSection {
 public:
  /** @name Basics
   *  Copy / move / assignment
   */
  ///@{
  CrossSection();
  ~CrossSection();

  CrossSection(const CrossSection& other);
  CrossSection& operator=(const CrossSection& other);
  CrossSection(CrossSection&&) noexcept;
  CrossSection& operator=(CrossSection&&) noexcept;
  ///@}

  /**
   * Specifies the treatment of path/contour joins (corners) when offseting
   * CrossSections; alias of manifold::JoinType (see common.h), shared with the
   * polygon offset implementation.
   */
  using JoinType = ::manifold::JoinType;

  /** @name Input & Output
   */
  ///@{
  CrossSection(const SimplePolygon& contour);
  CrossSection(const Polygons& contours);
  CrossSection(const Rect& rect);
  Polygons ToPolygons() const;
  ///@}

  /** @name Constructors
   * Topological ops and primitives
   */
  ///@{
  std::vector<CrossSection> Decompose() const;
  static CrossSection Square(const vec2 dims, bool center = false);
  static CrossSection Circle(double radius, int circularSegments = 0);
  ///@}

  /** @name Information
   *  Details of the cross-section
   */
  ///@{
  bool IsEmpty() const;
  size_t NumVert() const;
  size_t NumContour() const;
  Rect Bounds() const;
  double Area() const;
  double GetTolerance() const;
  CrossSection SetTolerance(double tolerance) const;
  ///@}

  /** @name Transformation
   */
  ///@{
  CrossSection Translate(const vec2 v) const;
  CrossSection Rotate(double degrees) const;
  CrossSection Scale(const vec2 s) const;
  CrossSection Mirror(const vec2 ax) const;
  CrossSection Transform(const mat2x3& m) const;
  CrossSection Warp(std::function<void(vec2&)> warpFunc) const;
  CrossSection WarpBatch(std::function<void(VecView<vec2>)> warpFunc) const;
  CrossSection Simplify(double tolerance = 0) const;
  CrossSection Offset(double delta, JoinType jt = JoinType::Round,
                      double miter_limit = 2.0, int circularSegments = 0) const;
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

  /** @name Convex Hull
   */
  ///@{
  CrossSection Hull() const;
  static CrossSection Hull(const std::vector<CrossSection>& crossSections);
  static CrossSection Hull(const SimplePolygon& pts);
  static CrossSection Hull(const Polygons& polys);
  ///@}

 private:
  mutable std::mutex pathsMutex_;
  mutable std::shared_ptr<const PathImpl> paths_;
  mutable mat2x3 transform_ = la::identity;
  // Propagated drift budget, analogous to Manifold::Impl::tolerance_.
  mutable double tolerance_ = 0.0;
  CrossSection(std::shared_ptr<const PathImpl> paths);
  std::shared_ptr<const PathImpl> GetPaths() const;
};
/** @} */
}  // namespace manifold

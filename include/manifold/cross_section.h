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
 * This class makes use of the
 * [Clipper2](http://www.angusj.com/clipper2/Docs/Overview.htm) library for
 * polygon clipping (boolean) and offsetting operations.
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
    Miter,  /*!< There's a necessary limit to mitered joins (to avoid narrow
             angled joins producing excessively long and narrow
             [spikes](http://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/Properties/MiterLimit.htm)).
             So where mitered joins would exceed a given maximum miter distance
             (relative to the offset distance), these are 'squared' instead. */
    Bevel   /*!< Bevelled joins are similar to 'squared' joins except that
             squaring won't occur at a fixed distance. While bevelled joins may
             not be as pretty as squared joins, bevelling is much easier (ie
             faster) than squaring. And perhaps this is why bevelling rather
             than squaring is preferred in numerous graphics display formats
             (including
             [SVG](https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/stroke-linejoin)
             and
             [PDF](https://helpx.adobe.com/indesign/using/applying-line-stroke-settings.html)
             document formats). */
  };

  /** @name Input & Output
   */
  ///@{
  CrossSection(const SimplePolygon& contour,
               FillRule fillrule = FillRule::Positive);
  CrossSection(const Polygons& contours,
               FillRule fillrule = FillRule::Positive);
  CrossSection(const Rect& rect);
  Polygons ToPolygons() const;
  ///@}

  /** @name Constructors
   * Topological ops and primitives
   */
  ///@{
  std::vector<CrossSection> Decompose() const;
  static CrossSection Compose(const std::vector<CrossSection>&);
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
  CrossSection Simplify(double epsilon = 1e-6) const;
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
  static CrossSection Hull(const SimplePolygon pts);
  static CrossSection Hull(const Polygons polys);
  ///@}

 private:
  mutable std::shared_ptr<const PathImpl> paths_;
  mutable mat2x3 transform_ = la::identity;
  CrossSection(std::shared_ptr<const PathImpl> paths);
  std::shared_ptr<const PathImpl> GetPaths() const;
};
/** @} */
}  // namespace manifold

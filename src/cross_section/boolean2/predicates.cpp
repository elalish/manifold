// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Leaf primitives for the Boolean2 pipeline.

#include "predicates.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../../shared.h"
#include "../../utils.h"

namespace manifold {
namespace boolean2 {

// Centered-shoelace signed area of a closed polygon loop. Same FP trick
// as the per-face area computation in winding_filter.h: subtract a
// reference vert
// before multiplying so products stay at edge-length scale instead of
// blowing up to O(L^2) at displaced coordinates. Total telescopes to the
// same answer as the raw shoelace because Sigma(b - a) around any closed
// loop is zero. Used by area-preservation regression tests and the
// area-drift tracking in DeepFuzz.
double SignedArea(const SimplePolygon& loop) {
  if (loop.size() < 3) return 0.0;
  const auto& r = loop[0];
  double sum = 0.0;
  for (size_t i = 0; i < loop.size(); ++i) {
    const auto& a = loop[i];
    const auto& b = loop[(i + 1) % loop.size()];
    const double ax = a.x - r.x, ay = a.y - r.y;
    const double bx = b.x - r.x, by = b.y - r.y;
    sum += ax * by - bx * ay;
  }
  return 0.5 * sum;
}

double TotalSignedArea(const Polygons& polys) {
  double total = 0.0;
  for (const auto& loop : polys) total += SignedArea(loop);
  return total;
}

// Choose epsilon for the operation. L = bounding box half-extent rounded
// up to power of 2. k_budget is the user's expected upper bound on how
// many times any one edge may be adjusted (default 1000 ~= 10^-12 L).
double EpsilonFromScale(double L, int k_budget) {
  // Round L up to power of 2 (Smith's analysis assumes this).
  if (L <= 0) return 0;
  int expBits;
  std::frexp(L, &expBits);
  const double L_pow2 = std::ldexp(1.0, expBits);
  return (k_budget + 1) * kAlphaCoeff * kU * L_pow2;
}

// 2D edge-edge symbolic intersection (BVH-friendly).
//
// The classical Kernel11 from boolean3.cpp can't be used in BVH-pair-query
// context: it requires "one endpoint inside, one outside" the other
// segment's projection, which sweep-line guarantees but BVH pair queries
// don't (most pairs have one segment fully nested in the other's
// projection).
//
// This kernel works for any pair by trimming both segments to their
// projection-axis overlap before applying Intersect. Steps:
//   1. CCW + SoS for cross-or-not (existing predicate, untouched).
//   2. Pick the axis (x or y) where BOTH segments have non-zero spread,
//      preferring the larger min-spread for stability.
//   3. Sort each segment's endpoints L-to-R along that axis.
//   4. Compute axis-overlap interval [overlapL, overlapR] = intersection
//      of a's and b's axis spans.
//   5. Use `Interpolate` (from shared.h) to evaluate each segment's
//      orthogonal coord at overlapL and overlapR. This produces four
//      (axis, ortho) points all spanning the same axis interval, which
//      is the precondition Intersect's closed-form expects.
//   6. Apply the Boolean intersection closed-form (smaller |dy| endpoint
//      picked for FP stability) to compute the intersection position.
//
// Trimming makes both segments span the same axis interval by
// construction, so the Kernel11 "inside/outside endpoint" precondition is
// satisfied even for the nested-axis cases that arise from BVH pairs.
double Coord(vec2 p, int axis) { return axis == 0 ? p.x : p.y; }

int CCW(vec2 a, vec2 b, vec2 c, double eps) {
  return manifold::CCW(a, b, c, eps);
}

bool IntersectSegments(vec2 a0, vec2 a1, vec2 b0, vec2 b1, double eps,
                       vec2* out) {
  // Cross-detection: CCW(A,B,C) is sign of (B-A) x (C-A) with a
  // collinearity threshold; we reject any case where one or more
  // endpoints lie on the other segment's line (T-junction or fully
  // collinear). Real point-intersection requires endpoints strictly on
  // opposite sides of each other.
  const int s1 = CCW(a0, a1, b0, eps);
  const int s2 = CCW(a0, a1, b1, eps);
  const int s3 = CCW(b0, b1, a0, eps);
  const int s4 = CCW(b0, b1, a1, eps);
  const int zeros = (s1 == 0) + (s2 == 0) + (s3 == 0) + (s4 == 0);
  // Any collinear case: no isolated point intersection. Fully collinear
  // (zeros==4) means the segments are coincident or overlapping along a
  // 1D segment, with no point-intersection; partial collinear (one or
  // more but not all zero) means a T-junction. Both are handled by
  // BuildEdgeVertLists (vertex-on-edge insertion) and Canonicalize
  // (sub-edge multiplicity sum). Returning false here avoids unstable,
  // order-dependent point intersections from collinear inputs.
  if (zeros > 0) return false;
  if (s1 == s2 || s3 == s4) return false;

  // Pick the axis where BOTH segments have non-zero spread, with
  // the larger spread of the two preferring stability (smaller |dy| works
  // better in Intersect when the trimmed segments are well-separated).
  // The min-spread per axis is what matters: a vertical segment has zero
  // x-spread, so x is unusable; we'd pick y. Without this check (just
  // bbox spread), the Smith hexagon's vertical CE segment causes
  // degenerate axis-overlap (overlapL == overlapR) and the kernel falsely
  // reports no intersection.
  const double aSpreadX = std::fabs(a1.x - a0.x);
  const double aSpreadY = std::fabs(a1.y - a0.y);
  const double bSpreadX = std::fabs(b1.x - b0.x);
  const double bSpreadY = std::fabs(b1.y - b0.y);
  const double xUsable = std::min(aSpreadX, bSpreadX);
  const double yUsable = std::min(aSpreadY, bSpreadY);

  // Special case: both segments are axis-aligned to opposite axes (one
  // horizontal, one vertical, exactly), so neither axis has both
  // segments contributing spread. Trim-and-Interpolate would degenerate
  // (zero-width overlap interval). Compute the intersection directly:
  // it's the cross of the vertical segment's constant x and the
  // horizontal segment's constant y. Common in real CAD/SVG inputs
  // (axis-aligned rectangles overlapping each other).
  if (xUsable == 0 && yUsable == 0) {
    const bool aHoriz = aSpreadX > 0 && aSpreadY == 0;
    const bool aVert = aSpreadY > 0 && aSpreadX == 0;
    const bool bHoriz = bSpreadX > 0 && bSpreadY == 0;
    const bool bVert = bSpreadY > 0 && bSpreadX == 0;
    if ((aHoriz && bVert) || (aVert && bHoriz)) {
      const double ix = aVert ? a0.x : b0.x;
      const double iy = aHoriz ? a0.y : b0.y;
      *out = vec2(ix, iy);
      return std::isfinite(out->x) && std::isfinite(out->y);
    }
    return false;  // both points, or other degenerate config
  }

  const int axis = xUsable >= yUsable ? 0 : 1;

  // Sort each segment along the chosen axis.
  vec2 aL = a0, aR = a1, bL = b0, bR = b1;
  if (Coord(aR, axis) < Coord(aL, axis)) std::swap(aL, aR);
  if (Coord(bR, axis) < Coord(bL, axis)) std::swap(bL, bR);

  // Axis-overlap interval. CCW already confirmed crossing, so the
  // overlap is non-empty (modulo FP noise; clamp on hairline).
  const double overlapL = std::max(Coord(aL, axis), Coord(bL, axis));
  const double overlapR = std::min(Coord(aR, axis), Coord(bR, axis));
  if (overlapR <= overlapL) return false;

  // Trim each segment to the overlap. Interpolate(aL, aR, x) wants
  // a vec3 with the projection axis as its x-component, so we permute when
  // axis == 1 (y becomes x in the call frame, x becomes the orthogonal
  // coord). The returned vec2's first component is the orthogonal coord
  // at the requested projection value.
  auto embed = [&](vec2 p) {
    return axis == 0 ? vec3(p.x, p.y, 0.0) : vec3(p.y, p.x, 0.0);
  };
  const vec3 aL3 = embed(aL), aR3 = embed(aR);
  const vec3 bL3 = embed(bL), bR3 = embed(bR);
  const double aOL = manifold::Interpolate(aL3, aR3, overlapL).x;
  const double aOR = manifold::Interpolate(aL3, aR3, overlapR).x;
  const double bOL = manifold::Interpolate(bL3, bR3, overlapL).x;
  const double bOR = manifold::Interpolate(bL3, bR3, overlapR).x;

  // Intersect closed-form. dyL/dyR are the ortho gaps at the two overlap
  // boundaries; pick whichever has smaller |dy| as the lambda basepoint for
  // FP stability.
  const double dyL = bOL - aOL;
  const double dyR = bOR - aOR;
  const bool useL = std::fabs(dyL) < std::fabs(dyR);
  const double dProj = overlapR - overlapL;
  double lambda = (useL ? dyL : dyR) / (dyL - dyR);
  if (!std::isfinite(lambda)) return false;
  const double outProj = lambda * dProj + (useL ? overlapL : overlapR);
  const double aDy = aOR - aOL;
  const double bDy = bOR - bOL;
  const bool useA = std::fabs(aDy) < std::fabs(bDy);
  const double outOrtho = lambda * (useA ? aDy : bDy) +
                          (useL ? (useA ? aOL : bOL) : (useA ? aOR : bOR));
  *out = axis == 0 ? vec2(outProj, outOrtho) : vec2(outOrtho, outProj);
  return std::isfinite(out->x) && std::isfinite(out->y);
}

}  // namespace boolean2
}  // namespace manifold

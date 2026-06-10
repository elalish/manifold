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

namespace {

vec2 SortLeftRight(vec2 p0, vec2 p1, int axis) {
  if (Coord(p1, axis) < Coord(p0, axis)) std::swap(p0, p1);
  return p0;
}

vec2 SortRight(vec2 p0, vec2 p1, int axis) {
  if (Coord(p1, axis) < Coord(p0, axis)) std::swap(p0, p1);
  return p1;
}

bool NearlyEqual(double a, double b, double eps) {
  return std::fabs(a - b) <= eps;
}

bool IsEndpointProjection(vec2 p, double x, int axis, double eps) {
  return NearlyEqual(Coord(p, axis), x, eps);
}

GraphOrderKind StrictOrder(double gap) {
  return gap > 0.0 ? GraphOrderKind::ALessOrtho : GraphOrderKind::AGreaterOrtho;
}

bool OppositeSigns(double a, double b) {
  return (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0);
}

bool StrictlyBetweenWithEndpointBand(double x, double p, double q, double eps) {
  return std::min(p, q) + eps < x && x < std::max(p, q) - eps;
}

bool AwayFromEndpoints(vec2 p, vec2 a, vec2 b, double eps) {
  const double eps2 = eps * eps;
  return dot(p - a, p - a) > eps2 && dot(p - b, p - b) > eps2;
}

int CompareDouble(double a, double b) { return (a > b) - (a < b); }

int ComparePointKey(vec2 a, vec2 b) {
  if (int c = CompareDouble(a.x, b.x)) return c;
  return CompareDouble(a.y, b.y);
}

int CompareSegmentKey(GraphSegment2D a, GraphSegment2D b) {
  if (ComparePointKey(a.p1, a.p0) < 0) std::swap(a.p0, a.p1);
  if (ComparePointKey(b.p1, b.p0) < 0) std::swap(b.p0, b.p1);
  if (int c = ComparePointKey(a.p0, b.p0)) return c;
  return ComparePointKey(a.p1, b.p1);
}

GraphOrderKind SymbolicTieOrder(const GraphSegment2D& a,
                                const GraphSegment2D& b) {
  int order = CompareSegmentKey(a, b);
  if (order == 0) {
    DEBUG_ASSERT(a.stableEdgeId != b.stableEdgeId, logicErr,
                 "Boolean2 graph order has unresolved symbolic tie");
    order = a.stableEdgeId < b.stableEdgeId ? -1 : 1;
  }
  return order < 0 ? GraphOrderKind::ALessOrtho : GraphOrderKind::AGreaterOrtho;
}

}  // namespace

// Centered-shoelace signed area of a closed polygon loop. Same FP trick
// as the per-face area computation in winding_filter.h: subtract a
// reference vert before multiplying so products stay at edge-length scale.
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

// Choose epsilon from the operation scale using Smith's rounded power-of-two
// length bound and the caller's adjustment budget.
double EpsilonFromScale(double L, int k_budget) {
  if (L <= 0) return 0;
  int expBits;
  std::frexp(L, &expBits);
  return std::ldexp((k_budget + 1) * kAlphaCoeff * kU, expBits);
}

double Coord(vec2 p, int axis) { return axis == 0 ? p.x : p.y; }

GraphOrder2D CompareProjectedOrder(const GraphSegment2D& a,
                                   const GraphSegment2D& b, int axis,
                                   double overlapL, double overlapR,
                                   double eps) {
  GraphOrder2D order;
  DEBUG_ASSERT(axis == 0 || axis == 1, logicErr,
               "Boolean2 graph order requires a 2D projection axis");
  if (overlapR <= overlapL) return order;

  const vec2 aL = SortLeftRight(a.p0, a.p1, axis);
  const vec2 aR = SortRight(a.p0, a.p1, axis);
  const vec2 bL = SortLeftRight(b.p0, b.p1, axis);
  const vec2 bR = SortRight(b.p0, b.p1, axis);
  DEBUG_ASSERT(
      Coord(aL, axis) < Coord(aR, axis) && Coord(bL, axis) < Coord(bR, axis),
      logicErr, "Boolean2 graph order requires nonzero axis spread");
  DEBUG_ASSERT(Coord(aL, axis) <= overlapL && overlapR <= Coord(aR, axis) &&
                   Coord(bL, axis) <= overlapL && overlapR <= Coord(bR, axis),
               logicErr, "Boolean2 graph order interval outside domain");
  const auto embed = [&](vec2 p) {
    return axis == 0 ? vec3(p.x, p.y, 0.0) : vec3(p.y, p.x, 0.0);
  };

  const vec3 aL3 = embed(aL), aR3 = embed(aR);
  const vec3 bL3 = embed(bL), bR3 = embed(bR);
  const double aOL = manifold::Interpolate(aL3, aR3, overlapL).x;
  const double aOR = manifold::Interpolate(aL3, aR3, overlapR).x;
  const double bOL = manifold::Interpolate(bL3, bR3, overlapL).x;
  const double bOR = manifold::Interpolate(bL3, bR3, overlapR).x;
  const double gapL = bOL - aOL;
  const double gapR = bOR - aOR;
  const bool tieL = NearlyEqual(gapL, 0.0, eps);
  const bool tieR = NearlyEqual(gapR, 0.0, eps);
  const bool signChanging = OppositeSigns(gapL, gapR);

  if (tieL && tieR && !signChanging) {
    order.coincidentOverlap = true;
    order.atMinProjection = SymbolicTieOrder(a, b);
    order.atMaxProjection = order.atMinProjection;
    return order;
  }

  const bool endpointL = IsEndpointProjection(aL, overlapL, axis, eps) ||
                         IsEndpointProjection(aR, overlapL, axis, eps) ||
                         IsEndpointProjection(bL, overlapL, axis, eps) ||
                         IsEndpointProjection(bR, overlapL, axis, eps);
  const bool endpointR = IsEndpointProjection(aL, overlapR, axis, eps) ||
                         IsEndpointProjection(aR, overlapR, axis, eps) ||
                         IsEndpointProjection(bL, overlapR, axis, eps) ||
                         IsEndpointProjection(bR, overlapR, axis, eps);
  const bool symbolicTieL = tieL && !signChanging;
  const bool symbolicTieR = tieR && !signChanging;

  order.atMinProjection =
      symbolicTieL
          ? (endpointL ? GraphOrderKind::EndpointTouch : SymbolicTieOrder(a, b))
          : StrictOrder(gapL);
  order.atMaxProjection =
      symbolicTieR
          ? (endpointR ? GraphOrderKind::EndpointTouch : SymbolicTieOrder(a, b))
          : StrictOrder(gapR);

  order.properCrossing =
      (order.atMinProjection == GraphOrderKind::ALessOrtho &&
       order.atMaxProjection == GraphOrderKind::AGreaterOrtho) ||
      (order.atMinProjection == GraphOrderKind::AGreaterOrtho &&
       order.atMaxProjection == GraphOrderKind::ALessOrtho);
  return order;
}

bool IntersectSegments(const GraphSegment2D& a, const GraphSegment2D& b,
                       double eps, vec2* out) {
  const vec2 a0 = a.p0, a1 = a.p1, b0 = b.p0, b1 = b.p1;
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
  // (zero-width overlap interval). Compute only strict interior crossings
  // directly; endpoint/near-line sliver contacts are degeneracies for the
  // vertex-on-edge phase.
  if (xUsable == 0 && yUsable == 0) {
    const bool aHoriz = aSpreadX > 0 && aSpreadY == 0;
    const bool aVert = aSpreadY > 0 && aSpreadX == 0;
    const bool bHoriz = bSpreadX > 0 && bSpreadY == 0;
    const bool bVert = bSpreadY > 0 && bSpreadX == 0;
    if ((aHoriz && bVert) || (aVert && bHoriz)) {
      const double ix = aVert ? a0.x : b0.x;
      const double iy = aHoriz ? a0.y : b0.y;
      if (!StrictlyBetweenWithEndpointBand(ix, a0.x, a1.x, eps) &&
          !StrictlyBetweenWithEndpointBand(iy, a0.y, a1.y, eps)) {
        return false;
      }
      if (!StrictlyBetweenWithEndpointBand(ix, b0.x, b1.x, eps) &&
          !StrictlyBetweenWithEndpointBand(iy, b0.y, b1.y, eps)) {
        return false;
      }
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

  // Axis-overlap interval. A proper crossing needs a positive-width shared
  // projection interval; endpoint-only contact is a degeneracy.
  const double overlapL = std::max(Coord(aL, axis), Coord(bL, axis));
  const double overlapR = std::min(Coord(aR, axis), Coord(bR, axis));
  if (overlapR <= overlapL) return false;

  const GraphOrder2D order =
      CompareProjectedOrder(a, b, axis, overlapL, overlapR, eps);
  if (!order.properCrossing) return false;

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

  const vec4 xyzz =
      manifold::Intersect(vec3(overlapL, aOL, 0.0), vec3(overlapR, aOR, 0.0),
                          vec3(overlapL, bOL, 0.0), vec3(overlapR, bOR, 0.0));
  *out = axis == 0 ? vec2(xyzz.x, xyzz.y) : vec2(xyzz.y, xyzz.x);
  return std::isfinite(out->x) && std::isfinite(out->y) &&
         AwayFromEndpoints(*out, a0, a1, eps) &&
         AwayFromEndpoints(*out, b0, b1, eps);
}

}  // namespace manifold

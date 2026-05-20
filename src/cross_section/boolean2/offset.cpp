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
// Polygon offset (a.k.a. inflate/inset, Clipper2's `InflatePaths`
// analog). Backs `CrossSection::Offset`.
//
// Algorithm:
//   1. For each input contour, walk vertex by vertex emitting one offset
//      polygon ring. Each edge contributes a parallel segment at distance
//      |delta| along its outward normal (for CCW outer; flipped sign for
//      CW hole). Each corner contributes a join cap (Square / Round /
//      Miter / Bevel) on the convex side, or a single miter point on the
//      concave side.
//   2. Fill all offset rings to resolve self-intersections introduced by
//      sharp concavities or large |delta|. Positive offsets use additive
//      winding; negative offsets use NonZero with a negative-winding fallback
//      for collapsed insets.
//
// Corner convention (per Clipper2):
//   - "Outward" is to the right of the directed edge for a CCW contour
//     (interior of a CCW outer ring is to the left of edges).
//   - Positive delta inflates; negative shrinks.
//   - Holes (CW rings) get the sign flipped because their "outward"
//     points the opposite direction in 2D.
//
// Join styles:
//   - Round:  arc centered at vertex, radius |delta|. Subdivided so each
//             chord's perpendicular error <= arc_tol. Matches Clipper2's
//             arc-tolerance semantics.
//   - Miter:  intersect the two adjacent offset edges; cap at
//             miter_limit * |delta| from the vertex by falling back to a
//             square chord.
//   - Bevel:  single straight chord between the two offset edge
//             endpoints, no extension. Always works.
//   - Square: like miter but pre-emptively capped at distance |delta|
//             from the vertex with a chord perpendicular to the
//             bisector.

#include "offset.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "boolean2.h"
#include "predicates.h"

namespace manifold {
namespace boolean2 {

namespace offset_detail {

constexpr double kStraightAngleSinTol = 1e-12;
// denom = 1 + dot(normals) = 2*cos^2(half-angle). Below this, the miter
// extension is at least ~1.4e6 * |delta|, so treat the corner as a reversal.
constexpr double kNearOppositeNormalsDenomTol = 1e-12;
constexpr double kMiterLimitDotTolUlp = 64.0;

double Perimeter(const SimplePolygon& loop) {
  if (loop.size() < 2) return 0.0;
  double total = 0.0;
  for (size_t i = 0; i < loop.size(); ++i) {
    total += length(loop[(i + 1) % loop.size()] - loop[i]);
  }
  return total;
}

double TotalPerimeter(const Polygons& polys) {
  double total = 0.0;
  for (const auto& loop : polys) total += Perimeter(loop);
  return total;
}

double AreaComparisonTol(const Polygons& a, const Polygons& b, double eps) {
  // If boundary vertices drift by O(eps), the induced first-order area change
  // is bounded by perimeter * eps. Use both operands' perimeters because this
  // tolerance compares two already-regularized polygon sets.
  return eps * (TotalPerimeter(a) + TotalPerimeter(b)) + eps * eps;
}

// Outward normal of a directed edge (right-perpendicular, unit length).
// For a CCW polygon, this points away from the interior.
vec2 OutwardNormal(vec2 edge) {
  const double len = std::sqrt(edge.x * edge.x + edge.y * edge.y);
  if (len == 0) return vec2(0, 0);
  return vec2(edge.y / len, -edge.x / len);
}

vec2 RotateDegrees(vec2 v, double angle) {
  const double c = cosd(angle);
  const double s = sind(angle);
  return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

double Cross(vec2 a, vec2 b) { return a.x * b.y - a.y * b.x; }

// Number of chords in a full circle at radius `r` such that each chord's
// perpendicular sagitta error stays <= arcTol. Since sagitta decreases
// monotonically with more chords, a binary search against deterministic cosd
// gives a stable minimal integer count.
int FullCircleChordCount(double r, double arcTol) {
  if (arcTol <= 0 || r <= 0) return 1;
  if (arcTol >= 2.0 * r) return 1;
  auto withinTol = [&](int n) { return (1.0 - cosd(180.0 / n)) * r <= arcTol; };
  int hi = 2;
  while (!withinTol(hi) && hi < (1 << 30)) hi *= 2;
  int lo = 1;
  while (lo + 1 < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (withinTol(mid)) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return hi;
}

bool BeforeSweepTarget(vec2 dir, vec2 target, double rotSign) {
  const double cross = Cross(dir, target);
  return rotSign > 0 ? cross > 0 : cross < 0;
}

// Append a round (arc) join between the offset endpoints `endPrev`
// (start of arc) and `startNext` (end of arc), centered at `V` with
// radius |delta|. Emits deterministic full-circle chord steps until the
// next step would reach or pass startNext; caller appends startNext.
void AppendRoundJoin(SimplePolygon& out, vec2 V, vec2 nPrev, vec2 nNext,
                     double delta, double arcTol) {
  // The arc sweeps the short side between nPrev and nNext on a circle
  // of radius |delta| around V. Caller invokes only on the convex
  // side, where this short side is < pi. The step direction is sign(delta)
  // so inset sweeps the mirror direction.
  const double absDelta = std::fabs(delta);
  const int fullCircleCount = FullCircleChordCount(absDelta, arcTol);
  const double step = 360.0 / fullCircleCount;
  const double rotSign = (delta >= 0) ? 1.0 : -1.0;
  for (int i = 1; i < fullCircleCount; ++i) {
    const vec2 dir = RotateDegrees(nPrev, rotSign * i * step);
    if (!BeforeSweepTarget(dir, nNext, rotSign)) break;
    out.push_back(V + delta * dir);
  }
}

// Bevel join: single chord, no extra vertices between endPrev and
// startNext. Caller is responsible for pushing those endpoints.
void AppendBevelJoin(SimplePolygon& /*out*/, vec2 /*V*/, vec2 /*nPrev*/,
                     vec2 /*nNext*/, double /*delta*/) {
  // intentionally empty: the chord is implicit in pushing
  // endPrev and startNext consecutively.
}

// Square join: two extra vertices forming a chord tangent to a circle
// of radius |delta| around V at the bisector, capping the corner with a
// flat. Matches Clipper2's `DoSquare`. For a corner with half-angle
// alpha (angle between bisector and either normal), the chord half-
// length is |delta| * tan(alpha/2) = |delta| * (1 - cos alpha) /
// sin alpha; using the more FP-stable form sin / (1 + cos).
void AppendSquareJoin(SimplePolygon& out, vec2 V, vec2 nPrev, vec2 nNext,
                      double delta) {
  vec2 bis(nPrev.x + nNext.x, nPrev.y + nNext.y);
  const double bisLen = std::sqrt(bis.x * bis.x + bis.y * bis.y);
  if (bisLen == 0) return;  // 180-deg reversal; nothing reasonable to emit
  bis.x /= bisLen;
  bis.y /= bisLen;
  const vec2 tang(-bis.y, bis.x);
  const double cosHalf = bis.x * nPrev.x + bis.y * nPrev.y;
  if (cosHalf <= 0) return;  // reflex; convex caller should not invoke this
  const double sinHalf = std::sqrt(std::max(0.0, 1.0 - cosHalf * cosHalf));
  const double half = std::fabs(delta) * sinHalf / (1.0 + cosHalf);
  const double absDelta = std::fabs(delta);
  const vec2 mid = V + absDelta * bis;
  out.push_back(mid - half * tang);
  out.push_back(mid + half * tang);
}

// Intersect the two offset edges (prev edge's offset line and next edge's
// offset line) at the vertex V. Returns the miter point. Falls back to V
// + delta * average-normal if lines are parallel.
vec2 MiterPoint(vec2 V, vec2 nPrev, vec2 nNext, double delta) {
  // The two offset lines pass through V + delta*nPrev (perp to ePrev)
  // and V + delta*nNext (perp to eNext). Their intersection lies along
  // the bisector at distance delta / cos(half-angle).
  const double dotN = nPrev.x * nNext.x + nPrev.y * nNext.y;
  const double denom = 1.0 + dotN;
  if (denom <= kNearOppositeNormalsDenomTol) {
    // Nearly opposite normals (sharp ~180-degree corner); miter is
    // unbounded. Caller should detect and fall back.
    return V + delta * nPrev;
  }
  return V +
         delta * vec2((nPrev.x + nNext.x) / denom, (nPrev.y + nNext.y) / denom);
}

// Offset a single input contour. Positive `delta` inflates the solid
// region (outer CCW rings expand outward; inner CW holes shrink as
// their boundary moves into the hole). Negative `delta` does the
// reverse. Orientation handling falls out automatically from
// OutwardNormal's right-of-edge-direction convention:
//   - CCW outer: right-of-edge = outward of solid.
//   - CW hole:   right-of-edge = into the hole = outward of solid.
// So `V + delta * OutwardNormal` always moves the boundary outward of
// the solid for delta > 0, regardless of ring orientation, and the
// convex/concave decision depends only on `cross * sign(delta)`.
SimplePolygon OffsetContour(const SimplePolygon& contour, double delta,
                            JoinType jt, double miterLimit, double arcTol) {
  const int n = static_cast<int>(contour.size());
  if (n < 3 || delta == 0) return contour;
  const double deltaSign = (delta >= 0) ? 1.0 : -1.0;

  SimplePolygon out;
  out.reserve(static_cast<size_t>(n) * 2);
  for (int i = 0; i < n; ++i) {
    const vec2 V = contour[i];
    const vec2 P = contour[(i + n - 1) % n];
    const vec2 N = contour[(i + 1) % n];
    const vec2 ePrev = vec2(V.x - P.x, V.y - P.y);
    const vec2 eNext = vec2(N.x - V.x, N.y - V.y);
    const vec2 nPrev = OutwardNormal(ePrev);
    const vec2 nNext = OutwardNormal(eNext);
    if (nPrev == vec2(0, 0) || nNext == vec2(0, 0)) continue;
    const vec2 endPrev = V + delta * nPrev;
    const vec2 startNext = V + delta * nNext;
    // Solid's convex/concave at this vertex: cross > 0 is a convex
    // outer corner of the solid regardless of ring orientation (CCW
    // outer left-turn = convex outer; CW hole left-turn = hole's
    // concave indent = solid's convex bulge into the hole). Negative
    // delta flips the offset role (a solid-convex corner becomes a
    // shrinking-corner that needs a miter), hence the `* deltaSign`.
    const double cross = ePrev.x * eNext.y - ePrev.y * eNext.x;
    const double convex = cross * deltaSign;
    // Scale-invariant collinearity gate: sin^2(theta) < 1e-24, i.e.
    // angle within ~1e-12 rad of straight. `cross` magnitude is
    // O(|ePrev| * |eNext| * sin(theta)) so the natural unitless gate is
    // cross^2 < tol^2 * |ePrev|^2 * |eNext|^2. Using a raw fabs(cross)
    // < 1e-12 threshold here would scale wrong with input magnitude:
    // large-coord inputs (GIS or millimeter-scale CAD parts pre-scaled
    // up) carry FP rounding O(1e-16 * coord^2) in cross, which can
    // exceed any absolute threshold even when the corner is exactly
    // collinear, leaving spurious mid-corner vertices in the output
    // until RemoveCollinear cleans them up downstream.
    const double ePrevLen2 = dot(ePrev, ePrev);
    const double eNextLen2 = dot(eNext, eNext);
    if (cross * cross <
        kStraightAngleSinTol * kStraightAngleSinTol * ePrevLen2 * eNextLen2) {
      // Collinear: nPrev == nNext, endPrev == startNext.
      out.push_back(endPrev);
      continue;
    }
    if (convex < 0) {
      // Concave for this offset direction: the two offset edges cross.
      // Emit a single miter point. The union pass downstream cleans up
      // any self-overlap if the miter pokes through the polygon.
      out.push_back(MiterPoint(V, nPrev, nNext, delta));
      continue;
    }
    // Convex corner: apply join.
    out.push_back(endPrev);
    switch (jt) {
      case JoinType::Round:
        AppendRoundJoin(out, V, nPrev, nNext, delta, arcTol);
        break;
      case JoinType::Miter: {
        // miterLen / |delta| = 1 / cos(half_angle) =
        // sqrt(2 / (1 + dot(nPrev, nNext))). The limit
        // miterLen <= miterLimit * |delta| rearranges to
        // dot >= 2 / miterLimit^2 - 1. Comparing on the dot product
        // directly avoids materialising the miter point in the
        // clamped case (and avoids FP-fragility of MiterPoint when
        // the denominator (1 + dot) approaches zero - exactly the
        // case where we'd clamp anyway).
        const double dotN = nPrev.x * nNext.x + nPrev.y * nNext.y;
        const double miterCosThresh = 2.0 / (miterLimit * miterLimit) - 1.0;
        // Equality is permitted by the miter limit. Regular triangles at the
        // default limit sit exactly on this boundary; allow a small ULP margin
        // so rounded unit normals do not spuriously square the join.
        const double miterTol =
            std::isfinite(miterCosThresh)
                ? kMiterLimitDotTolUlp * kU *
                      std::max(1.0, std::fabs(miterCosThresh))
                : 0.0;
        if (dotN + miterTol < miterCosThresh) {
          AppendSquareJoin(out, V, nPrev, nNext, delta);
        } else {
          out.push_back(MiterPoint(V, nPrev, nNext, delta));
        }
        break;
      }
      case JoinType::Square:
        AppendSquareJoin(out, V, nPrev, nNext, delta);
        break;
      case JoinType::Bevel:
        AppendBevelJoin(out, V, nPrev, nNext, delta);
        break;
    }
    out.push_back(startNext);
  }
  return out;
}

}  // namespace offset_detail

namespace offset_detail {

// Remove vertices that are collinear with their immediate neighbours
// (within tolerance `eps`). Offset output frequently contains such
// vertices because each input edge contributes a separate offset
// segment whose endpoints lie along a straight side. Clipper2's
// `InflatePaths` strips them as part of its finishing pass; replicating
// here keeps `NumVert()` parity for callers comparing vertex counts.
Polygons RemoveCollinear(Polygons polys, double eps) {
  const double eps2 = eps * eps;
  for (auto& loop : polys) {
    if (loop.size() < 3) continue;
    SimplePolygon kept;
    kept.reserve(loop.size());
    const int n = (int)loop.size();
    for (int i = 0; i < n; ++i) {
      const vec2 P = kept.empty() ? loop[(i + n - 1) % n] : kept.back();
      const vec2 V = loop[i];
      // Find first non-degenerate forward neighbour to compare against
      // (so a run of collinear verts collapses to a single vertex).
      vec2 N = loop[(i + 1) % n];
      const vec2 ePrev = vec2(V.x - P.x, V.y - P.y);
      const vec2 eNext = vec2(N.x - V.x, N.y - V.y);
      if (dot(ePrev, ePrev) < eps2) continue;  // zero-length back-edge
      if (dot(eNext, eNext) < eps2) continue;  // zero-length forward edge
      // Perpendicular squared distance from V to line PN; if less than
      // eps^2, V is essentially collinear with P-N and can be dropped.
      const vec2 pn = vec2(N.x - P.x, N.y - P.y);
      const double pnLen2 = dot(pn, pn);
      if (pnLen2 > 0) {
        const double cross = ePrev.x * eNext.y - ePrev.y * eNext.x;
        if (cross * cross < eps2 * pnLen2) continue;
      }
      kept.push_back(V);
    }
    // Wrap-around check: the first kept vertex may be collinear with
    // last-kept and kept[1] now that earlier collapses settled.
    while (kept.size() >= 3) {
      const vec2 P = kept.back();
      const vec2 V = kept.front();
      const vec2 N = kept[1];
      const vec2 ePrev = vec2(V.x - P.x, V.y - P.y);
      const vec2 eNext = vec2(N.x - V.x, N.y - V.y);
      const vec2 pn = vec2(N.x - P.x, N.y - P.y);
      const double pnLen2 = dot(pn, pn);
      const double cross = ePrev.x * eNext.y - ePrev.y * eNext.x;
      if (pnLen2 > 0 && cross * cross < eps2 * pnLen2) {
        kept.erase(kept.begin());
      } else {
        break;
      }
    }
    loop = std::move(kept);
  }
  // Drop sub-3-vertex degenerate rings that the collinear pass collapsed.
  polys.erase(
      std::remove_if(polys.begin(), polys.end(),
                     [](const SimplePolygon& l) { return l.size() < 3; }),
      polys.end());
  return polys;
}

}  // namespace offset_detail

// Public Offset API. `delta` is the offset distance (positive = inflate,
// negative = inset). `miterLimit` is relative to |delta|. `arcTol` is
// the maximum perpendicular chord-error for Round joins; same semantics
// as Clipper2's `arc_tolerance`.
//
// Each input contour produces one offset ring, then all rings are
// unioned via the NonZero fill rule to resolve self-intersections, and
// a final pass strips collinear vertices (matching Clipper2's
// `InflatePaths` finishing behaviour so callers see the same NumVert).
Polygons Offset(const Polygons& in, double delta, JoinType jt,
                double miterLimit, double arcTol) {
  if (delta == 0 || in.empty()) return in;
  // Reject non-finite delta and non-finite input verts. Garbage in
  // produces NaN-laden output and `assert(isfinite(...))` failures
  // downstream in DCEL face area centering; clean rejection is safer.
  // `miterLimit` and `arcTol` are clamped/treated as nonsense-but-OK
  // inside OffsetContour (negative or zero arcTol falls through to
  // GetCircularSegments default; miterLimit < 1 still produces a
  // bounded square fallback).
  if (!std::isfinite(delta)) return {};
  for (const auto& ring : in) {
    for (const auto& v : ring) {
      if (!std::isfinite(v.x) || !std::isfinite(v.y)) return {};
    }
  }
  Polygons offsetRings;
  offsetRings.reserve(in.size());
  for (const auto& ring : in) {
    auto off =
        offset_detail::OffsetContour(ring, delta, jt, miterLimit, arcTol);
    if (off.size() >= 3) offsetRings.push_back(std::move(off));
  }
  if (offsetRings.empty()) return {};
  const double eps = InferEps(offsetRings, {});
  // Resolve self-intersecting offset rings (e.g. when delta exceeds a
  // thin feature's half-width and the offset pinches itself into multiple
  // loops). Positive offsets keep the positive filled side; NonZero would
  // also preserve opposite-winding lobes from inverted concave joins.
  // Negative offsets normally use NonZero so holes and surviving inset
  // islands keep their winding, then fall back only when the inset has
  // impossibly grown in area.
  Polygons unioned;
  if (delta > 0) {
    unioned = FillByRule(offsetRings, WindRule::Add, eps);
    // A true outward offset cannot reduce the filled area. Extreme concave
    // bevel joins can invert large portions of the raw offset ring, causing
    // Add to keep only small positive-winding islands. NonZero preserves the
    // regularized expanded boundary in that case.
    const double inArea = std::fabs(TotalSignedArea(in));
    const double outArea = std::fabs(TotalSignedArea(unioned));
    if (outArea + offset_detail::AreaComparisonTol(in, unioned, eps) < inArea) {
      unioned = FillByRule(offsetRings, WindRule::NonZero, eps);
    }
  } else {
    unioned = FillByRule(offsetRings, WindRule::NonZero, eps);
    // A true inset cannot increase the filled area. If NonZero preserved
    // inverted runaway lobes from a collapsed contour, retry with the
    // opposite winding side, which regularizes the collapse to empty or to
    // the remaining interior islands.
    const double inArea = std::fabs(TotalSignedArea(in));
    const double outArea = std::fabs(TotalSignedArea(unioned));
    if (outArea > inArea + offset_detail::AreaComparisonTol(in, unioned, eps)) {
      unioned = FillByRule(offsetRings, WindRule::Negative, eps);
    }
  }
  return offset_detail::RemoveCollinear(std::move(unioned), eps);
}

}  // namespace boolean2
}  // namespace manifold

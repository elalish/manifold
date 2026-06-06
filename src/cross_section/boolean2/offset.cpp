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

#include <algorithm>
#include <cmath>
#include <vector>

#include "../../utils.h"
#include "boolean2.h"
#include "predicates.h"

// Polygon offset backing `CrossSection::Offset`.
namespace manifold {
namespace boolean2 {

namespace {

// Upper bound on round-join chords per 360 degrees, so a direct caller passing
// an absurd circularSegments cannot blow up the vertex count.
constexpr int kMaxRoundJoinSegments = 1 << 15;

struct ScaledDirection {
  vec2 unit = vec2(0, 0);
  double scale = 0;
};

ScaledDirection UnitFromScaled(vec2 v) {
  const double scale = std::max(std::fabs(v.x), std::fabs(v.y));
  if (scale == 0 || !std::isfinite(scale)) return {};
  const vec2 u = v / scale;
  const double len = la::length(u);
  if (len == 0 || !std::isfinite(len)) return {};
  return {u / len, scale};
}

// Outward normal of a directed edge (right-perpendicular, unit length).
// For a CCW polygon, this points away from the interior.
vec2 OutwardNormal(vec2 edge) {
  const ScaledDirection dir = UnitFromScaled(edge);
  // Keep this exact: public CrossSection inputs are regularized before
  // Offset, and a local eps threshold here can erase valid tiny-edge corners.
  if (dir.scale == 0) return vec2(0, 0);
  return vec2(dir.unit.y, -dir.unit.x);
}

vec2 RotateDegrees(vec2 v, double angle) {
  const double c = cosd(angle);
  const double s = sind(angle);
  return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

// Append a round (arc) join between the offset endpoints, sweeping the short
// (convex) side from nPrev to nNext on a circle of radius |delta| around V.
// Distributes `segments`-per-360 chords evenly across the corner so no short
// final chord is left; the caller appends startNext. Step direction is
// sign(delta) so inset sweeps the mirror direction.
void AppendRoundJoin(SimplePolygon& out, vec2 V, vec2 nPrev, vec2 nNext,
                     double delta, int segments) {
  const double rotSign = (delta >= 0) ? 1.0 : -1.0;
  const double fullStep = 360.0 / segments;
  const double sweep =
      degrees(std::acos(std::clamp(dot(nPrev, nNext), -1.0, 1.0)));
  const int nSub = std::max(1, static_cast<int>(std::ceil(sweep / fullStep)));
  const double subStep = sweep / nSub;
  for (int i = 1; i < nSub; ++i) {
    out.push_back(V + delta * RotateDegrees(nPrev, rotSign * i * subStep));
  }
}

// Square join: two extra vertices forming a chord tangent to a circle
// of radius |delta| around V at the bisector, capping the corner with a
// flat. Matches Clipper2's `DoSquare`. For a corner with half-angle
// alpha (angle between bisector and either normal), the chord half-
// length is |delta| * tan(alpha/2) = |delta| * (1 - cos alpha) /
// sin alpha; using the more FP-stable form sin / (1 + cos).
void AppendSquareJoin(SimplePolygon& out, vec2 V, vec2 nPrev, vec2 nNext,
                      double delta) {
  vec2 bisector(nPrev.x + nNext.x, nPrev.y + nNext.y);
  const ScaledDirection bisectorDir = UnitFromScaled(bisector);
  if (bisectorDir.scale == 0) return;  // 180-deg reversal; nothing reasonable
  bisector = bisectorDir.unit;
  const vec2 tangent(-bisector.y, bisector.x);
  // A convex corner can round cosHalf slightly below 0 as it approaches a
  // 180-degree reversal; clamp to 0 so the cap degrades to its limiting
  // half-width (|delta|) instead of vanishing. Exact reversals are handled by
  // the bisectorDir.scale == 0 guard above.
  const double cosHalf =
      std::max(0.0, bisector.x * nPrev.x + bisector.y * nPrev.y);
  const double sinHalf = std::sqrt(std::max(0.0, 1.0 - cosHalf * cosHalf));
  const double half = std::fabs(delta) * sinHalf / (1.0 + cosHalf);
  // Signed delta: the cap follows the offset side (outward for delta > 0,
  // inward for inset). Using std::fabs here would mirror the cap to the wrong
  // side for inset, which the Positive union cannot recover.
  const vec2 mid = V + delta * bisector;
  out.push_back(mid - half * tangent);
  out.push_back(mid + half * tangent);
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
  if (denom <= 0) {
    // Opposite normals make the miter unbounded. The caller's miter-limit
    // check handles near-opposite normals before this point.
    return V + delta * nPrev;
  }
  return V +
         delta * vec2((nPrev.x + nNext.x) / denom, (nPrev.y + nNext.y) / denom);
}

double ValidMiterLimit(double miterLimit) {
  return std::isfinite(miterLimit) && miterLimit >= 2.0 ? miterLimit : 2.0;
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
                            OffsetJoinType jt, double miterLimit,
                            int segments) {
  const int n = static_cast<int>(contour.size());
  if (n < 3 || delta == 0) return contour;
  const double deltaSign = (delta >= 0) ? 1.0 : -1.0;
  miterLimit = ValidMiterLimit(miterLimit);

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
    const double cross = la::cross(ePrev, eNext);
    // Collinearity reuses the canonical CCW predicate, with a scale-derived
    // tolerance from the larger adjacent edge.
    const double eps = EpsilonFromScale(
        std::sqrt(std::max(dot(ePrev, ePrev), dot(eNext, eNext))));
    if (CCW(P, V, N, eps) == 0) {
      // Zero cross is either a straight continuation (dot >= 0: nPrev == nNext,
      // so endPrev == startNext - one point suffices) or an antiparallel
      // reversal (dot < 0: nPrev == -nNext, so the offset endpoints are
      // distinct and on opposite sides). Emit both for the reversal to bevel
      // across the spike instead of collapsing it to a single point.
      out.push_back(endPrev);
      if (dot(ePrev, eNext) < 0) out.push_back(startNext);
      continue;
    }
    const bool convex = cross * deltaSign > 0;
    if (!convex) {
      // Concave joins intentionally create a self-intersection that the final
      // Positive union resolves, keeping the correct corner fill (matching
      // Clipper2's offset cleanup). Keep V so the union resolves on the right
      // side; a direct endPrev->startNext bevel would cut the corner the wrong
      // way and lose area.
      out.push_back(endPrev);
      out.push_back(V);
      out.push_back(startNext);
      continue;
    }
    // Convex corner: apply join.
    out.push_back(endPrev);
    switch (jt) {
      case OffsetJoinType::Round:
        AppendRoundJoin(out, V, nPrev, nNext, delta, segments);
        break;
      case OffsetJoinType::Miter: {
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
        // Equality is allowed by the miter limit. `dotN` comes from rounded
        // unit normals, so use only the baseline unit-scale predicate epsilon
        // to avoid squaring a corner that is exactly on the limit.
        const double miterTieTol = EpsilonFromScale(1.0, /*k_budget=*/0);
        // Near-opposite normals make MiterPoint unbounded (it scales as
        // 1/(1 + dotN)), and the miterLimit gate stops bounding it once
        // 2/miterLimit^2 underflows miterTieTol (i.e. for very large
        // miterLimit). Independently square such degenerate-sharp corners so
        // the emitted coordinate stays bounded by sqrt(2 / kMinMiterDenom) *
        // |delta|; 2e-12 caps it at ~1e6 * |delta|, far beyond any meaningful
        // miter, so honest finite limits are unaffected.
        constexpr double kMinMiterDenom = 2e-12;
        if (dotN + miterTieTol < miterCosThresh ||
            1.0 + dotN < kMinMiterDenom) {
          AppendSquareJoin(out, V, nPrev, nNext, delta);
        } else {
          out.push_back(MiterPoint(V, nPrev, nNext, delta));
        }
        break;
      }
      case OffsetJoinType::Square:
        AppendSquareJoin(out, V, nPrev, nNext, delta);
        break;
      case OffsetJoinType::Bevel:
        break;
    }
    out.push_back(startNext);
  }
  return out;
}

}  // namespace

// Public Offset API. `delta` is the offset distance (positive = inflate,
// negative = inset). `miterLimit` is relative to |delta|. `circularSegments`
// is the chord count per 360 degrees of Round join; values < 3 use the Quality
// default for radius |delta|.
//
// Each input contour produces one offset ring, then all rings are regularized
// with Positive/Add filling. Collinear vertices are left in place for the
// caller to `Simplify` if wanted, matching Manifold's tolerance model rather
// than Clipper2's finishing behaviour.
Polygons Offset(const Polygons& in, double delta, OffsetJoinType jt,
                double miterLimit, int circularSegments, double tolerance) {
  // Reject NaN/Inf delta and input coordinates.
  if (!std::isfinite(delta)) return {};
  for (const auto& ring : in) {
    for (const auto& v : ring) {
      if (!std::isfinite(v.x) || !std::isfinite(v.y)) return {};
    }
  }
  if (delta == 0 || in.empty()) return in;
  // Resolve the round-join segment count once: a public default (< 3) maps to
  // the Quality default for radius |delta|, and the result is clamped so a
  // direct caller cannot blow up the vertex count.
  int segments = circularSegments >= 3
                     ? circularSegments
                     : Quality::GetCircularSegments(std::fabs(delta));
  segments = std::min(std::max(segments, 3), kMaxRoundJoinSegments);
  Polygons offsetRings;
  offsetRings.reserve(in.size());
  for (const auto& ring : in) {
    auto off = OffsetContour(ring, delta, jt, miterLimit, segments);
    if (off.size() >= 3) offsetRings.push_back(std::move(off));
  }
  if (offsetRings.empty()) return {};
  const double eps = InferEps(offsetRings, {});
  // Resolve self-intersecting offset rings (e.g. when delta exceeds a
  // thin feature's half-width and the offset pinches itself into multiple
  // loops). CrossSection storage is normal-oriented before reaching Offset,
  // so Positive/Add cleanup keeps the filled side for both dilation and inset.
  return Simplify(offsetRings, eps, tolerance);
}

}  // namespace boolean2
}  // namespace manifold

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

#include "manifold/cross_section.h"

#include <gtest/gtest.h>

#include <cmath>
#ifdef MANIFOLD_DEBUG
#include <fstream>
#include <iomanip>
#include <ostream>
#endif
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include "../src/boolean2.h"
#include "manifold/common.h"
#include "manifold/manifold.h"
#include "test.h"

using namespace manifold;

namespace {

SimplePolygon StarRing(const std::vector<double>& radii) {
  SimplePolygon ring;
  ring.reserve(radii.size());
  const int n = static_cast<int>(radii.size());
  for (int i = 0; i < n; ++i) {
    const double r = 0.1 + std::fabs(radii[i]);
    const double theta = 2.0 * kPi * i / n;
    ring.push_back({r * std::cos(theta), r * std::sin(theta)});
  }
  return ring;
}

// Expand run-length radius specs for seeds with long repeated-radius runs:
// Runs({{3, 1000.}, {21, 332.47}}) is 3x 1000. then 21x 332.47.
std::vector<double> Runs(std::initializer_list<std::pair<int, double>> runs) {
  std::vector<double> out;
  for (const auto& run : runs) out.insert(out.end(), run.first, run.second);
  return out;
}

double AreaTol(const CrossSection& a, const CrossSection& b) {
  return 1e-6 * (1.0 + std::fabs(a.Area()) + std::fabs(b.Area()));
}

double AreaTol(const CrossSection& a, const CrossSection& b,
               const CrossSection& c) {
  return 1e-6 * (1.0 + std::fabs(a.Area()) + std::fabs(b.Area()) +
                 std::fabs(c.Area()));
}

void ExpectBooleanInvariants(const CrossSection& a, const CrossSection& b) {
  const auto aIb = a.Boolean(b, OpType::Intersect);
  const double tol = AreaTol(a, b);
  EXPECT_NEAR((a - b).Area() + aIb.Area(), a.Area(), tol)
      << "area(A - B) + area(A ∩ B) != area(A)";
  EXPECT_NEAR((b - a).Area() + aIb.Area(), b.Area(), tol)
      << "area(B - A) + area(A ∩ B) != area(B)";
  EXPECT_NEAR((a + b).Area(), a.Area() + b.Area() - aIb.Area(), tol)
      << "inclusion-exclusion violated";
}

void ExpectDistributesOverUnion(const CrossSection& a, const CrossSection& b,
                                const CrossSection& c) {
  const auto bUc = b + c;
  const auto left = a.Boolean(bUc, OpType::Intersect);
  const auto right =
      a.Boolean(b, OpType::Intersect) + a.Boolean(c, OpType::Intersect);
  const double tol = AreaTol(a, b, c);
  EXPECT_NEAR(left.Area(), right.Area(), tol)
      << "A ∩ (B ∪ C) != (A ∩ B) ∪ (A ∩ C)";
  EXPECT_NEAR((left - right).Area(), 0.0, tol)
      << "distributivity: left-right difference is non-empty";
  EXPECT_NEAR((right - left).Area(), 0.0, tol)
      << "distributivity: right-left difference is non-empty";
}

// Shared builder for the consolidated star-seed regression tables below
// (BooleanDistributivity/SubtractInvariants/BooleanCommutativity/
// BooleanAssociativity *Seeds). Each fuzz seed is a StarRing optionally
// shifted by a translation; rows carry the raw radii + translate verbatim.
struct Shape {
  std::vector<double> radii;
  vec2 translate{0, 0};
};

CrossSection MakeShape(const Shape& s) {
  CrossSection cs(StarRing(s.radii));
  if (s.translate != vec2{0, 0}) cs = cs.Translate(s.translate);
  return cs;
}

}  // namespace

TEST(CrossSection, Square) {
  auto a = Manifold::Cube({5, 5, 5});
  auto b = Manifold::Extrude(CrossSection::Square({5, 5}).ToPolygons(), 5);

  EXPECT_FLOAT_EQ((a - b).Volume(), 0.);
}

TEST(CrossSection, MirrorUnion) {
  auto a = CrossSection::Square({5., 5.}, true);
  auto b = a.Translate({2.5, 2.5});
  auto cross = a + b + b.Mirror({1, 1});
  auto result = Manifold::Extrude(cross.ToPolygons(), 5.);

  if (options.exportModels)
    WriteTestOBJ("cross_section_mirror_union.obj", result);

  EXPECT_FLOAT_EQ(2.5 * a.Area(), cross.Area());
  EXPECT_TRUE(a.Mirror(vec2(0.0)).IsEmpty());
}

TEST(CrossSection, MirrorCheckAxis) {
  auto tri = CrossSection({{0., 0.}, {5., 5.}, {0., 10.}});

  auto a = tri.Mirror({1., 1.}).Bounds();
  auto a_expected = CrossSection({{0., 0.}, {-10., 0.}, {-5., -5.}}).Bounds();

  EXPECT_NEAR(a.min.x, a_expected.min.x, 0.001);
  EXPECT_NEAR(a.min.y, a_expected.min.y, 0.001);
  EXPECT_NEAR(a.max.x, a_expected.max.x, 0.001);
  EXPECT_NEAR(a.max.y, a_expected.max.y, 0.001);

  auto b = tri.Mirror({-1., 1.}).Bounds();
  auto b_expected = CrossSection({{0., 0.}, {10., 0.}, {5., 5.}}).Bounds();

  EXPECT_NEAR(b.min.x, b_expected.min.x, 0.001);
  EXPECT_NEAR(b.min.y, b_expected.min.y, 0.001);
  EXPECT_NEAR(b.max.x, b_expected.max.x, 0.001);
  EXPECT_NEAR(b.max.y, b_expected.max.y, 0.001);
}

TEST(CrossSection, RoundOffset) {
  auto a = CrossSection::Square({20., 20.}, true);
  int segments = 20;
  auto rounded = a.Offset(5., CrossSection::JoinType::Round, 2, segments);
  auto result = Manifold::Extrude(rounded.ToPolygons(), 5.);

  if (options.exportModels)
    WriteTestOBJ("cross_section_round_offset.obj", result);

  EXPECT_EQ(result.Genus(), 0);
  EXPECT_NEAR(result.Volume(), 4386, 1);
  EXPECT_EQ(rounded.NumVert(), segments + 4);
}

TEST(CrossSection, BevelOffset) {
  auto a = CrossSection::Square({20., 20.}, true);
  int segments = 20;
  auto rounded = a.Offset(5., CrossSection::JoinType::Bevel, 2, segments);
  auto result = Manifold::Extrude(rounded.ToPolygons(), 5.);

  if (options.exportModels)
    WriteTestOBJ("cross_section_bevel_offset.obj", result);

  EXPECT_EQ(result.Genus(), 0);
  EXPECT_NEAR(result.Volume(),
              5 * (((20. + (2 * 5.)) * (20. + (2 * 5.))) - (2 * 5. * 5)), 1);
  EXPECT_EQ(rounded.NumVert(), 4 + 4);
}

TEST(CrossSection, MiterOffset) {
  auto square = CrossSection::Square({20., 20.}, true);
  auto offset = square.Offset(5., CrossSection::JoinType::Miter);
  auto result = Manifold::Extrude(offset.ToPolygons(), 1.);

  EXPECT_EQ(result.Genus(), 0);
  EXPECT_NEAR(result.Volume(), 30. * 30., 0.01);
  // Offset keeps collinear verts on the miter edges; stripping them is
  // Simplify's job, so check the simplified corner count (4)
  // backend-agnostically.
  EXPECT_EQ(offset.Simplify().NumVert(), 4);
}

TEST(CrossSection, OffsetWithHole) {
  SimplePolygon outer = {{-10, -10}, {10, -10}, {10, 10}, {-10, 10}};
  SimplePolygon hole = {{-2, -2}, {-2, 2}, {2, 2}, {2, -2}};
  // CCW outer + CW hole are already oriented for Positive fill, which yields
  // the same annulus as EvenOdd here and is the only rule CrossSection
  // supports.
  CrossSection cs({outer, hole}, CrossSection::FillRule::Positive);

  auto inflated = cs.Offset(1., CrossSection::JoinType::Miter);
  EXPECT_NEAR(inflated.Area(), 484. - 4., 0.01);

  auto inset = cs.Offset(-1., CrossSection::JoinType::Miter);
  EXPECT_NEAR(inset.Area(), 324. - 36., 0.01);
}

TEST(CrossSection, OffsetThinPinch) {
  CrossSection thin = CrossSection::Square({100., 2.}, true);
  auto inset = thin.Offset(-1., CrossSection::JoinType::Miter);

  EXPECT_TRUE(inset.IsEmpty() || inset.Area() < 0.01);
}

TEST(CrossSection, OffsetMultipleDisjointOuters) {
  SimplePolygon a = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  SimplePolygon b = {{10, 0}, {11, 0}, {11, 1}, {10, 1}};
  CrossSection cs({a, b});

  auto separated = cs.Offset(0.5, CrossSection::JoinType::Miter);
  EXPECT_EQ(separated.NumContour(), 2);
  EXPECT_NEAR(separated.Area(), 8., 0.01);

  auto merged = cs.Offset(5., CrossSection::JoinType::Miter);
  EXPECT_EQ(merged.NumContour(), 1);
}

TEST(CrossSection, FourConcurrentEdges) {
  auto rhomb = [](double angleDegrees) {
    const double angle = angleDegrees * kPi / 180.;
    const double cosA = std::cos(angle);
    const double sinA = std::sin(angle);
    constexpr double radius = 1.0;
    constexpr double width = 0.1;
    return SimplePolygon{{radius * cosA, radius * sinA},
                         {-width * sinA, width * cosA},
                         {-radius * cosA, -radius * sinA},
                         {width * sinA, -width * cosA}};
  };

  Polygons polys;
  for (double angle : {0., 45., 90., 135.}) polys.push_back(rhomb(angle));
  CrossSection cs(polys, CrossSection::FillRule::Positive);

  EXPECT_EQ(cs.NumContour(), 1);
  EXPECT_NEAR(cs.Area(), 0.644423, 1e-4);
  for (const auto& loop : cs.ToPolygons()) {
    for (const vec2& v : loop) {
      EXPECT_TRUE(std::isfinite(v.x));
      EXPECT_TRUE(std::isfinite(v.y));
      EXPECT_LE(la::length(v), 1.05);
    }
  }
}

TEST(CrossSection, ConcurrentIndependentEdgePairs) {
  auto rhomb = [](double angleDegrees) {
    const double angle = angleDegrees * kPi / 180.;
    const double cosA = std::cos(angle);
    const double sinA = std::sin(angle);
    constexpr double radius = 1.0;
    constexpr double width = 0.08;
    return SimplePolygon{{radius * cosA, radius * sinA},
                         {-width * sinA, width * cosA},
                         {-radius * cosA, -radius * sinA},
                         {width * sinA, -width * cosA}};
  };

  Polygons polys;
  for (double angle : {0., 30., 90., 120.}) polys.push_back(rhomb(angle));
  CrossSection cs(polys, CrossSection::FillRule::Positive);

  EXPECT_EQ(cs.NumContour(), 1);
  EXPECT_NEAR(cs.Area(), 0.527482, 1e-4);
  for (const auto& loop : cs.ToPolygons()) {
    for (const vec2& v : loop) {
      EXPECT_TRUE(std::isfinite(v.x));
      EXPECT_TRUE(std::isfinite(v.y));
      EXPECT_LE(la::length(v), 1.05);
    }
  }
}

TEST(CrossSection, TranslatedShallowConcurrentEdges) {
  auto rhomb = [](double angleDegrees, vec2 offset) {
    const double angle = angleDegrees * kPi / 180.;
    const double cosA = std::cos(angle);
    const double sinA = std::sin(angle);
    constexpr double radius = 1.0;
    constexpr double width = 0.08;
    return SimplePolygon{{offset.x + radius * cosA, offset.y + radius * sinA},
                         {offset.x - width * sinA, offset.y + width * cosA},
                         {offset.x - radius * cosA, offset.y - radius * sinA},
                         {offset.x + width * sinA, offset.y - width * cosA}};
  };

  auto polysAt = [&](vec2 offset) {
    Polygons polys;
    for (double angle : {0., 6., 90., 96.})
      polys.push_back(rhomb(angle, offset));
    return polys;
  };

  const double base = std::ldexp(1.0, 40);
  CrossSection origin(polysAt({0., 0.}), CrossSection::FillRule::Positive);
  CrossSection shifted(polysAt({base, -base}),
                       CrossSection::FillRule::Positive);
  CrossSection shiftedBack = shifted.Translate({-base, base});

  EXPECT_EQ(origin.NumContour(), 1);
  EXPECT_EQ(shifted.NumContour(), 1);
  EXPECT_EQ(shiftedBack.NumContour(), origin.NumContour());
  EXPECT_NEAR(shiftedBack.Area(), origin.Area(), 1e-4);
  EXPECT_NEAR(shiftedBack.Bounds().Size().x, origin.Bounds().Size().x, 1e-4);
  EXPECT_NEAR(shiftedBack.Bounds().Size().y, origin.Bounds().Size().y, 1e-4);
}

TEST(CrossSection, TranslatedSmallPolygonKeepsFeatures) {
  const double base = std::ldexp(1.0, 49) * 1.5;
  SimplePolygon square = {{base, -base},
                          {base + 10.0, -base},
                          {base + 10.0, -base + 10.0},
                          {base, -base + 10.0}};

  CrossSection cs(square);

  EXPECT_EQ(cs.NumContour(), 1);
  EXPECT_EQ(cs.NumVert(), 4);
  const vec2 size = cs.Bounds().Size();
  EXPECT_NEAR(size.x, 10.0, 1e-9);
  EXPECT_NEAR(size.y, 10.0, 1e-9);
}

// Regression test for the BR-cell hole pattern from Samples.Sponge4. Two
// CCW polygons that share an endpoint AND form a T-junction at the
// non-shared endpoint of one edge. Before the broad-phase / fused-pass
// shared-endpoint filter fix, the (long, short) edge pair was dropped at
// the broad phase, so the narrow phase never inserted the T-junction
// vertex on the long edge. Canonical sub-edges came out with the wrong
// multiplicities, the DCEL face traversal merged faces of different
// windings, and small CW holes were silently dropped from the output.
TEST(CrossSection, TJunctionAtSharedEndpoint) {
  // Outer CCW unit square plus a smaller CCW square sharing the (0,0)
  // corner. The outer's bottom edge (0,0)->(1,0) has the inner's
  // (0.5,0) vertex on its interior; both share endpoint (0,0).
  SimplePolygon outer = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
  SimplePolygon inner = {{0.0, 0.0}, {0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5}};

  CrossSection cs(Polygons{outer, inner});
  // Union with Positive fill rule: the inner CCW square is fully inside
  // the outer, so the result is just the outer's area.
  EXPECT_EQ(cs.NumContour(), 1);
  EXPECT_NEAR(cs.Area(), 1.0, 1e-9);

  // A more demanding case from the same family: a butterfly polygon with
  // cancel-pair retraces (one CW lobe around a unit cell) combined with
  // an adjacent CCW staircase polygon whose edge runs along the cell's
  // boundary, creating a T-junction at a vertex shared with the
  // butterfly. This is the minimal three-poly pattern that produced a
  // missing hole in Samples.Sponge4 before the fix.
  SimplePolygon hex = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};
  // CW butterfly enclosing the cell (-0.2, -0.2) .. (0.2, 0.2) with a
  // cancel-pair retrace at (-0.5, 0.0)/(0.0, -0.5) to mimic the Sponge16
  // shape. Net signed area is negative (a hole-bearing loop).
  SimplePolygon butterfly = {{-0.2, 0.0}, {-0.2, 0.2}, {-0.5, 0.2}, {-0.2, 0.2},
                             {0.0, 0.2},  {0.2, 0.2},  {0.2, 0.0},  {0.5, -0.2},
                             {0.2, -0.2}, {0.2, -0.5}, {0.2, -0.2}, {0.0, -0.2},
                             {-0.2, -0.2}};
  // CCW staircase whose v6->v7 edge ((0.2, -0.2) -> (-0.2, -0.2)) is the
  // butterfly's bottom edge, sharing both endpoints with butterfly
  // sub-edges (-0.2, -0.2)..(0.0, -0.2)..(0.2, -0.2) and creating a
  // T-junction at (0.0, -0.2).
  SimplePolygon staircase = {
      {0.2, -0.2}, {-0.2, -0.2}, {-0.2, -0.4}, {0.2, -0.4}};

  CrossSection csBR(Polygons{hex, butterfly, staircase});
  // The butterfly's full interior (cell 0.16 + SE detour triangle 0.03 =
  // 0.19) is a connected CW hole. The CCW staircase below the cell is
  // double-covered (winding 2: hex + staircase), so it stays filled.
  // Expected: outer hex CCW + one CW hole at the butterfly's interior.
  // Without the fix, the butterfly's edges along y=-0.2 (which share
  // endpoints with the staircase's edge) miss the T-junction split at
  // (0.0, -0.2), the canonical mults come out wrong, and the hole
  // collapses or merges with neighboring faces.
  EXPECT_EQ(csBR.NumContour(), 2);
  EXPECT_NEAR(csBR.Area(), 4.0 - 0.19, 1e-9);
}

// Audit follow-up: regression tests for boolean2 filters the post-
// Sponge4 audit argued were correct but didn't have a targeted case.
// Each test verifies a specific topology against a closed-form expected
// result.

// Audit target: canonical sub-edge multiplicity summing for collinear
// overlapping edges. Two CCW rectangles share a collinear segment along
// their boundary but not endpoints; the narrow phase must split A's
// edge at B's endpoints, then Finalize must sum the contributing mults
// correctly so the shared interior segment cancels and only the outer
// "T" outline remains. Without correct mult summing the result would
// drop the shared bottom strip entirely or carry a spurious interior
// edge.
TEST(CrossSection, CollinearSegmentOverlap) {
  SimplePolygon A = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 1.0}, {0.0, 1.0}};
  SimplePolygon B = {{3.0, 0.0}, {7.0, 0.0}, {7.0, 2.0}, {3.0, 2.0}};

  CrossSection cs(Polygons{A, B}, CrossSection::FillRule::Positive);

  EXPECT_EQ(cs.NumContour(), 1);
  // A = 10, B = 8, overlap (4x1 strip on shared bottom) = 4, union = 14.
  EXPECT_NEAR(cs.Area(), 14.0, 1e-9);
}

// Audit target: collinearity gate in CollectIntersectionPairs. N polygon
// wedges all share the origin as a vertex; every pair shares that
// endpoint. The gate must drop pairs that fan out at the center (no
// T-junction possible) while letting through pairs that need their
// non-shared endpoints checked. Result must be the inscribed regular
// N-gon: one contour, area = N/2 * sin(2pi/N). If the gate over-drops
// (filters something needed), interior wedge boundaries leak through;
// if it under-drops (filters nothing), the test still passes but the
// pre-1a057638 perf gain is gone.
TEST(CrossSection, ManyPolygonsShareCenterVertex) {
  constexpr int N = 8;
  Polygons polys;
  for (int i = 0; i < N; ++i) {
    const double a1 = 2.0 * kPi * i / N;
    const double a2 = 2.0 * kPi * (i + 1) / N;
    polys.push_back({{0.0, 0.0},
                     {std::cos(a1), std::sin(a1)},
                     {std::cos(a2), std::sin(a2)}});
  }
  CrossSection cs(polys, CrossSection::FillRule::Positive);

  EXPECT_EQ(cs.NumContour(), 1);
  const double expectedArea = 0.5 * N * std::sin(2.0 * kPi / N);
  EXPECT_NEAR(cs.Area(), expectedArea, 1e-9);
}

// 20-gon with
//   extreme-magnitude radii alternating between O(0.1) and O(8.9).
//   Strict Positive-only cleanup with the old single-miter concave raw-offset
//   emission returned a polygon with area ~ input.Area(), effectively a no-op.
//   Clipper2-style concave emission should let the positive cleanup expand it.
TEST(CrossSection, OffsetPositiveOnExtremeRadiusStar) {
  const std::vector<double> radii = {0.,
                                     0.40098345505108085,
                                     4.7498113621644447e-99,
                                     3.7120810186334277,
                                     8.8389852354367608,
                                     7.8626648130962875e-111,
                                     0.37816850000826657,
                                     0.,
                                     2.0448856906274785e-158,
                                     0.,
                                     0.2582179499017509,
                                     0.,
                                     7.224596115948677e-174,
                                     0.,
                                     0.21952411055214244,
                                     0.,
                                     9.550952284653982e-128,
                                     0.18006645730017631,
                                     0.,
                                     3.9220883255587997e-118};
  SimplePolygon ring;
  ring.reserve(radii.size());
  const int n = static_cast<int>(radii.size());
  for (int i = 0; i < n; ++i) {
    const double r = 0.1 + std::fabs(radii[i]);
    const double theta = 2.0 * kPi * i / n;
    ring.push_back({r * std::cos(theta), r * std::sin(theta)});
  }
  const CrossSection input(ring);
  ASSERT_FALSE(input.IsEmpty());
  ASSERT_GT(std::fabs(input.Area()), 1e-9);

  const double delta = 7.2097955766145416;
  const auto output = input.Offset(delta, CrossSection::JoinType::Bevel,
                                   /*miter_limit=*/2.0, /*circularSegments=*/0);
  EXPECT_FALSE(output.IsEmpty());
  // A positive offset on a non-self-intersecting ring should always
  // grow the area. Observed locally: output.Area() ~= input.Area(),
  // i.e. Offset is effectively a no-op for this input.
  EXPECT_GE(output.Area(), input.Area() - 1e-6 * (1.0 + input.Area()));
}

// Consolidated SubtractInvariants* fuzz-seed regressions, one row per seed.
// Each row carries that seed's radii + translate verbatim; SubtractKind
// selects which assertions the original standalone test made:
//   Invariants         - ExpectBooleanInvariants (both subtract identities
//                        plus inclusion-exclusion)
//   InclusionExclusion - only the inclusion-exclusion identity
// SubtractInvariantsEmptyIntersectionDrop stays standalone (below): it is
// boolean2-gated and asserts only a single subtract identity.
namespace {
enum class SubtractKind { Invariants, InclusionExclusion };

struct SubtractCase {
  const char* name;
  Shape a, b;
  SubtractKind kind = SubtractKind::Invariants;
};

const SubtractCase kSubtractInvariantsSeeds[] = {
    // Extreme magnitude mismatch
    //   between two star polygons - A is ~1e-9 scale, B is ~100 scale.
    //   At least one of the boolean algebraic invariants
    //   `area(A-B)+area(A∩B)=area(A)` etc. fails. Likely an eps-inference
    //   or vertex-merge cliff where A's tiny scale gets crushed by B's
    //   eps.
    {"TinyVsLargeStars",
     {{2.0371964671064575e-14, 1.814002251251902e-10, 0.,
       2.1825088357767143e-09}},
     {{113.5978182908662, 0., 114.34968677141997, 6.5076626333939721e-10},
      {0., 4.667562921730494e-33}}},
    // Two spiky stars at origin - A
    //   is 5-pointed with mixed magnitudes including two ~1000-radius
    //   spikes and a near-zero vertex; B is 4-pointed with three
    //   ~1000-scale spikes plus a near-zero vertex. One of the boolean
    //   algebraic invariants fails. Likely a different code path than
    //   the TinyVsLargeStars needle case - this is two large spiky
    //   shapes, not a tiny shape vs a needle. May share the off-axis
    //   T-junction root cause diagnosed earlier in this table, but the spike-
    //   collision geometry could be its own failure mode.
    {"SpikyStars",
     {{1000., 886.10628147264833, 1000., 1., 0.}},
     {{1000., 827.10387617193078, 548.20533789242359, 0.}}},
    // 17-vertex star with most radii zero collapsed to 0.1, plus a
    //   4-pointed star with one ~112 spike. area(b - a) + area(a ∩ b) =
    //   402.04, b.Area() = 634.18 - 232 units of area leak. Different
    //   geometry from the needle/spiky-star seeds above; the earlier
    //   winding-seed fix didn't cover this case. Reproduces via direct
    //   call with these exact args.
    {"LeakySparseStar",
     {{627.11994612906153, 0.11978140887764367, 601.04982226530046,
       102.31660501134466, 2.1549912703437601, 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0.}},
     {{112.00504648760347, 4.3823848227606392, 1.2795858846899296e-07,
       6.7216166802304542},
      {-2.7193614970785894e-29, 0.}}},
    // 19-vertex sparse star with only 2 nonzero radii (~39, ~21) vs a
    //   4-pointed star with one ~73 spike and tiny radii. area(b - a) +
    //   area(a ∩ b) is off
    //   by ~234 from b.Area() (~788). Different geometry from prior
    //   Subtract seeds.
    {"TwoNonzeroVsNeedle",
     {{0., 39.150613710409736, 0., 21.472413919643575, 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 3.237760857312173e-24}},
     {{72.814105930277677, 21.384583693573447, 1.6644635543899127e-243,
       1.6688953794582528e-229}}},
    // 34-vert star vs 20-vert star
    //   each with one dominant radius (~33 and ~36) and the rest very
    //   small. Inclusion-exclusion violated by ~8 absolute on a∪b≈10.
    //   Verified against post-68cbade7 binary.
    {"DominantSpikeStars",
     {{0.15588462038906103,
       0.00015361176400222398,
       5.8511202761938412e-16,
       3.3009196314697135e-10,
       2.7199740700330746e-09,
       4.690497141269639e-14,
       8.9922571322790269e-10,
       6.3553011716417212e-16,
       4.2872009761335588e-18,
       5.6483346599537904e-17,
       3.479258338980731e-12,
       9.9500479914355979e-14,
       1.822663758661907e-10,
       3.7366088533519133e-05,
       7.6137290475474603e-12,
       1.5917667211519553e-08,
       1.3293803091881353e-06,
       6.5804393417497265e-09,
       0.0015777855119048808,
       3.5317048943785664e-07,
       1.353313563107637e-09,
       1.1788667553671715e-05,
       0.00014847370669519226,
       1.509263403954591e-11,
       6.5516568176598574e-10,
       3.8749326674543751e-07,
       3.450117114134831e-06,
       8.3840226251340948e-08,
       3.0175612629932874e-07,
       33.778786246753299,
       0.033251289695133752,
       0.,
       0.,
       1.5817601023378236}},
     {{3.910135632588194e-07,
       1.0064704530951155e-13,
       2.0158855952179934e-14,
       1.4165380430901347e-21,
       3.6148752792123783e-11,
       1.3306453783491879e-10,
       2.3794160990202857e-13,
       4.0821728496033454e-14,
       4.6816579341479393e-18,
       1.7855060689360453e-17,
       2.4858744170474308e-10,
       1.5313588823631113e-11,
       6.6960358868004941e-08,
       2.1802650917628242e-10,
       5.5611296920831909e-13,
       1.9586931478995027e-08,
       0.,
       36.09032519236748,
       0.,
       7.5123873966542927e-05},
      {0.72140998591309213, 0.}},
     SubtractKind::InclusionExclusion},
};
}  // namespace

TEST(CrossSection, SubtractInvariantsSeeds) {
  for (const auto& c : kSubtractInvariantsSeeds) {
    SCOPED_TRACE(c.name);
    // Flush the seed name so a hard crash is still attributable.
    std::cerr << "[seed] " << c.name << std::endl;
    const CrossSection a = MakeShape(c.a), b = MakeShape(c.b);
    if (c.kind == SubtractKind::InclusionExclusion) {
      const auto aIb = a.Boolean(b, OpType::Intersect);
      const auto aUb = a + b;
      const double tol = AreaTol(a, b);
      EXPECT_NEAR(aUb.Area(), a.Area() + b.Area() - aIb.Area(), tol)
          << "inclusion-exclusion violated";
    } else {
      ExpectBooleanInvariants(a, b);
    }
  }
}

// Consolidated BooleanCommutativity* fuzz-seed regressions, one row per seed.
// Each row carries that seed's radii + translate verbatim. Every seed asserts
// union commutativity (A + B vs B + A: area + contour count); checkIntersect
// additionally asserts intersect commutativity (A ∩ B vs B ∩ A: area +
// contour count).
namespace {
struct CommCase {
  const char* name;
  Shape a, b;
  bool checkIntersect = false;
};

const CommCase kBooleanCommutativitySeeds[] = {
    // Asymmetric handling of two
    //   inputs in the boolean engine - A+B and B+A produce different
    //   results. A is an 11-pointed star with mixed magnitudes
    //   (~115 down to ~1e-32); B is a 6-pointed star with mixed
    //   magnitudes (~96 down to ~1e-53); translation (-0.72, -4.58).
    //   The boolean engine treats one input as "subject" and one as
    //   "clip"; an order-dependent bug in vertex merge or winding sign
    //   would surface here.
    {"MixedScaleStars",
     {{3.0969192681814191e-32, 2.3071236813515518e-31, 2.353005480384586e-24,
       115.24729490924352, 83.850539440722784, 4.3348536506605848,
       3.0129653594607548, 3.1208956318387746, 4.4747952332988792,
       51.973983183682101, 5.99364665010221e-15}},
     {{3.9905161863280855e-53, 0., 96.331001430191975, 0., 0.43952001502476951,
       48.319452987958854},
      {-0.72134106089064531, -4.5808240251267858}},
     /*checkIntersect=*/true},
    // 11-vertex star vs 12-vertex
    //   star both with extreme-magnitude radii ranging from O(1e-40)
    //   to O(55), translated by (-0.36, 2.83). A+B = 11.12 but B+A =
    //   72.75 - off by ~62. Different inputs from
    //   DISABLED_BooleanCommutativityMixedScaleStars.
    {"VeryMixedStars",
     {{5.5755616189085049e-09, 55.713117722084, 0.24425911685471227,
       12.355613962816753, 3.2696445771376686e-21, 7.7223346956456043e-27,
       9.1566312909847612e-29, 1.1757300951916251e-32, 7.5589164688107744e-40,
       2.1073720366622818e-34, 4.7311936525798036e-23}},
     {{0., 55.263016827250659, 0., 2.1568234275576556e-26,
       4.3751784176513804e-26, 3.0217717245002794e-34, 0., 0., 0.,
       5.2142580450244983e-22, 6.1161686141794186e-15, 2.8398350775198264},
      {-0.35827067719250716, 2.8298129605573439}}},
    // A+B and B+A produce different
    //   contour counts: A+B has 14, B+A has 13. Order-dependent
    //   commutativity violation: depending on which input is "subject"
    //   vs "clip", one contour appears/vanishes. Input shapes are
    //   adversarial - 42-vertex star A with extreme magnitudes
    //   (mix of 0, 1, 1000) vs 33-vertex star B with mostly small
    //   radii plus a few in the 700-900 range, translated by
    //   (1.71, -0.50). Likely a vertex-merge or sort tie-break path
    //   that depends on input order rather than canonical position.
    {"ExtremeMagStars",
     {Runs({{1, 849.58267006542974},
            {1, 1000.},
            {1, 1.},
            {1, 1000.},
            {2, 0.},
            {1, 1.},
            {2, 0.},
            {1, 673.91660850339099},
            {1, 641.26963903267847},
            {1, 0.84355834285842513},
            {6, 0.},
            {1, 633.12578243694929},
            {1, 52.949291739509349},
            {1, 596.41353494815576},
            {1, 659.59815503253992},
            {1, 907.85583546669454},
            {1, 2.3971527824334933},
            {1, 0.},
            {1, 744.77799184771879},
            {1, 0.},
            {1, 668.48707869911436},
            {1, 436.35135227432249},
            {3, 0.},
            {1, 1000.},
            {1, 0.},
            {1, 938.50101700824371},
            {1, 84.188304887665581},
            {1, 470.75422504731279},
            {1, 89.810808608112893},
            {1, 992.99533690532235},
            {1, 152.76119378800999},
            {1, 1000.},
            {1, 0.}})},
     {{681.1551582487607,
       1.,
       226.83285963348141,
       25.271695356113401,
       903.10565859434519,
       25.271695356113401,
       27.294237080468665,
       205.64018204902527,
       708.55499051253935,
       934.95934725410712,
       597.99208744386829,
       937.31857430425612,
       1.,
       3.0503172192082628,
       1.,
       1.,
       1000.,
       1000.,
       1.,
       1.,
       901.26525729886407,
       1.,
       1000.,
       1.6267135216510802,
       449.23493668120602,
       449.23493668120602,
       3.1828449559882372,
       6.9538585160876147,
       1.7485381697954843,
       696.88681861119551,
       1.,
       183.08811014255667,
       1.},
      {1.7126290778198534, -0.5023590407011973}}},
};
}  // namespace

TEST(CrossSection, BooleanCommutativitySeeds) {
  for (const auto& c : kBooleanCommutativitySeeds) {
    SCOPED_TRACE(c.name);
    // Flush the seed name so a hard crash is still attributable.
    std::cerr << "[seed] " << c.name << std::endl;
    const CrossSection a = MakeShape(c.a), b = MakeShape(c.b);
    const auto aPlusB = a + b;
    const auto bPlusA = b + a;
    const double tol = AreaTol(a, b);
    EXPECT_NEAR(aPlusB.Area(), bPlusA.Area(), tol) << "A + B != B + A";
    EXPECT_EQ(aPlusB.NumContour(), bPlusA.NumContour());
    if (c.checkIntersect) {
      const auto aIntB = a.Boolean(b, OpType::Intersect);
      const auto bIntA = b.Boolean(a, OpType::Intersect);
      EXPECT_NEAR(aIntB.Area(), bIntA.Area(), tol) << "A ∩ B != B ∩ A";
      EXPECT_EQ(aIntB.NumContour(), bIntA.NumContour());
    }
  }
}

// Consolidated PrismBooleanMatchesCrossSection fuzz-seed regressions, one row
// per seed. Each builds two regular triangles at near-equal circumradii,
// extrudes them to prisms, and asserts the union's Volume matches the 2D union
// area times the height. Rows differ only in the two radii. Ungated: runs on
// both backends.
namespace {
struct PrismCase {
  const char* name;
  double radiusA, radiusB;
};

const PrismCase kPrismSeeds[] = {
    // Two equilateral triangles with
    //   circumradii 0.1 and 0.1+6.88e-13, op=Add. Two near-IDENTICAL
    //   triangles - the union should be essentially one triangle.
    //   Volume check fails by ~0.26 absolute on h=5; the
    //   narrowing of Prism only skipped Project/Slice for Subtract,
    //   not the Volume check which catches this.
    {"NearIdenticalTriangles", 0.10000000000000001, 0.10000000000068791},
    // Two equilateral triangles
    //   with circumradii 1.675 vs 1.669, op=Add. Volume check fails by
    //   ~20 absolute on h=5. Different from the earlier
    //   PrismNearIdenticalTrianglesAdd seed which had radii differing
    //   by 6.88e-13; this one has a ~0.4% radius difference. Verified
    //   against post-68cbade7 binary..
    {"CloseRadiiTriangles", 1.6750962039134867, 1.6691810932888278},
};
}  // namespace

TEST(CrossSection, PrismSeeds) {
  auto regular = [](int sides, double radius) {
    SimplePolygon ring;
    const double r = 0.1 + std::fabs(radius);
    for (int i = 0; i < sides; ++i) {
      const double th = 2.0 * kPi * i / sides;
      ring.push_back({r * std::cos(th), r * std::sin(th)});
    }
    return ring;
  };
  for (const auto& c : kPrismSeeds) {
    SCOPED_TRACE(c.name);
    std::cerr << "[seed] " << c.name << std::endl;
    const CrossSection a(regular(3, c.radiusA));
    const CrossSection b(regular(3, c.radiusB));
    const auto expected = a + b;
    const double h = 5.0;
    const auto solidA = Manifold::Extrude(a.ToPolygons(), h);
    const auto solidB = Manifold::Extrude(b.ToPolygons(), h);
    const auto result = solidA + solidB;
    const double tolScale = 1.0 + std::fabs(a.Area()) + std::fabs(b.Area()) +
                            std::fabs(expected.Area());
    EXPECT_NEAR(result.Volume(), expected.Area() * h, 1e-5 * tolScale * h);
  }
}

// Consolidated BooleanAssociativity* fuzz-seed regressions, one row per seed.
// Each row carries that seed's radii + translate verbatim. Both seeds assert
// union-associativity area equality ((A ∪ B) ∪ C vs A ∪ (B ∪ C)); neither
// asserted intersect associativity.
namespace {
struct AssocCase {
  const char* name;
  Shape a, b, c;
};

const AssocCase kBooleanAssociativitySeeds[] = {
    // Three 4-vertex stars with
    //   mixed magnitudes; (A∪B)∪C = 2.994 but A∪(B∪C) = 5.472 - off
    //   by 2.48. Intersection associativity is fine (matches to 3e-17),
    //   so the asymmetry is in the union's binary-vs-batch path.
    //   Translations are all near-zero (~1e-35 to 1e-28); the trigger
    //   is the radii distribution, not translation.
    {"UnionMixedTriples",
     {{24.575460587300253, 2.4572617728922851e-10, 3.3952718785303559e-06, 0.}},
     {{29.731318514644453, 1.5729051003875837e-06, 0.0082858661009423962, 0.},
      {-9.1863209962415243e-35, -9.8444830049208569e-28}},
     {{0., 0., 6.5474426075871467e-16, 2.9296729904240054e-06},
      {-6.0489837474564476e-29, 0.}}},
    // Three 4-vertex stars with
    //   near-zero radii. (A∪B)∪C = 0.04 but A∪(B∪C) = 0.02 - the
    //   second form drops one unit of area, likely a contour-merge
    //   bug. Distinct from
    //   UnionMixedTriples seed above, which had
    //   larger radii. Verified against post-68cbade7 binary.
    {"TinyStars",
     {{0., 0., 0., 2.9078938473355343e-13}},
     {{1.4795645772678251e-12, 2.4342935254638573e-13, 1.4800031437954507e-12,
       1.1368244372782033e-305},
      {0., 0.}},
     {{0., 0., 0., 0.}}},
};
}  // namespace

TEST(CrossSection, BooleanAssociativitySeeds) {
  for (const auto& c : kBooleanAssociativitySeeds) {
    SCOPED_TRACE(c.name);
    // Flush the seed name so a hard crash is still attributable.
    std::cerr << "[seed] " << c.name << std::endl;
    const CrossSection a = MakeShape(c.a), b = MakeShape(c.b),
                       cc = MakeShape(c.c);
    const auto ab_c = (a + b) + cc;
    const auto a_bc = a + (b + cc);
    const double tol = AreaTol(a, b, cc);
    EXPECT_NEAR(ab_c.Area(), a_bc.Area(), tol) << "(A ∪ B) ∪ C != A ∪ (B ∪ C)";
  }
}

// Consolidated BooleanDistributivity* fuzz-seed regressions, one row per
// seed. Each row carries that seed's radii and translate inputs verbatim;
// DistribKind selects which assertions the original standalone test made:
//   AreaOnly     - only the area-equality check
//   Standard     - ExpectDistributesOverUnion (area-equality plus both
//                  one-directional area-difference checks)
//   Monotonicity - the Standard checks plus the two containment checks
//                  (A intersect B and A intersect C each inside the union)
// Kept as a single TEST in the CrossSection suite so the CrossSection.*
// CI filter keeps exercising every seed.
namespace {
enum class DistribKind { Standard, AreaOnly, Monotonicity };

struct DistribCase {
  const char* name;
  Shape a, b, c;
  DistribKind kind = DistribKind::Standard;
};

const DistribCase kDistributivitySeeds[] = {
    // Three 4-vertex stars with all
    //   near-zero radii. A∩(B∪C) returns 0 but (A∩B)∪(A∩C) returns
    //   0.02 - off by 0.02. Tiny inputs near eps; the distributivity
    //   path may have lost a contour during simplification. Verified
    //   against post-68cbade7 binary.
    {"TinyStars",
     {{0., 0., 0., 0.}},
     {{0., 0., 0., 2.6526376418441693e-13}, {0., 0.}},
     {{5.8033140339376039e-12, 4.8677534136980153e-13, 0., 0.},
      {-2.4305117516652688e-13, 0.}},
     DistribKind::AreaOnly},
    // Left side `A ∩ (B ∪ C)` has
    //   NumContour=24, right side `(A ∩ B) ∪ (A ∩ C)` has NumContour=23.
    //   Off-by-one in contour decomposition. Inputs are three mid-size
    //   stars (34/46/48 verts) with extreme-magnitude radii (mix of 0,
    //   1, 1000-ish) and translations in [-2.4, 3.9] x [-1.0, 2.4].
    //   Likely the interleaved Union+Intersect path produces or drops
    //   a contour relative to the (Intersect, Intersect, Union)
    //   ordering. Related to but distinct from previously-drained
    //   BooleanCommutativity / BooleanAssociativity classes.
    {"ExtremeMagStars",
     {{22.40352672886273,
       1000.,
       1.930469114705216,
       1000.,
       995.07901723564635,
       450.42497103653432,
       1.,
       1000.,
       442.67830885450564,
       974.26213405728731,
       762.40083808845225,
       375.22374612076686,
       311.70408869554416,
       4.6596927406080066,
       1000.,
       993.53293688808651,
       545.61942022366179,
       902.87869535025561,
       857.23036058152354,
       1.,
       997.14305195601548,
       998.81543502854458,
       4.287799655231888,
       1.9578049801330417,
       212.99717518614997,
       539.98252283023282,
       814.2147123777587,
       6.4135462648884491,
       386.69166945488905,
       642.25190405836315,
       671.02238904111891,
       1000.,
       0.,
       959.32882825167917}},
     {{596.78699570023025,
       0.,
       0.,
       4.8215130310908281,
       134.35969768300723,
       514.84369184653679,
       514.84369184653679,
       514.84369184653679,
       134.35969768300723,
       134.35969768300723,
       1000.,
       1.,
       1000.,
       0.,
       926.29500965174145,
       0.,
       134.35969768300723,
       0.90493025476122824,
       95.854999537523781,
       1000.,
       0.,
       0.,
       460.32237309958083,
       2.9791218075989327,
       30.186663600739344,
       1.,
       5.732788180417673,
       487.77994489187455,
       1.,
       639.36782925291504,
       1.,
       1.,
       405.85342012592389,
       405.85342012592389,
       405.85342012592389,
       706.56289376450047,
       405.85342012592389,
       405.85342012592389,
       1.,
       1.,
       872.8336783186561,
       5.1830997334306659,
       313.84753596061228,
       3.3318960586713984,
       22.317598513501498,
       4.1956604335614083},
      {3.4425093025001434, -0.89018350575014171}},
     {{279.68331311398725,
       670.56312431772176,
       812.15251441725718,
       1000.,
       0.,
       996.64035079388486,
       1.,
       1000.,
       1.,
       146.36259356992105,
       1000.,
       1.2440205144455736,
       2.4903214421341642,
       558.11442117895865,
       667.56514527125182,
       601.28991029625661,
       0.,
       0.,
       1.,
       485.29900078511309,
       205.35289543499823,
       1.,
       1.,
       999.13575735403208,
       0.,
       343.12355139003932,
       124.51182502694519,
       0.44553220583928932,
       512.45895098827191,
       2.5455428684940777,
       0.,
       1000.,
       1000.,
       0.,
       595.63567011806413,
       2.5851695628219766,
       3.2720224325713057,
       354.45821723238265,
       4.2476895774067867,
       956.03225557664916,
       0.,
       1000.,
       0.,
       98.799661756610732,
       0.,
       619.55626420041142,
       0.,
       999.86711403007541},
      {3.8936971336588151, 2.3137657561842939}},
     DistribKind::AreaOnly},
    // A ∩ (B ∪ C) != (A ∩ B) ∪ (A ∩ C)
    //   on large-area stars. left.Area=216374, right.Area=215961, diff=412
    //   (relative ~0.2%, exceeds 1e-6 * sum-of-areas tol of 2.34).
    //   Inputs: 16-vertex star A with mixed radii, 4-vertex star B (small),
    //   37-vertex star C with most radii repeated as 332.47 (degenerate-ish
    //   shape). Translations: B by (4.89, 3.65), C by (-1.53, -2.63).
    //   Distinct from previously-fixed BooleanDistributivityExtremeMagStars
    //   (hash 659ec969) - that was a contour-count off-by-one, this is
    //   area drift in absolute terms.
    {"RepeatedRadiusStarC",
     {{1., 283.64569337262878, 0., 705.96814039386641, 332.07185213769458,
       534.61157622175767, 1000., 1000., 1000., 1000., 1000., 1000., 1000., 1.,
       1., 1000.}},
     {{974.53903971669604, 443.79612063734504, 0.60212977272176105,
       601.54223181860084},
      {4.890230810667985, 3.6470913496183108}},
     {Runs({{1, 2.8991656937704362},
            {1, 1000.},
            {1, 1.},
            {21, 332.4725376862869},
            {2, 384.70682133429182},
            {2, 332.4725376862869},
            {1, 682.65150344752476},
            {1, 737.43471827249516},
            {1, 327.59982985461647},
            {2, 332.4725376862869},
            {1, 1.},
            {3, 1000.}}),
      {-1.5262817329731697, -2.6347725451611175}}},
    // Sibling of the just-drained
    //   RepeatedRadiusStarC. Smaller magnitude (left.Area=80787,
    //   right.Area=80781, diff=5.9 vs tol=1.28). Star C has many zeros
    //   mixed with mid-range values - a different degeneracy pattern
    //   than the previous all-repeated-radius case.
    {"ZeroMixStarC",
     {{1000.,
       1000.,
       255.79995725682645,
       1.,
       454.29432950304715,
       995.65401133735929,
       1000.,
       811.86733957046874,
       1.,
       437.04103091878892,
       195.25477231407794,
       1.8268273914683537,
       4.5954160688751484,
       3.2030613706914481,
       511.5935307510378,
       2.0771806846580496,
       570.21917979539887,
       1.9769538901315753,
       467.93526395973339,
       1.,
       959.79674877569607,
       0.076792444298594276,
       503.25635714266843,
       0.,
       89.215459767705454,
       0.,
       1000.,
       0.06374847983784826,
       995.14671048524883,
       1.,
       0.}},
     {{271.39306287742255, 997.96011166439871, 909.94463259654469,
       364.30795394492606},
      {0.12749108485042271, 2.533194291154123}},
     {{812.15251441725718,
       0.,
       601.28991029625661,
       0.,
       482.34411167146169,
       0.,
       588.04482477271347,
       0.,
       169.03839642543201,
       0.44553220583928932,
       4.0649812370918958,
       512.45895098827191,
       0.,
       1000.,
       0.,
       999.90271349806335,
       459.5889801989477,
       105.49723198609914,
       0.,
       0.,
       748.07738025389699,
       1.7632947108253827,
       0.,
       0.,
       4.9202747062496162,
       751.85659442712381,
       732.78647234489199,
       0.,
       375.32896975602767,
       0.},
      {-2.5843878979814856, -4.3357952411613461}}},
    // Residual distributivity failure
    //   after a9cfd4ff "Use nonzero union for boolean2 set add". A is a
    //   33-radii star, C is a 32-radii star with zeros interleaved with
    //   mid-large radii. Area asymmetry: left.Area ~ 71882, right.Area
    //   ~ 66448, diff 5435 (~7%). The right side (A ∩ B) ∪ (A ∩ C) is
    //   contained in left ((right - left).Area == 0) but is missing
    //   5435 area units. NumContour also diverges (left=1, right=5).
    //   The zero-radius vertices collapse to the 0.1 baseline floor,
    //   producing dense near-degenerate slivers that the intersect path
    //   appears to drop on one side of the distributivity identity.
    {"NonzeroUnionResidual",
     {{1000.,
       1000.,
       1.,
       454.29432950304715,
       995.65401133735929,
       1000.,
       1.,
       1000.,
       437.04103091878892,
       195.25477231407794,
       1.8268273914683537,
       4.5954160688751484,
       3.2030613706914481,
       511.5935307510378,
       2.0771806846580496,
       643.7557324509894,
       570.21917979539887,
       1.9769538901315753,
       467.93526395973339,
       2.8343333181297945,
       832.7231586632563,
       1.9935416909420567,
       3.4941254801335897,
       503.25635714266843,
       0.,
       677.87894521363603,
       1000.,
       0.,
       1000.,
       0.06374847983784826,
       995.14671048524883,
       1.,
       0.}},
     {{381.47868752207233, 271.38539902365324, 902.29172154497269,
       369.78142089265032},
      {-4.2594328040756597, -3.826038498814941}},
     {{812.15251441725718,
       0.,
       996.64035079388486,
       1.2440205144455736,
       845.2781504274202,
       1.7150332286725525,
       601.28991029625661,
       0.,
       482.34411167146169,
       0.,
       588.04482477271347,
       0.,
       169.03839642543201,
       512.45895098827191,
       3.4002817704521431,
       793.8531142632803,
       234.81914439627278,
       0.,
       999.90271349806335,
       105.49723198609914,
       0.,
       1000.,
       0.,
       705.33517104963164,
       2.2547666355380791,
       949.94413913719598,
       0.,
       4.9202747062496162,
       751.85659442712381,
       0.,
       375.32896975602767,
       0.},
      {-4.6033649212771799, -4.3357952411613461}},
     DistribKind::Monotonicity},
    // Companion to
    //   DISABLED_BooleanDistributivityNonzeroUnionResidual above, but
    //   with the zeros in A and an intermediate-sized B - 28-radii A
    //   with leading 0., 0. plus repeated 85.93 values, 16-radii B
    //   dominated by 1000s with a few small/zero values, 4-radii simple
    //   C. Same failure shape: left has 1 contour, right has 3, right
    //   strictly contained in left, missing ~4001 area units out of
    //   ~16255 (~25%). Confirms the intersect-path sliver-drop bug is
    //   not specific to which input carries the zeros.
    {"ZerosInANonzeroUnion",
     {Runs({{2, 0.},
            {2, 85.932045280269435},
            {1, 503.627431790542},
            {1, 85.932045280269435},
            {2, 90.069450175585047},
            {7, 1.},
            {1, 3.0409033274230679},
            {1, 90.340071855270295},
            {1, 88.580867960622072},
            {1, 261.53938936688053},
            {1, 1.},
            {6, 85.932045280269435},
            {1, 1000.},
            {1, 1.}})},
     {{0., 1000., 1000., 150.433665233841, 1000., 1000., 1., 1000., 1000.,
       1000., 1000., 177.35813950095334, 0.04587971958967612,
       635.76724699191789, 0., 801.00098835163021},
      {0.70686431311486997, 4.2881751021813148}},
     {{2.2620915191424817, 352.8271046252292, 124.40008725909412,
       583.15656615652506},
      {1.9937691377314026, 4.7666664390133615}},
     DistribKind::Monotonicity},
    // Small-residual failure on the
    //   OPPOSITE side from the previous nonzero-union residuals: here
    //   right > left, with left.Area=172524 and right.Area=172618 -
    //   diff only 94.6 (~0.06%) but (left-right).Area==0 and
    //   (right-left).Area=94. NumContour also reversed: left=25,
    //   right=8 - right side collapses many contours into fewer.
    //   Suggests the right-hand `(A∩B) ∪ (A∩C)` is over-merging
    //   adjacent components and accidentally including area that the
    //   left-hand `A ∩ (B∪C)` correctly carves out. Distinct from the
    //   "right is strictly contained in left, missing area" shape of
    //   86029efb/278f30ca seeds which were fixed by the recent
    //   nonzero-outer-face classifier change.
    {"RightOverMerges",
     {{1000., 995.07901723564635, 1., 1000., 974.26213405728731,
       974.26213405728731, 887.65436435582899, 1000., 0., 1000., 1000.,
       0.06374847983784826, 671.02238904111891, 671.02238904111891,
       671.02238904111891, 0.}},
     {Runs({{1, 165.9646574194592},
            {2, 134.35969768300723},
            {3, 0.},
            {1, 134.35969768300723},
            {1, 0.90493025476122824},
            {1, 95.854999537523781},
            {1, 0.},
            {1, 730.30962259021919},
            {1, 0.},
            {1, 460.32237309958083},
            {1, 2.9791218075989327},
            {1, 30.186663600739344},
            {1, 1.},
            {1, 90.446292042234916},
            {1, 5.732788180417673},
            {1, 487.77994489187455},
            {1, 4.5933105105744643},
            {1, 110.52720785200476},
            {3, 1.},
            {1, 338.4042053897046},
            {2, 1.},
            {2, 405.85342012592389},
            {1, 0.},
            {1, 1.},
            {1, 405.85342012592389},
            {1, 706.56289376450047},
            {2, 405.85342012592389},
            {9, 1.}}),
      {0.61958285855050033, -2.9878148015921839}},
     {{279.68331311398725,
       814.7527470368675,
       812.15251441725718,
       1000.,
       0.,
       996.64035079388486,
       1.,
       1000.,
       1.,
       0.,
       146.36259356992105,
       1000.,
       1.2440205144455736,
       2.4903214421341642,
       1.4267332381508839,
       1.0658012274682902,
       937.2477438999249,
       601.28991029625661,
       0.,
       482.34411167146169,
       0.,
       999.13575735403208,
       0.,
       124.51182502694519,
       0.44553220583928932,
       4.0649812370918958,
       512.45895098827191,
       1000.,
       0.,
       0.,
       0.,
       595.63567011806413,
       2.5851695628219766,
       354.45821723238265,
       4.2476895774067867,
       956.03225557664916,
       0.,
       0.,
       98.799661756610732,
       0.,
       1000.,
       1.},
      {2.4887547431653907, 4.8411440198518338}}},
    // Same failure shape as the
    //   already-drained 86029efb/278f30ca residuals - right strictly
    //   contained in left, missing area - but on larger inputs (34/19/45
    //   radii). Area diff 26631 (~19% of left), NumContour 11 vs 10.
    //   Demonstrates that the nonzero-outer-face classifier fix did not
    //   fully drain the bug class: the specific seeded counterexamples
    //   pass, but neighbors in the same family with larger zero/repeat
    //   counts still trigger the same misclassification.
    {"LargeInputsResidual",
     {Runs({{1, 215.54119679461166},
            {1, 0.},
            {1, 1000.},
            {2, 85.932045280269435},
            {1, 503.627431790542},
            {1, 85.932045280269435},
            {1, 1.6560313083633313},
            {2, 90.069450175585047},
            {4, 1.},
            {1, 0.},
            {4, 1.},
            {1, 87.027104618165325},
            {1, 85.932045280269435},
            {1, 258.81750993550418},
            {1, 1000.},
            {1, 1.},
            {1, 0.},
            {1, 89.294011990045064},
            {5, 85.932045280269435},
            {2, 1000.},
            {1, 1.}})},
     {{1000., 0., 997.02489346940774, 997.07890660881435, 319.71364680208376,
       675.57216392823045, 997.55565490398703, 619.18266326889307,
       345.22578984765374, 3.1502725292253997, 121.95185049831551,
       585.41131030666656, 999.93563090020007, 1000., 0., 177.35813950095334,
       0.04587971958967612, 882.24641989432973, 1.1946278115288917},
      {-4.3860291464683625, 2.2730433830580954}},
     {Runs({{1, 1.},
            {1, 1000.},
            {1, 1.},
            {1, 282.41410155737179},
            {1, 1.},
            {4, 282.41410155737179},
            {1, 1000.},
            {1, 1.},
            {2, 1000.},
            {2, 0.},
            {15, 1000.},
            {1, 578.55699544670244},
            {13, 1000.},
            {1, 1.}}),
      {0.99578244833742113, -4.0475870932225444}}},
};
}  // namespace

TEST(CrossSection, BooleanDistributivitySeeds) {
  for (const auto& c : kDistributivitySeeds) {
    SCOPED_TRACE(c.name);
    // Print the seed name unconditionally: SCOPED_TRACE is not flushed on a
    // raw abort/segfault, so this line is what attributes a hard crash to a
    // specific seed.
    std::cerr << "[seed] " << c.name << std::endl;
    const CrossSection a = MakeShape(c.a), b = MakeShape(c.b),
                       cc = MakeShape(c.c);
    if (c.kind == DistribKind::AreaOnly) {
      EXPECT_NEAR(
          a.Boolean(b + cc, OpType::Intersect).Area(),
          (a.Boolean(b, OpType::Intersect) + a.Boolean(cc, OpType::Intersect))
              .Area(),
          AreaTol(a, b, cc))
          << "A ∩ (B ∪ C) != (A ∩ B) ∪ (A ∩ C)";
    } else {
      ExpectDistributesOverUnion(a, b, cc);
    }
    if (c.kind == DistribKind::Monotonicity) {
      const double tol = AreaTol(a, b, cc);
      const auto right =
          a.Boolean(b, OpType::Intersect) + a.Boolean(cc, OpType::Intersect);
      EXPECT_NEAR((a.Boolean(b, OpType::Intersect) - right).Area(), 0.0, tol)
          << "union monotonicity: (A ∩ B) is not contained in right";
      EXPECT_NEAR((a.Boolean(cc, OpType::Intersect) - right).Area(), 0.0, tol)
          << "union monotonicity: (A ∩ C) is not contained in right";
    }
  }
}

// Regression in CrossSection::Offset (src/cross_section.cpp).
//   Regular triangle, Miter join exactly on the miter_limit=2.0
//   threshold; the Offset(d).Offset(-d) round-trip drifts -0.35% in area
//   in a way that's scale-invariant (same percentage at r=0.15, r=1.5,
//   r=15). n>=4 has zero drift (verified probe), so the bug is specific
//   to the miter-limit boundary of the equilateral triangle.
TEST(CrossSection, OffsetInverseTriangleMiter) {
  // Equilateral triangle inscribed in r=0.15 (effective radius via the
  // 0.1 + |radius| convention used in cross_section_fuzz).
  SimplePolygon ring;
  const double r = 0.15;
  for (int i = 0; i < 3; ++i) {
    const double theta = 2.0 * kPi * i / 3;
    ring.push_back({r * std::cos(theta), r * std::sin(theta)});
  }
  const CrossSection input(ring);
  const double delta = -0.0094938192047002799;

  const auto expanded =
      input.Offset(delta, CrossSection::JoinType::Miter, /*miter_limit=*/2.0,
                   /*circularSegments=*/0);
  const auto roundTrip = expanded.Offset(-delta, CrossSection::JoinType::Miter,
                                         /*miter_limit=*/2.0,
                                         /*circularSegments=*/0);

  // For Miter join with miter_limit=2.0 exactly at the equilateral triangle
  // threshold, equality should still take the miter path. Observed:
  // 0.357% absolute drift when rounded normals spuriously square the join.
  const double tol = 1e-4 * (1.0 + std::fabs(input.Area()));
  EXPECT_NEAR(roundTrip.Area(), input.Area(), tol)
      << "Triangle Miter Offset round-trip drifted by "
      << (roundTrip.Area() - input.Area()) / input.Area() * 100 << "%";
  EXPECT_EQ(roundTrip.NumContour(), input.NumContour());
}

// CrossSection::Decompose path,
//   most likely the containment/face-classification step that picks
//   which rings to emit per component. An 8-vertex star outer with
//   a small translated hole produces a 2-contour CrossSection
//   (NumContour=2: outer+hole, Area=911.40 = 911.70 outer - 0.30
//   hole). Decompose returns 1 component whose Area=911.70 and
//   NumContour=1 - the hole is silently dropped. Compose then gives
//   a full-outer result, losing 0.3 area worth of hole. Area
//   conservation invariant violated; bidirectional Decompose/Compose
//   should be the identity on multi-ring inputs.
TEST(CrossSection, DecomposeRecomposeOuterStarWithSmallHole) {
  const std::vector<double> outerRadii = {
      1., 1., 1., 50., 50., 0., 3.987798525003551, 1.};
  const std::vector<double> holeRadii = {0., 0., 1., 0.29444504003509697};
  const CrossSection outer(StarRing(outerRadii));
  const CrossSection hole =
      CrossSection(StarRing(holeRadii)).Translate({0., 1.});
  const auto holed = outer - hole;
  ASSERT_EQ(holed.NumContour(), 2u);  // sanity: subtract produced
                                      // outer + hole
  const double holedArea = holed.Area();

  const auto components = holed.Decompose();
  ASSERT_FALSE(components.empty());

  double componentArea = 0.0;
  size_t componentContourSum = 0;
  for (const auto& component : components) {
    componentArea += component.Area();
    componentContourSum += component.NumContour();
  }
  const double tol = 1e-6 * (1.0 + std::fabs(holedArea));
  EXPECT_NEAR(componentArea, holedArea, tol)
      << "Decompose lost area on holed input: " << "sum(components.Area)="
      << componentArea << " vs holed.Area()=" << holedArea;
  EXPECT_EQ(componentContourSum, holed.NumContour())
      << "Decompose lost contours on holed input: "
      << "sum(components.NumContour)=" << componentContourSum
      << " vs holed.NumContour()=" << holed.NumContour();

  const auto recomposed = CrossSection::Compose(components);
  EXPECT_NEAR(recomposed.Area(), holedArea, tol);
  EXPECT_EQ(recomposed.NumContour(), holed.NumContour());
}

// Same Decompose area-conservation
//   class as DecomposeRecomposeOuterStarWithSmallHole but a distinct
//   counterexample. 5-vertex outer star (radii skewed: 48.55, 5.05,
//   1, 1, 50 - one large spike) minus a 5-vertex inner star
//   (radii 1, 3.75, 0.573, 1, 0) translated by (0, 1.0012).
//   componentArea=1304.5424 but holed.Area()=1304.5751, diff 0.0327
//   exceeds tol 0.0013. Decompose drops a chunk of hole-adjacent
//   area rather than the entire hole this time. Possible: when the
//   hole is near-tangent to the outer (the 1.0012 offset puts it
//   very close to an outer edge), face classification miscategorizes
//   a sliver that gets dropped on the decompose path.
TEST(CrossSection, DecomposeRecomposeNearTangentSmallHole) {
  // Match the fuzz target's exact construction (outer - hole) so the
  // reproducer goes through the same Boolean Subtract -> Decompose
  // path that fuzzing exercised.
  const std::vector<double> outerRadii = {48.55001516665169, 5.0536385110757127,
                                          1., 1., 50.};
  const std::vector<double> holeRadii = {1., 3.7464608085566375,
                                         0.57299724595371804, 1., 0.};
  const CrossSection outer(StarRing(outerRadii));
  const CrossSection hole =
      CrossSection(StarRing(holeRadii)).Translate({-0., 1.0011960822937693});
  const auto holed = outer - hole;
  ASSERT_FALSE(holed.IsEmpty());
  ASSERT_GE(holed.NumContour(), 2u)
      << "expected outer + hole, got NumContour=" << holed.NumContour();
  const double holedArea = holed.Area();

  const auto components = holed.Decompose();
  ASSERT_FALSE(components.empty());

  double componentArea = 0.0;
  size_t componentContourSum = 0;
  for (const auto& c : components) {
    componentArea += c.Area();
    componentContourSum += c.NumContour();
  }
  const double tol = 1e-6 * (1.0 + std::fabs(holedArea));
  EXPECT_NEAR(componentArea, holedArea, tol)
      << "Decompose lost area on holed input: sum(components.Area)="
      << componentArea << " vs holed.Area()=" << holedArea;
  EXPECT_EQ(componentContourSum, holed.NumContour())
      << "Decompose split or merged contours unexpectedly";

  const auto recomposed = CrossSection::Compose(components);
  EXPECT_NEAR(recomposed.Area(), holedArea, tol)
      << "Compose(Decompose(holed)) changed area";
  EXPECT_EQ(recomposed.NumContour(), holed.NumContour())
      << "Compose(Decompose(holed)) changed contour count";
}

// Inclusion-exclusion fails when
//   one vertex of B is set exactly coincident with a vertex of A
//   (op=2 in the DegenerateInputFuzz harness). Both stars are 6
//   vertices with mid-range radii. unionAB.Area=436995 but
//   ca.Area+cb.Area-intersectAB.Area=434285, diff 2710 (~0.6%) far
//   above the 4.34 tol. intersectAB.Area only 0.51 - tiny sliver,
//   yet the union absorbs ~2710 extra area. Likely the coincident
//   vertex is mis-classified during merge, creating phantom area
//   in the union or losing it from the intersection.
TEST(CrossSection, DegenerateCoincidentVertexUnion) {
  const std::vector<double> radiiA = {1.0169016983060246, 1000., 1.,
                                      578.85382959129936, 0.,    0.};
  const std::vector<double> radiiB = {0.,
                                      1000.,
                                      999.83083173100238,
                                      7.275875880519048,
                                      726.89009231357352,
                                      3.880747251022969};
  SimplePolygon a = StarRing(radiiA);
  SimplePolygon b = StarRing(radiiB);
  for (auto& v : b) {
    v.x += 0.0030378301550779696;
  }
  // op=94 % 4 = 2 in DegenerateInputFuzz: set b[j] = a[i] to create
  // a coincident-vertex degeneracy. idxA=25 % 6 = 1, idxB=52 % 6 = 4.
  b[4] = a[1];

  const CrossSection ca(a);
  const CrossSection cb(b);
  const auto unionAB = ca + cb;
  const auto intersectAB = ca.Boolean(cb, OpType::Intersect);
  const double sum = ca.Area() + cb.Area() - intersectAB.Area();
  const double tol = 1e-5 * (1.0 + std::fabs(ca.Area()) + std::fabs(cb.Area()));
  EXPECT_NEAR(unionAB.Area(), sum, tol)
      << "Inclusion-exclusion violated on coincident-vertex degenerate input";
}

// Reduced from DegenerateCoincidentVertexUnion after constructor round-trip:
// already-regularized inputs still reproduce the original ~2710 area residual.
// This keeps both B triangles and drops two A vertices.
TEST(CrossSection, DegenerateCoincidentVertexUnionReduced) {
  const Polygons a = {{
      {500.05000000000018, 866.11200632481712},
      {-0.54999999999999716, 0.95262794416288443},
      {-578.95382959129938, 5.6843418860808015e-14},
      {-0.049999999999997158, -0.08660254037846471},
  }};
  const Polygons b = {
      {{500.04999974215514, 866.1120058797319},
       {-499.96237803534598, 865.96550230635103},
       {-7.3728380503639697, 0}},
      {{500.04999974215514, 866.1120058797319},
       {499.66038873910634, 865.43178211833663},
       {500.05303783015518, 866.11200632481712}},
  };

  const CrossSection ca(a);
  const CrossSection cb(b);
  const auto unionAB = ca + cb;
  const auto intersectAB = ca.Boolean(cb, OpType::Intersect);
  const CrossSection combined(Polygons{a[0], b[0], b[1]});
  const double sum = ca.Area() + cb.Area() - intersectAB.Area();
  const double tol = 1e-5 * (1.0 + std::fabs(ca.Area()) + std::fabs(cb.Area()));
  EXPECT_NEAR(unionAB.Area(), sum, tol)
      << "Inclusion-exclusion violated on reduced coincident-vertex input";
  EXPECT_NEAR(combined.Area(), sum, tol)
      << "Constructor edge soup lost area on reduced coincident-vertex input";
}

// Smaller 4+3 reduction from the same parked seed: keeping only the tiny top
// B triangle isolates the near-corner edge-vertex double-hit that used to make
// both binary union and single-constructor edge soup lose the large A contour.
TEST(CrossSection, DegenerateCoincidentVertexUnionTinyTriangle) {
  const Polygons a = {{
      {500.05000000000018, 866.11200632481712},
      {-0.54999999999999716, 0.95262794416288443},
      {-578.95382959129938, 5.6843418860808015e-14},
      {-0.049999999999997158, -0.08660254037846471},
  }};
  const Polygons b = {{
      {500.04999974215514, 866.1120058797319},
      {499.66038873910634, 865.43178211833663},
      {500.05303783015518, 866.11200632481712},
  }};

  auto translate = [](Polygons polys, double delta) {
    for (auto& loop : polys) {
      for (vec2& p : loop) {
        p.x += delta;
        p.y += delta;
      }
    }
    return polys;
  };

  for (double offset : {0.0, 1024.0, 4096.0, 8192.0}) {
    SCOPED_TRACE(offset);
    const Polygons shiftedA = translate(a, offset);
    const Polygons shiftedB = translate(b, offset);
    const CrossSection ca(shiftedA);
    const CrossSection cb(shiftedB);
    const auto unionAB = ca + cb;
    const auto intersectAB = ca.Boolean(cb, OpType::Intersect);
    const CrossSection combined(Polygons{shiftedA[0], shiftedB[0]});
    const double sum = ca.Area() + cb.Area() - intersectAB.Area();
    const double tol =
        1e-5 * (1.0 + std::fabs(ca.Area()) + std::fabs(cb.Area()));
    const double tinyContributionTol = 0.01 * cb.Area();
    EXPECT_GT(cb.Area(), 1e-4);
    EXPECT_GT(unionAB.Area(), ca.Area() + 0.5 * cb.Area())
        << "Binary union dropped most of the tiny triangle contribution";
    EXPECT_GT(combined.Area(), ca.Area() + 0.5 * cb.Area())
        << "Constructor edge soup dropped most of the tiny triangle "
           "contribution";
    EXPECT_NEAR(unionAB.Area(), sum, tol)
        << "Inclusion-exclusion violated on tiny-triangle reduction";
    EXPECT_NEAR(unionAB.Area(), sum, tinyContributionTol)
        << "Binary union dropped the tiny triangle contribution";
    EXPECT_NEAR(combined.Area(), sum, tinyContributionTol)
        << "Constructor edge soup dropped the tiny triangle contribution";
  }
}

TEST(CrossSection, NonFiniteInputReturnsEmpty) {
  const double inf = std::numeric_limits<double>::infinity();
  SimplePolygon bad = {{0.0, 0.0}, {1.0, 0.0}, {inf, 1.0}, {0.0, 1.0}};

  CrossSection constructed(bad);
  EXPECT_TRUE(constructed.IsEmpty());

  Polygons finite = {{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}};
  EXPECT_TRUE(Boolean2D(Polygons{bad}, finite, OpType::Add).empty());
}

TEST(CrossSection, OffsetIsInvariantUnderLargeTranslation) {
  const CrossSection square = CrossSection::Square({10.0, 10.0}, true);
  const CrossSection origin =
      square.Offset(1.0, CrossSection::JoinType::Round, 2.0, 8);
  const CrossSection translated =
      square.Translate({1e12, -1e12})
          .Offset(1.0, CrossSection::JoinType::Round, 2.0, 8)
          .Translate({-1e12, 1e12});

  EXPECT_EQ(translated.NumContour(), origin.NumContour());
  EXPECT_EQ(translated.NumVert(), origin.NumVert());
  EXPECT_NEAR(translated.Area(), origin.Area(), 1e-3);
}

TEST(CrossSection, SimplifyPostFiltersBoolean2Output) {
  const double apex = 1.0148512233354445e-6;
  const SimplePolygon tri = {{-1.0, 0.0}, {1.0, 0.0}, {0.0, apex}};
  const SimplePolygon quad = {
      {-0.05, -1.0}, {0.05, -1.0}, {0.05, 2.0}, {-0.05, 2.0}};
  const CrossSection input(Polygons{tri, quad},
                           CrossSection::FillRule::Positive);

  const CrossSection once = input.Simplify();
  const CrossSection twice = once.Simplify();

  // 10 verts (incl. the apex just above the 1e-6 default eps), matching
  // clipper2: Simplify regularizes at the geometry eps then RDP-reduces at eps.
  EXPECT_EQ(once.NumContour(), 1);
  EXPECT_EQ(once.NumVert(), 10);
  EXPECT_EQ(twice.NumContour(), once.NumContour());
  EXPECT_EQ(twice.NumVert(), once.NumVert());
  EXPECT_NEAR(twice.Area(), once.Area(), 1e-12);
}

TEST(CrossSection, SimplifyRemovesCollinearVertices) {
  // RDP: vertices within eps of the line through their neighbors (the edge
  // midpoints here) are dropped; the four corners are kept.
  const SimplePolygon boxWithMidpoints = {{0, 0}, {1, 0}, {2, 0}, {2, 1},
                                          {2, 2}, {1, 2}, {0, 2}, {0, 1}};
  const CrossSection simplified = CrossSection(boxWithMidpoints).Simplify(1e-3);
  EXPECT_EQ(simplified.NumVert(), 4);
  EXPECT_NEAR(simplified.Area(), 4.0, 1e-9);
}

TEST(CrossSection, SimplifyReducesCurveWithoutCollapsing) {
  // A moderate epsilon reduces a dense curve but must never delete it
  // (regression: feeding epsilon as the boolean merge tolerance collapsed
  // circles to empty once epsilon exceeded the vertex spacing).
  const CrossSection circle = CrossSection::Circle(10.0, 64);
  const double area0 = circle.Area();
  const CrossSection simplified = circle.Simplify(0.5);
  EXPECT_FALSE(simplified.IsEmpty());
  EXPECT_LT(simplified.NumVert(), 64u);
  EXPECT_GT(simplified.NumVert(), 4u);
  EXPECT_GT(simplified.Area(), 0.9 * area0);
}

TEST(CrossSection, Empty) {
  Polygons polys(2);
  auto e = CrossSection(polys);
  EXPECT_TRUE(e.IsEmpty());
}

TEST(CrossSection, Rect) {
  double w = 10;
  double h = 5;
  auto rect = Rect({0, 0}, {w, h});
  CrossSection cross(rect);
  auto area = rect.Area();

  EXPECT_FLOAT_EQ(area, w * h);
  EXPECT_FLOAT_EQ(area, cross.Area());
  EXPECT_TRUE(rect.Contains({5, 5}));
  EXPECT_TRUE(rect.Contains(cross.Bounds()));
  EXPECT_TRUE(rect.Contains(Rect()));
  EXPECT_TRUE(rect.DoesOverlap(Rect({5, 5}, {15, 15})));
  EXPECT_TRUE(Rect().IsEmpty());
}

TEST(CrossSection, Transform) {
  auto sq = CrossSection::Square({10., 10.});
  auto a = sq.Rotate(45).Scale({2, 3}).Translate({4, 5});

  mat3 trans({1.0, 0.0, 0.0},  //
             {0.0, 1.0, 0.0},  //
             {4.0, 5.0, 1.0});
  mat3 rot({cosd(45), sind(45), 0.0},   //
           {-sind(45), cosd(45), 0.0},  //
           {0.0, 0.0, 1.0});
  mat3 scale({2.0, 0.0, 0.0},  //
             {0.0, 3.0, 0.0},  //
             {0.0, 0.0, 1.0});

  auto b = sq.Transform(mat2x3(trans * scale * rot));
  auto b_copy = CrossSection(b);

  auto ex_b = Manifold::Extrude(b.ToPolygons(), 1.).GetMeshGL();
  Identical(Manifold::Extrude(a.ToPolygons(), 1.).GetMeshGL(), ex_b);

  // same transformations are applied in b_copy (giving same result)
  Identical(ex_b, Manifold::Extrude(b_copy.ToPolygons(), 1.).GetMeshGL());
}

TEST(CrossSection, Warp) {
  auto sq = CrossSection::Square({10., 10.});
  auto a = sq.Scale({2, 3}).Translate({4, 5});
  auto b = sq.Warp([](vec2& v) {
    v.x = v.x * 2 + 4;
    v.y = v.y * 3 + 5;
  });

  EXPECT_EQ(sq.NumVert(), 4);
  EXPECT_EQ(sq.NumContour(), 1);
}

TEST(CrossSection, Decompose) {
  auto a = CrossSection::Square({2., 2.}, true) -
           CrossSection::Square({1., 1.}, true);
  auto b = a.Translate({4, 4});
  auto ab = a + b;
  auto decomp = ab.Decompose();
  auto recomp = CrossSection::Compose(decomp);

  EXPECT_EQ(decomp.size(), 2);
  EXPECT_EQ(decomp[0].NumContour(), 2);
  EXPECT_EQ(decomp[1].NumContour(), 2);

  Identical(Manifold::Extrude(a.ToPolygons(), 1.).GetMeshGL(),
            Manifold::Extrude(decomp[0].ToPolygons(), 1.).GetMeshGL());
  Identical(Manifold::Extrude(b.ToPolygons(), 1.).GetMeshGL(),
            Manifold::Extrude(decomp[1].ToPolygons(), 1.).GetMeshGL());
  Identical(Manifold::Extrude(ab.ToPolygons(), 1.).GetMeshGL(),
            Manifold::Extrude(recomp.ToPolygons(), 1.).GetMeshGL());
}

TEST(CrossSection, DecomposeNestedHoleAndIsland) {
  SimplePolygon outer = {{-5, -5}, {5, -5}, {5, 5}, {-5, 5}};
  SimplePolygon hole = {{-3, -3}, {-3, 3}, {3, 3}, {3, -3}};
  SimplePolygon island = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  CrossSection input({outer, hole, island}, CrossSection::FillRule::Positive);

  auto components = input.Decompose();
  ASSERT_EQ(components.size(), 2);

  double totalArea = 0.0;
  size_t donutCount = 0;
  size_t islandCount = 0;
  for (const auto& component : components) {
    totalArea += component.Area();
    if (component.NumContour() == 2) {
      ++donutCount;
    } else if (component.NumContour() == 1) {
      ++islandCount;
    }
  }

  EXPECT_EQ(donutCount, 1);
  EXPECT_EQ(islandCount, 1);
  EXPECT_NEAR(totalArea, input.Area(), 1e-9);
  EXPECT_NEAR(CrossSection::Compose(components).Area(), input.Area(), 1e-9);
}

TEST(CrossSection, WarpAffineMatchesScaleTranslate) {
  CrossSection sq = CrossSection::Square({10, 10}, true);
  CrossSection scaled = sq.Scale({2, 3}).Translate({5, -1});
  CrossSection warped =
      sq.Warp([](vec2& v) { v = {2 * v.x + 5, 3 * v.y - 1}; });
  EXPECT_EQ(warped.NumVert(), scaled.NumVert());
  EXPECT_NEAR(warped.Area(), scaled.Area(), 1e-6);
  EXPECT_NEAR(warped.Bounds().min.x, scaled.Bounds().min.x, 1e-6);
}

TEST(CrossSection, ShearTransformPreservesArea) {
  CrossSection sq = CrossSection::Square({10, 10}, true);
  CrossSection sheared = sq.Transform(mat2x3({1, 0}, {0.5, 1}, {0, 0}));
  CrossSection warped = sq.Warp([](vec2& v) { v.x += 0.5 * v.y; });
  EXPECT_NEAR(sheared.Area(), 100.0, 1e-6);  // shear preserves area
  EXPECT_NEAR(sheared.Area(), warped.Area(), 1e-6);
}

// Native-only: the WASM build is single-threaded (no pthread), so std::thread
// construction throws, which aborts under -fno-exceptions.
#ifndef __EMSCRIPTEN__
TEST(CrossSection, ConcurrentConstAccessorsAreRaceFree) {
  // Meaningful under TSan (the upstream continuous-fuzz lane builds with it):
  // 16 threads materialize the lazy transform and read const accessors.
  const CrossSection cs = CrossSection::Circle(10, 64).Translate({1e6, -1e6});
  std::array<double, 16> results{};
  std::vector<std::thread> threads;
  for (int i = 0; i < 16; ++i) {
    threads.emplace_back(
        [&, i] { results[i] = cs.Area() + cs.Bounds().Size().x; });
  }
  for (auto& t : threads) t.join();
  for (double r : results) EXPECT_EQ(r, results[0]);
}
#endif

TEST(CrossSection, PublicRoundOffsetRoundTripsMismatchCount) {
  const int segments = 420;
  CrossSection square = CrossSection::Square({20, 20}, true);

  CrossSection rounded =
      square.Offset(5.0, CrossSection::JoinType::Round, 2.0, segments);

  EXPECT_EQ(rounded.NumVert(), segments + 4);
}

TEST(CrossSection, OffsetRoundNonFiniteAndHugeDelta) {
  // Round offset derives its default segment count from |delta| via
  // Quality::GetCircularSegments, whose internal 2*pi*r/edge -> int conversion
  // would be undefined for non-finite or very large radii; it is guarded at the
  // source (double, not int, intermediate). Non-finite delta is rejected by
  // Offset (empty); huge finite delta still yields a finite bounded result.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  CrossSection square = CrossSection::Square({10, 10}, true);

  EXPECT_EQ(square.Offset(nan, CrossSection::JoinType::Round).NumVert(), 0u);
  EXPECT_EQ(square.Offset(inf, CrossSection::JoinType::Round).NumVert(), 0u);

  CrossSection huge = square.Offset(1e9, CrossSection::JoinType::Round);
  EXPECT_GT(huge.NumVert(), 0u);
  for (const SimplePolygon& ring : huge.ToPolygons())
    for (const vec2& v : ring) {
      EXPECT_TRUE(std::isfinite(v.x));
      EXPECT_TRUE(std::isfinite(v.y));
    }
}

TEST(CrossSection, OffsetDoesNotInflateToleranceDownstream) {
  // A Round offset must not fold its round-join faceting sagitta into
  // tolerance_: that would silently over-merge features finer than the sagitta
  // in the next boolean. The offset output must behave like a fresh
  // reconstruction of the same paths.
  CrossSection sq = CrossSection::Square({100, 100}, true);
  CrossSection rounded = sq.Offset(50.0, CrossSection::JoinType::Round);
  CrossSection fresh(rounded.ToPolygons());  // same geometry, fresh tolerance
  // A thin sliver (~7.5 area) far below the faceting sagitta (~0.19); folding
  // that sagitta into tolerance_ on `rounded` would over-merge it, diverging
  // from fresh.
  CrossSection slivR = rounded - rounded.Scale({0.9999, 0.9999});
  CrossSection slivF = fresh - fresh.Scale({0.9999, 0.9999});
  ASSERT_GT(slivF.Area(), 1.0);  // the sliver is a real feature
  EXPECT_NEAR(slivR.Area(), slivF.Area(), 1e-6);
  EXPECT_EQ(slivR.NumVert(), slivF.NumVert());
}

TEST(CrossSection, PublicOffsetSquareInsetCapStaysOnSolidSide) {
  SimplePolygon L = {{0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}};

  CrossSection out(L);
  out = out.Offset(-0.25, CrossSection::JoinType::Square);

  for (const SimplePolygon& ring : out.ToPolygons()) {
    for (const vec2& v : ring) {
      const bool inNotch = v.x > 1.0 && v.x < 2.0 && v.y > 1.0 && v.y < 2.0;
      EXPECT_FALSE(inNotch) << "cap vertex (" << v.x << ", " << v.y
                            << ") is in the removed notch";
    }
  }
  EXPECT_LT(out.Area(), 1.35);
}

TEST(CrossSection, HullDegenerateInputIsEmpty) {
  // Collinear/coincident points have no 2D hull: empty, not a degenerate
  // 2-vertex zero-area contour that would extrude to an invalid solid.
  EXPECT_TRUE(
      CrossSection::Hull(SimplePolygon{{0, 0}, {1, 1}, {2, 2}}).IsEmpty());
  EXPECT_TRUE(
      CrossSection::Hull(SimplePolygon{{5, 5}, {5, 5}, {5, 5}}).IsEmpty());
  EXPECT_TRUE(CrossSection::Hull(SimplePolygon{{0, 0}, {1, 1}}).IsEmpty());
}

TEST(CrossSection, TransformNonFiniteIsNoOp) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CrossSection sq = CrossSection::Square({10, 10}, true);
  CrossSection t = sq.Transform(mat2x3({nan, 0}, {0, 1}, {0, 0}));
  EXPECT_NEAR(t.Area(), sq.Area(), 1e-9);  // garbage transform -> no-op
  EXPECT_TRUE(std::isfinite(t.Bounds().Size().x));
  EXPECT_TRUE(
      CrossSection::Hull(SimplePolygon{{0, 0}, {1, 0}, {nan, 1}}).IsEmpty());
}

TEST(CrossSection, SquareDegenerateDimIsEmpty) {
  // boolean2 returns empty for zero-extent inputs (clipper2 emits degenerate
  // contours here); pin the intended behavior.
  EXPECT_TRUE(CrossSection::Square({5, 0}, true).IsEmpty());
  EXPECT_TRUE(CrossSection::Square({0, 5}, false).IsEmpty());
  EXPECT_TRUE(CrossSection::Circle(0.0).IsEmpty());
  EXPECT_TRUE(CrossSection::BatchBoolean({}, OpType::Add).IsEmpty());
}

TEST(CrossSection, FillRule) {
  SimplePolygon polygon = {
      {-7, 13},   //
      {-7, 12},   //
      {-5, 9},    //
      {-5, 8.1},  //
      {-4.8, 8},  //
  };

  // CrossSection regularizes every input with the Positive fill rule; the
  // FillRule argument is retained for API compatibility but only Positive is
  // supported, so non-Positive rules are not exercised here.
  CrossSection positive(polygon);
  EXPECT_NEAR(positive.Area(), 0.683, 0.001);
}

TEST(CrossSection, Hull) {
  auto circ = CrossSection::Circle(10, 360);
  auto circs = std::vector<CrossSection>{circ, circ.Translate({0, 30}),
                                         circ.Translate({30, 0})};
  auto circ_tri = CrossSection::Hull(circs);
  auto centres = SimplePolygon{{0, 0}, {0, 30}, {30, 0}, {15, 5}};
  auto tri = CrossSection::Hull(centres);

  if (options.exportModels) {
    auto circ_tri_ex = Manifold::Extrude(circ_tri.ToPolygons(), 10);
    WriteTestOBJ("cross_section_hull_circ_tri.obj", circ_tri_ex);
  }

  auto circ_area = circ.Area();
  EXPECT_FLOAT_EQ(circ_area, (circ - circ.Scale({0.8, 0.8})).Hull().Area());
  EXPECT_FLOAT_EQ(
      circ_area * 2.5,
      (CrossSection::BatchBoolean(circs, OpType::Add) - tri).Area());
}

TEST(CrossSection, HullError) {
  auto rounded_rectangle = [](double x, double y, double radius, int segments) {
    auto circ = CrossSection::Circle(radius, segments);
    std::vector<CrossSection> vl{};
    vl.push_back(circ.Translate(vec2{radius, radius}));
    vl.push_back(circ.Translate(vec2{x - radius, radius}));
    vl.push_back(circ.Translate(vec2{x - radius, y - radius}));
    vl.push_back(circ.Translate(vec2{radius, y - radius}));
    return CrossSection::Hull(vl);
  };
  auto rr = rounded_rectangle(51, 36, 9.0, 36);

  auto rr_area = rr.Area();
  auto rr_verts = rr.NumVert();
  EXPECT_FLOAT_EQ(rr_area, 1765.1790375559026);
  EXPECT_FLOAT_EQ(rr_verts, 40);
}

TEST(CrossSection, BatchBoolean) {
  CrossSection square = CrossSection::Square({100, 100});
  CrossSection circle1 = CrossSection::Circle(30, 30).Translate({-10, 30});
  CrossSection circle2 = CrossSection::Circle(20, 30).Translate({110, 20});
  CrossSection circle3 = CrossSection::Circle(40, 30).Translate({50, 110});

  CrossSection intersect = CrossSection::BatchBoolean(
      {square, circle1, circle2, circle3}, OpType::Intersect);

  EXPECT_FLOAT_EQ(intersect.Area(), 0);
  EXPECT_FLOAT_EQ(intersect.NumVert(), 0);

  CrossSection add = CrossSection::BatchBoolean(
      {square, circle1, circle2, circle3}, OpType::Add);

  CrossSection subtract = CrossSection::BatchBoolean(
      {square, circle1, circle2, circle3}, OpType::Subtract);

  EXPECT_FLOAT_EQ(add.Area(), 16278.637002);
  EXPECT_FLOAT_EQ(add.NumVert(), 66);

  EXPECT_FLOAT_EQ(subtract.Area(), 7234.478452);
  EXPECT_FLOAT_EQ(subtract.NumVert(), 42);
}

// A is a 37-vertex star with many zero-radius vertices producing a dense
// near-degenerate shape; B is a 5-vertex star translated by ~5e-5 in x. The
// intersection A ∩ B returns area=0 even though both inputs are non-empty and
// overlap geometrically. (A - B) + (A ∩ B) ends up at 39605 versus
// a.Area=90895, missing ~51290 area units. The tiny x translation puts B's
// vertices just inside the eps band of A's; the intersect path appears to
// mis-classify the overlap region as empty.
TEST(CrossSection, SubtractInvariantsEmptyIntersectionDrop) {
  const std::vector<double> rA = Runs({{1, 1000.},
                                       {1, 0.},
                                       {2, 1000.},
                                       {1, 68.955135217866953},
                                       {1, 0.},
                                       {1, 1.},
                                       {9, 0.},
                                       {1, 805.7907155879783},
                                       {1, 1.},
                                       {5, 0.},
                                       {1, 4.7952714181195555},
                                       {3, 0.},
                                       {1, 1.2064064151490697},
                                       {1, 0.},
                                       {1, 3.2013961172280205},
                                       {3, 0.},
                                       {1, 13.934542696929423},
                                       {1, 363.29755584701081},
                                       {2, 0.}});
  const std::vector<double> rB = {1000., 731.51526963704362, 680.25141414501888,
                                  1000., 571.57429580441226};
  const CrossSection a(StarRing(rA));
  const CrossSection b =
      CrossSection(StarRing(rB)).Translate({-4.725175119801861e-05, -0.0});
  const auto aMinusB = a - b;
  const auto aIntersectB = a.Boolean(b, OpType::Intersect);
  const double tol = AreaTol(a, b);
  EXPECT_NEAR(aMinusB.Area() + aIntersectB.Area(), a.Area(), tol)
      << "area(A - B) + area(A ∩ B) != area(A)";
}

TEST(CrossSection, BooleanOperatorAssignments) {
  CrossSection a = CrossSection::Square({10, 10});
  CrossSection b = CrossSection::Square({10, 10}).Translate({5, 0});

  EXPECT_NEAR((a + b).Area(), 150.0, 1e-9);
  EXPECT_NEAR((a - b).Area(), 50.0, 1e-9);
  EXPECT_NEAR((a ^ b).Area(), 50.0, 1e-9);

  CrossSection add = a;
  add += b;
  EXPECT_NEAR(add.Area(), 150.0, 1e-9);

  CrossSection subtract = a;
  subtract -= b;
  EXPECT_NEAR(subtract.Area(), 50.0, 1e-9);

  CrossSection intersect = a;
  intersect ^= b;
  EXPECT_NEAR(intersect.Area(), 50.0, 1e-9);
}

TEST(CrossSection, NegativeOffset) {
  CrossSection plusSign = CrossSection::Square({30, 50}, true) +
                          CrossSection::Square({50, 30}, true);
  CrossSection dilated =
      plusSign.Offset(-10, CrossSection::JoinType::Round, 2.0, 1024);
  EXPECT_NEAR(dilated.Area(), 30 * 30 - 10 * 10 * kPi, 0.01);
}

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
#include <vector>

#include "manifold/common.h"
#include "manifold/manifold.h"
#include "test.h"

using namespace manifold;

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

TEST(CrossSection, FillRule) {
  SimplePolygon polygon = {
      {-7, 13},   //
      {-7, 12},   //
      {-5, 9},    //
      {-5, 8.1},  //
      {-4.8, 8},  //
  };

  CrossSection positive(polygon);
  EXPECT_NEAR(positive.Area(), 0.683, 0.001);

  CrossSection negative(polygon, CrossSection::FillRule::Negative);
  EXPECT_NEAR(negative.Area(), 0.193, 0.001);

  CrossSection evenOdd(polygon, CrossSection::FillRule::EvenOdd);
  EXPECT_NEAR(evenOdd.Area(), 0.875, 0.001);

  CrossSection nonZero(polygon, CrossSection::FillRule::NonZero);
  EXPECT_NEAR(nonZero.Area(), 0.875, 0.001);
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

TEST(CrossSection, NegativeOffset) {
  CrossSection plusSign = CrossSection::Square({30, 50}, true) +
                          CrossSection::Square({50, 30}, true);
  CrossSection dilated =
      plusSign.Offset(-10, CrossSection::JoinType::Round, 2.0, 1024);
  EXPECT_NEAR(dilated.Area(), 30 * 30 - 10 * 10 * kPi, 0.01);
}

namespace {

double NormalizeAngle(double angle) {
  while (angle < 0) angle += 2.0 * kPi;
  while (angle >= 2.0 * kPi) angle -= 2.0 * kPi;
  return angle;
}

void ExpectFilletAngles(const CrossSection& crossSection, int circularSegments) {
  const double dPhi = 2.0 * kPi / circularSegments;
  for (const SimplePolygon& loop : crossSection.ToPolygons()) {
    ASSERT_GE(loop.size(), 3);
    for (size_t i = 0; i < loop.size(); ++i) {
      const vec2 p1 = loop[i];
      const vec2 p2 = loop[(i + 1) % loop.size()];
      const vec2 p3 = loop[(i + 2) % loop.size()];
      const vec2 e1 = p2 - p1;
      const vec2 e2 = p3 - p2;

      EXPECT_TRUE(std::isfinite(p1.x));
      EXPECT_TRUE(std::isfinite(p1.y));
      EXPECT_GT(length2(e1), 0);
      EXPECT_GT(length2(e2), 0);

      const double angle =
          NormalizeAngle(std::atan2(e2.y, e2.x) - std::atan2(e1.y, e1.x));
      EXPECT_TRUE(angle > kPi || angle < dPhi + dPhi * 0.1);
    }
  }
}

void ExpectValidLoops(const CrossSection& crossSection) {
  for (const SimplePolygon& loop : crossSection.ToPolygons()) {
    ASSERT_GE(loop.size(), 3);
    for (size_t i = 0; i < loop.size(); ++i) {
      const vec2 p1 = loop[i];
      const vec2 p2 = loop[(i + 1) % loop.size()];

      EXPECT_TRUE(std::isfinite(p1.x));
      EXPECT_TRUE(std::isfinite(p1.y));
      EXPECT_GT(length2(p2 - p1), 0);
    }
  }
}

CrossSection ExpectFilletResult(const CrossSection& input, double radius,
                                int circularSegments) {
  const std::vector<CrossSection> pieces = input.Fillet(radius, circularSegments);
  const CrossSection composed = CrossSection::Compose(pieces);

  EXPECT_FALSE(pieces.empty());
  EXPECT_FALSE(composed.IsEmpty());
  if (radius < 0) {
    EXPECT_GT(composed.Area(), input.Area());
  } else {
    EXPECT_LT(composed.Area(), input.Area());
  }
  EXPECT_GT(std::abs(composed.Area() - input.Area()), 1e-9);

  for (const CrossSection& piece : pieces) {
    ExpectValidLoops(piece);
    if (radius > 0) ExpectFilletAngles(piece, circularSegments);
  }
  ExpectValidLoops(composed);
  if (radius > 0) ExpectFilletAngles(composed, circularSegments);
  return composed;
}

}  // namespace

TEST(CrossSection, FilletPositiveEESquareProducesGeometry) {
  CrossSection input = CrossSection::Square({10, 10});
  CrossSection filleted = ExpectFilletResult(input, 1, 16);

  EXPECT_GT(filleted.NumVert(), input.NumVert());
}

TEST(CrossSection, FilletPositiveEVAndVEConcaveL) {
  CrossSection input =
      CrossSection(Polygons{{{0, 0}, {6, 0}, {6, 2}, {2, 2}, {2, 6}, {0, 6}}});
  CrossSection filleted = ExpectFilletResult(input, 1, 20);

  EXPECT_GT(filleted.NumVert(), input.NumVert());
}

TEST(CrossSection, FilletPositiveVVStar) {
  CrossSection input = CrossSection(Polygons{{{3.18993, 0},
                                              {0.549068, 0.951013},
                                              {-1.47393, 2.55292},
                                              {-1.1997, 0},
                                              {-2.47915, -4.29402},
                                              {0.942809, -1.63299}}});
  CrossSection filleted = ExpectFilletResult(input, 1.2, 20);

  EXPECT_GT(filleted.NumVert(), input.NumVert());
}

TEST(CrossSection, FilletNegativeRadiusRoundsConcaveCorners) {
  CrossSection input =
      CrossSection(Polygons{{{0, 0},
                             {6, 0},
                             {6, 6},
                             {4, 6},
                             {4, 2},
                             {2, 2},
                             {2, 6},
                             {0, 6}}});
  CrossSection filleted = ExpectFilletResult(input, -1, 20);

  EXPECT_GT(filleted.NumVert(), input.NumVert());
}

TEST(CrossSection, FilletTooFlatForSegmentCount) {
  CrossSection input = CrossSection::Square({10, 10});
  std::vector<CrossSection> filleted = input.Fillet(1, 3);

  EXPECT_TRUE(filleted.empty());
}

TEST(CrossSection, FilletDeterministic) {
  CrossSection input = CrossSection::Square({10, 10});
  CrossSection a = CrossSection::Compose(input.Fillet(1, 16));
  CrossSection b = CrossSection::Compose(input.Fillet(1, 16));

  EXPECT_EQ(a.NumContour(), b.NumContour());
  EXPECT_EQ(a.NumVert(), b.NumVert());
  EXPECT_DOUBLE_EQ(a.Area(), b.Area());
}

TEST(CrossSection, FilletRadiusFitRejectsOversizedTriangle) {
  CrossSection input = CrossSection({{0, 0}, {10, 0}, {0, 10}});
  std::vector<CrossSection> filleted = input.Fillet(4, 16);

  EXPECT_TRUE(filleted.empty());
}

TEST(CrossSection, FilletNearColinearRejected) {
  CrossSection input = CrossSection({{0, 0}, {10, 0}, {20, 1e-9}});
  std::vector<CrossSection> filleted = input.Fillet(0.1, 16);

  EXPECT_TRUE(filleted.empty());
}

TEST(CrossSection, FilletNegativeRadiusStrictlyRejectedOnConvexSquare) {
  CrossSection input = CrossSection::Square({10, 10});
  std::vector<CrossSection> filleted = input.Fillet(-1, 16);

  EXPECT_TRUE(filleted.empty());
}

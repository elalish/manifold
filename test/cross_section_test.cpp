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

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("cross_section_mirror_union.glb", result.GetMeshGL(), {});
#endif

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

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("cross_section_round_offset.glb", result.GetMeshGL(), {});
#endif

  EXPECT_EQ(result.Genus(), 0);
  EXPECT_NEAR(result.Volume(), 4386, 1);
  EXPECT_EQ(rounded.NumVert(), segments + 4);
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

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    auto circ_tri_ex = Manifold::Extrude(circ_tri.ToPolygons(), 10);
    ExportMesh("cross_section_hull_circ_tri.glb", circ_tri_ex.GetMeshGL(), {});
  }
#endif

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

  EXPECT_EQ(intersect.NumVert(), 0);

  CrossSection add = CrossSection::BatchBoolean(
      {square, circle1, circle2, circle3}, OpType::Add);

  Polygons r1 = add.ToPolygons();
  SimplePolygon table1 = {{100, 2.67949},
                          {103.82, 0.97887},
                          {107.909, 0.109562},
                          {112.091, 0.109562},
                          {116.18, 0.97887},
                          {120, 2.67949},
                          {123.383, 5.1371},
                          {126.18, 8.24429},
                          {128.271, 11.8653},
                          {129.563, 15.8418},
                          {130, 20},
                          {129.563, 24.1582},
                          {128.271, 28.1347},
                          {126.18, 31.7557},
                          {123.383, 34.8629},
                          {120, 37.3205},
                          {116.18, 39.0211},
                          {112.091, 39.8904},
                          {107.909, 39.8904},
                          {103.82, 39.0211},
                          {100, 37.3205},
                          {100, 100},
                          {88.5789, 100},
                          {89.1259, 101.684},
                          {90, 110},
                          {89.1259, 118.316},
                          {86.5418, 126.269},
                          {82.3607, 133.511},
                          {76.7652, 139.726},
                          {70, 144.641},
                          {62.3607, 148.042},
                          {54.1811, 149.781},
                          {45.8189, 149.781},
                          {37.6393, 148.042},
                          {30, 144.641},
                          {23.2348, 139.726},
                          {17.6393, 133.511},
                          {13.4582, 126.269},
                          {10.8741, 118.316},
                          {10, 110},
                          {10.8741, 101.684},
                          {11.4211, 100},
                          {0, 100},
                          {0, 58.2069},
                          {-0.72949, 58.5317},
                          {-6.86415, 59.8357},
                          {-13.1359, 59.8357},
                          {-19.2705, 58.5317},
                          {-25, 55.9808},
                          {-30.0739, 52.2943},
                          {-34.2705, 47.6336},
                          {-37.4064, 42.2021},
                          {-39.3444, 36.2374},
                          {-40, 30},
                          {-39.3444, 23.7626},
                          {-37.4064, 17.7979},
                          {-34.2705, 12.3664},
                          {-30.0739, 7.70566},
                          {-25, 4.01924},
                          {-19.2705, 1.4683},
                          {-13.1359, 0.164343},
                          {-6.86415, 0.164343},
                          {-0.72949, 1.4683},
                          {0, 1.79309},
                          {0, 0},
                          {100, 0}};

  EXPECT_EQ(r1[0].size(), table1.size());

  bool flag = true;

  for (size_t i = 0; i != r1[0].size(); i++) {
    if (linalg::distance2(r1[0][i], table1[i]) > 0.001) {
      flag = false;
      break;
    }
  }

  EXPECT_TRUE(flag);

  CrossSection subtract = CrossSection::BatchBoolean(
      {square, circle1, circle2, circle3}, OpType::Subtract);

  Polygons r2 = subtract.ToPolygons();
  SimplePolygon table2 = {{100, 2.67949},
                          {96.6174, 5.1371},
                          {93.8197, 8.24429},
                          {91.7291, 11.8653},
                          {90.437, 15.8418},
                          {90, 20},
                          {90.437, 24.1582},
                          {91.7291, 28.1347},
                          {93.8197, 31.7557},
                          {96.6174, 34.8629},
                          {100, 37.3205},
                          {100, 100},
                          {88.5789, 100},
                          {86.5418, 93.7305},
                          {82.3607, 86.4886},
                          {76.7652, 80.2742},
                          {70, 75.359},
                          {62.3607, 71.9577},
                          {54.1811, 70.2191},
                          {45.8189, 70.2191},
                          {37.6393, 71.9577},
                          {30, 75.359},
                          {23.2348, 80.2742},
                          {17.6393, 86.4886},
                          {13.4582, 93.7305},
                          {11.4211, 100},
                          {0, 100},
                          {0, 58.2069},
                          {5, 55.9808},
                          {10.0739, 52.2943},
                          {14.2705, 47.6336},
                          {17.4064, 42.2021},
                          {19.3444, 36.2374},
                          {20, 30},
                          {19.3444, 23.7626},
                          {17.4064, 17.7979},
                          {14.2705, 12.3664},
                          {10.0739, 7.70566},
                          {5, 4.01924},
                          {0, 1.79309},
                          {0, 0},
                          {100, 0}};

  EXPECT_EQ(r2[0].size(), table2.size());

  for (size_t i = 0; flag && i != r2[0].size(); i++) {
    if (linalg::distance2(r2[0][i], table2[i]) > 0.001) {
      flag = false;
      break;
    }
  }

  EXPECT_TRUE(flag);
}
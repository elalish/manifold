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

#include "cross_section.h"

#include <gtest/gtest.h>

#include <vector>

#include "glm/geometric.hpp"
#include "manifold.h"
#include "polygon.h"
#include "public.h"
#include "test.h"

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

using namespace manifold;

TEST(CrossSection, Square) {
  auto a = Manifold::Cube({5, 5, 5});
  auto b = Manifold::Extrude(CrossSection::Square({5, 5}), 5);

  EXPECT_FLOAT_EQ((a - b).GetProperties().volume, 0.);
}

TEST(CrossSection, MirrorUnion) {
  auto a = CrossSection::Square({5., 5.}, true);
  auto b = a.Translate({2.5, 2.5});
  auto cross = a + b + b.Mirror({1, 1});
  auto result = Manifold::Extrude(cross, 5.);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("cross_section_mirror_union.glb", result.GetMesh(), {});
#endif

  EXPECT_FLOAT_EQ(2.5 * a.Area(), cross.Area());
  EXPECT_TRUE(a.Mirror(glm::vec2(0)).IsEmpty());
}

TEST(CrossSection, RoundOffset) {
  auto a = CrossSection::Square({20., 20.}, true);
  int segments = 20;
  auto rounded = a.Offset(5., CrossSection::JoinType::Round, 2, segments);
  auto result = Manifold::Extrude(rounded, 5.);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("cross_section_round_offset.glb", result.GetMesh(), {});
#endif

  EXPECT_EQ(result.Genus(), 0);
  EXPECT_NEAR(result.GetProperties().volume, 4386, 1);
  EXPECT_EQ(rounded.NumVert(), segments + 4);
}

TEST(CrossSection, Empty) {
  Polygons polys(2);
  auto e = CrossSection(polys);
  EXPECT_TRUE(e.IsEmpty());
}

TEST(CrossSection, Rect) {
  float w = 10;
  float h = 5;
  auto rect = Rect({0, 0}, {w, h});
  auto cross = rect.AsCrossSection();
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

  glm::mat3x3 trans(1.0f, 0.0f, 0.0f,  //
                    0.0f, 1.0f, 0.0f,  //
                    4.0f, 5.0f, 1.0f);
  glm::mat3x3 rot(cosd(45), sind(45), 0.0f,   //
                  -sind(45), cosd(45), 0.0f,  //
                  0.0f, 0.0f, 1.0f);
  glm::mat3x3 scale(2.0f, 0.0f, 0.0f,  //
                    0.0f, 3.0f, 0.0f,  //
                    0.0f, 0.0f, 1.0f);

  auto b = sq.Transform(trans * scale * rot);
  auto b_copy = CrossSection(b);

  auto ex_b = Manifold::Extrude(b, 1.).GetMesh();
  Identical(Manifold::Extrude(a, 1.).GetMesh(), ex_b);

  // same transformations are applied in b_copy (giving same result)
  Identical(ex_b, Manifold::Extrude(b_copy, 1.).GetMesh());
}

TEST(CrossSection, Warp) {
  auto sq = CrossSection::Square({10., 10.});
  auto a = sq.Scale({2, 3}).Translate({4, 5});
  auto b = sq.Warp([](glm::vec2 &v) {
    v.x = v.x * 2 + 4;
    v.y = v.y * 3 + 5;
  });

  EXPECT_EQ(sq.NumVert(), 4);
  EXPECT_EQ(sq.NumContour(), 1);
  Identical(Manifold::Extrude(a, 1.).GetMesh(),
            Manifold::Extrude(b, 1.).GetMesh());
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

  Identical(Manifold::Extrude(a, 1.).GetMesh(),
            Manifold::Extrude(decomp[0], 1.).GetMesh());
  Identical(Manifold::Extrude(b, 1.).GetMesh(),
            Manifold::Extrude(decomp[1], 1.).GetMesh());
  Identical(Manifold::Extrude(ab, 1.).GetMesh(),
            Manifold::Extrude(recomp, 1.).GetMesh());
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
    auto circ_tri_ex = Manifold::Extrude(circ_tri, 10);
    ExportMesh("cross_section_hull_circ_tri.glb", circ_tri_ex.GetMesh(), {});
  }
#endif

  auto circ_area = circ.Area();
  EXPECT_FLOAT_EQ(circ_area, (circ - circ.Scale({0.8, 0.8})).Hull().Area());
  EXPECT_FLOAT_EQ(
      circ_area * 2.5,
      (CrossSection::BatchBoolean(circs, OpType::Add) - tri).Area());
}

// Copyright 2021 Emmett Lalish
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

#include "samples.h"

#include "gtest/gtest.h"
#include "meshIO.h"
#include "polygon.h"

using namespace manifold;

// If you print this knot (with support), you can snap a half-inch marble into
// it and it'll roll around (dimensions in mm).
TEST(Samples, Knot13) {
  Manifold knot13 = TorusKnot(1, 3, 25, 10, 3.75);
  //   ExportMesh("knot13.stl", knot13.Extract(), {});
  EXPECT_TRUE(knot13.IsManifold());
  EXPECT_EQ(knot13.Genus(), 1);
  auto prop = knot13.GetProperties();
  EXPECT_NEAR(prop.volume, 20786, 1);
  EXPECT_NEAR(prop.surfaceArea, 11177, 1);
}

// This creates two interlinked knots.
TEST(Samples, Knot42) {
  Manifold knot42 = TorusKnot(4, 2, 15, 6, 5);
  //   ExportMesh("knot42.stl", knot42.Extract(), {});
  EXPECT_TRUE(knot42.IsManifold());
  EXPECT_TRUE(knot42.StrictlyMatchesTriNormals());
  std::vector<Manifold> knots = knot42.Decompose();
  ASSERT_EQ(knots.size(), 2);
  EXPECT_EQ(knots[0].Genus(), 1);
  EXPECT_EQ(knots[1].Genus(), 1);
  auto prop0 = knots[0].GetProperties();
  auto prop1 = knots[1].GetProperties();
  EXPECT_NEAR(prop0.volume, prop1.volume, 1);
  EXPECT_NEAR(prop0.surfaceArea, prop1.surfaceArea, 1);
}

TEST(Samples, Scallop) {
  Manifold scallop = Scallop();
  scallop.Refine(100);
  auto prop = scallop.GetProperties();
  EXPECT_NEAR(prop.volume, 41.3, 0.1);
  EXPECT_NEAR(prop.surfaceArea, 81.2, 0.1);

  // const Mesh out = scallop.Extract();
  // ExportOptions options;
  // options.faceted = false;
  // options.mat.roughness = 0.1;
  // const glm::vec4 blue(0, 0, 1, 1);
  // const glm::vec4 red(1, 0, 0, 1);
  // const float limit = 15;
  // for (float curvature : scallop.GetCurvature().vertMeanCurvature) {
  //   options.mat.vertColor.push_back(
  //       glm::mix(blue, red, glm::smoothstep(-limit, limit, curvature)));
  // }
  // ExportMesh("scallop.gltf", out, options);
}

TEST(Samples, TetPuzzle) {
  Manifold puzzle = TetPuzzle(50, 0.2, 50);
  EXPECT_TRUE(puzzle.IsManifold());
  EXPECT_TRUE(puzzle.StrictlyMatchesTriNormals());
  Manifold puzzle2 = puzzle;
  puzzle2.Rotate(0, 0, 180);
  EXPECT_TRUE((puzzle ^ puzzle2).IsEmpty());
  puzzle.Transform(RotateUp({1, -1, -1}));

  // const Mesh out = puzzle.Extract();
  // ExportOptions options;
  // options.faceted = false;
  // options.mat.vertColor.resize(puzzle.NumVert());
  // for (const glm::ivec3 tri : out.triVerts) {
  //   glm::vec3 v1 = out.vertPos[tri[1]] - out.vertPos[tri[0]];
  //   glm::vec3 v2 = out.vertPos[tri[2]] - out.vertPos[tri[0]];
  //   glm::vec3 crossP = glm::cross(v1, v2);
  //   float area2 = glm::dot(crossP, crossP);
  //   float base2 = glm::max(glm::dot(v1, v1), glm::dot(v2, v2));
  //   if (area2 < base2 * puzzle.Precision() * puzzle.Precision()) {
  //     std::cout << glm::normalize(crossP) << std::endl;
  //     for (int i : {0, 1, 2})
  //       std::cout << tri[i] << ", " << out.vertPos[tri[i]] << std::endl;
  //   }
  //   for (int i : {0, 1, 2}) {
  //     options.mat.vertColor[tri[i]] =
  //         area2 < base2 * puzzle.Precision() * puzzle.Precision()
  //             ? glm::vec4(1, 0, 0, 1)
  //             : glm::vec4(0, 1, 1, 1);
  //   }
  // }
  // ExportMesh("tetPuzzle.gltf", out, options);
}

TEST(Samples, FrameReduced) {
  Manifold::SetCircularSegments(4);
  Manifold frame = RoundedFrame(100, 10);
  EXPECT_TRUE(frame.IsManifold());
  EXPECT_TRUE(frame.StrictlyMatchesTriNormals());
  Manifold::SetCircularSegments(0);
  EXPECT_EQ(frame.Genus(), 5);
  auto prop = frame.GetProperties();
  EXPECT_NEAR(prop.volume, 227333, 10);
  EXPECT_NEAR(prop.surfaceArea, 62635, 1);
  // ExportMesh("roundedFrameReduced.gltf", frame.Extract(), {});
}

TEST(Samples, Frame) {
  Manifold frame = RoundedFrame(100, 10);
  EXPECT_TRUE(frame.IsManifold());
  EXPECT_TRUE(frame.StrictlyMatchesTriNormals());
  EXPECT_EQ(frame.Genus(), 5);
  // ExportMesh("roundedFrame.ply", frame.Extract(), {});
}

// This creates a bracelet sample which involves many operations between shapes
// that are not in general position, e.g. coplanar faces.
TEST(Samples, Bracelet) {
  Manifold bracelet = StretchyBracelet();
  EXPECT_TRUE(bracelet.IsManifold());
  EXPECT_TRUE(bracelet.StrictlyMatchesTriNormals());
  EXPECT_EQ(bracelet.Genus(), 1);
  // ExportMesh("bracelet.ply", bracelet.Extract(), {});
}

TEST(Samples, Sponge1) {
  Manifold sponge = MengerSponge(1);
  EXPECT_TRUE(sponge.IsManifold());
  EXPECT_TRUE(sponge.StrictlyMatchesTriNormals());
  EXPECT_EQ(sponge.Genus(), 5);
  // ExportMesh("mengerSponge1.gltf", sponge.Extract(), {});
}

// A fractal with many degenerate intersections, which also tests exact 90
// degree rotations.
TEST(Samples, Sponge4) {
  Manifold sponge = MengerSponge(4);
  EXPECT_TRUE(sponge.IsManifold());
  EXPECT_TRUE(sponge.StrictlyMatchesTriNormals());
  EXPECT_EQ(sponge.Genus(), 26433);  // should be 1:5, 2:81, 3:1409, 4:26433
  // ExportMesh("mengerSponge.gltf", sponge.Extract(), {});
  std::pair<Manifold, Manifold> cutSponge = sponge.SplitByPlane({1, 1, 1}, 0);
  EXPECT_TRUE(cutSponge.first.IsManifold());
  EXPECT_EQ(cutSponge.first.Genus(), 13394);
  EXPECT_TRUE(cutSponge.second.IsManifold());
  EXPECT_EQ(cutSponge.second.Genus(), 13394);
  // ExportMesh("mengerSponge.ply", cutSponge.first.Extract(), {});
}

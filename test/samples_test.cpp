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

void CheckManifold(const Manifold& manifold) {
  EXPECT_TRUE(manifold.IsManifold());
  EXPECT_TRUE(manifold.MatchesTriNormals());
  for (const glm::vec3& normal : manifold.GetMesh().vertNormal) {
    ASSERT_NEAR(glm::length(normal), 1, 0.0001);
  }
}

// If you print this knot (with support), you can snap a half-inch marble into
// it and it'll roll around (dimensions in mm).
TEST(Samples, Knot13) {
  Manifold knot13 = TorusKnot(1, 3, 25, 10, 3.75);
  //   ExportMesh("knot13.stl", knot13.GetMesh(), {});
  CheckManifold(knot13);
  EXPECT_EQ(knot13.Genus(), 1);
  auto prop = knot13.GetProperties();
  EXPECT_NEAR(prop.volume, 20786, 1);
  EXPECT_NEAR(prop.surfaceArea, 11177, 1);
}

// This creates two interlinked knots.
TEST(Samples, Knot42) {
  Manifold knot42 = TorusKnot(4, 2, 15, 6, 5);
  //   ExportMesh("knot42.stl", knot42.GetMesh(), {});
  CheckManifold(knot42);
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
  CheckManifold(scallop);
  auto prop = scallop.GetProperties();
  EXPECT_NEAR(prop.volume, 41.3, 0.1);
  EXPECT_NEAR(prop.surfaceArea, 81.2, 0.1);

  // const Mesh out = scallop.GetMesh();
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
  CheckManifold(puzzle);
  EXPECT_LE(puzzle.NumDegenerateTris(), 2);
  Manifold puzzle2 = puzzle;
  puzzle2.Rotate(0, 0, 180);
  EXPECT_TRUE((puzzle ^ puzzle2).IsEmpty());
  puzzle.Transform(RotateUp({1, -1, -1}));
  // ExportMesh("tetPuzzle.gltf", puzzle.GetMesh(), {});
}

TEST(Samples, FrameReduced) {
  Manifold::SetCircularSegments(4);
  Manifold frame = RoundedFrame(100, 10);
  CheckManifold(frame);
  EXPECT_EQ(frame.NumDegenerateTris(), 0);
  Manifold::SetCircularSegments(0);
  EXPECT_EQ(frame.Genus(), 5);
  auto prop = frame.GetProperties();
  EXPECT_NEAR(prop.volume, 227333, 10);
  EXPECT_NEAR(prop.surfaceArea, 62635, 1);
  // ExportMesh("roundedFrameReduced.gltf", frame.GetMesh(), {});
}

TEST(Samples, Frame) {
  Manifold frame = RoundedFrame(100, 10);
  CheckManifold(frame);
  EXPECT_EQ(frame.NumDegenerateTris(), 0);
  EXPECT_EQ(frame.Genus(), 5);
  // ExportMesh("roundedFrame.ply", frame.GetMesh(), {});
}

// This creates a bracelet sample which involves many operations between shapes
// that are not in general position, e.g. coplanar faces.
TEST(Samples, Bracelet) {
  Manifold bracelet = StretchyBracelet();
  CheckManifold(bracelet);
  EXPECT_LE(bracelet.NumDegenerateTris(), 11);
  EXPECT_EQ(bracelet.Genus(), 1);
  // ExportMesh("bracelet.ply", bracelet.GetMesh(), {});
}

TEST(Samples, Sponge1) {
  Manifold sponge = MengerSponge(1);
  CheckManifold(sponge);
  EXPECT_EQ(sponge.NumDegenerateTris(), 0);
  EXPECT_EQ(sponge.NumVert(), 40);
  EXPECT_EQ(sponge.Genus(), 5);
  // ExportMesh("mengerSponge1.gltf", sponge.GetMesh(), {});
}

// A fractal with many degenerate intersections, which also tests exact 90
// degree rotations.
TEST(Samples, Sponge4) {
  Manifold sponge = MengerSponge(4);
  CheckManifold(sponge);
  EXPECT_EQ(sponge.NumDegenerateTris(), 0);
  EXPECT_EQ(sponge.Genus(), 26433);  // should be 1:5, 2:81, 3:1409, 4:26433
  // ExportMesh("mengerSponge.gltf", sponge.GetMesh(), {});
  std::pair<Manifold, Manifold> cutSponge = sponge.SplitByPlane({1, 1, 1}, 0);
  EXPECT_TRUE(cutSponge.first.IsManifold());
  EXPECT_EQ(cutSponge.first.Genus(), 13394);
  EXPECT_TRUE(cutSponge.second.IsManifold());
  EXPECT_EQ(cutSponge.second.Genus(), 13394);
  // ExportMesh("mengerSponge.ply", cutSponge.first.GetMesh(), {});
}

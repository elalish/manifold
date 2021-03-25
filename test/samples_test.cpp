// Copyright 2019 Emmett Lalish
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
  //   ExportMesh("knot13.stl", knot13.Extract());
  EXPECT_TRUE(knot13.IsManifold());
  EXPECT_EQ(knot13.Genus(), 1);
  auto prop = knot13.GetProperties();
  EXPECT_NEAR(prop.volume, 20786, 1);
  EXPECT_NEAR(prop.surfaceArea, 11177, 1);
}

// This creates two interlinked knots.
TEST(Samples, Knot42) {
  Manifold knot42 = TorusKnot(4, 2, 15, 6, 5);
  //   ExportMesh("knot42.stl", knot42.Extract());
  EXPECT_TRUE(knot42.IsManifold());
  std::vector<Manifold> knots = knot42.Decompose();
  ASSERT_EQ(knots.size(), 2);
  EXPECT_EQ(knots[0].Genus(), 1);
  EXPECT_EQ(knots[1].Genus(), 1);
  auto prop0 = knots[0].GetProperties();
  auto prop1 = knots[1].GetProperties();
  EXPECT_NEAR(prop0.volume, prop1.volume, 1);
  EXPECT_NEAR(prop0.surfaceArea, prop1.surfaceArea, 1);
}

// This creates a bracelet sample which involves many operations between shapes
// that are not in general position, e.g. coplanar faces.
TEST(Samples, Bracelet) {
  Manifold bracelet = StretchyBracelet();
  EXPECT_TRUE(bracelet.IsManifold());
  EXPECT_EQ(bracelet.Genus(), 1);
  // ExportMesh("bracelet.ply", bracelet.Extract());
}

// A fractal with many degenerate intersections, which also tests exact 90
// degree rotations.
TEST(Samples, Sponge) {
  Manifold sponge = MengerSponge(4);
  EXPECT_TRUE(sponge.IsManifold());
  EXPECT_TRUE(sponge.MatchesTriNormals());
  EXPECT_EQ(sponge.Genus(), 26433);  // should be 1:5, 2:81, 3:1409, 4:26433
  ExportMesh("mengerSponge.gltf", sponge.Extract());
  std::pair<Manifold, Manifold> cutSponge = sponge.SplitByPlane({1, 1, 1}, 0);
  EXPECT_TRUE(cutSponge.first.IsManifold());
  EXPECT_EQ(cutSponge.first.Genus(), 13394);
  EXPECT_TRUE(cutSponge.second.IsManifold());
  EXPECT_EQ(cutSponge.second.Genus(), 13394);
  // ExportMesh("mengerSponge.ply", cutSponge.first.Extract());
}

TEST(Samples, Sponge1) {
  Manifold sponge = MengerSponge(1);
  EXPECT_TRUE(sponge.IsManifold());
  EXPECT_TRUE(sponge.MatchesTriNormals());
  EXPECT_EQ(sponge.Genus(), 5);
  ExportMesh("mengerSponge1.gltf", sponge.Extract());
}

TEST(Samples, FrameReduced) {
  Manifold::SetCircularSegments(4);
  Manifold frame = RoundedFrame(100, 10);
  EXPECT_TRUE(frame.IsManifold());
  Manifold::SetCircularSegments(0);
  EXPECT_EQ(frame.Genus(), 5);
  auto prop = frame.GetProperties();
  EXPECT_NEAR(prop.volume, 227333, 10);
  EXPECT_NEAR(prop.surfaceArea, 62635, 1);
  ExportMesh("roundedFrameReduced.gltf", frame.Extract());
}

TEST(Samples, Frame) {
  Manifold frame = RoundedFrame(100, 10);
  EXPECT_TRUE(frame.IsManifold());
  EXPECT_EQ(frame.Genus(), 5);
  // ExportMesh("roundedFrame.ply", frame.Extract());
}
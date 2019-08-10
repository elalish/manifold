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

using namespace manifold;

// If you print this knot (with support), you can snap a half-inch marble into
// it and it'll roll around (dimensions in mm).
TEST(Samples, Knot13) {
  Manifold knot13 = TorusKnot(1, 3, 25, 10, 3.75);
  //   ExportMesh("knot13.stl", knot13.Extract());
  ASSERT_TRUE(knot13.IsValid());
  EXPECT_EQ(knot13.Genus(), 1);
  EXPECT_NEAR(knot13.Volume(), 20786, 1);
  EXPECT_NEAR(knot13.SurfaceArea(), 11177, 1);
}

// This creates two interlinked knots.
TEST(Samples, Knot42) {
  Manifold knot42 = TorusKnot(4, 2, 15, 6, 5);
  //   ExportMesh("knot42.stl", knot42.Extract());
  ASSERT_TRUE(knot42.IsValid());
  std::vector<Manifold> knots = knot42.Decompose();
  ASSERT_EQ(knots.size(), 2);
  EXPECT_EQ(knots[0].Genus(), 1);
  EXPECT_EQ(knots[1].Genus(), 1);
  EXPECT_NEAR(knots[0].Volume(), knots[1].Volume(), 1);
  EXPECT_NEAR(knots[0].SurfaceArea(), knots[1].SurfaceArea(), 1);
}

TEST(Samples, Bracelet) {
  Manifold bracelet = StretchyBracelet();
  ExportMesh("bracelet.stl", bracelet.Extract());
}
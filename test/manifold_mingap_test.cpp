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

#include <algorithm>

#include "cross_section.h"
#include "manifold.h"
#include "samples.h"
#include "test.h"
#include "tri_dist.h"

using namespace manifold;

TEST(Manifold, MinGapCubeCube) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Translate({2, 2, 0});

  float distance = a.MinGap(b, 1.5f);

  EXPECT_FLOAT_EQ(distance, sqrt(2));
}

TEST(Manifold, MinGapCubeCube2) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Translate({3, 3, 0});

  float distance = a.MinGap(b, 3);

  EXPECT_FLOAT_EQ(distance, sqrt(2) * 2);
}

TEST(Manifold, MinGapCubeSphereOverlapping) {
  auto a = Manifold::Cube();
  auto b = Manifold::Sphere(1);

  float distance = a.MinGap(b, 0.1f);

  EXPECT_FLOAT_EQ(distance, 0);
}

TEST(Manifold, MinGapSphereSphere) {
  auto a = Manifold::Sphere(1);
  auto b = Manifold::Sphere(1).Translate({2, 2, 0});

  float distance = a.MinGap(b, 0.85f);

  EXPECT_FLOAT_EQ(distance, 2 * sqrt(2) - 2);
}

TEST(Manifold, MinGapSphereSphereOutOfBounds) {
  auto a = Manifold::Sphere(1);
  auto b = Manifold::Sphere(1).Translate({2, 2, 0});

  float distance = a.MinGap(b, 0.8f);

  EXPECT_FLOAT_EQ(distance, 0.8f);
}

TEST(Manifold, MinGapClosestPointOnEdge) {
  auto a = Manifold::Cube({1, 1, 1}, true).Rotate(0, 0, 45);
  auto b =
      Manifold::Cube({1, 1, 1}, true).Rotate(0, 45, 0).Translate({2, 0, 0});

  float distance = a.MinGap(b, 0.7f);

  EXPECT_FLOAT_EQ(distance, 2 - sqrt(2));
}

TEST(Manifold, MinGapClosestPointOnTriangleFace) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Scale({10, 10, 10}).Translate({2, -5, -1});

  float distance = a.MinGap(b, 1.1f);

  EXPECT_FLOAT_EQ(distance, 1);
}

TEST(Manifold, MingapAfterTransformations) {
  auto a = Manifold::Sphere(1, 512).Rotate(30, 30, 30);
  auto b =
      Manifold::Sphere(1, 512).Scale({3, 1, 1}).Rotate(0, 90, 45).Translate(
          {3, 0, 0});

  float distance = a.MinGap(b, 1.1f);

  ASSERT_NEAR(distance, 1, 0.001f);
}

TEST(Manifold, MingapStretchyBracelet) {
  auto a = StretchyBracelet();
  auto b = StretchyBracelet().Translate({0, 0, 20});

  float distance = a.MinGap(b, 10);

  ASSERT_NEAR(distance, 5, 0.001f);
}

TEST(Manifold, MinGapAfterTransformationsOutOfBounds) {
  auto a = Manifold::Sphere(1, 512).Rotate(30, 30, 30);
  auto b =
      Manifold::Sphere(1, 512).Scale({3, 1, 1}).Rotate(0, 90, 45).Translate(
          {3, 0, 0});

  float distance = a.MinGap(b, 0.95f);

  ASSERT_NEAR(distance, 0.95f, 0.001f);
}
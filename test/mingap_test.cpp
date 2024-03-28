// Copyright 2024 The Manifold Authors.
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

#include "manifold.h"
#include "test.h"

TEST(MinGap, CubeCube) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Translate({2.0f, 2.0f, 0.0f});

  float distance = a.MinGap(b, 1.5f);

  EXPECT_FLOAT_EQ(distance, sqrt(2));
}

TEST(MinGap, CubeCube2) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Translate({3.0f, 3.0f, 0.0f});

  float distance = a.MinGap(b, 3.0f);

  EXPECT_FLOAT_EQ(distance, sqrt(2) * 2);
}

TEST(MinGap, CubeCubeOutOfBounds) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Translate({3.0f, 3.0f, 0.0f});

  float distance = a.MinGap(b, 2.5f);

  EXPECT_FLOAT_EQ(distance, 2.5f);
}

TEST(MinGap, CubeSphereOverlapping) {
  auto a = Manifold::Cube();
  auto b = Manifold::Sphere(1.0f);

  float distance = a.MinGap(b, 0.1f);

  EXPECT_FLOAT_EQ(distance, 0.0f);
}

TEST(MinGap, SphereSphere) {
  auto a = Manifold::Sphere(1.0f);
  auto b = Manifold::Sphere(1.0f).Translate({5.0f, 0.0f, 0.0f});

  float distance = a.MinGap(b, 3.1f);

  EXPECT_FLOAT_EQ(distance, 3.0f);
}

TEST(MinGap, SphereSphere2) {
  auto a = Manifold::Sphere(1.0f);
  auto b = Manifold::Sphere(1.0f).Translate({2.0f, 2.0f, 0.0f});

  float distance = a.MinGap(b, 0.85f);

  EXPECT_FLOAT_EQ(distance, 2 * sqrt(2) - 2);
}

TEST(MinGap, SphereSphereOutOfBounds) {
  auto a = Manifold::Sphere(1.0f);
  auto b = Manifold::Sphere(1.0f).Translate({2.0f, 2.0f, 0.0f});

  float distance = a.MinGap(b, 0.8f);

  EXPECT_FLOAT_EQ(distance, 0.8f);
}

TEST(MinGap, ClosestPointOnEdge) {
  auto a = Manifold::Cube().Rotate(0.0f, 0.0f, 45.0f);
  auto b =
      Manifold::Cube().Rotate(0.0f, 45.0f, 0.0f).Translate({1.0f, 0.0f, 0.5f});

  float distance = a.MinGap(b, 0.3f);

  EXPECT_FLOAT_EQ(distance, 1.0f - sqrt(2) / 2);
}

TEST(MinGap, ClosestPointOnTriangleFace) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube()
               .Scale({10.0f, 10.0f, 10.0f})
               .Translate({2.0f, -5.0f, -1.0f});

  float distance = a.MinGap(b, 1.1f);

  EXPECT_FLOAT_EQ(distance, 1.0f);
}

TEST(MinGap, AfterTransformations) {
  auto a = Manifold::Sphere(1.0f, 100);
  auto b = Manifold::Sphere(1.0f, 100)
               .Scale({3.0f, 1.0f, 1.0f})
               .Rotate(0, 90, 45)
               .Translate({3.0f, 0.0f, 0.0f});

  float distance = a.MinGap(b, 1.1f);

  ASSERT_NEAR(distance, 1.0f, 0.001f);
}

TEST(MinGap, AfterTransformationsOutOfRange) {
  auto a = Manifold::Sphere(1.0f, 100);
  auto b = Manifold::Sphere(1.0f, 100)
               .Scale({3.0f, 1.0f, 1.0f})
               .Rotate(0, 90, 45)
               .Translate({3.0f, 0.0f, 0.0f});

  float distance = a.MinGap(b, 0.95f);

  ASSERT_NEAR(distance, 0.95f, 0.001f);
}

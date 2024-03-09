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

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

TEST(MinGap, CubeCube) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Translate({2.0f, 2.0f, 0.0f});

  float distance = a.MinGap(b, 10);

  EXPECT_FLOAT_EQ(distance, sqrt(2));
}

TEST(MinGap, CubeCube2) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Translate({3.0f, 3.0f, 0.0f});

  float distance = a.MinGap(b, 10);

  EXPECT_FLOAT_EQ(distance, sqrt(2) * 2);
}

TEST(MinGap, CubeCubeOutOfBounds) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Translate({3.0f, 3.0f, 0.0f});

  float distance = a.MinGap(b, 0.1f);

  EXPECT_FLOAT_EQ(distance, 0.1f);
}

TEST(MinGap, CubeSphereOverlapping) {
  auto a = Manifold::Cube();
  auto b = Manifold::Sphere(1.0f);

  float distance = a.MinGap(b, 10);

  EXPECT_FLOAT_EQ(distance, 0);
}

TEST(MinGap, SphereSphere) {
  auto a = Manifold::Sphere(1.0f);
  auto b = Manifold::Sphere(1.0f).Translate({5.0f, 0.0f, 0.0f});

  float distance = a.MinGap(b, 10);

  EXPECT_FLOAT_EQ(distance, 3);
}

TEST(MinGap, SphereSphere2) {
  auto a = Manifold::Sphere(1.0f);
  auto b = Manifold::Sphere(1.0f).Translate({2.0f, 2.0f, 0.0f});

  float distance = a.MinGap(b, 10);

  EXPECT_FLOAT_EQ(distance, 2 * sqrt(2) - 2);
}

TEST(MinGap, SphereSphereOutOfBounds) {
  auto a = Manifold::Sphere(1.0f);
  auto b = Manifold::Sphere(1.0f).Translate({2.0f, 2.0f, 0.0f});

  float distance = a.MinGap(b, 0.1f);

  EXPECT_FLOAT_EQ(distance, 0.1f);
}

TEST(MinGap, ClosestPointOnEdge) {
  auto a = Manifold::Cube();
  auto b = Manifold::Sphere(1.0f).Translate({3.0f, 0.0f, 0.5f});

  float distance = a.MinGap(b, 10);

  EXPECT_FLOAT_EQ(distance, 1);
}

TEST(MinGap, ClosestPointOnTriangleFace) {
  auto a = Manifold::Cube().Translate({0.0f, -0.25f, 0.0f});
  auto b = Manifold::Sphere(1.0f).Translate({3.0f, 0.0f, 0.5f});

  float distance = a.MinGap(b, 10);

  EXPECT_FLOAT_EQ(distance, 1);
}

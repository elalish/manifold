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

#include "../src/tri_dist.h"
#include "manifold/manifold.h"
#include "samples.h"
#include "test.h"

using namespace manifold;

/**
 * These tests verify the calculation of a manifold's geometric properties.
 */
TEST(Properties, GetProperties) {
  Manifold cube = Manifold::Cube();
  auto prop = cube.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 1.0);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 6.0);

  cube = cube.Scale(vec3(-1.0));
  prop = cube.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 1.0);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 6.0);
}

TEST(Properties, Precision) {
  Manifold cube = Manifold::Cube();
  EXPECT_FLOAT_EQ(cube.Precision(), kTolerance);
  cube = cube.Scale({0.1, 1, 10});
  EXPECT_FLOAT_EQ(cube.Precision(), 10 * kTolerance);
  cube = cube.Translate({-100, -10, -1});
  EXPECT_FLOAT_EQ(cube.Precision(), 100 * kTolerance);
}

TEST(Properties, Precision2) {
  Manifold cube = Manifold::Cube();
  cube = cube.Translate({-0.5, 0, 0}).Scale({2, 1, 1});
  EXPECT_FLOAT_EQ(cube.Precision(), 2 * kTolerance);
}

TEST(Properties, Precision3) {
  Manifold cylinder = Manifold::Cylinder(1, 1, 1, 1000);
  const auto prop = cylinder.GetProperties();

  MeshGL mesh = cylinder.GetMeshGL();
  mesh.precision = 0.001;
  mesh.faceID.clear();
  Manifold cylinder2(mesh);

  const auto prop2 = cylinder2.GetProperties();
  EXPECT_NEAR(prop.volume, prop2.volume, 0.001);
  EXPECT_NEAR(prop.surfaceArea, prop2.surfaceArea, 0.001);
}

/**
 * Curvature is the inverse of the radius of curvature, and signed such that
 * positive is convex and negative is concave. There are two orthogonal
 * principal curvatures at any point on a manifold, with one maximum and the
 * other minimum. Gaussian curvature is their product, while mean
 * curvature is their sum. Here we check our discrete approximations
 * calculated at each vertex against the constant expected values of spheres
 * of different radii and at different mesh resolutions.
 */
TEST(Properties, CalculateCurvature) {
  const float precision = 0.015;
  for (int n = 4; n < 100; n *= 2) {
    const int gaussianIdx = 3;
    const int meanIdx = 4;
    Manifold sphere = Manifold::Sphere(1, 64).CalculateCurvature(
        gaussianIdx - 3, meanIdx - 3);
    MeshGL sphereGL = sphere.GetMeshGL();
    ASSERT_EQ(sphereGL.numProp, 5);
    EXPECT_NEAR(GetMinProperty(sphereGL, meanIdx), 2, 2 * precision);
    EXPECT_NEAR(GetMaxProperty(sphereGL, meanIdx), 2, 2 * precision);
    EXPECT_NEAR(GetMinProperty(sphereGL, gaussianIdx), 1, precision);
    EXPECT_NEAR(GetMaxProperty(sphereGL, gaussianIdx), 1, precision);

    sphere = sphere.Scale(vec3(2.0)).CalculateCurvature(gaussianIdx - 3,
                                                        meanIdx - 3);
    sphereGL = sphere.GetMeshGL();
    ASSERT_EQ(sphereGL.numProp, 5);
    EXPECT_NEAR(GetMinProperty(sphereGL, meanIdx), 1, precision);
    EXPECT_NEAR(GetMaxProperty(sphereGL, meanIdx), 1, precision);
    EXPECT_NEAR(GetMinProperty(sphereGL, gaussianIdx), 0.25, 0.25 * precision);
    EXPECT_NEAR(GetMaxProperty(sphereGL, gaussianIdx), 0.25, 0.25 * precision);
  }
}

// These tests verify the calculation of MinGap functions.

TEST(Properties, MinGapCubeCube) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Translate({2, 2, 0});

  float distance = a.MinGap(b, 1.5);

  EXPECT_FLOAT_EQ(distance, sqrt(2));
}

TEST(Properties, MinGapCubeCube2) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Translate({3, 3, 0});

  float distance = a.MinGap(b, 3);

  EXPECT_FLOAT_EQ(distance, sqrt(2) * 2);
}

TEST(Properties, MinGapCubeSphereOverlapping) {
  auto a = Manifold::Cube();
  auto b = Manifold::Sphere(1);

  float distance = a.MinGap(b, 0.1);

  EXPECT_FLOAT_EQ(distance, 0);
}

TEST(Properties, MinGapSphereSphere) {
  auto a = Manifold::Sphere(1);
  auto b = Manifold::Sphere(1).Translate({2, 2, 0});

  float distance = a.MinGap(b, 0.85);

  EXPECT_FLOAT_EQ(distance, 2 * sqrt(2) - 2);
}

TEST(Properties, MinGapSphereSphereOutOfBounds) {
  auto a = Manifold::Sphere(1);
  auto b = Manifold::Sphere(1).Translate({2, 2, 0});

  float distance = a.MinGap(b, 0.8);

  EXPECT_FLOAT_EQ(distance, 0.8);
}

TEST(Properties, MinGapClosestPointOnEdge) {
  auto a = Manifold::Cube({1, 1, 1}, true).Rotate(0, 0, 45);
  auto b =
      Manifold::Cube({1, 1, 1}, true).Rotate(0, 45, 0).Translate({2, 0, 0});

  float distance = a.MinGap(b, 0.7);

  EXPECT_FLOAT_EQ(distance, 2 - sqrt(2));
}

TEST(Properties, MinGapClosestPointOnTriangleFace) {
  auto a = Manifold::Cube();
  auto b = Manifold::Cube().Scale({10, 10, 10}).Translate({2, -5, -1});

  float distance = a.MinGap(b, 1.1);

  EXPECT_FLOAT_EQ(distance, 1);
}

TEST(Properties, MingapAfterTransformations) {
  auto a = Manifold::Sphere(1, 512).Rotate(30, 30, 30);
  auto b =
      Manifold::Sphere(1, 512).Scale({3, 1, 1}).Rotate(0, 90, 45).Translate(
          {3, 0, 0});

  float distance = a.MinGap(b, 1.1);

  ASSERT_NEAR(distance, 1, 0.001);
}

#ifdef MANIFOLD_CROSS_SECTION
TEST(Properties, MingapStretchyBracelet) {
  auto a = StretchyBracelet();
  auto b = StretchyBracelet().Translate({0, 0, 20});

  float distance = a.MinGap(b, 10);

  ASSERT_NEAR(distance, 5, 0.001);
}
#endif

TEST(Properties, MinGapAfterTransformationsOutOfBounds) {
  auto a = Manifold::Sphere(1, 512).Rotate(30, 30, 30);
  auto b =
      Manifold::Sphere(1, 512).Scale({3, 1, 1}).Rotate(0, 90, 45).Translate(
          {3, 0, 0});

  float distance = a.MinGap(b, 0.95);

  ASSERT_NEAR(distance, 0.95, 0.001);
}

TEST(Properties, TriangleDistanceClosestPointsOnVertices) {
  std::array<vec3, 3> p = {vec3{-1, 0, 0}, vec3{1, 0, 0}, vec3{0, 1, 0}};

  std::array<vec3, 3> q = {vec3{2, 0, 0}, vec3{4, 0, 0}, vec3{3, 1, 0}};

  float distance = DistanceTriangleTriangleSquared(p, q);

  EXPECT_FLOAT_EQ(distance, 1);
}

TEST(Properties, TriangleDistanceClosestPointOnEdge) {
  std::array<vec3, 3> p = {vec3{-1, 0, 0}, vec3{1, 0, 0}, vec3{0, 1, 0}};

  std::array<vec3, 3> q = {vec3{-1, 2, 0}, vec3{1, 2, 0}, vec3{0, 3, 0}};

  float distance = DistanceTriangleTriangleSquared(p, q);

  EXPECT_FLOAT_EQ(distance, 1);
}

TEST(Properties, TriangleDistanceClosestPointOnEdge2) {
  std::array<vec3, 3> p = {vec3{-1, 0, 0}, vec3{1, 0, 0}, vec3{0, 1, 0}};

  std::array<vec3, 3> q = {vec3{1, 1, 0}, vec3{3, 1, 0}, vec3{2, 2, 0}};

  float distance = DistanceTriangleTriangleSquared(p, q);

  EXPECT_FLOAT_EQ(distance, 0.5);
}

TEST(Properties, TriangleDistanceClosestPointOnFace) {
  std::array<vec3, 3> p = {vec3{-1, 0, 0}, vec3{1, 0, 0}, vec3{0, 1, 0}};

  std::array<vec3, 3> q = {vec3{-1, 2, -0.5}, vec3{1, 2, -0.5},
                           vec3{0, 2, 1.5}};

  float distance = DistanceTriangleTriangleSquared(p, q);

  EXPECT_FLOAT_EQ(distance, 1);
}

TEST(Properties, TriangleDistanceOverlapping) {
  std::array<vec3, 3> p = {vec3{-1, 0, 0}, vec3{1, 0, 0}, vec3{0, 1, 0}};

  std::array<vec3, 3> q = {vec3{-1, 0, 0}, vec3{1, 0.5, 0}, vec3{0, 1, 0}};

  float distance = DistanceTriangleTriangleSquared(p, q);

  EXPECT_FLOAT_EQ(distance, 0);
}

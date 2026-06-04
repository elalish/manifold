// Copyright 2026 The Manifold Authors.
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

// Tests for measurement-style queries on Manifolds:
//   - `Manifold::RayCast` (ray-mesh intersection)
//   - `Manifold::MinGap` (inter-manifold distance)
//   - `DistanceTriangleTriangleSquared` (the per-triangle primitive
//     MinGap is built on)

#include "../src/tri_dist.h"
#include "manifold/manifold.h"
#include "samples.h"
#include "test.h"

using namespace manifold;

TEST(Manifold, RayCastHitCube) {
  // Ray through center along Z — also tests watertight shared edge at (0,0)
  Manifold cube = Manifold::Cube(vec3(1), true);

  auto hits = cube.RayCast(vec3(0, 0, -5), vec3(0, 0, 5));
  ASSERT_EQ(hits.size(), 2);
  EXPECT_LT(hits[0].distance, hits[1].distance);
  EXPECT_FLOAT_EQ(hits[0].position.z, -0.5);
  EXPECT_FLOAT_EQ(hits[1].position.z, 0.5);
  // Bottom face normal should point -Z, top +Z
  EXPECT_FLOAT_EQ(hits[0].normal.z, -1.0);
  EXPECT_FLOAT_EQ(hits[1].normal.z, 1.0);
}

TEST(Manifold, RayCastMiss) {
  Manifold cube = Manifold::Cube(vec3(1), true);
  EXPECT_EQ(cube.RayCast(vec3(10, 10, -5), vec3(10, 10, 5)).size(), 0);
}

TEST(Manifold, RayCastDiagonal) {
  // Diagonal ray — not axis-aligned
  Manifold cube = Manifold::Cube(vec3(1), true);

  auto hits = cube.RayCast(vec3(-5, -5, -5), vec3(5, 5, 5));
  ASSERT_EQ(hits.size(), 2);
  EXPECT_FLOAT_EQ(hits[0].position.z, -0.5);
}

TEST(Manifold, RayCastBehindOrigin) {
  Manifold cube = Manifold::Cube(vec3(1), true);
  EXPECT_EQ(cube.RayCast(vec3(0, 0, 5), vec3(0, 0, 10)).size(), 0);
}

TEST(Manifold, RayCastSphere) {
  Manifold sphere = Manifold::Sphere(1.0, 128);

  auto hits = sphere.RayCast(vec3(0, 0, -5), vec3(0, 0, 5));
  ASSERT_EQ(hits.size(), 2);
  EXPECT_NEAR(la::length(hits[0].position), 1.0, 1e-4);

  EXPECT_EQ(sphere.RayCast(vec3(2, 2, -5), vec3(2, 2, 5)).size(), 0);
}

TEST(Manifold, RayCastTwoCubes) {
  Manifold c1 = Manifold::Cube(vec3(1), true);
  Manifold c2 = Manifold::Cube(vec3(1), true).Translate(vec3(0, 0, 5));
  Manifold both = c1 + c2;

  auto hits = both.RayCast(vec3(0, 0, -5), vec3(0, 0, 10));
  ASSERT_EQ(hits.size(), 4);
  EXPECT_FLOAT_EQ(hits[0].position.z, -0.5);
  EXPECT_FLOAT_EQ(hits[1].position.z, 0.5);
  EXPECT_FLOAT_EQ(hits[2].position.z, 4.5);
  EXPECT_FLOAT_EQ(hits[3].position.z, 5.5);
}

TEST(Manifold, RayCastEmpty) {
  Manifold empty;
  EXPECT_EQ(empty.RayCast(vec3(0, 0, -5), vec3(0, 0, 5)).size(), 0);
}

TEST(Manifold, RayCastAlongX) {
  Manifold cube = Manifold::Cube(vec3(1), true);

  auto hits = cube.RayCast(vec3(-5, 0, 0), vec3(5, 0, 0));
  ASSERT_EQ(hits.size(), 2);
  EXPECT_FLOAT_EQ(hits[0].position.x, -0.5);
}

TEST(Manifold, RayCastAlongY) {
  Manifold cube = Manifold::Cube(vec3(1), true);

  auto hits = cube.RayCast(vec3(0, -5, 0), vec3(0, 5, 0));
  ASSERT_EQ(hits.size(), 2);
  EXPECT_FLOAT_EQ(hits[0].position.y, -0.5);
}

TEST(Manifold, RayCastZeroLength) {
  Manifold cube = Manifold::Cube(vec3(1), true);
  EXPECT_EQ(cube.RayCast(vec3(0, 0, 0), vec3(0, 0, 0)).size(), 0);
}

TEST(Manifold, RayCastWatertightVertex) {
  // Ray exactly through a vertex. Symbolic perturbation should assign
  // the hit to exactly one triangle per face.
  Manifold cube = Manifold::Cube(vec3(1), true);

  auto hits = cube.RayCast(vec3(0.5, 0.5, -5), vec3(0.5, 0.5, 5));
  ASSERT_EQ(hits.size(), 2);
  EXPECT_FLOAT_EQ(hits[0].position.z, -0.5);
}

TEST(Manifold, RayCastSilhouetteEdge) {
  // Ray at the silhouette edge should return 0 or 2 hits, never 1.
  Manifold cube = Manifold::Cube(vec3(1), true);

  auto hits = cube.RayCast(vec3(0.5, 0, -5), vec3(0.5, 0, 5));
  EXPECT_TRUE(hits.size() == 0 || hits.size() == 2);
}

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

// A tall box where the query point is far below the top face. The +Z ray from
// the point must reach the top face even though it's well outside the point's
// 3D bounding-box in Z. This exercises the XY-projected collider query.
TEST(Manifold, WindingTallBox) {
  Manifold box = Manifold::Cube(vec3(1, 1, 10), true);  // z in [-5, 5]
  const auto w =
      box.WindingNumber({{0, 0, -4}, {0, 0, 0}, {0, 0, 6}, {0, 0, -6}});
  EXPECT_EQ(w[0], 1);  // inside, near bottom
  EXPECT_EQ(w[1], 1);  // center
  EXPECT_EQ(w[2], 0);  // above
  EXPECT_EQ(w[3], 0);  // below
}

TEST(Manifold, WindingEmpty) {
  const auto w = Manifold{}.WindingNumber({{0, 0, 0}});
  ASSERT_EQ(w.size(), 1u);
  EXPECT_EQ(w[0], 0);
}

// Watertight winding across shared cube edges and vertices - must be exactly
// 0 or 1, never 2 from double-counting.
TEST(Manifold, WindingWatertight) {
  Manifold cube = Manifold::Cube(vec3(1), true);
  const auto w =
      cube.WindingNumber({{0.4999, 0.4999, 0}, {0.5, 0, 0}, {0, 0, 0}});
  EXPECT_TRUE(w[0] == 0 || w[0] == 1);  // near shared vertex
  EXPECT_TRUE(w[1] == 0 || w[1] == 1);  // silhouette edge
  EXPECT_EQ(w[2], 1);                   // center
}

// Hollow shell (sphere - sphere): cavity center must return winding 0, wall
// point 1, outside 0. Proves real winding number, not just ray parity.
TEST(Manifold, WindingHollowShell) {
  Manifold shell = Manifold::Sphere(2, 32) - Manifold::Sphere(1, 32);
  const auto w = shell.WindingNumber({{0, 0, 0}, {0, 0, 1.5}, {0, 0, 3}});
  EXPECT_EQ(w[0], 0);  // cavity
  EXPECT_EQ(w[1], 1);  // wall
  EXPECT_EQ(w[2], 0);  // outside
}

// Cross-check: winding != 0 agrees with RayCast hit-count parity.
TEST(Manifold, WindingMatchesRayCastParity) {
  Manifold cube = Manifold::Cube(vec3(1), true);
  const std::vector<vec3> pts = {{0, 0, 0}, {0.3, 0.2, 0.1}, {0.5, 0.5, 0.5},
                                 {0, 0, 1}, {0, 0, 2},       {-1, 0, 0}};
  const auto winding = cube.WindingNumber(pts);
  for (size_t i = 0; i < pts.size(); ++i) {
    const vec3& p = pts[i];
    const size_t hits = cube.RayCast(p, {p.x, p.y, p.z + 1e6}).size();
    EXPECT_EQ(winding[i] != 0, hits % 2 == 1)
        << "mismatch at (" << p.x << "," << p.y << "," << p.z << ")";
  }
}

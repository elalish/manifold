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

#include "../src/utils.h"
#include "manifold/manifold.h"
#include "test.h"

using namespace manifold;

/**
 * These tests verify the calculation of a manifold's geometric properties.
 */
TEST(Properties, Measurements) {
  Manifold cube = Manifold::Cube();
  EXPECT_FLOAT_EQ(cube.Volume(), 1.0);
  EXPECT_FLOAT_EQ(cube.SurfaceArea(), 6.0);

  cube = cube.Scale(vec3(-1.0));
  EXPECT_FLOAT_EQ(cube.Volume(), 1.0);
  EXPECT_FLOAT_EQ(cube.SurfaceArea(), 6.0);
}

TEST(Properties, Epsilon) {
  Manifold cube = Manifold::Cube();
  EXPECT_FLOAT_EQ(cube.GetEpsilon(), kPrecision);
  cube = cube.Scale({0.1, 1, 10});
  EXPECT_FLOAT_EQ(cube.GetEpsilon(), 10 * kPrecision);
  cube = cube.Translate({-100, -10, -1});
  EXPECT_FLOAT_EQ(cube.GetEpsilon(), 100 * kPrecision);
}

TEST(Properties, Epsilon2) {
  Manifold cube = Manifold::Cube();
  cube = cube.Translate({-0.5, 0, 0}).Scale({2, 1, 1});
  EXPECT_FLOAT_EQ(cube.GetEpsilon(), 2 * kPrecision);
}

TEST(Properties, Tolerance) {
  double degrees = 1;
  double tol = sind(degrees);
  Manifold cube = Manifold::Cube({1, 1, 1}, true);
  Manifold imperfect = (cube ^ cube.Rotate(degrees)).AsOriginal();
  EXPECT_EQ(imperfect.NumTri(), 28);

  Manifold imperfect2 = imperfect.Simplify(tol);
  EXPECT_EQ(imperfect2.NumTri(), 12);

  MeshGL mesh = imperfect.GetMeshGL();
  mesh.tolerance = tol;
  Manifold imperfect3(mesh);
  EXPECT_EQ(imperfect3.NumTri(), 28);  // Don't automatically simplify

  EXPECT_NEAR(imperfect.Volume(), imperfect2.Volume(), 0.01);
  EXPECT_NEAR(imperfect.SurfaceArea(), imperfect2.SurfaceArea(), 0.02);
  EXPECT_NEAR(imperfect2.Volume(), imperfect3.Volume(), 0.01);
  EXPECT_NEAR(imperfect2.SurfaceArea(), imperfect3.SurfaceArea(), 0.02);

  if (options.exportModels) {
    WriteTestOBJ("tolerance.obj", imperfect2);
    WriteTestOBJ("tolerance2.obj", imperfect3);
  }
}

TEST(Properties, ToleranceSphere) {
  const int n = 100;
  Manifold sphere = Manifold::Sphere(1, 4 * n);
  EXPECT_EQ(sphere.NumTri(), 8 * n * n);

  Manifold sphere2 = sphere.SetTolerance(0.01);
  EXPECT_LT(sphere2.NumTri(), 2500);
  EXPECT_EQ(sphere2.Genus(), 0);
  EXPECT_NEAR(sphere.Volume(), sphere2.Volume(), 0.05);
  EXPECT_NEAR(sphere.SurfaceArea(), sphere2.SurfaceArea(), 0.06);
  if (options.exportModels) WriteTestOBJ("sphere.obj", sphere2);
}

TEST(Properties, ToleranceCylinder) {
  const int n = 40;
  Manifold cylinder = Manifold::Cylinder(2, 1, 1, 4 * n);
  Manifold cylinder2 = cylinder.Simplify(0.01);
  EXPECT_LT(cylinder2.NumTri(), 130);
  EXPECT_EQ(cylinder2.Genus(), 0);
  EXPECT_NEAR(cylinder.Volume(), cylinder2.Volume(), 0.006);
  EXPECT_NEAR(cylinder.SurfaceArea(), cylinder2.SurfaceArea(), 0.006);
  if (options.exportModels) WriteTestOBJ("cylinder.obj", cylinder2);
}

TEST(Properties, ToleranceCube) {
  Manifold cube = Manifold::Cube().Refine(5);
  EXPECT_EQ(cube.NumTri(), 300);
  Manifold cube2 = cube.Simplify();
  EXPECT_LT(cube2.NumTri(), 40);
  EXPECT_EQ(cube2.Genus(), 0);
  EXPECT_DOUBLE_EQ(cube.Volume(), cube2.Volume());
  EXPECT_DOUBLE_EQ(cube.SurfaceArea(), cube2.SurfaceArea());
  if (options.exportModels) WriteTestOBJ("cube.obj", cube2);
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

TEST(Properties, CalculateNormals) {
  Manifold sphere = Manifold::Sphere(10);
  vec3 center(10);
  Manifold cut = (sphere - sphere.Translate(center));
  cut.Status();
  Manifold cut2(cut.GetMeshGL64());
  EXPECT_TRUE(cut.MatchesTriNormals());
  EXPECT_TRUE(cut2.MatchesTriNormals());
  MeshGL64 out = cut.CalculateNormals().GetMeshGL64();
  MeshGL64 out2 = cut2.CalculateNormals().GetMeshGL64();
  ASSERT_EQ(out.NumTri(), out2.NumTri());
  ASSERT_EQ(out.NumVert(), out2.NumVert());
  ASSERT_EQ(out.numProp, out2.numProp);
  const int np = out.numProp;
  int numBad = 0;
  int numBad2 = 0;
  for (int v = 0; v < out.NumVert(); ++v) {
    auto pos = out.GetVertPos(v);
    auto pos2 = out2.GetVertPos(v);
    auto norm = pos;
    auto norm2 = pos2;
    for (int j : {0, 1, 2}) {
      norm[j] = out.vertProperties[np * v + 3 + j];
      norm2[j] = out2.vertProperties[np * v + 3 + j];
      ASSERT_FLOAT_EQ(pos[j], pos2[j]);
      ASSERT_NEAR(norm[j], norm2[j], 1e-14);
    }
    if (dot(pos, norm) <= 0) ++numBad;
    if (dot(pos2, norm2) <= 0) ++numBad2;
    EXPECT_FLOAT_EQ(la::length(norm), 1.0);
    EXPECT_TRUE(dot(la::normalize(pos), norm) > 0.99 ||
                dot(la::normalize(center - pos), norm) > 0.99);
  }
  EXPECT_EQ(numBad, 0);
  EXPECT_EQ(numBad2, 0);
  for (int tv = out2.runIndex[0]; tv < out2.runIndex[1]; ++tv) {
    const int v = out2.triVerts[tv];
    auto pos = out2.GetVertPos(v);
    auto norm = pos;
    for (int j : {0, 1, 2}) {
      norm[j] = out2.vertProperties[np * v + 3 + j];
    }
    EXPECT_FLOAT_EQ(la::length(norm), 1.0);
    EXPECT_GT(dot(la::normalize(pos), norm), 0.99);
  }
}

TEST(Properties, Coplanar) {
  Manifold peg = Manifold::Cube({1, 1, 2}).Translate({1, 1, 0}).AsOriginal();
  const size_t pegID = peg.OriginalID();
  Manifold hole = Manifold::Cube({3, 3, 1}) - peg;
  hole = hole.AsOriginal();
  std::vector<MeshGL> input = {hole.GetMeshGL(), peg.GetMeshGL()};
  EXPECT_EQ(peg.Genus(), 0);
  EXPECT_EQ(hole.Genus(), 1);

  Manifold result = hole + peg;
  EXPECT_EQ(result.Genus(), 0);
  RelatedGL(result, input);

  MeshGL resultGL = result.GetMeshGL();
  float minPegZ = std::numeric_limits<float>::max();
  for (size_t run = 0; run < resultGL.runOriginalID.size(); run++) {
    if (resultGL.runOriginalID[run] == pegID) {
      for (size_t t3 = resultGL.runIndex[run]; t3 < resultGL.runIndex[run + 1];
           t3++) {
        const size_t v = resultGL.triVerts[t3];
        minPegZ = std::min(minPegZ,
                           resultGL.vertProperties[v * resultGL.numProp + 2]);
      }
    }
  }
  EXPECT_EQ(minPegZ, 0);

  if (options.exportModels) WriteTestOBJ("coplanar.obj", result);
}

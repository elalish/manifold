// Copyright 2022 The Manifold Authors.
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

#include "manifold/manifold.h"
#include "test.h"

using namespace manifold;

struct CubeVoid {
  double operator()(vec3 p) const {
    const vec3 min = p + vec3(1);
    const vec3 max = vec3(1) - p;
    const double min3 = std::min(min.x, std::min(min.y, min.z));
    const double max3 = std::min(max.x, std::min(max.y, max.z));
    return -1.0 * std::min(min3, max3);
  }
};

struct Layers {
  double operator()(vec3 p) const {
    int a = std::fmod(std::round(2 * p.z), 4.0);
    return a == 0 ? 1 : (a == 2 ? -1 : 0);
  }
};

TEST(SDF, SphereShell) {
  Manifold sphere = Manifold::LevelSet(
      [](vec3 pos) {
        const double r = la::length(pos);
        return la::min(1 - r, r - 0.995f);
      },
      {vec3(-1.1), vec3(1.1)}, 0.01, 0, 0.0001);

  EXPECT_NEAR(sphere.Genus(), 11500, 1000);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("sphereShellSDF.glb", sphere.GetMeshGL(), {});
#endif
}

TEST(SDF, CubeVoid) {
  CubeVoid voidSDF;

  EXPECT_EQ(voidSDF({0, 0, 0}), -1);
  EXPECT_EQ(voidSDF({0, 0, 1}), 0);
  EXPECT_EQ(voidSDF({0, 1, 1}), 0);
  EXPECT_EQ(voidSDF({-1, 0, 0}), 0);
  EXPECT_EQ(voidSDF({1, 1, -1}), 0);
  EXPECT_EQ(voidSDF({2, 0, 0}), 1);
  EXPECT_EQ(voidSDF({2, -2, 0}), 1);
  EXPECT_EQ(voidSDF({-2, 2, 2}), 1);
}

TEST(SDF, Bounds) {
  const double size = 4;
  const double edgeLength = 1;

  Manifold cubeVoid = Manifold::LevelSet(
      CubeVoid(), {vec3(-size / 2), vec3(size / 2)}, edgeLength);
  Box bounds = cubeVoid.BoundingBox();
  const double precision = cubeVoid.Precision();
#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("cubeVoid.glb", cubeVoid.GetMeshGL(), {});
#endif

  EXPECT_EQ(cubeVoid.Status(), Manifold::Error::NoError);
  EXPECT_EQ(cubeVoid.Genus(), -1);
  const double outerBound = size / 2 + edgeLength / 2;
  EXPECT_NEAR(bounds.min.x, -outerBound, precision);
  EXPECT_NEAR(bounds.min.y, -outerBound, precision);
  EXPECT_NEAR(bounds.min.z, -outerBound, precision);
  EXPECT_NEAR(bounds.max.x, outerBound, precision);
  EXPECT_NEAR(bounds.max.y, outerBound, precision);
  EXPECT_NEAR(bounds.max.z, outerBound, precision);
}

TEST(SDF, Bounds2) {
  const double size = 4;
  const double edgeLength = 1;

  Manifold cubeVoid = Manifold::LevelSet(
      CubeVoid(), {vec3(-size / 2), vec3(size / 2)}, edgeLength);
  Box bounds = cubeVoid.BoundingBox();
  const double precision = cubeVoid.Precision();
#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("cubeVoid2.glb", cubeVoid.GetMeshGL(), {});
#endif

  EXPECT_EQ(cubeVoid.Status(), Manifold::Error::NoError);
  EXPECT_EQ(cubeVoid.Genus(), -1);
  const double outerBound = size / 2 + edgeLength / 2;
  EXPECT_NEAR(bounds.min.x, -outerBound, precision);
  EXPECT_NEAR(bounds.min.y, -outerBound, precision);
  EXPECT_NEAR(bounds.min.z, -outerBound, precision);
  EXPECT_NEAR(bounds.max.x, outerBound, precision);
  EXPECT_NEAR(bounds.max.y, outerBound, precision);
  EXPECT_NEAR(bounds.max.z, outerBound, precision);
}

TEST(SDF, Surface) {
  const double size = 4;
  const double edgeLength = 0.5;

  Manifold cubeVoid = Manifold::LevelSet(
      CubeVoid(), {vec3(-size / 2), vec3(size / 2)}, edgeLength);

  Manifold cube = Manifold::Cube(vec3(size), true);
  cube -= cubeVoid;
  Box bounds = cube.BoundingBox();
  const double precision = cube.Precision();
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("cube.gltf", cube.GetMeshGL(), {});
#endif

  EXPECT_EQ(cubeVoid.Status(), Manifold::Error::NoError);
  EXPECT_EQ(cube.Genus(), 0);
  auto prop = cube.GetProperties();
  EXPECT_NEAR(prop.volume, 8, 0.001);
  EXPECT_NEAR(prop.surfaceArea, 24, 0.001);
  EXPECT_NEAR(bounds.min.x, -1, precision);
  EXPECT_NEAR(bounds.min.y, -1, precision);
  EXPECT_NEAR(bounds.min.z, -1, precision);
  EXPECT_NEAR(bounds.max.x, 1, precision);
  EXPECT_NEAR(bounds.max.y, 1, precision);
  EXPECT_NEAR(bounds.max.z, 1, precision);
}

TEST(SDF, Resize) {
  const double size = 20;
  Manifold layers = Manifold::LevelSet(Layers(), {vec3(0.0), vec3(size)}, 1);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("layers.gltf", layers.GetMeshGL(), {});
#endif

  EXPECT_EQ(layers.Status(), Manifold::Error::NoError);
  EXPECT_EQ(layers.Genus(), -8);
}

TEST(SDF, SineSurface) {
  Manifold surface = Manifold::LevelSet(
      [](vec3 p) {
        double mid = la::sin(p.x) + la::sin(p.y);
        return (p.z > mid - 0.5 && p.z < mid + 0.5) ? 1.0f : -1.0f;
      },
      {vec3(-1.75 * kPi), vec3(1.75 * kPi)}, 1);
  Manifold smoothed = surface.SmoothOut(180).RefineToLength(0.05);

  EXPECT_EQ(smoothed.Status(), Manifold::Error::NoError);
  EXPECT_EQ(smoothed.Genus(), 38);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("sinesurface.glb", smoothed.GetMeshGL(), {});
#endif
}

TEST(SDF, Blobs) {
  const double blend = 1;
  std::vector<vec4> balls = {{0, 0, 0, 2},     //
                             {1, 2, 3, 2},     //
                             {-2, 2, -2, 1},   //
                             {-2, -3, -2, 2},  //
                             {-3, -1, -3, 1},  //
                             {2, -3, -2, 2},   //
                             {-2, 3, 2, 2},    //
                             {-2, -3, 2, 2},   //
                             {1, -1, 1, -2},   //
                             {-4, -3, -2, 1}};
  Manifold blobs = Manifold::LevelSet(
      [&balls, blend](vec3 p) {
        double d = 0;
        for (const auto& ball : balls) {
          d += (ball.w > 0 ? 1 : -1) *
               smoothstep(-blend, blend,
                          std::abs(ball.w) - la::length(vec3(ball) - p));
        }
        return d;
      },
      {vec3(-5), vec3(5)}, 0.05, 0.5);

  const int chi = blobs.NumVert() - blobs.NumTri() / 2;
  const int genus = 1 - chi / 2;
  EXPECT_EQ(genus, 0);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("blobs.glb", blobs.GetMeshGL(), {});
#endif
}

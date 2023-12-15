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

#include "sdf.h"

#include "manifold.h"
#include "test.h"

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

using namespace manifold;

struct CubeVoid {
  double operator()(glm::dvec3 p) const {
    const glm::dvec3 min = p + glm::dvec3(1);
    const glm::dvec3 max = glm::dvec3(1) - p;
    const double min3 = glm::min(min.x, glm::min(min.y, min.z));
    const double max3 = glm::min(max.x, glm::min(max.y, max.z));
    return -1.0 * glm::min(min3, max3);
  }
};

struct Layers {
  double operator()(glm::dvec3 p) const {
    int a = glm::mod(glm::round(2 * p.z), 4.0);
    return a == 0 ? 1 : (a == 2 ? -1 : 0);
  }
};

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
  const double edgeLength = 0.5;

  Mesh levelSet = LevelSet(
      CubeVoid(), {glm::dvec3(-size / 2), glm::dvec3(size / 2)}, edgeLength);
  Manifold cubeVoid(levelSet);
  Box bounds = cubeVoid.BoundingBox();
  const double precision = cubeVoid.Precision();
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("cubeVoid.gltf", levelSet, {});
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

  Manifold cubeVoid(LevelSet(
      CubeVoid(), {glm::dvec3(-size / 2), glm::dvec3(size / 2)}, edgeLength));

  Manifold cube = Manifold::Cube(glm::dvec3(size), true);
  cube -= cubeVoid;
  Box bounds = cube.BoundingBox();
  const double precision = cube.Precision();
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("cube.gltf", cube.GetMesh(), {});
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
  Manifold layers(LevelSet(Layers(), {glm::dvec3(0), glm::dvec3(size)}, 1));
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("layers.gltf", layers.GetMesh(), {});
#endif

  EXPECT_EQ(layers.Status(), Manifold::Error::NoError);
  EXPECT_EQ(layers.Genus(), -8);
}

TEST(SDF, SineSurface) {
  Mesh surface(LevelSet(
      [](glm::dvec3 p) {
        double mid = glm::sin(p.x) + glm::sin(p.y);
        return (p.z > mid - 0.5 && p.z < mid + 0.5) ? 1 : 0;
      },
      {glm::dvec3(-4 * glm::pi<double>()), glm::dvec3(4 * glm::pi<double>())},
      1));
  Manifold smoothed = Manifold::Smooth(surface).Refine(2);

  EXPECT_EQ(smoothed.Status(), Manifold::Error::NoError);
  EXPECT_EQ(smoothed.Genus(), -2);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("sinesurface.glb", smoothed.GetMeshGL(), {});
#endif
}

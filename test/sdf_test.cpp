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
  float operator()(glm::vec3 p) const {
    const glm::vec3 min = p + glm::vec3(1);
    const glm::vec3 max = glm::vec3(1) - p;
    const float min3 = glm::min(min.x, glm::min(min.y, min.z));
    const float max3 = glm::min(max.x, glm::min(max.y, max.z));
    return -1.0f * glm::min(min3, max3);
  }
};

struct Layers {
  float operator()(glm::vec3 p) const {
    int a = glm::mod(glm::round(2 * p.z), 4.0f);
    return a == 0 ? 1 : (a == 2 ? -1 : 0);
  }
};

struct UnboundedGyroidSDF {
  float operator()(glm::vec3 p) const {
    auto qpi = glm::pi<float>() / 4.0;
    return cos(p.x - qpi) * sin(p.y - qpi) + 
           cos(p.y - qpi) * sin(p.z - qpi) +
           cos(p.z - qpi) * sin(p.x - qpi);
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

TEST(SDF, BatchValidation) {
  const float size = glm::two_pi<float>();
  const float edgeLength = size / 10;  // 15;

  Mesh levelSet = LevelSet(UnboundedGyroidSDF(),
      {glm::vec3(-size), glm::vec3(size)}, edgeLength);
  Mesh levelSetBatch = LevelSetBatch(
      [](std::vector<glm::vec3> pvec) {
        std::vector<float> dvec(pvec.size());
        for (int i = 0; i < pvec.size(); i++) {
          dvec[i] = UnboundedGyroidSDF()(pvec[i]);
        }
        return dvec;
      },
      {glm::vec3(-size), glm::vec3(size)}, edgeLength);

  
  EXPECT_EQ(levelSet.vertPos .size(), levelSetBatch.vertPos .size());
  EXPECT_EQ(levelSet.triVerts.size(), levelSetBatch.triVerts.size());

  Manifold      gyroid(levelSet);
  Manifold batchGyroid(levelSetBatch);

  Box            bounds      =      gyroid.BoundingBox();
  Box            boundsBatch = batchGyroid.BoundingBox();
  const float precision      =      gyroid.Precision();
  const float precisionBatch = batchGyroid.Precision();
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("batchcubeVoid.gltf", levelSetBatch, {});
#endif

  EXPECT_EQ  (     gyroid.Status(), Manifold::Error::NoError);
  EXPECT_EQ  (batchGyroid.Status(), Manifold::Error::NoError);
  EXPECT_EQ  (     gyroid.Genus (), batchGyroid.Genus ());
  const float outerBound = size + edgeLength / 2;
  EXPECT_NEAR(bounds.min.x, -outerBound, precision);
  EXPECT_NEAR(bounds.min.y, -outerBound, precision);
  EXPECT_NEAR(bounds.min.z, -outerBound, precision);
  EXPECT_NEAR(bounds.max.x,  outerBound, precision);
  EXPECT_NEAR(bounds.max.y,  outerBound, precision);
  EXPECT_NEAR(bounds.max.z,  outerBound, precision);
  EXPECT_NEAR(boundsBatch.min.x, -outerBound, precisionBatch);
  EXPECT_NEAR(boundsBatch.min.y, -outerBound, precisionBatch);
  EXPECT_NEAR(boundsBatch.min.z, -outerBound, precisionBatch);
  EXPECT_NEAR(boundsBatch.max.x,  outerBound, precisionBatch);
  EXPECT_NEAR(boundsBatch.max.y,  outerBound, precisionBatch);
  EXPECT_NEAR(boundsBatch.max.z,  outerBound, precisionBatch);
}

TEST(SDF, BatchBounds) {
  const float size = 4;
  const float edgeLength = 0.5;

  Mesh levelSet = LevelSetBatch(
      [](std::vector<glm::vec3> pvec) {
        std::vector<float> dvec(pvec.size());
        for (int i = 0; i < pvec.size(); i++) {
          dvec[i] = CubeVoid()(pvec[i]);
        }
        return dvec;
      }, {glm::vec3(-size / 2), glm::vec3(size / 2)}, edgeLength);
  Manifold cubeVoid(levelSet);
  Box bounds = cubeVoid.BoundingBox();
  const float precision = cubeVoid.Precision();
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("batchcubeVoid.gltf", levelSet, {});
#endif

  EXPECT_EQ(cubeVoid.Status(), Manifold::Error::NoError);
  EXPECT_EQ(cubeVoid.Genus(), -1);
  const float outerBound = size / 2 + edgeLength / 2;
  EXPECT_NEAR(bounds.min.x, -outerBound, precision);
  EXPECT_NEAR(bounds.min.y, -outerBound, precision);
  EXPECT_NEAR(bounds.min.z, -outerBound, precision);
  EXPECT_NEAR(bounds.max.x, outerBound, precision);
  EXPECT_NEAR(bounds.max.y, outerBound, precision);
  EXPECT_NEAR(bounds.max.z, outerBound, precision);
}

TEST(SDF, BatchSurface) {
  const float size = 4;
  const float edgeLength = 0.5;

  Manifold cubeVoid(LevelSetBatch(
      [](std::vector<glm::vec3> pvec) {
        std::vector<float> dvec(pvec.size());
        for (int i = 0; i < pvec.size(); i++) {
          dvec[i] = CubeVoid()(pvec[i]);
        }
        return dvec;
      }, {glm::vec3(-size / 2), glm::vec3(size / 2)}, edgeLength));

  Manifold cube = Manifold::Cube(glm::vec3(size), true);
  cube -= cubeVoid;
  Box bounds = cube.BoundingBox();
  const float precision = cube.Precision();
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("batchcube.gltf", cube.GetMesh(), {});
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

TEST(SDF, BatchResize) {
  const float size = 20;
  Manifold layers(LevelSetBatch(
      [](std::vector<glm::vec3> pvec) {
        std::vector<float> dvec(pvec.size());
        for (int i = 0; i < pvec.size(); i++) {
          dvec[i] = Layers()(pvec[i]);
        }
        return dvec;
      }, {glm::vec3(0), glm::vec3(size)}, 1));
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("layers.gltf", layers.GetMesh(), {});
#endif

  EXPECT_EQ(layers.Status(), Manifold::Error::NoError);
  EXPECT_EQ(layers.Genus(), -8);
}

TEST(SDF, BatchSineSurface) {
  Mesh surface(LevelSetBatch(
      [](std::vector<glm::vec3> pvec) {
        std::vector<float> dvec(pvec.size());
        for (int i = 0; i < pvec.size(); i++) {
          float mid = glm::sin(pvec[i].x) + glm::sin(pvec[i].y);
          dvec[i] = (pvec[i].z > mid - 0.5 && pvec[i].z < mid + 0.5) ? 1 : 0;
        }
        return dvec;
      },
      {glm::vec3(-4 * glm::pi<float>()), glm::vec3(4 * glm::pi<float>())}, 1));
  Manifold smoothed = Manifold::Smooth(surface).Refine(2);

  EXPECT_EQ(smoothed.Status(), Manifold::Error::NoError);
  EXPECT_EQ(smoothed.Genus(), -2);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("batchsinesurface.glb", smoothed.GetMeshGL(), {});
#endif
}


TEST(SDF, Bounds) {
  const float size = 4;
  const float edgeLength = 0.5;

  Mesh levelSet = LevelSet(
      CubeVoid(), {glm::vec3(-size / 2), glm::vec3(size / 2)}, edgeLength);
  Manifold cubeVoid(levelSet);
  Box bounds = cubeVoid.BoundingBox();
  const float precision = cubeVoid.Precision();
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("cubeVoid.gltf", levelSet, {});
#endif

  EXPECT_EQ(cubeVoid.Status(), Manifold::Error::NoError);
  EXPECT_EQ(cubeVoid.Genus(), -1);
  const float outerBound = size / 2 + edgeLength / 2;
  EXPECT_NEAR(bounds.min.x, -outerBound, precision);
  EXPECT_NEAR(bounds.min.y, -outerBound, precision);
  EXPECT_NEAR(bounds.min.z, -outerBound, precision);
  EXPECT_NEAR(bounds.max.x, outerBound, precision);
  EXPECT_NEAR(bounds.max.y, outerBound, precision);
  EXPECT_NEAR(bounds.max.z, outerBound, precision);
}

TEST(SDF, Surface) {
  const float size = 4;
  const float edgeLength = 0.5;

  Manifold cubeVoid(LevelSet(
      CubeVoid(), {glm::vec3(-size / 2), glm::vec3(size / 2)}, edgeLength));

  Manifold cube = Manifold::Cube(glm::vec3(size), true);
  cube -= cubeVoid;
  Box bounds = cube.BoundingBox();
  const float precision = cube.Precision();
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
  const float size = 20;
  Manifold layers(LevelSet(Layers(), {glm::vec3(0), glm::vec3(size)}, 1));
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("layers.gltf", layers.GetMesh(), {});
#endif

  EXPECT_EQ(layers.Status(), Manifold::Error::NoError);
  EXPECT_EQ(layers.Genus(), -8);
}

TEST(SDF, SineSurface) {
  Mesh surface(LevelSet(
      [](glm::vec3 p) {
        float mid = glm::sin(p.x) + glm::sin(p.y);
        return (p.z > mid - 0.5 && p.z < mid + 0.5) ? 1 : 0;
      },
      {glm::vec3(-4 * glm::pi<float>()), glm::vec3(4 * glm::pi<float>())}, 1));
  Manifold smoothed = Manifold::Smooth(surface).Refine(2);

  EXPECT_EQ(smoothed.Status(), Manifold::Error::NoError);
  EXPECT_EQ(smoothed.Genus(), -2);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("sinesurface.glb", smoothed.GetMeshGL(), {});
#endif
}
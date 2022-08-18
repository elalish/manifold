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
#include "meshIO.h"
#include "test.h"

using namespace manifold;

struct CubeVoid {
  __host__ __device__ float operator()(glm::vec3 p) const {
    const glm::vec3 min = glm::vec3(1) - p;
    const glm::vec3 max = p + glm::vec3(1);
    const float min3 = glm::min(min.x, glm::min(min.y, min.z));
    const float max3 = glm::min(max.x, glm::min(max.y, max.z));
    return -1.0f * glm::min(min3, max3);
  }
};

TEST(SDF, CubeVoid) {
  CubeVoid func;
  const SDF<CubeVoid> voidSDF(func);

  EXPECT_EQ(voidSDF({0, 0, 0}), -1);
  EXPECT_EQ(voidSDF({0, 0, 1}), 0);
  EXPECT_EQ(voidSDF({0, 1, 1}), 0);
  EXPECT_EQ(voidSDF({-1, 0, 0}), 0);
  EXPECT_EQ(voidSDF({1, 1, -1}), 0);
  EXPECT_EQ(voidSDF({2, 0, 0}), 1);
  EXPECT_EQ(voidSDF({2, -2, 0}), 1);
  EXPECT_EQ(voidSDF({-2, 2, 2}), 1);
}

TEST(SDF, Position) {
  CubeVoid func;
  const SDF<CubeVoid> voidSDF(func);

  const float size = 4;
  const float edgeLength = 0.5;

  Manifold cubeVoid(voidSDF.LevelSet(
      {glm::vec3(-size / 2), glm::vec3(size / 2)}, edgeLength));
  Box bounds = cubeVoid.BoundingBox();
  if (options.exportModels) ExportMesh("cubeVoid.gltf", cubeVoid.GetMesh(), {});

  EXPECT_TRUE(cubeVoid.IsManifold());
  EXPECT_EQ(cubeVoid.Genus(), -1);
  EXPECT_NEAR(bounds.min.x, -size / 2 - edgeLength / 2, 0.001);
  EXPECT_NEAR(bounds.min.y, -size / 2 - edgeLength / 2, 0.001);
  EXPECT_NEAR(bounds.min.z, -size / 2 - edgeLength / 2, 0.001);
  EXPECT_NEAR(bounds.max.x, size / 2 + edgeLength / 2, 0.001);
  EXPECT_NEAR(bounds.max.y, size / 2 + edgeLength / 2, 0.001);
  EXPECT_NEAR(bounds.max.z, size / 2 + edgeLength / 2, 0.001);
}
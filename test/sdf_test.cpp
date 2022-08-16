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
    const glm::vec3 max = p - glm::vec3(1);
    const float min3 = glm::max(min.x, glm::max(min.y, min.z));
    const float max3 = glm::max(max.x, glm::max(max.y, max.z));
    return glm::max(min3, max3);
  }
};

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
  EXPECT_NEAR(bounds.min.x, -1 - edgeLength / 2, 0.001);
  EXPECT_NEAR(bounds.min.y, -1 - edgeLength / 2, 0.001);
  EXPECT_NEAR(bounds.min.z, -1 - edgeLength / 2, 0.001);
  EXPECT_NEAR(bounds.max.x, 1 + edgeLength / 2, 0.001);
  EXPECT_NEAR(bounds.max.y, 1 + edgeLength / 2, 0.001);
  EXPECT_NEAR(bounds.max.z, 1 + edgeLength / 2, 0.001);
}
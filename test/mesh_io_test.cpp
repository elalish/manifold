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

#include <chrono>
#include <filesystem>

#include "../extras/meshIO.h"
#include "gtest/gtest.h"
#include "manifold/manifold.h"

using namespace manifold;

class MeshIORoundTrip : public testing::TestWithParam<const char*> {
 protected:
  std::filesystem::path directory;

  void SetUp() override {
    const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
    directory = std::filesystem::temp_directory_path() /
                ("manifold-mesh-io-" + std::to_string(id));
    ASSERT_TRUE(std::filesystem::create_directory(directory));
  }

  void TearDown() override { std::filesystem::remove_all(directory); }
};

TEST_P(MeshIORoundTrip, PreservesAsymmetricGeometryCoordinates) {
  const Manifold source = Manifold::Cube({2, 3, 5}).Translate({-1, 7, 13});
  const auto path = directory / (std::string("shape.") + GetParam());
  ExportMesh(path.string(), source.GetMeshGL(), {});
  const Manifold imported(ImportMesh(path.string(), true));
  ASSERT_EQ(imported.Status(), Manifold::Error::NoError);
  EXPECT_NEAR(imported.Volume(), source.Volume(), 1e-5);
  const Box expected = source.BoundingBox();
  const Box actual = imported.BoundingBox();
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_NEAR(actual.min[axis], expected.min[axis], 1e-5);
    EXPECT_NEAR(actual.max[axis], expected.max[axis], 1e-5);
  }
}

INSTANTIATE_TEST_SUITE_P(Assimp, MeshIORoundTrip,
                         testing::Values("obj", "stl", "glb"));

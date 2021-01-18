// Copyright 2019 Emmett Lalish
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

#include <random>

#include "gtest/gtest.h"
#include "manifold.h"
#include "meshIO.h"
#include "polygon.h"

namespace {

using namespace manifold;

void Identical(Mesh& mesh1, Mesh& mesh2) {
  ASSERT_EQ(mesh1.vertPos.size(), mesh2.vertPos.size());
  for (int i = 0; i < mesh1.vertPos.size(); ++i) {
    glm::vec3 v_in = mesh1.vertPos[i];
    glm::vec3 v_out = mesh2.vertPos[i];
    for (int j : {0, 1, 2}) ASSERT_NEAR(v_in[j], v_out[j], 0.0001);
  }
  ASSERT_EQ(mesh1.triVerts.size(), mesh2.triVerts.size());
  for (int i = 0; i < mesh1.triVerts.size(); ++i) {
    ASSERT_EQ(mesh1.triVerts[i], mesh2.triVerts[i]);
  }
}

void ExpectMeshes(Manifold& manifold,
                  const std::vector<std::pair<int, int>>& numVertTri) {
  EXPECT_TRUE(manifold.IsManifold());
  std::vector<Manifold> meshes = manifold.Decompose();
  ASSERT_EQ(meshes.size(), numVertTri.size());
  std::sort(meshes.begin(), meshes.end(),
            [](const Manifold& a, const Manifold& b) {
              return a.NumVert() != b.NumVert() ? a.NumVert() > b.NumVert()
                                                : a.NumTri() > b.NumTri();
            });
  for (int i = 0; i < meshes.size(); ++i) {
    EXPECT_TRUE(meshes[i].IsManifold());
    EXPECT_EQ(meshes[i].NumVert(), numVertTri[i].first);
    EXPECT_EQ(meshes[i].NumTri(), numVertTri[i].second);
  }
}

Polygons SquareHole(float xOffset = 0.0) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(2 + xOffset, 2), 0, Edge::kNoIdx},    //
      {glm::vec2(-2 + xOffset, 2), 0, Edge::kNoIdx},   //
      {glm::vec2(-2 + xOffset, -2), 0, Edge::kNoIdx},  //
      {glm::vec2(2 + xOffset, -2), 0, Edge::kNoIdx},   //
  });
  polys.push_back({
      {glm::vec2(-1 + xOffset, 1), 0, Edge::kNoIdx},   //
      {glm::vec2(1 + xOffset, 1), 0, Edge::kNoIdx},    //
      {glm::vec2(1 + xOffset, -1), 0, Edge::kNoIdx},   //
      {glm::vec2(-1 + xOffset, -1), 0, Edge::kNoIdx},  //
  });
  return polys;
}

}  // namespace

TEST(MeshIO, ReadWrite) {
  Mesh mesh = ImportMesh("data/gyroidpuzzle.ply");
  ExportMesh("data/gyroidpuzzle1.ply", mesh);
  Mesh mesh_out = ImportMesh("data/gyroidpuzzle1.ply");
  Identical(mesh, mesh_out);
}

/**
 * This tests that turning a mesh into a manifold and returning it to a mesh
 * produces a consistent result.
 */
TEST(Manifold, Regression) {
  Manifold manifold(ImportMesh("data/gyroidpuzzle.ply"));
  EXPECT_TRUE(manifold.IsManifold());

  Manifold mesh1 = manifold;
  mesh1.Translate(glm::vec3(5.0f));
  int num_overlaps = manifold.NumOverlaps(mesh1);
  ASSERT_EQ(num_overlaps, 237472);

  Mesh mesh_out = manifold.Extract();
  Manifold mesh2(mesh_out);
  Mesh mesh_out2 = mesh2.Extract();
  Identical(mesh_out, mesh_out2);
}

TEST(Manifold, Extract) {
  Manifold manifold = Manifold::Sphere(1);
  Mesh mesh_out = manifold.Extract();
  Manifold mesh2(mesh_out);
  Mesh mesh_out2 = mesh2.Extract();
  Identical(mesh_out, mesh_out2);
}

/**
 * ExpectMeshes performs a decomposition, so this test ensures that compose and
 * decompose are inverse operations.
 */
TEST(Manifold, Decompose) {
  std::vector<Manifold> meshList;
  meshList.push_back(Manifold::Tetrahedron());
  meshList.push_back(Manifold::Cube());
  meshList.push_back(Manifold::Octahedron());
  Manifold meshes = Manifold::Compose(meshList);

  ExpectMeshes(meshes, {{8, 12}, {6, 8}, {4, 4}});
}

/**
 * These tests check the various manifold constructors.
 */
TEST(Manifold, Sphere) {
  int n = 25;
  Manifold sphere = Manifold::Sphere(1.0f, 4 * n);
  EXPECT_TRUE(sphere.IsManifold());
  EXPECT_EQ(sphere.NumTri(), n * n * 8);
}

TEST(Manifold, Extrude) {
  Polygons polys = SquareHole();
  Manifold donut = Manifold::Extrude(polys, 1.0f, 3);
  EXPECT_TRUE(donut.IsManifold());
  EXPECT_EQ(donut.Genus(), 1);
  auto prop = donut.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 12.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 48.0f);
}

TEST(Manifold, ExtrudeCone) {
  Polygons polys = SquareHole();
  Manifold donut = Manifold::Extrude(polys, 1.0f, 0, 0, glm::vec2(0.0f));
  EXPECT_TRUE(donut.IsManifold());
  EXPECT_EQ(donut.Genus(), 0);
  EXPECT_FLOAT_EQ(donut.GetProperties().volume, 4.0f);
}

TEST(Manifold, Revolve) {
  Polygons polys = SquareHole();
  Manifold vug = Manifold::Revolve(polys, 48);
  EXPECT_TRUE(vug.IsManifold());
  EXPECT_EQ(vug.Genus(), -1);
  auto prop = vug.GetProperties();
  EXPECT_NEAR(prop.volume, 14.0f * glm::pi<float>(), 0.2f);
  EXPECT_NEAR(prop.surfaceArea, 30.0f * glm::pi<float>(), 0.2f);
}

TEST(Manifold, Revolve2) {
  Polygons polys = SquareHole(2.0f);
  Manifold donutHole = Manifold::Revolve(polys, 48);
  EXPECT_TRUE(donutHole.IsManifold());
  EXPECT_EQ(donutHole.Genus(), 0);
  auto prop = donutHole.GetProperties();
  EXPECT_NEAR(prop.volume, 48.0f * glm::pi<float>(), 1.0f);
  EXPECT_NEAR(prop.surfaceArea, 96.0f * glm::pi<float>(), 1.0f);
}

/**
 * These tests verify the calculation of a manifold's geometric properties.
 */
TEST(Manifold, GetProperties) {
  Manifold cube = Manifold::Cube();
  EXPECT_TRUE(cube.IsManifold());
  auto prop = cube.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 1.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 6.0f);

  cube.Scale(glm::vec3(-1.0f));
  prop = cube.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, -1.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 6.0f);
}

/**
 * The very simplest Boolean operation test.
 */
TEST(Manifold, BooleanTetra) {
  Manifold tetra = Manifold::Tetrahedron();
  EXPECT_TRUE(tetra.IsManifold());

  Manifold tetra2 = tetra;
  tetra2.Translate(glm::vec3(0.5f));
  Manifold result = tetra2 - tetra;

  ExpectMeshes(result, {{8, 12}});
}

/**
 * These tests check Boolean operations on coplanar faces.
 */
TEST(Manifold, SelfSubtract) {
  Manifold cube = Manifold::Cube();
  Manifold empty = cube - cube;
  EXPECT_TRUE(empty.IsManifold());
  EXPECT_TRUE(empty.IsEmpty());

  auto prop = empty.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 0.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 0.0f);
}

TEST(Manifold, Perturb) {
  Mesh tmp;
  tmp.vertPos = {{0.0f, 0.0f, 0.0f},
                 {0.0f, 1.0f, 0.0f},
                 {1.0f, 0.0f, 0.0f},
                 {0.0f, 0.0f, 1.0f}};
  tmp.triVerts = {{2, 0, 1}, {0, 3, 1}, {2, 3, 0}, {3, 2, 1}};
  Manifold corner(tmp);
  Manifold empty = corner - corner;
  EXPECT_TRUE(empty.IsManifold());
  // EXPECT_TRUE(empty.IsEmpty());

  auto prop = empty.GetProperties();
  // ExportMesh("perturb.ply", empty.Extract());
  EXPECT_FLOAT_EQ(prop.volume, 0.0f);
  // EXPECT_FLOAT_EQ(prop.surfaceArea, 0.0f);
}

TEST(Manifold, Coplanar) {
  Manifold cube = Manifold::Cylinder(1.0f, 1.0f);
  Manifold cube2 = cube;
  Manifold out = cube - cube2.Scale({0.5f, 0.5f, 1.0f})
                            .Rotate(0, 0, 15)
                            .Translate({0.25f, 0.25f, 0.0f});
  ExpectMeshes(out, {{60, 120}});
  EXPECT_EQ(out.Genus(), 1);
}

TEST(Manifold, MultiCoplanar) {
  Manifold cube = Manifold::Cube();
  Manifold cube2 = cube;
  Manifold first = cube - cube2.Translate({0.3f, 0.3f, 0.0f});
  cube.Translate({-0.3f, -0.3f, 0.0f});
  Manifold out = first - cube;
  EXPECT_TRUE(out.IsManifold());
  EXPECT_EQ(out.Genus(), -1);
  auto prop = out.GetProperties();
  EXPECT_NEAR(prop.volume, 0.18, 1e-5);
  EXPECT_NEAR(prop.surfaceArea, 2.76, 1e-5);
}

TEST(Manifold, EdgeUnion) {
  Manifold cubes = Manifold::Cube();
  auto propIn = cubes.GetProperties();
  Manifold cube2 = cubes;
  cubes += cube2.Translate({1, 1, 0});
  EXPECT_TRUE(cubes.IsManifold());
  EXPECT_EQ(cubes.Genus(), 0);
  auto prop = cubes.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 2 * propIn.volume);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 2 * propIn.surfaceArea);
}

TEST(Manifold, CornerUnion) {
  Manifold cubes = Manifold::Cube();
  auto propIn = cubes.GetProperties();
  Manifold cube2 = cubes;
  cubes += cube2.Translate({1, 1, 1});
  EXPECT_TRUE(cubes.IsManifold());
  EXPECT_EQ(cubes.Genus(), 0);
  auto prop = cubes.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 2 * propIn.volume);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 2 * propIn.surfaceArea);
}

/**
 * These tests verify that the spliting helper functions return meshes with
 * volumes that make sense.
 */
TEST(Manifold, Split) {
  Manifold cube = Manifold::Cube(glm::vec3(2.0f), true);
  Manifold oct = Manifold::Octahedron();
  oct.Translate(glm::vec3(0.0f, 0.0f, 1.0f));
  std::pair<Manifold, Manifold> splits = cube.Split(oct);
  EXPECT_TRUE(splits.first.IsManifold());
  EXPECT_TRUE(splits.second.IsManifold());
  EXPECT_FLOAT_EQ(splits.first.GetProperties().volume +
                      splits.second.GetProperties().volume,
                  cube.GetProperties().volume);
}

TEST(Manifold, SplitByPlane) {
  Manifold cube = Manifold::Cube(glm::vec3(2.0f), true);
  cube.Translate({0.0f, 1.0f, 0.0f});
  cube.Rotate(90.0f, 0.0f, 0.0f);
  std::pair<Manifold, Manifold> splits =
      cube.SplitByPlane({0.0f, 0.0f, 1.0f}, 1.0f);
  EXPECT_TRUE(splits.first.IsManifold());
  EXPECT_TRUE(splits.second.IsManifold());
  EXPECT_NEAR(splits.first.GetProperties().volume,
              splits.second.GetProperties().volume, 1e-5);
}

TEST(Manifold, SplitByPlane60) {
  Manifold cube = Manifold::Cube(glm::vec3(2.0f), true);
  cube.Translate({0.0f, 1.0f, 0.0f});
  cube.Rotate(0.0f, 0.0f, -60.0f);
  cube.Translate({2.0f, 0.0f, 0.0f});
  float phi = 30.0f;
  std::pair<Manifold, Manifold> splits =
      cube.SplitByPlane({sind(phi), -cosd(phi), 0.0f}, 1.0f);
  EXPECT_TRUE(splits.first.IsManifold());
  EXPECT_TRUE(splits.second.IsManifold());
  EXPECT_NEAR(splits.first.GetProperties().volume,
              splits.second.GetProperties().volume, 1e-5);
}

/**
 * This tests that non-intersecting geometry is properly retained.
 */
TEST(Manifold, BooleanVug) {
  Manifold cube = Manifold::Cube(glm::vec3(4.0f), true);
  Manifold vug = cube - Manifold::Cube();

  EXPECT_EQ(vug.Genus(), -1);

  Manifold half = vug.SplitByPlane({0.0f, 0.0f, 1.0f}, -1.0f).first;
  EXPECT_TRUE(half.IsManifold());
  EXPECT_EQ(half.Genus(), -1);

  auto prop = half.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 4.0 * 4.0 * 3.0 - 1.0);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 16.0 * 2 + 12.0 * 4 + 6.0);
}

TEST(Manifold, BooleanEmpty) {
  Manifold cube = Manifold::Cube();
  float cubeVol = cube.GetProperties().volume;
  Manifold empty;

  EXPECT_EQ((cube + empty).GetProperties().volume, cubeVol);
  EXPECT_EQ((cube - empty).GetProperties().volume, cubeVol);
  EXPECT_TRUE((empty - cube).IsEmpty());
  EXPECT_TRUE((cube ^ empty).IsEmpty());
}

TEST(Manifold, BooleanWinding) {
  std::vector<Manifold> cubes;
  cubes.push_back(Manifold::Cube(glm::vec3(3.0f), true));
  cubes.push_back(Manifold::Cube(glm::vec3(2.0f), true));
  Manifold doubled = Manifold::Compose(cubes);

  Manifold cube = Manifold::Cube(glm::vec3(1.0f), true);
  PolygonParams().suppressErrors = true;
  EXPECT_THROW(cube ^= doubled, runtimeErr);
  PolygonParams().suppressErrors = false;
}

TEST(Manifold, BooleanNonIntersecting) {
  Manifold cube1 = Manifold::Cube();
  float vol1 = cube1.GetProperties().volume;
  Manifold cube2 = cube1;
  cube2.Scale(glm::vec3(2)).Translate({3, 0, 0});
  float vol2 = cube2.GetProperties().volume;

  EXPECT_EQ((cube1 + cube2).GetProperties().volume, vol1 + vol2);
  EXPECT_EQ((cube1 - cube2).GetProperties().volume, vol1);
  EXPECT_TRUE((cube1 ^ cube2).IsEmpty());
}

/**
 * These tests verify correct topology and geometry for complex boolean
 * operations between valid shapes with many faces.
 */
TEST(Manifold, BooleanSphere) {
  Manifold sphere = Manifold::Sphere(1.0f, 12);
  Manifold sphere2 = sphere;
  sphere2.Translate(glm::vec3(0.5));
  Manifold result = sphere - sphere2;

  ExpectMeshes(result, {{74, 144}});
}

TEST(Manifold, Boolean3) {
  Manifold gyroid(ImportMesh("data/gyroidpuzzle.ply"));
  EXPECT_TRUE(gyroid.IsManifold());

  Manifold gyroid2 = gyroid;
  gyroid2.Translate(glm::vec3(5.0f));
  Manifold result = gyroid + gyroid2;

  ExpectMeshes(result, {{31733, 63606}});
}
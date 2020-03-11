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

void CheckIdx(int idx, int dir) {
  EdgeIdx edge(idx, dir);
  ASSERT_EQ(edge.Idx(), idx);
  ASSERT_EQ(edge.Dir(), dir);
}

void ExpectMeshes(const Manifold& manifold,
                  const std::vector<std::pair<int, int>>& numVertTri) {
  ASSERT_TRUE(manifold.IsValid());
  std::vector<Manifold> meshes = manifold.Decompose();
  ASSERT_EQ(meshes.size(), numVertTri.size());
  std::sort(meshes.begin(), meshes.end(),
            [](const Manifold& a, const Manifold& b) {
              return a.NumVert() != b.NumVert() ? a.NumVert() > b.NumVert()
                                                : a.NumTri() > b.NumTri();
            });
  for (int i = 0; i < meshes.size(); ++i) {
    EXPECT_TRUE(meshes[i].IsValid());
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

TEST(Manifold, EdgeIdx) {
  CheckIdx(0, 1);
  CheckIdx(0, -1);
  CheckIdx(1, 1);
  CheckIdx(1, -1);
}

/**
 * This tests that turning a mesh into a manifold and returning it to a mesh
 * produces a consistent result.
 */
TEST(Manifold, Regression) {
  Manifold manifold(ImportMesh("data/gyroidpuzzle.ply"));
  ASSERT_TRUE(manifold.IsValid());

  Manifold mesh1 = manifold.DeepCopy();
  mesh1.Translate(glm::vec3(5.0f));
  int num_overlaps = manifold.NumOverlaps(mesh1);
  ASSERT_EQ(num_overlaps, 237472);

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
  ASSERT_TRUE(meshes.IsValid());

  ExpectMeshes(meshes, {{8, 12}, {6, 8}, {4, 4}});
}

/**
 * These tests check the various manifold constructors.
 */
TEST(Manifold, Sphere) {
  int n = 25;
  Manifold sphere = Manifold::Sphere(1.0f, 4 * n);
  ASSERT_TRUE(sphere.IsValid());
  EXPECT_EQ(sphere.NumTri(), n * n * 8);
}

TEST(Manifold, Extrude) {
  Polygons polys = SquareHole();
  Manifold donut = Manifold::Extrude(polys, 1.0f, 3);
  ASSERT_TRUE(donut.IsValid());
  EXPECT_EQ(donut.Genus(), 1);
  EXPECT_FLOAT_EQ(donut.Volume(), 12.0f);
  EXPECT_FLOAT_EQ(donut.SurfaceArea(), 48.0f);
}

TEST(Manifold, ExtrudeCone) {
  Polygons polys = SquareHole();
  Manifold donut = Manifold::Extrude(polys, 1.0f, 0, 0, glm::vec2(0.0f));
  ASSERT_TRUE(donut.IsValid());
  EXPECT_EQ(donut.Genus(), 0);
  EXPECT_FLOAT_EQ(donut.Volume(), 4.0f);
}

TEST(Manifold, Revolve) {
  Polygons polys = SquareHole();
  Manifold vug = Manifold::Revolve(polys, 48);
  ASSERT_TRUE(vug.IsValid());
  EXPECT_EQ(vug.Genus(), -1);
  EXPECT_NEAR(vug.Volume(), 14.0f * glm::pi<float>(), 0.2f);
  EXPECT_NEAR(vug.SurfaceArea(), 30.0f * glm::pi<float>(), 0.2f);
}

TEST(Manifold, Revolve2) {
  Polygons polys = SquareHole(2.0f);
  Manifold donutHole = Manifold::Revolve(polys, 48);
  ASSERT_TRUE(donutHole.IsValid());
  EXPECT_EQ(donutHole.Genus(), 0);
  EXPECT_NEAR(donutHole.Volume(), 48.0f * glm::pi<float>(), 1.0f);
  EXPECT_NEAR(donutHole.SurfaceArea(), 96.0f * glm::pi<float>(), 1.0f);
}

/**
 * These tests verify the calculation of a manifold's geometric properties.
 */
TEST(Manifold, Volume) {
  Manifold cube = Manifold::Cube();
  ASSERT_TRUE(cube.IsValid());
  float vol = cube.Volume();
  EXPECT_FLOAT_EQ(vol, 1.0f);

  cube.Scale(glm::vec3(-1.0f));
  vol = cube.Volume();
  EXPECT_FLOAT_EQ(vol, -1.0f);
}

TEST(Manifold, SurfaceArea) {
  Manifold cube = Manifold::Cube();
  ASSERT_TRUE(cube.IsValid());
  float area = cube.SurfaceArea();
  EXPECT_FLOAT_EQ(area, 6.0f);

  cube.Scale(glm::vec3(-1.0f));
  area = cube.SurfaceArea();
  EXPECT_FLOAT_EQ(area, 6.0f);
}

/**
 * The very simplest Boolean operation test.
 */
TEST(Manifold, BooleanTetra) {
  Manifold::SetExpectGeometry(true);
  Manifold tetra = Manifold::Tetrahedron();
  ASSERT_TRUE(tetra.IsValid());

  Manifold tetra2 = tetra.DeepCopy();
  tetra2.Translate(glm::vec3(0.5f));
  Manifold result = tetra2 - tetra;

  ExpectMeshes(result, {{8, 12}});
}

/**
 * These tests check Boolean operations on coplanar faces.
 */
TEST(Manifold, SelfSubtract) {
  Manifold::SetExpectGeometry(true);
  Manifold cube = Manifold::Cube();
  Manifold empty = cube - cube;
  EXPECT_TRUE(empty.IsValid());
  EXPECT_FLOAT_EQ(empty.Volume(), 0.0f);
  EXPECT_FLOAT_EQ(empty.SurfaceArea(), 0.0f);
  EXPECT_TRUE(empty.IsEmpty());
}

TEST(Manifold, Perturb) {
  Manifold::SetExpectGeometry(true);
  Mesh tmp;
  tmp.vertPos = {{0.0f, 0.0f, 0.0f},
                 {0.0f, 1.0f, 0.0f},
                 {1.0f, 0.0f, 0.0f},
                 {0.0f, 0.0f, 1.0f}};
  tmp.triVerts = {{2, 0, 1}, {0, 3, 1}, {2, 3, 0}, {3, 2, 1}};
  std::vector<Manifold> meshList;
  meshList.push_back(tmp);
  meshList.push_back(meshList[0].DeepCopy());
  Manifold out = meshList[1] - meshList[0];
  EXPECT_TRUE(out.IsEmpty());
}

TEST(Manifold, Coplanar) {
  Manifold::SetExpectGeometry(true);
  Manifold cube = Manifold::Cylinder(1.0f, 1.0f);
  Manifold cube2 = cube.DeepCopy();
  Manifold out = cube -
                 cube2.Scale({0.5f, 0.5f, 1.0f})
                     .Rotate(0, 0, 15)
                     .Translate({0.25f, 0.25f, 0.0f});
}

TEST(Manifold, MultiCoplanar) {
  Manifold::SetExpectGeometry(true);
  Manifold cube = Manifold::Cube();
  Manifold cube2 = cube.DeepCopy();
  Manifold out = cube - cube2.Translate({0.3f, 0.3f, 0.0f});
  out = out - cube.Translate({-0.3f, -0.3f, 0.0f});
  EXPECT_EQ(out.Genus(), -1);
}

/**
 * These tests verify that the spliting helper functions return meshes with
 * volumes that make sense.
 */
TEST(Manifold, Split) {
  Manifold::SetExpectGeometry(true);
  Manifold cube = Manifold::Cube(glm::vec3(2.0f), true);
  Manifold oct = Manifold::Octahedron();
  oct.Translate(glm::vec3(0.0f, 0.0f, 1.0f));
  std::pair<Manifold, Manifold> splits = cube.Split(oct);
  EXPECT_FLOAT_EQ(splits.first.Volume() + splits.second.Volume(),
                  cube.Volume());
}

TEST(Manifold, SplitByPlane) {
  Manifold::SetExpectGeometry(true);
  Manifold cube = Manifold::Cube(glm::vec3(2.0f), true);
  cube.Translate({0.0f, 1.0f, 0.0f});
  cube.Rotate(90.0f, 0.0f, 0.0f);
  std::pair<Manifold, Manifold> splits =
      cube.SplitByPlane({0.0f, 0.0f, 1.0f}, 1.0f);
  EXPECT_NEAR(splits.first.Volume(), splits.second.Volume(), 1e-5);
}

TEST(Manifold, SplitByPlane60) {
  Manifold::SetExpectGeometry(true);
  Manifold cube = Manifold::Cube(glm::vec3(2.0f), true);
  cube.Translate({0.0f, 1.0f, 0.0f});
  cube.Rotate(0.0f, 0.0f, -60.0f);
  cube.Translate({2.0f, 0.0f, 0.0f});
  float phi = 30.0f;
  std::pair<Manifold, Manifold> splits =
      cube.SplitByPlane({sind(phi), -cosd(phi), 0.0f}, 1.0f);
  EXPECT_NEAR(splits.first.Volume(), splits.second.Volume(), 1e-5);
}

/**
 * This tests that non-intersecting geometry is properly retained.
 */
TEST(Manifold, BooleanVug) {
  Manifold::SetExpectGeometry(true);
  Manifold cube = Manifold::Cube(glm::vec3(4.0f), true);
  Manifold vug = cube - Manifold::Cube();

  EXPECT_EQ(vug.Genus(), -1);

  Manifold half = vug.SplitByPlane(glm::vec3(0.0f, 0.0f, 1.0f), -1.0f).first;

  EXPECT_EQ(half.Genus(), -1);
  EXPECT_FLOAT_EQ(half.Volume(), 4.0 * 4.0 * 3.0 - 1.0);
  EXPECT_FLOAT_EQ(half.SurfaceArea(), 16.0 * 2 + 12.0 * 4 + 6.0);
}

/**
 * These tests verify correct topology and geometry for complex boolean
 * operations between valid shapes with many faces.
 */
TEST(Manifold, BooleanSphere) {
  Manifold::SetExpectGeometry(true);
  Manifold sphere = Manifold::Sphere(1.0f, 12);
  Manifold sphere2 = sphere.DeepCopy();
  sphere2.Translate(glm::vec3(0.5));
  Manifold result = sphere - sphere2;

  ExpectMeshes(result, {{74, 144}});
}

TEST(Manifold, Boolean3) {
  Manifold::SetExpectGeometry(true);
  Manifold gyroid(ImportMesh("data/gyroidpuzzle.ply"));
  ASSERT_TRUE(gyroid.IsValid());

  Manifold gyroid2 = gyroid.DeepCopy();
  gyroid2.Translate(glm::vec3(5.0f));
  Manifold result = gyroid + gyroid2;

  ExpectMeshes(result, {{31733, 63606}});
}

/**
 * These tests create self-intersecting geometry using the Compose() method with
 * overlapping shapes. Though we cannot expect the resulting Boolean products to
 * be geometrically valid, we can expect the topology to be sane, and by testing
 * the number of vertices and triangles produced, we verify that the
 * triangulation has not produced strange results.
 */
TEST(Manifold, BooleanSelfIntersecting) {
  Manifold::SetExpectGeometry(false);
  std::vector<Manifold> meshList;
  meshList.push_back(Manifold::Tetrahedron());
  meshList.push_back(Manifold::Tetrahedron());
  meshList[1].Translate(glm::vec3(0, 0, 0.25));
  Manifold tetras = Manifold::Compose(meshList);
  ASSERT_TRUE(tetras.IsValid());

  meshList[0].Translate(glm::vec3(0, 0, 0.5f));
  Manifold result = meshList[0] - tetras;

  ExpectMeshes(result, {{8, 12}, {4, 4}});
}

TEST(Manifold, BooleanSelfIntersectingAlt) {
  Manifold::SetExpectGeometry(false);
  std::vector<Manifold> meshList;
  meshList.push_back(Manifold::Tetrahedron());
  meshList.push_back(Manifold::Tetrahedron());
  meshList[1].Translate(glm::vec3(0, 0, 0.5));
  Manifold tetras = Manifold::Compose(meshList);
  ASSERT_TRUE(tetras.IsValid());

  meshList[0].Translate(glm::vec3(0, 0, -0.5f));
  Manifold result = meshList[0] - tetras;

  ExpectMeshes(result, {{8, 12}, {4, 4}});
}

TEST(Manifold, BooleanWinding) {
  Manifold::SetExpectGeometry(false);
  std::vector<Manifold> meshList;
  meshList.push_back(Manifold::Tetrahedron());
  meshList.push_back(Manifold::Tetrahedron());
  meshList[1].Translate(glm::vec3(0.25));
  Manifold tetras = Manifold::Compose(meshList);
  ASSERT_TRUE(tetras.IsValid());

  meshList[0].Translate(glm::vec3(-0.25f));
  Manifold result = tetras - meshList[0];

  ExpectMeshes(result, {{11, 18}, {11, 18}});
}

/**
 * These tests take the worst (most self-intersected) general-position geometry
 * and perform an intersection to test that the topology stays sane in extremely
 * complex and unexpected situations.
 */
TEST(Manifold, BooleanHorrible) {
  Manifold::SetExpectGeometry(false);
  Manifold random = Manifold::Sphere(1.0f, 8);
  std::mt19937 gen(12345);  // Standard mersenne_twister_engine
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  random.Warp([&dis, &gen](glm::vec3& v) {
    v = glm::vec3(dis(gen), dis(gen), dis(gen));
  });
  Manifold random2 = random.DeepCopy();
  random2.Rotate(90);
  Manifold result = random ^ random2;
  EXPECT_TRUE(result.IsValid());

  Manifold::SetExpectGeometry(true);
  Manifold::SetSuppressErrors(true);
  EXPECT_THROW(result = random ^ random2, runtimeErr);
  Manifold::SetSuppressErrors(false);
}

TEST(Manifold, BooleanHorrible2) {
  Manifold::SetExpectGeometry(false);
  Manifold random = Manifold::Sphere(1.0f, 32);
  std::mt19937 gen(54321);  // Standard mersenne_twister_engine
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  random.Warp([&dis, &gen](glm::vec3& v) {
    v = glm::vec3(dis(gen), dis(gen), dis(gen));
  });
  Manifold random2 = random.DeepCopy();
  random2.Rotate(90);
  Manifold result = random ^ random2;
  EXPECT_TRUE(result.IsValid());
}

/**
 * This test is the most extreme of all, taking random geometry and projecting
 * it to a plane to ensure that it is not in general position (lots of coplanar
 * geometry). Two of these planes are rotated and intersected and verified to
 * produce a line. Note that due to the rotations, none of the points are
 * exactly coplanar, but within rounding error, which verifies the topology is
 * still correct even when Euclidean geometry gives inconsistent results due to
 * floating point error.
 */
TEST(Manifold, BooleanHorriblePlanar) {
  Manifold::SetExpectGeometry(false);
  Manifold random = Manifold::Sphere(1.0f, 32);
  std::mt19937 gen(654321);  // Standard mersenne_twister_engine
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  random.Warp(
      [&dis, &gen](glm::vec3& v) { v = glm::vec3(dis(gen), dis(gen), 0.0f); });
  Manifold random2 = random.DeepCopy();
  float a = 0.2;
  float phi = asin(a);
  random.Rotate(0, glm::degrees(-phi));
  random2.Rotate(glm::degrees(phi));
  Manifold result = random ^ random2;
  result.Rotate(0, 0, 45).Rotate(glm::degrees(atan(sqrt(2.0f) / tan(phi))));
  EXPECT_TRUE(result.IsEmpty());
  // Box BB = result.BoundingBox();
  // float tol = 1e-7;
  // EXPECT_NEAR(BB.Center().x, 0.0f, tol);
  // EXPECT_NEAR(BB.Center().y, 0.0f, tol);
  // EXPECT_NEAR(BB.Size().x, 0.0f, tol);
  // EXPECT_NEAR(BB.Size().y, 0.0f, tol);
  // EXPECT_GT(BB.Size().z, 1.0f);
  // EXPECT_LT(BB.Size().z, 4.0f);
}
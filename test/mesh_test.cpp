// Copyright 2021 Emmett Lalish
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

void Identical(glm::vec3 v0, glm::vec3 v1) {
  for (int j : {0, 1, 2}) ASSERT_NEAR(v0[j], v1[j], 0.0001);
}

void Identical(Mesh& mesh1, Mesh& mesh2) {
  ASSERT_EQ(mesh1.vertPos.size(), mesh2.vertPos.size());
  for (int i = 0; i < mesh1.vertPos.size(); ++i)
    Identical(mesh1.vertPos[i], mesh2.vertPos[i]);

  ASSERT_EQ(mesh1.triVerts.size(), mesh2.triVerts.size());
  for (int i = 0; i < mesh1.triVerts.size(); ++i)
    ASSERT_EQ(mesh1.triVerts[i], mesh2.triVerts[i]);
}

void Related(const Manifold& out, const Mesh& input) {
  Mesh output = out.Extract();
  MeshRelation relation = out.GetMeshRelation();
  for (int i = 0; i < out.NumTri(); ++i) {
    int inTri = relation.triBary[i].tri;
    ASSERT_LT(inTri, input.triVerts.size());
    for (int j : {0, 1, 2}) {
      int v = relation.triBary[i].vertBary[j];
      if (v < 0) {
        Identical(output.vertPos[output.triVerts[i][j]],
                  input.vertPos[input.triVerts[inTri][j]]);
      } else {
        ASSERT_LT(v, relation.barycentric.size());
        glm::vec3 uvw = relation.barycentric[v];
        glm::vec3 vPos = uvw[0] * input.vertPos[input.triVerts[inTri][0]] +
                         uvw[1] * input.vertPos[input.triVerts[inTri][1]] +
                         uvw[2] * input.vertPos[input.triVerts[inTri][2]];
        Identical(output.vertPos[output.triVerts[i][j]], vPos);
      }
    }
  }
}

void ExpectMeshes(const Manifold& manifold,
                  const std::vector<std::pair<int, int>>& numVertTri) {
  EXPECT_TRUE(manifold.IsManifold());
  EXPECT_TRUE(manifold.MatchesTriNormals());
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
      {glm::vec2(2 + xOffset, 2), 0},    //
      {glm::vec2(-2 + xOffset, 2), 0},   //
      {glm::vec2(-2 + xOffset, -2), 0},  //
      {glm::vec2(2 + xOffset, -2), 0},   //
  });
  polys.push_back({
      {glm::vec2(-1 + xOffset, 1), 0},   //
      {glm::vec2(1 + xOffset, 1), 0},    //
      {glm::vec2(1 + xOffset, -1), 0},   //
      {glm::vec2(-1 + xOffset, -1), 0},  //
  });
  return polys;
}

}  // namespace

TEST(MeshIO, ReadWrite) {
  Mesh mesh = ImportMesh("data/gyroidpuzzle.ply");
  ExportMesh("data/gyroidpuzzle1.ply", mesh, {});
  Mesh mesh_out = ImportMesh("data/gyroidpuzzle1.ply");
  Identical(mesh, mesh_out);
}

/**
 * This tests that turning a mesh into a manifold and returning it to a mesh
 * produces a consistent result.
 */
TEST(Manifold, Extract) {
  Manifold manifold = Manifold::Sphere(1);
  Mesh mesh_out = manifold.Extract();
  Manifold mesh2(mesh_out);
  Mesh mesh_out2 = mesh2.Extract();
  Identical(mesh_out, mesh_out2);
}

TEST(Manifold, Regression) {
  Manifold manifold(ImportMesh("data/gyroidpuzzle.ply"));
  EXPECT_TRUE(manifold.IsManifold());

  Manifold mesh1 = manifold;
  mesh1.Translate(glm::vec3(5.0f));
  int num_overlaps = manifold.NumOverlaps(mesh1);
  ASSERT_EQ(num_overlaps, 222653);

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
  meshList.push_back(Manifold::Sphere(1, 4));
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

TEST(Manifold, Normals) {
  Mesh cube = Manifold::Cube(glm::vec3(1), true).Extract();
  const int nVert = cube.vertPos.size();
  for (int i = 0; i < nVert; ++i) {
    glm::vec3 v = glm::normalize(cube.vertPos[i]);
    glm::vec3& n = cube.vertNormal[i];
    EXPECT_FLOAT_EQ(v.x, n.x);
    EXPECT_FLOAT_EQ(v.y, n.y);
    EXPECT_FLOAT_EQ(v.z, n.z);
  }
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

TEST(Manifold, Smooth) {
  Manifold tet = Manifold::Tetrahedron();
  Manifold smooth = Manifold::Smooth(tet.Extract());
  smooth.Refine(100);
  ExpectMeshes(smooth, {{20002, 40000}});
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 17.38, 0.1);
  EXPECT_NEAR(prop.surfaceArea, 33.38, 0.1);
  // ExportMesh("smoothTet.gltf", smooth.Extract());
}

TEST(Manifold, ManualSmooth) {
  // Unit Octahedron
  const Mesh oct = Manifold::Sphere(1, 4).Extract();
  Mesh smooth = Manifold::Smooth(oct).Extract();
  // Sharpen the edge from vert 4 to 5
  smooth.halfedgeTangent[6] = {0, 0, 0, 1};
  smooth.halfedgeTangent[22] = {0, 0, 0, 1};
  smooth.halfedgeTangent[16] = {0, 0, 0, 1};
  smooth.halfedgeTangent[18] = {0, 0, 0, 1};
  Manifold interp(smooth);
  interp.Refine(100);

  ExpectMeshes(interp, {{40002, 80000}});
  auto prop = interp.GetProperties();
  EXPECT_NEAR(prop.volume, 3.53, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 11.39, 0.01);

  const Mesh out = interp.Extract();
  ExportOptions options;
  options.faceted = false;
  options.mat.roughness = 0.1;

  options.mat.vertColor.resize(interp.NumVert());
  MeshRelation rel = interp.GetMeshRelation();
  const glm::vec4 red(1, 0, 0, 1);
  const glm::vec4 purple(1, 0, 1, 1);
  for (int tri = 0; tri < interp.NumTri(); ++tri) {
    for (int i : {0, 1, 2}) {
      const glm::vec3& uvw = rel.barycentric[rel.triBary[tri].vertBary[i]];
      const float alpha = glm::min(uvw[0], glm::min(uvw[1], uvw[2]));
      options.mat.vertColor[out.triVerts[tri][i]] =
          glm::mix(purple, red, glm::smoothstep(0.0f, 0.2f, alpha));
    }
  }
  // ExportMesh("sharpenedSphere.gltf", out, options);
}

TEST(Manifold, Csaszar) {
  Manifold csaszar = Manifold::Smooth(ImportMesh("data/Csaszar.ply"));
  csaszar.Refine(100);
  ExpectMeshes(csaszar, {{70000, 140000}});
  auto prop = csaszar.GetProperties();
  EXPECT_NEAR(prop.volume, 84699, 10);
  EXPECT_NEAR(prop.surfaceArea, 14796, 10);

  // const Mesh out = csaszar.Extract();
  // ExportOptions options;
  // options.faceted = false;
  // options.mat.roughness = 0.1;

  // options.mat.vertColor.resize(csaszar.NumVert());
  // MeshRelation rel = csaszar.GetMeshRelation();
  // const glm::vec4 blue(0, 0, 1, 1);
  // const glm::vec4 yellow(1, 1, 0, 1);
  // for (int tri = 0; tri < csaszar.NumTri(); ++tri) {
  //   for (int i : {0, 1, 2}) {
  //     const glm::vec3& uvw = rel.barycentric[rel.triBary[tri].vertBary[i]];
  //     const float alpha = glm::min(uvw[0], glm::min(uvw[1], uvw[2]));
  //     options.mat.vertColor[out.triVerts[tri][i]] =
  //         glm::mix(yellow, blue, glm::smoothstep(0.0f, 0.2f, alpha));
  //   }
  // }
  // ExportMesh("smoothCsaszar.gltf", out, options);
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

TEST(Manifold, Precision) {
  Manifold cube = Manifold::Cube();
  EXPECT_FLOAT_EQ(cube.Precision(), kTolerance);
  cube.Scale({0.1, 1, 10});
  EXPECT_FLOAT_EQ(cube.Precision(), 10 * kTolerance);
  cube.Translate({-100, -10, -1});
  EXPECT_FLOAT_EQ(cube.Precision(), 100 * kTolerance);
}

/**
 * Testing more advanced Manifold operations.
 */

TEST(Manifold, MeshRelation) {
  Mesh input = ImportMesh("data/Csaszar.ply");
  Manifold csaszar(input);
  Related(csaszar, input);
  Mesh sortedInput = csaszar.Extract();
  csaszar.Refine(4);
  Related(csaszar, sortedInput);
}

/**
 * The very simplest Boolean operation test.
 */
TEST(Boolean, Tetra) {
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
TEST(Boolean, SelfSubtract) {
  Manifold cube = Manifold::Cube();
  Manifold empty = cube - cube;
  EXPECT_TRUE(empty.IsManifold());
  EXPECT_TRUE(empty.IsEmpty());

  auto prop = empty.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 0.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 0.0f);
}

TEST(Boolean, Perturb) {
  Mesh tmp;
  tmp.vertPos = {{0.0f, 0.0f, 0.0f},
                 {0.0f, 1.0f, 0.0f},
                 {1.0f, 0.0f, 0.0f},
                 {0.0f, 0.0f, 1.0f}};
  tmp.triVerts = {{2, 0, 1}, {0, 3, 1}, {2, 3, 0}, {3, 2, 1}};
  Manifold corner(tmp);
  Manifold empty = corner - corner;
  EXPECT_TRUE(empty.IsManifold());
  EXPECT_TRUE(empty.IsEmpty());

  auto prop = empty.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 0.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 0.0f);
}

TEST(Boolean, Coplanar) {
  Manifold cube = Manifold::Cylinder(1.0f, 1.0f);
  Manifold cube2 = cube;
  Manifold out = cube - cube2.Scale({0.5f, 0.5f, 1.0f})
                            .Rotate(0, 0, 15)
                            .Translate({0.25f, 0.25f, 0.0f});
  ExpectMeshes(out, {{32, 64}});
  EXPECT_EQ(out.Genus(), 1);
  // ExportMesh("coplanar.gltf", out.Extract());
}

TEST(Boolean, MultiCoplanar) {
  Manifold cube = Manifold::Cube();
  Manifold cube2 = cube;
  Manifold first = cube - cube2.Translate({0.3f, 0.3f, 0.0f});
  cube.Translate({-0.3f, -0.3f, 0.0f});
  Manifold out = first - cube;
  EXPECT_TRUE(out.IsManifold());
  EXPECT_TRUE(out.MatchesTriNormals());
  EXPECT_EQ(out.Genus(), -1);
  auto prop = out.GetProperties();
  EXPECT_NEAR(prop.volume, 0.18, 1e-5);
  EXPECT_NEAR(prop.surfaceArea, 2.76, 1e-5);
}

TEST(Boolean, FaceUnion) {
  Manifold cubes = Manifold::Cube();
  Manifold cube2 = cubes;
  cubes += cube2.Translate({1, 0, 0});
  EXPECT_EQ(cubes.Genus(), 0);
  // TODO: This should be {12, 20} once CollapseDegenerates is restricted to
  // only degenerate triangles.
  ExpectMeshes(cubes, {{8, 12}});
  auto prop = cubes.GetProperties();
  EXPECT_NEAR(prop.volume, 2, 1e-5);
  EXPECT_NEAR(prop.surfaceArea, 10, 1e-5);
  // ExportMesh("faceUnion.gltf", cubes.Extract(), {});
}

TEST(Boolean, EdgeUnion) {
  Manifold cubes = Manifold::Cube();
  Manifold cube2 = cubes;
  cubes += cube2.Translate({1, 1, 0});
  ExpectMeshes(cubes, {{8, 12}, {8, 12}});
}

TEST(Boolean, EdgeUnion2) {
  Manifold tets = Manifold::Tetrahedron();
  Manifold cube2 = tets;
  tets.Translate({0, 0, -1});
  tets += cube2.Translate({0, 0, 1}).Rotate(0, 0, 90);
  ExpectMeshes(tets, {{4, 4}, {4, 4}});
}

TEST(Boolean, CornerUnion) {
  Manifold cubes = Manifold::Cube();
  Manifold cube2 = cubes;
  cubes += cube2.Translate({1, 1, 1});
  ExpectMeshes(cubes, {{8, 12}, {8, 12}});
}

/**
 * These tests verify that the spliting helper functions return meshes with
 * volumes that make sense.
 */
TEST(Boolean, Split) {
  Manifold cube = Manifold::Cube(glm::vec3(2.0f), true);
  Manifold oct = Manifold::Sphere(1, 4);
  oct.Translate(glm::vec3(0.0f, 0.0f, 1.0f));
  std::pair<Manifold, Manifold> splits = cube.Split(oct);
  EXPECT_TRUE(splits.first.IsManifold());
  EXPECT_TRUE(splits.first.MatchesTriNormals());
  EXPECT_TRUE(splits.second.IsManifold());
  EXPECT_TRUE(splits.second.MatchesTriNormals());
  EXPECT_FLOAT_EQ(splits.first.GetProperties().volume +
                      splits.second.GetProperties().volume,
                  cube.GetProperties().volume);
}

TEST(Boolean, SplitByPlane) {
  Manifold cube = Manifold::Cube(glm::vec3(2.0f), true);
  cube.Translate({0.0f, 1.0f, 0.0f});
  cube.Rotate(90.0f, 0.0f, 0.0f);
  std::pair<Manifold, Manifold> splits =
      cube.SplitByPlane({0.0f, 0.0f, 1.0f}, 1.0f);
  EXPECT_TRUE(splits.first.IsManifold());
  EXPECT_TRUE(splits.first.MatchesTriNormals());
  EXPECT_TRUE(splits.second.IsManifold());
  EXPECT_TRUE(splits.second.MatchesTriNormals());
  EXPECT_NEAR(splits.first.GetProperties().volume,
              splits.second.GetProperties().volume, 1e-5);

  Manifold first = cube.TrimByPlane({0.0f, 0.0f, 1.0f}, 1.0f);
  // Verify trim returns the same result as the first split by checking that
  // their bounding boxes contain each other, thus they are equal.
  EXPECT_TRUE(splits.first.BoundingBox().Contains(first.BoundingBox()));
  EXPECT_TRUE(first.BoundingBox().Contains(splits.first.BoundingBox()));
}

TEST(Boolean, SplitByPlane60) {
  Manifold cube = Manifold::Cube(glm::vec3(2.0f), true);
  cube.Translate({0.0f, 1.0f, 0.0f});
  cube.Rotate(0.0f, 0.0f, -60.0f);
  cube.Translate({2.0f, 0.0f, 0.0f});
  float phi = 30.0f;
  std::pair<Manifold, Manifold> splits =
      cube.SplitByPlane({sind(phi), -cosd(phi), 0.0f}, 1.0f);
  EXPECT_TRUE(splits.first.IsManifold());
  EXPECT_TRUE(splits.first.MatchesTriNormals());
  EXPECT_TRUE(splits.second.IsManifold());
  EXPECT_TRUE(splits.second.MatchesTriNormals());
  EXPECT_NEAR(splits.first.GetProperties().volume,
              splits.second.GetProperties().volume, 1e-5);
}

/**
 * This tests that non-intersecting geometry is properly retained.
 */
TEST(Boolean, Vug) {
  Manifold cube = Manifold::Cube(glm::vec3(4.0f), true);
  Manifold vug = cube - Manifold::Cube();

  EXPECT_EQ(vug.Genus(), -1);

  Manifold half = vug.SplitByPlane({0.0f, 0.0f, 1.0f}, -1.0f).first;
  EXPECT_TRUE(half.IsManifold());
  EXPECT_TRUE(half.MatchesTriNormals());
  EXPECT_EQ(half.Genus(), -1);

  auto prop = half.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 4.0 * 4.0 * 3.0 - 1.0);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 16.0 * 2 + 12.0 * 4 + 6.0);
}

TEST(Boolean, Empty) {
  Manifold cube = Manifold::Cube();
  float cubeVol = cube.GetProperties().volume;
  Manifold empty;

  EXPECT_EQ((cube + empty).GetProperties().volume, cubeVol);
  EXPECT_EQ((cube - empty).GetProperties().volume, cubeVol);
  EXPECT_TRUE((empty - cube).IsEmpty());
  EXPECT_TRUE((cube ^ empty).IsEmpty());
}

TEST(Boolean, Winding) {
  std::vector<Manifold> cubes;
  cubes.push_back(Manifold::Cube(glm::vec3(3.0f), true));
  cubes.push_back(Manifold::Cube(glm::vec3(2.0f), true));
  Manifold doubled = Manifold::Compose(cubes);

  Manifold cube = Manifold::Cube(glm::vec3(1.0f), true);
  EXPECT_TRUE((cube ^= doubled).IsManifold());
}

TEST(Boolean, NonIntersecting) {
  Manifold cube1 = Manifold::Cube();
  float vol1 = cube1.GetProperties().volume;
  Manifold cube2 = cube1;
  cube2.Scale(glm::vec3(2)).Translate({3, 0, 0});
  float vol2 = cube2.GetProperties().volume;

  EXPECT_EQ((cube1 + cube2).GetProperties().volume, vol1 + vol2);
  EXPECT_EQ((cube1 - cube2).GetProperties().volume, vol1);
  EXPECT_TRUE((cube1 ^ cube2).IsEmpty());
}

TEST(Boolean, Precision) {
  Manifold cube = Manifold::Cube();
  Manifold cube2 = cube;
  Manifold cube3 = cube;
  float distance = 100;
  float scale = distance * kTolerance;
  cube2.Scale(glm::vec3(scale)).Translate({distance, 0, 0});

  cube += cube2;
  ExpectMeshes(cube, {{8, 12}});

  cube3.Scale(glm::vec3(2 * scale)).Translate({distance, 0, 0});
  cube += cube3;
  ExpectMeshes(cube, {{8, 12}, {8, 12}});
}

TEST(Boolean, Precision2) {
  float scale = 1000;
  Manifold cube = Manifold::Cube(glm::vec3(scale));
  Manifold cube2 = cube;
  float distance = scale * (1 - kTolerance / 2);

  cube2.Translate(glm::vec3(-distance));
  EXPECT_TRUE((cube ^ cube2).IsEmpty());

  cube2.Translate(glm::vec3(scale * kTolerance));
  EXPECT_FALSE((cube ^ cube2).IsEmpty());
}

/**
 * These tests verify correct topology and geometry for complex boolean
 * operations between valid shapes with many faces.
 */
TEST(Boolean, Sphere) {
  Manifold sphere = Manifold::Sphere(1.0f, 12);
  Manifold sphere2 = sphere;
  sphere2.Translate(glm::vec3(0.5));
  Manifold result = sphere - sphere2;

  ExpectMeshes(result, {{74, 144}});
}

TEST(Boolean, Gyroid) {
  Manifold gyroid(ImportMesh("data/gyroidpuzzle.ply"));
  EXPECT_TRUE(gyroid.IsManifold());
  EXPECT_TRUE(gyroid.MatchesTriNormals());

  Manifold gyroid2 = gyroid;
  gyroid2.Translate(glm::vec3(5.0f));
  Manifold result = gyroid + gyroid2;
  // ExportMesh("gyroidUnion.gltf", result.Extract(), {});

  EXPECT_TRUE(result.IsManifold());
  EXPECT_TRUE(result.MatchesTriNormals());
  EXPECT_EQ(result.Decompose().size(), 1);
  auto prop = result.GetProperties();
  EXPECT_NEAR(prop.volume, 7692, 1);
  EXPECT_NEAR(prop.surfaceArea, 9642, 1);
}
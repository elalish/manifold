// Copyright 2021 The Manifold Authors.
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

#include "manifold.h"
#include "polygon.h"
#include "test.h"

using namespace manifold;

/**
 * The very simplest Boolean operation test.
 */
TEST(Boolean, Tetra) {
  Manifold tetra = Manifold::Tetrahedron();
  MeshGL tetraGL = WithPositionColors(tetra);
  tetra = Manifold(tetraGL);
  EXPECT_TRUE(!tetra.IsEmpty());

  Manifold tetra2 = tetra.Translate(glm::vec3(0.5f));
  Manifold result = tetra2 - tetra;

  ExpectMeshes(result, {{8, 12, 3, 11}});

  RelatedGL(result, {tetraGL});
}

TEST(Boolean, MeshGLRoundTrip) {
  Manifold cube = Manifold::Cube(glm::vec3(2));
  ASSERT_GE(cube.OriginalID(), 0);
  const MeshGL original = cube.GetMeshGL();

  Manifold result = cube + cube.Translate({1, 1, 0});

  ASSERT_LT(result.OriginalID(), 0);
  ExpectMeshes(result, {{18, 32}});
  RelatedGL(result, {original});

  MeshGL inGL = result.GetMeshGL();
  ASSERT_EQ(inGL.runOriginalID.size(), 2);
  const Manifold result2(inGL);

  ASSERT_LT(result2.OriginalID(), 0);
  ExpectMeshes(result2, {{16, 28}});
  RelatedGL(result2, {original});

  const MeshGL outGL = result2.GetMeshGL();
  ASSERT_EQ(outGL.runOriginalID.size(), 2);
}

TEST(Boolean, Normals) {
  MeshGL cubeGL = CubeSTL();
  cubeGL.Merge();
  const Manifold cube(cubeGL);
  const MeshGL sphereGL = WithNormals(Manifold::Sphere(60));
  const Manifold sphere(sphereGL);

  Manifold result =
      cube.Scale(glm::vec3(100)) -
      (sphere.Rotate(180) -
       sphere.Scale(glm::vec3(0.5)).Rotate(90).Translate({40, 40, 40}));

  RelatedGL(result, {cubeGL, sphereGL}, true, true);

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.faceted = false;
  opt.mat.roughness = 0;
  opt.mat.normalChannels = {3, 4, 5};
  if (options.exportModels)
    ExportMesh("normals.glb", result.GetMeshGL({3, 4, 5}), opt);
#endif

  MeshGL output = result.GetMeshGL({3, 4, 5});
  output.mergeFromVert.clear();
  output.mergeToVert.clear();
  output.Merge();
  Manifold roundTrip(output);

  RelatedGL(roundTrip, {cubeGL, sphereGL}, true, false);
}

TEST(Boolean, EmptyOriginal) {
  const Manifold cube = Manifold::Cube();
  const Manifold tet = Manifold::Tetrahedron();
  const Manifold result = tet - cube.Translate({3, 4, 5});
  const MeshGL mesh = result.GetMeshGL();
  ASSERT_EQ(mesh.runIndex.size(), 3);
  EXPECT_EQ(mesh.runIndex[0], 0);
  EXPECT_EQ(mesh.runIndex[1], mesh.triVerts.size());
  EXPECT_EQ(mesh.runIndex[2], mesh.triVerts.size());
  ASSERT_EQ(mesh.runOriginalID.size(), 2);
  EXPECT_EQ(mesh.runOriginalID[0], tet.OriginalID());
  EXPECT_EQ(mesh.runOriginalID[1], cube.OriginalID());
  ASSERT_EQ(mesh.runTransform.size(), 24);
  EXPECT_EQ(mesh.runTransform[9], 0);
  EXPECT_EQ(mesh.runTransform[10], 0);
  EXPECT_EQ(mesh.runTransform[11], 0);
  EXPECT_EQ(mesh.runTransform[12 + 9], 3);
  EXPECT_EQ(mesh.runTransform[12 + 10], 4);
  EXPECT_EQ(mesh.runTransform[12 + 11], 5);
}

TEST(Boolean, Mirrored) {
  Manifold cube = Manifold::Cube(glm::vec3(1)).Scale({1, -1, 1});
  EXPECT_TRUE(cube.MatchesTriNormals());

  Manifold cube2 = Manifold::Cube(glm::vec3(1)).Scale({0.5, -1, 0.5});
  Manifold result = cube - cube2;

  ExpectMeshes(result, {{12, 20}});

  auto prop = result.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 0.75);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 5.5);
}

TEST(Boolean, Cubes) {
  Manifold result = Manifold::Cube({1.2, 1, 1}, true).Translate({0, -0.5, 0.5});
  result += Manifold::Cube({1, 0.8, 0.5}).Translate({-0.5, 0, 0.5});
  result += Manifold::Cube({1.2, 0.1, 0.5}).Translate({-0.6, -0.1, 0});

  EXPECT_TRUE(result.MatchesTriNormals());
  EXPECT_LE(result.NumDegenerateTris(), 0);
  auto prop = result.GetProperties();
  EXPECT_NEAR(prop.volume, 1.6, 0.001);
  EXPECT_NEAR(prop.surfaceArea, 9.2, 0.01);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("cubes.glb", result.GetMesh(), {});
#endif
}

TEST(Boolean, NoRetainedVerts) {
  Manifold cube = Manifold::Cube(glm::vec3(1), true);
  Manifold oct = Manifold::Sphere(1, 4);
  EXPECT_NEAR(cube.GetProperties().volume, 1, 0.001);
  EXPECT_NEAR(oct.GetProperties().volume, 1.333, 0.001);
  EXPECT_NEAR((cube ^ oct).GetProperties().volume, 0.833, 0.001);
}

TEST(Boolean, PropertiesNoIntersection) {
  MeshGL cubeUV = CubeUV();
  Manifold m0(cubeUV);
  Manifold m1 = m0.Translate(glm::vec3(1.5));
  Manifold result = m0 + m1;
  EXPECT_EQ(result.NumProp(), 2);
  RelatedGL(result, {cubeUV});
}

TEST(Boolean, MixedProperties) {
  MeshGL cubeUV = CubeUV();
  Manifold m0(cubeUV);
  Manifold m1 = Manifold::Cube();
  Manifold result = m0 + m1.Translate(glm::vec3(0.5));
  EXPECT_EQ(result.NumProp(), 2);
  RelatedGL(result, {cubeUV, m1.GetMeshGL()});
}

TEST(Boolean, MixedNumProp) {
  MeshGL cubeUV = CubeUV();
  Manifold m0(cubeUV);
  Manifold m1 = Manifold::Cube();
  Manifold result =
      m0 + m1.SetProperties(1, [](float* prop, glm::vec3 p, const float* n) {
               prop[0] = 1;
             }).Translate(glm::vec3(0.5));
  EXPECT_EQ(result.NumProp(), 2);
  RelatedGL(result, {cubeUV, m1.GetMeshGL()});
}

TEST(Boolean, UnionDifference) {
  Manifold block = Manifold::Cube({1, 1, 1}, true) - Manifold::Cylinder(1, 0.5);
  Manifold result = block + block.Translate({0, 0, 1});
  float resultsize = result.GetProperties().volume;
  float blocksize = block.GetProperties().volume;
  EXPECT_NEAR(resultsize, blocksize * 2, 0.0001);
}

TEST(Boolean, TreeTransforms) {
  auto a = (Manifold::Cube({1, 1, 1}) + Manifold::Cube({1, 1, 1}))
               .Translate({1, 0, 0});
  auto b = (Manifold::Cube({1, 1, 1}) + Manifold::Cube({1, 1, 1}));

  EXPECT_FLOAT_EQ((a + b).GetProperties().volume, 2);
}

TEST(Boolean, CreatePropertiesSlow) {
  Manifold a = Manifold::Sphere(10, 1024).SetProperties(
      3, [](float* newprop, glm::vec3 pos, const float* old) {
        for (int i = 0; i < 3; i++) newprop[i] = 0;
      });
  Manifold b = Manifold::Sphere(10, 1024).Translate({5, 0, 0});
  Manifold result = a + b;
  EXPECT_EQ(result.NumProp(), 3);
}

/**
 * These tests check Boolean operations on coplanar faces.
 */
TEST(Boolean, SelfSubtract) {
  Manifold cube = Manifold::Cube();
  Manifold empty = cube - cube;
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
  EXPECT_TRUE(empty.IsEmpty());

  auto prop = empty.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 0.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 0.0f);
}

TEST(Boolean, Coplanar) {
  Manifold cylinder = Manifold::Cylinder(1.0f, 1.0f);
  MeshGL cylinderGL = WithPositionColors(cylinder);
  cylinder = Manifold(cylinderGL);

  Manifold cylinder2 = cylinder.Scale({0.8f, 0.8f, 1.0f}).Rotate(0, 0, 185);
  Manifold out = cylinder - cylinder2;
  ExpectMeshes(out, {{32, 64, 3, 48}});
  EXPECT_EQ(out.NumDegenerateTris(), 0);
  EXPECT_EQ(out.Genus(), 1);

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorChannels = glm::ivec4(3, 4, 5, -1);
  if (options.exportModels) ExportMesh("coplanar.glb", out.GetMeshGL(), opt);
#endif

  RelatedGL(out, {cylinderGL});
}

/**
 * Colinear edges are not collapsed like above due to non-coplanar properties.
 */
TEST(Boolean, CoplanarProp) {
  Manifold cylinder = Manifold::Cylinder(1.0f, 1.0f);
  MeshGL cylinderGL = WithIndexColors(cylinder.GetMesh());
  cylinder = Manifold(cylinderGL);

  Manifold cylinder2 = cylinder.Scale({0.8f, 0.8f, 1.0f}).Rotate(0, 0, 185);
  Manifold out = cylinder - cylinder2;
  ExpectMeshes(out, {{52, 104, 3, 88}});
  EXPECT_EQ(out.NumDegenerateTris(), 0);
  EXPECT_EQ(out.Genus(), 1);

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorChannels = glm::ivec4(3, 4, 5, -1);
  if (options.exportModels) ExportMesh("coplanar.glb", out.GetMeshGL(), opt);
#endif

  RelatedGL(out, {cylinderGL});
}

TEST(Boolean, MultiCoplanar) {
  Manifold cube = Manifold::Cube();
  Manifold first = cube - cube.Translate({0.3f, 0.3f, 0.0f});
  cube = cube.Translate({-0.3f, -0.3f, 0.0f});
  Manifold out = first - cube;
  CheckStrictly(out);
  EXPECT_EQ(out.Genus(), -1);
  auto prop = out.GetProperties();
  EXPECT_NEAR(prop.volume, 0.18, 1e-5);
  EXPECT_NEAR(prop.surfaceArea, 2.76, 1e-5);
}

TEST(Boolean, FaceUnion) {
  Manifold cubes = Manifold::Cube();
  cubes += cubes.Translate({1, 0, 0});
  EXPECT_EQ(cubes.Genus(), 0);
  ExpectMeshes(cubes, {{12, 20}});
  auto prop = cubes.GetProperties();
  EXPECT_NEAR(prop.volume, 2, 1e-5);
  EXPECT_NEAR(prop.surfaceArea, 10, 1e-5);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("faceUnion.glb", cubes.GetMesh(), {});
#endif
}

TEST(Boolean, EdgeUnion) {
  Manifold cubes = Manifold::Cube();
  cubes += cubes.Translate({1, 1, 0});
  ExpectMeshes(cubes, {{8, 12}, {8, 12}});
}

TEST(Boolean, EdgeUnion2) {
  Manifold tets = Manifold::Tetrahedron();
  Manifold cube2 = tets;
  tets = tets.Translate({0, 0, -1});
  tets += cube2.Translate({0, 0, 1}).Rotate(0, 0, 90);
  ExpectMeshes(tets, {{4, 4}, {4, 4}});
}

TEST(Boolean, CornerUnion) {
  Manifold cubes = Manifold::Cube();
  cubes += cubes.Translate({1, 1, 1});
  ExpectMeshes(cubes, {{8, 12}, {8, 12}});
}

/**
 * These tests verify that the spliting helper functions return meshes with
 * volumes that make sense.
 */
TEST(Boolean, Split) {
  Manifold cube = Manifold::Cube(glm::vec3(2.0f), true);
  Manifold oct = Manifold::Sphere(1, 4).Translate(glm::vec3(0.0f, 0.0f, 1.0f));
  std::pair<Manifold, Manifold> splits = cube.Split(oct);
  CheckStrictly(splits.first);
  CheckStrictly(splits.second);
  EXPECT_FLOAT_EQ(splits.first.GetProperties().volume +
                      splits.second.GetProperties().volume,
                  cube.GetProperties().volume);
}

TEST(Boolean, SplitByPlane) {
  Manifold cube = Manifold::Cube(glm::vec3(2.0f), true);
  cube = cube.Translate({0.0f, 1.0f, 0.0f});
  cube = cube.Rotate(90.0f, 0.0f, 0.0f);
  std::pair<Manifold, Manifold> splits =
      cube.SplitByPlane({0.0f, 0.0f, 1.0f}, 1.0f);
  CheckStrictly(splits.first);
  CheckStrictly(splits.second);
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
  cube = cube.Translate({0.0f, 1.0f, 0.0f});
  cube = cube.Rotate(0.0f, 0.0f, -60.0f);
  cube = cube.Translate({2.0f, 0.0f, 0.0f});
  float phi = 30.0f;
  std::pair<Manifold, Manifold> splits =
      cube.SplitByPlane({sind(phi), -cosd(phi), 0.0f}, 1.0f);
  CheckStrictly(splits.first);
  CheckStrictly(splits.second);
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
  CheckStrictly(half);
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
  cubes.emplace_back(Manifold::Cube(glm::vec3(3.0f), true));
  cubes.emplace_back(Manifold::Cube(glm::vec3(2.0f), true));
  Manifold doubled = Manifold::Compose(cubes);

  Manifold cube = Manifold::Cube(glm::vec3(1.0f), true);
  EXPECT_FALSE((cube ^= doubled).IsEmpty());
}

TEST(Boolean, NonIntersecting) {
  Manifold cube1 = Manifold::Cube();
  float vol1 = cube1.GetProperties().volume;
  Manifold cube2 = cube1.Scale(glm::vec3(2)).Translate({3, 0, 0});
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
  cube2 = cube2.Scale(glm::vec3(scale)).Translate({distance, 0, 0});

  cube += cube2;
  ExpectMeshes(cube, {{8, 12}});

  cube3 = cube3.Scale(glm::vec3(2 * scale)).Translate({distance, 0, 0});
  cube += cube3;
  ExpectMeshes(cube, {{8, 12}, {8, 12}});
}

TEST(Boolean, Precision2) {
  float scale = 1000;
  Manifold cube = Manifold::Cube(glm::vec3(scale));
  Manifold cube2 = cube;
  float distance = scale * (1 - kTolerance / 2);

  cube2 = cube2.Translate(glm::vec3(-distance));
  EXPECT_TRUE((cube ^ cube2).IsEmpty());

  cube2 = cube2.Translate(glm::vec3(scale * kTolerance));
  EXPECT_FALSE((cube ^ cube2).IsEmpty());
}

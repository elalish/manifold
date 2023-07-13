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

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

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
  ExpectMeshes(result2, {{18, 32}});
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

  Manifold cylinder2 = cylinder.Scale({0.5f, 0.5f, 1.0f})
                           .Rotate(0, 0, 15)
                           .Translate({0.25f, 0.25f, 0.0f});
  Manifold out = cylinder - cylinder2;
  ExpectMeshes(out, {{32, 64, 3, 49}});
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

  Manifold cylinder2 = cylinder.Scale({0.5f, 0.5f, 1.0f})
                           .Rotate(0, 0, 15)
                           .Translate({0.25f, 0.25f, 0.0f});
  Manifold out = cylinder - cylinder2;
  ExpectMeshes(out, {{42, 84, 3, 68}});
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

/**
 * These tests verify correct topology and geometry for complex boolean
 * operations between valid shapes with many faces.
 */
TEST(Boolean, Sphere) {
  Manifold sphere = Manifold::Sphere(1.0f, 12);
  MeshGL sphereGL = WithPositionColors(sphere);
  sphere = Manifold(sphereGL);

  Manifold sphere2 = sphere.Translate(glm::vec3(0.5));
  Manifold result = sphere - sphere2;

  ExpectMeshes(result, {{74, 144, 3, 110}});
  EXPECT_EQ(result.NumDegenerateTris(), 0);

  RelatedGL(result, {sphereGL});
}

TEST(Boolean, MeshRelation) {
  Mesh gyroidMesh = Gyroid();
  MeshGL gyroidMeshGL = WithPositionColors(gyroidMesh);
  Manifold gyroid(gyroidMeshGL);

  Manifold gyroid2 = gyroid.Translate(glm::vec3(2.0f));

  EXPECT_FALSE(gyroid.IsEmpty());
  EXPECT_TRUE(gyroid.MatchesTriNormals());
  EXPECT_LE(gyroid.NumDegenerateTris(), 0);
  Manifold result = gyroid + gyroid2;

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorChannels = glm::ivec4(3, 4, 5, -1);
  if (options.exportModels)
    ExportMesh("gyroidUnion.glb", result.GetMeshGL(), opt);
#endif

  EXPECT_TRUE(result.MatchesTriNormals());
  EXPECT_LE(result.NumDegenerateTris(), 1);
  EXPECT_EQ(result.Decompose().size(), 1);
  auto prop = result.GetProperties();
  EXPECT_NEAR(prop.volume, 226, 1);
  EXPECT_NEAR(prop.surfaceArea, 387, 1);

  RelatedGL(result, {gyroidMeshGL});
}

TEST(Boolean, Cylinders) {
  Manifold rod = Manifold::Cylinder(1.0, 0.4, -1.0, 12);
  float arrays1[][12] = {
      {0, 0, 1, 3,    //
       -1, 0, 0, 3,   //
       0, -1, 0, 6},  //
      {0, 0, 1, 2,    //
       -1, 0, 0, 3,   //
       0, -1, 0, 8},  //

      {0, 0, 1, 1,    //
       -1, 0, 0, 2,   //
       0, -1, 0, 7},  //
      {1, 0, 0, 3,    //
       0, 1, 0, 2,    //
       0, 0, 1, 6},   //
      {0, 0, 1, 3,    //
       -1, 0, 0, 3,   //
       0, -1, 0, 7},  //
      {0, 0, 1, 1,    //
       -1, 0, 0, 3,   //
       0, -1, 0, 7},  //
      {1, 0, 0, 3,    //
       0, 0, 1, 4,    //
       0, -1, 0, 6},  //
      {1, 0, 0, 4,    //
       0, 0, 1, 4,    //
       0, -1, 0, 6},  //
  };
  float arrays2[][12] = {
      {1, 0, 0, 3,    //
       0, 0, 1, 2,    //
       0, -1, 0, 6},  //
      {1, 0, 0, 4,    //
       0, 1, 0, 3,    //
       0, 0, 1, 6},   //

      {0, 0, 1, 2,    //
       -1, 0, 0, 2,   //
       0, -1, 0, 7},  //
      {1, 0, 0, 3,    //
       0, 1, 0, 3,    //
       0, 0, 1, 7},   //
      {1, 0, 0, 2,    //
       0, 1, 0, 3,    //
       0, 0, 1, 7},   //
      {1, 0, 0, 1,    //
       0, 1, 0, 3,    //
       0, 0, 1, 7},   //
      {1, 0, 0, 3,    //
       0, 1, 0, 4,    //
       0, 0, 1, 7},   //
      {1, 0, 0, 3,    //
       0, 1, 0, 5,    //
       0, 0, 1, 6},   //
      {0, 0, 1, 3,    //
       -1, 0, 0, 4,   //
       0, -1, 0, 6},  //
  };

  Manifold m1;
  for (auto& array : arrays1) {
    glm::mat4x3 mat;
    for (const int i : {0, 1, 2, 3}) {
      for (const int j : {0, 1, 2}) {
        mat[i][j] = array[j * 4 + i];
      }
    }
    m1 += rod.Transform(mat);
  }

  Manifold m2;
  for (auto& array : arrays2) {
    glm::mat4x3 mat;
    for (const int i : {0, 1, 2, 3}) {
      for (const int j : {0, 1, 2}) {
        mat[i][j] = array[j * 4 + i];
      }
    }
    m2 += rod.Transform(mat);
  }
  m1 += m2;

  EXPECT_TRUE(m1.MatchesTriNormals());
  EXPECT_LE(m1.NumDegenerateTris(), 12);
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

TEST(Boolean, Subtract) {
  Mesh firstMesh;
  firstMesh.vertPos = {{0, 0, 0},           {1540, 0, 0},
                       {1540, 70, 0},       {0, 70, 0},
                       {0, 0, -278.282},    {1540, 70, -278.282},
                       {1540, 0, -278.282}, {0, 70, -278.282}};
  firstMesh.triVerts = {
      {0, 1, 2}, {2, 3, 0}, {4, 5, 6}, {5, 4, 7}, {6, 2, 1}, {6, 5, 2},
      {5, 3, 2}, {5, 7, 3}, {7, 0, 3}, {7, 4, 0}, {4, 1, 0}, {4, 6, 1},
  };

  Mesh secondMesh;
  secondMesh.vertPos = {
      {2.04636e-12, 70, 50000},       {2.04636e-12, -1.27898e-13, 50000},
      {1470, -1.27898e-13, 50000},    {1540, 70, 50000},
      {2.04636e-12, 70, -28.2818},    {1470, -1.27898e-13, 0},
      {2.04636e-12, -1.27898e-13, 0}, {1540, 70, -28.2818}};
  secondMesh.triVerts = {{0, 1, 2}, {2, 3, 0}, {4, 5, 6}, {5, 4, 7},
                         {6, 2, 1}, {6, 5, 2}, {5, 3, 2}, {5, 7, 3},
                         {7, 0, 3}, {7, 4, 0}, {4, 1, 0}, {4, 6, 1}};

  Manifold first(firstMesh);
  Manifold second(secondMesh);

  first -= second;
  first.GetMesh();
}

TEST(Boolean, Close) {
  PolygonParams().processOverlaps = true;

  const float r = 10;
  Manifold a = Manifold::Sphere(r, 256);
  Manifold result = a;
  for (int i = 0; i < 10; i++) {
    // std::cout << i << std::endl;
    result ^= a.Translate({a.Precision() / 10 * i, 0.0, 0.0});
  }
  auto prop = result.GetProperties();
  const float tol = 0.004;
  EXPECT_NEAR(prop.volume, (4.0f / 3.0f) * glm::pi<float>() * r * r * r,
              tol * r * r * r);
  EXPECT_NEAR(prop.surfaceArea, 4 * glm::pi<float>() * r * r, tol * r * r);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("close.glb", result.GetMesh(), {});
#endif

  PolygonParams().processOverlaps = false;
}

TEST(Boolean, UnionDifference) {
  Manifold block = Manifold::Cube({1, 1, 1}, true) - Manifold::Cylinder(1, 0.5);
  Manifold result = block + block.Translate({0, 0, 1});
  float resultsize = result.GetProperties().volume;
  float blocksize = block.GetProperties().volume;
  EXPECT_NEAR(resultsize, blocksize * 2, 0.0001);
}

TEST(Boolean, BooleanVolumes) {
  glm::mat4 m = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f));

  // Define solids which volumes are easy to compute w/ bit arithmetics:
  // m1, m2, m4 are unique, non intersecting "bits" (of volume 1, 2, 4)
  // m3 = m1 + m2
  // m7 = m1 + m2 + m3
  auto m1 = Manifold::Cube({1, 1, 1});
  auto m2 = Manifold::Cube({2, 1, 1}).Transform(
      glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0, 0)));
  auto m4 = Manifold::Cube({4, 1, 1}).Transform(
      glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 0, 0)));
  auto m3 = Manifold::Cube({3, 1, 1});
  auto m7 = Manifold::Cube({7, 1, 1});

  EXPECT_FLOAT_EQ((m1 ^ m2).GetProperties().volume, 0);
  EXPECT_FLOAT_EQ((m1 + m2 + m4).GetProperties().volume, 7);
  EXPECT_FLOAT_EQ((m1 + m2 - m4).GetProperties().volume, 3);
  EXPECT_FLOAT_EQ((m1 + (m2 ^ m4)).GetProperties().volume, 1);
  EXPECT_FLOAT_EQ((m7 ^ m4).GetProperties().volume, 4);
  EXPECT_FLOAT_EQ((m7 ^ m3 ^ m1).GetProperties().volume, 1);
  EXPECT_FLOAT_EQ((m7 ^ (m1 + m2)).GetProperties().volume, 3);
  EXPECT_FLOAT_EQ((m7 - m4).GetProperties().volume, 3);
  EXPECT_FLOAT_EQ((m7 - m4 - m2).GetProperties().volume, 1);
  EXPECT_FLOAT_EQ((m7 - (m7 - m1)).GetProperties().volume, 1);
  EXPECT_FLOAT_EQ((m7 - (m1 + m2)).GetProperties().volume, 4);
}

TEST(Boolean, TreeTransforms) {
  auto a = (Manifold::Cube({1, 1, 1}) + Manifold::Cube({1, 1, 1}))
               .Translate({1, 0, 0});
  auto b = (Manifold::Cube({1, 1, 1}) + Manifold::Cube({1, 1, 1}));

  EXPECT_FLOAT_EQ((a + b).GetProperties().volume, 2);
}

TEST(Boolean, Sweep) {
  PolygonParams().processOverlaps = true;

  // generate the minimum equivalent positive angle
  auto minPosAngle = [](float angle) {
    float div = angle / glm::two_pi<float>();
    float wholeDiv = floor(div);
    return angle - wholeDiv * glm::two_pi<float>();
  };

  // calculate determinant
  auto det = [](glm::vec2 v1, glm::vec2 v2) {
    return v1.x * v2.y - v1.y * v2.x;
  };

  // generate sweep profile
  auto generateProfile = []() {
    float filletRadius = 2.5;
    float filletWidth = 5;
    int numberOfArcPoints = 10;
    glm::vec2 arcCenterPoint =
        glm::vec2(filletWidth - filletRadius, filletRadius);
    std::vector<glm::vec2> arcPoints;

    for (int i = 0; i < numberOfArcPoints; i++) {
      float angle = i * glm::pi<float>() / numberOfArcPoints;
      float y = arcCenterPoint.y - cos(angle) * filletRadius;
      float x = arcCenterPoint.x + sin(angle) * filletRadius;
      arcPoints.push_back(glm::vec2(x, y));
    }

    std::vector<glm::vec2> profile;
    profile.push_back(glm::vec2(0, 0));
    profile.push_back(glm::vec2(filletWidth - filletRadius, 0));
    for (int i = 0; i < numberOfArcPoints; i++) {
      profile.push_back(arcPoints[i]);
    }
    profile.push_back(glm::vec2(0, filletWidth));

    CrossSection profileCrossSection = CrossSection(profile);
    return profileCrossSection;
  };

  CrossSection profile = generateProfile();

  auto partialRevolve = [minPosAngle, profile](float startAngle, float endAngle,
                                               int nSegmentsPerRotation) {
    float posEndAngle = minPosAngle(endAngle);
    float totalAngle = 0;
    if (startAngle < 0 && endAngle < 0 && startAngle < endAngle) {
      totalAngle = endAngle - startAngle;
    } else {
      totalAngle = posEndAngle - startAngle;
    }

    int nSegments =
        ceil(totalAngle / glm::two_pi<float>() * nSegmentsPerRotation + 1);
    if (nSegments < 2) {
      nSegments = 2;
    }

    float angleStep = totalAngle / (nSegments - 1);
    auto warpFunc = [nSegments, angleStep, startAngle](glm::vec3& vertex) {
      float zIndex = nSegments - 1 - vertex.z;
      float angle = zIndex * angleStep + startAngle;

      // transform
      vertex.z = vertex.y;
      vertex.y = vertex.x * sin(angle);
      vertex.x = vertex.x * cos(angle);
    };

    return Manifold::Extrude(profile, nSegments - 1, nSegments - 2)
        .Warp(warpFunc);
  };

  auto cutterPrimitives = [det, partialRevolve, profile](
                              glm::vec2 p1, glm::vec2 p2, glm::vec2 p3) {
    glm::vec2 diff = p2 - p1;
    glm::vec2 vec1 = p1 - p2;
    glm::vec2 vec2 = p3 - p2;
    float determinant = det(vec1, vec2);

    float startAngle = atan2(vec1.x, -vec1.y);
    float endAngle = atan2(-vec2.x, vec2.y);

    Manifold round = partialRevolve(startAngle, endAngle, 20)
                         .Translate(glm::vec3(p2.x, p2.y, 0));

    float distance = sqrt(diff.x * diff.x + diff.y * diff.y);
    float angle = atan2(diff.y, diff.x);
    Manifold extrusionPrimitive =
        Manifold::Extrude(profile, distance)
            .Rotate(90, 0, -90)
            .Translate(glm::vec3(distance, 0, 0))
            .Rotate(0, 0, angle * 180 / glm::pi<float>())
            .Translate(glm::vec3(p1.x, p1.y, 0));

    std::vector<Manifold> result;

    if (determinant < 0) {
      result.push_back(round);
      result.push_back(extrusionPrimitive);
    } else {
      result.push_back(extrusionPrimitive);
    }

    return result;
  };

  auto scalePath = [](std::vector<glm::vec2> path, float scale) {
    std::vector<glm::vec2> newPath;
    for (glm::vec2 point : path) {
      newPath.push_back(scale * point);
    }
    return newPath;
  };

  std::vector<glm::vec2> pathPoints = {
      glm::vec2(-21.707751473606564, 10.04202769267855),
      glm::vec2(-21.840846948218307, 9.535474475521578),
      glm::vec2(-21.940954413815387, 9.048287386171369),
      glm::vec2(-22.005569458385835, 8.587741145234093),
      glm::vec2(-22.032187669917704, 8.16111047331591),
      glm::vec2(-22.022356960178296, 7.755456475810721),
      glm::vec2(-21.9823319178086, 7.356408291345673),
      glm::vec2(-21.91208498286602, 6.964505631629036),
      glm::vec2(-21.811437268778267, 6.579251589515578),
      glm::vec2(-21.68020988897306, 6.200149257860059),
      glm::vec2(-21.51822395687812, 5.82670172951726),
      glm::vec2(-21.254086890521585, 5.336709200579579),
      glm::vec2(-21.01963533308061, 4.974523796623895),
      glm::vec2(-20.658228140926262, 4.497743844638198),
      glm::vec2(-20.350337020134603, 4.144115181723373),
      glm::vec2(-19.9542029967, 3.7276501717684054),
      glm::vec2(-20.6969129296381, 3.110639833377638),
      glm::vec2(-21.026318197401537, 2.793796378245609),
      glm::vec2(-21.454710558515973, 2.3418076758544806),
      glm::vec2(-21.735944543382722, 2.014266362004704),
      glm::vec2(-21.958999535447845, 1.7205197644485681),
      glm::vec2(-22.170169612837164, 1.3912359628761894),
      glm::vec2(-22.376940405634056, 1.0213515348242117),
      glm::vec2(-22.62545385249271, 0.507889651991388),
      glm::vec2(-22.77620002102207, 0.13973666928102288),
      glm::vec2(-22.8689989640578, -0.135962138067232),
      glm::vec2(-22.974385239894364, -0.5322784681448909),
      glm::vec2(-23.05966775687304, -0.9551466941218276),
      glm::vec2(-23.102914137841445, -1.2774406685179822),
      glm::vec2(-23.14134824916783, -1.8152432718003662),
      glm::vec2(-23.152085124298473, -2.241104719188421),
      glm::vec2(-23.121576743285054, -2.976332948223073),
      glm::vec2(-23.020491352156856, -3.6736813934577914),
      glm::vec2(-22.843552165110886, -4.364810769710428),
      glm::vec2(-22.60334013490563, -5.033012850282157),
      glm::vec2(-22.305015243491663, -5.67461444847819),
      glm::vec2(-21.942709324216615, -6.330962778427178),
      glm::vec2(-21.648491707764062, -6.799117771996025),
      glm::vec2(-21.15330508818782, -7.496539096945377),
      glm::vec2(-21.10687739725184, -7.656798276710632),
      glm::vec2(-21.01253055778545, -8.364144493707382),
      glm::vec2(-20.923211927856293, -8.782280691344269),
      glm::vec2(-20.771325204062215, -9.258087073404687),
      glm::vec2(-20.554404009259198, -9.72613360625344),
      glm::vec2(-20.384050989017144, -9.985885743112847),
      glm::vec2(-20.134404839253612, -10.263023004626703),
      glm::vec2(-19.756998832033442, -10.613109670467736),
      glm::vec2(-18.83161393127597, -15.68768837402245),
      glm::vec2(-19.155593463785983, -17.65410871259763),
      glm::vec2(-17.930304365744544, -19.005810988385562),
      glm::vec2(-16.893408103100064, -19.50558228186199),
      glm::vec2(-16.27514960757635, -19.8288501942628),
      glm::vec2(-15.183033464853374, -20.47781203017123),
      glm::vec2(-14.906850387751492, -20.693472553142833),
      glm::vec2(-14.585198957236713, -21.015257964547136),
      glm::vec2(-11.013839210807205, -34.70394287828328),
      glm::vec2(-8.79778020674896, -36.17434400175442),
      glm::vec2(-7.850491148257242, -36.48835987119041),
      glm::vec2(-6.982497182376991, -36.74546968896842),
      glm::vec2(-6.6361688522576, -36.81653354539242),
      glm::vec2(-6.0701080598244035, -36.964332993204),
      glm::vec2(-5.472439187922815, -37.08824838436714),
      glm::vec2(-4.802871164820756, -37.20127157090685),
      glm::vec2(-3.6605994233344745, -37.34427653957914),
      glm::vec2(-1.7314396363710867, -37.46415201430501),
      glm::vec2(-0.7021130485987349, -37.5),
      glm::vec2(0.01918509410483974, -37.49359541901704),
      glm::vec2(1.2107837650065625, -37.45093992812552),
      glm::vec2(3.375529069920302, 32.21823383780513),
      glm::vec2(1.9041980552754056, 32.89839543047101),
      glm::vec2(1.4107184651094313, 33.16556804736585),
      glm::vec2(1.1315552947605065, 33.34344755450097),
      glm::vec2(0.8882931135353977, 33.52377699790175),
      glm::vec2(0.6775397019893341, 33.708817857198056),
      glm::vec2(0.49590284067753837, 33.900831612019715),
      glm::vec2(0.2291596803839543, 34.27380625039597),
      glm::vec2(0.03901816126171688, 34.66402375075138),
      glm::vec2(-0.02952797094655369, 34.8933309389416),
      glm::vec2(-0.0561772851849209, 35.044928843125824),
      glm::vec2(-0.067490756643705, 35.27129875796868),
      glm::vec2(-0.05587453990569748, 35.42204271802184),
      glm::vec2(0.013497378362074697, 35.72471438137191),
      glm::vec2(0.07132375113026912, 35.877348797053145),
      glm::vec2(0.18708820875448923, 36.108917464873215),
      glm::vec2(0.39580614140195136, 36.424415957998825),
      glm::vec2(0.8433687814267005, 36.964365016108914),
      glm::vec2(0.7078417131710703, 37.172455373435916),
      glm::vec2(0.5992848016685662, 37.27482757003058),
      glm::vec2(0.40594743344375905, 37.36664006036318),
      glm::vec2(0.1397973410299913, 37.434752779117005)};

  int numPoints = pathPoints.size();
  pathPoints = scalePath(pathPoints, 0.9);

  std::vector<Manifold> result;

  for (int i = 0; i < numPoints; i++) {
    // std::cerr << i << std::endl;
    std::vector<Manifold> primitives =
        cutterPrimitives(pathPoints[i], pathPoints[(i + 1) % numPoints],
                         pathPoints[(i + 2) % numPoints]);

    for (Manifold primitive : primitives) {
      result.push_back(primitive);
    }
  }

  // all primitives should be valid
  for (Manifold primitive : result) {
    manifold::Properties properties = primitive.GetProperties();
    if (properties.volume < 0) {
      std::cerr << "INVALID PRIMITIVE" << std::endl;
    }
  }

  Manifold shape = Manifold::BatchBoolean(result, OpType::Add);
  auto prop = shape.GetProperties();

  EXPECT_NEAR(prop.volume, 3757, 1);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("unionError.glb", shape.GetMesh(), {});
#endif

  PolygonParams().processOverlaps = false;
}
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

#include "../src/utils.h"
#include "manifold/manifold.h"
#include "test.h"

using namespace manifold;

/**
 * The very simplest Boolean operation test.
 */
TEST(Boolean, Tetra) {
  Manifold tetra = WithPositionColors(Manifold::Tetrahedron());
  MeshGL tetraGL = tetra.GetMeshGL();
  EXPECT_TRUE(!tetra.IsEmpty());

  Manifold tetra2 = tetra.Translate(vec3(0.5));
  Manifold result = tetra2 - tetra;

  ExpectMeshes(result, {{8, 12, 3, 11}});

  RelatedGL(result, {tetraGL});
}

TEST(Boolean, MeshGLRoundTrip) {
  Manifold cube = Manifold::Cube(vec3(2));
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
  const Manifold sphere = Manifold::Sphere(60).CalculateNormals(0);
  const MeshGL sphereGL = sphere.GetMeshGL();

  Manifold result =
      cube.Scale(vec3(100)) -
      (sphere.Rotate(180) -
       sphere.Scale(vec3(0.5)).Rotate(90).Translate({40, 40, 40}));

  RelatedGL(result, {cubeGL, sphereGL}, true, true);

  MeshGL output = result.GetMeshGL(0);

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.faceted = false;
  opt.mat.roughness = 0;
  opt.mat.normalIdx = 0;
  if (options.exportModels) ExportMesh("normals.glb", output, opt);
#endif

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
  Manifold cube = Manifold::Cube(vec3(1)).Scale({1, -1, 1});
  EXPECT_TRUE(cube.MatchesTriNormals());

  Manifold cube2 = Manifold::Cube(vec3(1)).Scale({0.5, -1, 0.5});
  Manifold result = cube - cube2;

  ExpectMeshes(result, {{12, 20}});

  EXPECT_FLOAT_EQ(result.Volume(), 0.75);
  EXPECT_FLOAT_EQ(result.SurfaceArea(), 5.5);
}

TEST(Boolean, Cubes) {
  Manifold result = Manifold::Cube({1.2, 1, 1}, true).Translate({0, -0.5, 0.5});
  result += Manifold::Cube({1, 0.8, 0.5}).Translate({-0.5, 0, 0.5});
  result += Manifold::Cube({1.2, 0.1, 0.5}).Translate({-0.6, -0.1, 0});

  EXPECT_TRUE(result.MatchesTriNormals());
  EXPECT_LE(result.NumDegenerateTris(), 0);
  EXPECT_NEAR(result.Volume(), 1.6, 0.001);
  EXPECT_NEAR(result.SurfaceArea(), 9.2, 0.01);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("cubes.glb", result.GetMeshGL(), {});
#endif
}

TEST(Boolean, NoRetainedVerts) {
  Manifold cube = Manifold::Cube(vec3(1), true);
  Manifold oct = Manifold::Sphere(1, 4);
  EXPECT_NEAR(cube.Volume(), 1, 0.001);
  EXPECT_NEAR(oct.Volume(), 1.333, 0.001);
  EXPECT_NEAR((cube ^ oct).Volume(), 0.833, 0.001);
}

TEST(Boolean, PropertiesNoIntersection) {
  MeshGL cubeUV = CubeUV();
  Manifold m0(cubeUV);
  Manifold m1 = m0.Translate(vec3(1.5));
  Manifold result = m0 + m1;
  EXPECT_EQ(result.NumProp(), 2);
  RelatedGL(result, {cubeUV});
}

TEST(Boolean, MixedProperties) {
  MeshGL cubeUV = CubeUV();
  Manifold m0(cubeUV);
  Manifold m1 = Manifold::Cube();
  Manifold result = m0 + m1.Translate(vec3(0.5));
  EXPECT_EQ(result.NumProp(), 2);
  RelatedGL(result, {cubeUV, m1.GetMeshGL()});
}

TEST(Boolean, MixedNumProp) {
  MeshGL cubeUV = CubeUV();
  Manifold m0(cubeUV);
  Manifold m1 = Manifold::Cube();
  Manifold result =
      m0 + m1.SetProperties(1, [](double* prop, vec3 p, const double* n) {
               prop[0] = 1;
             }).Translate(vec3(0.5));
  EXPECT_EQ(result.NumProp(), 2);
  RelatedGL(result, {cubeUV, m1.GetMeshGL()});
}

TEST(Boolean, UnionDifference) {
  Manifold block = Manifold::Cube({1, 1, 1}, true) - Manifold::Cylinder(1, 0.5);
  Manifold result = block + block.Translate({0, 0, 1});
  double resultsize = result.Volume();
  double blocksize = block.Volume();
  EXPECT_NEAR(resultsize, blocksize * 2, 0.0001);
}

TEST(Boolean, TreeTransforms) {
  auto a = (Manifold::Cube({1, 1, 1}) + Manifold::Cube({1, 1, 1}))
               .Translate({1, 0, 0});
  auto b = (Manifold::Cube({1, 1, 1}) + Manifold::Cube({1, 1, 1}));

  EXPECT_FLOAT_EQ((a + b).Volume(), 2);
}

TEST(Boolean, CreatePropertiesSlow) {
  Manifold a = Manifold::Sphere(10, 1024).SetProperties(
      3, [](double* newprop, vec3 pos, const double* old) {
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

  EXPECT_FLOAT_EQ(empty.Volume(), 0.0);
  EXPECT_FLOAT_EQ(empty.SurfaceArea(), 0.0);
}

TEST(Boolean, Perturb) {
  MeshGL tmp;
  tmp.vertProperties = {
      0.0, 0.0, 0.0,  //
      0.0, 1.0, 0.0,  //
      1.0, 0.0, 0.0,  //
      0.0, 0.0, 1.0   //
  };
  tmp.triVerts = {
      2, 0, 1,  //
      0, 3, 1,  //
      2, 3, 0,  //
      3, 2, 1   //
  };
  Manifold corner(tmp);
  Manifold empty = corner - corner;
  EXPECT_TRUE(empty.IsEmpty());

  EXPECT_FLOAT_EQ(empty.Volume(), 0.0);
  EXPECT_FLOAT_EQ(empty.SurfaceArea(), 0.0);
}

TEST(Boolean, Coplanar) {
  Manifold cylinder = WithPositionColors(Manifold::Cylinder(1.0, 1.0));
  MeshGL cylinderGL = cylinder.GetMeshGL();

  Manifold cylinder2 = cylinder.Scale({0.8, 0.8, 1.0}).Rotate(0, 0, 185);
  Manifold out = cylinder - cylinder2;
  ExpectMeshes(out, {{32, 64, 3, 48}});
  EXPECT_EQ(out.NumDegenerateTris(), 0);
  EXPECT_EQ(out.Genus(), 1);

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorIdx = 0;
  if (options.exportModels) ExportMesh("coplanar.glb", out.GetMeshGL(), opt);
#endif

  RelatedGL(out, {cylinderGL});
}

TEST(Boolean, MultiCoplanar) {
  Manifold cube = Manifold::Cube();
  Manifold first = cube - cube.Translate({0.3, 0.3, 0.0});
  cube = cube.Translate({-0.3, -0.3, 0.0});
  Manifold out = first - cube;
  CheckStrictly(out);
  EXPECT_EQ(out.Genus(), -1);
  EXPECT_NEAR(out.Volume(), 0.18, 1e-5);
  EXPECT_NEAR(out.SurfaceArea(), 2.76, 1e-5);
}

TEST(Boolean, FaceUnion) {
  Manifold cubes = Manifold::Cube();
  cubes += cubes.Translate({1, 0, 0});
  EXPECT_EQ(cubes.Genus(), 0);
  ExpectMeshes(cubes, {{12, 20}});
  EXPECT_NEAR(cubes.Volume(), 2, 1e-5);
  EXPECT_NEAR(cubes.SurfaceArea(), 10, 1e-5);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("faceUnion.glb", cubes.GetMeshGL(), {});
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
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  Manifold oct = Manifold::Sphere(1, 4).Translate(vec3(0.0, 0.0, 1.0));
  std::pair<Manifold, Manifold> splits = cube.Split(oct);
  CheckStrictly(splits.first);
  CheckStrictly(splits.second);
  EXPECT_FLOAT_EQ(splits.first.Volume() + splits.second.Volume(),
                  cube.Volume());
}

TEST(Boolean, SplitByPlane) {
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  cube = cube.Translate({0.0, 1.0, 0.0});
  cube = cube.Rotate(90.0, 0.0, 0.0);
  std::pair<Manifold, Manifold> splits =
      cube.SplitByPlane({0.0, 0.0, 1.0}, 1.0);
  CheckStrictly(splits.first);
  CheckStrictly(splits.second);
  EXPECT_NEAR(splits.first.Volume(), splits.second.Volume(), 1e-5);

  Manifold first = cube.TrimByPlane({0.0, 0.0, 1.0}, 1.0);
  // Verify trim returns the same result as the first split by checking that
  // their bounding boxes contain each other, thus they are equal.
  EXPECT_TRUE(splits.first.BoundingBox().Contains(first.BoundingBox()));
  EXPECT_TRUE(first.BoundingBox().Contains(splits.first.BoundingBox()));
}

TEST(Boolean, SplitByPlane60) {
  Manifold cube = Manifold::Cube(vec3(2.0), true);
  cube = cube.Translate({0.0, 1.0, 0.0});
  cube = cube.Rotate(0.0, 0.0, -60.0);
  cube = cube.Translate({2.0, 0.0, 0.0});
  double phi = 30.0;
  std::pair<Manifold, Manifold> splits =
      cube.SplitByPlane({sind(phi), -cosd(phi), 0.0}, 1.0);
  CheckStrictly(splits.first);
  CheckStrictly(splits.second);
  EXPECT_NEAR(splits.first.Volume(), splits.second.Volume(), 1e-5);
}

/**
 * This tests that non-intersecting geometry is properly retained.
 */
TEST(Boolean, Vug) {
  Manifold cube = Manifold::Cube(vec3(4.0), true);
  Manifold vug = cube - Manifold::Cube();

  EXPECT_EQ(vug.Genus(), -1);

  Manifold half = vug.SplitByPlane({0.0, 0.0, 1.0}, -1.0).first;
  CheckStrictly(half);
  EXPECT_EQ(half.Genus(), -1);

  EXPECT_FLOAT_EQ(half.Volume(), 4.0 * 4.0 * 3.0 - 1.0);
  EXPECT_FLOAT_EQ(half.SurfaceArea(), 16.0 * 2 + 12.0 * 4 + 6.0);
}

TEST(Boolean, Empty) {
  Manifold cube = Manifold::Cube();
  double cubeVol = cube.Volume();
  Manifold empty;

  EXPECT_EQ((cube + empty).Volume(), cubeVol);
  EXPECT_EQ((cube - empty).Volume(), cubeVol);
  EXPECT_TRUE((empty - cube).IsEmpty());
  EXPECT_TRUE((cube ^ empty).IsEmpty());
}

TEST(Boolean, Winding) {
  std::vector<Manifold> cubes;
  cubes.emplace_back(Manifold::Cube(vec3(3.0), true));
  cubes.emplace_back(Manifold::Cube(vec3(2.0), true));
  Manifold doubled = Manifold::Compose(cubes);

  Manifold cube = Manifold::Cube(vec3(1.0), true);
  EXPECT_FALSE((cube ^= doubled).IsEmpty());
}

TEST(Boolean, NonIntersecting) {
  Manifold cube1 = Manifold::Cube();
  double vol1 = cube1.Volume();
  Manifold cube2 = cube1.Scale(vec3(2)).Translate({3, 0, 0});
  double vol2 = cube2.Volume();

  EXPECT_EQ((cube1 + cube2).Volume(), vol1 + vol2);
  EXPECT_EQ((cube1 - cube2).Volume(), vol1);
  EXPECT_TRUE((cube1 ^ cube2).IsEmpty());
}

TEST(Boolean, Precision) {
  Manifold cube = Manifold::Cube();
  Manifold cube2 = cube;
  Manifold cube3 = cube;
  double distance = 100;
  double scale = distance * kPrecision;
  cube2 = cube2.Scale(vec3(scale)).Translate({distance, 0, 0});

  cube += cube2;
  ExpectMeshes(cube, {{8, 12}});

  cube3 = cube3.Scale(vec3(2 * scale)).Translate({distance, 0, 0});
  cube += cube3;
  ExpectMeshes(cube, {{8, 12}, {8, 12}});
}

TEST(Boolean, Precision2) {
  double scale = 1000;
  Manifold cube = Manifold::Cube(vec3(scale));
  Manifold cube2 = cube;
  double distance = scale * (1 - kPrecision / 2);

  cube2 = cube2.Translate(vec3(-distance));
  EXPECT_TRUE((cube ^ cube2).IsEmpty());

  cube2 = cube2.Translate(vec3(scale * kPrecision));
  EXPECT_FALSE((cube ^ cube2).IsEmpty());
}

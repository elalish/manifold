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
  ExpectMeshes(result2, {{18, 32}});
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

  if (options.exportModels) WriteTestOBJ("normals.obj", result);

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

  if (options.exportModels) WriteTestOBJ("cubes.obj", result);
}

TEST(Boolean, Simplify) {
  const int n = 10;
  MeshGL cubeGL = Manifold::Cube().Refine(n).GetMeshGL();
  size_t tri = 0;
  for (auto& id : cubeGL.faceID) {
    id = tri++;
  }
  Manifold cube(cubeGL);

  const int nExpected = 20 * n * n;
  Manifold result = cube + cube.Translate({1, 0, 0});
  EXPECT_EQ(result.NumTri(), nExpected);
  result = result.Simplify();
  EXPECT_EQ(result.NumTri(), nExpected);

  MeshGL resultGL = result.GetMeshGL();
  resultGL.faceID.clear();
  Manifold result2(resultGL);
  EXPECT_EQ(result2.NumTri(), nExpected);
  result2 = result2.Simplify();
  EXPECT_EQ(result2.NumTri(), 20);
}

TEST(Boolean, SimplifyCracks) {
  Manifold cylinder =
      Manifold::Cylinder(2, 50, 50, 180)
          .Rotate(
              -89.999999999999)  // Rotating by -90 makes the result too perfect
          .Translate(vec3(50, 0, 50));
  Manifold cube = Manifold::Cube(vec3(100, 2, 50));
  Manifold refined = (cylinder + cube).RefineToLength(1);
  Manifold deformed =
      refined.Warp([](vec3& p) { p.y += p.x - (p.x * p.x) / 100.0; });
  Manifold simplified = deformed.Simplify(0.005);

  // If Simplify adds cracks, volume decreases and surface area increases
  EXPECT_EQ(deformed.Genus(), 0);
  EXPECT_EQ(simplified.Genus(), 0);
  EXPECT_NEAR(simplified.Volume(), deformed.Volume(), 10);
  EXPECT_NEAR(simplified.SurfaceArea(), deformed.SurfaceArea(), 1);

  if (options.exportModels) WriteTestOBJ("cracks.obj", simplified);
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

TEST(Boolean, PropsMismatch) {
  Manifold ma = Manifold::Cylinder(1, 1);
  Manifold mb =
      Manifold::Cube()
          .Translate({50, 0, 0})
          .SetProperties(1, [](double* newProp, vec3 pos, const double* _) {
            newProp[0] = pos.x;
          });

  Manifold result = ma + mb;
  EXPECT_EQ(result.Status(), Manifold::Error::NoError);
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

TEST(Boolean, Perturb1) {
  const Manifold big = Manifold::Extrude(
      {{{0, 2}, {2, 0}, {4, 2}, {2, 4}}, {{1, 2}, {2, 3}, {3, 2}, {2, 1}}},
      1.0);
  const Manifold little =
      Manifold::Extrude({{{2, 1}, {3, 2}, {2, 3}, {1, 2}}}, 1.0)
          .Translate({0, 0, 1});
  const Manifold punchHole =
      Manifold::Extrude({{{1, 2}, {2, 2}, {2, 3}}}, 1.0).Translate({0, 0, 1});
  const Manifold result = (big + little) - punchHole;

  EXPECT_EQ(result.NumDegenerateTris(), 0);
  EXPECT_EQ(result.NumVert(), 24);
  EXPECT_FLOAT_EQ(result.Volume(), 7.5);
  EXPECT_NEAR(result.SurfaceArea(), 38.2, 0.1);

  if (options.exportModels) WriteTestOBJ("perturb1.obj", result);
}

TEST(Boolean, Perturb2) {
  Manifold cube = Manifold::Cube(vec3(2), true);
  MeshGL cubeGL = cube.GetMeshGL();

  // Rotate so that nothing is axis-aligned
  Manifold result = cube.Rotate(5, 10, 15);

  for (size_t tri = 0; tri < cubeGL.NumTri(); ++tri) {
    MeshGL prism;
    prism.numProp = 3;
    prism.triVerts = {4, 2, 0, 1, 3, 5};
    for (const int v0 : {0, 1, 2}) {
      const int v1 = (v0 + 1) % 3;
      const int vIn0 = cubeGL.triVerts[3 * tri + v0];
      const int vIn1 = cubeGL.triVerts[3 * tri + v1];
      if (vIn1 > vIn0) {
        prism.triVerts.push_back(2 * v0);
        prism.triVerts.push_back(2 * v1);
        prism.triVerts.push_back(2 * v1 + 1);
        prism.triVerts.push_back(2 * v0);
        prism.triVerts.push_back(2 * v1 + 1);
        prism.triVerts.push_back(2 * v0 + 1);
      } else {
        prism.triVerts.push_back(2 * v0);
        prism.triVerts.push_back(2 * v1);
        prism.triVerts.push_back(2 * v0 + 1);
        prism.triVerts.push_back(2 * v1);
        prism.triVerts.push_back(2 * v1 + 1);
        prism.triVerts.push_back(2 * v0 + 1);
      }
      for (const int j : {0, 1, 2}) {
        prism.vertProperties.push_back(cubeGL.vertProperties[3 * vIn0 + j]);
      }
      for (const int j : {0, 1, 2}) {
        prism.vertProperties.push_back(2 * cubeGL.vertProperties[3 * vIn0 + j]);
      }
    }
    // All verts should be floating-point identical to one of 16 positions: the
    // 8 starting result cube verts, or exactly double these coordinates.
    result += Manifold(prism).Rotate(5, 10, 15);
  }
  // The result should be a double-sized cube, 4 units to a side.
  // If symbolic perturbation fails, the number of verts and the surface area
  // will increase, indicating cracks and internal geometry.
  EXPECT_EQ(result.NumDegenerateTris(), 0);
  EXPECT_EQ(result.NumVert(), 8);
  EXPECT_FLOAT_EQ(result.Volume(), 64.0);
  EXPECT_FLOAT_EQ(result.SurfaceArea(), 96.0);
}

TEST(Boolean, Perturb3) {
  // Create a nasty gear pattern with many rotated cubes that creates
  // antiparallel slivers (triangles with normals ~180 degrees apart)
  // https://github.com/BrunoLevy/thingiCSG/blob/main/DATABASE/Basic/nasty_gear_1.scad

  const int N = 16;  // Number of rotations for the gear pattern
  const double alpha = 90.0 / N;

  // Create outer gear - many rotated cubes unioned together
  std::vector<Manifold> outerCubes;
  const Manifold cube = Manifold::Cube({1, 1, 1}, true);
  for (int i = 0; i < N; i++) {
    outerCubes.push_back(cube.Rotate(0, 0, alpha * i));
  }
  Manifold gear = Manifold::BatchBoolean(outerCubes, OpType::Add);
  Manifold outerGear = gear.Scale({2, 2, 1});

  // Subtract inner from outer to create the nasty gear with slivers
  Manifold nastyGear = outerGear - gear;

  // const float topArea = CrossSection(gear.Project()).Area();
  // const float sideArea = gear.SurfaceArea() - 2 * topArea;
  const float expectedArea = 26.972;  // 3 * sideArea + 6 * topArea;
  const float expectedVolume = outerGear.Volume() - gear.Volume();

  // The gear should be valid and manifold
  EXPECT_EQ(nastyGear.Status(), Manifold::Error::NoError);
  EXPECT_FALSE(nastyGear.IsEmpty());
  EXPECT_EQ(nastyGear.Genus(), 1);
  EXPECT_NEAR(nastyGear.Volume(), expectedVolume, 1e-5);
  EXPECT_NEAR(nastyGear.SurfaceArea(), expectedArea, 1e-4);

  if (options.exportModels) WriteTestOBJ("nastyGear.obj", nastyGear);
}

TEST(Boolean, Coplanar) {
  Manifold cylinder = WithPositionColors(Manifold::Cylinder(1.0, 1.0));
  MeshGL cylinderGL = cylinder.GetMeshGL();

  Manifold cylinder2 = cylinder.Scale({0.8, 0.8, 1.0}).Rotate(0, 0, 185);
  Manifold out = cylinder - cylinder2;
  ExpectMeshes(out, {{32, 64, 3, 48}});
  EXPECT_EQ(out.NumDegenerateTris(), 0);
  EXPECT_EQ(out.Genus(), 1);

  if (options.exportModels) WriteTestOBJ("coplanar.obj", out);

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

TEST(Boolean, AlmostCoplanar) {
  Manifold tet = Manifold::Tetrahedron();
  Manifold result =
      tet + tet.Rotate(0.001, -0.08472872823860228, 0.055910459615905288) + tet;
  ExpectMeshes(result, {{20, 36}});
}

TEST(Boolean, FaceUnion) {
  Manifold cubes = Manifold::Cube();
  cubes += cubes.Translate({1, 0, 0});
  EXPECT_EQ(cubes.Genus(), 0);
  ExpectMeshes(cubes, {{12, 20}});
  EXPECT_NEAR(cubes.Volume(), 2, 1e-5);
  EXPECT_NEAR(cubes.SurfaceArea(), 10, 1e-5);

  if (options.exportModels) WriteTestOBJ("faceUnion.obj", cubes);
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

TEST(Boolean, ConvexConvexMinkowski) {
  double r = 0.1;
  double w = 2.0;
  Manifold sphere = Manifold::Sphere(r, 20);
  Manifold cube = Manifold::Cube({w, w, w});
  Manifold sum = cube.MinkowskiSum(sphere);
  // Analytical volume of rounded cuboid: cube + 6 slabs + 12 quarter-cylinders
  // + 8 sphere octants = w³ + 6w²r + 3πwr² + (4/3)πr³
  double analyticalVolume = w * w * w + 6 * w * w * r + 3 * kPi * w * r * r +
                            (4.0 / 3) * kPi * r * r * r;
  // Analytical surface area: 6 faces + 12 quarter-cylinders + 8 octants
  // = 6w² + 6πwr + 4πr²
  double analyticalArea = 6 * w * w + 6 * kPi * w * r + 4 * kPi * r * r;
  // Discrete sphere approximation differs from analytical by ~1%
  EXPECT_NEAR(sum.Volume(), analyticalVolume, 0.15);
  EXPECT_NEAR(sum.SurfaceArea(), analyticalArea, 0.5);
  EXPECT_EQ(sum.Genus(), 0);

  if (options.exportModels) WriteTestOBJ("minkowski-convex-convex.obj", sum);
}

TEST(Boolean, ConvexConvexMinkowskiDecompose) {
  double r = 0.1;
  double w = 2.0;
  Manifold sphere = Manifold::Sphere(r, 20);
  Manifold cube = Manifold::Cube({w, w, w});
  Manifold sum = cube.MinkowskiSum(sphere, true);
  double analyticalVolume = w * w * w + 6 * w * w * r + 3 * kPi * w * r * r +
                            (4.0 / 3) * kPi * r * r * r;
  double analyticalArea = 6 * w * w + 6 * kPi * w * r + 4 * kPi * r * r;
  EXPECT_NEAR(sum.Volume(), analyticalVolume, 0.15);
  EXPECT_NEAR(sum.SurfaceArea(), analyticalArea, 0.5);
  EXPECT_EQ(sum.Genus(), 0);
}

TEST(Boolean, ConvexConvexMinkowskiDifference) {
  ManifoldParamGuard guard;
  ManifoldParams().processOverlaps = true;

  double r = 0.1;
  double w = 2.0;
  Manifold sphere = Manifold::Sphere(r, 20);
  Manifold cube = Manifold::Cube({w, w, w});
  Manifold difference = cube.MinkowskiDifference(sphere);
  // Analytical volume of eroded cube: (w-2r)³
  double analyticalVolume = (w - 2 * r) * (w - 2 * r) * (w - 2 * r);
  EXPECT_NEAR(difference.Volume(), analyticalVolume, 0.1);
  // Analytical surface area: 6*(w-2r)²
  double analyticalArea = 6 * (w - 2 * r) * (w - 2 * r);
  EXPECT_NEAR(difference.SurfaceArea(), analyticalArea, 0.1);
  EXPECT_EQ(difference.Genus(), 0);

  if (options.exportModels)
    WriteTestOBJ("minkowski-convex-convex-difference.obj", difference);
}

TEST(Boolean, NonConvexConvexMinkowskiSum) {
  ManifoldParamGuard guard;
  ManifoldParams().processOverlaps = true;

  Manifold sphere = Manifold::Sphere(1.2, 20);
  Manifold cube = Manifold::Cube({2.0, 2.0, 2.0}, true);
  Manifold nonConvex = cube - sphere;
  Manifold sum = nonConvex.MinkowskiSum(Manifold::Sphere(0.1, 20));
  EXPECT_NEAR(sum.Volume(), 4.841, 1e-3);
  EXPECT_NEAR(sum.SurfaceArea(), 34.06, 1e-2);
  EXPECT_EQ(sum.Genus(), 5);

  if (options.exportModels)
    WriteTestOBJ("minkowski-nonconvex-convex-sum.obj", sum);
}

TEST(Boolean, NonConvexConvexMinkowskiSumDecompose) {
  ManifoldParamGuard guard;
  ManifoldParams().processOverlaps = true;

  Manifold sphere = Manifold::Sphere(1.2, 20);
  Manifold cube = Manifold::Cube({2.0, 2.0, 2.0}, true);
  Manifold nonConvex = cube - sphere;
  Manifold sum = nonConvex.MinkowskiSum(Manifold::Sphere(0.1, 20), true);
  // Curved surfaces produce non-convex DT-clipped pieces, so the
  // vertex-addition Minkowski is approximate for curved geometry.
  // Volume/area are close but genus may differ.
  EXPECT_NEAR(sum.Volume(), 4.841, 0.1);
  EXPECT_NEAR(sum.SurfaceArea(), 34.06, 0.5);
  EXPECT_GE(sum.Genus(), 0);
}

TEST(Boolean, NonConvexConvexMinkowskiDifference) {
  ManifoldParamGuard guard;
  ManifoldParams().processOverlaps = true;

  Manifold sphere = Manifold::Sphere(1.2, 20);
  Manifold cube = Manifold::Cube({2.0, 2.0, 2.0}, true);
  Manifold nonConvex = cube - sphere;
  Manifold difference =
      nonConvex.MinkowskiDifference(Manifold::Sphere(0.05, 20));
  EXPECT_NEAR(difference.Volume(), 0.778, 1e-3);
  EXPECT_NEAR(difference.SurfaceArea(), 16.70, 1e-2);
  EXPECT_EQ(difference.Genus(), 5);

  if (options.exportModels)
    WriteTestOBJ("minkowski-nonconvex-convex-difference.obj", difference);
}

TEST(Boolean, NonConvexNonConvexMinkowskiSum) {
  ManifoldParamGuard guard;
  ManifoldParams().processOverlaps = true;

  Manifold tet = Manifold::Tetrahedron();
  Manifold nonConvex = tet - tet.Rotate(0, 0, 90).Translate(vec3(1));

  Manifold sum = nonConvex.MinkowskiSum(nonConvex.Scale(vec3(0.5)));
  EXPECT_NEAR(sum.Volume(), 8.65625, 1e-5);
  EXPECT_NEAR(sum.SurfaceArea(), 31.17691, 1e-5);
  EXPECT_EQ(sum.Genus(), 0);

  if (options.exportModels)
    WriteTestOBJ("minkowski-nonconvex-nonconvex-sum.obj", sum);
}

TEST(Boolean, NonConvexNonConvexMinkowskiSumDecompose) {
  ManifoldParamGuard guard;
  ManifoldParams().processOverlaps = true;

  Manifold tet = Manifold::Tetrahedron();
  Manifold nonConvex = tet - tet.Rotate(0, 0, 90).Translate(vec3(1));

  Manifold sum = nonConvex.MinkowskiSum(nonConvex.Scale(vec3(0.5)), true);
  // Without hull-snapping, decomposition preserves exact topology
  EXPECT_NEAR(sum.Volume(), 8.65625, 1e-5);
  EXPECT_NEAR(sum.SurfaceArea(), 31.17691, 1e-5);
  EXPECT_EQ(sum.Genus(), 0);
}

TEST(Boolean, NonConvexNonConvexMinkowskiDifference) {
  ManifoldParamGuard guard;
  ManifoldParams().processOverlaps = true;

  Manifold tet = Manifold::Tetrahedron();
  Manifold nonConvex = tet - tet.Rotate(0, 0, 90).Translate(vec3(1));

  Manifold difference =
      nonConvex.MinkowskiDifference(nonConvex.Scale(vec3(0.1)));
  EXPECT_NEAR(difference.Volume(), 0.815542, 1e-5);
  EXPECT_NEAR(difference.SurfaceArea(), 6.95045, 1e-5);
  EXPECT_EQ(difference.Genus(), 0);

  if (options.exportModels)
    WriteTestOBJ("minkowski-nonconvex-nonconvex-diff.obj", difference);
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
  Manifold doubled = Manifold::BatchBoolean(cubes, OpType::Add);

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

TEST(Boolean, SimpleCubeRegression) {
  ManifoldParamGuard guard;
  ManifoldParams().selfIntersectionChecks = true;
  Manifold result =
      Manifold::Cube().Rotate(-0.10000000000000001, 0.10000000000000001, -1.) +
      Manifold::Cube() -
      Manifold::Cube().Rotate(-0.10000000000000001, -0.10000000000066571, -1.);
  EXPECT_EQ(result.Status(), Manifold::Error::NoError);
}

TEST(Boolean, BatchBoolean) {
  Manifold cube = Manifold::Cube({100, 100, 1});
  Manifold cylinder1 = Manifold::Cylinder(1, 30).Translate({-10, 30, 0});
  Manifold cylinder2 = Manifold::Cylinder(1, 20).Translate({110, 20, 0});
  Manifold cylinder3 = Manifold::Cylinder(1, 40).Translate({50, 110, 0});

  Manifold intersect = Manifold::BatchBoolean(
      {cube, cylinder1, cylinder2, cylinder3}, OpType::Intersect);

  EXPECT_TRUE(intersect.IsEmpty());

  Manifold add = Manifold::BatchBoolean({cube, cylinder1, cylinder2, cylinder3},
                                        OpType::Add);

  ExpectMeshes(add, {{152, 300}});
  EXPECT_FLOAT_EQ(add.Volume(), 16290.478);
  EXPECT_FLOAT_EQ(add.SurfaceArea(), 33156.594);

  Manifold subtract = Manifold::BatchBoolean(
      {cube, cylinder1, cylinder2, cylinder3}, OpType::Subtract);

  ExpectMeshes(subtract, {{102, 200}});
  EXPECT_FLOAT_EQ(subtract.Volume(), 7226.043);
  EXPECT_FLOAT_EQ(subtract.SurfaceArea(), 14904.597);
}

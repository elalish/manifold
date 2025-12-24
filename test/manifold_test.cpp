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

#include "manifold/manifold.h"

#include <algorithm>

#ifdef MANIFOLD_CROSS_SECTION
#include "manifold/cross_section.h"
#endif
#include "test.h"

namespace {

using namespace manifold;

template <typename T>
int NumUnique(const std::vector<T>& in) {
  std::set<int> unique;
  for (const T& v : in) {
    unique.emplace(v);
  }
  return unique.size();
}

}  // namespace

/**
 * This tests that turning a mesh into a manifold and returning it to a mesh
 * produces a consistent result.
 */
TEST(Manifold, GetMeshGL) {
  Manifold manifold = Manifold::Sphere(0.01);
  auto mesh_out = manifold.GetMeshGL();
  Manifold manifold2(mesh_out);
  auto mesh_out2 = manifold2.GetMeshGL();
  Identical(mesh_out, mesh_out2);
}

TEST(Manifold, MeshDeterminism) {
  Manifold cube1 = Manifold::Cube(vec3(2.0, 2.0, 2.0), true);
  Manifold cube2 = Manifold::Cube(vec3(2.0, 2.0, 2.0), true)
                       .Translate(vec3(-1.1091, 0.88509, 1.3099));

  Manifold result = cube1 - cube2;
  MeshGL out = result.GetMeshGL();

  uint32_t triVerts[]{0,  2,  7,  0,  10, 1,  0,  6,  10, 0, 1,  2,  1, 3,  2,
                      1,  5,  3,  1,  11, 5,  0,  7,  6,  6, 7,  8,  6, 8,  13,
                      10, 12, 11, 1,  10, 11, 11, 13, 5,  6, 12, 10, 6, 13, 12,
                      13, 9,  5,  13, 8,  9,  11, 12, 13, 4, 2,  3,  4, 3,  5,
                      4,  7,  2,  4,  5,  8,  4,  8,  7,  9, 8,  5};

  float vertProperties[]{-1,      -1,       -1,     -1,      -1,       1,
                         -1,      -0.11491, 0.3099, -1,      -0.11491, 1,
                         -0.1091, -0.11491, 0.3099, -0.1091, -0.11491, 1,
                         -1,      1,        -1,     -1,      1,        0.3099,
                         -0.1091, 1,        0.3099, -0.1091, 1,        1,
                         1,       -1,       -1,     1,       -1,       1,
                         1,       1,        -1,     1,       1,        1};

  bool flag = true;
  for (size_t i = 0; i != out.triVerts.size(); i++) {
    if (out.triVerts[i] != triVerts[i]) {
      flag = false;
      break;
    }
  }

  for (size_t i = 0; flag && i != out.vertProperties.size(); i++) {
    if (out.vertProperties[i] != vertProperties[i]) {
      flag = false;
      break;
    }
  }

  EXPECT_TRUE(flag);
}

TEST(Manifold, Empty) {
  MeshGL emptyMesh;
  Manifold empty(emptyMesh);

  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_EQ(empty.Status(), Manifold::Error::NoError);
}

TEST(Manifold, ValidInput) {
  MeshGL tetGL = TetGL();
  Manifold tet(tetGL);
  EXPECT_FALSE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NoError);
}

TEST(Manifold, ValidInputOneRunIndex) {
  MeshGL emptyMesh;
  emptyMesh.runIndex = {0};
  Manifold empty(emptyMesh);
  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_EQ(empty.Status(), Manifold::Error::NoError);
}

TEST(Manifold, InvalidInput1) {
  MeshGL in = TetGL();
  in.vertProperties[2 * 3 + 1] = NAN;
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, InvalidInput2) {
  MeshGL in = TetGL();
  std::swap(in.triVerts[2 * 3 + 1], in.triVerts[2 * 3 + 2]);
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NotManifold);
}

TEST(Manifold, InvalidInput3) {
  MeshGL in = TetGL();
  for (uint32_t& triVert : in.triVerts) {
    if (triVert == 2) triVert = -2;
  }
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VertexOutOfBounds);
}

TEST(Manifold, InvalidInput4) {
  MeshGL in = TetGL();
  for (uint32_t& triVert : in.triVerts) {
    if (triVert == 2) triVert = 4;
  }
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NotManifold);
}

TEST(Manifold, InvalidInput5) {
  MeshGL tetGL = TetGL();
  tetGL.mergeFromVert[tetGL.mergeFromVert.size() - 1] = 7;
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::MergeIndexOutOfBounds);
}

TEST(Manifold, InvalidInput6) {
  MeshGL tetGL = TetGL();
  tetGL.triVerts[tetGL.triVerts.size() - 1] = 7;
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VertexOutOfBounds);
}

TEST(Manifold, InvalidInput7) {
  MeshGL cube = CubeUV();
  cube.runIndex = {0, 1, static_cast<uint32_t>(cube.triVerts.size())};
  Manifold tet(cube);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::RunIndexWrongLength);
}

TEST(Manifold, OppositeFace) {
  MeshGL gl;
  gl.vertProperties = {
      0, 0, 0,  //
      1, 0, 0,  //
      0, 1, 0,  //
      1, 1, 0,  //
      0, 0, 1,  //
      1, 0, 1,  //
      0, 1, 1,  //
      1, 1, 1,  //
      //
      2, 0, 0,  //
      2, 1, 0,  //
      2, 0, 1,  //
      2, 1, 1,  //
  };
  gl.triVerts = {
      0, 1,  4,   //
      0, 2,  3,   //
      0, 3,  1,   //
      0, 4,  2,   //
      1, 3,  5,   //
      1, 3,  9,   //
      1, 5,  3,   //
      1, 5,  4,   //
      1, 8,  5,   //
      1, 9,  8,   //
      2, 4,  6,   //
      2, 6,  7,   //
      2, 7,  3,   //
      3, 5,  7,   //
      3, 7,  5,   //
      3, 7,  11,  //
      3, 11, 9,   //
      4, 5,  6,   //
      5, 7,  6,   //
      5, 8,  10,  //
      5, 10, 7,   //
      7, 10, 11,  //
      8, 9,  10,  //
      9, 11, 10,  //
  };
  Manifold man(gl);
  EXPECT_EQ(man.Status(), Manifold::Error::NoError);
  EXPECT_EQ(man.NumVert(), 12);
  EXPECT_FLOAT_EQ(man.Volume(), 2);
}

/**
 * ExpectMeshes performs a decomposition, so this test ensures that compose and
 * decompose are inverse operations.
 */
TEST(Manifold, Decompose) {
  std::vector<Manifold> manifoldList;
  manifoldList.emplace_back(Manifold::Tetrahedron());
  manifoldList.emplace_back(Manifold::Cube());
  manifoldList.emplace_back(Manifold::Sphere(1, 4));
  Manifold manifolds = Manifold::Compose(manifoldList);

  ExpectMeshes(manifolds, {{8, 12}, {6, 8}, {4, 4}});

  std::vector<MeshGL> input;

  for (const Manifold& manifold : manifoldList) {
    EXPECT_GE(manifold.OriginalID(), 0);
    input.emplace_back(manifold.GetMeshGL());
  }

  RelatedGL(manifolds, input);
}

TEST(Manifold, DecomposeProps) {
  std::vector<MeshGL> input;
  std::vector<Manifold> manifoldList;
  auto tet = WithPositionColors(Manifold::Tetrahedron());
  manifoldList.emplace_back(tet);
  input.emplace_back(tet.GetMeshGL());
  auto cube = WithPositionColors(Manifold::Cube());
  manifoldList.emplace_back(cube);
  input.emplace_back(cube.GetMeshGL());
  auto sphere = WithPositionColors(Manifold::Sphere(1, 4));
  manifoldList.emplace_back(sphere);
  input.emplace_back(sphere.GetMeshGL());
  Manifold manifolds = Manifold::Compose(manifoldList);

  ExpectMeshes(manifolds, {{8, 12, 3}, {6, 8, 3}, {4, 4, 3}});

  RelatedGL(manifolds, input);

  for (const Manifold& manifold : manifolds.Decompose()) {
    RelatedGL(manifold, input);
  }
}

/**
 * These tests check the various manifold constructors.
 */
TEST(Manifold, Sphere) {
  int n = 25;
  Manifold sphere = Manifold::Sphere(1.0, 4 * n);
  EXPECT_EQ(sphere.NumTri(), n * n * 8);
}

TEST(Manifold, Cylinder) {
  int n = 10000;
  Manifold cylinder = Manifold::Cylinder(2, 2, 2, n);
  EXPECT_EQ(cylinder.NumTri(), 4 * n - 4);
}

TEST(Manifold, Extrude) {
  Polygons polys = SquareHole();
  Manifold donut = Manifold::Extrude(polys, 1.0, 3);
  EXPECT_EQ(donut.Genus(), 1);
  EXPECT_FLOAT_EQ(donut.Volume(), 12.0);
  EXPECT_FLOAT_EQ(donut.SurfaceArea(), 48.0);
}

TEST(Manifold, ExtrudeCone) {
  Polygons polys = SquareHole();
  Manifold donut = Manifold::Extrude(polys, 1.0, 0, 0, vec2(0.0));
  EXPECT_EQ(donut.Genus(), 0);
  EXPECT_FLOAT_EQ(donut.Volume(), 4.0);
}

Polygons RotatePolygons(Polygons polys, const int index) {
  Polygons rotatedPolys;
  for (auto& polygon : polys) {
    auto rotatedPolygon = polygon;
    std::rotate(rotatedPolygon.begin(), rotatedPolygon.begin() + index,
                rotatedPolygon.end());
    rotatedPolys.push_back(rotatedPolygon);
  }
  return rotatedPolys;
}

TEST(Manifold, Revolve) {
  Polygons polys = SquareHole();
  Manifold vug;
  for (size_t i = 0; i < polys[0].size(); i++) {
    Polygons rotatedPolys = RotatePolygons(polys, i);
    vug = Manifold::Revolve(rotatedPolys, 48);
    EXPECT_EQ(vug.Genus(), -1);
    EXPECT_NEAR(vug.Volume(), 14.0 * kPi, 0.2);
    EXPECT_NEAR(vug.SurfaceArea(), 30.0 * kPi, 0.2);
  }
}

TEST(Manifold, Revolve2) {
  Polygons polys = SquareHole(2.0);
  Manifold donutHole = Manifold::Revolve(polys, 48);
  EXPECT_EQ(donutHole.Genus(), 0);
  EXPECT_NEAR(donutHole.Volume(), 48.0 * kPi, 1.0);
  EXPECT_NEAR(donutHole.SurfaceArea(), 96.0 * kPi, 1.0);
}

#ifdef MANIFOLD_CROSS_SECTION
TEST(Manifold, Revolve3) {
  CrossSection circle = CrossSection::Circle(1, 32);
  Manifold sphere = Manifold::Revolve(circle.ToPolygons(), 32);
  EXPECT_NEAR(sphere.Volume(), 4.0 / 3.0 * kPi, 0.1);
  EXPECT_NEAR(sphere.SurfaceArea(), 4 * kPi, 0.15);
}
#endif

TEST(Manifold, RevolveClip) {
  Polygons polys = {{{-5, -10}, {5, 0}, {-5, 10}}};
  Polygons clipped = {{{0, -5}, {5, 0}, {0, 5}}};
  Manifold first = Manifold::Revolve(polys, 48);
  Manifold second = Manifold::Revolve(clipped, 48);
  EXPECT_EQ(first.Genus(), second.Genus());
  EXPECT_EQ(first.Volume(), second.Volume());
  EXPECT_EQ(first.SurfaceArea(), second.SurfaceArea());
}

TEST(Manifold, PartialRevolveOnYAxis) {
  Polygons polys = SquareHole(2.0);
  Polygons offsetPolys = SquareHole(10.0);

  Manifold revolute;
  for (size_t i = 0; i < polys[0].size(); i++) {
    Polygons rotatedPolys = RotatePolygons(polys, i);
    revolute = Manifold::Revolve(rotatedPolys, 48, 180);
    EXPECT_EQ(revolute.Genus(), 1);
    EXPECT_NEAR(revolute.Volume(), 24.0 * kPi, 1.0);
    EXPECT_NEAR(revolute.SurfaceArea(),
                48.0 * kPi + 4.0 * 4.0 * 2.0 - 2.0 * 2.0 * 2.0, 1.0);
  }
}

TEST(Manifold, PartialRevolveOffset) {
  Polygons polys = SquareHole(10.0);

  Manifold revolute;
  for (size_t i = 0; i < polys[0].size(); i++) {
    Polygons rotatedPolys = RotatePolygons(polys, i);
    revolute = Manifold::Revolve(rotatedPolys, 48, 180);
    EXPECT_EQ(revolute.Genus(), 1);
    EXPECT_NEAR(revolute.SurfaceArea(), 777.0, 1.0);
    EXPECT_NEAR(revolute.Volume(), 376.0, 1.0);
  }
}

#ifdef MANIFOLD_CROSS_SECTION
TEST(Manifold, Warp) {
  CrossSection square = CrossSection::Square({1, 1});
  Manifold shape =
      Manifold::Extrude(square.ToPolygons(), 2, 10).Warp([](vec3& v) {
        v.x += v.z * v.z;
      });

  Manifold simplified = Manifold::Compose({shape});

  EXPECT_NEAR(shape.Volume(), simplified.Volume(), 0.0001);
  EXPECT_NEAR(shape.SurfaceArea(), simplified.SurfaceArea(), 0.0001);
  EXPECT_NEAR(shape.Volume(), 2, 0.0001);
}

TEST(Manifold, Warp2) {
  CrossSection circle = CrossSection::Circle(5, 20).Translate(vec2(10.0, 10.0));

  Manifold shape =
      Manifold::Extrude(circle.ToPolygons(), 2, 10).Warp([](vec3& v) {
        int nSegments = 10;
        double angleStep = 2.0 / 3.0 * kPi / nSegments;
        int zIndex = nSegments - 1 - std::round(v.z);
        double angle = zIndex * angleStep;
        v.z = v.y;
        v.y = v.x * sin(angle);
        v.x = v.x * cos(angle);
      });

  Manifold simplified = Manifold::Compose({shape});

  EXPECT_NEAR(shape.Volume(), simplified.Volume(), 0.0001);
  EXPECT_NEAR(shape.SurfaceArea(), simplified.SurfaceArea(), 0.0001);
  EXPECT_NEAR(shape.Volume(), 321, 1);
}
#endif

TEST(Manifold, WarpBatch) {
  Manifold cube = Manifold::Cube({2, 3, 4});
  const int id = cube.OriginalID();

  Manifold shape1 = cube.Warp([](vec3& v) { v.x += v.z * v.z; });
  Manifold shape2 = cube.WarpBatch([](VecView<vec3> vecs) {
    for (vec3& v : vecs) {
      v.x += v.z * v.z;
    }
  });

  EXPECT_GE(id, 0);
  EXPECT_EQ(shape1.OriginalID(), -1);
  EXPECT_EQ(shape2.OriginalID(), -1);
  std::vector<uint32_t> runOriginalID1 = shape1.GetMeshGL().runOriginalID;
  EXPECT_EQ(runOriginalID1.size(), 1);
  EXPECT_EQ(runOriginalID1[0], id);
  std::vector<uint32_t> runOriginalID2 = shape2.GetMeshGL().runOriginalID;
  EXPECT_EQ(runOriginalID2.size(), 1);
  EXPECT_EQ(runOriginalID2[0], id);
  EXPECT_EQ(shape1.Volume(), shape2.Volume());
  EXPECT_EQ(shape1.SurfaceArea(), shape2.SurfaceArea());
}

#ifdef MANIFOLD_CROSS_SECTION
TEST(Manifold, Project) {
  MeshGL input;
  input.numProp = 3;
  input.vertProperties = {0,    0,       0,     //
                          -2,   -0.7,    -0.1,  //
                          -2,   -0.7,    0,     //
                          -1.9, -0.7,    -0.1,  //
                          -1.9, -0.6901, -0.1,  //
                          -1.9, -0.7,    0,     //
                          -1.9, -0.6901, 0,     //
                          -2,   -1,      3,     //
                          -1.9, -1,      3,     //
                          -2,   -1,      4,     //
                          -1.9, -1,      4,     //
                          -1.9, -0.6901, 3,     //
                          -1.9, -0.6901, 4,     //
                          -1.7, -0.6901, 3,     //
                          -1.7, -0.6901, 3.2,   //
                          -2,   0,       -0.1,  //
                          -2,   0,       0,     //
                          -2,   0,       3,     //
                          -2,   0,       4,     //
                          -1.7, 0,       3,     //
                          -1.7, 0,       3.2,   //
                          -1,   -0.6901, -0.1,  //
                          -1,   -0.6901, 0,     //
                          -1,   -0.6901, 3.2,   //
                          -1,   -0.6901, 4,     //
                          -1,   0,       -0.1,  //
                          -1,   0,       0,     //
                          -1,   0,       3.2,   //
                          -1,   0,       4};
  input.triVerts = {1,  3,  2,   //
                    1,  4,  3,   //
                    2,  3,  5,   //
                    5,  6,  2,   //
                    3,  4,  6,   //
                    5,  3,  6,   //
                    6,  4,  21,  //
                    26, 22, 25,  //
                    21, 25, 22,  //
                    25, 15, 26,  //
                    26, 6,  22,  //
                    21, 4,  25,  //
                    21, 22, 6,   //
                    16, 26, 15,  //
                    16, 6,  26,  //
                    4,  15, 25,  //
                    15, 1,  16,  //
                    16, 2,  6,   //
                    4,  1,  15,  //
                    1,  2,  16,  //
                    12, 14, 23,  //
                    12, 13, 14,  //
                    12, 11, 13,  //
                    18, 9,  12,  //
                    11, 7,  17,  //
                    7,  9,  18,  //
                    17, 7,  18,  //
                    13, 11, 19,  //
                    17, 18, 20,  //
                    19, 11, 17,  //
                    19, 17, 20,  //
                    14, 13, 20,  //
                    18, 12, 24,  //
                    20, 13, 19,  //
                    20, 18, 27,  //
                    12, 10, 11,  //
                    24, 12, 23,  //
                    9,  10, 12,  //
                    9,  8,  10,  //
                    8,  11, 10,  //
                    8,  7,  11,  //
                    8,  9,  7,   //
                    14, 20, 27,  //
                    24, 28, 18,  //
                    27, 18, 28,  //
                    23, 14, 27,  //
                    24, 23, 28,  //
                    28, 23, 27};
  Manifold in(input);
  CrossSection projected = in.Project();
  EXPECT_NEAR(projected.Area(), 0.72, 0.01);
}
#endif

/**
 * Testing more advanced Manifold operations.
 */

TEST(Manifold, Transform) {
  Manifold cube = Manifold::Cube({1, 2, 3});
  Manifold cube2 = cube;
  cube = cube.Rotate(30, 40, 50).Scale({6, 5, 4}).Translate({1, 2, 3});

  mat3 rX({1.0, 0.0, 0.0},            //
          {0.0, cosd(30), sind(30)},  //
          {0.0, -sind(30), cosd(30)});
  mat3 rY({cosd(40), 0.0, -sind(40)},  //
          {0.0, 1.0, 0.0},             //
          {sind(40), 0.0, cosd(40)});
  mat3 rZ({cosd(50), sind(50), 0.0},   //
          {-sind(50), cosd(50), 0.0},  //
          {0.0, 0.0, 1.0});
  mat3 s;
  s[0][0] = 6;
  s[1][1] = 5;
  s[2][2] = 4;
  mat3x4 transform = mat3x4(s * rZ * rY * rX, vec3(0.0));
  transform[3] = vec3(1, 2, 3);
  cube2 = cube2.Transform(transform);

  Identical(cube.GetMeshGL(), cube2.GetMeshGL());
}

#ifdef MANIFOLD_CROSS_SECTION
TEST(Manifold, Slice) {
  Manifold cube = Manifold::Cube();
  CrossSection bottom = cube.Slice();
  CrossSection top = cube.Slice(1);
  EXPECT_EQ(bottom.Area(), 1);
  EXPECT_EQ(top.Area(), 0);
}

TEST(Manifold, SliceEmptyObject) {
  Manifold empty;
  EXPECT_TRUE(empty.IsEmpty());
  CrossSection bottom = empty.Slice();
}

TEST(Manifold, Simplify) {
  Polygons polyCircle =
      CrossSection::Circle(1, 20).Translate({10, 0}).ToPolygons();
  Manifold torus = Manifold::Revolve(polyCircle, 100);
  Manifold simplified = torus.Simplify(0.4);
  EXPECT_NEAR(torus.Volume(), simplified.Volume(), 20);
  EXPECT_NEAR(torus.SurfaceArea(), simplified.SurfaceArea(), 10);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("torus.glb", simplified.GetMeshGL(), {});
#endif
}
#endif

TEST(Manifold, MeshID) {
  const Manifold cube = Manifold::Cube();
  MeshGL cubeGL = cube.GetMeshGL();
  cubeGL.runIndex.clear();
  cubeGL.runOriginalID.clear();
  Manifold cube1 = Manifold(cubeGL);
  Manifold cube2 = Manifold(cubeGL);
  EXPECT_NE(cube1.GetMeshGL().runOriginalID[0],
            cube2.GetMeshGL().runOriginalID[0]);
}

TEST(Manifold, MeshRelation) {
  Manifold gyroid = WithPositionColors(Gyroid());
  MeshGL gyroidMeshGL = gyroid.GetMeshGL();
  gyroid = gyroid.Simplify();

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorIdx = 0;
  if (options.exportModels) ExportMesh("gyroid.glb", gyroid.GetMeshGL(), opt);
#endif

  RelatedGL(gyroid, {gyroidMeshGL});
}

TEST(Manifold, MeshRelationTransform) {
  const Manifold cube = Manifold::Cube();
  const MeshGL cubeGL = cube.GetMeshGL();
  const Manifold turned = cube.Rotate(45, 90);

  RelatedGL(turned, {cubeGL});
}

TEST(Manifold, MeshRelationRefine) {
  Manifold csaszar = WithPositionColors(Csaszar()).AsOriginal();
  MeshGL inGL = csaszar.GetMeshGL();

  RelatedGL(csaszar, {inGL});
  csaszar = csaszar.RefineToLength(1);
  ExpectMeshes(csaszar, {{9019, 18038, 3}});
  RelatedGL(csaszar, {inGL});

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorIdx = 0;
  if (options.exportModels) ExportMesh("csaszar.glb", csaszar.GetMeshGL(), opt);
#endif
}

TEST(Manifold, MeshRelationRefinePrecision) {
  MeshGL inGL = WithPositionColors(Csaszar()).GetMeshGL();
  const int id = inGL.runOriginalID[0];
  Manifold csaszar = Manifold::Smooth(inGL);

  csaszar = csaszar.RefineToTolerance(0.05);
  ExpectMeshes(csaszar, {{2684, 5368, 3}});
  std::vector<uint32_t> runOriginalID = csaszar.GetMeshGL().runOriginalID;
  EXPECT_EQ(runOriginalID.size(), 1);
  EXPECT_EQ(runOriginalID[0], id);

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorIdx = 0;
  if (options.exportModels)
    ExportMesh("csaszarSmooth.glb", csaszar.GetMeshGL(), opt);
#endif
}

TEST(Manifold, MeshGLRoundTrip) {
  const Manifold cylinder = Manifold::Cylinder(2, 1);
  ASSERT_GE(cylinder.OriginalID(), 0);
  MeshGL inGL = cylinder.GetMeshGL();
  const Manifold cylinder2(inGL);
  const MeshGL outGL = cylinder2.GetMeshGL();

  ASSERT_EQ(inGL.runOriginalID.size(), 1);
  ASSERT_EQ(outGL.runOriginalID.size(), 1);
  ASSERT_EQ(outGL.runOriginalID[0], inGL.runOriginalID[0]);

  RelatedGL(cylinder2, {inGL});
}

void CheckCube(const MeshGL& cubeSTL) {
  Manifold cube(cubeSTL);
  cube = cube.AsOriginal();
  EXPECT_EQ(cube.NumTri(), 12);
  EXPECT_EQ(cube.NumVert(), 8);
  EXPECT_EQ(cube.NumPropVert(), 24);

  EXPECT_FLOAT_EQ(cube.Volume(), 1.0);
  EXPECT_FLOAT_EQ(cube.SurfaceArea(), 6.0);
}

TEST(Manifold, Merge) {
  MeshGL cubeSTL = CubeSTL();
  EXPECT_EQ(cubeSTL.NumTri(), 12);
  EXPECT_EQ(cubeSTL.NumVert(), 36);

  Manifold cubeBad(cubeSTL);
  EXPECT_TRUE(cubeBad.IsEmpty());
  EXPECT_EQ(cubeBad.Status(), Manifold::Error::NotManifold);

  EXPECT_TRUE(cubeSTL.Merge());
  EXPECT_EQ(cubeSTL.mergeFromVert.size(), 28);
  CheckCube(cubeSTL);

  EXPECT_FALSE(cubeSTL.Merge());
  EXPECT_EQ(cubeSTL.mergeFromVert.size(), 28);
  cubeSTL.mergeFromVert.resize(14);
  cubeSTL.mergeToVert.resize(14);

  EXPECT_TRUE(cubeSTL.Merge());
  EXPECT_EQ(cubeSTL.mergeFromVert.size(), 28);
  CheckCube(cubeSTL);
}

TEST(Manifold, MergeEmpty) {
  MeshGL shape;
  shape.numProp = 7;
  shape.triVerts = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
  shape.vertProperties = {0.0,  0.5,  0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          0.0,  -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          0.0,  0.5,  -0.43450000882149, 0.0, 0.0, 0.0, 1.0,
                          0.0,  -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          0.0,  0.5,  0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          0.0,  -0.5, 0.434500008821487, 0.0, 0.0, 1.0, 0.0,
                          0.0,  0.5,  0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          -0.0, 0.5,  -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          -0.0, 0.5,  0.434500008821487, 0.0, 0.0, 0.0, 1.0,
                          -0.0, 0.5,  -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          0.0,  0.5,  0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          0.0,  0.5,  -0.43450000882149, 0.0, 0.0, 1.0, 0.0,
                          0.0,  0.5,  0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          -0.0, -0.5, 0.434500008821487, 0.0, 0.0, 1.0, 1.0,
                          0.0,  -0.5, 0.434500008821487, 0.0, 0.0, 0.0, 1.0,
                          -0.0, -0.5, 0.434500008821487, 0.0, 0.0, 1.0, 1.0,
                          0.0,  0.5,  0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          -0.0, 0.5,  0.434500008821487, 0.0, 0.0, 1.0, 0.0,
                          -0.0, 0.5,  -0.43450000882149, 0.0, 0.0, 0.0, 0.0,
                          -0.0, -0.5, 0.434500008821487, 0.0, 0.0, 1.0, 1.0,
                          -0.0, 0.5,  0.434500008821487, 0.0, 0.0, 0.0, 1.0,
                          -0.0, -0.5, 0.434500008821487, 0.0, 0.0, 1.0, 1.0,
                          -0.0, 0.5,  -0.43450000882149, 0.0, 0.0, 0.0, 0.0,
                          -0.0, -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 0.0,
                          -0.0, -0.5, 0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          0.0,  -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          0.0,  -0.5, 0.434500008821487, 0.0, 0.0, 0.0, 1.0,
                          0.0,  -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          -0.0, -0.5, 0.434500008821487, 0.0, 0.0, 0.0, 0.0,
                          -0.0, -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 0.0,
                          0.0,  -0.5, -0.43450000882149, 0.0, 0.0, 0.0, 0.0,
                          -0.0, 0.5,  -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          0.0,  0.5,  -0.43450000882149, 0.0, 0.0, 0.0, 1.0,
                          -0.0, 0.5,  -0.43450000882149, 0.0, 0.0, 1.0, 1.0,
                          0.0,  -0.5, -0.43450000882149, 0.0, 0.0, 0.0, 0.0,
                          -0.0, -0.5, -0.43450000882149, 0.0, 0.0, 1.0, 0.0};
  EXPECT_TRUE(shape.Merge());
  Manifold man(shape);
  EXPECT_EQ(man.Status(), Manifold::Error::NoError);
  EXPECT_EQ(man.NumTri(), 4);
  EXPECT_TRUE(man.Simplify().IsEmpty());
}

TEST(Manifold, PinchedVert) {
  MeshGL shape;
  shape.numProp = 3;
  shape.vertProperties = {0,        0,  0,   //
                          1,        1,  0,   //
                          1,        -1, 0,   //
                          -0.00001, 0,  0,   //
                          -1,       -1, -0,  //
                          -1,       1,  0,   //
                          0,        0,  2,   //
                          0,        0,  -2};
  shape.triVerts = {0, 2, 6,  //
                    2, 1, 6,  //
                    1, 0, 6,  //
                    4, 3, 6,  //
                    3, 5, 6,  //
                    5, 4, 6,  //
                    2, 0, 4,  //
                    0, 3, 4,  //
                    3, 0, 1,  //
                    3, 1, 5,  //
                    7, 2, 4,  //
                    7, 4, 5,  //
                    7, 5, 1,  //
                    7, 1, 2};
  Manifold touch(shape);
  EXPECT_FALSE(touch.IsEmpty());
  EXPECT_EQ(touch.Status(), Manifold::Error::NoError);
  EXPECT_EQ(touch.Genus(), 0);
}

TEST(Manifold, FaceIDRoundTrip) {
  const Manifold cube = Manifold::Cube();
  ASSERT_GE(cube.OriginalID(), 0);
  MeshGL inGL = cube.GetMeshGL();
  ASSERT_EQ(NumUnique(inGL.faceID), 6);
  inGL.faceID = {3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5};

  const Manifold cube2(inGL);
  const MeshGL outGL = cube2.GetMeshGL();
  ASSERT_EQ(NumUnique(outGL.faceID), 2);
}

TEST(Manifold, MirrorUnion) {
  auto a = Manifold::Cube({5., 5., 5.}, true);
  auto b = a.Translate({2.5, 2.5, 2.5});
  auto result = a + b + b.Mirror({1, 1, 0});

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("manifold_mirror_union.glb", result.GetMeshGL(), {});
#endif

  auto vol_a = a.Volume();
  EXPECT_FLOAT_EQ(vol_a * 2.75, result.Volume());
  EXPECT_TRUE(a.Mirror(vec3(0.0)).IsEmpty());
}

TEST(Manifold, MirrorUnion2) {
  auto a = Manifold::Cube();
  auto result = Manifold::Compose({a.Mirror({1, 0, 0})});
  EXPECT_TRUE(result.MatchesTriNormals());
}

#ifdef MANIFOLD_CROSS_SECTION
TEST(Manifold, Invalid) {
  auto invalid = Manifold::Error::InvalidConstruction;
  auto circ = CrossSection::Circle(10.);
  auto empty_circ = CrossSection::Circle(-2.);
  auto empty_sq = CrossSection::Square(vec2(0.0));

  EXPECT_EQ(Manifold::Sphere(0).Status(), invalid);
  EXPECT_EQ(Manifold::Cylinder(0, 5).Status(), invalid);
  EXPECT_EQ(Manifold::Cylinder(2, -5).Status(), invalid);
  EXPECT_EQ(Manifold::Cube(vec3(0.0)).Status(), invalid);
  EXPECT_EQ(Manifold::Cube({-1, 1, 1}).Status(), invalid);
  EXPECT_EQ(Manifold::Extrude(circ.ToPolygons(), 0.).Status(), invalid);
  EXPECT_EQ(Manifold::Extrude(empty_circ.ToPolygons(), 10.).Status(), invalid);
  EXPECT_EQ(Manifold::Revolve(empty_sq.ToPolygons()).Status(), invalid);
}
#endif

TEST(Manifold, MultiCompose) {
  auto part = Manifold::Compose({Manifold::Cube({10, 10, 10})});
  auto finalAssembly =
      Manifold::Compose({part, part.Translate({0, 10, 0}),
                         part.Mirror({1, 0, 0}).Translate({10, 0, 0}),
                         part.Mirror({1, 0, 0}).Translate({10, 10, 0})});
  EXPECT_FLOAT_EQ(finalAssembly.Volume(), 4000);
}

TEST(Manifold, MergeDegenerates) {
  MeshGL cube = Manifold::Cube(vec3(1), true).GetMeshGL();
  MeshGL squash;
  squash.vertProperties = cube.vertProperties;
  squash.triVerts = cube.triVerts;
  // Move one vert to the position of its neighbor and remove one triangle
  // linking them to break the manifold.
  squash.vertProperties[squash.vertProperties.size() - 1] *= -1;
  squash.triVerts.resize(squash.triVerts.size() - 3);
  // Rotate the degenerate triangle to the middle to catch more problems.
  std::rotate(squash.triVerts.begin(), squash.triVerts.begin() + 3 * 5,
              squash.triVerts.end());
  // Merge should remove the now duplicate vertex.
  EXPECT_TRUE(squash.Merge());
  // Manifold should remove the triangle with two references to the same vert.
  Manifold squashed = Manifold(squash);
  EXPECT_FALSE(squashed.IsEmpty());
  EXPECT_EQ(squashed.Status(), Manifold::Error::NoError);
}

#ifdef MANIFOLD_EXPORT
TEST(Manifold, MergeRefine) {
  MeshGL mesh;
  mesh.tolerance = 1e-5;
  mesh.vertProperties = {
      0.01383194,  -1.88699961, -1.09618223, -0.72694844, -1.30568409,
      -1.09618223, -0.72694844, -1.30568397, -1.09618223, -1.14769018,
      0.00177073,  -1.09618223, -0.72777849, 1.30731535,  -1.09618223,
      -0.72777843, 1.30731547,  -1.09618223, 0.01344040,  1.88713050,
      -1.09618223, 0.75422078,  1.30581498,  -1.09618223, 0.75422555,
      1.30580890,  -1.09618223, 1.14608598,  -0.00000003, -1.09618223,
      0.75343317,  -1.30844986, -1.09618223, 0.00032274,  2.45551205,
      0.70305431,  -0.00045545, 0.49932927,  1.63340366,  -0.00045485,
      0.49933004,  1.63340306,  0.59210259,  2.54137087,  0.46526274,
      -0.00083422, -0.50377685, 1.63340366,  -0.00153223, -2.45697832,
      0.70305431,  0.59018338,  -2.54327321, 0.46526241,  0.59018356,
      -2.54327297, 0.46526247,  -0.00083402, -0.50377715, 1.63340342,
      -0.00153225, -2.45697832, 0.70305425,  -0.00083453, -0.50377727,
      1.63340330,  -0.59329146, -2.54207492, 0.46526280,  -0.00045549,
      0.49932927,  1.63340366,  -0.59137100, 2.54256892,  0.46526301,
      -0.59137243, 2.54256916,  0.46526241,  -1.09916806, 1.80846310,
      0.62867594,  -1.09916782, 1.80846262,  0.62867624,  1.09932983,
      1.80623639,  0.62867647,  1.09933031,  1.80623722,  0.62867594,
      0.59210330,  2.54137087,  0.46526241,  1.09796500,  -1.80969083,
      0.62867600,  -1.10053313, -1.80746412, 0.62867641,  -1.10053349,
      -1.80746484, 0.62867594,  -0.59329236, -2.54207492, 0.46526238,
      -0.00045545, 0.49932924,  1.63340366,  -0.45734894, -0.00169915,
      1.59454787,  -0.00065268, -0.00251516, 1.81522143,  0.45606264,
      -0.00262397, 1.59454787,  0.45606312,  -0.00262326, 1.59454739,
      0.45606264,  -0.00262397, 1.59454787,  -0.00065267, -0.00251517,
      1.81522143,  -0.00083422, -0.50377685, 1.63340366,  1.09796488,
      -1.80969071, 0.62867606,  0.45606279,  -0.00262419, 1.59454775,
      -0.45734891, -0.00169916, 1.59454787,  -0.45734927, -0.00169968,
      1.59454751,  -1.56936228, 1.35052788,  0.52379429,  -0.45734900,
      -0.00169913, 1.59454787,  1.56918728,  1.34735024,  0.52379429,
      1.56811047,  -1.35147583, 0.52380723,  1.56808507,  -1.35131860,
      0.52387440,  -1.57038140, -1.34824181, 0.52379429,  -1.56936240,
      1.35052788,  0.52379423,  -1.44328797, 0.00054518,  0.88277382,
      -1.44327283, 0.00043958,  0.88280576,  1.44195509,  -0.00248179,
      0.88287276,  1.44223559,  -0.00054497, 0.88228679,  1.56918740,
      1.34735012,  0.52379423,  -1.57040441, -1.34818101, 0.52378970,
      -1.57038152, -1.34824169, 0.52379423,  0.00060985,  3.12254715,
      -0.12592883, 0.00060586,  3.12254739,  -0.12593007, -0.21464077,
      3.10757208,  -0.15445986, -0.42353559, 3.05566168,  -0.13569236,
      -0.62559998, 2.98231292,  -0.08878353, 0.62670720,  2.98100901,
      -0.08873826, 0.62661505,  2.98107791,  -0.08880345, 0.42469645,
      3.05480289,  -0.13569044, 0.21584108,  3.10713625,  -0.15445758,
      -0.00174801, -3.12165236, -0.12629783, 0.21322219,  -3.10669637,
      -0.15490441, 0.42281535,  -3.05461216, -0.13599920, 0.62440765,
      -2.98143482, -0.08901373, 0.62447023,  -2.98138809, -0.08896933,
      -0.62786478, -2.98014331, -0.08899876, -0.62786037, -2.98014665,
      -0.08900189, -0.42584255, -3.05390787, -0.13610038, -0.21698712,
      -3.10624146, -0.15494294, -0.00175192, -3.12165260, -0.12629905,
      -0.62560898, 2.98230958,  -0.08879187, -0.78726906, 2.86185288,
      -0.16757384, -0.93646783, 2.72652268,  -0.21237959, -1.07508647,
      2.58084154,  -0.23096853, -1.20526373, 2.42826319,  -0.23021236,
      -1.32975245, 2.27157402,  -0.21704006, -1.45062196, 2.05095077,
      -0.12514453, -1.45065486, 2.05087924,  -0.12510653, 1.45106840,
      2.04791498,  -0.12508209, 1.45102346,  2.04801273,  -0.12513401,
      1.33032060,  2.26888084,  -0.21701908, 1.20595014,  2.42582202,
      -0.23018987, 1.07588828,  2.57866383,  -0.23094597, 0.93737990,
      2.72462535,  -0.21235918, 0.78851759,  2.86004448,  -0.16762893,
      0.78625345,  -2.86083961, -0.16809504, 0.93532181,  -2.72562766,
      -0.21302269, 1.07394040,  -2.57994676, -0.23167852, 1.20411777,
      -2.42736840, -0.23091961, 1.32860637,  -2.27067924, -0.21769993,
      1.44944537,  -2.05011153, -0.12549706, 1.56815219,  -1.35151458,
      0.52370584,  -1.45212758, -2.04719448, -0.12551625, -1.33146667,
      -2.26798582, -0.21772107, -1.20709634, -2.42492723, -0.23094229,
      -1.07703435, -2.57776904, -0.23170128, -0.93852592, -2.72373033,
      -0.21304333, -0.78942937, -2.85936260, -0.16807115, -1.80910003,
      1.01058388,  -0.03365797, -1.84168148, 0.80862963,  -0.10271203,
      -1.86654651, 0.60610521,  -0.15332751, -1.88623869, 0.40378988,
      -0.19160835, -1.90002000, 0.20241861,  -0.21582428, -1.90505385,
      0.00237573,  -0.21922338, -1.90501344, 0.00071962,  -0.21905561,
      -1.90017104, -0.19767725, -0.21568473, -1.88654184, -0.39907673,
      -0.19146930, -1.86700249, -0.60143071, -0.15318951, -1.84229040,
      -0.80400622, -0.10257494, -1.80987811, -1.00592279, -0.03355779,
      1.90388918,  -0.00071938, -0.21917287, 1.89902496,  0.19857220,
      -0.21578738, 1.88539588,  0.39997119,  -0.19157638, 1.86585641,
      0.60232592,  -0.15330335, 1.84114432,  0.80490118,  -0.10269797,
      1.80871546,  1.00692093,  -0.03365800, 1.80794466,  -1.00973761,
      -0.03369782, 1.80795383,  -1.00968957, -0.03371784, 1.84053540,
      -0.80773503, -0.10275956, 1.86540043,  -0.60521013, -0.15336604,
      1.88509274,  -0.40289518, -0.19163992, 1.89881432,  -0.20239551,
      -0.21574667, -1.54677641, 1.84270406,  -0.22365679, -1.63208210,
      1.62775970,  -0.25862685, -1.70706439, 1.40805590,  -0.23586856,
      -1.77522779, 1.18613660,  -0.17659405, -1.80909991, 1.01058459,
      -0.03365869, 1.80871534,  1.00692177,  -0.03365877, 1.77497590,
      1.18254220,  -0.17653592, 1.70699000,  1.40456688,  -0.23577766,
      1.63230419,  1.62443995,  -0.25918254, 1.54714429,  1.83953130,
      -0.22410297, -1.80986142, -1.00602686, -0.03365880, -1.77612185,
      -1.18164730, -0.17739406, -1.70812643, -1.40370321, -0.23700003,
      -1.63330925, -1.62356102, -0.25988522, -1.54816461, -1.83868098,
      -0.22471850, -1.45216942, -2.04711795, -0.12558441, 1.44947588,
      -2.05005574, -0.12554680, 1.54517174,  -1.84302914, -0.22456542,
      1.63157570,  -1.62538552, -0.26028806, 1.70592797,  -1.40712976,
      -0.23690872, 1.77408183,  -1.18524182, -0.17733525, 0.00060586,
      3.12254739,  -0.46959737, -0.21464077, 3.10757208,  -0.45468479,
      0.00060591,  3.12254739,  -0.46959740, 0.21584108,  3.10713625,
      -0.45459676, -0.00175192, -3.12165260, -0.46959740, 0.21322219,
      -3.10669637, -0.45459723, -0.00175193, -3.12165260, -0.46959740,
      -0.21698712, -3.10624146, -0.45467842, -0.42353559, 3.05566168,
      -0.46107247, 0.42469645,  3.05480289,  -0.46099803, 0.42281535,
      -3.05461216, -0.46102279, -0.42584255, -3.05390787, -0.46106711,
      -0.62560898, 2.98230958,  -0.48016420, 0.62440765,  -2.98143482,
      -0.48015898, 0.62661505,  2.98107791,  -0.48015478, -0.62786037,
      -2.98014665, -0.48016420, -0.78726906, 2.86185288,  -0.45703074,
      0.62445766,  -2.98139739, -0.48017445, 0.78625345,  -2.86083961,
      -0.45658660, 0.78851759,  2.86004448,  -0.45661709, 0.62670475,
      2.98101068,  -0.48018256, -0.78942937, -2.85936260, -0.45704728,
      -0.62786043, -2.98014665, -0.48016423, -0.93646783, 2.72652268,
      -0.44622171, 0.93532181,  -2.72562766, -0.44563186, -0.93852592,
      -2.72373033, -0.44624415, 0.93737990,  2.72462535,  -0.44568357,
      -1.07508647, 2.58084154,  -0.44488293, 1.07394040,  -2.57994676,
      -0.44435173, -1.07703435, -2.57776904, -0.44490317, 1.07588828,
      2.57866383,  -0.44439828, -1.20526373, 2.42826319,  -0.45050618,
      1.20411777,  -2.42736840, -0.45019045, -1.20709634, -2.42492723,
      -0.45051819, 1.20595014,  2.42582202,  -0.45021811, -1.32975245,
      2.27157426,  -0.46058667, -1.32975245, 2.27157402,  -0.46058667,
      1.32860637,  -2.27067924, -0.46058667, -1.33146667, -2.26798582,
      -0.46058667, -1.33146656, -2.26798606, -0.46058667, 1.33032060,
      2.26888084,  -0.46058667, 1.33032048,  2.26888084,  -0.46058667,
      -1.45062196, 2.05095077,  -0.45238319, 1.32860649,  -2.27067924,
      -0.46058670, 1.44947588,  -2.05005574, -0.45129201, -1.45216942,
      -2.04711795, -0.45243445, 1.45102346,  2.04801273,  -0.45137730,
      -1.54677641, 1.84270406,  -0.45860177, 1.54517174,  -1.84302914,
      -0.45671692, -1.54816461, -1.83868098, -0.45867091, 1.54714429,
      1.83953130,  -0.45686778, -1.63208210, 1.62775970,  -0.47587365,
      -1.63330925, -1.62356102, -0.47593805, 1.63157570,  -1.62538552,
      -0.47375324, 1.63230419,  1.62443995,  -0.47368667, -1.70706439,
      1.40805590,  -0.50303531, -1.70812643, -1.40370321, -0.50307423,
      1.70592797,  -1.40712976, -0.50065041, 1.70699000,  1.40456688,
      -0.50072700, -1.77522779, 1.18613660,  -0.53636789, -1.77612185,
      -1.18164730, -0.53636789, 1.77308404,  -1.18848991, -0.53344285,
      1.77408183,  -1.18524182, -0.53272378, 1.77497590,  1.18254220,
      -0.53206521, 1.77341759,  1.18763101,  -0.53318721, -1.80909991,
      1.01058459,  -0.51588774, -1.80986142, -1.00602686, -0.51592720,
      -1.77612185, -1.18164718, -0.53636789, 1.80795383,  -1.00968957,
      -0.51267529, 1.80871534,  1.00692177,  -0.51214570, -1.84168148,
      0.80862963,  -0.49827772, -1.84229040, -0.80400622, -0.49834538,
      1.84053540,  -0.80773503, -0.49560285, 1.84114432,  0.80490118,
      -0.49522084, -1.86654651, 0.60610521,  -0.48789516, -1.86700249,
      -0.60143071, -0.48796996, 1.86540043,  -0.60521013, -0.48580948,
      1.86585641,  0.60232592,  -0.48557436, -1.88623869, 0.40378988,
      -0.48232013, -1.88654184, -0.39907673, -0.48238823, 1.88509274,
      -0.40289518, -0.48085603, 1.88539588,  0.39997119,  -0.48076698,
      -1.90002000, 0.20241861,  -0.48219424, -1.90017104, -0.19767725,
      -0.48223990, 1.89881432,  -0.20239551, -0.48138389, 1.89902496,
      0.19857220,  -0.48143867, 1.89892781,  0.20000805,  -0.48143691,
      -1.90505385, 0.00237582,  -0.49013662, -1.90505385, 0.00237573,
      -0.49013659, 1.90388918,  -0.00071975, -0.21917289, 1.90388918,
      -0.00071975, -0.49010351, 1.90388906,  -0.00072469, -0.49010390,
      1.90128529,  0.00611682,  -0.49218670, 1.40738547,  1.02835560,
      -0.88730782, 1.40482545,  -1.03119016, -0.88952076, 1.40470743,
      -1.02958739, -0.88973862, -1.41726339, 1.01772320,  -0.88060957,
      -1.90505385, 0.00237566,  -0.49013662, -1.41691077, -1.01434207,
      -0.88060957, 1.32860637,  -2.27067900, -0.46058673, 1.40398669,
      -1.03246403, -0.88968676, 1.40373409,  1.02831805,  -0.89023006,
      -1.32975245, 2.27157402,  -0.46058670, -1.41726327, 1.01772332,
      -0.88060963, -1.41726327, 1.01772320,  -0.88060963, -1.41691077,
      -1.01434207, -0.88060963, -1.14769018, 0.00177076,  -1.09618223,
      1.14696825,  -0.00000003, -1.09586132, 1.40373087,  1.02831531,
      -0.89023262, 0.40889478,  -2.28727150, -0.89325231, 0.40938482,
      2.28683925,  -0.89325225, -0.62560904, 2.98230958,  -0.48016423,
      -0.39072043, 2.28764248,  -0.89325225, -0.39121491, -2.28645349,
      -0.89325231, 0.40889484,  -2.28727150, -0.89325231, 0.75434840,
      -1.30917358, -1.09577036, 0.40938476,  2.28683901,  -0.89325231,
      -0.39072037, 2.28764224,  -0.89325231, -0.00175192, -3.12165236,
      -0.46959746, 0.40889472,  -2.28727150, -0.89325231, 0.00060591,
      3.12254691,  -0.46959764, -0.39072025, 2.28764224,  -0.89325237,
      0.40938473,  2.28683901,  -0.89325237, 0.01344044,  1.88713050,
      -1.09618223, 0.00032271,  2.45551205,  0.70305431};
  mesh.triVerts = {
      0,   1,   2,   0,   2,   3,   0,   3,   4,   0,   4,   5,   0,   5,   6,
      0,   6,   7,   0,   7,   8,   0,   8,   9,   9,   10,  0,   11,  12,  13,
      13,  14,  11,  15,  16,  17,  15,  17,  18,  18,  19,  15,  20,  15,  21,
      21,  22,  20,  23,  24,  25,  23,  25,  26,  26,  27,  23,  14,  13,  28,
      14,  28,  29,  29,  30,  14,  19,  18,  31,  22,  21,  32,  22,  32,  33,
      33,  34,  22,  35,  27,  36,  36,  37,  35,  12,  37,  38,  12,  38,  39,
      12,  39,  28,  40,  41,  42,  40,  42,  19,  40,  19,  43,  43,  44,  40,
      42,  41,  45,  42,  45,  46,  42,  46,  32,  32,  21,  42,  47,  48,  27,
      27,  26,  47,  28,  39,  49,  49,  29,  28,  43,  31,  50,  43,  50,  51,
      51,  44,  43,  32,  46,  52,  52,  33,  32,  48,  47,  53,  48,  53,  54,
      54,  55,  48,  44,  51,  56,  56,  40,  44,  38,  56,  57,  38,  57,  58,
      38,  58,  49,  49,  39,  38,  45,  55,  59,  45,  59,  60,  45,  60,  52,
      52,  46,  45,  24,  11,  61,  24,  61,  62,  24,  62,  63,  24,  63,  64,
      24,  64,  65,  65,  25,  24,  61,  11,  14,  61,  14,  30,  61,  30,  66,
      61,  66,  67,  61,  67,  68,  68,  69,  61,  17,  20,  70,  17,  70,  71,
      17,  71,  72,  17,  72,  73,  73,  74,  17,  70,  20,  22,  70,  22,  34,
      70,  34,  75,  70,  75,  76,  70,  76,  77,  70,  77,  78,  78,  79,  70,
      25,  65,  80,  25,  80,  81,  25,  81,  82,  25,  82,  83,  25,  83,  84,
      25,  84,  85,  25,  85,  86,  25,  86,  87,  25,  87,  53,  25,  53,  47,
      47,  26,  25,  66,  30,  29,  66,  29,  49,  66,  49,  58,  66,  58,  88,
      66,  88,  89,  66,  89,  90,  66,  90,  91,  66,  91,  92,  66,  92,  93,
      93,  94,  66,  17,  74,  95,  17,  95,  96,  17,  96,  97,  17,  97,  98,
      17,  98,  99,  17,  99,  100, 17,  100, 101, 17,  101, 50,  17,  50,  31,
      52,  60,  102, 52,  102, 103, 52,  103, 104, 52,  104, 105, 52,  105, 106,
      52,  106, 107, 52,  107, 75,  52,  75,  34,  34,  33,  52,  54,  53,  108,
      54,  108, 109, 54,  109, 110, 54,  110, 111, 54,  111, 112, 54,  112, 113,
      113, 114, 54,  54,  114, 115, 54,  115, 116, 54,  116, 117, 54,  117, 118,
      54,  118, 119, 54,  119, 59,  59,  55,  54,  58,  57,  120, 58,  120, 121,
      58,  121, 122, 58,  122, 123, 58,  123, 124, 124, 125, 58,  120, 57,  56,
      120, 56,  51,  120, 51,  50,  120, 50,  101, 120, 101, 126, 120, 126, 127,
      120, 127, 128, 120, 128, 129, 120, 129, 130, 120, 130, 131, 53,  87,  132,
      53,  132, 133, 53,  133, 134, 53,  134, 135, 53,  135, 136, 136, 108, 53,
      58,  125, 137, 58,  137, 138, 58,  138, 139, 58,  139, 140, 58,  140, 141,
      141, 88,  58,  59,  119, 142, 59,  142, 143, 59,  143, 144, 59,  144, 145,
      59,  145, 146, 59,  146, 147, 59,  147, 102, 102, 60,  59,  101, 100, 148,
      101, 148, 149, 101, 149, 150, 101, 150, 151, 101, 151, 152, 152, 126, 101,
      62,  153, 154, 154, 63,  62,  155, 62,  61,  155, 61,  69,  69,  156, 155,
      79,  157, 158, 79,  158, 71,  71,  70,  79,  159, 79,  78,  78,  160, 159,
      63,  154, 161, 161, 64,  63,  69,  68,  162, 162, 156, 69,  71,  158, 163,
      163, 72,  71,  78,  77,  164, 164, 160, 78,  64,  161, 165, 64,  165, 80,
      80,  65,  64,  72,  163, 166, 166, 73,  72,  68,  67,  167, 167, 162, 68,
      77,  76,  168, 168, 164, 77,  80,  165, 169, 169, 81,  80,  73,  166, 170,
      73,  170, 171, 73,  171, 95,  95,  74,  73,  66,  94,  172, 66,  172, 173,
      66,  173, 167, 167, 67,  66,  75,  107, 174, 75,  174, 175, 175, 76,  75,
      81,  169, 176, 176, 82,  81,  95,  171, 177, 177, 96,  95,  107, 106, 178,
      178, 174, 107, 94,  93,  179, 179, 172, 94,  82,  176, 180, 180, 83,  82,
      96,  177, 181, 181, 97,  96,  106, 105, 182, 182, 178, 106, 93,  92,  183,
      183, 179, 93,  83,  180, 184, 184, 84,  83,  97,  181, 185, 185, 98,  97,
      105, 104, 186, 186, 182, 105, 92,  91,  187, 187, 183, 92,  84,  184, 188,
      189, 85,  84,  98,  185, 190, 190, 99,  98,  104, 103, 191, 192, 186, 104,
      91,  90,  193, 194, 187, 91,  85,  189, 195, 195, 86,  85,  99,  196, 197,
      99,  197, 148, 148, 100, 99,  102, 147, 198, 102, 198, 191, 191, 103, 102,
      90,  89,  199, 199, 193, 90,  86,  195, 200, 86,  200, 132, 132, 87,  86,
      148, 197, 201, 201, 149, 148, 147, 146, 202, 202, 198, 147, 88,  141, 203,
      88,  203, 199, 199, 89,  88,  132, 200, 204, 204, 133, 132, 146, 145, 205,
      205, 202, 146, 149, 201, 206, 206, 150, 149, 141, 140, 207, 207, 203, 141,
      133, 204, 208, 208, 134, 133, 145, 144, 209, 209, 205, 145, 150, 206, 210,
      210, 151, 150, 140, 139, 211, 211, 207, 140, 134, 208, 212, 212, 135, 134,
      144, 143, 213, 213, 209, 144, 151, 210, 214, 151, 214, 215, 215, 152, 151,
      139, 138, 216, 139, 216, 217, 217, 211, 139, 135, 212, 218, 218, 136, 135,
      143, 142, 219, 143, 219, 220, 152, 215, 221, 152, 221, 127, 127, 126, 152,
      138, 137, 222, 222, 216, 138, 136, 218, 223, 136, 223, 109, 109, 108, 136,
      119, 118, 224, 119, 224, 219, 219, 142, 119, 127, 221, 225, 225, 128, 127,
      125, 124, 226, 125, 226, 222, 222, 137, 125, 109, 223, 227, 227, 110, 109,
      118, 117, 228, 228, 224, 118, 128, 225, 229, 229, 129, 128, 124, 123, 230,
      230, 226, 124, 110, 227, 231, 231, 111, 110, 117, 116, 232, 232, 228, 117,
      129, 229, 233, 233, 130, 129, 123, 122, 234, 234, 230, 123, 111, 231, 235,
      235, 112, 111, 116, 115, 236, 236, 232, 116, 130, 233, 237, 237, 131, 130,
      122, 121, 238, 122, 238, 239, 239, 234, 122, 112, 235, 240, 240, 113, 112,
      113, 241, 236, 113, 236, 115, 115, 114, 113, 238, 121, 120, 238, 120, 242,
      242, 243, 238, 243, 242, 131, 243, 131, 237, 243, 237, 244, 243, 245, 239,
      239, 238, 243, 239, 245, 246, 239, 246, 217, 239, 217, 216, 239, 216, 222,
      239, 222, 226, 239, 226, 230, 230, 234, 239, 214, 247, 248, 214, 248, 244,
      214, 244, 237, 214, 237, 233, 214, 233, 229, 214, 229, 225, 214, 225, 221,
      221, 215, 214, 212, 249, 240, 212, 240, 235, 212, 235, 231, 212, 231, 227,
      212, 227, 223, 223, 218, 212, 236, 250, 251, 236, 251, 220, 236, 220, 219,
      236, 219, 224, 236, 224, 228, 228, 232, 236, 197, 252, 253, 197, 253, 247,
      197, 247, 214, 197, 214, 210, 197, 210, 206, 206, 201, 197, 217, 246, 254,
      217, 254, 194, 217, 194, 193, 217, 193, 199, 217, 199, 203, 217, 203, 207,
      207, 211, 217, 195, 255, 256, 195, 257, 249, 195, 249, 212, 195, 212, 208,
      195, 208, 204, 204, 200, 195, 220, 258, 192, 220, 191, 198, 220, 198, 202,
      220, 202, 205, 220, 205, 209, 259, 251, 250, 259, 250, 240, 259, 240, 249,
      249, 257, 259, 244, 248, 260, 244, 260, 261, 244, 261, 254, 244, 254, 246,
      244, 246, 245, 171, 170, 262, 171, 262, 252, 171, 190, 185, 171, 185, 181,
      181, 177, 171, 187, 194, 263, 187, 263, 173, 187, 173, 172, 187, 172, 179,
      179, 183, 187, 169, 264, 265, 169, 265, 255, 169, 255, 188, 169, 188, 184,
      169, 184, 180, 180, 176, 169, 192, 266, 175, 192, 175, 174, 192, 174, 178,
      192, 178, 182, 182, 186, 192, 252, 267, 268, 268, 253, 252, 254, 261, 8,
      254, 8,   7,   254, 7,   269, 254, 269, 263, 263, 194, 254, 265, 270, 5,
      265, 4,   256, 256, 255, 265, 192, 258, 2,   192, 2,   1,   1,   266, 192,
      158, 271, 272, 158, 272, 170, 158, 170, 166, 166, 163, 158, 263, 269, 273,
      263, 273, 155, 263, 155, 156, 263, 156, 162, 263, 162, 167, 167, 173, 263,
      271, 160, 164, 271, 164, 168, 168, 266, 271, 153, 273, 274, 153, 274, 270,
      153, 270, 265, 153, 265, 264, 153, 264, 161, 161, 154, 153, 271, 266, 0,
      0,   272, 271, 274, 273, 275, 275, 276, 274, 260, 9,   8,   8,   261, 260,
      9,   260, 248, 9,   248, 247, 9,   247, 253, 9,   253, 268, 268, 10,  9,
      256, 4,   259, 3,   2,   258, 272, 0,   10,  272, 10,  268, 268, 267, 272,
      7,   276, 275, 274, 6,   5,   5,   270, 274, 0,   266, 1,   277, 24,  23};
  mesh.Merge();
  Manifold manifold(mesh);
  manifold = manifold.RefineToLength(1.0);
  EXPECT_NEAR(manifold.Volume(), 31.21, 0.01);
}
#endif

#ifdef MANIFOLD_DEBUG
TEST(Manifold, DISABLED_TriangulationNonManifold) {
  ManifoldParamGuard guard;
  ManifoldParams().intermediateChecks = false;
  Manifold m = ReadTestOBJ("openscad-nonmanifold-crash.obj");
  // m is not empty
  EXPECT_EQ(m.IsEmpty(), false);
  Manifold m2 = m + m.Translate({0, 0.6, 0});
  EXPECT_EQ(m2.IsEmpty(), false);
}
#endif

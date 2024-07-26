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

#include <algorithm>

#include "cross_section.h"
#include "samples.h"
#include "test.h"
#include "tri_dist.h"

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
TEST(Manifold, GetMesh) {
  Manifold manifold = Manifold::Sphere(1);
  Mesh mesh_out = manifold.GetMesh();
  Manifold manifold2(mesh_out);
  Mesh mesh_out2 = manifold2.GetMesh();
  Identical(mesh_out, mesh_out2);
}

TEST(Manifold, GetMeshGL) {
  Manifold manifold = Manifold::Sphere(1);
  Mesh mesh_out = manifold.GetMesh();
  MeshGL meshGL_out = manifold.GetMeshGL();
  ASSERT_EQ(meshGL_out.NumVert(), mesh_out.vertPos.size());
  ASSERT_EQ(meshGL_out.NumTri(), mesh_out.triVerts.size());
  for (size_t i = 0; i < meshGL_out.NumVert(); ++i) {
    for (const int j : {0, 1, 2}) {
      ASSERT_EQ(meshGL_out.vertProperties[3 * i + j], mesh_out.vertPos[i][j]);
    }
  }
  for (size_t i = 0; i < meshGL_out.NumTri(); ++i) {
    for (const int j : {0, 1, 2})
      ASSERT_EQ(meshGL_out.triVerts[3 * i + j], mesh_out.triVerts[i][j]);
  }
}

TEST(Manifold, Empty) {
  Mesh emptyMesh;
  Manifold empty(emptyMesh);

  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_EQ(empty.Status(), Manifold::Error::NoError);
}

TEST(Manifold, ValidInput) {
  std::vector<float> propTol = {0.1, 0.2};
  MeshGL tetGL = TetGL();
  Manifold tet(tetGL, propTol);
  EXPECT_FALSE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NoError);
}

TEST(Manifold, InvalidInput1) {
  Mesh in = Tet();
  in.vertPos[2][1] = NAN;
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NonFiniteVertex);
}

TEST(Manifold, InvalidInput2) {
  Mesh in = Tet();
  std::swap(in.triVerts[2][1], in.triVerts[2][2]);
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NotManifold);
}

TEST(Manifold, InvalidInput3) {
  Mesh in = Tet();
  for (glm::ivec3& tri : in.triVerts) {
    for (int i : {0, 1, 2}) {
      if (tri[i] == 2) tri[i] = -2;
    }
  }
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VertexOutOfBounds);
}

TEST(Manifold, InvalidInput4) {
  Mesh in = Tet();
  for (glm::ivec3& tri : in.triVerts) {
    for (int i : {0, 1, 2}) {
      if (tri[i] == 2) tri[i] = 4;
    }
  }
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VertexOutOfBounds);
}

TEST(Manifold, InvalidInput5) {
  MeshGL tetGL = TetGL();
  tetGL.mergeFromVert[tetGL.mergeFromVert.size() - 1] = 7;
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::MergeIndexOutOfBounds);
}

TEST(Manifold, InvalidInput7) {
  MeshGL tetGL = TetGL();
  tetGL.triVerts[tetGL.triVerts.size() - 1] = 7;
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VertexOutOfBounds);
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
  input.emplace_back(tet);
  auto cube = WithPositionColors(Manifold::Cube());
  manifoldList.emplace_back(cube);
  input.emplace_back(cube);
  auto sphere = WithPositionColors(Manifold::Sphere(1, 4));
  manifoldList.emplace_back(sphere);
  input.emplace_back(sphere);
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
  Manifold sphere = Manifold::Sphere(1.0f, 4 * n);
  EXPECT_EQ(sphere.NumTri(), n * n * 8);
}

TEST(Manifold, Cylinder) {
  int n = 10000;
  Manifold cylinder = Manifold::Cylinder(2, 2, 2, n);
  EXPECT_EQ(cylinder.NumTri(), 4 * n - 4);
}

TEST(Manifold, Normals) {
  Mesh cube = Manifold::Cube(glm::vec3(1), true).GetMesh();
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
  EXPECT_EQ(donut.Genus(), 1);
  auto prop = donut.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 12.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 48.0f);
}

TEST(Manifold, ExtrudeCone) {
  Polygons polys = SquareHole();
  Manifold donut = Manifold::Extrude(polys, 1.0f, 0, 0, glm::vec2(0.0f));
  EXPECT_EQ(donut.Genus(), 0);
  EXPECT_FLOAT_EQ(donut.GetProperties().volume, 4.0f);
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
    auto prop = vug.GetProperties();
    EXPECT_NEAR(prop.volume, 14.0f * glm::pi<float>(), 0.2f);
    EXPECT_NEAR(prop.surfaceArea, 30.0f * glm::pi<float>(), 0.2f);
  }
}

TEST(Manifold, Revolve2) {
  Polygons polys = SquareHole(2.0f);
  Manifold donutHole = Manifold::Revolve(polys, 48);
  EXPECT_EQ(donutHole.Genus(), 0);
  auto prop = donutHole.GetProperties();
  EXPECT_NEAR(prop.volume, 48.0f * glm::pi<float>(), 1.0f);
  EXPECT_NEAR(prop.surfaceArea, 96.0f * glm::pi<float>(), 1.0f);
}

TEST(Manifold, Revolve3) {
  CrossSection circle = CrossSection::Circle(1, 32);
  Manifold sphere = Manifold::Revolve(circle.ToPolygons(), 32);
  auto prop = sphere.GetProperties();
  EXPECT_NEAR(prop.volume, 4.0f / 3.0f * glm::pi<float>(), 0.1);
  EXPECT_NEAR(prop.surfaceArea, 4 * glm::pi<float>(), 0.15);
}

TEST(Manifold, PartialRevolveOnYAxis) {
  Polygons polys = SquareHole(2.0f);
  Polygons offsetPolys = SquareHole(10.0f);

  Manifold revolute;
  for (size_t i = 0; i < polys[0].size(); i++) {
    Polygons rotatedPolys = RotatePolygons(polys, i);
    revolute = Manifold::Revolve(rotatedPolys, 48, 180);
    EXPECT_EQ(revolute.Genus(), 1);
    auto prop = revolute.GetProperties();
    EXPECT_NEAR(prop.volume, 24.0f * glm::pi<float>(), 1.0f);
    EXPECT_NEAR(
        prop.surfaceArea,
        48.0f * glm::pi<float>() + 4.0f * 4.0f * 2.0f - 2.0f * 2.0f * 2.0f,
        1.0f);
  }
}

TEST(Manifold, PartialRevolveOffset) {
  Polygons polys = SquareHole(10.0f);

  Manifold revolute;
  for (size_t i = 0; i < polys[0].size(); i++) {
    Polygons rotatedPolys = RotatePolygons(polys, i);
    revolute = Manifold::Revolve(rotatedPolys, 48, 180);
    auto prop = revolute.GetProperties();
    EXPECT_EQ(revolute.Genus(), 1);
    EXPECT_NEAR(prop.surfaceArea, 777.0f, 1.0f);
    EXPECT_NEAR(prop.volume, 376.0f, 1.0f);
  }
}

TEST(Manifold, Warp) {
  CrossSection square = CrossSection::Square({1, 1});
  Manifold shape =
      Manifold::Extrude(square.ToPolygons(), 2, 10).Warp([](glm::vec3& v) {
        v.x += v.z * v.z;
      });
  auto propBefore = shape.GetProperties();

  Manifold simplified = Manifold::Compose({shape});
  auto propAfter = simplified.GetProperties();

  EXPECT_NEAR(propBefore.volume, propAfter.volume, 0.0001);
  EXPECT_NEAR(propBefore.surfaceArea, propAfter.surfaceArea, 0.0001);
  EXPECT_NEAR(propBefore.volume, 2, 0.0001);
}

TEST(Manifold, Warp2) {
  CrossSection circle =
      CrossSection::Circle(5, 20).Translate(glm::vec2(10.0, 10.0));

  Manifold shape =
      Manifold::Extrude(circle.ToPolygons(), 2, 10).Warp([](glm::vec3& v) {
        int nSegments = 10;
        double angleStep = 2.0 / 3.0 * glm::pi<float>() / nSegments;
        int zIndex = nSegments - 1 - std::round(v.z);
        double angle = zIndex * angleStep;
        v.z = v.y;
        v.y = v.x * sin(angle);
        v.x = v.x * cos(angle);
      });

  auto propBefore = shape.GetProperties();

  Manifold simplified = Manifold::Compose({shape});
  auto propAfter = simplified.GetProperties();

  EXPECT_NEAR(propBefore.volume, propAfter.volume, 0.0001);
  EXPECT_NEAR(propBefore.surfaceArea, propAfter.surfaceArea, 0.0001);
  EXPECT_NEAR(propBefore.volume, 321, 1);
}

TEST(Manifold, WarpBatch) {
  Manifold shape1 =
      Manifold::Cube({2, 3, 4}).Warp([](glm::vec3& v) { v.x += v.z * v.z; });
  auto prop1 = shape1.GetProperties();

  Manifold shape2 =
      Manifold::Cube({2, 3, 4}).WarpBatch([](VecView<glm::vec3> vecs) {
        for (glm::vec3& v : vecs) {
          v.x += v.z * v.z;
        }
      });
  auto prop2 = shape2.GetProperties();

  EXPECT_EQ(prop1.volume, prop2.volume);
  EXPECT_EQ(prop1.surfaceArea, prop2.surfaceArea);
}

TEST(Manifold, Project) {
  Mesh input;
  input.vertPos = {{0, 0, 0},
                   {-2, -0.7, -0.1},
                   {-2, -0.7, 0},
                   {-1.9, -0.7, -0.1},
                   {-1.9, -0.6901, -0.1},
                   {-1.9, -0.7, 0},
                   {-1.9, -0.6901, 0},
                   {-2, -1, 3},
                   {-1.9, -1, 3},
                   {-2, -1, 4},
                   {-1.9, -1, 4},
                   {-1.9, -0.6901, 3},
                   {-1.9, -0.6901, 4},
                   {-1.7, -0.6901, 3},
                   {-1.7, -0.6901, 3.2},
                   {-2, 0, -0.1},
                   {-2, 0, 0},
                   {-2, 0, 3},
                   {-2, 0, 4},
                   {-1.7, 0, 3},
                   {-1.7, 0, 3.2},
                   {-1, -0.6901, -0.1},
                   {-1, -0.6901, 0},
                   {-1, -0.6901, 3.2},
                   {-1, -0.6901, 4},
                   {-1, 0, -0.1},
                   {-1, 0, 0},
                   {-1, 0, 3.2},
                   {-1, 0, 4}};
  input.triVerts = {
      {1, 3, 2},    {1, 4, 3},    {2, 3, 5},    {5, 6, 2},    {3, 4, 6},
      {5, 3, 6},    {6, 4, 21},   {26, 22, 25}, {21, 25, 22}, {25, 15, 26},
      {26, 6, 22},  {21, 4, 25},  {21, 22, 6},  {16, 26, 15}, {16, 6, 26},
      {4, 15, 25},  {15, 1, 16},  {16, 2, 6},   {4, 1, 15},   {1, 2, 16},
      {12, 14, 23}, {12, 13, 14}, {12, 11, 13}, {18, 9, 12},  {11, 7, 17},
      {7, 9, 18},   {17, 7, 18},  {13, 11, 19}, {17, 18, 20}, {19, 11, 17},
      {19, 17, 20}, {14, 13, 20}, {18, 12, 24}, {20, 13, 19}, {20, 18, 27},
      {12, 10, 11}, {24, 12, 23}, {9, 10, 12},  {9, 8, 10},   {8, 11, 10},
      {8, 7, 11},   {8, 9, 7},    {14, 20, 27}, {24, 28, 18}, {27, 18, 28},
      {23, 14, 27}, {24, 23, 28}, {28, 23, 27}};
  Manifold in(input);
  CrossSection projected = in.Project();
  EXPECT_NEAR(projected.Area(), 0.72, 0.01);
}

/**
 * Testing more advanced Manifold operations.
 */

TEST(Manifold, Transform) {
  Manifold cube = Manifold::Cube({1, 2, 3});
  Manifold cube2 = cube;
  cube = cube.Rotate(30, 40, 50).Scale({6, 5, 4}).Translate({1, 2, 3});

  glm::mat3 rX(1.0f, 0.0f, 0.0f,          //
               0.0f, cosd(30), sind(30),  //
               0.0f, -sind(30), cosd(30));
  glm::mat3 rY(cosd(40), 0.0f, -sind(40),  //
               0.0f, 1.0f, 0.0f,           //
               sind(40), 0.0f, cosd(40));
  glm::mat3 rZ(cosd(50), sind(50), 0.0f,   //
               -sind(50), cosd(50), 0.0f,  //
               0.0f, 0.0f, 1.0f);
  glm::mat3 s = glm::mat3(1.0f);
  s[0][0] = 6;
  s[1][1] = 5;
  s[2][2] = 4;
  glm::mat4x3 transform = glm::mat4x3(s * rZ * rY * rX);
  transform[3] = glm::vec3(1, 2, 3);
  cube2 = cube2.Transform(transform);

  Identical(cube.GetMesh(), cube2.GetMesh());
}

TEST(Manifold, Slice) {
  Manifold cube = Manifold::Cube();
  CrossSection bottom = cube.Slice();
  CrossSection top = cube.Slice(1);
  EXPECT_EQ(bottom.Area(), 1);
  EXPECT_EQ(top.Area(), 0);
}

TEST(Manifold, MeshRelation) {
  MeshGL gyroidMeshGL = WithIndexColors(Gyroid());
  Manifold gyroid(gyroidMeshGL);

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorChannels = glm::ivec4(3, 4, 5, -1);
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
  const Mesh in = Csaszar();
  MeshGL inGL = WithIndexColors(in);
  Manifold csaszar(inGL);

  RelatedGL(csaszar, {inGL});
  csaszar = csaszar.RefineToLength(1);
  ExpectMeshes(csaszar, {{9019, 18038, 3}});
  RelatedGL(csaszar, {inGL});

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorChannels = glm::ivec4(3, 4, 5, -1);
  if (options.exportModels) ExportMesh("csaszar.glb", csaszar.GetMeshGL(), opt);
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
  EXPECT_EQ(cube.NumTri(), 12);
  EXPECT_EQ(cube.NumVert(), 8);
  EXPECT_EQ(cube.NumPropVert(), 24);

  auto prop = cube.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 1.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 6.0f);
}

TEST(Manifold, Merge) {
  MeshGL cubeSTL = CubeSTL();
  EXPECT_EQ(cubeSTL.NumTri(), 12);
  EXPECT_EQ(cubeSTL.NumVert(), 36);

  Manifold cubeBad(cubeSTL);
  EXPECT_TRUE(cubeBad.IsEmpty());
  EXPECT_EQ(cubeBad.Status(), Manifold::Error::NotManifold);

  EXPECT_TRUE(cubeSTL.Merge());
  CheckCube(cubeSTL);

  EXPECT_FALSE(cubeSTL.Merge());
  EXPECT_EQ(cubeSTL.mergeFromVert.size(), 28);
  cubeSTL.mergeFromVert.resize(14);
  cubeSTL.mergeToVert.resize(14);

  EXPECT_TRUE(cubeSTL.Merge());
  EXPECT_EQ(cubeSTL.mergeFromVert.size(), 28);
  CheckCube(cubeSTL);
}

TEST(Manifold, PinchedVert) {
  Mesh shape;
  shape.vertPos = {{0, 0, 0},         //
                   {1, 1, 0},         //
                   {1, -1, 0},        //
                   {-0.00001, 0, 0},  //
                   {-1, -1, -0},      //
                   {-1, 1, 0},        //
                   {0, 0, 2},         //
                   {0, 0, -2}};
  shape.triVerts = {{0, 2, 6},  //
                    {2, 1, 6},  //
                    {1, 0, 6},  //
                    {4, 3, 6},  //
                    {3, 5, 6},  //
                    {5, 4, 6},  //
                    {2, 0, 4},  //
                    {0, 3, 4},  //
                    {3, 0, 1},  //
                    {3, 1, 5},  //
                    {7, 2, 4},  //
                    {7, 4, 5},  //
                    {7, 5, 1},  //
                    {7, 1, 2}};
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
  inGL.faceID = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  const Manifold cube2(inGL);
  const MeshGL outGL = cube2.GetMeshGL();
  ASSERT_EQ(NumUnique(outGL.faceID), 12);
}

TEST(Manifold, MirrorUnion) {
  auto a = Manifold::Cube({5., 5., 5.}, true);
  auto b = a.Translate({2.5, 2.5, 2.5});
  auto result = a + b + b.Mirror({1, 1, 0});

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("manifold_mirror_union.glb", result.GetMesh(), {});
#endif

  auto vol_a = a.GetProperties().volume;
  EXPECT_FLOAT_EQ(vol_a * 2.75, result.GetProperties().volume);
  EXPECT_TRUE(a.Mirror(glm::vec3(0)).IsEmpty());
}

TEST(Manifold, MirrorUnion2) {
  auto a = Manifold::Cube();
  auto result = Manifold::Compose({a.Mirror({1, 0, 0})});
  EXPECT_TRUE(result.MatchesTriNormals());
}

TEST(Manifold, Invalid) {
  auto invalid = Manifold::Error::InvalidConstruction;
  auto circ = CrossSection::Circle(10.);
  auto empty_circ = CrossSection::Circle(-2.);
  auto empty_sq = CrossSection::Square(glm::vec2(0.0f));

  EXPECT_EQ(Manifold::Sphere(0).Status(), invalid);
  EXPECT_EQ(Manifold::Cylinder(0, 5).Status(), invalid);
  EXPECT_EQ(Manifold::Cylinder(2, -5).Status(), invalid);
  EXPECT_EQ(Manifold::Cube(glm::vec3(0.0f)).Status(), invalid);
  EXPECT_EQ(Manifold::Cube({-1, 1, 1}).Status(), invalid);
  EXPECT_EQ(Manifold::Extrude(circ.ToPolygons(), 0.).Status(), invalid);
  EXPECT_EQ(Manifold::Extrude(empty_circ.ToPolygons(), 10.).Status(), invalid);
  EXPECT_EQ(Manifold::Revolve(empty_sq.ToPolygons()).Status(), invalid);
}

TEST(Manifold, MultiCompose) {
  auto part = Manifold::Compose({Manifold::Cube({10, 10, 10})});
  auto finalAssembly =
      Manifold::Compose({part, part.Translate({0, 10, 0}),
                         part.Mirror({1, 0, 0}).Translate({10, 0, 0}),
                         part.Mirror({1, 0, 0}).Translate({10, 10, 0})});
  EXPECT_FLOAT_EQ(finalAssembly.GetProperties().volume, 4000);
}

TEST(Manifold, MergeDegenerates) {
  MeshGL cube = Manifold::Cube(glm::vec3(1), true).GetMeshGL();
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

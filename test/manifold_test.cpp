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
#include "test.h"

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

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
  for (int i = 0; i < meshGL_out.NumVert(); ++i) {
    for (const int j : {0, 1, 2}) {
      ASSERT_EQ(meshGL_out.vertProperties[3 * i + j], mesh_out.vertPos[i][j]);
    }
  }
  for (int i = 0; i < meshGL_out.NumTri(); ++i) {
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

/**
 * These tests check the various manifold constructors.
 */
TEST(Manifold, Sphere) {
  int n = 25;
  Manifold sphere = Manifold::Sphere(1.0f, 4 * n);
  EXPECT_EQ(sphere.NumTri(), n * n * 8);
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
  for (int i = 0; i < polys[0].size(); i++) {
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

TEST(Manifold, PartialRevolveOnYAxis) {
  Polygons polys = SquareHole(2.0f);
  Polygons offsetPolys = SquareHole(10.0f);

  Manifold revolute;
  for (int i = 0; i < polys[0].size(); i++) {
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
  for (int i = 0; i < polys[0].size(); i++) {
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
  Manifold shape = Manifold::Extrude(square, 2, 10).Warp([](glm::vec3& v) {
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

  Manifold shape = Manifold::Extrude(circle, 2, 10).Warp([](glm::vec3& v) {
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

TEST(Manifold, Smooth) {
  Manifold tet = Manifold::Tetrahedron();
  Manifold smooth = Manifold::Smooth(tet.GetMesh());
  smooth = smooth.Refine(100);
  ExpectMeshes(smooth, {{20002, 40000}});
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 17.38, 0.1);
  EXPECT_NEAR(prop.surfaceArea, 33.38, 0.1);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("smoothTet.glb", smooth.GetMesh(), {});
#endif
}

TEST(Manifold, SmoothSphere) {
  int n[5] = {4, 8, 16, 32, 64};
  float precision[5] = {0.03, 0.003, 0.003, 0.0005, 0.00006};
  for (int i = 0; i < 5; ++i) {
    Manifold sphere = Manifold::Sphere(1, n[i]);
    // Refine(odd) puts a center point in the triangle, which is the worst case.
    Manifold smoothed = Manifold::Smooth(sphere.GetMesh()).Refine(7);
    Mesh out = smoothed.GetMesh();
    auto bounds =
        std::minmax_element(out.vertPos.begin(), out.vertPos.end(),
                            [](const glm::vec3& a, const glm::vec3& b) {
                              return glm::dot(a, a) < glm::dot(b, b);
                            });
    float min = glm::length(*bounds.first);
    float max = glm::length(*bounds.second);
    EXPECT_NEAR(min, 1, precision[i]);
    EXPECT_NEAR(max, 1, precision[i]);
  }
}

TEST(Manifold, ManualSmooth) {
  // Unit Octahedron
  const Mesh oct = Manifold::Sphere(1, 4).GetMesh();
  Mesh smooth = Manifold::Smooth(oct).GetMesh();
  // Sharpen the edge from vert 4 to 5
  smooth.halfedgeTangent[6] = {0, 0, 0, 1};
  smooth.halfedgeTangent[22] = {0, 0, 0, 1};
  smooth.halfedgeTangent[16] = {0, 0, 0, 1};
  smooth.halfedgeTangent[18] = {0, 0, 0, 1};
  Manifold interp(smooth);
  interp = interp.Refine(100);

  ExpectMeshes(interp, {{40002, 80000}});
  auto prop = interp.GetProperties();
  EXPECT_NEAR(prop.volume, 3.53, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 11.39, 0.01);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    const Mesh out = interp.GetMesh();
    ExportOptions options;
    options.faceted = false;
    options.mat.roughness = 0.1;

    options.mat.vertColor.resize(interp.NumVert());
    const glm::vec4 red(1, 0, 0, 1);
    const glm::vec4 purple(1, 0, 1, 1);
    for (int tri = 0; tri < interp.NumTri(); ++tri) {
      for (int i : {0, 1, 2}) {
        const glm::vec3& uvw = {0.5, 0.5, 0.0};
        const float alpha = glm::min(uvw[0], glm::min(uvw[1], uvw[2]));
        options.mat.vertColor[out.triVerts[tri][i]] =
            glm::mix(purple, red, glm::smoothstep(0.0f, 0.2f, alpha));
      }
    }
    ExportMesh("sharpenedSphere.glb", out, options);
  }
#endif
}

TEST(Manifold, SmoothMirrored) {
  const Mesh tet = Manifold::Tetrahedron().GetMesh();
  Manifold smooth = Manifold::Smooth(tet);
  Manifold mirror = smooth.Scale({-1, 1, 2}).Refine(10);
  smooth = smooth.Refine(10).Scale({1, 1, 2});

  auto prop0 = smooth.GetProperties();
  auto prop1 = mirror.GetProperties();
  EXPECT_NEAR(prop0.volume, prop1.volume, 0.1);
  EXPECT_NEAR(prop0.surfaceArea, prop1.surfaceArea, 0.1);
}

TEST(Manifold, Csaszar) {
  Manifold csaszar = Manifold::Smooth(Csaszar());
  csaszar = csaszar.Refine(100);
  ExpectMeshes(csaszar, {{70000, 140000}});
  auto prop = csaszar.GetProperties();
  EXPECT_NEAR(prop.volume, 84699, 10);
  EXPECT_NEAR(prop.surfaceArea, 14796, 10);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    const Mesh out = csaszar.GetMesh();
    ExportOptions options;
    options.faceted = false;
    options.mat.roughness = 0.1;

    options.mat.vertColor.resize(csaszar.NumVert());
    const glm::vec4 blue(0, 0, 1, 1);
    const glm::vec4 yellow(1, 1, 0, 1);
    for (int tri = 0; tri < csaszar.NumTri(); ++tri) {
      for (int i : {0, 1, 2}) {
        const glm::vec3& uvw = {0.5, 0.5, 0.0};
        const float alpha = glm::min(uvw[0], glm::min(uvw[1], uvw[2]));
        options.mat.vertColor[out.triVerts[tri][i]] =
            glm::mix(yellow, blue, glm::smoothstep(0.0f, 0.2f, alpha));
      }
    }
    ExportMesh("smoothCsaszar.glb", out, options);
  }
#endif
}

/**
 * These tests verify the calculation of a manifold's geometric properties.
 */
TEST(Manifold, GetProperties) {
  Manifold cube = Manifold::Cube();
  auto prop = cube.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 1.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 6.0f);

  cube = cube.Scale(glm::vec3(-1.0f));
  prop = cube.GetProperties();
  EXPECT_FLOAT_EQ(prop.volume, 1.0f);
  EXPECT_FLOAT_EQ(prop.surfaceArea, 6.0f);
}

TEST(Manifold, Precision) {
  Manifold cube = Manifold::Cube();
  EXPECT_FLOAT_EQ(cube.Precision(), kTolerance);
  cube = cube.Scale({0.1, 1, 10});
  EXPECT_FLOAT_EQ(cube.Precision(), 10 * kTolerance);
  cube = cube.Translate({-100, -10, -1});
  EXPECT_FLOAT_EQ(cube.Precision(), 100 * kTolerance);
}

TEST(Manifold, Precision2) {
  Manifold cube = Manifold::Cube();
  cube = cube.Translate({-0.5, 0, 0}).Scale({2, 1, 1});
  EXPECT_FLOAT_EQ(cube.Precision(), 2 * kTolerance);
}

TEST(Manifold, Precision3) {
  Manifold cylinder = Manifold::Cylinder(1, 1, 1, 1000);
  const auto prop = cylinder.GetProperties();

  MeshGL mesh = cylinder.GetMeshGL();
  mesh.precision = 0.001;
  mesh.faceID.clear();
  Manifold cylinder2(mesh);

  const auto prop2 = cylinder2.GetProperties();
  EXPECT_NEAR(prop.volume, prop2.volume, 0.001);
  EXPECT_NEAR(prop.surfaceArea, prop2.surfaceArea, 0.001);
}

/**
 * Curvature is the inverse of the radius of curvature, and signed such that
 * positive is convex and negative is concave. There are two orthogonal
 * principal curvatures at any point on a manifold, with one maximum and the
 * other minimum. Gaussian curvature is their product, while mean
 * curvature is their sum. Here we check our discrete approximations calculated
 * at each vertex against the constant expected values of spheres of different
 * radii and at different mesh resolutions.
 */
TEST(Manifold, CalculateCurvature) {
  const float precision = 0.015;
  for (int n = 4; n < 100; n *= 2) {
    const int gaussianIdx = 3;
    const int meanIdx = 4;
    Manifold sphere = Manifold::Sphere(1, 64).CalculateCurvature(
        gaussianIdx - 3, meanIdx - 3);
    MeshGL sphereGL = sphere.GetMeshGL();
    ASSERT_EQ(sphereGL.numProp, 5);
    EXPECT_NEAR(GetMinProperty(sphereGL, meanIdx), 2, 2 * precision);
    EXPECT_NEAR(GetMaxProperty(sphereGL, meanIdx), 2, 2 * precision);
    EXPECT_NEAR(GetMinProperty(sphereGL, gaussianIdx), 1, precision);
    EXPECT_NEAR(GetMaxProperty(sphereGL, gaussianIdx), 1, precision);

    sphere = sphere.Scale(glm::vec3(2.0f))
                 .CalculateCurvature(gaussianIdx - 3, meanIdx - 3);
    sphereGL = sphere.GetMeshGL();
    ASSERT_EQ(sphereGL.numProp, 5);
    EXPECT_NEAR(GetMinProperty(sphereGL, meanIdx), 1, precision);
    EXPECT_NEAR(GetMaxProperty(sphereGL, meanIdx), 1, precision);
    EXPECT_NEAR(GetMinProperty(sphereGL, gaussianIdx), 0.25, 0.25 * precision);
    EXPECT_NEAR(GetMaxProperty(sphereGL, gaussianIdx), 0.25, 0.25 * precision);
  }
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

TEST(Manifold, MeshRelation) {
  Mesh gyroidMesh = Gyroid();
  MeshGL gyroidMeshGL = WithIndexColors(gyroidMesh);
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
  csaszar.Refine(4);
  RelatedGL(csaszar, {inGL});
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
  EXPECT_EQ(Manifold::Extrude(circ, 0.).Status(), invalid);
  EXPECT_EQ(Manifold::Extrude(empty_circ, 10.).Status(), invalid);
  EXPECT_EQ(Manifold::Revolve(empty_sq).Status(), invalid);
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

TEST(Manifold, TictacHull) {
  const float tictacRad = 100;
  const float tictacHeight = 500;
  const int tictacSeg = 1000;
  const float tictacMid = tictacHeight - 2 * tictacRad;
  const auto sphere = Manifold::Sphere(tictacRad, tictacSeg);
  const std::vector<Manifold> spheres{sphere,
                                      sphere.Translate({0, 0, tictacMid})};
  const auto tictac = Manifold::Hull(spheres);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportMesh("tictac_hull.glb", tictac.GetMesh(), {});
  }
#endif

  EXPECT_EQ(sphere.NumVert() + tictacSeg, tictac.NumVert());
}

TEST(Manifold, HollowHull) {
  auto sphere = Manifold::Sphere(100, 360);
  auto hollow = sphere - sphere.Scale({0.8, 0.8, 0.8});
  const float sphere_vol = sphere.GetProperties().volume;
  EXPECT_FLOAT_EQ(hollow.Hull().GetProperties().volume, sphere_vol);
}

TEST(Manifold, CubeHull) {
  std::vector<glm::vec3> cubePts = {
      {0, 0, 0},       {1, 0, 0},   {0, 1, 0},      {0, 0, 1},  // corners
      {1, 1, 0},       {0, 1, 1},   {1, 0, 1},      {1, 1, 1},  // corners
      {0.5, 0.5, 0.5}, {0.5, 0, 0}, {0.5, 0.7, 0.2}  // internal points
  };
  auto cube = Manifold::Hull(cubePts);
  EXPECT_FLOAT_EQ(cube.GetProperties().volume, 1);
}

TEST(Manifold, EmptyHull) {
  const std::vector<glm::vec3> tooFew{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  EXPECT_TRUE(Manifold::Hull(tooFew).IsEmpty());

  const std::vector<glm::vec3> coplanar{
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
  EXPECT_TRUE(Manifold::Hull(coplanar).IsEmpty());
}

TEST(Manifold, InterpolatedNormals) {
  MeshGL a;
  a.numProp = 8;
  a.vertProperties = {
      // 0
      -409.0570983886719, -300, -198.83624267578125, 0, -1, 0,
      590.9429321289062, 301.1637268066406,
      // 1
      -1000, -300, 500, 0, -1, 0, 0, 1000,
      // 2
      -1000, -300, -500, 0, -1, 0, 0, 0,
      // 3
      -1000, -300, -500, -1, 0, 0, 600, 0,
      // 4
      -1000, -300, 500, -1, 0, 0, 600, 1000,
      // 5
      -1000, 300, -500, -1, 0, 0, 0, 0,
      // 6
      7.179656982421875, -300, -330.03717041015625, 0, -0.9999999403953552, 0,
      1007.1796264648438, 169.9628448486328,
      // 7
      1000, 300, 500, 0, 0, 1, 2000, 600,
      // 8
      403.5837097167969, 300, 500, 0, 0, 1, 1403.583740234375, 600,
      // 9
      564.2904052734375, 21.64801025390625, 500, 0, 0, 1, 1564.29052734375,
      321.64801025390625,
      // 10
      1000, -300, -500, 0, 0, -1, 2000, 600,
      // 11
      -1000, -300, -500, 0, 0, -1, 0, 600,
      // 12
      -1000, 300, -500, 0, 0, -1, 0, 0,
      // 13
      1000, 300, 500, 0, 1, 0, 0, 1000,
      // 14
      1000, 300, -500, 0, 1, 0, 0, 0,
      // 15
      724.5271606445312, 300, 398.83624267578125, 0, 1, 0, 275.47283935546875,
      898.8362426757812,
      // 16
      -115.35255432128906, -300, 500, 0, -1.0000001192092896, 0,
      884.6475219726562, 1000.0001220703125,
      // 17
      -384.7195129394531, 166.55722045898438, 500, 0, 0, 1, 615.280517578125,
      466.5572509765625,
      // 18
      -1000, -300, 500, 0, 0, 1, 0, 0,
      // 19
      -161.6136932373047, -219.87335205078125, 500, 0, 0, 1, 838.3862915039062,
      80.12664794921875,
      // 20
      1000, -300, 500, 0, 0, 1, 2000, 0,
      // 21
      -115.35255432128906, -300, 500, 0, 0, 1, 884.6475219726562, 0,
      // 22
      1000, 300, 500, 1, 0, 0, 600, 1000,
      // 23
      1000, -300, 500, 1, 0, 0, 0, 1000,
      // 24
      1000, 300, -500, 1, 0, 0, 600, 0,
      // 25
      566.6257934570312, 300, 23.1280517578125, 0, 1, 0, 433.3742370605469,
      523.1281127929688,
      // 26
      411.5867004394531, -66.51548767089844, -500, 0, 0, -1, 1411.586669921875,
      366.5155029296875,
      // 27
      375.7498779296875, -4.444300651550293, -500, 0, 0, -1, 1375.7498779296875,
      304.4443054199219,
      // 28
      346.7673034667969, 300, -500, 0, 1, 0, 653.2326049804688, 0,
      // 29
      -153.58984375, 300, -388.552490234375, 0, 1, 0, 1153.58984375,
      111.447509765625,
      // 30
      199.9788818359375, 300, -500, 0, 1, 0, 800.0211791992188, 0,
      // 31
      -1000, 300, -500, 0, 1, 0, 2000, 0,
      // 32
      -153.58987426757812, 300, 44.22247314453125, 0, 1, 0, 1153.58984375,
      544.2224731445312,
      // 33
      199.9788818359375, 300, -500, 0, 0, -1, 1199.9791259765625, 0,
      // 34
      521.6780395507812, -2.9542479515075684, -500, 0, 0, -1, 1521.677978515625,
      302.9542541503906,
      // 35
      346.7673034667969, 300, -500, 0, 0, -1, 1346.767333984375, 0,
      // 36
      1000, 300, -500, 0, 0, -1, 2000, 0,
      // 37
      -1000, 300, 500, -1, 0, 0, 0, 1000,
      // 38
      -1000, 300, 500, 0, 0, 1, 0, 600,
      // 39
      -1000, 300, 500, 0, 1, 0, 2000, 1000,
      // 40
      -153.58985900878906, 300, 500, 0, 0, 1, 846.4102172851562, 600,
      // 41
      88.46627807617188, -253.06915283203125, 500, 0, 0, 1, 1088.4664306640625,
      46.93084716796875,
      // 42
      -153.58985900878906, 300, 500, 0, 1, 0, 1153.58984375, 1000,
      // 43
      7.1796698570251465, -300, 500, 0, 0, 1, 1007.1797485351562, 0,
      // 44
      1000, -300, -500, 0, -1, 0, 2000, 0,
      // 45
      1000, -300, 500, 0, -1, 0, 2000, 1000,
      // 46
      7.1796698570251465, -300, 500, 0, -1, 0, 1007.1796264648438, 1000,
      // 47
      403.5837097167969, 300, 500, 0, 1, 0, 596.4163208007812, 1000,
      // 48
      1000, -300, -500, 1, 0, 0, 0, 0,
      // 49
      492.3005676269531, -19.915321350097656, -500, 0, 0, -1, 1492.300537109375,
      319.91534423828125,
      // 50
      411.5867004394531, -66.51548767089844, -500, 0.5, -0.8660253882408142, 0,
      880.5439453125, 0,
      // 51
      7.179656982421875, -300, -330.03717041015625, 0.5, -0.8660253286361694, 0,
      383.6058654785156, 0,
      // 52
      492.3005676269531, -19.915321350097656, -500, 0.5, -0.8660253882408142, 0,
      968.1235961914062, 31.876384735107422,
      // 53
      7.1796698570251465, -300, 500, 0.5, -0.8660253286361694, 0,
      99.71644592285156, 779.979736328125,
      // 54
      88.46627807617188, -253.06915283203125, 500, 0.5, -0.8660253882408142, 0,
      187.91758728027344, 812.0823974609375,
      // 55
      -153.58985900878906, 300, 500, -0.4999999701976776, 0.8660253286361694, 0,
      749.2095947265625, 834.9661865234375,
      // 56
      -384.7195129394531, 166.55722045898438, 500, -0.5, 0.8660253286361694, 0,
      1000, 743.6859741210938,
      // 57
      -153.58987426757812, 300, 44.22247314453125, -0.5, 0.8660253882408142, 0,
      593.3245239257812, 406.6754455566406,
      // 58
      564.2904052734375, 21.64801025390625, 500, 0.5, -0.866025447845459, 0,
      704.217041015625, 1000.0000610351562,
      // 59
      -604.9979248046875, 39.37942886352539, -198.83624267578125, -0.5,
      0.8660253882408142, 0, 1000, 0,
      // 60
      199.9788818359375, 300, -500, -0.29619812965393066, -0.1710100769996643,
      -0.9396926164627075, 880.5438842773438, 176.7843475341797,
      // 61
      -153.58984375, 300, -388.552490234375, -0.29619812965393066,
      -0.1710100769996643, -0.9396926164627075, 554.6932373046875, 0,
      // 62
      375.7498779296875, -4.444300651550293, -500, -0.2961980998516083,
      -0.1710100769996643, -0.9396925568580627, 880.5438842773438,
      528.3263549804688,
      // 63
      566.6257934570312, 300, 23.1280517578125, 0.8137975931167603,
      0.46984630823135376, -0.3420201539993286, 239.89218139648438, 600.1796875,
      // 64
      346.7673034667969, 300, -500, 0.813797652721405, 0.46984627842903137,
      -0.3420201539993286, 349.8214111328125, 43.478458404541016,
      // 65
      521.6780395507812, -2.9542479515075684, -500, 0.813797652721405,
      0.46984627842903137, -0.3420201539993286, 0, 43.478458404541016,
      // 66
      804.9979248046875, 160.62057495117188, 398.83624267578125, 0.5,
      -0.8660253882408142, 0, 1000, 1000,
      // 67
      521.6780395507812, -2.9542479515075684, -500, 0.5, -0.8660253882408142, 0,
      1000, 43.47837829589844,
      // 68
      -153.58984375, 300, -388.552490234375, -0.5, 0.8660253882408142, 0,
      445.3067626953125, 0,
      // 69
      -604.9979248046875, 39.37942886352539, -198.83624267578125,
      -0.29619812965393066, -0.1710100769996643, -0.9396926164627075, 0, 0,
      // 70
      804.9979248046875, 160.62057495117188, 398.83624267578125,
      0.813797652721405, 0.46984630823135376, -0.3420201539993286, 0, 1000,
      // 71
      -161.6136932373047, -219.87335205078125, 500, -0.8137977123260498,
      -0.46984630823135376, 0.3420201539993286, 446.21160888671875,
      743.68603515625,
      // 72
      -604.9979248046875, 39.37942886352539, -198.83624267578125,
      -0.813797652721405, -0.46984630823135376, 0.3420201539993286, 0, 0,
      // 73
      -384.7195129394531, 166.55722045898438, 500, -0.8137975931167603,
      -0.46984630823135376, 0.3420201539993286, 0, 743.6859741210938,
      // 74
      -115.35255432128906, -300, 500, -0.813797652721405, -0.46984633803367615,
      0.3420201539993286, 538.73388671875, 743.68603515625,
      // 75
      -409.0570983886719, -300, -198.83624267578125, -0.813797652721405,
      -0.46984630823135376, 0.3420201539993286, 391.8816223144531, 0,
      // 76
      7.179656982421875, -300, -330.03717041015625, -0.29619812965393066,
      -0.1710100769996643, -0.9396926164627075, 383.6058654785156, 600,
      // 77
      564.2904052734375, 21.64801025390625, 500, 0.2961980998516083,
      0.1710100769996643, 0.939692497253418, 704.2169189453125,
      -0.000030517578125,
      // 78
      403.5837097167969, 300, 500, 0.29619812965393066, 0.1710100919008255,
      0.9396926164627075, 704.2169799804688, 321.4132385253906,
      // 79
      724.5271606445312, 300, 398.83624267578125, 0.29619812965393066,
      0.1710100769996643, 0.9396925568580627, 1000, 160.94149780273438,
      // 80
      804.9979248046875, 160.62057495117188, 398.83624267578125,
      0.29619812965393066, 0.1710100769996643, 0.9396926164627075, 1000, 0,
      // 81
      -409.0570983886719, -300, -198.83624267578125, -0.29619812965393066,
      -0.1710100769996643, -0.9396926164627075, 0, 391.88165283203125,
      // 82
      724.5271606445312, 300, 398.83624267578125, 0.813797652721405,
      0.46984630823135376, -0.342020183801651, 160.94149780273438,
      1000.0000610351562,
      // 83
      411.5867004394531, -66.51548767089844, -500, -0.29619815945625305,
      -0.1710100769996643, -0.9396926164627075, 880.5440063476562,
      600.0000610351562};
  a.triVerts = {// 0
                0, 1, 2,
                // 1
                3, 4, 5,
                // 2
                6, 0, 2,
                // 3
                7, 8, 9,
                // 4
                10, 11, 12,
                // 5
                13, 14, 15,
                // 6
                0, 16, 1,
                // 7
                17, 18, 19,
                // 8
                9, 20, 7,
                // 9
                18, 21, 19,
                // 10
                22, 23, 24,
                // 11
                14, 25, 15,
                // 12
                26, 12, 27,
                // 13
                14, 28, 25,
                // 14
                29, 30, 31,
                // 15
                29, 31, 32,
                // 16
                12, 33, 27,
                // 17
                34, 35, 36,
                // 18
                5, 4, 37,
                // 19
                17, 38, 18,
                // 20
                31, 39, 32,
                // 21
                40, 38, 17,
                // 22
                9, 41, 20,
                // 23
                39, 42, 32,
                // 24
                41, 43, 20,
                // 25
                6, 2, 44,
                // 26
                6, 45, 46,
                // 27
                26, 10, 12,
                // 28
                47, 13, 15,
                // 29
                48, 24, 23,
                // 30
                6, 44, 45,
                // 31
                26, 49, 10,
                // 32
                49, 34, 10,
                // 33
                34, 36, 10,
                // 34
                50, 51, 52,
                // 35
                51, 53, 54,
                // 36
                51, 54, 52,
                // 37
                55, 56, 57,
                // 38
                52, 54, 58,
                // 39
                59, 57, 56,
                // 40
                60, 61, 62,
                // 41
                63, 64, 65,
                // 42
                52, 66, 67,
                // 43
                59, 68, 57,
                // 44
                69, 62, 61,
                // 45
                65, 70, 63,
                // 46
                71, 72, 73,
                // 47
                52, 58, 66,
                // 48
                74, 72, 71,
                // 49
                74, 75, 72,
                // 50
                62, 69, 76,
                // 51
                77, 78, 79,
                // 52
                79, 80, 77,
                // 53
                69, 81, 76,
                // 54
                63, 70, 82,
                // 55
                76, 83, 62};
  a.mergeFromVert = {3,  4,  11, 12, 13, 18, 21, 22, 23, 24, 31, 33, 35, 36,
                     38, 39, 42, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55,
                     56, 57, 58, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71,
                     72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83};
  a.mergeToVert = {2,  1,  2,  5,  7,  1,  16, 7,  20, 14, 5,  30, 28, 14,
                   37, 37, 40, 10, 20, 43, 8,  10, 26, 6,  49, 43, 41, 40,
                   17, 32, 9,  30, 29, 27, 25, 28, 34, 34, 29, 59, 66, 19,
                   59, 17, 16, 0,  6,  9,  8,  15, 66, 0,  15, 26};

  MeshGL b;
  b.numProp = 8;
  b.vertProperties = {// 0
                      -1700, -600, -1000, -1, 0, 0, 1200, 0,
                      // 1
                      -1700, -600, 1000, -1, 0, 0, 1200, 2000,
                      // 2
                      -1700, 600, -1000, -1, 0, 0, 0, 0,
                      // 3
                      -1700, -600, -1000, 0, -1, 0, 0, 0,
                      // 4
                      300, -600, -1000, 0, -1, 0, 2000, 0,
                      // 5
                      -1700, -600, 1000, 0, -1, 0, 0, 2000,
                      // 6
                      -1700, -600, -1000, 0, 0, -1, 0, 1200,
                      // 7
                      -1700, 600, -1000, 0, 0, -1, 0, 0,
                      // 8
                      300, -600, -1000, 0, 0, -1, 2000, 1200,
                      // 9
                      -1700, -600, 1000, 0, 0, 1, 0, 0,
                      // 10
                      300, -600, 1000, 0, 0, 1, 2000, 0,
                      // 11
                      -1700, 600, 1000, 0, 0, 1, 0, 1200,
                      // 12
                      -1700, 600, 1000, -1, 0, 0, 0, 2000,
                      // 13
                      -1700, 600, -1000, 0, 1, 0, 2000, 0,
                      // 14
                      -1700, 600, 1000, 0, 1, 0, 2000, 2000,
                      // 15
                      300, 600, 1000, 0, 1, 0, 0, 2000,
                      // 16
                      300, -600, -1000, 1, 0, 0, 0, 0,
                      // 17
                      300, 600, -1000, 1, 0, 0, 1200, 0,
                      // 18
                      300, -600, 1000, 1, 0, 0, 0, 2000,
                      // 19
                      300, -600, 1000, 0, -1, 0, 2000, 2000,
                      // 20
                      300, 600, -1000, 0, 0, -1, 2000, 0,
                      // 21
                      300, 600, -1000, 0, 1, 0, 0, 0,
                      // 22
                      300, 600, 1000, 0, 0, 1, 2000, 1200,
                      // 23
                      300, 600, 1000, 1, 0, 0, 1200, 2000};
  b.triVerts = {// 0
                0, 1, 2,
                // 1
                3, 4, 5,
                // 2
                6, 7, 8,
                // 3
                9, 10, 11,
                // 4
                1, 12, 2,
                // 5
                13, 14, 15,
                // 6
                16, 17, 18,
                // 7
                4, 19, 5,
                // 8
                7, 20, 8,
                // 9
                21, 13, 15,
                // 10
                10, 22, 11,
                // 11
                17, 23, 18};
  b.mergeFromVert = {3, 5, 6, 7, 8, 9, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23};
  b.mergeToVert = {0, 1, 0, 2, 4, 1, 11, 2, 11, 4, 10, 10, 17, 17, 15, 15};

  a.runOriginalID = {Manifold::ReserveIDs(1)};
  b.runOriginalID = {Manifold::ReserveIDs(1)};

  Manifold aManifold(a);
  Manifold bManifold(b);

  auto aMinusB = aManifold - bManifold;

  std::vector<MeshGL> meshList;
  meshList.emplace_back(a);
  meshList.emplace_back(b);

  RelatedGL(aMinusB, meshList, true, true);
}
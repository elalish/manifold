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

#include "manifold.h"
#include "meshIO.h"
#include "test.h"

namespace {

using namespace manifold;

Mesh Csaszar() {
  Mesh csaszar;
  csaszar.vertPos = {{-20, -20, -10},  //
                     {-20, 20, -15},   //
                     {-5, -8, 8},      //
                     {0, 0, 30},       //
                     {5, 8, 8},        //
                     {20, -20, -15},   //
                     {20, 20, -10}};
  csaszar.triVerts = {{1, 3, 6},  //
                      {1, 6, 5},  //
                      {2, 5, 6},  //
                      {0, 2, 6},  //
                      {0, 6, 4},  //
                      {3, 4, 6},  //
                      {1, 2, 3},  //
                      {1, 4, 2},  //
                      {1, 0, 4},  //
                      {1, 5, 0},  //
                      {3, 5, 4},  //
                      {0, 5, 3},  //
                      {0, 3, 2},  //
                      {2, 4, 5}};
  return csaszar;
}

void Identical(const Mesh& mesh1, const Mesh& mesh2) {
  ASSERT_EQ(mesh1.vertPos.size(), mesh2.vertPos.size());
  for (int i = 0; i < mesh1.vertPos.size(); ++i)
    for (int j : {0, 1, 2})
      ASSERT_NEAR(mesh1.vertPos[i][j], mesh2.vertPos[i][j], 0.0001);

  ASSERT_EQ(mesh1.triVerts.size(), mesh2.triVerts.size());
  for (int i = 0; i < mesh1.triVerts.size(); ++i)
    ASSERT_EQ(mesh1.triVerts[i], mesh2.triVerts[i]);
}

void Related(const Manifold& out, const std::vector<Mesh>& input,
             const std::map<int, int>& meshID2idx) {
  Mesh output = out.GetMesh();
  MeshRelation relation = out.GetMeshRelation();
  for (int tri = 0; tri < out.NumTri(); ++tri) {
    int meshID = relation.triBary[tri].meshID;
    int meshIdx = meshID2idx.at(meshID);
    ASSERT_LT(meshIdx, input.size());
    const Mesh& inMesh = input[meshIdx];
    int inTri = relation.triBary[tri].tri;
    ASSERT_LT(inTri, inMesh.triVerts.size());
    glm::mat3 inTriangle = {inMesh.vertPos[inMesh.triVerts[inTri][0]],
                            inMesh.vertPos[inMesh.triVerts[inTri][1]],
                            inMesh.vertPos[inMesh.triVerts[inTri][2]]};
    for (int j : {0, 1, 2}) {
      glm::vec3 vPos = output.vertPos[output.triVerts[tri][j]];
      glm::vec3 uvw = relation.UVW(tri, j);
      ASSERT_NEAR(uvw[0] + uvw[1] + uvw[2], 1, 0.0001);
      glm::vec3 vRelation = inTriangle * uvw;
      for (int k : {0, 1, 2})
        ASSERT_NEAR(vPos[k], vRelation[k], 5 * out.Precision());
    }
  }
}

void RelatedOp(const Manifold& inP, const Manifold& inQ, const Manifold& outR) {
  std::vector<Mesh> input;
  std::map<int, int> meshID2idx;

  std::vector<int> meshIDs = inP.GetMeshIDs();
  EXPECT_EQ(meshIDs.size(), 1);
  meshID2idx[meshIDs[0]] = input.size();
  input.push_back(inP.GetMesh());

  meshIDs = inQ.GetMeshIDs();
  EXPECT_EQ(meshIDs.size(), 1);
  meshID2idx[meshIDs[0]] = input.size();
  input.push_back(inQ.GetMesh());

  Related(outR, input, meshID2idx);
}

void ExpectMeshes(const Manifold& manifold,
                  const std::vector<std::pair<int, int>>& numVertTri) {
  EXPECT_TRUE(manifold.IsManifold());
  EXPECT_TRUE(manifold.MatchesTriNormals());
  std::vector<Manifold> manifolds = manifold.Decompose();
  ASSERT_EQ(manifolds.size(), numVertTri.size());
  std::sort(manifolds.begin(), manifolds.end(),
            [](const Manifold& a, const Manifold& b) {
              return a.NumVert() != b.NumVert() ? a.NumVert() > b.NumVert()
                                                : a.NumTri() > b.NumTri();
            });
  for (int i = 0; i < manifolds.size(); ++i) {
    EXPECT_TRUE(manifolds[i].IsManifold());
    EXPECT_EQ(manifolds[i].NumVert(), numVertTri[i].first);
    EXPECT_EQ(manifolds[i].NumTri(), numVertTri[i].second);
    const Mesh mesh = manifolds[i].GetMesh();
    for (const glm::vec3& normal : mesh.vertNormal) {
      ASSERT_NEAR(glm::length(normal), 1, 0.0001);
    }
  }
}

void CheckStrictly(const Manifold& manifold) {
  EXPECT_TRUE(manifold.IsManifold());
  EXPECT_TRUE(manifold.MatchesTriNormals());
  EXPECT_EQ(manifold.NumDegenerateTris(), 0);
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
TEST(Manifold, GetMesh) {
  Manifold manifold = Manifold::Sphere(1);
  Mesh mesh_out = manifold.GetMesh();
  Manifold manifold2(mesh_out);
  Mesh mesh_out2 = manifold2.GetMesh();
  Identical(mesh_out, mesh_out2);
}

// There is still some non-determinism, especially in parallel. Likely in the
// edge collapse step. Not a huge problem, but best to fix for user's sanity.
TEST(Manifold, DISABLED_Determinism) {
  Manifold manifold(ImportMesh("data/gyroidpuzzle.ply"));
  EXPECT_TRUE(manifold.IsManifold());

  Manifold manifold1 = manifold;
  manifold1.Translate(glm::vec3(5.0f));
  int num_overlaps = manifold.NumOverlaps(manifold1);
  ASSERT_EQ(num_overlaps, 229611);

  Mesh mesh_out = manifold.GetMesh();
  Manifold manifold2(mesh_out);
  Mesh mesh_out2 = manifold2.GetMesh();
  // Identical(mesh_out, mesh_out2);
}

/**
 * ExpectMeshes performs a decomposition, so this test ensures that compose and
 * decompose are inverse operations.
 */
TEST(Manifold, Decompose) {
  std::vector<Manifold> manifoldList;
  manifoldList.push_back(Manifold::Tetrahedron());
  manifoldList.push_back(Manifold::Cube());
  manifoldList.push_back(Manifold::Sphere(1, 4));
  Manifold manifolds = Manifold::Compose(manifoldList);

  ExpectMeshes(manifolds, {{8, 12}, {6, 8}, {4, 4}});

  std::vector<Mesh> input;
  std::map<int, int> meshID2idx;

  for (const Manifold& manifold : manifoldList) {
    std::vector<int> meshIDs = manifold.GetMeshIDs();
    EXPECT_EQ(meshIDs.size(), 1);
    meshID2idx[meshIDs[0]] = input.size();
    input.push_back(manifold.GetMesh());
  }

  Related(manifolds, input, meshID2idx);
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
  Manifold smooth = Manifold::Smooth(tet.GetMesh());
  smooth.Refine(100);
  ExpectMeshes(smooth, {{20002, 40000}});
  auto prop = smooth.GetProperties();
  EXPECT_NEAR(prop.volume, 17.38, 0.1);
  EXPECT_NEAR(prop.surfaceArea, 33.38, 0.1);

  if (options.exportModels) ExportMesh("smoothTet.glb", smooth.GetMesh(), {});
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
  interp.Refine(100);

  ExpectMeshes(interp, {{40002, 80000}});
  auto prop = interp.GetProperties();
  EXPECT_NEAR(prop.volume, 3.53, 0.01);
  EXPECT_NEAR(prop.surfaceArea, 11.39, 0.01);

  if (options.exportModels) {
    const Mesh out = interp.GetMesh();
    ExportOptions options;
    options.faceted = false;
    options.mat.roughness = 0.1;

    options.mat.vertColor.resize(interp.NumVert());
    MeshRelation rel = interp.GetMeshRelation();
    const glm::vec4 red(1, 0, 0, 1);
    const glm::vec4 purple(1, 0, 1, 1);
    for (int tri = 0; tri < interp.NumTri(); ++tri) {
      for (int i : {0, 1, 2}) {
        const glm::vec3& uvw = rel.UVW(tri, i);
        const float alpha = glm::min(uvw[0], glm::min(uvw[1], uvw[2]));
        options.mat.vertColor[out.triVerts[tri][i]] =
            glm::mix(purple, red, glm::smoothstep(0.0f, 0.2f, alpha));
      }
    }
    ExportMesh("sharpenedSphere.glb", out, options);
  }
}

TEST(Manifold, Csaszar) {
  Manifold csaszar = Manifold::Smooth(Csaszar());
  csaszar.Refine(100);
  ExpectMeshes(csaszar, {{70000, 140000}});
  auto prop = csaszar.GetProperties();
  EXPECT_NEAR(prop.volume, 84699, 10);
  EXPECT_NEAR(prop.surfaceArea, 14796, 10);

  if (options.exportModels) {
    const Mesh out = csaszar.GetMesh();
    ExportOptions options;
    options.faceted = false;
    options.mat.roughness = 0.1;

    options.mat.vertColor.resize(csaszar.NumVert());
    MeshRelation rel = csaszar.GetMeshRelation();
    const glm::vec4 blue(0, 0, 1, 1);
    const glm::vec4 yellow(1, 1, 0, 1);
    for (int tri = 0; tri < csaszar.NumTri(); ++tri) {
      for (int i : {0, 1, 2}) {
        const glm::vec3& uvw = rel.barycentric[rel.triBary[tri].vertBary[i]];
        const float alpha = glm::min(uvw[0], glm::min(uvw[1], uvw[2]));
        options.mat.vertColor[out.triVerts[tri][i]] =
            glm::mix(yellow, blue, glm::smoothstep(0.0f, 0.2f, alpha));
      }
    }
    ExportMesh("smoothCsaszar.glb", out, options);
  }
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
 * Curvature is the inverse of the radius of curvature, and signed such that
 * positive is convex and negative is concave. There are two orthogonal
 * principal curvatures at any point on a manifold, with one maximum and the
 * other minimum. Gaussian curvature is their product, while mean
 * curvature is their sum. Here we check our discrete appoximations calculated
 * at each vertex against the constant expected values of spheres of different
 * radii and at different mesh resolutions.
 */
TEST(Manifold, GetCurvature) {
  const float precision = 0.015;
  for (int n = 4; n < 100; n *= 2) {
    Manifold sphere = Manifold::Sphere(1, 64);
    Curvature curvature = sphere.GetCurvature();
    EXPECT_NEAR(curvature.minMeanCurvature, 2, 2 * precision);
    EXPECT_NEAR(curvature.maxMeanCurvature, 2, 2 * precision);
    EXPECT_NEAR(curvature.minGaussianCurvature, 1, precision);
    EXPECT_NEAR(curvature.maxGaussianCurvature, 1, precision);

    sphere.Scale(glm::vec3(2.0f));
    curvature = sphere.GetCurvature();
    EXPECT_NEAR(curvature.minMeanCurvature, 1, precision);
    EXPECT_NEAR(curvature.maxMeanCurvature, 1, precision);
    EXPECT_NEAR(curvature.minGaussianCurvature, 0.25, 0.25 * precision);
    EXPECT_NEAR(curvature.maxGaussianCurvature, 0.25, 0.25 * precision);
  }
}

/**
 * Testing more advanced Manifold operations.
 */

TEST(Manifold, Transform) {
  Manifold cube = Manifold::Cube({1, 2, 3});
  Manifold cube2 = cube;
  cube.Rotate(30, 40, 50).Scale({6, 5, 4}).Translate({1, 2, 3});

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
  cube2.Transform(transform);

  Identical(cube.GetMesh(), cube2.GetMesh());
}

TEST(Manifold, MeshRelation) {
  std::vector<Mesh> input;
  std::map<int, int> meshID2idx;

  input.push_back(ImportMesh("data/gyroidpuzzle.ply"));
  Manifold gyroid(input[0]);

  std::vector<int> meshIDs = gyroid.GetMeshIDs();
  EXPECT_EQ(meshIDs.size(), 1);
  meshID2idx[meshIDs[0]] = input.size() - 1;

  Related(gyroid, input, meshID2idx);
}

TEST(Manifold, MeshRelationRefine) {
  std::vector<Mesh> input;
  std::map<int, int> meshID2idx;

  input.push_back(Csaszar());
  Manifold csaszar(input[0]);

  std::vector<int> meshIDs = csaszar.GetMeshIDs();
  EXPECT_EQ(meshIDs.size(), 1);
  meshID2idx[meshIDs[0]] = input.size() - 1;

  Related(csaszar, input, meshID2idx);
  csaszar.Refine(4);
  Related(csaszar, input, meshID2idx);
}

/**
 * The very simplest Boolean operation test.
 */
TEST(Boolean, Tetra) {
  Manifold tetra = Manifold::Tetrahedron();
  EXPECT_TRUE(tetra.IsManifold());

  Manifold tetra2 = tetra;
  tetra2.Translate(glm::vec3(0.5f));
  tetra2.SetAsOriginal();
  Manifold result = tetra2 - tetra;

  ExpectMeshes(result, {{8, 12}});

  RelatedOp(tetra, tetra2, result);
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
  Manifold cylinder = Manifold::Cylinder(1.0f, 1.0f);
  Manifold cylinder2 = cylinder;
  cylinder2.SetAsOriginal();
  Manifold out = cylinder - cylinder2.Scale({0.5f, 0.5f, 1.0f})
                                .Rotate(0, 0, 15)
                                .Translate({0.25f, 0.25f, 0.0f});
  ExpectMeshes(out, {{32, 64}});
  EXPECT_EQ(out.NumDegenerateTris(), 0);
  EXPECT_EQ(out.Genus(), 1);

  if (options.exportModels) ExportMesh("coplanar.glb", out.GetMesh(), {});

  RelatedOp(cylinder, cylinder2, out);
}

TEST(Boolean, MultiCoplanar) {
  Manifold cube = Manifold::Cube();
  Manifold cube2 = cube;
  Manifold first = cube - cube2.Translate({0.3f, 0.3f, 0.0f});
  cube.Translate({-0.3f, -0.3f, 0.0f});
  Manifold out = first - cube;
  CheckStrictly(out);
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
  ExpectMeshes(cubes, {{12, 20}});
  auto prop = cubes.GetProperties();
  EXPECT_NEAR(prop.volume, 2, 1e-5);
  EXPECT_NEAR(prop.surfaceArea, 10, 1e-5);

  if (options.exportModels) ExportMesh("faceUnion.glb", cubes.GetMesh(), {});
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
  CheckStrictly(splits.first);
  CheckStrictly(splits.second);
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
  cube.Translate({0.0f, 1.0f, 0.0f});
  cube.Rotate(0.0f, 0.0f, -60.0f);
  cube.Translate({2.0f, 0.0f, 0.0f});
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
  sphere2.SetAsOriginal();
  Manifold result = sphere - sphere2;

  ExpectMeshes(result, {{74, 144}});
  EXPECT_EQ(result.NumDegenerateTris(), 0);

  RelatedOp(sphere, sphere2, result);
}

TEST(Boolean, Gyroid) {
  Mesh gyroidpuzzle = ImportMesh("data/gyroidpuzzle.ply");
  Manifold gyroid(gyroidpuzzle);

  Mesh gyroidpuzzle2 = gyroidpuzzle;
  std::transform(gyroidpuzzle.vertPos.begin(), gyroidpuzzle.vertPos.end(),
                 gyroidpuzzle2.vertPos.begin(),
                 [](const glm::vec3& v) { return v + glm::vec3(5.0f); });
  Manifold gyroid2(gyroidpuzzle2);

  EXPECT_TRUE(gyroid.IsManifold());
  EXPECT_TRUE(gyroid.MatchesTriNormals());
  EXPECT_LE(gyroid.NumDegenerateTris(), 12);
  Manifold result = gyroid + gyroid2;

  if (options.exportModels) ExportMesh("gyroidUnion.glb", result.GetMesh(), {});

  EXPECT_TRUE(result.IsManifold());
  EXPECT_TRUE(result.MatchesTriNormals());
  EXPECT_LE(result.NumDegenerateTris(), 42);
  EXPECT_EQ(result.Decompose().size(), 1);
  auto prop = result.GetProperties();
  EXPECT_NEAR(prop.volume, 7692, 1);
  EXPECT_NEAR(prop.surfaceArea, 9642, 1);

  std::vector<Mesh> input;
  std::map<int, int> meshID2idx;

  std::vector<int> meshIDs = gyroid.GetMeshIDs();
  EXPECT_EQ(meshIDs.size(), 1);
  meshID2idx[meshIDs[0]] = input.size();
  input.push_back(gyroidpuzzle);

  meshIDs = gyroid2.GetMeshIDs();
  EXPECT_EQ(meshIDs.size(), 1);
  meshID2idx[meshIDs[0]] = input.size();
  input.push_back(gyroidpuzzle2);

  Related(result, input, meshID2idx);
}

TEST(Boolean, DISABLED_Cylinders) {
  Manifold rod = Manifold::Cylinder(1.0, 0.4, -1.0, 20);
  float arrays1[][12] = {
      {0, 0, 1, 1,    //
       -1, 0, 0, 2,   //
       0, -1, 0, 7},  //
      {1, 0, 0, 3,    //
       0, 1, 0, 2,    //
       0, 0, 1, 6},   //
      {0, 0, 1, 3,    //
       -1, 0, 0, 3,   //
       0, -1, 0, 6},  //
      {0, 0, 1, 3,    //
       -1, 0, 0, 3,   //
       0, -1, 0, 7},  //
      {0, 0, 1, 2,    //
       -1, 0, 0, 3,   //
       0, -1, 0, 8},  //
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
      {0, 0, 1, 2,    //
       -1, 0, 0, 2,   //
       0, -1, 0, 7},  //
      {1, 0, 0, 3,    //
       0, 0, 1, 2,    //
       0, -1, 0, 6},  //
      {1, 0, 0, 4,    //
       0, 1, 0, 3,    //
       0, 0, 1, 6},   //
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
    m1 += Manifold(rod).Transform(mat);
  }

  Manifold m2;
  for (auto& array : arrays2) {
    glm::mat4x3 mat;
    for (const int i : {0, 1, 2, 3}) {
      for (const int j : {0, 1, 2}) {
        mat[i][j] = array[j * 4 + i];
      }
    }
    m2 += Manifold(rod).Transform(mat);
  }

  m1 += m2;

  EXPECT_TRUE(m1.IsManifold());
  EXPECT_TRUE(m1.MatchesTriNormals());
  EXPECT_LE(m1.NumDegenerateTris(), 12);
}

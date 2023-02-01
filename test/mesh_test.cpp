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

#include <random>

#include "manifold.h"
#include "polygon.h"
#include "sdf.h"
#include "test.h"

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

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

Mesh Tet() {
  Mesh tet;
  tet.vertPos = {{-1.0f, -1.0f, 1.0f},
                 {-1.0f, 1.0f, -1.0f},
                 {1.0f, -1.0f, -1.0f},
                 {1.0f, 1.0f, 1.0f}};
  tet.triVerts = {{2, 0, 1}, {0, 3, 1}, {2, 3, 0}, {3, 2, 1}};
  return tet;
}

MeshGL TetGL() {
  MeshGL tet;
  tet.numProp = 5;
  tet.vertProperties = {-1, -1, 1,  0, 0,   //
                        -1, 1,  -1, 1, -1,  //
                        1,  -1, -1, 2, -2,  //
                        1,  1,  1,  3, -3,  //
                        -1, 1,  -1, 4, -4,  //
                        1,  -1, -1, 5, -5,  //
                        1,  1,  1,  6, -6};
  tet.triVerts = {2, 0, 1, 0, 3, 1, 2, 3, 0, 6, 5, 4};
  tet.mergeFromVert = {4, 5, 6};
  tet.mergeToVert = {1, 2, 3};
  return tet;
}

MeshGL WithIndexColors(const Mesh& in) {
  MeshGL inGL(in);
  const int numVert = in.vertPos.size();
  inGL.numProp = 6;
  inGL.vertProperties.resize(6 * numVert);
  for (int i = 0; i < numVert; ++i) {
    for (int j : {0, 1, 2}) inGL.vertProperties[6 * i + j] = in.vertPos[i][j];
    // vertex colors
    double a;
    inGL.vertProperties[6 * i + 3] = powf(modf(i * sqrt(2.0), &a), 2.2);
    inGL.vertProperties[6 * i + 4] = powf(modf(i * sqrt(3.0), &a), 2.2);
    inGL.vertProperties[6 * i + 5] = powf(modf(i * sqrt(5.0), &a), 2.2);
  }
  inGL.runIndex = {0, 3u * inGL.NumTri()};
  inGL.originalID.push_back(Manifold::ReserveIDs(1));
  inGL.transform = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
  return inGL;
}

MeshGL WithPositionColors(const Manifold& in) {
  MeshGL inGL = in.GetMeshGL();
  const int numVert = in.NumVert();
  const Box bbox = in.BoundingBox();
  const glm::vec3 size = bbox.Size();
  const std::vector<float> oldProp = inGL.vertProperties;
  inGL.numProp = 6;
  inGL.vertProperties.resize(6 * numVert);
  for (int i = 0; i < numVert; ++i) {
    for (int j : {0, 1, 2}) inGL.vertProperties[6 * i + j] = oldProp[3 * i + j];
    // vertex colors
    inGL.vertProperties[6 * i + 3] = (oldProp[3 * i] - bbox.min.x) / size.x;
    inGL.vertProperties[6 * i + 4] = (oldProp[3 * i + 1] - bbox.min.y) / size.y;
    inGL.vertProperties[6 * i + 5] = (oldProp[3 * i + 2] - bbox.min.z) / size.z;
  }
  return inGL;
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

void RelatedGL(const Manifold& out, const std::vector<MeshGL>& originals) {
  ASSERT_FALSE(out.IsEmpty());
  MeshGL output = out.GetMeshGL();
  for (int run = 0; run < output.originalID.size(); ++run) {
    const float* m = output.transform.data() + 12 * run;
    const glm::mat4x3 transform(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                                m[8], m[9], m[10], m[11]);
    int i = 0;
    for (; i < originals.size(); ++i) {
      ASSERT_EQ(originals[i].originalID.size(), 1);
      if (originals[i].originalID[0] == output.originalID[run]) break;
    }
    ASSERT_LT(i, originals.size());
    const MeshGL& inMesh = originals[i];
    for (int tri = output.runIndex[run] / 3; tri < output.runIndex[run + 1] / 3;
         ++tri) {
      ASSERT_LT(tri, output.faceID.size());
      const int inTri = output.faceID[tri];
      ASSERT_LT(inTri, inMesh.triVerts.size() / 3);
      glm::ivec3 inTriangle = {inMesh.triVerts[3 * inTri],
                               inMesh.triVerts[3 * inTri + 1],
                               inMesh.triVerts[3 * inTri + 2]};
      inTriangle *= inMesh.numProp;

      glm::mat3 inTriPos;
      for (int j : {0, 1, 2}) {
        glm::vec4 pos;
        for (int k : {0, 1, 2})
          pos[k] = inMesh.vertProperties[inTriangle[j] + k];
        pos[3] = 1;
        inTriPos[j] = transform * pos;
      }
      glm::vec3 normal =
          glm::cross(inTriPos[1] - inTriPos[0], inTriPos[2] - inTriPos[0]);
      const float area = glm::length(normal);
      if (area == 0) continue;
      normal /= area;

      for (int j : {0, 1, 2}) {
        const int vert = output.triVerts[3 * tri + j];
        glm::vec3 outPos;
        for (int k : {0, 1, 2})
          outPos[k] = output.vertProperties[vert * output.numProp + k];

        glm::vec3 edges[3];
        for (int k : {0, 1, 2}) edges[k] = inTriPos[k] - outPos;
        const float volume = glm::dot(edges[0], glm::cross(edges[1], edges[2]));
        ASSERT_LE(volume, area * 100 * out.Precision());

        for (int p = 3; p < output.numProp; ++p) {
          const float propOut =
              output.vertProperties[vert * output.numProp + p];

          glm::vec3 inProp = {inMesh.vertProperties[inTriangle[0] + p],
                              inMesh.vertProperties[inTriangle[1] + p],
                              inMesh.vertProperties[inTriangle[2] + p]};
          glm::vec3 edgesP[3];
          for (int k : {0, 1, 2}) {
            edgesP[k] = edges[k] + normal * inProp[k] - normal * propOut;
          }
          const float volumeP =
              glm::dot(edgesP[0], glm::cross(edgesP[1], edgesP[2]));

          ASSERT_LE(volumeP, area * 100 * out.Precision());
        }
      }
    }
  }
}

void RelatedOp(const Manifold& inP, const Manifold& inQ, const Manifold& outR) {
  EXPECT_GE(inP.OriginalID(), 0);
  EXPECT_GE(inQ.OriginalID(), 0);
  std::vector<MeshGL> inputGL;
  inputGL.emplace_back(inP.GetMeshGL());
  inputGL.emplace_back(inQ.GetMeshGL());
  RelatedGL(outR, inputGL);
}

struct MeshSize {
  int numVert, numTri;
  int numProp = 0;
  int numPropVert = numVert;
};

void ExpectMeshes(const Manifold& manifold,
                  const std::vector<MeshSize>& meshSize) {
  EXPECT_FALSE(manifold.IsEmpty());
  EXPECT_TRUE(manifold.IsManifold());
  EXPECT_TRUE(manifold.MatchesTriNormals());
  std::vector<Manifold> manifolds = manifold.Decompose();
  ASSERT_EQ(manifolds.size(), meshSize.size());
  std::sort(manifolds.begin(), manifolds.end(),
            [](const Manifold& a, const Manifold& b) {
              return a.NumVert() != b.NumVert() ? a.NumVert() > b.NumVert()
                                                : a.NumTri() > b.NumTri();
            });
  for (int i = 0; i < manifolds.size(); ++i) {
    EXPECT_TRUE(manifolds[i].IsManifold());
    EXPECT_EQ(manifolds[i].NumVert(), meshSize[i].numVert);
    EXPECT_EQ(manifolds[i].NumTri(), meshSize[i].numTri);
    EXPECT_EQ(manifolds[i].NumProp(), meshSize[i].numProp);
    EXPECT_EQ(manifolds[i].NumPropVert(), meshSize[i].numPropVert);
    const MeshGL meshGL = manifolds[i].GetMeshGL();
    EXPECT_EQ(meshGL.mergeFromVert.size(), meshGL.mergeToVert.size());
    EXPECT_EQ(meshGL.mergeFromVert.size(),
              meshGL.NumVert() - manifolds[i].NumVert());
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

struct Gyroid {
  __host__ __device__ float operator()(glm::vec3 p) const {
    const glm::vec3 min = p;
    const glm::vec3 max = glm::vec3(glm::two_pi<float>()) - p;
    const float min3 = glm::min(min.x, glm::min(min.y, min.z));
    const float max3 = glm::min(max.x, glm::min(max.y, max.z));
    const float bound = glm::min(min3, max3);
    const float gyroid =
        cos(p.x) * sin(p.y) + cos(p.y) * sin(p.z) + cos(p.z) * sin(p.x);
    return glm::min(gyroid, bound);
  }
};

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
  EXPECT_EQ(empty.Status(), Manifold::Error::NO_ERROR);
  EXPECT_TRUE(empty.IsManifold());
}

TEST(Manifold, ValidInput) {
  std::vector<float> propTol = {0.1, 0.2};
  Manifold tet(TetGL(), propTol);
  EXPECT_FALSE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NO_ERROR);
  EXPECT_TRUE(tet.IsManifold());
}

TEST(Manifold, InvalidInput1) {
  Mesh in = Tet();
  in.vertPos[2][1] = NAN;
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NON_FINITE_VERTEX);
  EXPECT_TRUE(tet.IsManifold());
}

TEST(Manifold, InvalidInput2) {
  Mesh in = Tet();
  std::swap(in.triVerts[2][1], in.triVerts[2][2]);
  Manifold tet(in);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::NOT_MANIFOLD);
  EXPECT_TRUE(tet.IsManifold());
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
  EXPECT_EQ(tet.Status(), Manifold::Error::VERTEX_INDEX_OUT_OF_BOUNDS);
  EXPECT_TRUE(tet.IsManifold());
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
  EXPECT_EQ(tet.Status(), Manifold::Error::VERTEX_INDEX_OUT_OF_BOUNDS);
  EXPECT_TRUE(tet.IsManifold());
}

TEST(Manifold, InvalidInput5) {
  MeshGL tetGL = TetGL();
  tetGL.mergeFromVert[tetGL.mergeFromVert.size() - 1] = 7;
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::MERGE_INDEX_OUT_OF_BOUNDS);
  EXPECT_TRUE(tet.IsManifold());
}

TEST(Manifold, InvalidInput7) {
  MeshGL tetGL = TetGL();
  tetGL.triVerts[tetGL.triVerts.size() - 1] = 7;
  Manifold tet(tetGL);
  EXPECT_TRUE(tet.IsEmpty());
  EXPECT_EQ(tet.Status(), Manifold::Error::VERTEX_INDEX_OUT_OF_BOUNDS);
  EXPECT_TRUE(tet.IsManifold());
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
  EXPECT_TRUE(cube.IsManifold());
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

/**
 * Curvature is the inverse of the radius of curvature, and signed such that
 * positive is convex and negative is concave. There are two orthogonal
 * principal curvatures at any point on a manifold, with one maximum and the
 * other minimum. Gaussian curvature is their product, while mean
 * curvature is their sum. Here we check our discrete approximations calculated
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

    sphere = sphere.Scale(glm::vec3(2.0f));
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
  const float period = glm::two_pi<float>();

  Mesh gyroidMesh = LevelSet(Gyroid(), {glm::vec3(0), glm::vec3(period)}, 3);
  MeshGL gyroidMeshGL = WithIndexColors(gyroidMesh);
  std::vector<float> tol(3, 0.0001);
  Manifold gyroid(gyroidMeshGL, tol);

  std::vector<MeshGL> inputGL;

#ifdef MANIFOLD_EXPORT
  ExportOptions opt;
  opt.mat.roughness = 1;
  opt.mat.colorChannels = glm::ivec4(3, 4, 5, -1);
  if (options.exportModels) ExportMesh("gyroid.glb", gyroid.GetMeshGL(), opt);
#endif

  EXPECT_GE(gyroid.OriginalID(), 0);
  inputGL.emplace_back(gyroidMeshGL);

  RelatedGL(gyroid, inputGL);
}

TEST(Manifold, MeshRelationRefine) {
  std::vector<MeshGL> inputGL;

  const Mesh in = Csaszar();
  MeshGL inGL = WithIndexColors(in);

  inputGL.emplace_back(inGL);
  std::vector<float> tol(3, 0);
  Manifold csaszar(inGL, tol);

  int meshID = csaszar.OriginalID();
  EXPECT_GE(meshID, 0);

  RelatedGL(csaszar, inputGL);
  csaszar.Refine(4);
  RelatedGL(csaszar, inputGL);
}

/**
 * The very simplest Boolean operation test.
 */
TEST(Boolean, Tetra) {
  Manifold tetra = Manifold::Tetrahedron();
  MeshGL tetraGL = WithPositionColors(tetra);
  std::vector<float> tol(3, 0.01);
  tetra = Manifold(tetraGL, tol);
  EXPECT_TRUE(!tetra.IsEmpty());

  Manifold tetra2 = tetra;
  tetra2 = tetra2.Translate(glm::vec3(0.5f)).AsOriginal();
  Manifold result = tetra2 - tetra;

  ExpectMeshes(result, {{8, 12, 3, 11}});

  RelatedOp(tetra, tetra2, result);
}

TEST(Boolean, Mirrored) {
  Manifold cube = Manifold::Cube(glm::vec3(1)).Scale({1, -1, 1});
  EXPECT_TRUE(cube.IsManifold());
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
  MeshGL cylinderGL = WithPositionColors(cylinder);
  std::vector<float> tol(3, 0.0001);
  cylinder = Manifold(cylinderGL, tol);

  Manifold cylinder2 = Manifold(cylinderGL, tol);
  cylinder2 = cylinder2.Scale({0.5f, 0.5f, 1.0f})
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
  std::vector<float> tol(3, 0.0001);
  cylinder = Manifold(cylinderGL, tol);

  Manifold cylinder2 = Manifold(cylinderGL, tol);
  cylinder2 = cylinder2.Scale({0.5f, 0.5f, 1.0f})
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

  RelatedOp(cylinder, cylinder2, out);
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
  EXPECT_TRUE((cube ^= doubled).IsManifold());
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
  std::vector<float> tol(3, 0.01);
  sphere = Manifold(sphereGL, tol);

  Manifold sphere2 = sphere;
  sphere2 = sphere2.Translate(glm::vec3(0.5));
  sphere2 = sphere2.AsOriginal();
  Manifold result = sphere - sphere2;

  ExpectMeshes(result, {{74, 144, 3, 110}});
  EXPECT_EQ(result.NumDegenerateTris(), 0);

  RelatedOp(sphere, sphere2, result);
}

TEST(Boolean, MeshRelation) {
  const float period = glm::two_pi<float>();

  Mesh gyroidMesh = LevelSet(Gyroid(), {glm::vec3(0), glm::vec3(period)}, 0.5);
  MeshGL gyroidMeshGL = WithPositionColors(gyroidMesh);
  std::vector<float> tol(3, 0.0001);
  Manifold gyroid(gyroidMeshGL, tol);

  Mesh gyroidMesh2 = gyroidMesh;
  std::transform(gyroidMesh.vertPos.begin(), gyroidMesh.vertPos.end(),
                 gyroidMesh2.vertPos.begin(),
                 [](const glm::vec3& v) { return v + glm::vec3(2.0f); });
  MeshGL gyroidMeshGL2 = WithPositionColors(gyroidMesh2);
  Manifold gyroid2(gyroidMeshGL2, tol);

  EXPECT_FALSE(gyroid.IsEmpty());
  EXPECT_TRUE(gyroid.IsManifold());
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

  EXPECT_TRUE(result.IsManifold());
  EXPECT_TRUE(result.MatchesTriNormals());
  EXPECT_LE(result.NumDegenerateTris(), 1);
  EXPECT_EQ(result.Decompose().size(), 1);
  auto prop = result.GetProperties();
  EXPECT_NEAR(prop.volume, 226, 1);
  EXPECT_NEAR(prop.surfaceArea, 387, 1);

  std::vector<MeshGL> inputGL;

  EXPECT_GE(gyroid.OriginalID(), 0);
  inputGL.emplace_back(gyroidMeshGL);

  EXPECT_GE(gyroid2.OriginalID(), 0);
  inputGL.emplace_back(gyroidMeshGL2);

  RelatedGL(result, inputGL);
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

  EXPECT_TRUE(m1.IsManifold());
  EXPECT_TRUE(m1.MatchesTriNormals());
  EXPECT_LE(m1.NumDegenerateTris(), 12);
}

TEST(Boolean, Cubes) {
  Manifold result = Manifold::Cube({1.2, 1, 1}, true).Translate({0, -0.5, 0.5});
  result += Manifold::Cube({1, 0.8, 0.5}).Translate({-0.5, 0, 0.5});
  result += Manifold::Cube({1.2, 0.1, 0.5}).Translate({-0.6, -0.1, 0});

  EXPECT_TRUE(result.IsManifold());
  EXPECT_TRUE(result.MatchesTriNormals());
  EXPECT_LE(result.NumDegenerateTris(), 0);
  auto prop = result.GetProperties();
  EXPECT_NEAR(prop.volume, 1.6, 0.001);
  EXPECT_NEAR(prop.surfaceArea, 9.2, 0.01);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) ExportMesh("cubes.glb", result.GetMesh(), {});
#endif
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

// FIXME: test is failing on Mac CI (passing on others)
TEST(Boolean, DISABLED_Close) {
  PolygonParams().processOverlaps = true;

  const float r = 10;
  Manifold a = Manifold::Sphere(r, 256);
  Manifold result = a;
  for (int i = 0; i < 10; i++) {
    // std::cout << i << std::endl;
    result ^= a.Translate({a.Precision() / 10 * i, 0.0, 0.0});
    EXPECT_TRUE(result.IsManifold());
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

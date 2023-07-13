// Copyright 2022 The Manifold Authors.
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
#include "sdf.h"
#include "test.h"

using namespace manifold;

Options options;

void print_usage() {
  printf("-------------------------------\n");
  printf("manifold_test specific options:\n");
  printf("  -h: Print this message\n");
  printf("  -e: Export GLB models of samples\n");
  printf(
      "  -v: Enable verbose output (only works if compiled with MANIFOLD_DEBUG "
      "flag)\n");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  for (int i = 1; i < argc; i++) {
    if (argv[i][0] != '-') {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      print_usage();
      return 1;
    }
    switch (argv[i][1]) {
      case 'h':
        print_usage();
        return 0;
      case 'e':
#ifndef MANIFOLD_EXPORT
        printf(
            "Export not possible because MANIFOLD_EXPORT compile flag is not "
            "set.\n");
#endif
        options.exportModels = true;
        break;
      case 'v':
        options.params.verbose = true;
        manifold::ManifoldParams().verbose = true;
        manifold::ManifoldParams().intermediateChecks = true;
        break;
      default:
        fprintf(stderr, "Unknown option: %s\n", argv[i]);
        print_usage();
        return 1;
    }
  }

  manifold::PolygonParams().intermediateChecks = true;
  manifold::PolygonParams().processOverlaps = false;

  return RUN_ALL_TESTS();
}

Polygons SquareHole(float xOffset) {
  Polygons polys;
  polys.push_back({
      {2 + xOffset, 2},    //
      {-2 + xOffset, 2},   //
      {-2 + xOffset, -2},  //
      {2 + xOffset, -2},   //
  });
  polys.push_back({
      {-1 + xOffset, 1},   //
      {1 + xOffset, 1},    //
      {1 + xOffset, -1},   //
      {-1 + xOffset, -1},  //
  });
  return polys;
}

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

struct GyroidSDF {
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

Mesh Gyroid() {
  const float period = glm::two_pi<float>();
  return LevelSet(GyroidSDF(), {glm::vec3(0), glm::vec3(period)}, 0.5);
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

// STL-style meshGL with face normals. Not manifold, requires Merge().
MeshGL CubeSTL() {
  const MeshGL cubeIn = Manifold::Cube(glm::vec3(1), true).GetMeshGL();
  MeshGL cube;
  cube.numProp = 6;

  for (int tri = 0, vert = 0; tri < cubeIn.NumTri(); tri++) {
    glm::mat3 triPos;
    for (const int i : {0, 1, 2}) {
      cube.triVerts.push_back(vert++);

      for (const int j : {0, 1, 2}) {
        triPos[i][j] =
            cubeIn
                .vertProperties[cubeIn.numProp * cubeIn.triVerts[3 * tri + i] +
                                j];
      }
    }

    const glm::vec3 normal = glm::normalize(
        glm::cross(triPos[1] - triPos[0], triPos[2] - triPos[0]));
    for (const int i : {0, 1, 2}) {
      for (const int j : {0, 1, 2}) {
        cube.vertProperties.push_back(triPos[i][j]);
      }
      for (const int j : {0, 1, 2}) {
        cube.vertProperties.push_back(normal[j]);
      }
    }
  }

  cube.runOriginalID.push_back(Manifold::ReserveIDs(1));

  return cube;
}

MeshGL WithIndexColors(const Mesh& in) {
  MeshGL inGL(in);
  inGL.runOriginalID = {Manifold::ReserveIDs(1)};
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
  return inGL;
}

MeshGL WithPositionColors(const Manifold& in) {
  const Box bbox = in.BoundingBox();
  const glm::vec3 size = bbox.Size();

  Manifold out = in.SetProperties(
      3, [bbox, size](float* prop, glm::vec3 pos, const float* oldProp) {
        for (int i : {0, 1, 2}) {
          prop[i] = (pos[i] - bbox.min[i]) / size[i];
        }
      });

  MeshGL outGL = out.GetMeshGL();
  outGL.runIndex.clear();
  outGL.runOriginalID.clear();
  outGL.runTransform.clear();
  outGL.faceID.clear();
  outGL.runOriginalID = {Manifold::ReserveIDs(1)};
  return outGL;
}

MeshGL WithNormals(const Manifold& in) {
  const Mesh mesh = in.GetMesh();
  MeshGL out;
  out.runOriginalID = {Manifold::ReserveIDs(1)};
  out.numProp = 6;
  out.vertProperties.resize(out.numProp * mesh.vertPos.size());
  for (int i = 0; i < mesh.vertPos.size(); ++i) {
    for (int j : {0, 1, 2}) {
      out.vertProperties[6 * i + j] = mesh.vertPos[i][j];
      out.vertProperties[6 * i + 3 + j] = mesh.vertNormal[i][j];
    }
  }
  out.triVerts.resize(3 * mesh.triVerts.size());
  for (int i = 0; i < mesh.triVerts.size(); ++i) {
    for (int j : {0, 1, 2}) out.triVerts[3 * i + j] = mesh.triVerts[i][j];
  }
  return out;
}

MeshGL CubeUV() {
  MeshGL mgl;
  mgl.numProp = 5;
  mgl.vertProperties = {0.5,  -0.5, 0.5,  0.5,  0.66,  //
                        -0.5, -0.5, 0.5,  0.25, 0.66,  //
                        0.5,  0.5,  0.5,  0.5,  0.33,  //
                        -0.5, 0.5,  0.5,  0.25, 0.33,  //
                        -0.5, -0.5, -0.5, 1.0,  0.66,  //
                        0.5,  -0.5, -0.5, 0.75, 0.66,  //
                        -0.5, 0.5,  -0.5, 1.0,  0.33,  //
                        0.5,  0.5,  -0.5, 0.75, 0.33,  //
                        -0.5, -0.5, -0.5, 0.0,  0.66,  //
                        -0.5, 0.5,  -0.5, 0.0,  0.33,  //
                        -0.5, 0.5,  -0.5, 0.25, 0.0,   //
                        0.5,  0.5,  -0.5, 0.5,  0.0,   //
                        -0.5, -0.5, -0.5, 0.25, 1.0,   //
                        0.5,  -0.5, -0.5, 0.5,  1.0};
  mgl.triVerts = {3, 1, 0, 3, 0, 2, 7,  5,  4, 7,  4, 6, 2, 0, 5,  2, 5,  7,
                  9, 8, 1, 9, 1, 3, 11, 10, 3, 11, 3, 2, 0, 1, 12, 0, 12, 13};
  mgl.mergeFromVert = {8, 12, 13, 9, 10, 11};
  mgl.mergeToVert = {4, 4, 5, 6, 6, 7};
  mgl.runOriginalID.push_back(Manifold::ReserveIDs(1));
  return mgl;
}

float GetMaxProperty(const MeshGL& mesh, int channel) {
  float max = -std::numeric_limits<float>::infinity();
  const int numVert = mesh.NumVert();
  for (int i = 0; i < numVert; ++i) {
    max = glm::max(max, mesh.vertProperties[i * mesh.numProp + channel]);
  }
  return max;
}

float GetMinProperty(const MeshGL& mesh, int channel) {
  float min = std::numeric_limits<float>::infinity();
  const int numVert = mesh.NumVert();
  for (int i = 0; i < numVert; ++i) {
    min = glm::min(min, mesh.vertProperties[i * mesh.numProp + channel]);
  }
  return min;
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

void RelatedGL(const Manifold& out, const std::vector<MeshGL>& originals,
               bool checkNormals, bool updateNormals) {
  ASSERT_FALSE(out.IsEmpty());
  const glm::ivec3 normalIdx =
      updateNormals ? glm::ivec3(3, 4, 5) : glm::ivec3(0);
  MeshGL output = out.GetMeshGL(normalIdx);
  for (int run = 0; run < output.runOriginalID.size(); ++run) {
    const float* m = output.runTransform.data() + 12 * run;
    const glm::mat4x3 transform =
        output.runTransform.empty()
            ? glm::mat4x3(1.0f)
            : glm::mat4x3(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
                          m[9], m[10], m[11]);
    int i = 0;
    for (; i < originals.size(); ++i) {
      ASSERT_EQ(originals[i].runOriginalID.size(), 1);
      if (originals[i].runOriginalID[0] == output.runOriginalID[run]) break;
    }
    ASSERT_LT(i, originals.size());
    const MeshGL& inMesh = originals[i];
    for (int tri = output.runIndex[run] / 3; tri < output.runIndex[run + 1] / 3;
         ++tri) {
      if (!output.faceID.empty()) {
        ASSERT_LT(tri, output.faceID.size());
      }
      const int inTri = output.faceID.empty() ? tri : output.faceID[tri];
      ASSERT_LT(inTri, inMesh.triVerts.size() / 3);
      glm::ivec3 inTriangle = {inMesh.triVerts[3 * inTri],
                               inMesh.triVerts[3 * inTri + 1],
                               inMesh.triVerts[3 * inTri + 2]};
      inTriangle *= inMesh.numProp;

      glm::mat3 inTriPos;
      glm::mat3 outTriPos;
      for (int j : {0, 1, 2}) {
        const int vert = output.triVerts[3 * tri + j];
        glm::vec4 pos;
        for (int k : {0, 1, 2}) {
          pos[k] = inMesh.vertProperties[inTriangle[j] + k];
          outTriPos[j][k] = output.vertProperties[vert * output.numProp + k];
        }
        pos[3] = 1;
        inTriPos[j] = transform * pos;
      }
      glm::vec3 outNormal =
          glm::cross(outTriPos[1] - outTriPos[0], outTriPos[2] - outTriPos[0]);
      glm::vec3 inNormal =
          glm::cross(inTriPos[1] - inTriPos[0], inTriPos[2] - inTriPos[0]);
      const float area = glm::length(inNormal);
      if (area == 0) continue;
      inNormal /= area;

      for (int j : {0, 1, 2}) {
        const int vert = output.triVerts[3 * tri + j];
        glm::vec3 edges[3];
        for (int k : {0, 1, 2}) edges[k] = inTriPos[k] - outTriPos[j];
        const float volume = glm::dot(edges[0], glm::cross(edges[1], edges[2]));
        ASSERT_LE(volume, area * 100 * out.Precision());

        if (checkNormals) {
          glm::vec3 normal;
          for (int k : {0, 1, 2})
            normal[k] = output.vertProperties[vert * output.numProp + 3 + k];
          ASSERT_NEAR(glm::length(normal), 1, 0.0001);
          ASSERT_GT(glm::dot(normal, outNormal), 0);
        } else {
          for (int p = 3; p < inMesh.numProp; ++p) {
            const float propOut =
                output.vertProperties[vert * output.numProp + p];

            glm::vec3 inProp = {inMesh.vertProperties[inTriangle[0] + p],
                                inMesh.vertProperties[inTriangle[1] + p],
                                inMesh.vertProperties[inTriangle[2] + p]};
            glm::vec3 edgesP[3];
            for (int k : {0, 1, 2}) {
              edgesP[k] = edges[k] + inNormal * inProp[k] - inNormal * propOut;
            }
            const float volumeP =
                glm::dot(edgesP[0], glm::cross(edgesP[1], edgesP[2]));

            ASSERT_LE(volumeP, area * 100 * out.Precision());
          }
        }
      }
    }
  }
}

void ExpectMeshes(const Manifold& manifold,
                  const std::vector<MeshSize>& meshSize) {
  EXPECT_FALSE(manifold.IsEmpty());
  EXPECT_TRUE(manifold.MatchesTriNormals());
  std::vector<Manifold> manifolds = manifold.Decompose();
  ASSERT_EQ(manifolds.size(), meshSize.size());
  std::sort(manifolds.begin(), manifolds.end(),
            [](const Manifold& a, const Manifold& b) {
              return a.NumVert() != b.NumVert() ? a.NumVert() > b.NumVert()
                                                : a.NumTri() > b.NumTri();
            });
  for (int i = 0; i < manifolds.size(); ++i) {
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

void CheckNormals(const Manifold& manifold) {
  EXPECT_TRUE(manifold.MatchesTriNormals());
  for (const glm::vec3& normal : manifold.GetMesh().vertNormal) {
    ASSERT_NEAR(glm::length(normal), 1, 0.0001);
  }
}

void CheckStrictly(const Manifold& manifold) {
  CheckNormals(manifold);
  EXPECT_EQ(manifold.NumDegenerateTris(), 0);
}

void CheckGL(const Manifold& manifold) {
  ASSERT_FALSE(manifold.IsEmpty());
  const MeshGL meshGL = manifold.GetMeshGL();
  EXPECT_EQ(meshGL.mergeFromVert.size(), meshGL.mergeToVert.size());
  EXPECT_EQ(meshGL.mergeFromVert.size(), meshGL.NumVert() - manifold.NumVert());
  EXPECT_EQ(meshGL.runIndex.size(), meshGL.runOriginalID.size() + 1);
  EXPECT_EQ(meshGL.runIndex.front(), 0);
  EXPECT_EQ(meshGL.runIndex.back(), 3 * meshGL.NumTri());
  if (!meshGL.runTransform.empty()) {
    EXPECT_EQ(meshGL.runTransform.size(), 12 * meshGL.runOriginalID.size());
  }
  EXPECT_EQ(meshGL.faceID.size(), meshGL.NumTri());
}
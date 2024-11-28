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
#include <algorithm>

#include "manifold/manifold.h"
#include "manifold/polygon.h"
#include "test.h"

#if (MANIFOLD_PAR == 1)
#include <oneapi/tbb/parallel_for.h>
#endif

// we need to call some tracy API to establish the connection
#if __has_include(<tracy/Tracy.hpp>)
#include <tracy/Tracy.hpp>
#else
#define FrameMarkStart(x)
#define FrameMarkEnd(x)
#endif

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

  const char* name = "test setup";
  FrameMarkStart(name);

  // warmup tbb for emscripten, according to
  // https://github.com/oneapi-src/oneTBB/blob/master/WASM_Support.md#limitations
#if defined(__EMSCRIPTEN__) && (MANIFOLD_PAR == 1)
  int num_threads = tbb::this_task_arena::max_concurrency();
  std::atomic<int> barrier{num_threads};
  tbb::parallel_for(
      0, num_threads,
      [&barrier](int) {
        barrier--;
        while (barrier > 0) {
          // Send browser thread to event loop
          std::this_thread::yield();
        }
      },
      tbb::static_partitioner{});
#endif

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

  FrameMarkEnd(name);

  RegisterPolygonTests();
  return RUN_ALL_TESTS();
}

Polygons SquareHole(double xOffset) {
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

MeshGL Csaszar() {
  MeshGL csaszar;
  csaszar.numProp = 3;
  csaszar.vertProperties = {-20, -20, -10,  //
                            -20, 20,  -15,  //
                            -5,  -8,  8,    //
                            0,   0,   30,   //
                            5,   8,   8,    //
                            20,  -20, -15,  //
                            20,  20,  -10};
  csaszar.triVerts = {1, 3, 6,  //
                      1, 6, 5,  //
                      2, 5, 6,  //
                      0, 2, 6,  //
                      0, 6, 4,  //
                      3, 4, 6,  //
                      1, 2, 3,  //
                      1, 4, 2,  //
                      1, 0, 4,  //
                      1, 5, 0,  //
                      3, 5, 4,  //
                      0, 5, 3,  //
                      0, 3, 2,  //
                      2, 4, 5};
  csaszar.runOriginalID = {Manifold::ReserveIDs(1)};
  return csaszar;
}

struct GyroidSDF {
  double operator()(vec3 p) const {
    const vec3 min = p;
    const vec3 max = vec3(kTwoPi) - p;
    const double min3 = std::min(min.x, std::min(min.y, min.z));
    const double max3 = std::min(max.x, std::min(max.y, max.z));
    const double bound = std::min(min3, max3);
    const double gyroid =
        cos(p.x) * sin(p.y) + cos(p.y) * sin(p.z) + cos(p.z) * sin(p.x);
    return std::min(gyroid, bound);
  }
};

Manifold Gyroid() {
  const double period = kTwoPi;
  return Manifold::LevelSet(GyroidSDF(), {vec3(0.0), vec3(period)}, 0.5);
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
  const MeshGL cubeIn = Manifold::Cube(vec3(1), true).GetMeshGL();
  MeshGL cube;
  cube.numProp = 6;

  for (size_t tri = 0, vert = 0; tri < cubeIn.NumTri(); tri++) {
    mat3 triPos;
    for (const int i : {0, 1, 2}) {
      cube.triVerts.push_back(vert++);

      for (const int j : {0, 1, 2}) {
        triPos[i][j] =
            cubeIn
                .vertProperties[cubeIn.numProp * cubeIn.triVerts[3 * tri + i] +
                                j];
      }
    }

    const vec3 normal =
        la::normalize(la::cross(triPos[1] - triPos[0], triPos[2] - triPos[0]));
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

Manifold WithPositionColors(const Manifold& in) {
  const Box bbox = in.BoundingBox();
  const vec3 size = bbox.Size();

  return in.SetProperties(
      3, [bbox, size](double* prop, vec3 pos, const double* oldProp) {
        for (int i : {0, 1, 2}) {
          prop[i] = (pos[i] - bbox.min[i]) / size[i];
        }
      });
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
    max = std::max(max, mesh.vertProperties[i * mesh.numProp + channel]);
  }
  return max;
}

float GetMinProperty(const MeshGL& mesh, int channel) {
  float min = std::numeric_limits<float>::infinity();
  const int numVert = mesh.NumVert();
  for (int i = 0; i < numVert; ++i) {
    min = std::min(min, mesh.vertProperties[i * mesh.numProp + channel]);
  }
  return min;
}

void CheckFinite(const MeshGL& mesh) {
  for (float v : mesh.vertProperties) {
    ASSERT_TRUE(std::isfinite(v));
  }
  for (float v : mesh.runTransform) {
    ASSERT_TRUE(std::isfinite(v));
  }
  for (float v : mesh.halfedgeTangent) {
    ASSERT_TRUE(std::isfinite(v));
  }
}

void Identical(const MeshGL& mesh1, const MeshGL& mesh2) {
  ASSERT_EQ(mesh1.vertProperties.size() / mesh1.numProp,
            mesh2.vertProperties.size() / mesh2.numProp);
  for (size_t i = 0; i < mesh1.vertProperties.size() / mesh1.numProp; ++i)
    ASSERT_LE(la::length(mesh1.GetVertPos(i) - mesh2.GetVertPos(i)), 0.0001);

  ASSERT_EQ(mesh1.triVerts.size(), mesh2.triVerts.size());

  // reorder faces
  std::vector<ivec3> triVerts1(mesh1.triVerts.size() / 3);
  std::vector<ivec3> triVerts2(mesh1.triVerts.size() / 3);

  for (size_t i = 0; i < triVerts1.size(); ++i) {
    triVerts1[i] = ivec3(mesh1.triVerts[3 * i], mesh1.triVerts[3 * i + 1],
                         mesh1.triVerts[3 * i + 2]);
    triVerts2[i] = ivec3(mesh2.triVerts[3 * i], mesh2.triVerts[3 * i + 1],
                         mesh2.triVerts[3 * i + 2]);
  }
  auto comp = [](const ivec3& a, const ivec3& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y) ||
           (a.x == b.x && a.y == b.y && a.z < b.z);
  };
  std::sort(triVerts1.begin(), triVerts1.end(), comp);
  std::sort(triVerts2.begin(), triVerts2.end(), comp);
  for (size_t i = 0; i < triVerts1.size(); ++i)
    ASSERT_EQ(triVerts1[i], triVerts2[i]);
}

void RelatedGL(const Manifold& out, const std::vector<MeshGL>& originals,
               bool checkNormals, bool updateNormals) {
  ASSERT_FALSE(out.IsEmpty());
  const int normalIdx = updateNormals ? 0 : -1;
  MeshGL output = out.GetMeshGL(normalIdx);

  for (size_t run = 0; run < output.runOriginalID.size(); ++run) {
    const float* m = output.runTransform.data() + 12 * run;
    const mat3x4 transform =
        output.runTransform.empty()
            ? la::identity
            : mat3x4({m[0], m[1], m[2]}, {m[3], m[4], m[5]}, {m[6], m[7], m[8]},
                     {m[9], m[10], m[11]});
    size_t i = 0;
    for (; i < originals.size(); ++i) {
      ASSERT_EQ(originals[i].runOriginalID.size(), 1);
      if (originals[i].runOriginalID[0] == output.runOriginalID[run]) break;
    }
    ASSERT_LT(i, originals.size());
    const MeshGL& inMesh = originals[i];
    const float tolerance =
        3 * std::max(static_cast<float>(out.GetTolerance()), inMesh.tolerance);

    for (uint32_t tri = output.runIndex[run] / 3;
         tri < output.runIndex[run + 1] / 3; ++tri) {
      if (!output.faceID.empty()) {
        ASSERT_LT(tri, output.faceID.size());
      }
      const int inTri = output.faceID.empty() ? tri : output.faceID[tri];
      ASSERT_LT(inTri, inMesh.triVerts.size() / 3);
      ivec3 inTriangle(inMesh.triVerts[3 * inTri],
                       inMesh.triVerts[3 * inTri + 1],
                       inMesh.triVerts[3 * inTri + 2]);
      inTriangle *= static_cast<int>(inMesh.numProp);

      mat3 inTriPos;
      mat3 outTriPos;
      for (int j : {0, 1, 2}) {
        const int vert = output.triVerts[3 * tri + j];
        vec4 pos;
        for (int k : {0, 1, 2}) {
          pos[k] = inMesh.vertProperties[inTriangle[j] + k];
          outTriPos[j][k] = output.vertProperties[vert * output.numProp + k];
        }
        pos[3] = 1;
        inTriPos[j] = transform * pos;
      }
      vec3 outNormal =
          la::cross(outTriPos[1] - outTriPos[0], outTriPos[2] - outTriPos[0]);
      vec3 inNormal =
          la::cross(inTriPos[1] - inTriPos[0], inTriPos[2] - inTriPos[0]);
      const double area = la::length(inNormal);
      if (area == 0) continue;
      inNormal /= area;

      for (int j : {0, 1, 2}) {
        const int vert = output.triVerts[3 * tri + j];
        vec3 edges[3];
        for (int k : {0, 1, 2}) edges[k] = inTriPos[k] - outTriPos[j];
        const double volume = la::dot(edges[0], la::cross(edges[1], edges[2]));
        ASSERT_LE(volume, area * tolerance);

        if (checkNormals) {
          vec3 normal;
          for (int k : {0, 1, 2})
            normal[k] = output.vertProperties[vert * output.numProp + 3 + k];
          ASSERT_NEAR(la::length(normal), 1, 0.0001);
          ASSERT_GT(la::dot(normal, outNormal), 0);
        } else {
          for (size_t p = 3; p < inMesh.numProp; ++p) {
            const double propOut =
                output.vertProperties[vert * output.numProp + p];

            vec3 inProp = {inMesh.vertProperties[inTriangle[0] + p],
                           inMesh.vertProperties[inTriangle[1] + p],
                           inMesh.vertProperties[inTriangle[2] + p]};
            vec3 edgesP[3];
            for (int k : {0, 1, 2}) {
              edgesP[k] = edges[k] + inNormal * inProp[k] - inNormal * propOut;
            }
            const double volumeP =
                la::dot(edgesP[0], la::cross(edgesP[1], edgesP[2]));

            ASSERT_LE(volumeP, area * tolerance);
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
  for (size_t i = 0; i < manifolds.size(); ++i) {
    EXPECT_EQ(manifolds[i].NumVert(), meshSize[i].numVert);
    EXPECT_EQ(manifolds[i].NumTri(), meshSize[i].numTri);
    EXPECT_EQ(manifolds[i].NumProp(), meshSize[i].numProp);
    EXPECT_EQ(manifolds[i].NumPropVert(), meshSize[i].numPropVert);
    const MeshGL meshGL = manifolds[i].GetMeshGL();
    EXPECT_EQ(meshGL.mergeFromVert.size(), meshGL.mergeToVert.size());
    EXPECT_EQ(meshGL.mergeFromVert.size(),
              meshGL.NumVert() - manifolds[i].NumVert());
  }
}

void CheckStrictly(const Manifold& manifold) {
  EXPECT_EQ(manifold.NumDegenerateTris(), 0);
}

void CheckGL(const Manifold& manifold, bool noMerge) {
  ASSERT_FALSE(manifold.IsEmpty());
  const MeshGL meshGL = manifold.GetMeshGL();
  if (noMerge) {
    EXPECT_EQ(manifold.NumVert(), meshGL.NumVert());
  }
  EXPECT_EQ(meshGL.mergeFromVert.size(), meshGL.mergeToVert.size());
  EXPECT_EQ(meshGL.mergeFromVert.size(), meshGL.NumVert() - manifold.NumVert());
  EXPECT_EQ(meshGL.runIndex.size(), meshGL.runOriginalID.size() + 1);
  EXPECT_EQ(meshGL.runIndex.front(), 0);
  EXPECT_EQ(meshGL.runIndex.back(), 3 * meshGL.NumTri());
  if (!meshGL.runTransform.empty()) {
    EXPECT_EQ(meshGL.runTransform.size(), 12 * meshGL.runOriginalID.size());
  }
  EXPECT_EQ(meshGL.faceID.size(), meshGL.NumTri());
  CheckFinite(meshGL);
}

#ifdef MANIFOLD_EXPORT
Manifold ReadMesh(const std::string& filename) {
  std::string file = __FILE__;
  std::string dir = file.substr(0, file.rfind('/'));
  return Manifold(ImportMesh(dir + "/models/" + filename));
}
#endif

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

#include <algorithm>

#include "../src/tri_dist.h"
#include "manifold/manifold.h"
#include "samples.h"
#include "test.h"

using namespace manifold;

// Check if the mesh remains convex after adding new faces
bool isMeshConvex(Manifold hullManifold, double epsilon = 0.0000001) {
  // Get the mesh from the manifold
  MeshGL64 mesh = hullManifold.GetMeshGL64();

  const auto numTri = mesh.NumTri();
  const auto numVert = mesh.NumVert();
  const auto numProp = mesh.numProp;

  // Iterate over each triangle
  for (size_t t = 0; t < numTri; ++t) {
    // Get the vertices of the triangle
    auto tri = mesh.GetTriVerts(t);
    vec3 v0 = mesh.GetVertPos(tri[0]);
    vec3 v1 = mesh.GetVertPos(tri[1]);
    vec3 v2 = mesh.GetVertPos(tri[2]);

    // Compute the normal of the triangle
    vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

    // Check all other vertices
    for (size_t i = 0; i < numVert; ++i) {
      if (i == tri[0] || i == tri[1] || i == tri[2])
        continue;  // Skip vertices of the current triangle

      // Get the vertex
      vec3 v = mesh.GetVertPos(i);

      // Compute the signed distance from the plane
      double distance = glm::dot(normal, v - v0);

      // If any vertex lies on the opposite side of the normal direction
      if (distance > epsilon) {
        std::cout << distance << std::endl;
        // The manifold is not convex
        return false;
      }
    }
  }
  // If we didn't find any vertex on the opposite side for any triangle, it's
  // convex
  return true;
}

TEST(Hull, Tictac) {
  const double tictacRad = 100;
  const double tictacHeight = 500;
  const int tictacSeg = 500;
  const double tictacMid = tictacHeight - 2 * tictacRad;
  const auto sphere = Manifold::Sphere(tictacRad, tictacSeg);
  const std::vector<Manifold> spheres{sphere,
                                      sphere.Translate({0, 0, tictacMid})};
  const auto tictac = Manifold::Hull(spheres);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportMesh("tictac_hull.glb", tictac.GetMeshGL(), {});
  }
#endif

  EXPECT_NEAR(sphere.NumVert() + tictacSeg, tictac.NumVert(), 1);
}

TEST(Hull, Hollow) {
  auto sphere = Manifold::Sphere(100, 360);
  auto hollow = sphere - sphere.Scale({0.8, 0.8, 0.8});
  const double sphere_vol = sphere.GetProperties().volume;
  EXPECT_FLOAT_EQ(hollow.Hull().GetProperties().volume, sphere_vol);
}

TEST(Hull, Cube) {
  std::vector<vec3> cubePts = {
      {0, 0, 0},       {1, 0, 0},   {0, 1, 0},      {0, 0, 1},  // corners
      {1, 1, 0},       {0, 1, 1},   {1, 0, 1},      {1, 1, 1},  // corners
      {0.5, 0.5, 0.5}, {0.5, 0, 0}, {0.5, 0.7, 0.2}  // internal points
  };
  auto cube = Manifold::Hull(cubePts);
  EXPECT_FLOAT_EQ(cube.GetProperties().volume, 1);
}

TEST(Hull, Empty) {
  const std::vector<vec3> tooFew{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  EXPECT_TRUE(Manifold::Hull(tooFew).AsOriginal().IsEmpty());

  const std::vector<vec3> coplanar{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
  EXPECT_TRUE(Manifold::Hull(coplanar).AsOriginal().IsEmpty());
}

TEST(Hull, MengerSponge) {
  Manifold sponge = MengerSponge(4);
  sponge = sponge.Rotate(10, 20, 30);
  Manifold spongeHull = sponge.Hull();
  EXPECT_EQ(spongeHull.NumTri(), 12);
  EXPECT_FLOAT_EQ(spongeHull.GetProperties().surfaceArea, 6);
  EXPECT_FLOAT_EQ(spongeHull.GetProperties().volume, 1);
}

TEST(Hull, Sphere) {
  Manifold sphere = Manifold::Sphere(1, 1500);
  sphere = sphere.Translate(vec3(0.5));
  Manifold sphereHull = sphere.Hull();
  EXPECT_EQ(sphereHull.NumTri(), sphere.NumTri());
  EXPECT_FLOAT_EQ(sphereHull.GetProperties().volume,
                  sphere.GetProperties().volume);
}

TEST(Hull, FailingTest1) {
  // 39202.stl
  const std::vector<vec3> hullPts = {
      {-24.983196259f, -43.272167206f, 52.710712433f},
      {-25.0f, -12.7726717f, 49.907142639f},
      {-23.016393661f, 39.865562439f, 79.083930969f},
      {-24.983196259f, -40.272167206f, 52.710712433f},
      {-4.5177311897f, -28.633184433f, 50.405872345f},
      {11.176083565f, -22.357545853f, 45.275596619f},
      {-25.0f, 21.885698318f, 49.907142639f},
      {-17.633232117f, -17.341972351f, 89.96282196f},
      {26.922552109f, 10.344738007f, 57.146999359f},
      {-24.949174881f, 1.5f, 54.598075867f},
      {9.2058267593f, -23.47851944f, 55.334011078f},
      {13.26748085f, -19.979951859f, 28.117856979f},
      {-18.286884308f, 31.673814774f, 2.1749999523f},
      {18.419618607f, -18.215343475f, 52.450099945f},
      {-24.983196259f, 43.272167206f, 52.710712433f},
      {-1.6232370138f, -29.794223785f, 48.394889832f},
      {49.865573883f, -0.0f, 55.507141113f},
      {-18.627283096f, -39.544368744f, 55.507141113f},
      {-20.442623138f, -35.407661438f, 8.2749996185f},
      {10.229375839f, -14.717799187f, 10.508025169f}};
  auto hull = Manifold::Hull(hullPts);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportMesh("failing_test1.glb", hull.GetMeshGL(), {});
  }
#endif
  EXPECT_TRUE(isMeshConvex(hull, 1.09375e-05));
}

TEST(Hull, FailingTest2) {
  // 1750623.stl
  const std::vector<vec3> hullPts = {
      {174.17001343f, -12.022000313f, 29.562002182f},
      {174.51400757f, -10.858000755f, -3.3340001106f},
      {187.50801086f, 22.826000214f, 23.486001968f},
      {172.42800903f, 12.018000603f, 28.120000839f},
      {180.98001099f, -26.866001129f, 6.9100003242f},
      {172.42800903f, -12.022000313f, 28.120000839f},
      {174.17001343f, 19.498001099f, 29.562002182f},
      {213.96600342f, 2.9400000572f, -11.100000381f},
      {182.53001404f, -22.49200058f, 23.644001007f},
      {175.89401245f, 19.900001526f, 16.118000031f},
      {211.38601685f, 3.0200002193f, -14.250000954f},
      {183.7440033f, 12.018000603f, 18.090000153f},
      {210.51000977f, 2.5040001869f, -11.100000381f},
      {204.13601685f, 34.724002838f, -11.250000954f},
      {193.23400879f, -24.704000473f, 17.768001556f},
      {171.62800598f, -19.502000809f, 27.320001602f},
      {189.67401123f, 8.486000061f, -5.4080004692f},
      {193.23800659f, 24.704000473f, 17.758001328f},
      {165.36801147f, -6.5600004196f, -14.250000954f},
      {174.17001343f, -19.502000809f, 29.562002182f},
      {190.06401062f, -0.81000006199f, -14.250000954f}};
  auto hull = Manifold::Hull(hullPts);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportMesh("failing_test2.glb", hull.GetMeshGL(), {});
  }
#endif
  EXPECT_TRUE(isMeshConvex(hull, 2.13966e-05));
}

TEST(Hull, DisabledFaceTest) {
  // 101213.stl
  const std::vector<vec3> hullPts = {
      {65.398902893, 58.303115845, 58.765388489},
      {42.147319794, 44.512584686, 75.703102112},
      {89.208251953, 97.092460632, 41.632453918},
      {69.860748291, 69.860748291, 56.492958069},
      {45.375354767, 39.067985535, 64.844772339},
      {26.555616379, 18.671405792, 81.067504883},
      {88.179382324, 81.083595276, 43.981628418},
      {51.823883057, 50.247039795, 70.359062195},
      {58.489616394, 72.681190491, 51.274829865},
      {110, 10, 65},
      {29.590316772, 20.917686462, 73.143547058},
      {101.61526489, 98.461585999, 30.909877777}};
  auto hull = Manifold::Hull(hullPts);
#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportMesh("disabledFaceTest.glb", hull.GetMeshGL(), {});
  }
#endif
  EXPECT_TRUE(!hull.IsEmpty());
  EXPECT_TRUE(isMeshConvex(hull));
}

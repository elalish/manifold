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

#include "gtest/gtest.h"
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

  // Iterate over each triangle
  for (size_t t = 0; t < numTri; ++t) {
    // Get the vertices of the triangle
    auto tri = mesh.GetTriVerts(t);
    vec3 v0 = mesh.GetVertPos(tri[0]);
    vec3 v1 = mesh.GetVertPos(tri[1]);
    vec3 v2 = mesh.GetVertPos(tri[2]);

    // Compute the normal of the triangle
    vec3 normal = la::normalize(la::cross(v1 - v0, v2 - v0));

    // Check all other vertices
    for (size_t i = 0; i < numVert; ++i) {
      if (i == tri[0] || i == tri[1] || i == tri[2])
        continue;  // Skip vertices of the current triangle

      // Get the vertex
      vec3 v = mesh.GetVertPos(i);

      // Compute the signed distance from the plane
      double distance = la::dot(normal, v - v0);

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
  const auto sphere = Manifold::Sphere(tictacRad, {tictacSeg});
  const std::vector<Manifold> spheres{sphere,
                                      sphere.Translate({0, 0, tictacMid})};
  const auto tictac = Manifold::Hull(spheres);

  if (options.exportModels) WriteTestOBJ("tictac_hull.obj", tictac);

  EXPECT_NEAR(sphere.NumVert() + tictacSeg, tictac.NumVert(), 1);
}

TEST(Hull, Hollow) {
  auto sphere = Manifold::Sphere(100, {360});
  auto hollow = sphere - sphere.Scale({0.8, 0.8, 0.8});
  const double sphere_vol = sphere.Volume();
  EXPECT_FLOAT_EQ(hollow.Hull().Volume(), sphere_vol);
}

TEST(Hull, Cube) {
  std::vector<vec3> cubePts = {
      {0, 0, 0},       {1, 0, 0},   {0, 1, 0},      {0, 0, 1},  // corners
      {1, 1, 0},       {0, 1, 1},   {1, 0, 1},      {1, 1, 1},  // corners
      {0.5, 0.5, 0.5}, {0.5, 0, 0}, {0.5, 0.7, 0.2}  // internal points
  };
  auto cube = Manifold::Hull(cubePts);
  EXPECT_FLOAT_EQ(cube.Volume(), 1);
}

TEST(Hull, Empty) {
  const std::vector<vec3> tooFew{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  EXPECT_TRUE(Manifold::Hull(tooFew).Simplify().IsEmpty());

  const std::vector<vec3> coplanar{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
  EXPECT_TRUE(Manifold::Hull(coplanar).Simplify().IsEmpty());
}

TEST(Hull, MengerSponge) {
  Manifold sponge = MengerSponge(4);
  sponge = sponge.Rotate(10, 20, 30);
  Manifold spongeHull = sponge.Hull();
  EXPECT_EQ(spongeHull.NumTri(), 12);
  EXPECT_FLOAT_EQ(spongeHull.SurfaceArea(), 6);
  EXPECT_FLOAT_EQ(spongeHull.Volume(), 1);
}

TEST(Hull, Sphere) {
  Manifold sphere = Manifold::Sphere(1, {1500});
  sphere = sphere.Translate(vec3(0.5));
  Manifold sphereHull = sphere.Hull();
  EXPECT_EQ(sphereHull.NumTri(), sphere.NumTri());
  EXPECT_FLOAT_EQ(sphereHull.Volume(), sphere.Volume());
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
  if (options.exportModels) WriteTestOBJ("failing_test1.obj", hull);
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
  if (options.exportModels) WriteTestOBJ("failing_test2.obj", hull);
  EXPECT_TRUE(isMeshConvex(hull, 2.13966e-05));
}

TEST(Hull, CoplanarClusters) {
  // A convex polygon swept by an octahedron (a Minkowski sum): every corner
  // of the polygon shifted by each octahedron vertex. Large exactly coplanar
  // clusters; the visibility test used to remove faces the new point merely
  // lay on, leaving inward-facing faces and points far outside the hull.
  const std::vector<vec3> hullPts = {
      {2.2965078353881836, 1.8417460918426514, 2.041705846786499},
      {2.3199396133422852, 1.8417460918426514, 2.041705846786499},
      {2.3082237243652344, 1.8300302028656006, 2.041705846786499},
      {2.3082237243652344, 1.8534619808197021, 2.041705846786499},
      {2.3082237243652344, 1.8417460918426514, 2.0299899578094482},
      {2.3082237243652344, 1.8417460918426514, 2.0534217357635498},
      {2.2965078353881836, 1.7583343982696533, 2.041705846786499},
      {2.3199396133422852, 1.7583343982696533, 2.041705846786499},
      {2.3082237243652344, 1.7466185092926025, 2.041705846786499},
      {2.3082237243652344, 1.7700502872467041, 2.041705846786499},
      {2.3082237243652344, 1.7583343982696533, 2.0299899578094482},
      {2.3082237243652344, 1.7583343982696533, 2.0534217357635498},
      {2.8965883255004883, 1.1582539081573486, 2.6417863368988037},
      {2.9200201034545898, 1.1582539081573486, 2.6417863368988037},
      {2.9083042144775391, 1.1465380191802979, 2.6417863368988037},
      {2.9083042144775391, 1.1699697971343994, 2.6417863368988037},
      {2.9083042144775391, 1.1582539081573486, 2.6300704479217529},
      {2.9083042144775391, 1.1582539081573486, 2.6535022258758545},
      {2.9465782642364502, 1.1582539081573486, 2.6917762756347656},
      {2.9700100421905518, 1.1582539081573486, 2.6917762756347656},
      {2.958294153213501, 1.1465380191802979, 2.6917762756347656},
      {2.958294153213501, 1.1699697971343994, 2.6917762756347656},
      {2.958294153213501, 1.1582539081573486, 2.6800603866577148},
      {2.958294153213501, 1.1582539081573486, 2.7034921646118164},
      {2.9465782642364502, 1.2416656017303467, 2.6917762756347656},
      {2.9700100421905518, 1.2416656017303467, 2.6917762756347656},
      {2.958294153213501, 1.2299497127532959, 2.6917762756347656},
      {2.958294153213501, 1.2533814907073975, 2.6917762756347656},
      {2.958294153213501, 1.2416656017303467, 2.6800603866577148},
      {2.958294153213501, 1.2416656017303467, 2.7034921646118164},
      {2.3464977741241455, 1.8417460918426514, 2.0916957855224609},
      {2.3699295520782471, 1.8417460918426514, 2.0916957855224609},
      {2.3582136631011963, 1.8300302028656006, 2.0916957855224609},
      {2.3582136631011963, 1.8534619808197021, 2.0916957855224609},
      {2.3582136631011963, 1.8417460918426514, 2.0799798965454102},
      {2.3582136631011963, 1.8417460918426514, 2.1034116744995117},
  };
  auto hull = Manifold::Hull(hullPts);
  if (options.exportModels) WriteTestOBJ("coplanar_clusters.obj", hull);
  EXPECT_TRUE(isMeshConvex(hull));
  EXPECT_NEAR(hull.Volume(), 0.002259329608, 1e-12);
}

TEST(Hull, NearCoplanarFold) {
  // A point within epsilon above a face, on the inner side of one of its edges.
  // Keeping that face folded the new triangle back over it.
  const std::vector<vec3> hullPts = {
      {-0x1.4ee12a11cf174p-4, -0x1.54f3a76ac6401p-2, -0x1.f000bf2f2f9c5p-2},
      {-0x1.4ecbc87cb1179p-4, 0x1.8a96e99eda124p-4, -0x1.f923c4e27e38ep-2},
      {-0x1.4eaef28b4d685p-4, -0x1.5bed770629debp-2, -0x1.f0150e32d645bp-2},
      {-0x1.4e64d1230c345p-4, -0x1.817ceb61c0841p-2, -0x1.e1122d52d828fp-2},
      {-0x1.4dcc517664b17p-4, 0x1.e10bfaa7998c1p-2, -0x1.965107a45f08ap-2},
      {-0x1.4ca0401984232p-4, 0x1.e3c2aab017514p-2, -0x1.986dae5e5ec5dp-2},
      {-0x1.49d8282619b71p-4, -0x1.e3c2a684ddd9p-2, -0x1.b6010f66d62d4p-2},
      {-0x1.2e84095d867bdp-4, -0x1.51613a8b9f827p-2, -0x1.f27bc3c913456p-2},
      {-0x1.293ffb61e9c6p-4, -0x1.5b782bd003b8ap-2, -0x1.f033cf476fe1fp-2},
      {-0x1.0037fd87c2667p-4, 0x1.502742c3321cbp-4, -0x1.fcdcdb7c2f6d1p-2},
      {-0x1.001ede75832b6p-4, -0x1.5557e99548e9ap-2, -0x1.f28d7cb16ed5bp-2},
      {-0x1.fcccc7cf1850bp-5, 0x1.57e4fe6dd25ebp-4, -0x1.fce925a53b944p-2},
      {-0x1.f5c599f6874e3p-5, -0x1.54bb60a44277ep-2, -0x1.f2c912dd66f3ep-2},
      {-0x1.a78efeecd05dfp-5, 0x1.db88f2caacf16p-2, 0x1.fffffbd4c8dbp-2},
      {-0x1.a0d005257579cp-5, 0x1.58264c6396bd7p-4, -0x1.fb9b79efef2bep-2},
      {-0x1.976edc1180893p-5, -0x1.e1a6dce8e74f1p-2, -0x1.b9a4efc10f291p-2},
      {-0x1.93f158cb2a18fp-5, -0x1.8241e0257232ep-2, -0x1.e0beff162d33dp-2},
      {0x1.3e3ace9f2b869p-8, 0x1.8125132b8e86fp-2, -0x1.b2d261ffff0e5p-2},
      {0x1.507a39ae251c8p-8, -0x1.4298ab00e9de6p-2, -0x1.f980149e8ee6cp-2},
      {0x1.5ff810094da5p-8, 0x1.1b740b120c501p-4, -0x1.0000024e80a4bp-1},
      {0x1.0e5a7ed137cadp-4, -0x1.b9ba3be96352fp-2, 0x1.dbcc72343909cp-2},
      {0x1.0f38c9fb6f6dep-4, -0x1.b5bebab8a8276p-2, -0x1.c1f8dd4d0920bp-2},
      {0x1.4d56820f1e318p-4, 0x1.58b2b09fd8e31p-2, 0x1.f0251f71aa1acp-2},
      {0x1.4d65df1096f0cp-4, 0x1.537ea666762f1p-2, -0x1.ad33618cf4ca3p-2},
      {0x1.4ecc7d6a254d2p-4, -0x1.98fb1d84ef894p-2, -0x1.ccba4c60d3242p-2},
  };
  auto hull = Manifold::Hull(hullPts);
  if (options.exportModels) WriteTestOBJ("near_coplanar_fold.obj", hull);
  EXPECT_TRUE(isMeshConvex(hull));
}

TEST(Hull, NearCoplanarFoldSecondEdge) {
  // As above, but the face is first reached across an edge that does not fold.
  const std::vector<vec3> hullPts = {
      {-0x1.6102379d5c083p-6, -0x1.17bc079edcf4dp-2, 0x1.3a88a7214d888p-6},
      {-0x1p-4, -0x1.db5c49ccbb67p-5, 0x1.a86c03926efa8p-2},
      {-0x1p-4, 0x1.cd60574187958p-2, -0x1.18cdf12a9c652p-2},
      {-0x1.fdc66576cdf94p-5, 0x1.ce0a03a128b8p-2, -0x1.18934d8ff79aap-2},
      {0x1.00000bd4c86ecp-4, 0x1.c3b46259b547cp-4, -0x1.3607eebf9fffp-2},
      {0x1.5555270cfcdabp-5, -0x1.95d22eea3f368p-2, -0x1.6d5f91501d8e7p-3},
      {-0x1p-4, -0x1.e8ff2e59ce758p-4, -0x1.5f4a547baa548p-5},
      {-0x1.59ae538e35d69p-5, -0x1.1161b1036d90fp-3, -0x1.ca11ae796f81cp-2},
      {-0x1.00000682d34bap-4, -0x1.f74ca59a66c38p-3, -0x1.51df270a377c9p-6},
      {-0x1.00000b3e6363cp-4, -0x1.fc0a8d7c000fdp-3, 0x1.ef7b3552e6021p-3},
      {0x1.589d96a27ca94p-6, 0x1.5173c6cea866p-2, 0x1.b59bc7598479cp-2},
  };
  auto hull = Manifold::Hull(hullPts);
  if (options.exportModels) WriteTestOBJ("near_coplanar_fold_edge.obj", hull);
  EXPECT_TRUE(isMeshConvex(hull));
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
  if (options.exportModels) WriteTestOBJ("disabledFaceTest.obj", hull);
  EXPECT_TRUE(!hull.IsEmpty());
  EXPECT_TRUE(isMeshConvex(hull));
}

TEST(Hull, Degenerate2D) {
  // issue 1491
  // note that we need 5 points to trigger this bug
  Manifold hull = Manifold::Hull({
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 1.0},
      {0.5, 0.0, 0.0},
      {0.5, 0.0, 0.0},
      {0.5, 0.0, 1.0},
  });
  EXPECT_TRUE(hull.IsEmpty());
}

TEST(Hull, Degenerate1D) {
  Manifold hull = Manifold::Hull({
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
      {0.5, 0.0, 0.0},
      {0.5, 0.0, 0.0},
      {0.5, 0.0, 0.0},
  });
  EXPECT_TRUE(hull.IsEmpty());
}

TEST(Hull, NotEnoughPoints) {
  Manifold hull = Manifold::Hull({
      {0.0, 0.0, 0.0},
      {0.5, 0.0, 0.0},
  });
  EXPECT_TRUE(hull.IsEmpty());
}

TEST(Hull, EmptyHull) {
  Manifold hull = Manifold::Hull(std::vector<vec3>());
  EXPECT_TRUE(hull.IsEmpty());
}

TEST(Hull, CoarseDeterministic) {
  // Coarse precision makes RemoveDegenerates collapse edges, which depends on
  // the halfedge order QuickHull produces. The hull needs enough halfedges to
  // run in parallel.
  for (const double precision : {1e-3, 1e-2}) {
    const Manifold s = Manifold::Sphere(0.02, {96}).WithPrecision(precision);
    std::vector<Manifold> parts;
    for (const vec3 v : {vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0)})
      parts.push_back(s.Translate(v));
    const MeshGL64 first = Manifold::Hull(parts).GetMeshGL64();
    for (int i = 0; i < 5; ++i) {
      const MeshGL64 mesh = Manifold::Hull(parts).GetMeshGL64();
      EXPECT_EQ(mesh.vertProperties, first.vertProperties) << precision;
      EXPECT_EQ(mesh.triVerts, first.triVerts) << precision;
    }
  }
}

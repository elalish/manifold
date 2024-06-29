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

#include "cross_section.h"
#include "manifold.h"
#include "samples.h"
#include "sdf.h"
#include "test.h"
#include "tri_dist.h"

using namespace manifold;

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

#ifdef MANIFOLD_EXPORT
TEST(Manifold, HullFail) {
  Manifold body = ReadMesh("hull-body.glb");
  Manifold mask = ReadMesh("hull-mask.glb");
  Manifold ret = body - mask;
  MeshGL mesh = ret.GetMesh();
}
#endif

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

TEST(Manifold, VHACDHULL) {
  const float tictacRad = 100;
  const float tictacHeight = 500;
  const int tictacSeg = 100;
  const float tictacMid = tictacHeight - 2 * tictacRad;
  const auto sphere = Manifold::Sphere(tictacRad, tictacSeg);
  const std::vector<Manifold> spheres{sphere,
                                      sphere.Translate({0, 0, tictacMid})};
  const auto tictac = Manifold::Hull2(spheres);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportMesh("tictac_hull2.glb", tictac.GetMesh(), {});
  }
#endif

  EXPECT_EQ(sphere.NumVert() + tictacSeg, tictac.NumVert());
}

TEST(Manifold, HollowHull2) {
  auto sphere = Manifold::Sphere(100, 360);
  auto hollow = sphere - sphere.Scale({0.8, 0.8, 0.8});
  const float sphere_vol = sphere.GetProperties().volume;
  EXPECT_FLOAT_EQ(hollow.Hull2().GetProperties().volume, sphere_vol);
}

TEST(Manifold, CubeHull2) {
  std::vector<glm::vec3> cubePts = {
      {0, 0, 0},       {1, 0, 0},   {0, 1, 0},      {0, 0, 1},  // corners
      {1, 1, 0},       {0, 1, 1},   {1, 0, 1},      {1, 1, 1},  // corners
      {0.5, 0.5, 0.5}, {0.5, 0, 0}, {0.5, 0.7, 0.2}  // internal points
  };
  auto cube = Manifold::Hull2(cubePts);
  EXPECT_FLOAT_EQ(cube.GetProperties().volume, 1);
}

TEST(Manifold, EmptyHull2) {
  const std::vector<glm::vec3> tooFew{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  EXPECT_TRUE(Manifold::Hull2(tooFew).IsEmpty());

  const std::vector<glm::vec3> coplanar{
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
  EXPECT_TRUE(Manifold::Hull2(coplanar).IsEmpty());
}

TEST(Manifold, QUICKHULL2) {
  const float tictacRad = 100;
  const float tictacHeight = 500;
  const int tictacSeg = 50;
  const float tictacMid = tictacHeight - 2 * tictacRad;
  const auto sphere = Manifold::Sphere(tictacRad, tictacSeg);
  const std::vector<Manifold> spheres{sphere,
                                      sphere.Translate({0, 0, tictacMid})};
  const auto tictac = Manifold::Hull3(spheres);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    ExportMesh("tictac_hull3.glb", tictac.GetMesh(), {});
  }
#endif

  EXPECT_NEAR(sphere.NumVert() + tictacSeg, tictac.NumVert(), 3);
}

// TEST(Manifold, HollowHull3) {
//   auto sphere = Manifold::Sphere(100, 100);
//   auto hollow = sphere - sphere.Scale({0.8, 0.8, 0.8});
//   const float sphere_vol = sphere.GetProperties().volume;
//   EXPECT_FLOAT_EQ(hollow.Hull3().GetProperties().volume, sphere_vol);
// }

// TEST(Manifold, CubeHull3) {
//   std::vector<glm::vec3> cubePts = {
//       {0, 0, 0},       {1, 0, 0},   {0, 1, 0},      {0, 0, 1},  // corners
//       {1, 1, 0},       {0, 1, 1},   {1, 0, 1},      {1, 1, 1},  // corners
//       {0.5, 0.5, 0.5}, {0.5, 0, 0}, {0.5, 0.7, 0.2}  // internal points
//   };
//   auto cube = Manifold::Hull3(cubePts);
//   EXPECT_FLOAT_EQ(cube.GetProperties().volume, 1);
// }

// TEST(Manifold, EmptyHull3) {
//   const std::vector<glm::vec3> tooFew{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
//   EXPECT_TRUE(Manifold::Hull3(tooFew).IsEmpty());

//   const std::vector<glm::vec3> coplanar{
//       {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
//   EXPECT_TRUE(Manifold::Hull3(coplanar).IsEmpty());
// }

TEST(Manifold, TICTACHULL4) {
  const float tictacRad = 100;
  const float tictacHeight = 500;
  const int tictacSeg = 50;
  const float tictacMid = tictacHeight - 2 * tictacRad;
  const auto sphere = Manifold::Sphere(tictacRad, tictacSeg);
  const std::vector<Manifold> spheres{sphere,
                                      sphere.Translate({0, 0, tictacMid})};
  const auto tictac = Manifold::Hull4(spheres);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    // std::cout << "Ok" << std::endl;
    ExportMesh("tictac_hull4.glb", tictac.GetMesh(), {});
  }
#endif

  EXPECT_NEAR(sphere.NumVert() + tictacSeg, tictac.NumVert(), 3);
}

TEST(Manifold, HollowHull4) {
  auto sphere = Manifold::Sphere(100, 100);
  auto hollow = sphere - sphere.Scale({0.8, 0.8, 0.8});
  const float sphere_vol = sphere.GetProperties().volume;
  EXPECT_FLOAT_EQ(hollow.Hull4().GetProperties().volume, sphere_vol);
}

TEST(Manifold, CubeHull4) {
  std::vector<glm::vec3> cubePts = {
      {0, 0, 0},       {1, 0, 0},   {0, 1, 0},      {0, 0, 1},  // corners
      {1, 1, 0},       {0, 1, 1},   {1, 0, 1},      {1, 1, 1},  // corners
      {0.5, 0.5, 0.5}, {0.5, 0, 0}, {0.5, 0.7, 0.2}  // internal points
  };
  auto cube = Manifold::Hull4(cubePts);
  EXPECT_FLOAT_EQ(cube.GetProperties().volume, 1);
}

TEST(Manifold, EmptyHull4) {
  const std::vector<glm::vec3> tooFew{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  EXPECT_TRUE(Manifold::Hull4(tooFew).IsEmpty());

  const std::vector<glm::vec3> coplanar{
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
  EXPECT_TRUE(Manifold::Hull4(coplanar).IsEmpty());
}

TEST(Manifold, TICTACHULL5) {
  const float tictacRad = 100;
  const float tictacHeight = 500;
  const int tictacSeg = 50;
  const float tictacMid = tictacHeight - 2 * tictacRad;
  const auto sphere = Manifold::Sphere(tictacRad, tictacSeg);
  const std::vector<Manifold> spheres{sphere,
                                      sphere.Translate({0, 0, tictacMid})};
  const auto tictac = Manifold::Hull5(spheres);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    // std::cout << "Ok" << std::endl;
    ExportMesh("tictac_hull5.glb", tictac.GetMesh(), {});
  }
#endif

  EXPECT_NEAR(sphere.NumVert() + tictacSeg, tictac.NumVert(), 3);
}

TEST(Manifold, HollowHull5) {
  auto sphere = Manifold::Sphere(100, 100);
  auto hollow = sphere - sphere.Scale({0.8, 0.8, 0.8});
  const float sphere_vol = sphere.GetProperties().volume;
  EXPECT_FLOAT_EQ(hollow.Hull5().GetProperties().volume, sphere_vol);
}

TEST(Manifold, CubeHull5) {
  std::vector<glm::vec3> cubePts = {
      {0, 0, 0},       {1, 0, 0},   {0, 1, 0},      {0, 0, 1},  // corners
      {1, 1, 0},       {0, 1, 1},   {1, 0, 1},      {1, 1, 1},  // corners
      {0.5, 0.5, 0.5}, {0.5, 0, 0}, {0.5, 0.7, 0.2}  // internal points
  };
  auto cube = Manifold::Hull5(cubePts);
  EXPECT_FLOAT_EQ(cube.GetProperties().volume, 1);
}

TEST(Manifold, EmptyHull5) {
  const std::vector<glm::vec3> tooFew{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  EXPECT_TRUE(Manifold::Hull5(tooFew).IsEmpty());

  const std::vector<glm::vec3> coplanar{
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
  EXPECT_TRUE(Manifold::Hull5(coplanar).IsEmpty());
}

TEST(Manifold, TICTACHULL6) {
  const float tictacRad = 100;
  const float tictacHeight = 500;
  const int tictacSeg = 50;
  const float tictacMid = tictacHeight - 2 * tictacRad;
  const auto sphere = Manifold::Sphere(tictacRad, tictacSeg);
  const std::vector<Manifold> spheres{sphere,
                                      sphere.Translate({0, 0, tictacMid})};
  const auto tictac = Manifold::Hull6(spheres);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels) {
    // std::cout << "Ok" << std::endl;
    ExportMesh("tictac_hull6.glb", tictac.GetMesh(), {});
  }
#endif

  EXPECT_NEAR(sphere.NumVert() + tictacSeg, tictac.NumVert(), 3);
}

TEST(Manifold, HollowHull6) {
  auto sphere = Manifold::Sphere(100, 100);
  auto hollow = sphere - sphere.Scale({0.8, 0.8, 0.8});
  const float sphere_vol = sphere.GetProperties().volume;
  EXPECT_FLOAT_EQ(hollow.Hull6().GetProperties().volume, sphere_vol);
}

TEST(Manifold, CubeHull6) {
  std::vector<glm::vec3> cubePts = {
      {0, 0, 0},       {1, 0, 0},   {0, 1, 0},      {0, 0, 1},  // corners
      {1, 1, 0},       {0, 1, 1},   {1, 0, 1},      {1, 1, 1},  // corners
      {0.5, 0.5, 0.5}, {0.5, 0, 0}, {0.5, 0.7, 0.2}  // internal points
  };
  auto cube = Manifold::Hull6(cubePts);
  EXPECT_FLOAT_EQ(cube.GetProperties().volume, 1);
}

TEST(Manifold, EmptyHull6) {
  const std::vector<glm::vec3> tooFew{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  EXPECT_TRUE(Manifold::Hull6(tooFew).IsEmpty());

  const std::vector<glm::vec3> coplanar{
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
  EXPECT_TRUE(Manifold::Hull6(coplanar).IsEmpty());
}

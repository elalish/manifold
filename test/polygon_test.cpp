// Copyright 2019 Emmett Lalish
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

#include "polygon.h"
#include <algorithm>
#include <random>
#include "gtest/gtest.h"

constexpr bool kVerbose = false;

namespace {

using namespace manifold;

void StandardizePoly(SimplePolygon &p) {
  auto start = std::min_element(
      p.begin(), p.end(),
      [](const PolyVert &v1, const PolyVert &v2) { return v1.idx < v2.idx; });
  std::rotate(p.begin(), start, p.end());
}

void StandardizePolys(Polygons &polys) {
  for (auto &p : polys) StandardizePoly(p);
  std::sort(polys.begin(), polys.end(),
            [](SimplePolygon &p1, SimplePolygon &p2) {
              return p1[0].idx < p2[0].idx;
            });
}

void Identical(Polygons p1, Polygons p2) {
  ASSERT_EQ(p1.size(), p2.size());
  StandardizePolys(p1);
  StandardizePolys(p2);
  for (int i = 0; i < p1.size(); ++i) {
    ASSERT_EQ(p1[i].size(), p2[i].size());
    for (int j = 0; j < p1[i].size(); ++j) {
      ASSERT_EQ(p1[i][j].idx, p2[i][j].idx);
    }
  }
}

void TestAssemble(const Polygons &polys) {
  std::vector<EdgeVerts> edges = Polygons2Edges(polys);
  std::mt19937 g;
  std::shuffle(edges.begin(), edges.end(), g);
  Polygons polys_out = Assemble(edges);
  Identical(polys, polys_out);
}

void TestPoly(const Polygons &polys, int expectedNumTri) {
  TestAssemble(polys);

  std::vector<glm::ivec3> triangles = BackupTriangulate(polys);
  if (kVerbose)
    for (auto tri : triangles) {
      std::cout << tri.x << ", " << tri.y << ", " << tri.z << std::endl;
    }
  CheckManifold(triangles, polys);

  triangles = PrimaryTriangulate(polys);
  if (kVerbose)
    for (auto tri : triangles) {
      std::cout << tri.x << ", " << tri.y << ", " << tri.z << std::endl;
    }
  CheckManifold(triangles, polys);
  ASSERT_EQ(triangles.size(), expectedNumTri);
}
}  // namespace

TEST(Polygon, NoAssemble) {
  std::vector<EdgeVerts> edges;
  edges.push_back({0, 2});
  edges.push_back({7, 0});
  edges.push_back({9, 7});
  ASSERT_THROW(Assemble(edges), runtimeErr);
}

TEST(Polygon, SimpleHole) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(0, -2), 11, Edge::kNoIdx},  //
      {glm::vec2(2, 2), 21, Edge::kNoIdx},   //
      {glm::vec2(0, 4), 47, Edge::kNoIdx},   //
      {glm::vec2(-3, 3), 14, Edge::kNoIdx},  //
  });
  polys.push_back({
      {glm::vec2(0, -1), 5, Edge::kNoIdx},  //
      {glm::vec2(-1, 1), 8, Edge::kNoIdx},  //
      {glm::vec2(1, 1), 3, Edge::kNoIdx},   //
  });
  TestPoly(polys, 7);
}

TEST(Polygon, SimpleHole2) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(0, 1.63299), 0, Edge::kNoIdx},           //
      {glm::vec2(-1.41421, -0.816496), 1, Edge::kNoIdx},  //
      {glm::vec2(1.41421, -0.816496), 2, Edge::kNoIdx},   //
  });
  polys.push_back({
      {glm::vec2(0, 1.02062), 3, Edge::kNoIdx},           //
      {glm::vec2(0.883883, -0.51031), 4, Edge::kNoIdx},   //
      {glm::vec2(-0.883883, -0.51031), 5, Edge::kNoIdx},  //
  });
  TestPoly(polys, 6);
}

TEST(Polygon, MultiMerge) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-7, 0), 5, Edge::kNoIdx},    //
      {glm::vec2(-6, 3), 4, Edge::kNoIdx},    //
      {glm::vec2(-5, 1), 7, Edge::kNoIdx},    //
      {glm::vec2(-4, 6), 1, Edge::kNoIdx},    //
      {glm::vec2(-3, 2), 2, Edge::kNoIdx},    //
      {glm::vec2(-2, 5), 9, Edge::kNoIdx},    //
      {glm::vec2(-1, 4), 3, Edge::kNoIdx},    //
      {glm::vec2(0, 12), 6, Edge::kNoIdx},    //
      {glm::vec2(-6, 10), 12, Edge::kNoIdx},  //
      {glm::vec2(-8, 11), 18, Edge::kNoIdx},  //
  });
  polys.push_back({
      {glm::vec2(-5, 7), 11, Edge::kNoIdx},  //
      {glm::vec2(-6, 8), 21, Edge::kNoIdx},  //
      {glm::vec2(-5, 9), 47, Edge::kNoIdx},  //
  });
  TestPoly(polys, 13);
}

TEST(Polygon, Colinear) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-5.48368, -3.73905), 0, Edge::kNoIdx},   //
      {glm::vec2(-4.9881, -4.51552), 1, Edge::kNoIdx},    //
      {glm::vec2(-4.78988, -4.13186), 2, Edge::kNoIdx},   //
      {glm::vec2(-4.82012, -4.13999), 3, Edge::kNoIdx},   //
      {glm::vec2(-4.84314, -4.14617), 4, Edge::kNoIdx},   //
      {glm::vec2(-4.85738, -4.13581), 5, Edge::kNoIdx},   //
      {glm::vec2(-4.86772, -4.12831), 6, Edge::kNoIdx},   //
      {glm::vec2(-4.87337, -4.12422), 7, Edge::kNoIdx},   //
      {glm::vec2(-4.88097, -4.1187), 8, Edge::kNoIdx},    //
      {glm::vec2(-4.89799, -4.10634), 9, Edge::kNoIdx},   //
      {glm::vec2(-4.90219, -4.10329), 10, Edge::kNoIdx},  //
      {glm::vec2(-4.90826, -4.09887), 11, Edge::kNoIdx},  //
      {glm::vec2(-4.90846, -4.09873), 12, Edge::kNoIdx},  //
      {glm::vec2(-4.91227, -4.09597), 13, Edge::kNoIdx},  //
      {glm::vec2(-4.92199, -4.0889), 14, Edge::kNoIdx},   //
      {glm::vec2(-5.0245, -4.01443), 15, Edge::kNoIdx},   //
      {glm::vec2(-5.02494, -4.01412), 16, Edge::kNoIdx},  //
      {glm::vec2(-5.02536, -4.01381), 17, Edge::kNoIdx},  //
      {glm::vec2(-5.0316, -4.00927), 18, Edge::kNoIdx},   //
      {glm::vec2(-5.03211, -4.00891), 19, Edge::kNoIdx},  //
      {glm::vec2(-5.05197, -3.99448), 20, Edge::kNoIdx},  //
      {glm::vec2(-5.14757, -3.92504), 21, Edge::kNoIdx},  //
      {glm::vec2(-5.21287, -3.8776), 22, Edge::kNoIdx},   //
      {glm::vec2(-5.29419, -3.81853), 23, Edge::kNoIdx},  //
      {glm::vec2(-5.29907, -3.81499), 24, Edge::kNoIdx},  //
      {glm::vec2(-5.36732, -3.76541), 25, Edge::kNoIdx},  //
  });
  TestPoly(polys, 24);
}

TEST(Polygon, Merges) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-3.22039, 10.2769), 0, Edge::kNoIdx},   //
      {glm::vec2(-3.12437, 10.4147), 1, Edge::kNoIdx},   //
      {glm::vec2(-3.99093, 10.1781), 2, Edge::kNoIdx},   //
      {glm::vec2(-3.8154, 10.0716), 3, Edge::kNoIdx},    //
      {glm::vec2(-3.78982, 10.0893), 4, Edge::kNoIdx},   //
      {glm::vec2(-3.55033, 10.2558), 5, Edge::kNoIdx},   //
      {glm::vec2(-3.50073, 10.2549), 6, Edge::kNoIdx},   //
      {glm::vec2(-3.47018, 10.2572), 7, Edge::kNoIdx},   //
      {glm::vec2(-3.42633, 10.2605), 8, Edge::kNoIdx},   //
      {glm::vec2(-3.34311, 10.2604), 9, Edge::kNoIdx},   //
      {glm::vec2(-3.32096, 10.2633), 10, Edge::kNoIdx},  //
  });
  TestPoly(polys, 9);
}

TEST(Polygon, ColinearY) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(0, 0), 0, Edge::kNoIdx},   //
      {glm::vec2(-1, 1), 5, Edge::kNoIdx},  //
      {glm::vec2(-2, 1), 2, Edge::kNoIdx},  //
      {glm::vec2(-3, 1), 1, Edge::kNoIdx},  //
      {glm::vec2(-4, 1), 7, Edge::kNoIdx},  //
      {glm::vec2(-4, 2), 8, Edge::kNoIdx},  //
      {glm::vec2(-3, 2), 3, Edge::kNoIdx},  //
      {glm::vec2(-2, 2), 9, Edge::kNoIdx},  //
      {glm::vec2(-1, 2), 4, Edge::kNoIdx},  //
      {glm::vec2(0, 3), 10, Edge::kNoIdx},  //
      {glm::vec2(1, 2), 11, Edge::kNoIdx},  //
      {glm::vec2(2, 2), 12, Edge::kNoIdx},  //
      {glm::vec2(3, 2), 14, Edge::kNoIdx},  //
      {glm::vec2(4, 2), 13, Edge::kNoIdx},  //
      {glm::vec2(4, 1), 15, Edge::kNoIdx},  //
      {glm::vec2(3, 1), 18, Edge::kNoIdx},  //
      {glm::vec2(2, 1), 19, Edge::kNoIdx},  //
      {glm::vec2(1, 1), 16, Edge::kNoIdx},  //
  });
  TestPoly(polys, 16);
}

TEST(Polygon, Inverted) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(0, 2.04124), 0, Edge::kNoIdx},           //
      {glm::vec2(-1.41421, -0.408248), 1, Edge::kNoIdx},  //
      {glm::vec2(-1.23744, -0.408248), 5, Edge::kNoIdx},  //
      {glm::vec2(0, 1.73506), 9, Edge::kNoIdx},           //
      {glm::vec2(1.23744, -0.408248), 7, Edge::kNoIdx},   //
      {glm::vec2(1.41421, -0.408248), 2, Edge::kNoIdx},   //
  });
  polys.push_back({
      {glm::vec2(-1.06066, -0.408248), 4, Edge::kNoIdx},  //
      {glm::vec2(0, 1.42887), 8, Edge::kNoIdx},           //
      {glm::vec2(1.06066, -0.408248), 6, Edge::kNoIdx},   //
  });
  TestPoly(polys, 5);
}

TEST(Polygons, DISABLED_Ugly) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(0.550049, -0.484235), 0, Edge::kNoIdx},      //
      {glm::vec2(0.411479, -0.20602), 165, Edge::kNoIdx},     //
      {glm::vec2(0.515815, -0.205784), 1067, Edge::kNoIdx},   //
      {glm::vec2(0.548218, -0.172791), 283, Edge::kNoIdx},    //
      {glm::vec2(0.547651, -0.0762587), 280, Edge::kNoIdx},   //
      {glm::vec2(0.358672, -0.0999957), 164, Edge::kNoIdx},   //
      {glm::vec2(0.3837, -0.150245), 169, Edge::kNoIdx},      //
      {glm::vec2(0.505506, -0.154427), 1072, Edge::kNoIdx},   //
      {glm::vec2(0.546988, 0.0365588), 284, Edge::kNoIdx},    //
      {glm::vec2(0.546255, 0.161176), 285, Edge::kNoIdx},     //
      {glm::vec2(0.47977, -0.0196254), 1070, Edge::kNoIdx},   //
      {glm::vec2(0.492479, 0.0576864), 1071, Edge::kNoIdx},   //
      {glm::vec2(0.549446, -0.381719), 286, Edge::kNoIdx},    //
      {glm::vec2(0.547831, -0.106901), 282, Edge::kNoIdx},    //
      {glm::vec2(0.36829, -0.00350715), 1066, Edge::kNoIdx},  //
      {glm::vec2(0.301766, 0.0142589), 166, Edge::kNoIdx},    //
      {glm::vec2(0.205266, 0.208008), 167, Edge::kNoIdx},     //
      {glm::vec2(0.340096, 0.562179), 1068, Edge::kNoIdx},    //
      {glm::vec2(0.255078, 0.464996), 1063, Edge::kNoIdx},    //
      {glm::vec2(0.545534, 0.283844), 278, Edge::kNoIdx},     //
      {glm::vec2(0.543549, 0.621731), 287, Edge::kNoIdx},     //
      {glm::vec2(0.363675, 0.790003), 1073, Edge::kNoIdx},    //
      {glm::vec2(0.102785, 0.413765), 170, Edge::kNoIdx},     //
      {glm::vec2(0.152009, 0.314934), 174, Edge::kNoIdx},     //
      {glm::vec2(0.198766, 0.33883), 1076, Edge::kNoIdx},     //
      {glm::vec2(0.344385, -0.0713115), 173, Edge::kNoIdx},   //
      {glm::vec2(0.261844, 0.0944117), 163, Edge::kNoIdx},    //
      {glm::vec2(0.546909, 0.0499438), 279, Edge::kNoIdx},    //
      {glm::vec2(0.544994, 0.375805), 277, Edge::kNoIdx},     //
      {glm::vec2(0.112526, 0.732868), 1062, Edge::kNoIdx},    //
      {glm::vec2(0.205606, 0.429447), 1064, Edge::kNoIdx},    //
      {glm::vec2(0.0249926, 0.763546), 1065, Edge::kNoIdx},   //
      {glm::vec2(-0.0523676, 0.725275), 162, Edge::kNoIdx},   //
      {glm::vec2(0.124038, 0.371095), 161, Edge::kNoIdx},     //
      {glm::vec2(0.0406817, 0.937559), 1069, Edge::kNoIdx},   //
      {glm::vec2(-0.176136, 0.973774), 168, Edge::kNoIdx},    //
      {glm::vec2(-0.111522, 0.844043), 172, Edge::kNoIdx},    //
      {glm::vec2(0.396158, 0.428476), 1075, Edge::kNoIdx},    //
      {glm::vec2(0.549669, -0.419631), 281, Edge::kNoIdx},    //
  });
  polys.push_back({
      {glm::vec2(-0.102551, 0.826033), 171, Edge::kNoIdx},    //
      {glm::vec2(-0.0320386, 0.876701), 1074, Edge::kNoIdx},  //
      {glm::vec2(0.54275, 0.757552), 288, Edge::kNoIdx},      //
      {glm::vec2(0.542592, 0.784444), 289, Edge::kNoIdx},     //
      {glm::vec2(0.163827, 0.854684), 1077, Edge::kNoIdx},    //
      {glm::vec2(-0.0562895, 0.733149), 175, Edge::kNoIdx},   //
  });
  TestPoly(polys, 5);
}

TEST(Polygon, BadEdges) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(1, -1), 2, 9},    //
      {glm::vec2(1, 1), 3, 11},    //
      {glm::vec2(1, 1), 10, -1},   //
      {glm::vec2(1, -1), 16, -1},  //
      {glm::vec2(1, -1), 15, -1},  //
      {glm::vec2(-1, -1), 9, 11},  //
      {glm::vec2(-1, -1), 6, 10},  //
  });
  std::vector<glm::ivec3> triangles = BackupTriangulate(polys);
  if (kVerbose)
    for (auto tri : triangles) {
      std::cout << tri.x << ", " << tri.y << ", " << tri.z << std::endl;
    }
  CheckManifold(triangles, polys);
}

TEST(Polygon, BadEdges2) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-0.292598, -0.322469), 1, 0},    //
      {glm::vec2(-0.282797, -0.340069), 2, -1},   //
      {glm::vec2(-0.158295, -0.30762), 26, 8},    //
      {glm::vec2(-0.189351, -0.230253), 27, -1},  //
      {glm::vec2(-0.329733, -0.255784), 3, 0},    //
      {glm::vec2(-0.342412, -0.233016), 4, -1},   //
      {glm::vec2(-0.202167, -0.198325), 28, 8},   //
      {glm::vec2(-0.223625, -0.144868), 25, -1},  //
  });
  std::vector<glm::ivec3> triangles = BackupTriangulate(polys);
  if (kVerbose)
    for (auto tri : triangles) {
      std::cout << tri.x << ", " << tri.y << ", " << tri.z << std::endl;
    }
  EXPECT_THROW(CheckManifold(triangles, polys), runtimeErr);
}

// void fnExit() { throw std::runtime_error("Someone called Exit()!"); }

int main(int argc, char **argv) {
  // atexit(fnExit);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
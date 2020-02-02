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

  std::vector<glm::vec2> vertPos;
  for (const auto &poly : polys) {
    for (const PolyVert &vert : poly) {
      if (vert.idx >= vertPos.size()) vertPos.resize(vert.idx + 1);
      vertPos[vert.idx] = vert.pos;
    }
  }
  std::vector<int> vertAssignment(vertPos.size());
  std::iota(vertAssignment.begin(), vertAssignment.end(), 0);

  Polygons polys_out = Assemble(vertAssignment, edges,
                                [&vertPos](int vert) { return vertPos[vert]; });
  Identical(polys, polys_out);
}

void TestPoly(const Polygons &polys, int expectedNumTri,
              bool expectGeometry = true) {
  // PolygonParams().verbose = true;
  PolygonParams().checkGeometry = expectGeometry;
  PolygonParams().intermediateChecks = true;
  TestAssemble(polys);
  std::vector<glm::ivec3> triangles = Triangulate(polys);
  ASSERT_EQ(triangles.size(), expectedNumTri);
}
}  // namespace

TEST(Polygon, NoAssemble) {
  std::vector<EdgeVerts> edges;
  edges.push_back({0, 2});
  edges.push_back({1, 2});
  edges.push_back({0, 1});
  std::vector<int> vertAssignment = {0, 1, 2};
  ASSERT_THROW(
      Assemble(vertAssignment, edges, [](int) { return glm::vec2(0.0 / 0.0); }),
      runtimeErr);
}

/**
 * These polygons are all valid geometry. Some are clearly valid, while many are
 * marginal, but all should produce correct topology and geometry, within
 * tolerance.
 */
TEST(Polygon, SimpleHole) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(0, -2), 0, Edge::kNoIdx},  //
      {glm::vec2(2, 2), 1, Edge::kNoIdx},   //
      {glm::vec2(0, 4), 2, Edge::kNoIdx},   //
      {glm::vec2(-3, 3), 3, Edge::kNoIdx},  //
  });
  polys.push_back({
      {glm::vec2(0, -1), 4, Edge::kNoIdx},  //
      {glm::vec2(-1, 1), 5, Edge::kNoIdx},  //
      {glm::vec2(1, 1), 6, Edge::kNoIdx},   //
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
      {glm::vec2(-7, 0), 0, Edge::kNoIdx},   //
      {glm::vec2(-6, 3), 1, Edge::kNoIdx},   //
      {glm::vec2(-5, 1), 2, Edge::kNoIdx},   //
      {glm::vec2(-4, 6), 3, Edge::kNoIdx},   //
      {glm::vec2(-3, 2), 4, Edge::kNoIdx},   //
      {glm::vec2(-2, 5), 5, Edge::kNoIdx},   //
      {glm::vec2(-1, 4), 6, Edge::kNoIdx},   //
      {glm::vec2(0, 12), 7, Edge::kNoIdx},   //
      {glm::vec2(-6, 10), 8, Edge::kNoIdx},  //
      {glm::vec2(-8, 11), 9, Edge::kNoIdx},  //
  });
  polys.push_back({
      {glm::vec2(-5, 7), 10, Edge::kNoIdx},  //
      {glm::vec2(-6, 8), 11, Edge::kNoIdx},  //
      {glm::vec2(-5, 9), 12, Edge::kNoIdx},  //
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
      {glm::vec2(0, 0), 0, Edge::kNoIdx},    //
      {glm::vec2(1, 1), 1, Edge::kNoIdx},    //
      {glm::vec2(2, 1), 2, Edge::kNoIdx},    //
      {glm::vec2(3, 1), 3, Edge::kNoIdx},    //
      {glm::vec2(4, 1), 4, Edge::kNoIdx},    //
      {glm::vec2(4, 2), 5, Edge::kNoIdx},    //
      {glm::vec2(3, 2), 6, Edge::kNoIdx},    //
      {glm::vec2(2, 2), 7, Edge::kNoIdx},    //
      {glm::vec2(1, 2), 8, Edge::kNoIdx},    //
      {glm::vec2(0, 3), 9, Edge::kNoIdx},    //
      {glm::vec2(-1, 2), 10, Edge::kNoIdx},  //
      {glm::vec2(-2, 2), 11, Edge::kNoIdx},  //
      {glm::vec2(-3, 2), 12, Edge::kNoIdx},  //
      {glm::vec2(-4, 2), 13, Edge::kNoIdx},  //
      {glm::vec2(-4, 1), 14, Edge::kNoIdx},  //
      {glm::vec2(-3, 1), 15, Edge::kNoIdx},  //
      {glm::vec2(-2, 1), 16, Edge::kNoIdx},  //
      {glm::vec2(-1, 1), 17, Edge::kNoIdx},  //
  });
  TestPoly(polys, 16);
}

TEST(Polygon, Concave) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-0.707107, -0.707107), 0, 10},    //
      {glm::vec2(1, 0), 1, 21},                    //
      {glm::vec2(0.683013, 0), 2, -1},             //
      {glm::vec2(0.37941, -0.232963), 3, -1},      //
      {glm::vec2(0.37941, -0.232963), 4, -1},      //
      {glm::vec2(1.49012e-08, -0.183013), 5, -1},  //
      {glm::vec2(1.49012e-08, -0.183013), 6, -1},  //
      {glm::vec2(-0.140431, 0), 7, 21},            //
      {glm::vec2(-1, 0), 8, 6},                    //
  });
  TestPoly(polys, 7);
}

TEST(Polygon, Sliver) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(2.82003, 0), 0, 181},         //
      {glm::vec2(2.82003, 0), 1, -1},          //
      {glm::vec2(2.06106, 0), 2, -1},          //
      {glm::vec2(2.05793, 0.0680379), 3, -1},  //
      {glm::vec2(2.0641, 0.206908), 4, -1},    //
      {glm::vec2(2.28446, 1.04696), 5, -1},    //
      {glm::vec2(2.35006, 1.2499), 6, 181},    //
      {glm::vec2(-2.82003, 15), 7, 191},       //
      {glm::vec2(-2.82003, 0), 8, 179},        //
  });
  TestPoly(polys, 7);
}

TEST(Polygon, Duplicate) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-32.0774, -10.431), 0, 36},     //
      {glm::vec2(-31.7347, -6.10349), 1, -1},    //
      {glm::vec2(-31.8646, -5.61859), 2, -1},    //
      {glm::vec2(-31.8646, -5.61859), 3, -1},    //
      {glm::vec2(-31.8646, -5.61859), 4, -1},    //
      {glm::vec2(-31.8646, -5.61859), 5, -1},    //
      {glm::vec2(-31.8646, -5.61859), 6, -1},    //
      {glm::vec2(-31.8646, -5.61859), 7, -1},    //
      {glm::vec2(-31.8646, -5.61859), 8, -1},    //
      {glm::vec2(-31.8646, -5.61859), 9, -1},    //
      {glm::vec2(-31.8646, -5.61859), 10, -1},   //
      {glm::vec2(-31.8646, -5.61859), 11, -1},   //
      {glm::vec2(-31.8646, -5.61859), 12, -1},   //
      {glm::vec2(-31.8646, -5.61859), 13, -1},   //
      {glm::vec2(-31.8646, -5.61859), 14, -1},   //
      {glm::vec2(-31.8646, -5.61859), 15, -1},   //
      {glm::vec2(-31.8646, -5.61859), 16, -1},   //
      {glm::vec2(-31.8646, -5.61859), 17, -1},   //
      {glm::vec2(-31.8646, -5.61859), 18, -1},   //
      {glm::vec2(-31.8646, -5.61859), 19, -1},   //
      {glm::vec2(-31.8646, -5.61859), 20, -1},   //
      {glm::vec2(-32.0774, -3.18655), 21, 226},  //
  });
  TestPoly(polys, 20);
}

TEST(Polygon, Folded) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(2.82003, 0), 0, 110},          //
      {glm::vec2(1.23707, 4.20995), 1, -1},     //
      {glm::vec2(1.14141, 4.09091), 2, -1},     //
      {glm::vec2(1.05896, 3.94496), 3, -1},     //
      {glm::vec2(0.00757742, 2.72727), 4, -1},  //
      {glm::vec2(-0.468092, 1.94364), 5, -1},   //
      {glm::vec2(-1.06107, 1.36364), 6, -1},    //
      {glm::vec2(-1.79214, 0.34649), 7, -1},    //
      {glm::vec2(-2.27417, 0), 8, -1},          //
      {glm::vec2(-2.82003, 0), 9, 174},         //
      {glm::vec2(-2.82003, 0), 10, 108},        //
  });
  TestPoly(polys, 9);
}

TEST(Polygon, NearlyLinear) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(2.82003, -8.22814e-05), 0, 231},   //
      {glm::vec2(2.82003, -8.22814e-05), 1, -1},    //
      {glm::vec2(2.31802, -8.22814e-05), 2, -1},    //
      {glm::vec2(-0.164567, -8.22813e-05), 3, -1},  //
      {glm::vec2(-0.857388, -8.22814e-05), 4, -1},  //
      {glm::vec2(-1.01091, -8.22814e-05), 5, 257},  //
      {glm::vec2(-1.01091, -8.22814e-05), 6, 233},  //
  });
  TestPoly(polys, 5);
}

TEST(Polygon, Sliver2) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(27.4996014, 8.6873703), 74, 151},    //
      {glm::vec2(28.27701, 9.52887344), 76, 156},     //
      {glm::vec2(27.6687469, 10.8811588), 104, 152},  //
      {glm::vec2(27.5080414, 8.79682922), 242, -1},   //
      {glm::vec2(27.5594807, 8.75218964), 207, -1},   //
      {glm::vec2(27.4996014, 8.6873703), 268, 152},   //
  });
  TestPoly(polys, 4);
}

TEST(Polygon, Sliver3) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(0, -2.65168381), 369, 1173},               //
      {glm::vec2(0, -0.792692184), 1889, 5777},             //
      {glm::vec2(0, -0.792692184), 2330, -1},               //
      {glm::vec2(0, -1.04356134), 2430, -1},                //
      {glm::vec2(-0.953957975, -0.768045247), 2331, 5777},  //
      {glm::vec2(-1.36363637, -0.757460594), 1892, 1174},   //
  });
  TestPoly(polys, 4);
}

TEST(Polygon, Colinear2) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(11.7864399, -7.4572401), 4176, 13521},    //
      {glm::vec2(11.6818037, -7.30982304), 24873, -1},     //
      {glm::vec2(11.6777582, -7.30626202), 28498, -1},     //
      {glm::vec2(11.6789398, -7.30578804), 24872, 13521},  //
      {glm::vec2(11.3459997, -6.83671999), 4889, 16146},   //
      {glm::vec2(11.25597, -6.9267602), 4888, 13520},      //
  });
  TestPoly(polys, 4);
}

TEST(Polygon, Split) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-0.707106769, -0.707106769), 1, 10},     //
      {glm::vec2(1, 0), 14, 21},                          //
      {glm::vec2(0.683012664, 0), 25, -1},                //
      {glm::vec2(0.379409522, -0.232962906), 33, -1},     //
      {glm::vec2(0.379409522, -0.232962906), 32, -1},     //
      {glm::vec2(1.49011612e-08, -0.183012664), 31, -1},  //
      {glm::vec2(1.49011612e-08, -0.183012664), 30, -1},  //
      {glm::vec2(-0.14043057, 0), 24, 21},                //
      {glm::vec2(-1, 0), 4, 6},                           //
  });
  TestPoly(polys, 7);
}

TEST(Polygon, Duplicates) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-15, -8.10255623), 1648, 151},        //
      {glm::vec2(-15, -9.02439785), 1650, 157},        //
      {glm::vec2(-13.636364, -9.4640789), 1678, 152},  //
      {glm::vec2(-14.996314, -8.10623646), 1916, -1},  //
      {glm::vec2(-15, -8.10639), 1845, -1},            //
      {glm::vec2(-15, -8.10255623), 1922, 152},        //
  });
  TestPoly(polys, 4);
}

TEST(Polygon, Simple1) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(4.04059982, -4.01843977), 2872, 8988},   //
      {glm::vec2(3.95867562, -4.25263977), 24604, -1},    //
      {glm::vec2(4.23459578, -4.30138493), 28274, -1},    //
      {glm::vec2(4.235569, -4.30127287), 28273, -1},      //
      {glm::vec2(4.23782539, -4.30141878), 24602, 8986},  //
  });
  TestPoly(polys, 3);
}

TEST(Polygon, Simple2) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-1, -1), 1, 8},       //
      {glm::vec2(-0.5, -0.5), 9, -1},  //
      {glm::vec2(-1, 0), 11, -1},      //
      {glm::vec2(0, 1), 12, -1},       //
      {glm::vec2(0.5, 0.5), 10, 8},    //
      {glm::vec2(1, 1), 7, 12},        //
      {glm::vec2(-1, 1), 3, 6},        //
  });
  TestPoly(polys, 5);
}

TEST(Polygon, Simple3) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(19.7193489, 6.15445995), 19798, 28537},  //
      {glm::vec2(20.2308197, 5.64299059), 31187, -1},     //
      {glm::vec2(20.3464642, 5.65459776), 27273, -1},     //
      {glm::vec2(20.3733711, 5.65404081), 27274, -1},     //
      {glm::vec2(20.373394, 5.65404034), 31188, 28538},   //
      {glm::vec2(20.8738098, 6.15445995), 19801, 28541},  //
  });
  TestPoly(polys, 4);
}

TEST(Polygon, Simple4) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(15, -12.7135563), 287, 346},          //
      {glm::vec2(15, -10.6843739), 288, 350},          //
      {glm::vec2(15, -10.6843739), 492, -1},           //
      {glm::vec2(15, -11.0041418), 413, -1},           //
      {glm::vec2(15, -11.4550743), 409, -1},           //
      {glm::vec2(15, -11.4550743), 411, -1},           //
      {glm::vec2(14.9958763, -11.4545326), 408, -1},   //
      {glm::vec2(14.4307623, -11.3802214), 412, -1},   //
      {glm::vec2(13.9298496, -11.2768612), 480, 349},  //
  });
  TestPoly(polys, 7);
}

/**
 * This polygon is geometrically valid except that it is wound CW instead of
 * CCW. Therefore we expect it to fail the geometric check.
 */
TEST(Polygon, Inverted) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(0, 2.04124), 0, Edge::kNoIdx},           //
      {glm::vec2(-1.41421, -0.408248), 1, Edge::kNoIdx},  //
      {glm::vec2(-1.23744, -0.408248), 2, Edge::kNoIdx},  //
      {glm::vec2(0, 1.73506), 3, Edge::kNoIdx},           //
      {glm::vec2(1.23744, -0.408248), 4, Edge::kNoIdx},   //
      {glm::vec2(1.41421, -0.408248), 5, Edge::kNoIdx},   //
  });
  polys.push_back({
      {glm::vec2(-1.06066, -0.408248), 6, Edge::kNoIdx},  //
      {glm::vec2(0, 1.42887), 7, Edge::kNoIdx},           //
      {glm::vec2(1.06066, -0.408248), 8, Edge::kNoIdx},   //
  });
  TestPoly(polys, 5, false);
}

/**
 * These polygons are horribly self-intersected. They are not expected to pass
 * geometric checks, but should still produce valid topology.
 */
TEST(Polygon, Ugly) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(0.550049, -0.484235), 0, Edge::kNoIdx},    //
      {glm::vec2(0.411479, -0.20602), 1, Edge::kNoIdx},     //
      {glm::vec2(0.515815, -0.205784), 2, Edge::kNoIdx},    //
      {glm::vec2(0.548218, -0.172791), 3, Edge::kNoIdx},    //
      {glm::vec2(0.547651, -0.0762587), 4, Edge::kNoIdx},   //
      {glm::vec2(0.358672, -0.0999957), 5, Edge::kNoIdx},   //
      {glm::vec2(0.3837, -0.150245), 6, Edge::kNoIdx},      //
      {glm::vec2(0.505506, -0.154427), 7, Edge::kNoIdx},    //
      {glm::vec2(0.546988, 0.0365588), 8, Edge::kNoIdx},    //
      {glm::vec2(0.546255, 0.161176), 9, Edge::kNoIdx},     //
      {glm::vec2(0.47977, -0.0196254), 10, Edge::kNoIdx},   //
      {glm::vec2(0.492479, 0.0576864), 11, Edge::kNoIdx},   //
      {glm::vec2(0.549446, -0.381719), 12, Edge::kNoIdx},   //
      {glm::vec2(0.547831, -0.106901), 13, Edge::kNoIdx},   //
      {glm::vec2(0.36829, -0.00350715), 14, Edge::kNoIdx},  //
      {glm::vec2(0.301766, 0.0142589), 15, Edge::kNoIdx},   //
      {glm::vec2(0.205266, 0.208008), 16, Edge::kNoIdx},    //
      {glm::vec2(0.340096, 0.562179), 17, Edge::kNoIdx},    //
      {glm::vec2(0.255078, 0.464996), 18, Edge::kNoIdx},    //
      {glm::vec2(0.545534, 0.283844), 19, Edge::kNoIdx},    //
      {glm::vec2(0.543549, 0.621731), 20, Edge::kNoIdx},    //
      {glm::vec2(0.363675, 0.790003), 21, Edge::kNoIdx},    //
      {glm::vec2(0.102785, 0.413765), 22, Edge::kNoIdx},    //
      {glm::vec2(0.152009, 0.314934), 23, Edge::kNoIdx},    //
      {glm::vec2(0.198766, 0.33883), 24, Edge::kNoIdx},     //
      {glm::vec2(0.344385, -0.0713115), 25, Edge::kNoIdx},  //
      {glm::vec2(0.261844, 0.0944117), 26, Edge::kNoIdx},   //
      {glm::vec2(0.546909, 0.0499438), 27, Edge::kNoIdx},   //
      {glm::vec2(0.544994, 0.375805), 28, Edge::kNoIdx},    //
      {glm::vec2(0.112526, 0.732868), 29, Edge::kNoIdx},    //
      {glm::vec2(0.205606, 0.429447), 30, Edge::kNoIdx},    //
      {glm::vec2(0.0249926, 0.763546), 31, Edge::kNoIdx},   //
      {glm::vec2(-0.0523676, 0.725275), 32, Edge::kNoIdx},  //
      {glm::vec2(0.124038, 0.371095), 33, Edge::kNoIdx},    //
      {glm::vec2(0.0406817, 0.937559), 34, Edge::kNoIdx},   //
      {glm::vec2(-0.176136, 0.973774), 35, Edge::kNoIdx},   //
      {glm::vec2(-0.111522, 0.844043), 36, Edge::kNoIdx},   //
      {glm::vec2(0.396158, 0.428476), 37, Edge::kNoIdx},    //
      {glm::vec2(0.549669, -0.419631), 38, Edge::kNoIdx},   //
  });
  polys.push_back({
      {glm::vec2(-0.102551, 0.826033), 39, Edge::kNoIdx},   //
      {glm::vec2(-0.0320386, 0.876701), 40, Edge::kNoIdx},  //
      {glm::vec2(0.54275, 0.757552), 41, Edge::kNoIdx},     //
      {glm::vec2(0.542592, 0.784444), 42, Edge::kNoIdx},    //
      {glm::vec2(0.163827, 0.854684), 43, Edge::kNoIdx},    //
      {glm::vec2(-0.0562895, 0.733149), 44, Edge::kNoIdx},  //
  });
  TestPoly(polys, 41, false);
}

TEST(Polygon, Intersected) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(0.20988664, 0.645049632), 9, -1},     //
      {glm::vec2(0.140454829, 0.684921205), 61, 20},   //
      {glm::vec2(-0.368753016, 0.453292787), 62, -1},  //
      {glm::vec2(-0.325120926, 0.444164693), 10, 4},   //
  });
  polys.push_back({
      {glm::vec2(-0.321355969, 0.445578367), 11, -1},  //
      {glm::vec2(-0.355203509, 0.459456205), 63, 20},  //
      {glm::vec2(-0.486033946, 0.399944067), 66, -1},  //
      {glm::vec2(-0.47572422, 0.387616128), 14, 4},    //
  });
  polys.push_back({
      {glm::vec2(-0.441979349, 0.400286674), 12, 4},   //
      {glm::vec2(-0.18760772, 0.49579826), 13, -1},    //
      {glm::vec2(-0.18741411, 0.535780251), 65, 20},   //
      {glm::vec2(-0.435777694, 0.422804594), 64, -1},  //
  });
  TestPoly(polys, 6, false);
}

/**
 * This polygon is degenerate (but still geometrically valid) and demonstrates
 * why the SharesEdge() check is necessary in the triangulator.
 */
TEST(Polygon, BadEdges) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(1, -1), 0, 9},    //
      {glm::vec2(1, 1), 1, 11},    //
      {glm::vec2(1, 1), 2, -1},    //
      {glm::vec2(1, -1), 3, -1},   //
      {glm::vec2(1, -1), 4, -1},   //
      {glm::vec2(-1, -1), 5, 11},  //
      {glm::vec2(-1, -1), 6, 10},  //
  });
  TestPoly(polys, 5);
}

/**
 * This polygon is self-intersected and demonstrates a simple example for which
 * no triangulation exists that satisfies the edge constraints. This is the
 * reason the Boolean contains a fallback triangulator that adds an extra
 * vertex. This situation is only possible with input that is not geometrically
 * valid.
 */
TEST(Polygon, BadEdges2) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-0.292598, -0.322469), 0, 0},   //
      {glm::vec2(-0.282797, -0.340069), 1, -1},  //
      {glm::vec2(-0.158295, -0.30762), 2, 8},    //
      {glm::vec2(-0.189351, -0.230253), 3, -1},  //
      {glm::vec2(-0.329733, -0.255784), 4, 0},   //
      {glm::vec2(-0.342412, -0.233016), 5, -1},  //
      {glm::vec2(-0.202167, -0.198325), 6, 8},   //
      {glm::vec2(-0.223625, -0.144868), 7, -1},  //
  });
  EXPECT_THROW(TestPoly(polys, 6, false), runtimeErr);
}

/**
 * These self-intersected polygons demonstrate a situation where the first step
 * of the triangulator does not produce monotone polygons. Unfortunately, the
 * polygon it produces confuses the second step such that it outputs triangles
 * with duplicate indices. This is another example where the Boolean will use
 * its fallback, and again it will only occur with self-intersected input (and
 * only a small fraction of that).
 */
TEST(Polygon, Intersected2) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-0.542905211, 0.26293695), 41, 12},    //
      {glm::vec2(-0.534729958, 0.262017727), 43, -1},   //
      {glm::vec2(-0.154050604, 0.501501143), 197, -1},  //
      {glm::vec2(-0.266216218, 0.616827428), 198, -1},  //
      {glm::vec2(-0.532943189, 0.2618168), 44, 12},     //
      {glm::vec2(-0.433016717, 0.250580698), 42, -1},   //
      {glm::vec2(-0.320804417, 0.694312572), 196, -1},  //
  });
  EXPECT_THROW(TestPoly(polys, 5, false), runtimeErr);
}

TEST(Polygon, Ugly2) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-0.101033807, -0.224459052), 42, 78},        //
      {glm::vec2(-0.114276297, -0.21651797), 2795, -1},       //
      {glm::vec2(0.0252860114, 0.0279429853), 40516, -1},     //
      {glm::vec2(-0.109061785, -0.219644934), 2794, 78},      //
      {glm::vec2(-0.12047147, -0.212802932), 2799, -1},       //
      {glm::vec2(-0.0944778621, -0.155659199), 40523, -1},    //
      {glm::vec2(-0.0563188158, -0.163132995), 10780, 234},   //
      {glm::vec2(-0.0680458918, -0.179216534), 10763, -1},    //
      {glm::vec2(-0.127449349, 0.181831047), 2813, 80},       //
      {glm::vec2(-0.12863557, 0.181802988), 2809, -1},        //
      {glm::vec2(0.0389348865, -0.0324937254), 10752, 234},   //
      {glm::vec2(0.0478278585, -0.020297125), 10776, -1},     //
      {glm::vec2(0.0475635156, -0.0147248516), 40517, -1},    //
      {glm::vec2(0.0629266277, 0.000410649925), 10775, 234},  //
      {glm::vec2(0.114835069, 0.0716024339), 10769, -1},      //
      {glm::vec2(0.0509060621, 0.186050385), 2818, 80},       //
      {glm::vec2(-0.00853413343, 0.184644222), 2820, -1},     //
      {glm::vec2(-0.235355899, -0.143910572), 2793, 78},      //
      {glm::vec2(-0.231330231, -0.146324635), 2792, -1},      //
      {glm::vec2(-0.126363248, -0.0016354993), 40514, -1},    //
      {glm::vec2(0.00572538376, 0.184981555), 2819, 80},      //
      {glm::vec2(-0.111232482, 0.182214692), 2807, -1},       //
      {glm::vec2(-0.490230888, 0.00892945938), 2782, 78},     //
      {glm::vec2(-0.512604475, 0.0223461706), 2783, -1},      //
      {glm::vec2(-0.212485209, 0.179819345), 2808, 80},       //
      {glm::vec2(-0.381284058, 0.175826088), 2815, -1},       //
      {glm::vec2(-0.549054563, 0.0442040525), 2789, 78},      //
      {glm::vec2(-0.538524151, 0.0378893279), 2800, -1},      //
      {glm::vec2(-0.0595484748, -0.16756244), 10783, 234},    //
      {glm::vec2(-0.0874335244, -0.205806434), 10750, -1},    //
      {glm::vec2(-0.154888913, -0.192163944), 2784, 78},      //
      {glm::vec2(-0.142032966, -0.199873224), 2797, -1},      //
      {glm::vec2(-0.0615496635, -0.103677124), 40519, -1},    //
      {glm::vec2(0.0466179252, 0.0825909078), 40518, -1},     //
      {glm::vec2(0.0504311398, -0.0167267546), 10777, 234},   //
      {glm::vec2(0.101100922, 0.0527662188), 10781, -1},      //
      {glm::vec2(0.144673079, 0.188268632), 2824, 80},        //
      {glm::vec2(0.00604920089, 0.184989214), 2821, -1},      //
      {glm::vec2(0.128165483, 0.0898849294), 10771, 234},     //
      {glm::vec2(0.169255152, 0.146238893), 10753, -1},       //
      {glm::vec2(0.150734439, 0.17369315), 40515, -1},        //
      {glm::vec2(0.130970299, 0.0937316865), 10774, 234},     //
      {glm::vec2(0.18448028, 0.16711998), 10745, -1},         //
      {glm::vec2(0.148723006, 0.188364446), 2804, 80},        //
      {glm::vec2(0.0231154487, 0.185392946), 2806, -1},       //
      {glm::vec2(0.135846302, 0.100419097), 10748, 234},      //
      {glm::vec2(0.188570037, 0.17272903), 10764, -1},        //
      {glm::vec2(0.178380936, 0.189066067), 2816, 80},        //
      {glm::vec2(0.109236225, 0.187430307), 2810, -1},        //
      {glm::vec2(0.0425353609, -0.027555719), 10759, 234},    //
      {glm::vec2(0.0659840852, 0.00460390747), 10778, -1},    //
      {glm::vec2(-0.141682729, -0.200083241), 2796, 78},      //
      {glm::vec2(-0.128434882, -0.208027542), 2781, -1},      //
      {glm::vec2(-0.130571827, -0.0580008253), 40510, -1},    //
      {glm::vec2(-0.0784455761, -0.193479568), 10749, 234},   //
      {glm::vec2(-0.101033807, -0.224459052), 43, 78},        //
      {glm::vec2(-0.101587415, -0.224127069), 2785, -1},      //
      {glm::vec2(-0.100215822, -0.223337188), 10751, 234},    //
  });
  polys.push_back({
      {glm::vec2(0.200869858, 0.189598083), 83, 80},          //
      {glm::vec2(0.189127877, 0.189320311), 2829, -1},        //
      {glm::vec2(-0.0800229609, -0.0271971524), 40524, -1},   //
      {glm::vec2(0.00411255658, -0.080252111), 10772, 234},   //
      {glm::vec2(0.00320341345, -0.0814989954), 10755, -1},   //
      {glm::vec2(-0.00508844247, -0.0574787036), 40512, -1},  //
      {glm::vec2(0.061883457, -0.00102004036), 10757, 234},   //
      {glm::vec2(0.10504882, 0.0581807196), 10756, -1},       //
      {glm::vec2(-0.0992391109, 0.0890228674), 40511, -1},    //
      {glm::vec2(0.150255814, 0.120181553), 10754, 234},      //
      {glm::vec2(0.200869858, 0.189598083), 85, 80},          //
      {glm::vec2(0.151743054, 0.188435897), 2817, -1},        //
      {glm::vec2(0.184203118, 0.166739851), 10766, 234},      //
  });
  polys.push_back({
      {glm::vec2(0.200869858, 0.189598083), 84, 80},      //
      {glm::vec2(0.177050918, 0.189034596), 2826, -1},    //
      {glm::vec2(0.160961747, 0.134864599), 10782, 234},  //
  });
  polys.push_back({
      {glm::vec2(-0.249479041, -0.135441393), 2786, 78},     //
      {glm::vec2(-0.317177385, -0.0948449373), 2787, -1},    //
      {glm::vec2(0.0774079114, 0.0202715453), 10762, 234},   //
      {glm::vec2(0.121518858, 0.0807691664), 10773, -1},     //
      {glm::vec2(-0.034029603, 0.184041068), 2822, 80},      //
      {glm::vec2(-0.116880283, 0.182081074), 2823, -1},      //
      {glm::vec2(-0.0138434581, -0.104878575), 10779, 234},  //
      {glm::vec2(-0.0269743055, -0.122887366), 10768, -1},   //
      {glm::vec2(-0.225466788, -0.149840742), 2791, 78},     //
      {glm::vec2(-0.176685899, -0.179093003), 2790, -1},     //
      {glm::vec2(-0.0581585802, -0.165656209), 10767, 234},  //
      {glm::vec2(-0.080177553, -0.195854962), 10747, -1},    //
      {glm::vec2(-0.109769076, 0.182249308), 2805, 80},      //
      {glm::vec2(-0.122621775, 0.181945249), 2803, -1},      //
      {glm::vec2(-0.112711385, 0.035688974), 40509, -1},     //
      {glm::vec2(-0.0184414312, -0.111184642), 10746, 234},  //
      {glm::vec2(-0.040281795, -0.141138434), 10760, -1},    //
      {glm::vec2(-0.00668276846, 0.184688017), 2811, 80},    //
      {glm::vec2(-0.112469606, 0.182185411), 2812, -1},      //
      {glm::vec2(0.0327619836, -0.0409597605), 10761, 234},  //
      {glm::vec2(0.0371489823, -0.0349430591), 10758, -1},   //
  });
  polys.push_back({
      {glm::vec2(-0.528274596, 0.0317429788), 2788, -1},   //
      {glm::vec2(-0.136698052, 0.181612253), 2814, 80},    //
      {glm::vec2(-0.361992508, 0.176282465), 2825, -1},    //
      {glm::vec2(-0.520582616, 0.0789725408), 40522, -1},  //
      {glm::vec2(-0.516868532, 0.0754575953), 40521, -1},  //
      {glm::vec2(-0.480212837, 0.0658109486), 40520, -1},  //
      {glm::vec2(-0.55114758, 0.0454591736), 2798, 78},    //
      {glm::vec2(-0.569476902, 0.0564506426), 2802, -1},   //
      {glm::vec2(-0.488848627, 0.173281431), 2828, 80},    //
      {glm::vec2(-0.409982115, 0.175147176), 2827, -1},    //
      {glm::vec2(-0.517921209, 0.0255344063), 2801, 78},   //
  });
  polys.push_back({
      {glm::vec2(0.102886759, 0.0552154705), 10765, 234},  //
      {glm::vec2(0.168070927, 0.144614756), 10770, -1},    //
      {glm::vec2(0.0853726789, 0.068075642), 40513, -1},   //
  });
  ASSERT_THROW(TestPoly(polys, 5, false), runtimeErr);
}

// void fnExit() { throw std::runtime_error("Someone called Exit()!"); }

int main(int argc, char **argv) {
  // atexit(fnExit);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
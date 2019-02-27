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
  Polygons polys_out = Assemble(edges);
  Identical(polys, polys_out);
}

void TestPoly(const Polygons &polys) {
  TestAssemble(polys);
  std::vector<TriVerts> triangles;
  Triangulate(triangles, polys);
  CheckManifold(triangles, polys);
}
}  // namespace

TEST(SimplePolygon, SimpleHole) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(0, -2), 11},  //
      {glm::vec2(2, 2), 21},   //
      {glm::vec2(0, 4), 47},   //
      {glm::vec2(-3, 3), 14},  //
  });
  polys.push_back({
      {glm::vec2(0, -1), 5},  //
      {glm::vec2(-1, 1), 8},  //
      {glm::vec2(1, 1), 3},   //
  });
  TestPoly(polys);
}

TEST(SimplePolygon, MultiMerge) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-7, 0), 5},    //
      {glm::vec2(-6, 3), 4},    //
      {glm::vec2(-5, 1), 7},    //
      {glm::vec2(-4, 6), 1},    //
      {glm::vec2(-3, 2), 2},    //
      {glm::vec2(-2, 5), 9},    //
      {glm::vec2(-1, 4), 3},    //
      {glm::vec2(0, 12), 6},    //
      {glm::vec2(-6, 10), 12},  //
      {glm::vec2(-8, 11), 18},  //
  });
  polys.push_back({
      {glm::vec2(-5, 7), 11},  //
      {glm::vec2(-6, 8), 21},  //
      {glm::vec2(-5, 9), 47},  //
  });
  TestPoly(polys);
}

TEST(SimplePolygon, Colinear) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-5.48368, -3.73905), 0},   //
      {glm::vec2(-4.9881, -4.51552), 1},    //
      {glm::vec2(-4.78988, -4.13186), 2},   //
      {glm::vec2(-4.82012, -4.13999), 3},   //
      {glm::vec2(-4.84314, -4.14617), 4},   //
      {glm::vec2(-4.85738, -4.13581), 5},   //
      {glm::vec2(-4.86772, -4.12831), 6},   //
      {glm::vec2(-4.87337, -4.12422), 7},   //
      {glm::vec2(-4.88097, -4.1187), 8},    //
      {glm::vec2(-4.89799, -4.10634), 9},   //
      {glm::vec2(-4.90219, -4.10329), 10},  //
      {glm::vec2(-4.90826, -4.09887), 11},  //
      {glm::vec2(-4.90846, -4.09873), 12},  //
      {glm::vec2(-4.91227, -4.09597), 13},  //
      {glm::vec2(-4.92199, -4.0889), 14},   //
      {glm::vec2(-5.0245, -4.01443), 15},   //
      {glm::vec2(-5.02494, -4.01412), 16},  //
      {glm::vec2(-5.02536, -4.01381), 17},  //
      {glm::vec2(-5.0316, -4.00927), 18},   //
      {glm::vec2(-5.03211, -4.00891), 19},  //
      {glm::vec2(-5.05197, -3.99448), 20},  //
      {glm::vec2(-5.14757, -3.92504), 21},  //
      {glm::vec2(-5.21287, -3.8776), 22},   //
      {glm::vec2(-5.29419, -3.81853), 23},  //
      {glm::vec2(-5.29907, -3.81499), 24},  //
      {glm::vec2(-5.36732, -3.76541), 25},  //
  });
  TestPoly(polys);
}

TEST(SimplePolygon, Merges) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(-3.22039, 10.2769), 0},   //
      {glm::vec2(-3.12437, 10.4147), 1},   //
      {glm::vec2(-3.99093, 10.1781), 2},   //
      {glm::vec2(-3.8154, 10.0716), 3},    //
      {glm::vec2(-3.78982, 10.0893), 4},   //
      {glm::vec2(-3.55033, 10.2558), 5},   //
      {glm::vec2(-3.50073, 10.2549), 6},   //
      {glm::vec2(-3.47018, 10.2572), 7},   //
      {glm::vec2(-3.42633, 10.2605), 8},   //
      {glm::vec2(-3.34311, 10.2604), 9},   //
      {glm::vec2(-3.32096, 10.2633), 10},  //
  });
  TestPoly(polys);
}

TEST(SimplePolygon, ColinearY) {
  Polygons polys;
  polys.push_back({
      {glm::vec2(0, 0), 0},   //
      {glm::vec2(-1, 1), 5},  //
      {glm::vec2(-2, 1), 2},  //
      {glm::vec2(-3, 1), 1},  //
      {glm::vec2(-4, 1), 7},  //
      {glm::vec2(-4, 2), 8},  //
      {glm::vec2(-3, 2), 3},  //
      {glm::vec2(-2, 2), 9},  //
      {glm::vec2(-1, 2), 4},  //
      {glm::vec2(0, 3), 10},  //
      {glm::vec2(1, 2), 11},  //
      {glm::vec2(2, 2), 12},  //
      {glm::vec2(3, 2), 14},  //
      {glm::vec2(4, 2), 13},  //
      {glm::vec2(4, 1), 15},  //
      {glm::vec2(3, 1), 18},  //
      {glm::vec2(2, 1), 19},  //
      {glm::vec2(1, 1), 16},  //
  });
  TestPoly(polys);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
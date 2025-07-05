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

#include "manifold/polygon.h"

#include <algorithm>
#include <fstream>
#include <limits>

#include "manifold/cross_section.h"
#include "test.h"

namespace {

using namespace manifold;

Polygons Turn180(Polygons polys) {
  for (SimplePolygon &poly : polys) {
    for (vec2 &vert : poly) {
      vert *= -1.0;
    }
  }
  return polys;
}

Polygons Duplicate(Polygons polys) {
  double xMin = std::numeric_limits<double>::infinity();
  double xMax = -std::numeric_limits<double>::infinity();
  for (SimplePolygon &poly : polys) {
    for (vec2 &vert : poly) {
      xMin = std::min(xMin, vert.x);
      xMax = std::max(xMax, vert.x);
    }
  }
  const double shift = xMax - xMin;

  const int nPolys = polys.size();
  for (int i = 0; i < nPolys; ++i) {
    SimplePolygon poly = polys[i];
    for (vec2 &vert : poly) {
      vert.x += shift;
    }
    polys.push_back(poly);
  }
  return polys;
}

void TestPoly(const Polygons &polys, int expectedNumTri,
              double epsilon = -1.0) {
  std::vector<ivec3> triangles;
  EXPECT_NO_THROW(triangles = Triangulate(polys, epsilon));
  EXPECT_EQ(triangles.size(), expectedNumTri) << "Basic";

  EXPECT_NO_THROW(triangles = Triangulate(Turn180(polys), epsilon));
  EXPECT_EQ(triangles.size(), expectedNumTri) << "Turn 180";

  EXPECT_NO_THROW(triangles = Triangulate(Duplicate(polys), epsilon));
  EXPECT_EQ(triangles.size(), 2 * expectedNumTri) << "Duplicate";
}

class PolygonTestFixture : public testing::Test {
 public:
  Polygons polys;
  double epsilon;
  int expectedNumTri;
  explicit PolygonTestFixture(Polygons polys, double epsilon,
                              int expectedNumTri)
      : polys(polys), epsilon(epsilon), expectedNumTri(expectedNumTri) {}
  void TestBody() { TestPoly(polys, expectedNumTri, epsilon); }
};

void RegisterPolygonTestsFile(const std::string &filename) {
  auto f = std::ifstream(filename);
  EXPECT_TRUE(f.is_open());

  // for each test:
  //   test name, expectedNumTri, epsilon, num polygons
  //   for each polygon:
  //     num points
  //     for each vertex:
  //       x coord, y coord
  //
  // note that we should not have commas in the file

  std::string name;
  double epsilon, x, y;
  int expectedNumTri, numPolys, numPoints;

  while (1) {
    f >> name;
    if (f.eof()) break;
    f >> expectedNumTri >> epsilon >> numPolys;
    Polygons polys;
    for (int i = 0; i < numPolys; i++) {
      polys.emplace_back();
      f >> numPoints;
      for (int j = 0; j < numPoints; j++) {
        f >> x >> y;
        polys.back().emplace_back(x, y);
      }
    }
    testing::RegisterTest(
        "Polygon", name.c_str(), nullptr, nullptr, __FILE__, __LINE__,
        [=, polys = std::move(polys)]() -> PolygonTestFixture * {
          return new PolygonTestFixture(polys, epsilon, expectedNumTri);
        });
  }
  f.close();
}
}  // namespace

void RegisterPolygonTests() {
  std::string files[] = {"polygon_corpus.txt", "sponge.txt", "zebra.txt",
                         "zebra3.txt"};

#ifdef __EMSCRIPTEN__
  for (auto f : files) RegisterPolygonTestsFile("/polygons/" + f);
#else
  std::string file = __FILE__;
  auto end = std::min(file.rfind('\\'), file.rfind('/'));
  std::string dir = file.substr(0, end);
  for (auto f : files) RegisterPolygonTestsFile(dir + "/polygons/" + f);
#endif
}

struct PolygonTest {
  PolygonTest(const manifold::Polygons &polygons)
      : name("Result"), polygons(polygons) {};

  std::string name;
  int expectedNumTri = -1;
  double epsilon = -1;

  manifold::Polygons polygons;
};

void Save(const std::string &filename, const std::vector<PolygonTest> &result) {
  // Open a file stream for writing.
  std::ofstream outFile(filename);

  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }

  // Write each test case to the file.
  for (const auto &test : result) {
    // Write the header for the test.
    outFile << test.name << " " << test.expectedNumTri << " " << test.epsilon
            << " " << test.polygons.size() << "\n";

    // Write each polygon within the test.
    for (const auto &poly : test.polygons) {
      // Write the number of points for the current polygon.
      outFile << poly.size() << "\n";
      // Write the coordinates for each point in the polygon.
      for (const auto &point : poly) {
        outFile << point.x << " " << point.y << "\n";
      }
    }
  }

  outFile.close();
  std::cout << "Successfully saved " << result.size() << " tests to "
            << filename << std::endl;
}

TEST(Polygons, Fillet) {
  manifold::Polygons Rect{{vec2{0, 0}, vec2{0, 5}, vec2{5, 5}, vec2{5, 0}}},
      Tri{{vec2{0, 0}, vec2{0, 5}, vec2{5, 0}}}, AShape{{vec2{}}},
      UShape{{vec2{0, 0}, vec2{-1, 5}, vec2{3, 1}, vec2{7, 5}, vec2{6, 0}}},
      // Corner testcase
      ZShape{{vec2{0, 0}, vec2{4, 4}, vec2{0, 6}, vec2{6, 6}, vec2{3, 1},
              vec2{6, 0}}},
      WShape{{vec2{0, 0}, vec2{-2, 5}, vec2{0, 3}, vec2{2, 5}, vec2{4, 3},
              vec2{6, 5}, vec2{4, 0}, vec2{2, 3}}},
      TShape{{vec2{0, 0}, vec2{0, 5}, vec2{2, 5}, vec2{0, 8}, vec2{4, 8},
              vec2{3, 5}, vec2{5, 5}, vec2{5, 0}}},
      // Spike case
      Spike1{{vec2{0, 0}, vec2{0, 5}, vec2{5, 5}, vec2{5, 0}, vec2{3, 0},
              vec2{3.5, -0.3}, vec2{2.9, 0}}},
      Spike2{{vec2{0, 0}, vec2{-1, 5}, vec2{2, 1}, vec2{4, 1}, vec2{7, 5},
              vec2{6, 0}, vec2{2.6, 0}, vec2{2.9, -0.1}, vec2{2.5, 0}}},
      Spike3{{vec2{0, 0}, vec2{-1, 5}, vec2{2, 1}, vec2{4, 1}, vec2{7, 5},
              vec2{6, 0}, vec2{2.6, 0}, vec2{5, -1}, vec2{2.5, 0}}};

  const manifold::Polygons polygon = ZShape;
  const double radius = 0.7;

  manifold::ManifoldParams().verbose = true;

  std::vector<PolygonTest> result{
      // poly,
      // PolygonTest(VertexByVertex(radius, poly)),
      manifold::CrossSection::Fillet(polygon, radius, 20),
  };

  // UnionFind

  Save("result.txt", result);
}
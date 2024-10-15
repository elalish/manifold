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
              double precision = -1.0) {
  PolygonParams().verbose = options.params.verbose;

  std::vector<ivec3> triangles;
  EXPECT_NO_THROW(triangles = Triangulate(polys, precision));
  EXPECT_EQ(triangles.size(), expectedNumTri) << "Basic";

  EXPECT_NO_THROW(triangles = Triangulate(Turn180(polys), precision));
  EXPECT_EQ(triangles.size(), expectedNumTri) << "Turn 180";

  EXPECT_NO_THROW(triangles = Triangulate(Duplicate(polys), precision));
  EXPECT_EQ(triangles.size(), 2 * expectedNumTri) << "Duplicate";

  PolygonParams().verbose = false;
}

class PolygonTestFixture : public testing::Test {
 public:
  Polygons polys;
  double precision;
  int expectedNumTri;
  explicit PolygonTestFixture(Polygons polys, double precision,
                              int expectedNumTri)
      : polys(polys), precision(precision), expectedNumTri(expectedNumTri) {}
  void TestBody() { TestPoly(polys, expectedNumTri, precision); }
};

void RegisterPolygonTestsFile(const std::string &filename) {
  auto f = std::ifstream(filename);
  EXPECT_TRUE(f.is_open());

  // for each test:
  //   test name, expectedNumTri, precision, num polygons
  //   for each polygon:
  //     num points
  //     for each vertex:
  //       x coord, y coord
  //
  // note that we should not have commas in the file

  std::string name;
  double precision, x, y;
  int expectedNumTri, numPolys, numPoints;

  while (1) {
    f >> name;
    if (f.eof()) break;
    f >> expectedNumTri >> precision >> numPolys;
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
          return new PolygonTestFixture(polys, precision, expectedNumTri);
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

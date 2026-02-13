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
#include <iomanip>

#include "../src/vec.h"
#include "manifold/cross_section.h"
#include "test.h"

namespace {

using namespace manifold;

Polygons Turn180(Polygons polys) {
  for (SimplePolygon& poly : polys) {
    for (vec2& vert : poly) {
      vert *= -1.0;
    }
  }
  return polys;
}

Polygons Duplicate(Polygons polys) {
  double xMin = std::numeric_limits<double>::infinity();
  double xMax = -std::numeric_limits<double>::infinity();
  for (SimplePolygon& poly : polys) {
    for (vec2& vert : poly) {
      xMin = std::min(xMin, vert.x);
      xMax = std::max(xMax, vert.x);
    }
  }
  const double shift = xMax - xMin;

  const int nPolys = polys.size();
  for (int i = 0; i < nPolys; ++i) {
    SimplePolygon poly = polys[i];
    for (vec2& vert : poly) {
      vert.x += shift;
    }
    polys.push_back(poly);
  }
  return polys;
}

void TestPoly(const Polygons& polys, int expectedNumTri,
              double epsilon = -1.0) {
  std::vector<ivec3> triangles;
  EXPECT_NO_THROW(triangles = Triangulate(polys, epsilon));
  EXPECT_EQ(triangles.size(), expectedNumTri) << "Basic";

  EXPECT_NO_THROW(triangles = Triangulate(Turn180(polys), epsilon));
  EXPECT_EQ(triangles.size(), expectedNumTri) << "Turn 180";

  EXPECT_NO_THROW(triangles = Triangulate(Duplicate(polys), epsilon));
  EXPECT_EQ(triangles.size(), 2 * expectedNumTri) << "Duplicate";
}

bool TestFillet(const Polygons& polys, CrossSection input, double radius,
                int inputCircularSegments) {
  const int circularSegments = inputCircularSegments > 2
                                   ? inputCircularSegments
                                   : Quality::GetCircularSegments(radius);

  auto r = input.Fillet(radius, circularSegments);
  auto rc = manifold::CrossSection::Compose(r);

#ifdef MANIFOLD_DEBUG
  std::cout << "[==========] Testing radius: " << radius << std::endl;
#endif

  EXPECT_TRUE((manifold::CrossSection(polys).Area() == 0) ||
              (rc.Area() < manifold::CrossSection(polys).Area()));

  auto toRad = [](const vec2& v) -> double { return atan2(v.y, v.x); };

  auto normalizeAngle = [](double angle) -> double {
    while (angle < 0) angle += 2.0 * M_PI;
    while (angle >= 2.0 * M_PI) angle -= 2.0 * M_PI;
    return angle;
  };

  for (const auto& crossSection : r) {
    auto polygon = crossSection.ToPolygons();
    for (const auto& loop : polygon) {
      const auto& cs = CrossSection(loop);

      bool isCCW = cs.Area() > 0;

      for (size_t i = 0; i != loop.size(); i++) {
        vec2 p1 = loop[i], p2 = loop[(i + 1) % loop.size()],
             p3 = loop[(i + 2) % loop.size()];

        vec2 e1 = p2 - p1, e2 = p3 - p2;

        // Check angle between edge
        double angle = normalizeAngle(toRad(e2) - toRad(e1));

        const double dPhi = 2.0 * M_PI / circularSegments;

        // Is the threshold too low?
        // Specify to Polygon.Fillet.CoincidentHole4
        EXPECT_TRUE(angle > M_PI || angle < (dPhi + dPhi * 0.1));
      }
    }
  }

  // Use the result run again, check the result is almost same.
  // Check idempotent
  if (true) {
    auto rr = rc.Fillet(radius, circularSegments);
    auto rrc = manifold::CrossSection::Compose(rr);

    EXPECT_NEAR(rc.Area(), rrc.Area(), 0.1 * (input.Area() - rc.Area()));
  }

  return rc.Area() > 0 && std::abs((input.Area() - rc.Area())) > 1E-12;
};

void BuildFillet(const Polygons& polys, double epsilon = -1.0) {
  manifold::ManifoldParams().verbose = false;
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

  const int inputCircularSegments = 20;
  const CrossSection input = CrossSection(polys);

#ifdef MANIFOLD_DEBUG
  if (false) {
    double radius = 0.5;
    TestFillet(polys, input, radius, inputCircularSegments);
  } else {
#endif

    const vec2 bbox = input.Bounds().Size();
    const double min = std::min(bbox.x, bbox.y);

    // Testing Positive Radius

    double low = 1E-6, high = 0.8 * min;

    TestFillet(polys, input, low, inputCircularSegments);

    // Maximum attempt 20 times
    // Early stop if 10 case result non zero.
    for (size_t i = 0, j = 0; i != 20 && j != 10; i++) {
      double mid = low + (high - low) * 0.5;

      if (std::abs(low - mid) < 1E-6) break;

      // Area non zero
      if (TestFillet(polys, input, mid, inputCircularSegments)) {
        low = mid;
        j++;
      } else {
        high = mid;
      }
    }

    // Testing Negative Radius

    low = -1E-6, high = -0.8 * min;

    TestFillet(polys, input, low, inputCircularSegments);

    // Maximum attempt 20 times
    // Early stop if 10 case result non zero.
    for (size_t i = 0, j = 0; i != 20 && j != 10; i++) {
      double mid = low + (high - low) * 0.5;

      // Area non zero
      if (TestFillet(polys, input, mid, inputCircularSegments)) {
        high = mid;
        j++;
      } else {
        low = mid;
      }
    }

#ifdef MANIFOLD_DEBUG
  }
#endif
}

}  // namespace

class PolygonTestFixture : public testing::Test {
 public:
  Polygons polys;
  double epsilon;
  int expectedNumTri;
  std::string name;

  explicit PolygonTestFixture(Polygons polys, double epsilon,
                              int expectedNumTri, const std::string& name)
      : polys(polys),
        epsilon(epsilon),
        expectedNumTri(expectedNumTri),
        name(name) {}

  void TestBody() { TestPoly(polys, expectedNumTri, epsilon); }
};

class FilletTestFixture : public testing::Test {
 public:
  Polygons polys;
  double epsilon;
  int expectedNumTri;
  std::string name;

  explicit FilletTestFixture(Polygons polys, double epsilon, int expectedNumTri,
                             const std::string& name)
      : polys(polys),
        epsilon(epsilon),
        expectedNumTri(expectedNumTri),
        name(name) {}

  void TestBody() { BuildFillet(polys, epsilon); }
};

template <typename TestFixture>
void RegisterPolygonTestsFile(const std::string& suitename,
                              const std::string& filename) {
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
        suitename.c_str(), name.c_str(), nullptr, nullptr, __FILE__, __LINE__,
        [=, polys = std::move(polys)]() -> TestFixture* {
          return new TestFixture(polys, epsilon, expectedNumTri, name);
        });
  }
  f.close();
}

void RegisterPolygonTests() {
  std::string files[] = {"polygon_corpus.txt", "sponge.txt", "zebra.txt",
                         "zebra3.txt"};

#ifdef __EMSCRIPTEN__
  for (auto f : files) RegisterPolygonTestsFile("/polygons/" + f);
#else
  std::string file = __FILE__;
  auto end = std::min(file.rfind('\\'), file.rfind('/'));
  std::string dir = file.substr(0, end);
  for (auto f : files)
    RegisterPolygonTestsFile<PolygonTestFixture>("Polygon",
                                                 dir + "/polygons/" + f);
#endif
}

void RegisterFilletTests() {
  // std::string files[] = {"fillet.txt", "polygon_corpus.txt", "sponge.txt",
  //                        "zebra.txt", "zebra3.txt"};

  std::string files[] = {"fillet.txt"};

#ifdef __EMSCRIPTEN__
  for (auto f : files) RegisterPolygonTestsFile("/polygons/" + f);
#else
  std::string file = __FILE__;
  auto end = std::min(file.rfind('\\'), file.rfind('/'));
  std::string dir = file.substr(0, end);
  for (auto f : files)
    RegisterPolygonTestsFile<FilletTestFixture>("Polygon.Fillet",
                                                dir + "/polygons/" + f);
#endif
}
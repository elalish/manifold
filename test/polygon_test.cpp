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

#define _USE_MATH_DEFINES

#include "manifold/polygon.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../src/vec.h"
#include "manifold/cross_section.h"
#include "test.h"

namespace {

using namespace manifold;

struct FilletResult {
  bool valid;
  double area;
};

class PositiveFilletAreaTracker {
 public:
  void Check(double radius, const FilletResult& result) {
    if (radius <= 0 || !result.valid) return;

    for (const auto& sample : samples_) {
      const double otherRadius = sample.first;
      const double otherArea = sample.second;

      if (radius == otherRadius) continue;

      const double smallerRadius = std::min(radius, otherRadius);
      const double largerRadius = std::max(radius, otherRadius);
      const double smallerArea = radius < otherRadius ? result.area : otherArea;
      const double largerArea = radius < otherRadius ? otherArea : result.area;
      const double tolerance =
          1e-9 * std::max({1.0, std::abs(smallerArea), std::abs(largerArea)});

      if (largerArea > smallerArea + tolerance) {
        std::ostringstream message;
        message << std::setprecision(std::numeric_limits<double>::max_digits10)
                << "Positive fillet area increased as radius increased: "
                << "r=" << smallerRadius << " area=" << smallerArea
                << ", r=" << largerRadius << " area=" << largerArea;
        // throw std::runtime_error(message.str());
      }
    }

    samples_.push_back({radius, result.area});
  }

 private:
  std::vector<std::pair<double, double>> samples_;
};

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

FilletResult TestFillet(const Polygons& polys, CrossSection input,
                        double radius, int inputCircularSegments) {
  const int circularSegments =
      inputCircularSegments > 2
          ? inputCircularSegments
          : Quality::GetCircularSegments(std::abs(radius));

  auto r = input.Fillet(radius, circularSegments);
  auto rc = manifold::CrossSection::Compose(r);

#ifdef MANIFOLD_DEBUG
  std::cout << "[==========] Testing radius: " << radius << std::endl;
#endif

  const double inArea = manifold::CrossSection(polys).Area();
  const double outArea = rc.Area();

  const bool expectGrowth = radius < 0;
  if (outArea > 0) {
    EXPECT_TRUE((inArea == 0) ||
                (expectGrowth ? outArea > inArea : outArea < inArea));
  }

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
        EXPECT_TRUE(std::isfinite(p1.x));
        EXPECT_TRUE(std::isfinite(p1.y));
        EXPECT_GT(length2(e1), 0);
        EXPECT_GT(length2(e2), 0);

        if (radius > 0) {
          // Check angle between edge
          double angle = normalizeAngle(toRad(e2) - toRad(e1));

          const double dPhi = 2.0 * M_PI / circularSegments;

          // Is the threshold too low?
          // Specify to Polygon.Fillet.CoincidentHole4
          EXPECT_TRUE(angle > M_PI || angle < (dPhi + dPhi * 0.1));
        }
      }
    }
  }

  // Use the result run again, check the result is almost same.
  // Check idempotent
  if (false) {
    auto rr = rc.Fillet(radius, circularSegments);
    auto rrc = manifold::CrossSection::Compose(rr);

    EXPECT_NEAR(rc.Area(), rrc.Area(), 0.1 * (input.Area() - rc.Area()));
  }

  return {rc.Area() > 0 && std::abs((input.Area() - rc.Area())) > 1E-12,
          outArea};
};

void BuildFillet(const Polygons& polys, double epsilon = -1.0,
                 double maxCircleRadius = 0.0) {
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

  const int inputCircularSegments = 20;
  const CrossSection input = CrossSection(polys);
  PositiveFilletAreaTracker positiveAreaTracker;
  auto testFillet = [&](double radius) {
    FilletResult result =
        TestFillet(polys, input, radius, inputCircularSegments);
    positiveAreaTracker.Check(radius, result);
    return result.valid;
  };

#ifdef MANIFOLD_DEBUG
  manifold::ManifoldParams().verbose = false;
  int idx = 2;

  if (idx == 0) {
    double radius = 0.1;
    testFillet(radius);
  } else if (idx == 1) {
    auto bbox = input.Bounds().Size();

    double min = std::min(bbox.x, bbox.y), max = std::max(bbox.x, bbox.y);

    std::vector<double> radiusVec;
    if (true) {
      std::array<double, 6> multipliers{1E-4, 1E-3, 1E-2, 0.1, 0.5, 1};
      // std::array<double, 1> multipliers{1E-3};
      for (auto it = multipliers.begin(); it != multipliers.end(); it++) {
        double mmin = *it * min, mmax = *it * max;
        if (std::abs(mmin - mmax) < 1E-6) {
          radiusVec.push_back(mmin);
          // radiusVec.push_back(-1.0 * mmin);
        } else {
          radiusVec.push_back(min);
          // radiusVec.push_back(max);

          // radiusVec.push_back(-1.0 * mmin);
          // radiusVec.push_back(-1.0 * mmax);
        }
      }
    } else {
      radiusVec.push_back(0.7);
    }

    std::sort(radiusVec.begin(), radiusVec.end());

    for (auto it = radiusVec.begin(); it != radiusVec.end(); it++)
      testFillet(*it);
  } else {
#endif

    const vec2 bbox = input.Bounds().Size();
    const double min = std::min(bbox.x, bbox.y);

    // Testing Positive Radius

    double low = 1E-6;
    const double positiveHigh =
        maxCircleRadius > low ? 1.05 * maxCircleRadius : 0.8 * min;
    double high = positiveHigh;

    testFillet(low);

    // Maximum attempt 20 times
    // Early stop if 10 case result non zero.

    const size_t MAXIMUM = 100, EARLYSTOP = 100;
    for (size_t i = 0, j = 0; i != MAXIMUM && j != EARLYSTOP; i++) {
      double mid = low + (high - low) * 0.5;

      if (std::abs(low - mid) < 1E-20) break;

      // Area non zero
      if (testFillet(mid)) {
        low = mid;
        j++;
      } else {
        high = mid;
      }
    }

    if (maxCircleRadius > 0 &&
        (low < maxCircleRadius || !testFillet((low + maxCircleRadius) * 0.5)) &&
        testFillet(maxCircleRadius)) {
      low = maxCircleRadius;
      high = positiveHigh;
      const size_t SCAN = 128;
      for (size_t i = 1; i <= SCAN; ++i) {
        const double radius =
            maxCircleRadius + (positiveHigh - maxCircleRadius) * i / SCAN;
        if (testFillet(radius)) {
          low = radius;
          continue;
        }

        high = radius;
        for (size_t j = 0; j != MAXIMUM; ++j) {
          double mid = low + (high - low) * 0.5;

          if (testFillet(mid)) {
            low = mid;
          } else {
            high = mid;
          }
        }
        break;
      }
    }

    EXPECT_NEAR(low, maxCircleRadius, std::max(1E-6, 1E-3 * maxCircleRadius));

    // Testing Negative Radius

    low = -1E-6, high = -0.8 * min;

    testFillet(low);

    // Maximum attempt 20 times
    // Early stop if 10 case result non zero.
    for (size_t i = 0, j = 0; i != MAXIMUM && j != EARLYSTOP; i++) {
      double mid = low + (high - low) * 0.5;

      // Area non zero
      if (testFillet(mid)) {
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
                              int expectedNumTri, const std::string& name,
                              double maxCircleRadius)
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
  double maxCircleRadius;
  int expectedNumTri;
  std::string name;

  explicit FilletTestFixture(Polygons polys, double epsilon, int expectedNumTri,
                             const std::string& name, double maxCircleRadius)
      : polys(polys),
        epsilon(epsilon),
        maxCircleRadius(maxCircleRadius),
        expectedNumTri(expectedNumTri),
        name(name) {}

  void TestBody() { BuildFillet(polys, epsilon, maxCircleRadius); }
};

template <typename TestFixture>
void RegisterPolygonTestsFile(const std::string& suitename,
                              const std::string& filename) {
  auto f = std::ifstream(filename);
  EXPECT_TRUE(f.is_open());

  // for each test:
  //   test name, expectedNumTri, epsilon, num polygons, max circle radius
  //   for each polygon:
  //     num points
  //     for each vertex:
  //       x coord, y coord
  //
  // note that we should not have commas in the file

  std::string name;
  double epsilon, x, y, maxCircleRadius;
  int expectedNumTri, numPolys, numPoints;

  while (1) {
    f >> name;
    if (f.eof()) break;
    f >> expectedNumTri >> epsilon >> numPolys >> maxCircleRadius;
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
          return new TestFixture(polys, epsilon, expectedNumTri, name,
                                 maxCircleRadius);
        });
  }
  f.close();
}

void RegisterPolygonTests() {
  std::string files[] = {"fillet.txt", "polygon_corpus.txt", "sponge.txt",
                         "zebra.txt", "zebra3.txt"};
  // std::string files[] = {"fillet.txt"};

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
  // std::string files[] = {"fillet.txt"};

  std::string files[] = {"fillet.txt", "polygon_corpus.txt", "sponge.txt",
                         "zebra.txt", "zebra3.txt"};

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

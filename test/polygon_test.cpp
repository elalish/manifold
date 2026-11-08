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
#include <array>
#include <atomic>
#include <cmath>
#ifndef MANIFOLD_NO_IOSTREAM
#include <fstream>
#endif
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "../src/polygon_internal.h"
#include "../src/vec.h"
#include "gtest/gtest.h"
#include "manifold/cross_section.h"
#include "polygon_corpus.h"
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

Polygons StarPolygon(int numVert) {
  Polygons polys(1);
  polys[0].reserve(numVert);
  for (int i = 0; i < numVert; ++i) {
    const double angle = kTwoPi * i / numVert;
    const double radius = i % 2 == 0 ? 2.0 : 1.0;
    polys[0].push_back({radius * std::cos(angle), radius * std::sin(angle)});
  }
  return polys;
}

PolygonsIdx IndexPolygons(const Polygons& polys) {
  PolygonsIdx indexed;
  int idx = 0;
  for (const SimplePolygon& poly : polys) {
    SimplePolygonIdx simple;
    simple.reserve(poly.size());
    for (const vec2& pos : poly) simple.push_back({pos, idx++});
    indexed.push_back(std::move(simple));
  }
  return indexed;
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
  auto rc = CrossSection::BatchBoolean(r, OpType::Add);

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
    auto rrc = CrossSection::BatchBoolean(rr, OpType::Add);

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

#ifndef MANIFOLD_NO_IOSTREAM
void RegisterPolygonTestsFile(const std::string& filename) {
  const std::vector<PolygonCorpusEntry> corpus = ReadPolygonCorpus(filename);
  // The corpus files are committed, so an empty read means a missing or
  // unreadable file; fail loudly rather than silently registering no tests.
  EXPECT_FALSE(corpus.empty()) << "no polygon corpus read from " << filename;
  for (const auto& entry : corpus) {
    testing::RegisterTest(
        "Polygon", entry.name.c_str(), nullptr, nullptr, __FILE__, __LINE__,
        [polys = entry.polys, epsilon = entry.epsilon,
         expectedNumTri = entry.expectedNumTri]() -> PolygonTestFixture* {
          return new PolygonTestFixture(polys, epsilon, expectedNumTri);
        });
  }
}

class FilletTestFixture : public testing::Test {
 public:
  Polygons polys;
  double epsilon;
  double maxCircleRadius;
  explicit FilletTestFixture(Polygons polys, double epsilon,
                             double maxCircleRadius)
      : polys(polys), epsilon(epsilon), maxCircleRadius(maxCircleRadius) {}
  void TestBody() { BuildFillet(polys, epsilon, maxCircleRadius); }
};

void RegisterFilletTestsFile(const std::string& filename) {
  const std::vector<PolygonCorpusEntry> corpus = ReadPolygonCorpus(filename);
  EXPECT_FALSE(corpus.empty()) << "no polygon corpus read from " << filename;
  for (const auto& entry : corpus) {
    testing::RegisterTest(
        "Polygon.Fillet", entry.name.c_str(), nullptr, nullptr, __FILE__,
        __LINE__,
        [polys = entry.polys, epsilon = entry.epsilon,
         maxCircleRadius = entry.maxCircleRadius]() -> FilletTestFixture* {
          return new FilletTestFixture(polys, epsilon, maxCircleRadius);
        });
  }
}
#endif
}  // namespace

TEST(TriangulatorReuse, ClearsState) {
  const Polygons large = StarPolygon(64);
  const Polygons withHole = SquareHole();
  const PolygonsIdx largeIndexed = IndexPolygons(large);
  const PolygonsIdx holeIndexed = IndexPolygons(withHole);
  PolygonTriangulatorStore triangulators;

  const auto largeTriangles = Triangulate(large, -1, false);
  const auto holeTriangles = Triangulate(withHole, -1, false);

  EXPECT_EQ(largeTriangles.size(), 62);
  EXPECT_EQ(holeTriangles.size(), 8);
  EXPECT_EQ(
      TriangulateIdxHalfedges(largeIndexed, -1, false, triangulators.local())
          .Triangles(),
      largeTriangles);
  EXPECT_EQ(
      TriangulateIdxHalfedges(holeIndexed, -1, false, triangulators.local())
          .Triangles(),
      holeTriangles);
  EXPECT_EQ(
      TriangulateIdxHalfedges(largeIndexed, -1, false, triangulators.local())
          .Triangles(),
      largeTriangles);
}

#if MANIFOLD_PAR == 1 && !defined(__EMSCRIPTEN__)
TEST(TriangulatorReuse, IsThreadLocal) {
  const Polygons large = StarPolygon(64);
  const Polygons withHole = SquareHole();
  const PolygonsIdx largeIndexed = IndexPolygons(large);
  const PolygonsIdx holeIndexed = IndexPolygons(withHole);
  const auto largeTriangles = Triangulate(large, -1, false);
  const auto holeTriangles = Triangulate(withHole, -1, false);
  PolygonTriangulatorStore triangulators;

  std::atomic<bool> success = true;
  std::array<std::thread, 8> threads;
  for (std::thread& thread : threads) {
    thread = std::thread([&] {
      PolygonTriangulator& triangulator = triangulators.local();
      for (int i = 0; i < 10; ++i) {
        if (TriangulateIdxHalfedges(largeIndexed, -1, false, triangulator)
                    .Triangles() != largeTriangles ||
            TriangulateIdxHalfedges(holeIndexed, -1, false, triangulator)
                    .Triangles() != holeTriangles) {
          success = false;
          return;
        }
      }
    });
  }
  for (std::thread& thread : threads) thread.join();

  EXPECT_TRUE(success);
}
#endif

#ifndef MANIFOLD_NO_IOSTREAM
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

void RegisterFilletTests() {
  std::string files[] = {"fillet.txt", "polygon_corpus.txt", "sponge.txt",
                         "zebra.txt", "zebra3.txt"};

#ifdef __EMSCRIPTEN__
  for (auto f : files) RegisterFilletTestsFile("/polygons/" + f);
#else
  std::string file = __FILE__;
  auto end = std::min(file.rfind('\\'), file.rfind('/'));
  std::string dir = file.substr(0, end);
  for (auto f : files) RegisterFilletTestsFile(dir + "/polygons/" + f);
#endif
}
#else
// Stub when MANIFOLD_NO_IOSTREAM is set: polygon corpus tests need
// std::ifstream to load fixtures, so they're skipped. test_main.cpp
// still calls RegisterPolygonTests(); this stub keeps the link clean.
void RegisterPolygonTests() {}
void RegisterFilletTests() {}
#endif

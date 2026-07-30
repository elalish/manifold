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
#include <limits>
#include <thread>
#include <utility>

#include "../src/polygon_internal.h"
#include "polygon_corpus.h"
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
#else
// Stub when MANIFOLD_NO_IOSTREAM is set: polygon corpus tests need
// std::ifstream to load fixtures, so they're skipped. test_main.cpp
// still calls RegisterPolygonTests(); this stub keeps the link clean.
void RegisterPolygonTests() {}
#endif

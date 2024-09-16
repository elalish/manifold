// Copyright 2023 The Manifold Authors.
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

#include <atomic>
#include <fstream>
#include <future>

#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"
#include "manifold/polygon.h"

using namespace fuzztest;

void TriangulationNoCrash(
    std::vector<std::vector<std::pair<float, float>>> input, float precision) {
  if (precision < 0) precision = -1;
  manifold::PolygonParams().intermediateChecks = true;
  manifold::Polygons polys;
  size_t size = 0;
  for (const auto &simplePoly : input) {
    polys.emplace_back();
    for (const auto &p : simplePoly) {
      polys.back().emplace_back(p.first, p.second);
      size++;
    }
  }
  std::atomic<pid_t> tid;
  std::atomic<bool> faulted(true);
  auto asyncFuture =
      std::async(std::launch::async, [&polys, &faulted, &tid, precision]() {
        tid.store(gettid());
        try {
          manifold::Triangulate(polys, precision);
          faulted.store(false);
        } catch (manifold::geometryErr e) {
          // geometryErr is fine
          faulted.store(false);
        } catch (...) {
          printf("got unexpected error\n");
        }
      });
  if (asyncFuture.wait_for(std::chrono::milliseconds(
          std::max<size_t>(size, 10000))) == std::future_status::timeout) {
    printf("timeout after %ldms...\n", std::max<size_t>(size, 10000));
    pthread_cancel(tid.load());
  }

  EXPECT_FALSE(faulted.load());
}

void TriangulationNoCrashRounded(
    std::vector<std::vector<std::pair<float, float>>> input, float precision) {
  TriangulationNoCrash(std::move(input), precision);
}

using Polygons = std::vector<std::vector<std::pair<float, float>>>;
using TestCase = std::tuple<Polygons, float>;

std::vector<TestCase> SeedProvider();

auto PolygonDomain =
    ContainerOf<Polygons>(ContainerOf<std::vector<std::pair<float, float>>>(
                              PairOf(Finite<float>(), Finite<float>()))
                              .WithMinSize(3)
                              .WithMaxSize(10000))
        .WithMinSize(1)
        .WithMaxSize(10000);

FUZZ_TEST(PolygonFuzz, TriangulationNoCrash)
    .WithDomains(PolygonDomain, InRange<float>(-1.0, 0.1))
    .WithSeeds(SeedProvider);

FUZZ_TEST(PolygonFuzz, TriangulationNoCrashRounded)
    .WithDomains(ReversibleMap(
                     [](auto input) {
                       for (auto &poly : input) {
                         for (auto &pair : poly) {
                           pair.first = std::round(pair.first);
                           pair.second = std::round(pair.second);
                         }
                       }
                       return input;
                     },
                     [](auto input) {
                       return std::optional{std::tuple{input}};
                     },
                     PolygonDomain),
                 InRange<float>(-1.0, 0.1))
    .WithSeeds(SeedProvider);

static std::vector<TestCase> TestCases;
void TestPoly(Polygons polys, int _unused, float precision = -1.0) {
  TestCases.push_back({polys, precision});
}

std::vector<TestCase> SeedProvider() {
  std::string file = __FILE__;
  std::string dir = file.substr(0, file.rfind('/'));
  auto f = std::ifstream(dir + "/polygons/" + "polygon_corpus.txt");
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

  std::vector<TestCase> TestCases;
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
    TestCases.push_back({polys, precision});
  }
  f.close();
  return TestCases;
}

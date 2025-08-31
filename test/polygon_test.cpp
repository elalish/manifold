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

#include "../src/vec.h"
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
  std::string name;

  explicit PolygonTestFixture(Polygons polys, double epsilon,
                              int expectedNumTri, const std::string &name)
      : polys(polys),
        epsilon(epsilon),
        expectedNumTri(expectedNumTri),
        name(name) {}

  void TestBody() { TestPoly(polys, expectedNumTri, epsilon); }
};

template <typename TestFixture>
void RegisterPolygonTestsFile(const std::string &suitename,
                              const std::string &filename) {
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
        [=, polys = std::move(polys)]() -> TestFixture * {
          return new TestFixture(polys, epsilon, expectedNumTri, name);
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
  for (auto f : files)
    RegisterPolygonTestsFile<PolygonTestFixture>("Polygon",
                                                 dir + "/polygons/" + f);
#endif
}

struct FilletResult {
  FilletResult(const std::vector<CrossSection> &crossSections,
               const std::string &name)
      : name(name), crossSections(crossSections) {};

  std::string name;

  std::vector<CrossSection> crossSections;
};

class FilletTestFixture : public PolygonTestFixture {
 public:
  using PolygonTestFixture::PolygonTestFixture;

  void TestBody() override {
    // result->emplace_back(FilletResult({CrossSection(polys)}, name));

    TestFillet(polys, expectedNumTri, epsilon);
  }

  void TestFillet(const Polygons &polys, int expectedNumTri,
                  double epsilon = -1.0);

 private:
  static std::unique_ptr<std::vector<FilletResult>,
                         void (*)(std::vector<FilletResult> *)>
      result;

  static std::unique_ptr<std::vector<FilletResult>,
                         void (*)(std::vector<FilletResult> *)>
      input;
};

void FilletTestFixture::TestFillet(const Polygons &polys, int expectedNumTri,
                                   double epsilon) {
  // const double radius = 0.7;

  const int inputCircularSegments = 20;

  manifold::ManifoldParams().verbose = true;

  auto input = CrossSection(polys);
  auto bbox = input.Bounds().Size();

  double min = std::min(bbox.x, bbox.y), max = std::max(bbox.x, bbox.y);

  std::vector<double> radiusVec;
  if (false) {
    std::array<double, 10> multipliers{1E-4, 1E-3, 1E-2, 0.1, 0.5, 1};
    for (auto it = multipliers.begin(); it != multipliers.end(); it++) {
      radiusVec.push_back(*it * min);
      radiusVec.push_back(*it * max);
    }
  } else {
    radiusVec.push_back(0.7);
  }

  for (auto it = radiusVec.begin(); it != radiusVec.end(); it++) {
    const double radius = *it;

    const int circularSegments = inputCircularSegments > 2
                                     ? inputCircularSegments
                                     : Quality::GetCircularSegments(radius);

    auto r = input.Fillet(radius, circularSegments);

    auto rc = manifold::CrossSection::Compose(r);
    EXPECT_TRUE(rc.Area() < manifold::CrossSection(polys).Area());

    for (const auto &crossSection : r) {
      auto polygon = crossSection.ToPolygons();
      for (const auto &loop : polygon) {
        const auto &cs = CrossSection(loop);

        bool isCCW = cs.Area() > 0;

        for (size_t i = 0; i != loop.size(); i++) {
          vec2 p1 = loop[i], p2 = loop[(i + 1) % loop.size()],
               p3 = loop[(i + 2) % loop.size()];

          // Check edge direction
          vec2 e1 = p2 - p1, e2 = p3 - p2;
          double det = la::cross(e1, e2);
          EXPECT_TRUE(isCCW && (det > 0));

          // Check angle between edge
          double angle = la::asin(det / (la::length(e1) * la::length(e2)));

          const double dPhi = M_PI_2 / circularSegments;
          EXPECT_TRUE(angle < dPhi || angle > M_PI);
        }
      }
    }

    // Check idempotent
    {
      auto rr = rc.Fillet(radius, circularSegments);
      auto rrc = manifold::CrossSection::Compose(r);

      EXPECT_NEAR(rc.Area(), rrc.Area(), 0.01 * (rc.Area() - input.Area()));
    }

    result->emplace_back(FilletResult(r, name + "_" + std::to_string(radius)));
  }
}

void RegisterFilletTests() {
  // std::string files[] = {"polygon_corpus.txt", "sponge.txt", "zebra.txt",
  //                        "zebra3.txt"};

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

void Save(const std::string &filename,
          const std::vector<FilletResult> &result) {
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
    outFile << test.name << " " << test.crossSections.size() << "\n";

    // Write each CrossSection within the test.
    for (const auto &crossSection : test.crossSections) {
      const auto polygons = crossSection.ToPolygons();

      outFile << polygons.size() << "\n";
      for (const auto &loop : polygons) {
        outFile << loop.size() << "\n";
        for (const auto &point : loop) {
          outFile << point.x << " " << point.y << "\n";
        }
      }
    }
  }

  outFile.close();
  std::cout << "Successfully saved " << result.size() << " tests to "
            << filename << std::endl;
}

std::unique_ptr<std::vector<FilletResult>,
                void (*)(std::vector<FilletResult> *)>
    FilletTestFixture::result =
        std::unique_ptr<std::vector<FilletResult>,
                        void (*)(std::vector<FilletResult> *)>(
            new std::vector<FilletResult>(),
            [](std::vector<FilletResult> *v) -> void {
              Save("result.txt", *v);

              delete v;
            });
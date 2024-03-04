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

#include "polygon.h"

#include <algorithm>
#include <limits>
#include <random>

#include "test.h"

namespace {

using namespace manifold;

Polygons Turn180(Polygons polys) {
  for (SimplePolygon &poly : polys) {
    for (glm::vec2 &vert : poly) {
      vert *= -1;
    }
  }
  return polys;
}

Polygons Duplicate(Polygons polys) {
  float xMin = std::numeric_limits<float>::infinity();
  float xMax = -std::numeric_limits<float>::infinity();
  for (SimplePolygon &poly : polys) {
    for (glm::vec2 &vert : poly) {
      xMin = std::min(xMin, vert.x);
      xMax = std::max(xMax, vert.x);
    }
  }
  const float shift = xMax - xMin;

  const int nPolys = polys.size();
  for (int i = 0; i < nPolys; ++i) {
    SimplePolygon poly = polys[i];
    for (glm::vec2 &vert : poly) {
      vert.x += shift;
    }
    polys.push_back(poly);
  }
  return polys;
}

void TestPoly(const Polygons &polys, int expectedNumTri,
              float precision = -1.0f) {
  PolygonParams().verbose = options.params.verbose;

  std::vector<glm::ivec3> triangles;
  EXPECT_NO_THROW(triangles = Triangulate(polys, precision));
  EXPECT_EQ(triangles.size(), expectedNumTri) << "Basic";

  EXPECT_NO_THROW(triangles = Triangulate(Turn180(polys), precision));
  EXPECT_EQ(triangles.size(), expectedNumTri) << "Turn 180";

  EXPECT_NO_THROW(triangles = Triangulate(Duplicate(polys), precision));
  EXPECT_EQ(triangles.size(), 2 * expectedNumTri) << "Duplicate";

  PolygonParams().verbose = false;
}
}  // namespace

#include "polygon_corpus.cpp"

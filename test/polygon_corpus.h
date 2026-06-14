// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "manifold/common.h"

#ifndef MANIFOLD_NO_IOSTREAM
#include <fstream>
#endif

namespace manifold {

// One entry of the shared polygon test corpus: a test name, its expected
// triangle count and epsilon, and the polygon set.
struct PolygonCorpusEntry {
  std::string name;
  int expectedNumTri;
  double epsilon;
  Polygons polys;
};

#ifndef MANIFOLD_NO_IOSTREAM
// Read a polygon corpus file - per entry: name, expectedNumTri, epsilon,
// numPolys, then each polygon's point count and its vertices. Returns empty if
// the file cannot be opened.
inline std::vector<PolygonCorpusEntry> ReadPolygonCorpus(
    const std::string& path) {
  std::vector<PolygonCorpusEntry> corpus;
  std::ifstream f(path);
  if (!f.is_open()) return corpus;
  std::string name;
  double epsilon, x, y;
  int expectedNumTri, numPolys, numPoints;
  while (f >> name) {
    f >> expectedNumTri >> epsilon >> numPolys;
    Polygons polys;
    for (int i = 0; i < numPolys; ++i) {
      polys.emplace_back();
      f >> numPoints;
      for (int j = 0; j < numPoints; ++j) {
        f >> x >> y;
        polys.back().emplace_back(x, y);
      }
    }
    corpus.push_back(
        {std::move(name), expectedNumTri, epsilon, std::move(polys)});
  }
  return corpus;
}
#endif  // MANIFOLD_NO_IOSTREAM

}  // namespace manifold

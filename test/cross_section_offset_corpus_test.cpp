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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "manifold/cross_section.h"
#include "manifold/polygon.h"

// Offset the polygon triangulation corpus across every join type and verify
// each result triangulates cleanly (a valid, non-overlapping fill). Triangulate
// only throws on overlap under MANIFOLD_DEBUG, so the non-overlap teeth need a
// debug build. Raw first pass: it runs the whole corpus, so some pathological
// cases may still need filtering before this becomes a hard gate.

namespace {
using namespace manifold;

// Read the polygon corpus (name, expectedNumTri, epsilon, numPolys, then the
// polygons), keeping just the name and the polygon set.
std::vector<std::pair<std::string, Polygons>> ReadCorpus(
    const std::string& path) {
  std::vector<std::pair<std::string, Polygons>> corpus;
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
    corpus.emplace_back(name, std::move(polys));
  }
  return corpus;
}

std::string CorpusPath() {
#ifdef __EMSCRIPTEN__
  // test/polygons is preloaded into the Emscripten FS at /polygons (see
  // test/CMakeLists.txt); the __FILE__ source path does not exist there.
  return "/polygons/polygon_corpus.txt";
#else
  std::string file = __FILE__;
  const auto end = std::min(file.rfind('\\'), file.rfind('/'));
  return file.substr(0, end) + "/polygons/polygon_corpus.txt";
#endif
}

}  // namespace

TEST(CrossSectionOffsetCorpus, TriangulatesCleanly) {
  const auto corpus = ReadCorpus(CorpusPath());
  ASSERT_FALSE(corpus.empty());
  const CrossSection::JoinType joins[] = {
      CrossSection::JoinType::Round, CrossSection::JoinType::Miter,
      CrossSection::JoinType::Square, CrossSection::JoinType::Bevel};
  int nonEmpty = 0;
  for (const auto& entry : corpus) {
    const CrossSection cs(entry.second);
    const vec2 size = cs.Bounds().Size();
    const double diag = std::sqrt(size.x * size.x + size.y * size.y);
    if (!std::isfinite(diag) || diag <= 0) continue;
    // Offset out and in, by fractions of the bbox diagonal, on every join type.
    for (const double frac : {0.05, -0.02}) {
      for (const CrossSection::JoinType jt : joins) {
        SCOPED_TRACE(entry.first + " frac=" + std::to_string(frac));
        const CrossSection off = cs.Offset(frac * diag, jt);
        const Polygons result = off.ToPolygons();
        if (!result.empty()) ++nonEmpty;
        EXPECT_NO_THROW(Triangulate(result));
      }
    }
  }
  // Guard against a regression that silently empties every offset, which would
  // make the triangulation checks vacuous.
  EXPECT_GT(nonEmpty, 0);
}

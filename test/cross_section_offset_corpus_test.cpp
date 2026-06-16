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
#include <string>

#include "gtest/gtest.h"
#include "manifold/cross_section.h"
#include "manifold/polygon.h"
#include "polygon_corpus.h"

// Offset the polygon triangulation corpus across every join type and verify
// each result triangulates cleanly (a valid, non-overlapping fill). Triangulate
// only throws on overlap under MANIFOLD_DEBUG, so the non-overlap teeth need a
// debug build. Raw first pass: it runs the whole corpus, so some pathological
// cases may still need filtering before this becomes a hard gate.

#ifndef MANIFOLD_NO_IOSTREAM
namespace {
using namespace manifold;

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
  const auto corpus = ReadPolygonCorpus(CorpusPath());
  ASSERT_FALSE(corpus.empty());
  const CrossSection::JoinType joins[] = {
      CrossSection::JoinType::Round, CrossSection::JoinType::Miter,
      CrossSection::JoinType::Square, CrossSection::JoinType::Bevel};
  int nonEmpty = 0;
  for (const auto& entry : corpus) {
    const CrossSection cs(entry.polys);
    const double inArea = cs.Area();
    const vec2 size = cs.Bounds().Size();
    const double diag = la::length(size);
    if (!std::isfinite(diag) || diag <= 0) continue;
    // Offset out and in, by fractions of the bbox diagonal, on every join type.
    double outset[4], inset[4];
    for (const double frac : {0.05, -0.02}) {
      for (int j = 0; j < 4; ++j) {
        SCOPED_TRACE(entry.name + " frac=" + std::to_string(frac));
        const CrossSection off = cs.Offset(frac * diag, joins[j]);
        const Polygons result = off.ToPolygons();
        if (!result.empty()) ++nonEmpty;
        EXPECT_NO_THROW(Triangulate(result));
        (frac > 0 ? outset : inset)[j] = off.Area();
      }
    }
    if (inArea <= 0) continue;
    // An outset never loses area and an inset never gains it; and across a
    // given outset the join-type areas order Miter >= Square >= Round >= Bevel
    // (verified over the whole corpus). joins[] is {Round, Miter, Square,
    // Bevel}.
    const double tol = 1e-6 * inArea;
    SCOPED_TRACE(entry.name + " area ordering");
    for (int j = 0; j < 4; ++j) {
      EXPECT_GE(outset[j], inArea - tol) << "outset lost area";
      EXPECT_LE(inset[j], inArea + tol) << "inset gained area";
    }
    EXPECT_GE(outset[1], outset[2] - tol);  // Miter >= Square
    EXPECT_GE(outset[2], outset[0] - tol);  // Square >= Round
    EXPECT_GE(outset[0], outset[3] - tol);  // Round >= Bevel
  }
  // Guard against a regression that silently empties every offset, which would
  // make the triangulation checks vacuous.
  EXPECT_GT(nonEmpty, 0);
}
#else
// The offset corpus test loads fixtures via ReadPolygonCorpus (std::ifstream),
// so it is skipped under MANIFOLD_NO_IOSTREAM.
#endif

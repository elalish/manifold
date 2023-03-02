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

#include "cross_section.h"

#include <gtest/gtest.h>

#include "manifold.h"
#include "polygon.h"
#include "public.h"
#include "test.h"

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

using namespace manifold;

TEST(CrossSection, MirrorUnion) {
  auto a = CrossSection::Square({5., 5.}, true);
  auto b = a.Translate({2.5, 2.5});
  auto cross = a + b + b.Mirror({-1, 1});
  auto result = Manifold::Extrude(cross, 5.);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("cross_section_mirror_union.glb", result.GetMesh(), {});
#endif

  auto area_a = a.Area();
  EXPECT_EQ(area_a + 1.5 * area_a, cross.Area());
  EXPECT_TRUE(a.Mirror(glm::vec2(0)).IsEmpty());
}

TEST(CrossSection, RoundOffset) {
  auto a = CrossSection::Square({20., 20.}, true);
  auto rounded = a.Offset(5., CrossSection::JoinType::Round);
  auto result = Manifold::Extrude(rounded, 5.);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("cross_section_round_offset.glb", result.GetMesh(), {});
#endif
}

TEST(CrossSection, Empty) {
  Polygons polys(2);
  auto e = CrossSection(polys);
  EXPECT_TRUE(e.IsEmpty());
}

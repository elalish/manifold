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

#include "cross_section.h"

#include "manifold.h"
#include "polygon.h"
#include "test.h"

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

using namespace manifold;

TEST(CrossSection, Union) {
  auto a = CrossSection::Square({5., 5.}, true);
  auto b = a.Translate({2.5, 2.5});
  auto result = Manifold::Extrude(a + b, 5.);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("cross_section_union.glb", result.GetMesh(), {});
#endif
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

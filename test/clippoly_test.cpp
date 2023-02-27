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

#include "clippoly.h"

#include "manifold.h"
#include "polygon.h"
#include "test.h"

#ifdef MANIFOLD_EXPORT
#include "meshIO.h"
#endif

using namespace manifold;

TEST(Clippoly, Union) {
  auto a = Clippoly::Square({5., 5.}, true);
  auto b = a.Translate({2.5, 2.5});
  auto cross = (a + b);
  auto result = Manifold::Extrude(cross, 5.);

#ifdef MANIFOLD_EXPORT
  if (options.exportModels)
    ExportMesh("clippoly_union.glb", result.GetMesh(), {});
#endif
}

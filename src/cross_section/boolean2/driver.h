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

#include <vector>

#include "predicates.h"
#include "winding_filter.h"

namespace manifold {
namespace boolean2 {

struct Trace;

struct OverlapResult {
  std::vector<vec2> verts;
  std::vector<OutEdge> edges;
  std::vector<int> inputVert2Merged;
  int numMergedVerts;
};

// `eps` is the fresh per-op FP-noise bound (3D analogue: Impl::epsilon_).
// `tolerance` is the propagated drift bound (3D analogue: Impl::tolerance_);
// floored to `eps` if smaller. Nearby-intersection merging widens
// new-to-old snaps to `tolerance`; MergeVerts stays at `eps`.
OverlapResult RemoveOverlaps2D(const std::vector<vec2>& vertsIn,
                               const std::vector<EdgeM>& edgesIn, double eps,
                               double tolerance = 0.0, bool debug = false,
                               WindRule pred = WindRule::Add,
                               Trace* trace = nullptr);

}  // namespace boolean2
}  // namespace manifold

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

#include <utility>
#include <vector>

#include "predicates.h"
#include "vertex_merge.h"

namespace manifold {
namespace boolean2 {

struct NarrowPhaseResult {
  std::vector<std::vector<int>> lists;
  std::vector<IntersectionPoint> intersections;
};

// Combined narrow phase over broad-phase edge pairs. Produces sorted
// edge-vertex split lists and independent proper edge-edge intersection
// candidates without mutating `verts` or `edges`; serial vs TBB execution is
// an internal thresholded implementation detail.
NarrowPhaseResult BuildListsAndFindIntersections(
    const std::vector<EdgeM>& edges, const std::vector<vec2>& verts, double eps,
    const std::vector<std::pair<int, int>>& pairs);

}  // namespace boolean2
}  // namespace manifold

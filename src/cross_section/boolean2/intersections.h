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

#include "bvh.h"
#include "predicates.h"
#include "vertex_merge.h"

namespace manifold {
namespace boolean2 {

inline constexpr double kIntersectionMergeEpsFactor = 10.0;

void CollectIntersectionPairs(const std::vector<EdgeM>& edges,
                              const std::vector<vec2>& verts, double eps,
                              const std::vector<Box2>& edgeBoxes,
                              const BVH& bvh,
                              std::vector<std::pair<int, int>>* pairs);

// Serially materialize precomputed proper intersections into caller-owned
// containers. `verts` and `lists` are taken by value so callers can move in the
// post-narrow-phase state; the returned fields are those same containers after
// appending/snapping intersection vertices and updating edge split lists.
// `vertEdges` records which edges meet at each materialized or snapped
// intersection vertex; nearby-intersection merge uses that incidence to
// distinguish unrelated near vertices from vertices that share an intersection
// source.
struct IntersectionInsertion {
  std::vector<vec2> verts;
  std::vector<std::vector<int>> lists;
  std::vector<std::vector<int>> vertEdges;
};

IntersectionInsertion FindAndInsertIntersections(
    const std::vector<EdgeM>& edges, std::vector<vec2> verts,
    std::vector<std::vector<int>> lists, double eps,
    const std::vector<Box2>& edgeBoxes, const BVH& bvh,
    const std::vector<IntersectionPoint>& precomputedIntersections);

}  // namespace boolean2
}  // namespace manifold

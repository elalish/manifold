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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "../../collider.h"
#include "../../parallel.h"

namespace manifold {
namespace boolean2 {

#ifndef BOOLEAN2_EDGE_BVH_THRESHOLD
#define BOOLEAN2_EDGE_BVH_THRESHOLD 256
#endif
inline constexpr int kEdgeBvhThreshold = BOOLEAN2_EDGE_BVH_THRESHOLD;
inline constexpr int kEdgePairBvhThreshold = 1024;

struct Box2 {
  vec2 min = vec2(std::numeric_limits<double>::infinity());
  vec2 max = vec2(-std::numeric_limits<double>::infinity());

  Box2() = default;
  Box2(vec2 a, vec2 b)
      : min(std::min(a.x, b.x), std::min(a.y, b.y)),
        max(std::max(a.x, b.x), std::max(a.y, b.y)) {}

  Box2 Union(const Box2& b) const {
    return {vec2(std::min(min.x, b.min.x), std::min(min.y, b.min.y)),
            vec2(std::max(max.x, b.max.x), std::max(max.y, b.max.y))};
  }

  vec2 Center() const { return (min + max) * 0.5; }

  bool DoesOverlap(const Box2& b) const {
    return min.x <= b.max.x && max.x >= b.min.x && min.y <= b.max.y &&
           max.y >= b.min.y;
  }
};

Box2 BoxOf2DPoint(vec2 p, double eps);
Box2 BoxOf2DEdge(vec2 p0, vec2 p1, double eps);
uint32_t MortonCode2(vec2 position, Box2 bBox);

struct BVH {
  std::vector<Box2> nodeBBox;
  std::vector<std::pair<int, int>> internalChildren;
  std::vector<int> leafToOrig;

  bool Empty() const { return internalChildren.empty(); }
};

BVH BVHBuildFromBoxes(const std::vector<Box2>& boxes);

template <const bool selfCollision = false, typename Recorder, typename F>
inline void BVHCollisions(const BVH& bvh, Recorder& recorder, F&& queryBox,
                          int n, bool parallel) {
  using namespace collider_internal;
  if (bvh.Empty()) return;
  auto collideOne = [&](int queryIdx) {
    const Box2 query = queryBox(queryIdx);
    int stack[64];
    int top = -1;
    int node = kRoot;
    auto& local = recorder.local();
    while (true) {
      const int internal = Node2Internal(node);
      const int child1 = bvh.internalChildren[internal].first;
      const int child2 = bvh.internalChildren[internal].second;
      auto recordOrTraverse = [&](int child) {
        const bool overlaps = bvh.nodeBBox[child].DoesOverlap(query);
        if (overlaps && IsLeaf(child)) {
          const int leafIdx = Node2Leaf(child);
          if (!selfCollision || leafIdx != queryIdx)
            recorder.record(queryIdx, leafIdx, local);
        }
        return overlaps && IsInternal(child);
      };
      const bool traverse1 = recordOrTraverse(child1);
      const bool traverse2 = recordOrTraverse(child2);
      if (!traverse1 && !traverse2) {
        if (top < 0) break;
        node = stack[top--];
      } else {
        node = traverse1 ? child1 : child2;
        if (traverse1 && traverse2) stack[++top] = child2;
      }
    }
  };
  manifold::for_each_n(parallel ? autoPolicy(n, 512) : ExecutionPolicy::Seq,
                       countAt(0), n, collideOne);
}

template <typename F>
inline void CollidePairs(const BVH& bvh, const std::vector<Box2>& queries,
                         F&& f) {
  if (bvh.Empty() || queries.empty()) return;
  auto adapter = [&](int qi, int leafIdx) { f(qi, bvh.leafToOrig[leafIdx]); };
  auto recorder = MakeSimpleRecorder(adapter);
  auto qf = [&](int i) { return queries[i]; };
  BVHCollisions<false>(bvh, recorder, qf, static_cast<int>(queries.size()),
                       /*parallel=*/false);
}

}  // namespace boolean2
}  // namespace manifold

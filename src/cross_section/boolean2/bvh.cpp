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
//
// BVH helpers for eps-padded 2D box queries.

#include "../../boolean2.h"

namespace manifold {

namespace {

constexpr int kRadixTreeBuildGrainSize = 10000;

}  // namespace

Box2 BoxOf2DPoint(vec2 p, double eps) {
  const vec2 pad(eps, eps);
  return Box2(p - pad, p + pad);
}

Box2 BoxOf2DEdge(vec2 p0, vec2 p1, double eps) {
  const vec2 pad(eps, eps);
  Box2 b(p0, p1);
  return Box2(b.min - pad, b.max + pad);
}

uint32_t MortonCode2(vec2 position, Box2 bBox) {
  using collider_internal::SpreadBits3;
  const vec2 size = bBox.max - bBox.min;
  const double xNorm = size.x > 0 ? (position.x - bBox.min.x) / size.x : 0.5;
  const double yNorm = size.y > 0 ? (position.y - bBox.min.y) / size.y : 0.5;
  const double xClamped = std::min(1023.0, std::max(0.0, 1024.0 * xNorm));
  const double yClamped = std::min(1023.0, std::max(0.0, 1024.0 * yNorm));
  const uint32_t x = SpreadBits3(static_cast<uint32_t>(xClamped));
  const uint32_t y = SpreadBits3(static_cast<uint32_t>(yClamped));
  return x * 2 + y;
}

BVH BVHBuildFromBoxes(const std::vector<Box2>& boxes) {
  using namespace collider_internal;
  const int n = static_cast<int>(boxes.size());
  BVH out;
  out.leafToOrig.resize(n);
  for (int i = 0; i < n; ++i) out.leafToOrig[i] = i;
  if (n == 0) return out;
  Box2 bbox = boxes[0];
  for (const auto& b : boxes) bbox = bbox.Union(b);
  std::vector<uint32_t> morton(n);
  for (int i = 0; i < n; ++i) morton[i] = MortonCode2(boxes[i].Center(), bbox);
  manifold::stable_sort(out.leafToOrig.begin(), out.leafToOrig.end(),
                        [&](int a, int b) { return morton[a] < morton[b]; });
  std::vector<uint32_t> sortedMorton(n);
  for (int i = 0; i < n; ++i) {
    sortedMorton[i] = morton[out.leafToOrig[i]];
  }
  const int numNodes = 2 * n - 1;
  out.nodeBBox.resize(numNodes);
  std::vector<int> nodeParent(numNodes, -1);
  out.internalChildren.resize(n - 1, std::make_pair(-1, -1));
  // Radix-tree node creation does little work per index, so use a coarser
  // grain than BVH query traversal/narrow predicates to avoid scheduling
  // overhead dominating construction.
  manifold::for_each_n(
      autoPolicy(n - 1, kRadixTreeBuildGrainSize), countAt(0), n - 1,
      CreateRadixTree(
          {VecView<int>(nodeParent.data(), nodeParent.size()),
           VecView<std::pair<int, int>>(out.internalChildren.data(),
                                        out.internalChildren.size()),
           VecView<const uint32_t>(sortedMorton)}));
  for (int i = 0; i < n; ++i)
    out.nodeBBox[Leaf2Node(i)] = boxes[out.leafToOrig[i]];
  auto buildNode = [&](auto&& self, int node) -> Box2 {
    if (IsLeaf(node)) return out.nodeBBox[node];
    const auto [left, right] = out.internalChildren[Node2Internal(node)];
    out.nodeBBox[node] = self(self, left).Union(self(self, right));
    return out.nodeBBox[node];
  };
  if (n > 1) buildNode(buildNode, kRoot);
  return out;
}

}  // namespace manifold

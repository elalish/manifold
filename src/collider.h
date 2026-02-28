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

#pragma once
#include "manifold/common.h"
#include "parallel.h"
#include "utils.h"
#include "vec.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

#if (MANIFOLD_PAR == 1)
#include <tbb/combinable.h>
#endif

namespace manifold {

namespace collider_internal {
// Adjustable parameters
constexpr int kInitialLength = 128;
constexpr int kLengthMultiple = 4;
constexpr int kSequentialThreshold = 512;
// Fundamental constants
constexpr int kRoot = 1;

#ifdef _MSC_VER

#ifndef _WINDEF_
using DWORD = unsigned long;
#endif

uint32_t inline ctz(uint32_t value) {
  DWORD trailing_zero = 0;

  if (_BitScanForward(&trailing_zero, value)) {
    return trailing_zero;
  } else {
    // This is undefined, I better choose 32 than 0
    return 32;
  }
}

uint32_t inline clz(uint32_t value) {
  DWORD leading_zero = 0;

  if (_BitScanReverse(&leading_zero, value)) {
    return 31 - leading_zero;
  } else {
    // Same remarks as above
    return 32;
  }
}
#endif

constexpr inline bool IsLeaf(int node) { return node % 2 == 0; }
constexpr inline bool IsInternal(int node) { return node % 2 == 1; }
constexpr inline int Node2Internal(int node) { return (node - 1) / 2; }
constexpr inline int Internal2Node(int internal) { return internal * 2 + 1; }
constexpr inline int Node2Leaf(int node) { return node / 2; }
constexpr inline int Leaf2Node(int leaf) { return leaf * 2; }

struct CreateRadixTree {
  VecView<int> nodeParent_;
  VecView<std::pair<int, int>> internalChildren_;
  const VecView<const uint32_t> leafMorton_;

  int PrefixLength(uint32_t a, uint32_t b) const {
// count-leading-zeros is used to find the number of identical highest-order
// bits
#ifdef _MSC_VER
    // return __lzcnt(a ^ b);
    return clz(a ^ b);
#else
    return __builtin_clz(a ^ b);
#endif
  }

  int PrefixLength(int i, int j) const {
    if (j < 0 || j >= static_cast<int>(leafMorton_.size())) {
      return -1;
    } else {
      int out;
      if (leafMorton_[i] == leafMorton_[j])
        // use index to disambiguate
        out = 32 +
              PrefixLength(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
      else
        out = PrefixLength(leafMorton_[i], leafMorton_[j]);
      return out;
    }
  }

  int RangeEnd(int i) const {
    // Determine direction of range (+1 or -1)
    int dir = PrefixLength(i, i + 1) - PrefixLength(i, i - 1);
    dir = (dir > 0) - (dir < 0);
    // Compute conservative range length with exponential increase
    int commonPrefix = PrefixLength(i, i - dir);
    int max_length = kInitialLength;
    while (PrefixLength(i, i + dir * max_length) > commonPrefix)
      max_length *= kLengthMultiple;
    // Compute precise range length with binary search
    int length = 0;
    for (int step = max_length / 2; step > 0; step /= 2) {
      if (PrefixLength(i, i + dir * (length + step)) > commonPrefix)
        length += step;
    }
    return i + dir * length;
  }

  int FindSplit(int first, int last) const {
    int commonPrefix = PrefixLength(first, last);
    // Find the furthest object that shares more than commonPrefix bits with the
    // first one, using binary search.
    int split = first;
    int step = last - first;
    do {
      step = (step + 1) >> 1;  // divide by 2, rounding up
      int newSplit = split + step;
      if (newSplit < last) {
        int splitPrefix = PrefixLength(first, newSplit);
        if (splitPrefix > commonPrefix) split = newSplit;
      }
    } while (step > 1);
    return split;
  }

  void operator()(int internal) {
    int first = internal;
    // Find the range of objects with a common prefix
    int last = RangeEnd(first);
    if (first > last) std::swap(first, last);
    // Determine where the next-highest difference occurs
    int split = FindSplit(first, last);
    int child1 = split == first ? Leaf2Node(split) : Internal2Node(split);
    ++split;
    int child2 = split == last ? Leaf2Node(split) : Internal2Node(split);
    // Record parent_child relationships.
    internalChildren_[internal].first = child1;
    internalChildren_[internal].second = child2;
    int node = Internal2Node(internal);
    nodeParent_[child1] = node;
    nodeParent_[child2] = node;
  }
};

template <typename F, const bool selfCollision, const bool hasTransform,
          typename Recorder>
struct FindCollision {
  F& f;
  VecView<const Box> nodeBBox_;
  VecView<const std::pair<int, int>> internalChildren_;
  Recorder& recorder;
  mat3x4 transform;

  using Local = typename Recorder::Local;

  inline int RecordCollision(std::invoke_result_t<F, const int> query, int node,
                             const int queryIdx, Local& local) {
    auto box = nodeBBox_[node];
    if (hasTransform) box = box.Transform(transform);
    bool overlaps = box.DoesOverlap(query);
    if (overlaps && IsLeaf(node)) {
      int leafIdx = Node2Leaf(node);
      if (!selfCollision || leafIdx != queryIdx) {
        recorder.record(queryIdx, leafIdx, local);
      }
    }
    return overlaps && IsInternal(node);  // Should traverse into node
  }

  void operator()(const int queryIdx) {
    auto query = f(queryIdx);
    // stack cannot overflow because radix tree has max depth 30 (Morton code) +
    // 32 (index).
    int stack[64];
    int top = -1;
    // Depth-first search
    int node = kRoot;
    Local& local = recorder.local();
    while (1) {
      int internal = Node2Internal(node);
      int child1 = internalChildren_[internal].first;
      int child2 = internalChildren_[internal].second;

      int traverse1 = RecordCollision(query, child1, queryIdx, local);
      int traverse2 = RecordCollision(query, child2, queryIdx, local);

      if (!traverse1 && !traverse2) {
        if (top < 0) break;   // done
        node = stack[top--];  // get a saved node
      } else {
        node = traverse1 ? child1 : child2;  // go here next
        if (traverse1 && traverse2) {
          stack[++top] = child2;  // save the other for later
        }
      }
    }
  }
};

template <typename Recorder>
struct DualTraversal {
  using Local = typename Recorder::Local;

  VecView<const Box> nodeBBox1;
  VecView<const std::pair<int, int>> internalChildren1;
  const std::optional<mat3x4>& transform1;

  VecView<const Box> nodeBBox2;
  VecView<const std::pair<int, int>> internalChildren2;
  const std::optional<mat3x4>& transform2;

  Recorder& recorder;

#if MANIFOLD_PAR == 1
  tbb::task_group group;
#endif

  void populateChildren(bool isCollider1, int node, Box b,
                        std::array<std::pair<int, Box>, 2>& children,
                        int& length) const {
    if (IsLeaf(node)) {
      children[length++] = std::make_pair(node, b);
    } else {
      int internal = Node2Internal(node);
      const VecView<const Box>& nodeBBox = isCollider1 ? nodeBBox1 : nodeBBox2;
      const VecView<const std::pair<int, int>>& internalChildren =
          isCollider1 ? internalChildren1 : internalChildren2;
      const std::optional<mat3x4>& transform =
          isCollider1 ? transform1 : transform2;
      auto [child1, child2] = internalChildren[internal];
      for (int c : {child1, child2}) {
        Box bb = nodeBBox[c];
        if (transform) bb = bb.Transform(transform.value());
        children[length++] = std::make_pair(c, bb);
      }
    }
  }

  void check(int node1, int node2, Box b1, Box b2, Local& local,
             int splitDepth = 0) {
    if (IsLeaf(node1) && IsLeaf(node2)) {
      recorder.record(Node2Leaf(node1), Node2Leaf(node2), b1, b2, local);
    } else {
      std::array<std::pair<int, Box>, 2> children1;
      std::array<std::pair<int, Box>, 2> children2;
      std::array<std::tuple<int, int, Box, Box>, 4> toCheck;
      int children1Length = 0;
      int children2Length = 0;
      int toCheckLength = 0;

      populateChildren(true, node1, b1, children1, children1Length);
      populateChildren(false, node2, b2, children2, children2Length);
      for (int i = 0; i < children1Length; i++) {
        for (int j = 0; j < children2Length; j++) {
          auto [c1, bb1] = children1[i];
          auto [c2, bb2] = children2[j];
          if (bb1.DoesOverlap(bb2))
            toCheck[toCheckLength++] = std::make_tuple(c1, c2, bb1, bb2);
        }
      }
      for (int i = 0; i < toCheckLength; i++) {
        int n1 = std::get<0>(toCheck[i]);
        int n2 = std::get<1>(toCheck[i]);
        Box bb1 = std::get<2>(toCheck[i]);
        Box bb2 = std::get<3>(toCheck[i]);
#if MANIFOLD_PAR == 1
        if (splitDepth < 5 && toCheckLength > 1)
          group.run([n1, n2, bb1, bb2, splitDepth, this]() {
            check(n1, n2, bb1, bb2, recorder.local(), splitDepth + 1);
          });
        else
#endif
          check(n1, n2, bb1, bb2, local, splitDepth);
      }
    }
  }
};

struct BuildInternalBoxes {
  VecView<Box> nodeBBox_;
  VecView<int> counter_;
  const VecView<int> nodeParent_;
  const VecView<std::pair<int, int>> internalChildren_;

  void operator()(int leaf) {
    int node = Leaf2Node(leaf);
    do {
      node = nodeParent_[node];
      int internal = Node2Internal(node);
      if (AtomicAdd(counter_[internal], 1) == 0) return;
      nodeBBox_[node] = nodeBBox_[internalChildren_[internal].first].Union(
          nodeBBox_[internalChildren_[internal].second]);
    } while (node != kRoot);
  }
};

constexpr inline uint32_t SpreadBits3(uint32_t v) {
  v = 0xFF0000FFu & (v * 0x00010001u);
  v = 0x0F00F00Fu & (v * 0x00000101u);
  v = 0xC30C30C3u & (v * 0x00000011u);
  v = 0x49249249u & (v * 0x00000005u);
  return v;
}
}  // namespace collider_internal

template <typename F>
struct SimpleRecorder {
  using Local = F;
  F& f;

  inline void record(int queryIdx, int leafIdx, F& f) const {
    f(queryIdx, leafIdx);
  }
  Local& local() { return f; }
};

template <typename F>
inline SimpleRecorder<F> MakeSimpleRecorder(F& f) {
  return SimpleRecorder<F>{f};
}

/** @ingroup Private */
class Collider {
 public:
  Collider() {};

  Collider(const VecView<const Box>& leafBB,
           const VecView<const uint32_t>& leafMorton) {
    ZoneScoped;
    DEBUG_ASSERT(leafBB.size() == leafMorton.size(), userErr,
                 "vectors must be the same length");
    if (leafBB.size() == 0) return;
    int num_nodes = 2 * leafBB.size() - 1;
    // assign and allocate members
    nodeBBox_.resize_nofill(num_nodes);
    nodeParent_.resize(num_nodes, -1);
    internalChildren_.resize(leafBB.size() - 1, std::make_pair(-1, -1));
    // organize tree
    for_each_n(autoPolicy(NumInternal(), 1e4), countAt(0), NumInternal(),
               collider_internal::CreateRadixTree(
                   {nodeParent_, internalChildren_, leafMorton}));
    UpdateBoxes(leafBB);
  }

  void UpdateBoxes(const VecView<const Box>& leafBB) {
    ZoneScoped;
    DEBUG_ASSERT(leafBB.size() == NumLeaves(), userErr,
                 "must have the same number of updated boxes as original");
    // copy in leaf node Boxes
    auto leaves = StridedRange(nodeBBox_.begin(), nodeBBox_.end(), 2);
    copy(leafBB.cbegin(), leafBB.cend(), leaves.begin());
    // create global counters
    Vec<int> counter(NumInternal(), 0);
    // kernel over leaves to save internal Boxes
    for_each_n(autoPolicy(NumInternal(), 1e3), countAt(0), NumLeaves(),
               collider_internal::BuildInternalBoxes(
                   {nodeBBox_, counter, nodeParent_, internalChildren_}));
  }

  template <typename Recorder>
  void DualTraversal(Recorder& recorder,
                     const std::optional<mat3x4>& transform1,
                     const Collider& collider2,
                     const std::optional<mat3x4>& transform2) const {
    using collider_internal::DualTraversal;
    using collider_internal::kRoot;
    if (internalChildren_.empty() || collider2.internalChildren_.empty())
      return;
    DualTraversal<Recorder> traversal{nodeBBox_,
                                      internalChildren_,
                                      transform1,
                                      collider2.nodeBBox_,
                                      collider2.internalChildren_,
                                      transform2,
                                      recorder};
    Box b1 = nodeBBox_[kRoot];
    Box b2 = collider2.nodeBBox_[kRoot];
    if (transform1) b1 = b1.Transform(transform1.value());
    if (transform2) b2 = b2.Transform(transform2.value());
    traversal.check(kRoot, kRoot, b1, b2, recorder.local(),
                    nodeBBox_.size() > kSeqThreshold &&
                            collider2.nodeBBox_.size() > kSeqThreshold
                        ? 0
                        : 7);
#if MANIFOLD_PAR == 1
    traversal.group.wait();
#endif
  }

  template <const bool selfCollision = false, typename F, typename Recorder>
  void Collisions(Recorder& recorder, std::optional<mat3x4> transform, F f,
                  int n, bool parallel = true) const {
    ZoneScoped;
    using collider_internal::FindCollision;
    if (internalChildren_.empty()) return;
    if (transform) {
      for_each_n(
          parallel ? autoPolicy(n, collider_internal::kSequentialThreshold)
                   : ExecutionPolicy::Seq,
          countAt(0), n,
          FindCollision<decltype(f), selfCollision, true, Recorder>{
              f, nodeBBox_, internalChildren_, recorder, transform.value()});
    } else {
      for_each_n(parallel
                     ? autoPolicy(n, collider_internal::kSequentialThreshold)
                     : ExecutionPolicy::Seq,
                 countAt(0), n,
                 FindCollision<decltype(f), selfCollision, false, Recorder>{
                     f, nodeBBox_, internalChildren_, recorder, la::identity});
    }
  }

  // This function iterates over queriesIn and calls recorder.record(queryIdx,
  // leafIdx, local) for each collision it found.
  // If selfCollisionl is true, it will skip the case where queryIdx == leafIdx.
  // The recorder should provide a local() method that returns a Recorder::Local
  // type, representing thread local storage. By default, recorder.record can
  // run in parallel and the thread local storage can be combined at the end.
  // If parallel is false, the function will run in sequential mode.
  //
  // If thread local storage is not needed, use SimpleRecorder.
  template <const bool selfCollision = false, typename T, typename Recorder>
  void Collisions(Recorder& recorder, std::optional<mat3x4> transform,
                  const VecView<const T>& queriesIn,
                  bool parallel = true) const {
    auto f = [queriesIn](const int i) { return queriesIn[i]; };
    Collisions<selfCollision>(recorder, transform, f, queriesIn.size(),
                              parallel);
  }

  static uint32_t MortonCode(vec3 position, Box bBox) {
    using collider_internal::SpreadBits3;
    vec3 xyz = (position - bBox.min) / (bBox.max - bBox.min);
    xyz = la::min(vec3(1023.0), la::max(vec3(0.0), 1024.0 * xyz));
    uint32_t x = SpreadBits3(static_cast<uint32_t>(xyz.x));
    uint32_t y = SpreadBits3(static_cast<uint32_t>(xyz.y));
    uint32_t z = SpreadBits3(static_cast<uint32_t>(xyz.z));
    return x * 4 + y * 2 + z;
  }

 private:
  Vec<Box> nodeBBox_;
  Vec<int> nodeParent_;
  // even nodes are leaves, odd nodes are internal, root is 1
  Vec<std::pair<int, int>> internalChildren_;

  size_t NumInternal() const { return internalChildren_.size(); };
  size_t NumLeaves() const {
    return internalChildren_.empty() ? 0 : (NumInternal() + 1);
  };
};

}  // namespace manifold

// Copyright 2025 The Manifold Authors.
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

#include <algorithm>

#include "./utils.h"
#include "manifold/common.h"
#include "manifold/optional_assert.h"
#include "manifold/vec_view.h"

namespace manifold {

template <typename P, bool sortX = true>
void BuildTwoDTreeImpl(VecView<P> points) {
  std::sort(points.begin(), points.end(), [](const P& a, const P& b) {
    return sortX ? a.pt.x < b.pt.x : a.pt.y < b.pt.y;
  });
  if (points.size() < 2) return;
  BuildTwoDTreeImpl<P, !sortX>(points.view(0, points.size() / 2));
  BuildTwoDTreeImpl<P, !sortX>(points.view(points.size() / 2 + 1));
}

// Not really a proper KD-tree, but a kd tree with k = 2 and alternating x/y
// partition.
// Recursive sorting is not the most efficient, but simple and guaranteed to
// result in a balanced tree.
//
// Make it multiway...
template <typename P>
void BuildTwoDTree(VecView<P> points) {
  ZoneScoped;
  // don't even bother...
  if (points.size() <= 8) return;
  BuildTwoDTreeImpl<P, true>(points);
}

template <typename P, typename F>
void QueryTwoDTree(VecView<P> points, Rect r, F f) {
  ZoneScoped;
  if (points.size() <= 8) {
    for (const auto& p : points)
      if (r.Contains(p.pt)) f(p);
    return;
  }
  Rect current;
  current.min = vec2(-std::numeric_limits<double>::infinity());
  current.max = vec2(std::numeric_limits<double>::infinity());

  int level = 0;
  VecView<P> currentView = points;
  std::array<Rect, 64> rectStack;
  std::array<VecView<P>, 64> viewStack;
  std::array<int, 64> levelStack;
  int stackPointer = 0;

  while (1) {
    if (currentView.size() <= 2) {
      for (const auto& p : currentView)
        if (r.Contains(p.pt)) f(p);
      if (--stackPointer < 0) break;
      level = levelStack[stackPointer];
      currentView = viewStack[stackPointer];
      current = rectStack[stackPointer];
      continue;
    }

    // these are conceptual left/right trees
    Rect left = current;
    Rect right = current;
    const P middle = currentView[currentView.size() / 2];
    if (level % 2 == 0)
      left.max.x = right.min.x = middle.pt.x;
    else
      left.max.y = right.min.y = middle.pt.y;

    if (r.Contains(middle.pt)) f(middle);
    if (left.DoesOverlap(r)) {
      if (right.DoesOverlap(r)) {
        DEBUG_ASSERT(stackPointer < 64, logicErr, "Stack overflow");
        rectStack[stackPointer] = right;
        viewStack[stackPointer] = currentView.view(currentView.size() / 2 + 1);
        levelStack[stackPointer] = level + 1;
        stackPointer++;
      }
      current = left;
      currentView = currentView.view(0, currentView.size() / 2);
      level++;
    } else {
      current = right;
      currentView = currentView.view(currentView.size() / 2 + 1);
      level++;
    }
  }
}

template <typename P>
void VerifyTwoDTree(VecView<P> points) {
  Rect current;
  current.min = vec2(-std::numeric_limits<double>::infinity());
  current.max = vec2(std::numeric_limits<double>::infinity());

  int level = 0;
  VecView<P> currentView = points;
  std::array<Rect, 64> rectStack;
  std::array<VecView<P>, 64> viewStack;
  std::array<int, 64> levelStack;
  int stackPointer = 0;

  while (1) {
    if (currentView.size() <= 2) {
      if (--stackPointer < 0) break;
      level = levelStack[stackPointer];
      currentView = viewStack[stackPointer];
      current = rectStack[stackPointer];
      continue;
    }

    Rect left = current;
    Rect right = current;
    const P middle = currentView[currentView.size() / 2];
    if (level % 2 == 0)
      left.max.x = right.min.x = middle.pt.x;
    else
      left.max.y = right.min.y = middle.pt.y;

    DEBUG_ASSERT(stackPointer < 64, logicErr, "Stack overflow");

    DEBUG_ASSERT(
        std::all_of(currentView.begin(),
                    currentView.begin() + (currentView.size() / 2),
                    [&left](const P& p) { return left.Contains(p.pt); }),
        logicErr, "Left tree contains invalid point");
    if (!std::all_of(currentView.begin() + (currentView.size() / 2 + 1),
                     currentView.end(),
                     [&right](const P& p) { return right.Contains(p.pt); })) {
      printf("level = %d, length = %ld\n", level, currentView.size());
    }
    DEBUG_ASSERT(
        std::all_of(currentView.begin() + (currentView.size() / 2 + 1),
                    currentView.end(),
                    [&right](const P& p) { return right.Contains(p.pt); }),
        logicErr, "Right tree contains invalid point");

    rectStack[stackPointer] = right;
    viewStack[stackPointer] = currentView.view(currentView.size() / 2 + 1);
    levelStack[stackPointer] = level + 1;
    stackPointer++;
    current = left;
    currentView = currentView.view(0, currentView.size() / 2);
    level++;
  }
}
}  // namespace manifold

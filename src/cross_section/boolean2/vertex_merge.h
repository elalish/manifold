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

namespace manifold {

struct IntersectionPoint {
  int i;
  int j;
  vec2 p;
};

struct VertexMerge {
  std::vector<int> inputVert2Merged;
  std::vector<vec2> verts;
};

VertexMerge MergeVerts(const std::vector<vec2>& in, double eps);
bool VESetContains(const std::vector<int>& vec, int x);
void VESetInsert(std::vector<int>* vec, int x);
std::vector<EdgeM> RemapAndCollapse(const std::vector<EdgeM>& edges,
                                    const std::vector<int>& inputVert2Merged);

}  // namespace manifold

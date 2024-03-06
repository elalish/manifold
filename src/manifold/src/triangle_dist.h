// Copyright 2024 The Manifold Authors.
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
#include <array>

#include "public.h"

namespace manifold {

// The algorithm needs to handle the following cases:
//    1. The minimum distance is along the edges of both triangles (guaranteed
//      in 2D)
//    2. One of the closest points is a vertex of one triangle and the other
//      closest point is on the face of the other triangle
//    3. The triangles intersect (can't happen in MinGap)
//    4. An edge of one triangle is parallel to the face of the other triangle
//      (handled by case 1)
//    5. One or both triangles are degenerate (we may ignore this?)
//
// An implementation can be found in https://gamma.cs.unc.edu/SSV/, TriDist.cpp.
// Probably create an implementation based on this.
//
// Explanation and sources taken from here:
// https://stackoverflow.com/questions/53602907/algorithm-to-find-minimum-distance-between-two-triangles

/**
 * Returns the minimum distance between two triangles.
 *
 * @param t1 First  triangle.
 * @param t2 Second triangle.
 */
float TriangleDistance(const std::array<glm::vec3, 3>& t1,
                       const std::array<glm::vec3, 3>& t2);
}  // namespace manifold

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

#include "samples.h"

namespace manifold {

/**
 * A smoothed manifold demonstrating selective edge sharpening with
 * Manifold.Smooth(). Use Manifold.Refine() before export to see the curvature.
 */
Manifold Scallop() {
  constexpr double height = 1;
  constexpr double radius = 3;
  constexpr double offset = 2;
  constexpr int wiggles = 12;
  constexpr double sharpness = 0.8;
  constexpr double lean = 0.2;

  MeshGL64 scallop;
  scallop.numProp = 3;
  scallop.vertProperties = {-offset, 0, height, -offset, 0, -height};

  const double len = kPi * radius / (3 * wiggles);
  const vec3 topNormal = normalize(vec3(-lean, 0, 1));
  vec3 lastCenterTan(-1, 0, 0);
  lastCenterTan = lastCenterTan - la::dot(lastCenterTan, topNormal) * topNormal;
  const vec3 lastEdgeTangent(0, len, 0);

  const double delta = kPi / wiggles;
  for (uint32_t i = 1; i <= 2 * wiggles; ++i) {
    const uint32_t j = i == 2 * wiggles ? 0 : i;
    const double theta = i * delta;
    const double amp = 0.5 * height * (la::cos(theta) + 1);

    vec3 v = vec3(radius * la::cos(theta), radius * la::sin(theta),
                  amp * (i % 2 == 0 ? 1 : -1));

    scallop.vertProperties.insert(scallop.vertProperties.end(),
                                  {v.x, v.y, v.z});
    const vec3 edgeTangent(-len * la::sin(theta), len * la::cos(theta), 0);
    vec3 centerTan = v - vec3(-offset, 0, height);
    centerTan /= 3;
    centerTan = centerTan - la::dot(centerTan, topNormal) * topNormal;

    scallop.triVerts.insert(scallop.triVerts.end(), {0, 2 + i - 1, 2 + j});
    scallop.halfedgeTangent.insert(
        scallop.halfedgeTangent.end(),
        {lastCenterTan.x, lastCenterTan.y, lastCenterTan.z, 1,        //
         lastEdgeTangent.x, lastEdgeTangent.y, lastEdgeTangent.z, 1,  //
         0, 0, len, 1});

    scallop.triVerts.insert(scallop.triVerts.end(), {1, 2 + j, 2 + i - 1});
    scallop.halfedgeTangent.insert(
        scallop.halfedgeTangent.end(),
        {centerTan.x, centerTan.y, -centerTan.z, 1,          //
         -edgeTangent.x, -edgeTangent.y, -edgeTangent.z, 1,  //
         0, 0, -len, 1});

    lastCenterTan = centerTan;
  }

  return Manifold(scallop);
}
}  // namespace manifold

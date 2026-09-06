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
  constexpr double lean = 0.3;

  MeshGL64 scallop;
  scallop.numProp = 3;
  scallop.vertProperties = {-offset, 0, height, -offset, 0, -height};

  const double len = kPi * radius / (2 * wiggles);
  const vec3 topNormal = normalize(vec3(-lean, 0, 1));

  auto centerTangent = [&](vec3 point) {
    vec3 tangent = point - vec3(-offset, 0, height);
    tangent /= 2;
    return tangent - la::dot(tangent, topNormal) * topNormal;
  };

  auto addTri = [&](std::array<uint32_t, 3> triVerts, vec3 firstTangent,
                    vec3 secondTangent, double thirdZ) {
    scallop.triVerts.insert(scallop.triVerts.end(), triVerts.begin(),
                            triVerts.end());
    scallop.halfedgeTangent.insert(
        scallop.halfedgeTangent.end(),
        {firstTangent.x, firstTangent.y, firstTangent.z, 1,     //
         secondTangent.x, secondTangent.y, secondTangent.z, 1,  //
         0, 0, thirdZ, 1});
  };

  const vec3 v0(radius, 0, 0.5 * height);
  scallop.vertProperties.insert(scallop.vertProperties.end(),
                                {v0.x, v0.y, v0.z});

  vec3 lastCenterTan = centerTangent(v0);
  const vec3 firstCenterTan = lastCenterTan;
  const vec3 firstEdgeTangent(0, len, 0);
  vec3 lastEdgeTangent = firstEdgeTangent;

  const double delta = kPi / wiggles;
  for (uint32_t i = 1; i < 2 * wiggles; ++i) {
    const double theta = i * delta;
    const double amp = 0.25 * height * (la::cos(theta) + 1);

    vec3 v = vec3(radius * la::cos(theta), radius * la::sin(theta),
                  amp * (i % 2 == 0 ? 1 : -1));

    scallop.vertProperties.insert(scallop.vertProperties.end(),
                                  {v.x, v.y, v.z});
    const vec3 edgeTangent(-len * la::sin(theta), len * la::cos(theta), 0);
    vec3 centerTan = centerTangent(v);
    const uint32_t prev = 1 + i;
    const uint32_t curr = 2 + i;

    addTri({0, prev, curr}, lastCenterTan, lastEdgeTangent, len);

    addTri({1, curr, prev}, {centerTan.x, centerTan.y, -centerTan.z},
           -edgeTangent, -len);

    lastCenterTan = centerTan;
    lastEdgeTangent = edgeTangent;
  }

  const uint32_t last = 1 + 2 * wiggles;
  addTri({0, last, 2}, lastCenterTan, lastEdgeTangent, len);

  addTri({1, 2, last}, {firstCenterTan.x, firstCenterTan.y, -firstCenterTan.z},
         -firstEdgeTangent, -len);

  return Manifold(scallop);
}
}  // namespace manifold

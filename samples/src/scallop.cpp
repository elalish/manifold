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
 * A smoothed manifold demonstrating manual smoothing by assigning tangents to
 * halfedges. Use Manifold.Refine() before export to see the curvature.
 */
Manifold Scallop() {
  constexpr double height = 1;
  constexpr double radius = 3;
  constexpr double offset = 2;
  constexpr int wiggles = 12;
  constexpr double lean = 0.3;
  constexpr double frontalSharpness = 1;

  MeshGL64 scallop;
  scallop.numProp = 3;
  scallop.vertProperties = {-offset, 0, height, -offset, 0, -height};

  const double len = kPi * radius / (2 * wiggles);
  const vec3 topNormal = normalize(vec3(-lean, 0, 1));

  const double delta = kPi / wiggles;
  std::array<vec3, 2 * wiggles> centerTangents;
  std::array<vec3, 2 * wiggles> edgeTangents;
  for (uint32_t i = 0; i < 2 * wiggles; ++i) {
    const double theta = i * delta;
    const double amp = 0.5 * height * (la::cos(theta) + 1) / 2;
    const vec3 v(radius * la::cos(theta), radius * la::sin(theta),
                 amp * (i % 2 == 0 ? 1 : -1));

    scallop.vertProperties.insert(scallop.vertProperties.end(),
                                  {v.x, v.y, v.z});
    vec3 centerTan = v - vec3(-offset, 0, height);
    centerTan /= 2;
    centerTangents[i] = centerTan - la::dot(centerTan, topNormal) * topNormal;
    edgeTangents[i] = vec3(-len * la::sin(theta), len * la::cos(theta), 0);
  }

  for (uint32_t i = 0; i < 2 * wiggles; ++i) {
    const uint32_t next = i + 1 == 2 * wiggles ? 0 : i + 1;

    const vec3 radial = centerTangents[i];
    const vec3 nextRadial = centerTangents[next];
    const vec3 edge = edgeTangents[i];
    const vec3 nextEdge = -edgeTangents[next];
    const double sharpness = frontalSharpness * (la::cos(i * delta) + 1) / 4;
    const vec3 up =
        la::lerp(vec3(0, 0, len), vec3(nextEdge.y, -nextEdge.x, 0), sharpness);
    const vec3 down =
        la::lerp(vec3(0, 0, -len), vec3(-edge.y, edge.x, 0), sharpness);

    scallop.triVerts.insert(scallop.triVerts.end(), {0, 2 + i, 2 + next});
    scallop.halfedgeTangent.insert(        //
        scallop.halfedgeTangent.end(),     //
        {radial.x, radial.y, radial.z, 1,  //
         edge.x, edge.y, edge.z, 1,        //
         up.x, up.y, up.z, 1});

    scallop.triVerts.insert(scallop.triVerts.end(), {1, 2 + next, 2 + i});
    scallop.halfedgeTangent.insert(
        scallop.halfedgeTangent.end(),
        {nextRadial.x, nextRadial.y, -nextRadial.z, 1,  //
         nextEdge.x, nextEdge.y, nextEdge.z, 1,         //
         down.x, down.y, down.z, 1});
  }

  return Manifold(scallop);
}
}  // namespace manifold

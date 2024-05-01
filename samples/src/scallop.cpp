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
  constexpr float height = 1;
  constexpr float radius = 3;
  constexpr float offset = 2;
  constexpr int wiggles = 12;
  constexpr float sharpness = 0.8;

  Mesh scallop;
  std::vector<Smoothness> sharpenedEdges;
  scallop.vertPos = {{-offset, 0, height}, {-offset, 0, -height}};

  const float delta = glm::pi<float>() / wiggles;
  for (int i = 0; i < 2 * wiggles; ++i) {
    float theta = (i - wiggles) * delta;
    float amp = 0.5 * height * glm::max(glm::cos(0.8f * theta), 0.0f);

    scallop.vertPos.push_back({radius * glm::cos(theta),
                               radius * glm::sin(theta),
                               amp * (i % 2 == 0 ? 1 : -1)});
    int j = i + 1;
    if (j == 2 * wiggles) j = 0;

    float smoothness = 1 - sharpness * glm::cos((theta + delta / 2) / 2);
    int halfedge = 3 * scallop.triVerts.size() + 1;
    sharpenedEdges.push_back({halfedge, smoothness});
    scallop.triVerts.push_back({0, 2 + i, 2 + j});

    halfedge = 3 * scallop.triVerts.size() + 1;
    sharpenedEdges.push_back({halfedge, smoothness});
    scallop.triVerts.push_back({1, 2 + j, 2 + i});
  }

  return Manifold::Smooth(scallop, sharpenedEdges);
}
}  // namespace manifold
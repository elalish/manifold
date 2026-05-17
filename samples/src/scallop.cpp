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

  MeshGL64 scallop;
  std::vector<Smoothness> sharpenedEdges;
  scallop.numProp = 3;
  scallop.vertProperties = {-offset, 0, height, -offset, 0, -height};

  const double delta = kPi / wiggles;
  for (int i = 0; i < 2 * wiggles; ++i) {
    double theta = (i - wiggles) * delta;
    double amp = 0.5 * height * la::max(la::cos(0.8 * theta), 0.0);

    scallop.vertProperties.insert(
        scallop.vertProperties.end(),
        {radius * la::cos(theta), radius * la::sin(theta),
         amp * (i % 2 == 0 ? 1 : -1)});
    int j = i + 1;
    if (j == 2 * wiggles) j = 0;

    double smoothness = 1 - sharpness * la::cos((theta + delta / 2) / 2);
    size_t halfedge = scallop.triVerts.size() + 1;
    sharpenedEdges.push_back({halfedge, smoothness});
    scallop.triVerts.insert(
        scallop.triVerts.end(),
        {0, static_cast<uint32_t>(2 + i), static_cast<uint32_t>(2 + j)});

    halfedge = scallop.triVerts.size() + 1;
    sharpenedEdges.push_back({halfedge, smoothness});
    scallop.triVerts.insert(
        scallop.triVerts.end(),
        {1, static_cast<uint32_t>(2 + j), static_cast<uint32_t>(2 + i)});
  }

  return Manifold::Smooth(scallop, sharpenedEdges);
}
}  // namespace manifold
